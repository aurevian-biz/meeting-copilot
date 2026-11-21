#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""vision.py — let the voice assistant "take a look at the screen"

Pipeline: /dev/video0 (UVC capture card) → drop warm-up frames → scale → RapidOCR → text

## Why RapidOCR rather than PaddleOCR

Measured on Jetson AGX Orin, the same 1080p Chinese UI screenshot (50 lines of text):

| Engine | Time | Mean confidence |
|---|---|---|
| RapidOCR (onnxruntime) | **6.2s** | 0.894 |
| PaddleOCR default (server model) | 17.0s | 0.98 |
| PaddleOCR mobile | 24.6s | 0.749 |

On Jetson the mobile model is actually slower than server (the Jetson CPU likes large convolutions),
and its confidence drops noticeably. RapidOCR also drops paddle's few hundred MB of deps — asr-venv has them already.

## Why the input is scaled to a 1440 long edge first

The 1920 original takes 7.3s and reads 「窗口」 as 「官口」; at 1440 it takes only 5.7s and reads correctly.
RapidOCR internally squashes the long edge to a fixed size, so scaling with INTER_AREA first saves one resampling pass.
Below 960 small glyphs start to smear (「编辑」→「编冀」), so do not go lower.

## Why the OCR thread count is hard-coded

`RapidOCR()` without a thread count takes the onnxruntime default. This machine also runs vLLM, ASR and captions,
12 cores at 12 threads measures **9.10s**, explicit 8 threads **4.36s** — oversubscription costs far more than parallelism gains.
The flat region is 6~8, degradation starts at 10. See `OCR_THREADS`.

## Why no frames are dropped per audio segment, but 40 are dropped on device open

On MS2109, after every **device open or resolution switch** the first 33~36 frames are pure black (YMAX=7 measured). Skip them and you get
one black image whose OCR is empty, looking like "there is no text on screen". README 4.7 has the full record of this trap.

So warm-up happens "once per open", not "once per frame grab": `_Capture` keeps the handle resident,
**a process lifetime pays that 1.2s only once**. Frame grab drops from 2.87s to tens of milliseconds (2 frames are fresh enough).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path


# OCR thread count: **must be set explicitly, never inherit the default**.
# onnxruntime defaults to saturating all cores (Orin 12 cores → 12 threads), while vLLM/ASR/captions run alongside,
# 12 threads measures 9.10s, 8 threads 4.36s — oversubscription halves the speed (thread migration + core contention).
# Measured flat region is 6~8 (4.34 / 4.49s), degradation starts at 10 (5.11s), collapse at 12 (9.10s).
# Pick 8: leaves 4 cores for other resident processes, and the difference from 6 is within noise.
OCR_THREADS = 8

# Voice intents that mean "take a look at the screen". Any hit triggers OCR.
# Do not add overly broad words like "see" — far too easy to false-trigger in daily conversation.
LOOK_HINTS = (
    "看看", "看一看", "看一眼", "看一下", "瞧瞧", "瞧一眼",
    "屏幕上", "画面上", "屏幕里", "画面里",
    "这是什么", "这是啥", "写的什么", "写了什么", "读一下", "念一下",
    "认得吗", "认识吗", "你看得见", "能看到",
)


def is_look_intent(text: str) -> bool:
    """Is this utterance asking to look at the screen?"""
    if not text:
        return False
    return any(h in text for h in LOOK_HINTS)


# ---------------------------------------------------------------- Redaction
# Screens often show tokens / keys / phone numbers; mask the obvious ones before full OCR text reaches the LLM
import re

_SECRET = re.compile(r"\b(sk-|ghp_|mul_|xox[baprs]-|AKIA)[A-Za-z0-9_\-]{8,}")
_LONGKEY = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")


def redact(text: str) -> tuple[str, int]:
    """Mask suspected keys / long random strings / phone numbers; returns (processed text, hit count)."""
    n = 0

    def count(_m: "re.Match") -> str:
        nonlocal n
        n += 1
        return _m.group(0)

    for pat in (_SECRET, _LONGKEY, _PHONE):
        text = pat.sub(count, text)
    # Final substitution: keep the prefix so a human can tell which kind of key it is
    text = _SECRET.sub(lambda m: m.group(1) + "***", text)
    text = _LONGKEY.sub("***", text)
    text = _PHONE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[-4:], text)
    return text, n


class VisionError(RuntimeError):
    pass


class _Capture:
    """Resident capture-card handle: open once, warm up once, then read only a frame or two each time.

    Why not decode continuously in a background thread: 1080p MJPG decoding is real CPU cost, while "look at the
    screen" is used only a few times per meeting. Keeping "handle always open, frames read on demand" gets both — the
    1.2s warm-up and device negotiation are skipped, no CPU burns while idle, and V4L2's queue still yields the current picture.

    Thread safety: one internal lock serialises access, and the device is exclusive anyway.
    """

    def __init__(self, dev: str, width: int, height: int, warmup: int,
                 *, keep_open: bool = True, idle_release: float = 0.0):
        self.dev = dev
        self.width = width
        self.height = height
        self.warmup = max(1, warmup)
        self.keep_open = keep_open
        self.idle_release = idle_release    # if >0, release the device after this much idle time so others can use it
        self._cap = None
        self._lock = threading.RLock()
        self.opens = 0          # open count (should be 1; more means it dropped and reconnected)
        self.warmups = 0        # warm-up rounds (one per open; an all-black retry adds 1)
        self.reads = 0          # successful frame grabs
        self.open_s = 0.0       # duration of the last "open + warm-up"
        self.grab_s = 0.0       # duration of the last frame grab
        self.last_read = 0.0
        self.reconnects = 0
        self.last_release = ""

    # ------------------------------------------------------------ Internal
    def _drain_locked(self, n: int):
        """Read n frames back to back; return the last good frame (None if none could be read)."""
        frame = None
        for _ in range(max(1, n)):
            ok, f = self._cap.read()
            if ok and f is not None:
                frame = f
        return frame

    def _open_locked(self):
        import cv2
        t0 = time.time()
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise VisionError(f"打不开 {self.dev}（设备被占用或未枚举）")
        # MJPG must be explicit: the default negotiates YUYV, which gives only 5fps for 1080p on USB2,
        # so a 40-frame warm-up takes 8s instead of 1.3s
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.opens += 1
        frame = self._drain_locked(self.warmup)     # drop the warm-up black frames in one go
        self.warmups += 1
        self.open_s = round(time.time() - t0, 2)
        if frame is None:
            self._release_locked("预热全程读不到帧")
            raise VisionError("取不到帧（HDMI 无信号或带宽不足）")
        return frame

    def _release_locked(self, why: str = "") -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._cap = None
        self.last_release = why

    def _idle_expired_locked(self) -> bool:
        return bool(self.idle_release and self._cap is not None and self.last_read
                    and time.time() - self.last_read > self.idle_release)

    # ------------------------------------------------------------ Public
    def read(self, fresh: int = 2):
        """Grab one frame of the current picture. If the handle is already open, read only `fresh` frames — no reopen, no re-warm."""
        with self._lock:
            if self._idle_expired_locked():
                self._release_locked("闲置释放")
            t0 = time.time()
            if self._cap is None:
                self._open_locked()
            frame = self._drain_locked(max(1, fresh))
            if frame is None:
                # The occasional dock dropout (once saw a 166s stream cut): releasing and reopening once brings it back
                self._release_locked("读帧失败")
                self.reconnects += 1
                self._open_locked()
                frame = self._drain_locked(max(1, fresh))
            if frame is None:
                raise VisionError("取不到帧（HDMI 无信号或带宽不足）")
            self.reads += 1
            self.last_read = time.time()
            self.grab_s = round(time.time() - t0, 3)
            if not self.keep_open:
                self._release_locked("keep_open=False")
            return frame

    def rewarm(self, budget_s: float = 1.5):
        """Drop one more round of warm-up frames. After a source resolution switch MS2109 emits black frames again; use this to clear them.

        Time-budgeted: **without signal the device lowers its frame rate** (measured 5s to read 40 frames), and with no
        cap the "no-signal decision" itself becomes very slow. Not reading the full 40 frames within the budget is fine;
        whether the picture recovered is re-checked by the caller.
        """
        with self._lock:
            if self._cap is None:
                return None
            self.warmups += 1
            t0 = time.time()
            frame = None
            for _ in range(self.warmup):
                ok, f = self._cap.read()
                if ok and f is not None:
                    frame = f
                if time.time() - t0 > budget_s:
                    break
            return frame

    def close(self) -> None:
        with self._lock:
            self._release_locked("close()")

    def status(self) -> dict:
        with self._lock:
            return {"open": self._cap is not None, "opens": self.opens,
                    "warmups": self.warmups, "reads": self.reads,
                    "reconnects": self.reconnects,
                    "open_s": self.open_s, "grab_s": self.grab_s}


class Vision:
    """Capture-card frame grab + OCR. Thread-safe (internally serialised; the device is exclusive)."""

    def __init__(self, dev: str = "/dev/video0", width: int = 1920, height: int = 1080,
                 warmup: int = 40, max_side: int = 1440, min_score: float = 0.35,
                 max_chars: int = 1400, cache_ttl: float = 8.0,
                 keep_open: bool = True, idle_release: float = 0.0,
                 ocr_threads: int = OCR_THREADS, trt_engine: str = ""):
        self.dev = dev
        self.width = width
        self.height = height
        self.warmup = warmup          # number of leading frames to discard (MS2109 black-frame warm-up)
        self.max_side = max_side
        self.min_score = min_score
        self.max_chars = max_chars
        self.cache_ttl = cache_ttl
        self.ocr_threads = ocr_threads
        self.trt_engine = trt_engine  # non-empty swaps the rec backend for TensorRT (~2.4×)
        self._trt = None
        self.trt_note = ""            # TensorRT attach result (ok / failure reason)
        self._lock = threading.Lock()
        self._engine = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self.calls = 0
        self.errors = 0
        self.last: dict | None = None
        self._grab_s = 0.0   # duration of the last frame grab; must hold a value even when grab fails
        self._cap = _Capture(dev, width, height, warmup,
                             keep_open=keep_open, idle_release=idle_release)

    # ------------------------------------------------------------ Engine
    def _ocr_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR   # lazy import: 0.4s, only paid when used
            try:
                self._engine = RapidOCR(intra_op_num_threads=self.ocr_threads)
            except TypeError:
                # older versions lack this argument: fall back to the default (oversubscribes on a 12-core box, 2× slower)
                self._engine = RapidOCR()
            self._attach_trt()
        return self._engine

    def _attach_trt(self) -> None:
        """Swap the rec backend for TensorRT. On failure stay on CPU — never let "look at the screen" break entirely.

        The engine is pinned to a TensorRT version (rebuild it when JetPack changes) and may have been copied from
        another machine, so every exception is swallowed here and only recorded as status — the CPU path always worked.
        """
        if not self.trt_engine or self._engine is None:
            return
        if not Path(self.trt_engine).exists():
            self.trt_note = f"engine 不存在：{self.trt_engine}"
            return
        try:
            import trt_rec
            self._trt = trt_rec.attach(self._engine, self.trt_engine)
            self.trt_note = "ok"
        except Exception as e:  # noqa: BLE001
            self._trt = None
            self.trt_note = f"{type(e).__name__}: {e}"

    def trt_status(self) -> dict | None:
        return self._trt.status() if self._trt is not None else None

    def warm_ocr(self) -> bool:
        """Build the OCR engine (including TensorRT) at startup.

        Lazy loading the engine saves 0.4s of startup, but TRT has to deserialise the engine and create a CUDA
        context, and the first forward pass still needs a warmup — those seconds land on the user as "the first
        screen look is very slow". So pay them up front on a background thread.
        """
        try:
            self._ocr_engine()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------ Frame grab
    def grab(self):
        """Return one frame of the current picture.

        The device handle is resident, so this **no longer** opens or closes the device and no longer drops 40
        warm-up frames — both are paid once at process start (`prewarm()`), leaving only the 2 frame reads per grab.
        """
        t0 = time.time()
        frame = self._cap.read()
        self._grab_s = time.time() - t0
        return frame

    def prewarm(self) -> bool:
        """Open and warm up the device at startup, so the first "look at the screen" need not wait another 1.2s.

        Failure only returns False and never raises: without a capture card the service still starts, "look at the screen" is just unavailable.
        """
        try:
            self._cap.read(fresh=1)
            return True
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        """Release the capture card (so another program can use the device)."""
        self._cap.close()
        if self._trt is not None:
            try:
                self._trt.close()
            except Exception:  # noqa: BLE001
                pass
            self._trt = None

    def capture_status(self) -> dict:
        return self._cap.status()

    # ------------------------------------------------------------ Frame fingerprint
    @staticmethod
    def _dhash(img, size: int = 16) -> int:
        """Horizontal gradient hash. Insensitive to cursor blink / clock ticking, yet distinguishes layouts.

        Pure numpy (no cv2): one hash is not worth a dependency, and it lets unit tests run on
        machines without opencv. Averages 16×17 blocks, about 1ms per call.
        """
        import numpy as np

        a = np.asarray(img, dtype=np.float32)
        gray = (0.114 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.299 * a[:, :, 2]
                if a.ndim == 3 else a)
        h, w = gray.shape
        th, tw = size, size + 1
        ys = np.linspace(0, h, th + 1).astype(int)
        xs = np.linspace(0, w, tw + 1).astype(int)
        small = np.empty((th, tw), dtype=np.float32)
        for i in range(th):
            y0, y1 = ys[i], max(ys[i] + 1, ys[i + 1])
            for j in range(tw):
                x0, x1 = xs[j], max(xs[j] + 1, xs[j + 1])
                small[i, j] = gray[y0:y1, x0:x1].mean()
        diff = small[:, 1:] > small[:, :-1]
        bits = 0
        for v in diff.flatten():
            bits = (bits << 1) | int(v)
        return bits

    # ------------------------------------------------------------ Recognition
    def _ocr(self, img) -> dict:
        import cv2
        h, w = img.shape[:2]
        scale = self.max_side / max(w, h)
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
        t0 = time.time()
        result, _elapse = self._ocr_engine()(img)
        ocr_s = time.time() - t0
        lines: list[str] = []
        scores: list[float] = []
        for item in result or []:
            try:
                _box, text, score = item[0], item[1], float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            if score >= self.min_score and text.strip():
                lines.append(text.strip())
                scores.append(score)
        text = "\n".join(lines)
        return {
            "text": text[: self.max_chars],
            "lines": len(lines),
            "score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "ocr_s": round(ocr_s, 2),
            "truncated": len(text) > self.max_chars,
        }

    def available(self) -> bool:
        return Path(self.dev).exists()

    # The no-signal criterion is "dark and flat", not "pure black".
    # Measured: with no signal MS2109 emits frames at mean=7.0 / std=1.5 (YMAX=7, sensor noise included),
    # so the old criterion mean<3 never fires on real hardware and treats no-signal as "a screen with no text".
    # A dark-theme IDE (mean 15~25) has std above 15, so it is not misjudged.
    BLANK_MEAN, BLANK_STD = 12.0, 6.0

    @classmethod
    def _is_blank(cls, img) -> bool:
        import numpy as np

        a = np.asarray(img, dtype=np.float32)
        return float(a.mean()) < cls.BLANK_MEAN and float(a.std()) < cls.BLANK_STD

    # ------------------------------------------------------------ Public
    def look(self, force: bool = False) -> dict:
        """Look at the screen. The same picture (same dHash) is served straight from cache within cache_ttl."""
        with self._lock:
            self.calls += 1
            t0 = time.time()
            try:
                img = self.grab()
                if self._is_blank(img):
                    # After a source resolution switch MS2109 emits another round of black frames. A resident
                    # handle never hits the open path but does hit this "mid-stream blackening", so drop one more
                    # warm-up round before declaring no-signal, or a normal picture is misreported as "HDMI no signal".
                    self._cap.rewarm()
                    img2 = self.grab()
                    if img2 is not None and not self._is_blank(img2):
                        img = img2
                    else:
                        self.errors += 1
                        out = {"ok": False, "reason": "blank",
                               "message": "画面是全黑的，HDMI 可能没信号",
                               "rewarmed": True,
                               "grab_s": round(self._grab_s, 2),
                               "total_s": round(time.time() - t0, 2)}
                        self.last = out
                        return out
                h = self._dhash(img)
                key = f"{h:0{16*4}x}"
                hit = self._cache.get(key)
                if hit and not force and time.time() - hit[0] < self.cache_ttl:
                    cached = dict(hit[1])
                    cached.update({"cached": True, "grab_s": round(self._grab_s, 2),
                                   "total_s": round(time.time() - t0, 2)})
                    self.last = cached
                    return cached
                info = self._ocr(img)
                info.update({"ok": True, "cached": False, "hash": key[:16],
                             "grab_s": round(self._grab_s, 2),
                             "total_s": round(time.time() - t0, 2)})
                info["text"], redacted = redact(info["text"])
                if redacted:
                    info["redacted"] = redacted
                self._cache[key] = (time.time(), info)
                # keep only the 8 most recent screen fingerprints so a resident process cannot grow without bound
                if len(self._cache) > 8:
                    oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
                    self._cache.pop(oldest, None)
                self.last = info
                return info
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                out = {"ok": False, "reason": "error", "message": f"{type(e).__name__}: {e}"}
                self.last = out
                return out


def format_context(info: dict, max_lines: int = 40) -> str:
    """Turn the OCR result into a context block fed to the LLM."""
    text = (info or {}).get("text") or ""
    lines = [ln for ln in text.splitlines() if ln.strip()][:max_lines]
    if not lines:
        return ""
    return "[屏幕文字]\n" + "\n".join(lines) + "\n[/屏幕文字]"


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="看一眼屏幕并 OCR")
    ap.add_argument("--dev", default="/dev/video0")
    ap.add_argument("--max-side", type=int, default=1440)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--image", default="", help="不给设备，直接读图片（无硬件时测）")
    args = ap.parse_args()

    if args.image:
        import cv2
        v = Vision(max_side=args.max_side)
        info = v._ocr(cv2.imread(args.image))
        info["ok"] = True
    else:
        v = Vision(args.dev, max_side=args.max_side)
        info = v.look(force=args.force)
    print(_json.dumps(info, ensure_ascii=False, indent=2))
