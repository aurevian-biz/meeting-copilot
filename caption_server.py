#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Meeting captions: capture-card audio -> segmentation -> SenseVoice -> zh/ja bilingual -> scrolling page.

Fully decoupled from voice_server; each owns one audio device (an MS2109 capture card here) without interference.

    ./asr-venv/bin/python caption_server.py                       # capture card -> 8096
    ./asr-venv/bin/python caption_server.py --wav test.wav        # run one file offline
    ./asr-venv/bin/python caption_server.py --device plughw:CARD=H340,DEV=0

Design trade-offs:
- Segment with an energy threshold instead of Silero: meeting audio comes straight off HDMI as a clean
  signal with no echo or noise, so energy is enough and VAD state never bleeds across segments.
- Force-cut every segment at 3s: a speaker who never breathes still won't emit one long block. Measured ASR
  takes 0.3s for 6s of audio, so the real latency is "waiting for the segment to end" -- shorter is faster.
- **Source first, translation patched in later** (two stages): push a source-only line as soon as recognition
  completes, then let a separate thread patch the translation into the same seq. Time-to-text drops to ASR only.
- Three decoupled pipeline stages: capture only segments, ASR only recognizes, translation only translates; a
  slow stage backs up in its own queue and never stalls capture (the old code idled frames for the full 1s translation).
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

# Must stay at module level: this file enables `from __future__ import annotations`,
# so annotations are strings and FastAPI resolves types via get_type_hints in module globals;
# if declared inside a function, it wouldn't recognize Request and would treat it as a query parameter.
from fastapi import Request  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  directory conventions: MC_DATA can override, see paths.py
import voice_probe as vp  # noqa: E402
from voice_server import Asr, clean_text  # noqa: E402

SR = 16000
FRAME_S = 0.02
FRAME_N = int(SR * FRAME_S)

LANG_TAG = re.compile(r"<\|(zh|ja|en|ko|yue)\|>")
KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
CJK = re.compile(r"[\u4e00-\u9FFF]")
# Characters with real content. When a noise segment slips past min_s=0.25, SenseVoice
# returns punctuation-only output like " 。": an empty caption line that also wastes a translation call and a seq.
SUBSTANCE = re.compile(r"[0-9A-Za-z\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")


# ------------------------------------------------------------------ capture
def resolve_pw_target(hint: str = "MACROSILICON") -> str:
    """Find the capture card's source node name from pw-dump.

    PipeWire here claims the capture card as the default source and holds the PCM permanently, so a direct
    ALSA connection (arecord -D plughw) reports "device or resource busy"; going through PipeWire is fine.
    """
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=8).stdout
        for node in json.loads(out):
            props = (node.get("info") or {}).get("props") or {}
            name = props.get("node.name") or ""
            if props.get("media.class") == "Audio/Source" and hint.lower() in name.lower():
                return name
    except Exception:  # noqa: BLE001
        pass
    return "@DEFAULT_AUDIO_SOURCE@"


def resolve_pw_source(hint: str = "MACROSILICON") -> str:
    return resolve_pw_target(hint or "MACROSILICON")


def resolve_pw_sink(hint: str = "H340") -> str:
    """Find the output node (headphones). Looks for H340 by name, else uses the system default sink."""
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=8).stdout
        for node in json.loads(out):
            props = (node.get("info") or {}).get("props") or {}
            name = props.get("node.name") or ""
            if props.get("media.class") == "Audio/Sink" and hint.lower() in name.lower():
                return name
    except Exception:  # noqa: BLE001
        pass
    return "@DEFAULT_AUDIO_SINK@"


class Capture:
    """Fixed 20ms frames. The reader thread owns the device; nothing else may open the same card.

    backend="pw" goes through PipeWire (non-exclusive, coexists with other audio clients);
    backend="alsa" uses a direct arecord connection (useful when PipeWire is not holding the device).
    """

    def __init__(self, device: str, sr: int = SR, *, backend: str = "pw",
                 target: str | None = None):
        self.device = device
        self.sr = sr
        self.backend = backend
        self.target = target
        self.frames: queue.Queue = queue.Queue(maxsize=600)   # 12 seconds; when full, drop the oldest
        self.proc: subprocess.Popen | None = None
        self.errors = 0
        self.last_error = ""
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _cmd(self) -> list[str]:
        if self.backend == "pw":
            tgt = self.target or resolve_pw_target()
            self.target = tgt
            return ["pw-record", "--target", tgt, "--rate", str(self.sr),
                    "--channels", "1", "--format", "s16", "-"]
        return ["arecord", "-q", "-D", self.device, "-f", "S16_LE",
                "-r", str(self.sr), "-c", "1", "-t", "raw"]

    def start(self) -> None:
        self.proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self._t = threading.Thread(target=self._run, name="capture", daemon=True)
        self._t.start()

    def _run(self) -> None:
        assert self.proc and self.proc.stdout
        nbytes = FRAME_N * 2
        buf = b""
        while not self._stop.is_set():
            chunk = self.proc.stdout.read(nbytes - len(buf))
            if not chunk:
                # Process exited: keep its last stderr line -- otherwise all we can guess is "device busy"
                err = ""
                if self.proc.poll() is not None and self.proc.stderr:
                    try:
                        err = (self.proc.stderr.read() or b"").decode("utf-8", "replace").strip()
                    except Exception:  # noqa: BLE001
                        pass
                self.last_error = err.splitlines()[-1][:160] if err else "无输出"
                self.errors += 1
                break
            buf += chunk
            if len(buf) < nbytes:
                continue
            frame = np.frombuffer(buf[:nbytes], dtype=np.int16)
            buf = buf[nbytes:]
            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MonitorPipe:
    """Play HDMI audio through the local headphones unchanged (monitor / playback).

    Purpose: the PC's HDMI output all lands in the capture card, so nobody hears it. Here on the Jetson
    the card's audio is tapped out to the headphones, effectively hanging a monitor tap off the HDMI chain.

    Implementation: two child processes, pw-record | pw-cat, wired straight together; Python moves no bytes.
    Samples travel through a kernel pipe, so latency is only the end buffers (20ms each) and CPU cost is nil.

    Why both can run at once: PipeWire lets multiple clients read one source, so the caption service's own
    16k capture stream is unaffected (two concurrent paths measured conflict-free).
    """

    def __init__(self, src: str, sink: str, rate: int = 44100,
                 latency_ms: int = 20):
        self.src = src
        self.sink = sink
        self.rate = rate
        self.latency_ms = latency_ms
        self.rec: subprocess.Popen | None = None
        self.play: subprocess.Popen | None = None
        self.started_at = 0.0
        self.starts = 0
        self.last_error = ""
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self.rec and self.play
                        and self.rec.poll() is None and self.play.poll() is None)

    def start(self) -> bool:
        with self._lock:
            if self.rec and self.play and self.rec.poll() is None:
                return False
            lat = f"{self.latency_ms}ms"
            rec_cmd = ["pw-record", "--target", self.src, "--rate", str(self.rate),
                       "--channels", "2", "--format", "s16", "--latency", lat, "-"]
            play_cmd = ["pw-cat", "--playback", "--target", self.sink,
                        "--rate", str(self.rate), "--channels", "2",
                        "--format", "s16", "--latency", lat, "-"]
            try:
                self.rec = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, bufsize=0)
                self.play = subprocess.Popen(play_cmd, stdin=self.rec.stdout,
                                             stderr=subprocess.PIPE)
                if self.rec.stdout:
                    self.rec.stdout.close()     # let play own the read end, so EOF isn't missed
            except FileNotFoundError as e:
                self.last_error = f"找不到 {e.filename}（需要 pipewire-bin）"
                self._kill()
                return False
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)[:120]
                self._kill()
                return False
            self.started_at = time.time()
            self.starts += 1
            self.last_error = ""
        return True

    def stop(self) -> None:
        with self._lock:
            self._kill()
        print("[monitor] 停止回放", flush=True)

    def _kill(self) -> None:
        for p in (self.play, self.rec):
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
        for p in (self.play, self.rec):
            if p is not None:
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        # Drain stderr after killing the children, otherwise the zombie pipe accumulates garbage
        for p in (self.play, self.rec):
            if p is not None and p.stderr:
                try:
                    err = p.stderr.read() or b""
                    if err and not self.last_error:
                        self.last_error = err.decode("utf-8", "replace").strip()[:120]
                except (OSError, ValueError):
                    pass
        self.rec = None
        self.play = None

    def status(self) -> dict:
        return {"on": self.running, "src": self.src, "sink": self.sink,
                "rate": self.rate, "latency_ms": self.latency_ms,
                "starts": self.starts,
                "up_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
                "error": self.last_error}


# ------------------------------------------------------------------ segmentation
class Segmenter:
    """Energy threshold plus silence hold. The floor is estimated from the opening quiet samples, then drops fast, rises slow.

    A hard cut (at max_s) looks back for the lowest-energy frame to cut on: slicing mid-syllable makes ASR emit
    half a word ("天真好" cut to "天真" translates as "sky"); backing into a syllable gap costs a few hundred ms but kills typos.

    The pause gate tiers by segment length (2026-11-21):
    a fixed 0.42s gate is a dilemma -- set tight, small inter-word pauses shred the speech
    (measured: "140 亿美元..." split into "...况且140。" + "亿美元对台均售...", number and unit parted);
    set loose, a finished sentence waits nearly a second before captions appear and all latency goes to waiting.
    So the gate follows "how long has this segment been accumulating":
      voiced < mid_s  -> wait for a long hold_long_s pause. Still short: better merged with the next sentence than shredded.
      voiced < long_s -> use the base hold_s; around 0.4s roughly matches a normal inter-sentence pause.
      voiced >= long_s -> cut at hold_short_s. Long enough already: accept a short pause, latency must not grow.
    max_s is still the fallback ceiling, but with tiering, hitting it degrades from "normal" to "occasional".
    """

    FLOOR_MIN, FLOOR_MAX = -70.0, -32.0

    def __init__(self, margin_db: float = 8.0, hold_s: float = 0.42,
                 min_s: float = 0.25, max_s: float = 3.0, pre_s: float = 0.25,
                 warm_s: float = 1.2, soft_s: float = 0.7,
                 hold_long_s: float = 0.90, hold_short_s: float = 0.30,
                 mid_s: float = 1.20, long_s: float = 2.40):
        self.margin_db = margin_db
        self.hold_frames = max(1, int(hold_s / FRAME_S))
        self.min_frames = max(1, int(min_s / FRAME_S))
        self.max_frames = max(1, int(max_s / FRAME_S))
        self.pre_frames = max(0, int(pre_s / FRAME_S))
        self.warm_frames = max(1, int(warm_s / FRAME_S))
        self.soft_frames = max(0, int(soft_s / FRAME_S))
        # Tiered gates: long-segment gate >= base, short-segment gate <= base, so a reversed argument can't break the logic
        self.hold_long_frames = max(self.hold_frames, int(hold_long_s / FRAME_S))
        self.hold_short_frames = min(self.hold_frames,
                                     max(1, int(hold_short_s / FRAME_S)))
        self.mid_frames = max(1, int(mid_s / FRAME_S))
        self.long_frames = max(self.mid_frames + 1, int(long_s / FRAME_S))

        self.floor = -70.0
        self._warm = 0
        self._samples: list[float] = []
        self._pre: collections.deque = collections.deque(maxlen=max(1, self.pre_frames))
        self._pre_db: collections.deque = collections.deque(maxlen=max(1, self.pre_frames))
        self._cur: list[np.ndarray] = []
        self._cur_db: list[float] = []
        self._sil = 0
        self._voiced_run = 0
        self.last_db = -99.0
        self.level_db = -99.0
        self.last_cut = ""          # "silence"=cut at a natural pause / "hard"=cut at the max_s ceiling

    def threshold(self) -> float:
        return self.floor + self.margin_db

    def hold_need(self) -> int:
        """Silence needed (in frames) before this segment may be cut. The longer the segment, the looser the requirement."""
        voiced = len(self._cur) - self._sil
        if voiced < self.mid_frames:
            return self.hold_long_frames
        if voiced < self.long_frames:
            return self.hold_frames
        return self.hold_short_frames

    def _clamp(self, v: float) -> float:
        return min(self.FLOOR_MAX, max(self.FLOOR_MIN, v))

    def push(self, frame: np.ndarray, db: float) -> np.ndarray | None:
        self.last_db = db
        # Slow-decay envelope for the meter so it doesn't flicker; floor at -99 so digital silence can't sink to -290
        self.level_db = max(-99.0, db, self.level_db - 1.2)

        # First 1.2s only estimates the floor, no speech decision; digital silence (-91) is not a valid sample
        if self._warm < self.warm_frames:
            self._warm += 1
            if db > -80.0:
                self._samples.append(db)
            if self._warm >= self.warm_frames and self._samples:
                self.floor = self._clamp(float(np.percentile(self._samples, 10)))
            self._pre.append(frame)             # keep pre-roll during warm-up too, so the first sentence's onset isn't lost
            self._pre_db.append(db)
            return None

        # Fast drop, slow rise: snap to the floor when quiet, barely move while speaking.
        # Digital silence (below -80, the floor when HDMI has no stream) is not "room floor" and
        # never updates it, else floor slides to the minimum and the gate rots until even AC hum counts as speech.
        if db > -80.0:
            if db < self.floor:
                self.floor = self._clamp(0.70 * self.floor + 0.30 * db)
            elif db < self.floor + 4.0:
                self.floor = self._clamp(0.995 * self.floor + 0.005 * db)

        thr = self.threshold()
        voiced = db > thr

        if not self._cur:
            self._pre.append(frame)
            self._pre_db.append(db)
            if voiced:
                self._voiced_run += 1
                if self._voiced_run >= 3:        # require 60ms in a row, which rejects clicks
                    self._cur = list(self._pre)
                    self._cur_db = list(self._pre_db)
                    self._pre.clear()
                    self._pre_db.clear()
                    self._sil = 0
                    self._voiced_run = 0
            else:
                self._voiced_run = 0
            return None

        self._cur.append(frame)
        self._cur_db.append(db)
        self._sil = 0 if voiced else self._sil + 1

        if self._sil >= self.hold_need() and len(self._cur) - self._sil >= self.min_frames:
            self.last_cut = "silence"
            return self._flush()
        if len(self._cur) >= self.max_frames:
            self.last_cut = "hard"
            return self._flush(look_back=self.soft_frames)
        return None

    def draft(self) -> tuple[np.ndarray, float] | None:
        """Draft audio for the current segment (not yet cut).

        Offline ASR produces text only after the whole segment is spoken, and a segment can run max_s seconds --
        on screen that is "two seconds of motion, four or five of nothing"; the draft fills that gap with gray text.
        """
        if not self._cur:
            return None
        return np.concatenate(self._cur), len(self._cur) * FRAME_S

    def _flush(self, look_back: int = 0) -> np.ndarray:
        """Wrap up. With look_back>0, cut at the lowest-energy point in the tail, leaving the rest for the next segment."""
        cur, cur_db = self._cur, self._cur_db
        if look_back and len(cur) > look_back + self.min_frames:
            tail = cur_db[-look_back:]
            cut = len(cur) - look_back + int(np.argmin(tail))
            if cut >= self.min_frames and len(cur) - cut >= 1:
                seg = np.concatenate(cur[:cut]) if cut else np.zeros(0, dtype=np.int16)
                self._cur = cur[cut:]
                self._cur_db = cur_db[cut:]
                self._sil = 0
                self._pre.clear()
                self._pre_db.clear()
                return seg
        seg = np.concatenate(cur) if cur else np.zeros(0, dtype=np.int16)
        self._cur = []
        self._cur_db = []
        self._pre.clear()
        self._pre_db.clear()
        self._sil = 0
        return seg


def seg_tuning(args) -> dict:
    """The four tiered-pause tuning knobs. Centralized here so live capture and --wav offline cannot diverge."""
    return {"hold_long_s": getattr(args, "hold_long", 0.90),
            "hold_short_s": getattr(args, "hold_short", 0.30),
            "mid_s": getattr(args, "seg_mid", 1.20),
            "long_s": getattr(args, "seg_long", 2.40)}


# ------------------------------------------------------------------ translation
# ------------------------------------------------------------------ filler removal
# Only the text handed to the translator is cleaned; the on-page source keeps every word:
# a verbatim transcript has value in a meeting, but fillers make the model treat "嗯...那个" as content words
# (measured: "嗯" -> "はい/ええ", "あの" -> "那个", skewing the whole sentence).
# The list is deliberately conservative -- pure fillers only. The "那个" in "那个项目" and
# the "あの" in "あの人" precede content words and are excluded by the "bounded by punctuation" rule.
_FILLERS = (
    "嗯嗯", "嗯", "呃", "额", "唔", "唉", "哎",
    "えーと", "えっと", "えー", "あのー", "そのー", "んー", "あの", "その",
    "umm", "um", "uhh", "uh", "erm", "er", "hmm", "ah",
)
_SEP = r"[\s，,、。．.!！?？…~～]"
_FILLER_RE = re.compile(
    rf"(?<![^{_SEP[1:-1]}])"                       # preceded by start of text or punctuation
    rf"(?:{'|'.join(map(re.escape, _FILLERS))})"   # a single filler token
    rf"(?![^{_SEP[1:-1]}])"                        # followed by punctuation or end of text
    rf"{_SEP}*",                                   # swallow the trailing separators too
    re.I)


def strip_fillers(text: str) -> str:
    """Strip pure fillers for the translator. If the whole sentence is fillers, return it as is -- never an empty string.

    「嗯，那个项目怎么样了」 -> 「那个项目怎么样了」
    「あの人は誰」 -> untouched (あの is followed by a content word)
    """
    if not text:
        return text
    out = _FILLER_RE.sub("", text)
    out = re.sub(rf"^{_SEP}+", "", out)            # leftover leading punctuation
    out = re.sub(r"([，,、])[，,、\s]+", r"\1", out)  # 「，，」 -> 「，」 (collapse doubled commas)
    out = re.sub(r"[，,、]\s*([。！？!?])", r"\1", out)
    out = out.strip()
    if not re.search(r"[^\s，,、。．.!！?？…~～]", out):
        return text                                # everything got stripped: fall back to the original
    return out


# Post-translation check: source-language features left in the output means it wasn't translated.
# Prompts only lower the odds of zh/ja copy-through, so a deterministic fallback check is required.
# For ja->zh, looking for kana is **zero false positive**; for zh->ja only Simplified-only glyphs work
# (Japanese uses traditional or different forms: 說/説, 對/対, 這/这, 時/时, 個/个).
# Never add 「会」「都」「那」 -- they are common kanji in Japanese and would flag valid translations.
_LEFTOVER = {
    "zh": re.compile(r"[\u3040-\u30FF\uFF66-\uFF9F]"),
    "ja": re.compile(r"[们给这说对很没让从把么吗呢吧啊嗯呃时个]"),
}
_RETRY_HINT = {
    "zh": "【注意】上一版译文混入了日文。这次必须从头到尾都是中文，"
          "一个日文词、一个假名都不许留（人名机构名除外）。",
    "ja": "【注意】上一版译文混入了中文。这次必须从头到尾都是日语，"
          "一个中文词都不许留（简体字「们给这说对很没让从把么吗呢吧」都是错的）。",
}
TR_STATS = {"retry": 0, "fixed": 0}         # exposes the retry hit rate via /api/status


def lang_clean(trans: str, target: str) -> bool:
    """Whether a translation still carries source-language residue. Languages without a test are passed.

    For zh->ja the character list only catches "mixed writing", not **whole-sentence copying**: 「我是主持人田英和」
    is all kanji that Japanese also uses, so a character scan looks clean although nothing was translated.
    Hence a structural test: sentence-length Japanese cannot be kana-free
    (particles は/が/を/に/の and endings です/ます necessarily carry kana). Pure kanji >= 6 chars counts as dirty.

    For ja->zh the risk is the opposite, false positives: proper nouns may keep their original form
    (「深セン」 is the Japanese spelling of Shenzhen), so a few kana in long Chinese text should not burn a retry.
    """
    if target == "ja" and len(trans) >= 6 and not re.search(r"[\u3040-\u30FF]", trans):
        return False
    if target == "zh":
        kana = len(re.findall(r"[\u3040-\u30FF]", trans))
        if kana == 0:
            return True
        cjk = len(re.findall(r"[\u4e00-\u9FFF]", trans))
        return kana <= 3 and cjk >= kana * 2      # a few kana + a long run of kanji = proper noun
    pat = _LEFTOVER.get(target)
    return True if pat is None else pat.search(trans) is None


# Magnitude's /inference/v1 requires Bearer auth; vLLM does not. Set via --llm-key.
_LLM_KEY = ""

# Translation prompt tier: full (long version with counter-examples, tuned in the 7B era) /
# short (the 35B short version). Measured in README 4.19: short cuts system from ~250 tokens to ~70.
TR_STYLE = "full"

# Engine-side timings from the latest translate_stream call (prompt_n / cache_n / prompt_ms).
# Used only as evidence by offline tools; not part of the business logic.
LAST_TIMINGS: dict = {}


def _llm_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _LLM_KEY:
        h["Authorization"] = f"Bearer {_LLM_KEY}"
    return h


def _tr_system_short(target: str) -> str:
    """Short version for the 35B model (~100 chars). Only three hard rules survive; examples and filler rules are cut.

    Why that is safe: the full `_tr_system` suppresses zh/ja kanji copy-through with "counter-examples + self-check",
    but it costs 250-300 tokens, the largest fixed cost in every request. 35B-A3B prefill reads nearly all
    experts, so token count converts straight to milliseconds -- hence measuring "does it hold without examples".
    The three rules that cannot go:
    - target-language-only plus the shared-kanji trap (weakest link without examples; held up by wording alone)
    - fix homophone typos from context (otherwise the typo gets translated faithfully)
    - translate a half sentence as a half sentence (short segments produce many of them)
    """
    zh = (
        "你是中日会议同传。把【本句】译成纯中文，只输出译文一行，不加解释。\n"
        "按【上文】纠正识别错字。中日共用汉字，禁止把日语原样当中文；人名可留原文。半句译半句。"
    )
    ja = (
        "你是中日会议同传。把【本句】译成纯日语，只输出译文一行，不加解释。\n"
        "按【上文】纠正识别错字。中日共用汉字，禁止把中文原样当日语"
        "（不得出现「们」「给」「这」「了」）；人名可留原文。半句译半句。"
    )
    if target == "zh":
        return zh
    if target == "ja":
        return ja
    name = target
    return (f"你是会议同传。把【本句】译成纯{name}，只输出译文一行；"
            f"按上文纠正识别错字；半句译半句。")


def _tr_system(target: str, style: str | None = None) -> str:
    """System prompt for translation. One per target language; a fixed prefix is what hits the vLLM cache.

    Why it looks like this (all learned the hard way):
    - **Shared zh/ja kanji is where things break most**. A 7B model copies words like 「坚持」「好处」「主持人」
      verbatim and emits "Japanese" with flawless kanji and broken grammar. Saying "must be pure target
      language" is not enough: concrete counter-examples plus a self-check are required.
    - **ASR homophone typos must be explicitly delegated for fixing**. Otherwise the model translates the
      typo faithfully ("中方科意" -> "中国側は科意", a word that doesn't exist).
    - Counter-examples use real bad outputs captured in testing; imitation beats abstract rules.
    """
    if (style or TR_STYLE) == "short":
        return _tr_system_short(target)
    zh = (
        "你是中日会议同传。把【本句】译成中文。\n"
        "第一步：按【上文】纠正语音识别的同音错字，译纠正后的意思而不是错字本身"
        "（如「科意」应为「刻意」、「三强贵物」应为「三强会晤」）。\n"
        "输出规则：\n"
        "1. 只输出译文一行，不要引号、不要解释、不要附原文。\n"
        "2. 译文必须 100% 是中文。中日共用汉字，最容易犯的错是把日语句子原样"
        "当中文输出。译完自查：只要有一处还是日语语法或残留假名，整句重写。\n"
        "3. 人名、机构名、缩写可保留原文写法，其余不得保留任何日文词。\n"
        "4. 原文是半句就译成半句，不要补主语、不要自行补完。\n"
        "5. 语气词（えーと、あの、ね、かな）略去不译；有实义的词要译。\n"
        "示例：\n"
        "原文：身体面にアメリカは坚持するべきです\n"
        "错误：在实质层面，美国应该坚持するべきです\n"
        "正确：在实质层面，美国应当坚持立场。\n"
        "\n"
        "原文：深zのアプリをどう評価するか\n"
        "错误：深zのアプリで何点给深zに評価するか\n"
        "正确：该怎么评价深z这款应用？"
    )
    ja = (
        "你是中日会议同传。把【本句】译成日语。\n"
        "第一步：按【上文】纠正语音识别的同音错字，译纠正后的意思而不是错字本身"
        "（如「科意」应为「刻意」、「三强贵物」应为「三强会晤」）。\n"
        "输出规则：\n"
        "1. 只输出译文一行，不要引号、不要解释、不要附原文。\n"
        "2. 译文必须 100% 是日语。中日共用汉字，最容易犯的错是把中文句子原样"
        "当日语输出。译完自查：只要有一处还是中文语法（如「们」「给」「这」「很」"
        "「了」），整句重写。\n"
        "3. 人名、机构名、缩写可保留原文写法，其余不得保留任何中文词。\n"
        "4. 原文是半句就译成半句，不要补主语、不要自行补完。\n"
        "5. 语气词（嗯、呃、那个）略去不译；有实义的词要译。\n"
        "示例：\n"
        "原文：我是主持人谭英文\n"
        "错误：私は主持人谭英文です\n"
        "正确：私は司会の谭英文です。\n"
        "\n"
        "原文：中方刻意给韩国好处\n"
        "错误：中国側は故意に韓国に给好处\n"
        "正确：中国側は意図的に韓国に便宜を図った。"
    )
    if target == "zh":
        return zh
    if target == "ja":
        return ja
    name = target
    return (f"你是会议同传。把【本句】译成{name}。只输出译文一行，不要引号或解释；"
            f"译文必须是纯{name}，不得保留任何原文词（专有名词除外）；"
            f"原文可能有语音识别错字，按上下文纠正后翻译；半句就译成半句，不要补完。")


# Qwen3.5 chat templates take the thinking branch whenever enable_thinking is undefined, so
# max_tokens=200 burns on reasoning and content stays empty -- captions show "no translation" and
# every line waits out the full timeout (measured 9~10s between lines). So it is disabled explicitly.
# Qwen2.5 templates ignore the variable, so passing it is harmless. Add --llm-think to enable thinking.
LLM_THINK = False


def _think_kw() -> dict:
    return {} if LLM_THINK else {"chat_template_kwargs": {"enable_thinking": False}}



def translate(base: str, model: str, text: str, target: str, timeout: float = 12.0,
              ctx: str = "", hint: str = "") -> str:
    """Output the translation only, short sentences read literally. Returns "" on failure so captions never block.

    ctx is the previous one or two source lines: short segments are often half sentences, and context aligns
    terminology, person, and tone; it is also the model's only basis for fixing homophone typos (a bare half
    sentence like "中方科意" gives no way to tell it means 「刻意」). hint is the retry reinforcement, used after lang_clean().
    """
    sys = _tr_system(target)
    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    if hint:
        user += "\n" + hint
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        "stream": False, "temperature": 0.1, "max_tokens": 200,
        **_think_kw(),
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers=_llm_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        out = (d["choices"][0]["message"]["content"] or "").strip()
        return out.strip('"“”「」 ')
    except Exception:  # noqa: BLE001
        return ""


def translate_stream(base: str, model: str, text: str, target: str,
                     timeout: float = 12.0, ctx: str = "",
                     on_delta=None, hint: str = "",
                     screen_hint: str = "") -> tuple[str, int, int]:
    """Streaming translation: call back as chunks arrive, return (full text, first-token ms, total ms).

    Why stream: vLLM prefix-caches the system prompt, so the **first token takes only ~96ms** while the
    full sentence takes 300~1300ms (the slow part is per-token generation, not prefill). Letting the
    translation surface character by character makes time-to-first-character 3~7x faster at identical total cost.

    Intermediate states are not stripped (that would flicker); cleaning happens once at the end.
    """
    sys = _tr_system(target)
    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    # The screen hint is non-empty only when this sentence has a homophone candidate; otherwise not a single token is added
    if screen_hint:
        user += "\n" + screen_hint
    if hint:
        user += "\n" + hint
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        "stream": True, "temperature": 0.1, "max_tokens": 200,
        **_think_kw(),
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers=_llm_headers())
    parts: list[str] = []
    t0 = time.time()
    ttft_ms = 0
    t_hdr = -1
    dbg: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            t_hdr = int((time.time() - t0) * 1000)   # connected and response headers received
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                if "timings" in ev and not dbg:
                    dbg = ev["timings"]
                ch = (ev.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content") or ""
                if not piece:
                    continue
                if not ttft_ms:
                    ttft_ms = int((time.time() - t0) * 1000)
                parts.append(piece)
                if on_delta is not None:
                    try:
                        on_delta("".join(parts))
                    except Exception:  # noqa: BLE001
                        pass          # a failed push must never crash the translation thread
    except Exception:  # noqa: BLE001
        # Still return whatever arrived: a partial translation beats a blank
        pass
    total_ms = int((time.time() - t0) * 1000)
    # Engine-side timings from the latest call. Offline tools (wav_seg_ab / ttft_prompt_ab) want
    # prompt_n / cache_n as evidence but don't want to rebuild the request body, so they read it here.
    global LAST_TIMINGS
    LAST_TIMINGS = dbg or {}
    if os.environ.get("CAP_DEBUG"):
        # Stage breakdown: hdr=connected and headers received, ttft=first token, total=whole sentence
        print(f"[tr] hdr={t_hdr}ms ttft={ttft_ms}ms total={total_ms}ms "
              f"| srv prompt_n={dbg.get('prompt_n')} cache_n={dbg.get('cache_n')} "
              f"prompt_ms={dbg.get('prompt_ms')} ttft_ms={dbg.get('time_to_first_token_ms')}",
              flush=True)
    final = "".join(parts).strip().strip('"“”「」 ')
    # Source-language residue triggers one retry (non-streaming, with the reinforcement hint).
    # Still dirty -> keep the first version: a mixed half sentence beats a blank, and retrying must not stall the thread.
    if final and not lang_clean(final, target):
        TR_STATS["retry"] += 1
        alt = translate(base, model, text, target, timeout=timeout, ctx=ctx,
                        hint=_RETRY_HINT.get(target, ""))
        if alt and lang_clean(alt, target):
            TR_STATS["fixed"] += 1
            total_ms = int((time.time() - t0) * 1000)
            if on_delta is not None:
                try:
                    on_delta(alt)          # replace the whole line with the clean version; applyUpd already replaces wholesale
                except Exception:  # noqa: BLE001
                    pass
            return alt, ttft_ms, total_ms
    return final, ttft_ms, total_ms


# ------------------------------------------------------------------ screen term hints
# Text already on screen (slide titles, lower-third name straps, window titles) is the cheapest
# source of proper-noun correction for ASR -- zero tokens, you only judge "is this a homophone".
#
# But **you must not stuff screen text into every sentence** (measured and rejected 2026-11-21):
#   - cost: 60 chars -> 72 extra prompt tokens -> first token 490->740ms; 120 chars is worse
#   - effect: 5 of 6 sentences rewritten, 3 of them for the worse ("三社" -> "首次会谈",
#     "三家公司" invented from nowhere, "特朗普那边" repeated), because the screen holds both our own
#     page (self-loop: the previous translation fed back as screen content) and images unrelated to the sentence
# So this only does "on-demand hints": emit one line only when a screen term and a span of this
# sentence are **phonetically close but written differently**, one term per span; no hit, add nothing.
_SCREEN_UI = ("会议字幕", "设备", "已识别", "延迟", "首字", "译完", "双语", "正在听",
              "停止", "清屏", "正在说", "运行中", "搜索", "创建", "Chrome",
              "youtube.com", "dB")

_SCREEN_RES = (
    re.compile(r"[0-9]{4,}"),                            # years / amounts / issue numbers
    re.compile(r"[ァ-ヴー]{4,10}"),                      # katakana nouns
    re.compile(r"[一-龥]{3,8}"),                         # kanji runs (names, organizations, terminology)
)
# Why no Latin letters: OpenJTalk reads "CloudflareObservability" letter by letter into 50 morae
# ("しーえるおー..."), never matching a Japanese reading, and just fills the list with emails and URLs
# (measured: 40 candidates, all such junk). An English word on screen that ASR already wrote correctly
# needs no hint; if it was heard as katakana, phonetics can't save it -- that case belongs to ASR hotwords.
_SCREEN_JUNK = ("@", "://", ".com", ".cn", ".jp", ".biz", ".net", "www.")
# Simplified-Chinese marker chars: text from chat windows or a Chinese UI mixed into a Japanese meeting's
# term list only creates false hits. Two or more marks a line as Chinese (Japanese does use 「的」, so one can't decide).
_SCREEN_CN = "这们为说时个还就没都很地对吗呢吧啊从给过应该"


def _hira(s: str) -> str:
    """Katakana to hiragana, keeping only characters whose reading can be compared (punctuation, Latin, digits are skipped)."""
    out = []
    for ch in s or "":
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        elif 0x3041 <= o <= 0x3096 or ch == "ー":
            out.append(ch)
    return "".join(out)


def _lev(a: str, b: str) -> int:
    """Edit distance. Only <=1 matters; a length gap over 1 short-circuits to 9 (skipping the whole table)."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class ScreenHints:
    """Look at the screen every few seconds in the background, compress it into a term list, hint the translator on demand.

    Why not open the capture card directly: `/dev/video0` is exclusive and voice-bench holds it permanently.
    So this goes through voice-bench's `/api/look` (which caches by dHash, making the same frame nearly free);
    the caption service never touches the device and pulls in no opencv/OCR dependency.
    """

    def __init__(self, api: str = "http://127.0.0.1:8095/api/look",
                 every: float = 12.0, max_terms: int = 40,
                 timeout: float = 45.0, recent=None, reader=None):
        self.api = api
        self.every = max(4.0, every)
        self.max_terms = max(4, max_terms)
        self.timeout = timeout
        self.recent = recent or (lambda: [])
        self._reader = reader          # test injection: text -> [(surface, hiragana reading)]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.terms: list[tuple[str, str]] = []
        self.polls = 0
        self.hits = 0
        self.fails = 0
        self.lines_n = 0
        self.hash = ""
        self.at = ""
        self.last_ms = 0
        self.last_err = ""
        self.last_hint = ""

    # ------------------------------------------------------------ readings
    def _morphemes(self, text: str) -> list[tuple[str, str]]:
        if self._reader is not None:
            return list(self._reader(text))
        try:
            import pyopenjtalk            # already shipped with the voice assistant's TTS, no extra install
        except Exception:                 # noqa: BLE001
            self.last_err = "没有 pyopenjtalk，屏幕提示不可用"
            return []
        try:
            return [(n.get("string", ""), _hira(n.get("read") or ""))
                    for n in pyopenjtalk.run_frontend(text)]
        except Exception as e:            # noqa: BLE001
            self.last_err = f"{type(e).__name__}: {e}"
            return []

    @staticmethod
    def _windows(morphs: list[tuple[str, str]],
                 max_join: int = 4, max_mora: int = 14):
        """Join morphemes into windows of 1~4, keeping only readings of 4+ morae (shorter ones always collide)."""
        out = []
        for i in range(len(morphs)):
            surf, rd = "", ""
            for j in range(i, min(i + max_join, len(morphs))):
                surf += morphs[j][0]
                rd += morphs[j][1]
                if len(rd) > max_mora:
                    break
                if len(rd) >= 4:
                    out.append((surf, rd))
        return out

    # ------------------------------------------------------------ grabbing the screen
    def _keep(self, lines: list[str]) -> list[str]:
        recent = [re.sub(r"\s+", "", x) for x in self.recent() if x]
        out = []
        for ln in lines:
            ln = ln.strip()
            if len(ln) < 3:
                continue
            if any(w in ln for w in _SCREEN_UI):
                continue
            if any(w in ln for w in _SCREEN_JUNK):   # emails / URLs: the whole line is worthless
                continue
            if sum(ln.count(c) for c in _SCREEN_CN) >= 2:   # Chinese UI / chat window
                continue
            # 8+ chars in common with a recent caption -> assumed copied from our own page, drop it
            flat = re.sub(r"\s+", "", ln)
            junk = False
            for r in recent:
                for i in range(0, len(r) - 7):
                    if r[i:i + 8] in flat:
                        junk = True
                        break
                if junk:
                    break
            if not junk:
                out.append(ln)
        return out

    def set_text(self, text: str) -> None:
        """Compress one screen of OCR text into a term list (filter UI and self-loop, take readings, cap the count)."""
        lines = [x for x in (text or "").splitlines() if x.strip()]
        self.lines_n = len(lines)
        kept = self._keep(lines)
        seen: dict[str, str] = {}
        for ln in kept:
            for rx in _SCREEN_RES:
                for m in rx.finditer(ln):
                    w = m.group()
                    if w in seen:
                        continue
                    rd = "".join(r for _s, r in self._morphemes(w))
                    if not 4 <= len(rd) <= 14:   # too short always collides, too long reads inaccurately
                        continue
                    seen[w] = rd
        # Longer readings first: the longer it is, the more specific and the less likely a false hit
        terms = sorted(seen.items(), key=lambda kv: -len(kv[1]))[:self.max_terms]
        with self._lock:
            self.terms = terms

    def fetch(self) -> None:
        t0 = time.time()
        req = urllib.request.Request(
            self.api, data=b"{}", method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            info = json.loads(r.read().decode("utf-8", "replace") or "{}")
        self.polls += 1
        self.last_ms = int((time.time() - t0) * 1000)
        if not info.get("ok"):
            self.fails += 1
            self.last_err = str(info.get("message") or info.get("reason") or "看屏幕失败")[:120]
            return
        self.hash = (info.get("hash") or self.hash)[:16]
        self.at = time.strftime("%H:%M:%S")
        self.set_text(info.get("text") or "")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.fetch()
            except Exception as e:        # noqa: BLE001
                self.fails += 1
                self.last_err = f"{type(e).__name__}: {e}"[:120]
            self._stop.wait(self.every)

    def start(self) -> None:
        self._t = threading.Thread(target=self._loop, name="screen-hints", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------ hits
    def suggest(self, text: str) -> str:
        """Return one hint line on a hit, otherwise "" (an empty string adds zero tokens to the request)."""
        with self._lock:
            terms = list(self.terms)
        if not terms or len(text) < 4:
            return ""
        wins = self._windows(self._morphemes(text))
        if not wins:
            return ""
        for word, wrd in terms:
            if word in text:              # already written correctly, no hint needed
                continue
            for surf, rd in wins:
                if surf == word or abs(len(rd) - len(wrd)) > 1:
                    continue
                # Share at least one character: real homophones look like 「畔蒜泰助 / 阪井泰介」 (share 「泰」)
                # or 「小委員会 / 小委部」 (share 「小委」). Coincidences that share no character are almost
                # always false hits (especially when Chinese UI terms are forced into Japanese readings by OpenJTalk).
                if not (set(surf) & set(word)):
                    continue
                if _lev(rd, wrd) <= 1:
                    self.hits += 1
                    self.last_hint = (f"【屏幕词】「{word}」——本句的「{surf}」"
                                      f"若指同一事物，请改用屏幕写法。")
                    return self.last_hint
        return ""

    def status(self) -> dict:
        with self._lock:
            n = len(self.terms)
            sample = [w for w, _r in self.terms[:6]]
        return {"on": True, "polls": self.polls, "hits": self.hits, "fails": self.fails,
                "terms": n, "sample": sample,
                "lines": self.lines_n, "hash": self.hash, "at": self.at,
                "last_ms": self.last_ms, "last_err": self.last_err or None,
                "last_hint": self.last_hint or None}


# ------------------------------------------------------------------ engine
class Captioner:
    """Recognition plus bilingual captions.

    The online path is a three-stage pipeline: `submit()` enqueues -> the asr thread recognizes -> a
    source-only caption line lands immediately -> the translation thread streams `trans` back into the same
    line (first token pushed at ~96ms) and registers a patch for the frontend; text never waits for translation.
    The offline path `handle()` runs synchronously with equivalent behavior, for wav self-tests and unit tests.
    """

    def __init__(self, asr: Asr, *, bilingual: bool = True,
                 llm_base: str = "http://127.0.0.1:8000/v1", llm_model: str = "qwen",
                 asr_workers: int = 1, tr_workers: int = 2, ctx_lines: int = 2,
                 stream_trans: bool = True, draft_on: bool = True,
                 draft_every: float = 0.8, screen=None):
        self.asr = asr
        self.bilingual = bilingual
        # Screen term hints (optional). A hint is passed to the translator only on a hit; see ScreenHints.
        self.screen = screen
        self.llm_base = llm_base
        self.llm_model = llm_model
        self.ctx_lines = ctx_lines
        self.stream_trans = stream_trans
        self.lines: collections.deque = collections.deque(maxlen=400)
        self.seq = itertools.count(1)
        self.segments = 0
        self.errors = 0
        self.dropped = 0
        self.started = time.time()
        self.asr_ms = 0
        self.trans_ms = 0
        self.ttft_ms = 0
        self.lag_ms = 0
        self.asr_workers = max(1, asr_workers)
        self.tr_workers = max(1, tr_workers)
        self._q_asr: queue.Queue = queue.Queue(maxsize=24)
        self._q_tr: queue.Queue = queue.Queue(maxsize=64)
        self._updates: collections.deque = collections.deque(maxlen=120)
        self._lock = threading.RLock()
        # The draft thread and real recognition share one recognizer. sherpa-onnx supports per-thread
        # create_stream, but decoding both at once fights over CPU cores; serializing costs the draft
        # only a 200ms yield while keeping the real path's lag stable.
        self._asr_lock = threading.Lock()
        self._started = False

        # Draft captions: gray text while speaking, replaced by the final line when the segment ends.
        # Single-slot queue (newest only): when draft recognition lags the speech rate, stale ones are dropped, never queued.
        self.draft_on = draft_on
        self.draft_every = draft_every
        self.draft_text = ""
        self.draft_epoch = 0
        self.draft_ms = 0
        self.drafts = 0
        self._q_draft: queue.Queue = queue.Queue(maxsize=1)

    @property
    def busy(self) -> bool:
        return not self._q_asr.empty() or not self._q_tr.empty()

    def start_workers(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        for i in range(self.asr_workers):
            threading.Thread(target=self._asr_loop, name=f"asr{i}", daemon=True).start()
        for i in range(self.tr_workers):
            threading.Thread(target=self._tr_loop, name=f"tr{i}", daemon=True).start()
        if self.draft_on:
            threading.Thread(target=self._draft_loop, name="draft", daemon=True).start()

    def submit(self, pcm: np.ndarray, dur: float, end_ts: float | None = None) -> None:
        """The capture thread only drops segments in here; it never waits for recognition."""
        if not self._started:
            self.start_workers()
        # This segment is final: the draft is void, and any draft result still in flight is rejected by epoch
        with self._lock:
            self.draft_epoch += 1
            self.draft_text = ""
        item = (pcm, dur, end_ts if end_ts else time.time())
        try:
            self._q_asr.put_nowait(item)
        except queue.Full:
            # Rather drop the oldest segment than let captions drift further behind (10s-late captions are worthless)
            self.dropped += 1
            try:
                self._q_asr.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q_asr.put_nowait(item)
            except queue.Full:
                pass

    def push_draft(self, pcm: np.ndarray, dur: float) -> None:
        """The capture thread drops "the current segment's audio so far" in for draft recognition.

        Single slot: a draft is a temporary state for humans, old ones are worthless, a new one replaces it.
        Never queue -- queuing makes the draft fall further and further behind until it shows speech from seconds ago.
        """
        if not (self.draft_on and self._started):
            return
        if self._q_draft.full():
            try:
                self._q_draft.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q_draft.put_nowait((pcm, dur, self.draft_epoch))
        except queue.Full:
            pass

    def _draft_loop(self) -> None:
        while True:
            pcm, _dur, ep = self._q_draft.get()
            # A draft is meaningless while a real segment is queued (it will be replaced right away), so step aside
            if not self._q_asr.empty():
                continue
            t0 = time.time()
            try:
                with self._asr_lock:
                    text, lang = self.recognize(pcm)
            except Exception:  # noqa: BLE001
                continue
            self.draft_ms = int((time.time() - t0) * 1000)
            self.drafts += 1
            with self._lock:
                # The segment may have gone final during recognition (submit bumps epoch), so drop the result
                if ep != self.draft_epoch:
                    continue
                self.draft_text = text
            if text:
                print(f"  [草稿] {self.draft_ms}ms {lang} {text}", flush=True)

    def take_updates(self) -> list[dict]:
        """Translation patches: drained on read. The line object's trans is also patched, so history stays complete."""
        out: list[dict] = []
        with self._lock:
            while self._updates:
                out.append(self._updates.popleft())
        return out

    def _push_update(self, seq: int, trans: str, final: bool,
                     trans_ms: int = 0, ttft_ms: int = 0) -> None:
        """Register a translation patch; only the newest one per seq is kept.

        The frontend polls every 250ms while streaming emits a delta every ~80ms. Without merging, one poll
        would receive 3 intermediate states for the same line and the page would repaint twice for nothing.
        """
        with self._lock:
            for u in self._updates:
                if u["seq"] == seq:
                    u["trans"] = trans
                    u["final"] = final
                    if final:
                        u["trans_ms"] = trans_ms
                        u["ttft_ms"] = ttft_ms
                    return
            self._updates.append({"seq": seq, "trans": trans, "final": final,
                                  "trans_ms": trans_ms, "ttft_ms": ttft_ms})

    def clear(self) -> None:
        with self._lock:
            self.lines.clear()
            self._updates.clear()
            self.draft_text = ""
            self.draft_epoch += 1

    def _ctx_before(self, seq: int) -> str:
        # ctx_lines=0 must really mean "no context": [-0:] equals [0:] and would stuff in the whole history,
        # which once inflated the prompt from 80 to 138 tokens and added 300ms to the first token.
        if self.ctx_lines <= 0:
            return ""
        with self._lock:
            prev = [l["text"] for l in self.lines if l["seq"] < seq][-self.ctx_lines:]
        return " / ".join(prev)

    def recognize(self, pcm: np.ndarray) -> tuple[str, str]:
        """Return (text, language). Language comes from SenseVoice's <|ja|> tag; if absent, guess from the script."""
        x = vp.highpass(pcm.astype(np.float64), 150.0, SR, 2)
        s = self.asr.rec.create_stream()
        s.accept_waveform(SR, (x / vp.FULL_SCALE).astype(np.float32))
        self.asr.rec.decode_stream(s)
        raw = s.result.text or ""
        m = LANG_TAG.search(raw)
        lang = m.group(1) if m else ("ja" if KANA.search(raw) else "zh" if CJK.search(raw) else "")
        text = clean_text(raw)
        if not SUBSTANCE.search(text):          # punctuation / whitespace only = noise segment, not a sentence
            return "", lang
        return text, lang

    def _asr_loop(self) -> None:
        while True:
            pcm, dur, end_ts = self._q_asr.get()
            t0 = time.time()
            try:
                with self._asr_lock:
                    text, lang = self.recognize(pcm)
            except Exception:  # noqa: BLE001
                self.errors += 1
                continue
            self.asr_ms = int((time.time() - t0) * 1000)
            if not text:
                continue
            lag = int((time.time() - end_ts) * 1000)
            self.lag_ms = lag
            want = self.bilingual and lang in ("zh", "ja")
            line = {
                "seq": next(self.seq), "text": text, "lang": lang, "trans": "",
                "pending": want, "dur": round(dur, 2), "asr_ms": self.asr_ms,
                "lag_ms": lag, "at": time.strftime("%H:%M:%S"),
            }
            with self._lock:
                self.segments += 1
                self.lines.append(line)
            print(f"  [{line['at']}] #{line['seq']} lag={lag}ms asr={self.asr_ms}ms "
                  f"q={self._q_tr.qsize()} {line['lang']} {text}", flush=True)
            if want:
                try:
                    self._q_tr.put_nowait(line)
                except queue.Full:
                    line["pending"] = False

    def _tr_loop(self) -> None:
        while True:
            line = self._q_tr.get()
            target = "ja" if line["lang"] == "zh" else "zh"
            # Clean only the text handed to the translator; line["text"] (shown on the page) stays verbatim
            ctx = strip_fillers(self._ctx_before(line["seq"]))
            src = strip_fillers(line["text"])
            seq = line["seq"]

            if not self.stream_trans:
                t0 = time.time()
                trans = translate(self.llm_base, self.llm_model, src, target, ctx=ctx)
                dt = int((time.time() - t0) * 1000)
                line["pending"] = False
                if not trans:
                    line["failed"] = True
                    continue
                line["trans"] = trans
                line["trans_ms"] = dt
                self.trans_ms = dt
                self._push_update(seq, trans, True, dt, 0)
                continue

            # Streaming: first token pushed at ~96ms, the whole sentence completes in 300~1300ms
            state = {"last": 0.0, "last_len": 0}
            # Screen term hints: a hint only when "the reading matches but the spelling differs"; otherwise "", zero cost
            sh = self.screen.suggest(src) if self.screen is not None else ""
            if sh:
                line["screen_hint"] = sh

            def on_delta(txt: str, _seq=seq, _line=line, _st=state) -> None:
                now = time.time()
                # Throttle to ~80ms or +4 chars; pushes merge by seq anyway, so extra ones don't cause repaints
                if now - _st["last"] < 0.08 and len(txt) - _st["last_len"] < 4:
                    return
                _st["last"] = now
                _st["last_len"] = len(txt)
                _line["trans"] = txt
                self._push_update(_seq, txt, False)

            trans, ttft, total = translate_stream(
                self.llm_base, self.llm_model, src, target, ctx=ctx,
                on_delta=on_delta, screen_hint=sh)
            line["pending"] = False
            line["ttft_ms"] = ttft
            if not trans:
                line["failed"] = True
                continue
            line["trans"] = trans
            line["trans_ms"] = total
            self.trans_ms = total
            self.ttft_ms = ttft
            self._push_update(seq, trans, True, total, ttft)

    def handle(self, pcm: np.ndarray, dur: float) -> dict | None:
        """Run one segment synchronously: for offline wav self-tests. The online path uses submit()."""
        t0 = time.time()
        try:
            text, lang = self.recognize(pcm)
        except Exception:  # noqa: BLE001
            self.errors += 1
            return None
        asr_s = time.time() - t0
        if not text:
            return None

        trans = ""
        if self.bilingual and lang in ("zh", "ja"):
            target = "ja" if lang == "zh" else "zh"
            t1 = time.time()
            trans = translate(self.llm_base, self.llm_model, strip_fillers(text),
                              target,
                              ctx=strip_fillers(self._ctx_before(10 ** 9)))
            self.trans_ms = int((time.time() - t1) * 1000)

        line = {
            "seq": next(self.seq), "text": text, "lang": lang, "trans": trans,
            "pending": False, "dur": round(dur, 2), "asr_ms": int(asr_s * 1000),
            "lag_ms": int(asr_s * 1000), "at": time.strftime("%H:%M:%S"),
        }
        with self._lock:
            self.segments += 1
            self.lines.append(line)
        return line

    def draft_state(self) -> dict:
        """What the draft line should show right now. An empty string means no draft; the frontend clears the gray line."""
        with self._lock:
            return {"text": self.draft_text, "epoch": self.draft_epoch,
                    "ms": self.draft_ms}

    def status(self) -> dict:
        return {"segments": self.segments, "errors": self.errors, "busy": self.busy,
                "bilingual": self.bilingual, "up_s": round(time.time() - self.started, 1),
                "asr_ms": self.asr_ms, "trans_ms": self.trans_ms,
                "ttft_ms": self.ttft_ms, "stream_trans": self.stream_trans,
                "lag_ms": self.lag_ms, "dropped": self.dropped,
                "draft_on": self.draft_on, "drafts": self.drafts,
                "draft_ms": self.draft_ms, "tr_prompt": TR_STYLE,
                "q_asr": self._q_asr.qsize(), "q_tr": self._q_tr.qsize(),
                "screen": self.screen.status() if self.screen is not None else None}


# ------------------------------------------------------------------ page
# Note: this must be a raw string. PAGE embeds JS; as a non-raw string Python would turn the JS
# '\n' into a real newline, truncating single-quoted strings -> the script fails to parse -> page sticks at "连接中…".
PAGE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>会议字幕</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{--bg:#0b0d11;--panel:#151920;--line:#252b36;--fg:#e8eefc;--dim:#7f8a9e;
        --accent:#5b9cff;--ok:#41c48a;--warn:#e0a34a;--err:#e0645a}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font-family:"Noto Sans CJK JP","Noto Sans CJK SC",system-ui,sans-serif}
  header{position:sticky;top:0;z-index:9;background:rgba(11,13,17,.92);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--line);
    padding:14px 22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.5px}
  .meter{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--dim)}
  .bar{width:140px;height:6px;background:#1d222b;border-radius:99px;overflow:hidden}
  .bar i{display:block;height:100%;width:0;background:var(--ok);transition:width .12s}
  .pill{font-size:12px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);
    background:var(--panel);color:var(--dim)}
  .pill b{color:var(--fg)}
  main{padding:16px 22px 60px;max-width:1100px;margin:0 auto;
    overflow-anchor:none}   /* we manage the scroll anchor ourselves; keep browser scroll anchoring out */
  .line{padding:10px 0;border-bottom:1px dashed #1e242e}
  .line .t{font-size:13px;color:#5f6a7d;font-variant-numeric:tabular-nums}
  .line .o{font-size:26px;line-height:1.45;margin:2px 0 0}
  .line .tr{font-size:19px;line-height:1.4;color:#9fb4d6;margin:4px 0 0}
  .line .tr.pend{color:#4d5766;font-style:italic;font-size:16px}
  .line .tr .cur{color:var(--accent);animation:blink 1s steps(2) infinite}
  @keyframes blink{50%{opacity:0}}
  .ja .o{color:#dbe8ff} .zh .o{color:#e6ffe9}
  .empty{color:#4d5766;font-style:italic;padding:40px 0;text-align:center}
  .err{color:var(--err);font-size:13px}
  button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--line);
    background:var(--panel);color:var(--fg);padding:6px 14px;font-size:13px;transition:.15s}
  button:hover{border-color:#3a4356;background:#1c2029}
  #stop.run{border-color:#6b3438;background:#2a191b;color:#ffb3ad}
  #stop.idle{border-color:#2f6b52;background:#152a22;color:#a8e6c8}
  #mon.run{border-color:#2f6b52;background:#152a22;color:#a8e6c8}
  .spacer{flex:1}
  /* Draft line: gray text while speaking, replaced by the final line when the segment ends.
     Deliberately dimmed and italic so it reads as "this may still change" and never gets quoted as final. */
  .line.draft{opacity:.62}
  .line.draft .o{font-style:italic;color:#b9c2d4}
  .line.draft .t{color:#5f6a7d}
</style></head><body>
<header>
  <h1>会议字幕</h1>
  <span class="pill">设备 <b>__DEV__</b></span>
  <span class="pill">已识别 <b id="seg">0</b> 段</span>
  <span class="pill" id="lag">延迟 <b>—</b></span>
  <span class="pill" id="bi">双语</span>
  <span class="pill" id="scr" style="display:none">屏幕</span>
  <div class="meter"><div class="bar"><i id="lvl"></i></div><span id="db">-99 dB</span></div>
  <span class="pill" id="live">连接中…</span>
  <span class="err" id="err"></span>
  <span class="spacer"></span>
  <button id="mon">听 HDMI</button>
  <button id="stop" class="run">停止</button>
  <button id="clear">清屏</button>
</header>
<main id="main"><div class="empty">等待声音…</div></main>
<script>
const $ = s => document.querySelector(s);
let last = 0, lines = [], running = true, idx = new Map(), draft = '';

// ---------- incremental rendering ----------
// Why "HTML string + innerHTML replace" is gone (changed 2026-11-21): the old build rebuilt the last
// 30 lines on every push (~80ms apart while streaming) and restored scroll from a pixel scrollY. As a
// long translation grows character by character the rebuilt height changes, so the same scrollY now
// points at different content -- the line being read is pushed out of view: "next sentence arrives,
// previous line vanishes". Longer sentences change height more, hence only they showed it. Incremental
// now: one persistent node per line, only changed text nodes are written, geometry never moves.
// Ordering (2026-11-21 afternoon): **newest on top, older lines pushed down**, draft at the very top,
// so the anchor moves from "bottom" to "top": pinned at top a new line shifts the screen down; scrolled
// down, compensate scrollY by the anchor element so the line being read isn't shoved off (old bug, mirrored).
const dom = new Map();      // seq -> {root, head, o, tr, key}
let draftEl = null;

function makeLine(l){
  const root = document.createElement('div');
  root.className = 'line ' + (l.lang || '');
  const head = document.createElement('div'); head.className = 't';
  const o = document.createElement('div'); o.className = 'o';
  const tr = document.createElement('div'); tr.className = 'tr';
  root.append(head, o, tr);
  return {root: root, head: head, o: o, tr: tr, key: {}};
}

function syncLine(l){
  let d = dom.get(l.seq);
  if(!d){
    d = makeLine(l);
    dom.set(l.seq, d);
    // Newest on top: insert the new line at the very front (syncDraft later lifts the draft above it)
    const first = $('#main').firstElementChild;
    if(first) $('#main').insertBefore(d.root, first);
    else $('#main').appendChild(d.root);
  }
  const head = (l.at || '') + ' · ' + (l.dur != null ? l.dur : '') + 's' +
    (l.lag_ms != null ? ' · 延迟 ' + (l.lag_ms/1000).toFixed(1) + 's' : '') +
    (l.trans_ms ? ' · 译 ' + (l.trans_ms/1000).toFixed(1) + 's' : '');
  if(d.key.head !== head){ d.head.textContent = head; d.key.head = head; }

  const o = l.text || '';
  if(d.key.o !== o){ d.o.textContent = o; d.key.o = o; }

  const t = l.trans || '';
  const pend = !!l.pending;
  const k = t + (pend ? '\u0001' : '');
  // Touch the DOM only when content really changed: while streaming this line changes every 80ms, the other 29 stay untouched
  if(d.key.tr !== k){
    d.tr.textContent = '';
    if(t){
      d.tr.className = 'tr';
      d.tr.appendChild(document.createTextNode(t));
      if(pend){
        const c = document.createElement('span');
        c.className = 'cur'; c.textContent = '▍';
        d.tr.appendChild(c);
      }
    } else if(pend){
      d.tr.className = 'tr pend';
      d.tr.textContent = '译文中…';
    } else {
      d.tr.className = 'tr';
    }
    d.key.tr = k;
  }
  const cls = 'line ' + (l.lang || '');
  if(d.root.className !== cls) d.root.className = cls;
}

function syncDraft(){
  if(!draftEl){
    draftEl = document.createElement('div');
    draftEl.className = 'line draft';
    const t = document.createElement('div'); t.className = 't'; t.textContent = '正在说…';
    const o = document.createElement('div'); o.className = 'o';
    draftEl.append(t, o);
    draftEl.txt = '';
  }
  const o = draftEl.lastElementChild;
  if(draft){
    if(!draftEl.isConnected){
      // The draft is "the newest entry" and always sits at the very top
      const first = $('#main').firstElementChild;
      if(first) $('#main').insertBefore(draftEl, first);
      else $('#main').appendChild(draftEl);
    }
    if(draftEl.txt !== draft){
      o.textContent = draft;
      const c = document.createElement('span'); c.className = 'cur'; c.textContent = '▍';
      o.appendChild(c);
      draftEl.txt = draft;
    }
  } else if(draftEl.isConnected){
    draftEl.remove();
    draftEl.txt = '';
  }
}

// Expected order: draft (newest) at the top, the rest by descending seq below it
function orderOk(m, visible){
  let prev = null;
  if(draftEl && draftEl.isConnected){
    if(m.firstElementChild !== draftEl) return false;
    prev = draftEl;
  }
  for(let i = visible.length - 1; i >= 0; i--){
    const r = dom.get(visible[i].seq).root;
    if(prev ? prev.nextElementSibling !== r : m.firstElementChild !== r) return false;
    prev = r;
  }
  return true;
}

function render(){
  const m = $('#main');
  if(!lines.length && !draft){
    if(!m.querySelector('.empty')){
      m.innerHTML = '<div class="empty">等待声音…</div>';
      dom.clear(); draftEl = null;
    }
    return;
  }
  const empty = m.querySelector('.empty');
  if(empty) empty.remove();

  // Visual anchor: pinned at top, don't scroll -- a new line pushes content down (that's "captions scrolling down");
  // scrolled down reading old lines, compensate by the anchor's shift, or the inserted line shoves the paragraph away.
  const anchor = m.firstElementChild;
  const anchorTop = anchor ? anchor.getBoundingClientRect().top : 0;
  const atTop = window.scrollY <= 24;

  const visible = lines.slice(-30);
  const keep = new Set(visible.map(l => l.seq));
  for(const [seq, d] of dom){
    if(!keep.has(seq)){ d.root.remove(); dom.delete(seq); }   // recycle only rows outside the window (now at the bottom)
  }
  for(const l of visible) syncLine(l);

  // Reorder only when genuinely out of order. appendChild on an existing node is a "move": no rebuild, no content loss
  if(!orderOk(m, visible)){
    if(draftEl && draftEl.isConnected) m.appendChild(draftEl);   // land it first, then attach the rest in order
    for(let i = visible.length - 1; i >= 0; i--) m.appendChild(dom.get(visible[i].seq).root);
  }

  syncDraft();

  if(atTop){
    if(window.scrollY) window.scrollTo(0, 0);
  }else if(anchor && anchor.isConnected){
    // Scroll by exactly the pixels the content was pushed down (top grew), keeping the eye on the same sentence
    const delta = anchor.getBoundingClientRect().top - anchorTop;
    if(Math.abs(delta) > 0.5) window.scrollBy(0, delta);
  }
}
function pushLines(arr){
  if(!arr || !arr.length) return false;
  let added = false;
  for(const l of arr){
    if(idx.has(l.seq)) continue;      // an SSE reconnect resends lines before after, so dedupe is mandatory
    lines.push(l); idx.set(l.seq, l); added = true;
  }
  if(!added) return false;
  if(lines.length > 400){
    lines = lines.slice(-300);
    idx = new Map(lines.map(l => [l.seq, l]));
  }
  return true;
}
// Translation patch: fill trans into the same seq, no new row
function applyUpd(list){
  if(!list || !list.length) return false;
  let hit = false;
  for(const u of list){
    const l = idx.get(u.seq);
    if(l){
      l.trans = u.trans;
      // Intermediate streaming states update text only; timings and status wait for final, or the numbers keep jumping
      if(u.final){
        l.trans_ms = u.trans_ms;
        if(u.ttft_ms) l.ttft_ms = u.ttft_ms;
        l.pending = false;
      }
      hit = true;
    }
  }
  return hit;
}

function paint(r){
  const b = $('#stop');
  if(b){ b.textContent = r ? '停止' : '开始'; b.className = r ? 'run' : 'idle'; }
  const lv = $('#live');
  if(lv) lv.textContent = r ? '运行中' : '已停止';
}

// "Listen to HDMI": play capture-card audio through the headphones, usable as a monitor tap in meetings
let monOn = null;
function paintMon(on, err){
  monOn = on;
  const b = $('#mon');
  if(!b) return;
  b.textContent = on ? '正在听 HDMI' : '听 HDMI';
  b.className = on ? 'run' : '';
  if(err) $('#err').textContent = err;
}
async function mon(){
  const b = $('#mon');
  if(b) b.disabled = true;
  try{
    const r = await fetch('/api/monitor', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({on: !monOn})});
    const j = await r.json();
    paintMon(j.monitor.on, j.monitor.error || '');
  }catch(e){ $('#err').textContent = '回放切换失败：' + e; }
  finally{ if(b) b.disabled = false; }
}
$('#mon').onclick = () => mon();

async function ctl(action){
  try{
    const r = await fetch('/api/control', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({action})});
    const j = await r.json();
    if(action === 'clear'){
      lines = []; idx = new Map(); last = 0; draft = '';
      dom.clear(); draftEl = null;
      $('#main').innerHTML = '<div class="empty">等待声音…</div>';
      return;
    }
    running = !!j.running;
    paint(running);
    $('#err').textContent = (!j.ok && j.error) ? j.error : '';
  }catch(e){ $('#err').textContent = '控制失败：' + e; }
}

// Handling for one payload, shared by SSE and polling
function applyPayload(j){
  const st = j.status || {};
  if(st.running !== running){ running = !!st.running; paint(running); }
  if(st.monitor && st.monitor.on !== monOn){ paintMon(st.monitor.on); }
  const got = pushLines(j.lines);
  if(got) last = Math.max(last, j.lines[j.lines.length-1].seq);
  const patched = applyUpd(j.upd);
  // Draft line: the server clears it on submit and the final line appears right away, so just follow the server
  const dt = (j.draft && j.draft.text) || '';
  const draftHit = dt !== draft;
  if(draftHit) draft = dt;
  $('#seg').textContent = st.segments ?? 0;
  const lg = $('#lag');
  if(lg) lg.innerHTML = '延迟 <b>' + (st.lag_ms ? (st.lag_ms/1000).toFixed(1) + 's' : '—') + '</b>'
                        + ' · 首字 <b>' + (st.ttft_ms ? st.ttft_ms + 'ms' : '—') + '</b>'
                        + ' · 译完 <b>' + (st.trans_ms ? (st.trans_ms/1000).toFixed(1) + 's' : '—') + '</b>';
  const db = running ? (j.level_db ?? -99) : -99;
  $('#db').textContent = running ? db.toFixed(0) + ' dB' : '—';
  const thr = j.thr_db ?? -50;
  const pct = Math.max(0, Math.min(100, (db + 70) / 60 * 100));
  const bar = $('#lvl'); bar.style.width = pct + '%';
  bar.style.background = db > thr ? '#41c48a' : '#3a4356';
  if(st.bilingual === false) $('#bi').textContent = '仅原文';
  // Screen term hints: invisible means useless, so show "N terms / M hits" right in the header
  const sc = $('#scr');
  if(sc){
    const s = st.screen;
    if(!s){ sc.style.display = 'none'; }
    else{
      sc.style.display = '';
      sc.innerHTML = '屏幕 <b>' + (s.terms || 0) + '</b> 词 · 命中 <b>' + (s.hits || 0) + '</b>';
      // PAGE is a raw string, so this \n reaches JS verbatim (making it non-raw would break the script)
      sc.title = (s.sample && s.sample.length ? '词表：' + s.sample.join('、') + '\n' : '')
               + (s.last_hint ? '最近命中：' + s.last_hint + '\n' : '')
               + '看屏幕一次 ' + (s.last_ms || 0) + 'ms'
               + (s.last_err ? '\n错误：' + s.last_err : '');
    }
  }
  $('#err').textContent = j.error || '';
  if(got || patched || draftHit) render();
}

// While SSE works the server pushes and polling stands down; if SSE errors it closes itself and falls back
let es = null;
function startSSE(){
  if(!window.EventSource) return false;
  try{ es = new EventSource('/api/events?after=' + last); }catch(e){ es = null; return false; }
  es.onmessage = ev => { try{ applyPayload(JSON.parse(ev.data)); }catch(e){} };
  es.onerror = () => { if(es){ es.close(); es = null; } setTimeout(tick, 300); };
  return true;
}

async function tick(){
  if(es) return;                       // SSE is running, stop polling
  try{
    const r = await fetch('/api/poll?after=' + last, {cache:'no-store'});
    applyPayload(await r.json());
  }catch(e){ $('#live').textContent = '断开'; }
  setTimeout(tick, es ? 1000 : 250);
}

$('#stop').onclick = () => ctl(running ? 'stop' : 'start');
$('#clear').onclick = () => ctl('clear');
paint(true);
if(!startSSE()) tick();     // push if EventSource exists, otherwise poll as a fallback
</script></body></html>"""


# ------------------------------------------------------------------ HTTP
def build_app(cap: Captioner, pipe: Pipeline, device: str, state: dict,
              monitor: MonitorPipe | None = None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="meeting captions")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return PAGE.replace("__DEV__", device)

    @app.get("/api/events")
    async def events(request: Request, after: int = Query(0)):
        """SSE push. Streaming emits a translation delta every ~80ms; polling waits up to 250ms to see it.

        Internally the server still checks every 50ms (cap is a synchronous object), but **the client no longer
        round-trips**: an intermediate state paints on arrival, and a dropped frontend falls back to /api/poll.

        This deliberately avoids cap.take_updates(): that is "drain on read", so two open pages would steal
        patches from each other and the second would sit at "译文中…" forever. Each connection instead
        compares its own translation snapshot, keeping the clients independent.
        """
        import asyncio

        async def gen():
            last = after
            seen: dict[int, str] = {}
            last_status = 0.0
            last_draft: tuple[str, int] = ("", -1)
            while True:
                if await request.is_disconnected():
                    break
                lines = [dict(l) for l in cap.lines if l["seq"] > last]
                if lines:
                    last = lines[-1]["seq"]
                    for l in lines:
                        seen[l["seq"]] = l.get("trans") or ""

                upd = []
                for l in cap.lines:
                    cur = l.get("trans") or ""
                    if cur and seen.get(l["seq"]) != cur:
                        seen[l["seq"]] = cur
                        upd.append({"seq": l["seq"], "trans": cur,
                                    "final": not l.get("pending"),
                                    "trans_ms": l.get("trans_ms") or 0,
                                    "ttft_ms": l.get("ttft_ms") or 0})

                # The draft line updates every ~0.8s while speaking and is replaced by the final line at segment end.
                # It must be pushed separately from translation patches: it isn't a row, it's the row growing at the list end.
                d = cap.draft_state()
                now = time.time()
                tick_status = (now - last_status) >= 1.0
                draft_changed = (d["text"], d["epoch"]) != last_draft
                if draft_changed:
                    last_draft = (d["text"], d["epoch"])

                if lines or upd or draft_changed or tick_status:
                    if tick_status:
                        last_status = now
                    payload = {
                        "lines": lines,
                        "upd": upd,
                        "draft": d,
                        "status": {**cap.status(), "running": pipe.running,
                                   "monitor": monitor.status() if monitor else None},
                        "level_db": round(state.get("level_db", -99.0), 1),
                        "thr_db": round(state.get("thr_db", -50.0), 1),
                        "error": state.get("error", ""),
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/poll")
    async def poll(after: int = Query(0)):
        # Copy: the translation thread patches line["trans"] in place; don't let serialization read a half state
        lines = [dict(l) for l in cap.lines if l["seq"] > after]
        return JSONResponse({
            "lines": lines,
            "upd": cap.take_updates(),
            "draft": cap.draft_state(),
            "status": {**cap.status(), "running": pipe.running,
                       "monitor": monitor.status() if monitor else None},
            "level_db": round(state.get("level_db", -99.0), 1),
            "thr_db": round(state.get("thr_db", -50.0), 1),
            "error": state.get("error", ""),
        })

    @app.get("/api/status")
    async def status():
        return {**cap.status(), "running": pipe.running,
                "level_db": state.get("level_db"), "thr_db": state.get("thr_db"),
                "device": device,
                # Echo the segmentation params verbatim: when segmentation looks off, the first step is to confirm the
                # live process really uses these values instead of guessing at defaults (explicit unit values win).
                "seg": {"max_s": pipe.args.max_s, "hold": pipe.args.hold,
                        "hold_short": pipe.args.hold_short,
                        "hold_long": pipe.args.hold_long,
                        "mid": pipe.args.seg_mid, "long": pipe.args.seg_long},
                "tr": dict(TR_STATS),          # retry/fixed: dirty-translation detections and rescues
                "monitor": monitor.status() if monitor else None}

    @app.post("/api/monitor")
    async def monitor_ctl(request: Request):
        """Play HDMI audio through the headphones: {on: true|false}."""
        import asyncio

        if monitor is None:
            raise HTTPException(503, "回放未启用（启动时没建 MonitorPipe）")
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        want = bool(payload.get("on", not monitor.running))
        loop = asyncio.get_running_loop()
        if want:
            ok = await loop.run_in_executor(None, monitor.start)
            # PipeWire is still negotiating right after start; wait briefly before reading status, or the first check is always false
            if ok:
                await asyncio.sleep(0.35)
        else:
            await loop.run_in_executor(None, monitor.stop)
        return {"ok": True, "monitor": monitor.status()}

    @app.post("/api/control")
    async def control(request: Request):
        """stop / start / clear. stop releases the sound card; start re-estimates the noise floor."""
        import asyncio

        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        action = str(payload.get("action") or "")
        loop = asyncio.get_running_loop()
        if action == "stop":
            # stop has a wait(timeout=2) and start waits for arecord to come up; neither may block the event loop
            await loop.run_in_executor(None, pipe.stop)
            return {"ok": True, "running": pipe.running}
        if action == "start":
            ok = await loop.run_in_executor(None, pipe.start)
            return {"ok": ok, "running": pipe.running,
                    "error": state.get("error", "") if not ok else ""}
        if action == "clear":
            cap.clear()
            return {"ok": True, "running": pipe.running}
        raise HTTPException(400, f"未知动作：{action}")

    return app


# ------------------------------------------------------------------ pipeline
class Pipeline:
    """Start/stop for the capture pipeline.

    Stopping must really release the sound card: halting the thread without terminating arecord makes the next
    start fail with "device or resource busy". Each start rebuilds the Segmenter, re-estimating the floor for this room.
    """

    def __init__(self, cap: Captioner, device: str, args, state: dict):
        self.cap = cap
        self.device = device
        self.args = args
        self.state = state
        self.running = False
        self._src: Capture | None = None
        self._seg: Segmenter | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self._seg = Segmenter(margin_db=self.args.margin, hold_s=self.args.hold,
                                  max_s=self.args.max_s,
                                  min_s=getattr(self.args, "min_s", 0.25),
                                  soft_s=getattr(self.args, "soft_cut", 0.7),
                                  **seg_tuning(self.args))
            # A terminated USB sound card takes a moment to actually be released by ALSA, so reopening
            # immediately always hits "device busy" -- hence several retries instead of one fatal verdict.
            last_err = "打开声卡失败：未知原因"
            for attempt in range(4):
                src = Capture(self.device, SR, backend=getattr(self.args, "backend", "pw"),
                              target=getattr(self.args, "pw_target", None))
                try:
                    src.start()
                except Exception as e:  # noqa: BLE001
                    last_err = f"打开声卡失败：{e}"
                    time.sleep(0.35)
                    continue
                time.sleep(0.45)                   # arecord starts its process asynchronously; wait for it to come up
                if src.alive():
                    self._src = src
                    self.state["error"] = ""
                    self.running = True
                    self._thread = threading.Thread(target=self._loop, name="capture-loop",
                                                    daemon=True)
                    self._thread.start()
                    print(f"[采集] 开始 {src.target or self.device}  backend={self.args.backend}",
                          flush=True)
                    return True
                src.stop()
                last_err = f"打开声卡失败：{src.last_error or '设备仍被占用'}"
                time.sleep(0.35 + 0.25 * attempt)
            self.state["error"] = last_err + "（已重试，稍等一下再点开始）"
            return False

    def stop(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.running = False
            if self._src:
                self._src.stop()
            self.state["level_db"] = -99.0
            self.state["thr_db"] = -50.0
            print("[采集] 停止", flush=True)
            return True

    def _loop(self) -> None:
        seg, src = self._seg, self._src
        assert seg and src
        last_draft = 0.0
        while self.running:
            try:
                frame = src.frames.get(timeout=0.5)
            except queue.Empty:
                if src.errors:
                    self.state["error"] = "采集中断，检查设备是否被占用"
                continue
            db = vp.db_of(frame.astype(np.float64))
            self.state["level_db"] = seg.level_db
            self.state["thr_db"] = seg.threshold()
            pcm = seg.push(frame, db)
            if pcm is not None and pcm.size:
                # Enqueue only, never wait for recognition: ASR and translation own their threads, capture never blocks
                self.cap.submit(pcm, pcm.size / SR, end_ts=time.time())
                last_draft = 0.0                 # new segment starts, restart the draft throttle timer
                continue
            # Segment not finished: send the accumulated audio to recognition every draft_every seconds so gray text shows first.
            # Without this, offline ASR waits for the whole segment (up to 6s) before the first line, which feels non-streaming.
            if self.cap.draft_on and pcm is None:
                now = time.time()
                if now - last_draft >= self.cap.draft_every:
                    d = seg.draft()
                    if d is not None and d[1] >= 0.8:      # a too-short draft carries no information
                        last_draft = now
                        self.cap.push_draft(*d)
        src.stop()


def run_wav(cap: Captioner, path: str, args) -> None:
    audio, sr = vp.read_wav(path)
    if sr != SR:                                   # linear resampling, for offline self-tests only
        n = int(round(audio.size * SR / sr))
        audio = np.interp(np.linspace(0, audio.size - 1, n),
                          np.arange(audio.size), audio)
    print(f"[文件] {path}  {audio.size/SR:.2f}s  {sr}Hz → {SR}Hz", flush=True)
    seg = Segmenter(margin_db=args.margin, hold_s=args.hold, max_s=args.max_s,
                    min_s=getattr(args, "min_s", 0.25),
                    soft_s=getattr(args, "soft_cut", 0.7),
                    **seg_tuning(args))
    # Append 1s of silence, otherwise the last sentence stays in the buffer and is never flushed
    audio = np.concatenate([audio, np.zeros(SR, dtype=audio.dtype)])
    for i in range(0, audio.size - FRAME_N, FRAME_N):
        frame = audio[i:i + FRAME_N]
        db = vp.db_of(frame.astype(np.float64))
        pcm = seg.push(frame, db)
        if pcm is not None and pcm.size:
            line = cap.handle(pcm, pcm.size / SR)
            if line:
                print(f"  [{line['at']}] {line['lang']} {line['text']}"
                      + (f"  → {line['trans']}" if line['trans'] else ""), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="会议字幕（采集卡 → SenseVoice → 中日双语）")
    p.add_argument("--device", default="plughw:CARD=MS2109,DEV=0")
    p.add_argument("--backend", default="pw", choices=["pw", "alsa"],
                   help="pw=PipeWire（不独占，推荐）；alsa=arecord 直连")
    p.add_argument("--pw-target", help="PipeWire 源节点名，默认自动找 MACROSILICON")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--wav", help="不采集，直接跑一个 wav 文件然后退出")
    p.add_argument("--asr-dir", default=str(paths.SENSEVOICE_DIR))
    p.add_argument("--lang", default="auto", help="SenseVoice 语种：auto/zh/ja/en")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--margin", type=float, default=8.0, help="高于本底多少 dB 算说话")
    p.add_argument("--hold", type=float, default=0.45,
                   help="基准静音断句门限（段长介于 --seg-mid 与 --seg-long 之间时用）。"
                        "开会要实时就压小：0.4 秒够分开句子，再小会把词切成两段")
    p.add_argument("--max-s", type=float, default=6.0,
                   help="一段最长多少秒，兜底硬切。4.5 秒仍有 26%% 的段被切断、把"
                        "「大会」切成「大」+「会议」；6.0 秒能把半句并成完整句。"
                        "别超 7 秒：实测 7 秒时同一段音频会出幻觉（「等一长会议」→「德拉兰会议」）")
    p.add_argument("--min-s", type=float, default=0.25,
                   help="短于这个长度的段直接丢（咔哒声、咳嗽）")
    p.add_argument("--soft-cut", type=float, default=0.7,
                   help="硬切前在尾巴里回退找能量低谷的搜索窗（秒）。"
                        "切在音节中间会出半截词，退几百毫秒就能避开")
    p.add_argument("--hold-long", type=float, default=0.90,
                   help="段还短（有声 < --seg-mid）时要等多久静音才断句。"
                        "放大它，短句会跟下一句并成一段，不再被词间小停顿切碎")
    p.add_argument("--hold-short", type=float, default=0.30,
                   help="段已长（有声 ≥ --seg-long）时的断句门限。"
                        "压低它，长句一出现停顿就出字幕，延迟不再往上堆")
    p.add_argument("--seg-mid", type=float, default=1.20,
                   help="多久算「段还短」（秒）。低于它用 --hold-long")
    p.add_argument("--seg-long", type=float, default=2.40,
                   help="多久算「段已长」（秒）。高于它用 --hold-short")
    p.add_argument("--asr-workers", type=int, default=1,
                   help="识别线程数。ASR 只占 0.3 秒/段，1 个够；不够会体现在 q_asr")
    p.add_argument("--tr-workers", type=int, default=2,
                   help="翻译线程数。vLLM 会自己 batch，2 路能把译文追上原文")
    p.add_argument("--ctx-lines", type=int, default=2,
                   help="翻译时带上前几句原文做上文，短句译文才连贯")
    p.add_argument("--no-stream-trans", action="store_true",
                   help="译文退化成整句出（不流式）。流式首字约 96ms，"
                        "整句出要等 300~1300ms；只在排查问题时才关")
    p.add_argument("--llm-think", action="store_true",
                   help="打开模型的思考模式（Qwen3.5 系）。默认关：思考会把 "
                        "max_tokens 吃光导致译文空白，且每句多几百毫秒")
    p.add_argument("--no-bilingual", action="store_true", help="只出原文，不翻译")
    p.add_argument("--no-draft", action="store_true",
                   help="关掉草稿字幕。草稿是说话过程中先出灰字、段结束再定稿，"
                        "用来填掉「等一整段说完」那几秒空白；关掉就退回整段出字")
    p.add_argument("--draft-every", type=float, default=0.8,
                   help="草稿刷新间隔（秒）。压太小会白跑识别抢 CPU，"
                        "放大到 1.5 以上屏幕又开始一顿一顿")
    p.add_argument("--llm-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--llm-model", default="qwen")
    p.add_argument("--llm-key", default="",
                   help="LLM 的 Bearer token。Magnitude 的 /inference/v1 必填，vLLM 留空")
    p.add_argument("--tr-prompt", choices=["full", "short"], default="full",
                   help="翻译 system prompt 档位。full 带反例（7B 时代调出来的，约 250 token），"
                        "short 是 35B 短版（约 70 token）。短版省的是每句的 prefill，"
                        "但丢了示例，质量要另测（tools/tr_quality.py）")
    p.add_argument("--screen-hints", action="store_true",
                   help="用屏幕上的专有名词纠正 ASR 同音错字。只有「读音相似、写法不同」"
                        "才给翻译递一句提示，没有命中就不加任何 token（默认关）")
    p.add_argument("--screen-api", default="http://127.0.0.1:8095/api/look",
                   help="向谁要屏幕 OCR。采集卡是独占的，字幕服务不自己开设备，"
                        "默认借 voice-bench 的 /api/look（它内部按画面指纹缓存）")
    p.add_argument("--screen-every", type=float, default=12.0,
                   help="隔几秒看一次屏幕。OCR 一次约 0.7~3.5 秒，别小于 4 秒")
    p.add_argument("--monitor", action="store_true",
                   help="启动时就把 HDMI 声音回放到耳机（也可在页面上开关）")
    p.add_argument("--monitor-sink", default="H340",
                   help="回放目标节点名关键字；空则用系统默认输出")
    p.add_argument("--monitor-rate", type=int, default=44100,
                   help="回放采样率。耳机节点是 44100，传同值可省一次重采样")
    p.add_argument("--monitor-latency", type=int, default=20,
                   help="回放缓冲毫秒。20ms 实测无欠载；开会听着别超过 60")
    args = p.parse_args(argv)

    global LLM_THINK, _LLM_KEY, TR_STYLE
    LLM_THINK = args.llm_think
    _LLM_KEY = args.llm_key
    TR_STYLE = args.tr_prompt

    print(f"[1/2] 加载 ASR … {args.lang}", flush=True)
    asr = Asr(Path(args.asr_dir), threads=args.threads, language=args.lang)
    screen = None
    if args.screen_hints:
        screen = ScreenHints(api=args.screen_api, every=args.screen_every)
    cap = Captioner(asr, bilingual=not args.no_bilingual,
                    llm_base=args.llm_base, llm_model=args.llm_model,
                    asr_workers=args.asr_workers, tr_workers=args.tr_workers,
                    ctx_lines=args.ctx_lines,
                    stream_trans=not args.no_stream_trans,
                    draft_on=not args.no_draft, draft_every=args.draft_every,
                    screen=screen)
    if screen is not None:
        # Self-loop guard: the screen term list must be able to recognize "this line is our own caption"
        screen.recent = lambda: [x for l in list(cap.lines)[-30:]
                                 for x in (l.get("text", ""), l.get("trans", ""))]
        screen.start()
        print(f"[屏幕] 每 {screen.every:.0f}s 看一次 {screen.api}", flush=True)

    if args.wav:
        print("[2/2] 离线模式", flush=True)
        run_wav(cap, args.wav, args)
        return 0

    state: dict = {"level_db": -99.0, "thr_db": -50.0, "error": ""}
    label = args.device if args.backend == "alsa" else "USB3.0 Capture (PipeWire)"
    pipe = Pipeline(cap, args.device, args, state)
    monitor = MonitorPipe(src=resolve_pw_source(args.pw_target or "MACROSILICON"),
                          sink=resolve_pw_sink(args.monitor_sink),
                          rate=args.monitor_rate,
                          latency_ms=args.monitor_latency)
    app = build_app(cap, pipe, label, state, monitor=monitor)
    if not pipe.start():
        print(f"!! 启动采集失败：{state['error']}", flush=True)
    if args.monitor:
        print(f"[监听] 开机即开：{monitor.src} → {monitor.sink}", flush=True)
        monitor.start()

    print(f"[2/2] 打开 http://<jetson-ip>:{args.port}   双语={'关' if args.no_bilingual else '开'}"
          f"   译文={'流式（首字先出）' if not args.no_stream_trans else '整句'}"
          f"   草稿={'开 %.1fs' % args.draft_every if not args.no_draft else '关'}",
          flush=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
