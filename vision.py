#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vision.py — 让语音助手「看一眼屏幕」

链路：/dev/video0（UVC 采集卡）→ 丢弃预热帧 → 缩放 → RapidOCR → 文本

## 为什么是 RapidOCR 而不是 PaddleOCR

Jetson AGX Orin 实测，同一张 1080p 中文界面截图（50 行文字）：

| 引擎 | 耗时 | 平均置信 |
|---|---|---|
| RapidOCR (onnxruntime) | **6.2s** | 0.894 |
| PaddleOCR 默认（server 模型） | 17.0s | 0.98 |
| PaddleOCR mobile | 24.6s | 0.749 |

mobile 模型在 Jetson 上反而比 server 更慢（Jetson CPU 对大卷积更友好），
且置信度明显下降。RapidOCR 还省掉 paddle 那几百 MB 依赖——asr-venv 里现成就有。

## 为什么输入先缩到长边 1440

1920 原图要 7.3s，而且「窗口」被读成「官口」；缩到 1440 只要 5.7s，反而读对了。
RapidOCR 内部会把长边压到固定尺寸，先自己用 INTER_AREA 缩能少一次重采样。
960 以下小字开始糊（「编辑」→「编冀」），不要更低。

## OCR 线程数为什么要写死

`RapidOCR()` 不传线程数就吃 onnxruntime 默认值。这台机同时跑 vLLM、ASR、字幕服务，
12 核吃满 12 线程实测 **9.10s**，显式 8 线程 **4.36s**——超订的代价远大于并行收益。
平坦区 6~8，10 开始退化。见 `OCR_THREADS`。

## 为什么每段音频不丢帧，但打开设备要丢 40 帧

MS2109 每次**打开设备或切换分辨率**后，前 33~36 帧是纯黑（实测 YMAX=7）。不丢就是
一张黑图，OCR 出来是空的，看起来像「屏幕上没字」。这个坑在 README 4.7 有完整记录。

所以预热是「每次 open 一次」，不是「每次取帧一次」：`_Capture` 把句柄常驻下来，
**一个进程生命周期只付这一次 1.2 秒**。取帧从 2.87 秒降到几十毫秒（读 2 帧就够新）。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path


# OCR 线程数：**必须显式指定，不能吃默认值**。
# onnxruntime 默认吃满核数（Orin 12 核 → 12 线程），而 vLLM/ASR/字幕服务同时在跑，
# 12 线程实测 9.10s，8 线程 4.36s——超订把速度打回一半（线程迁移 + 抢核）。
# 实测平坦区在 6~8（4.34 / 4.49s），10 开始退化（5.11s），12 崩掉（9.10s）。
# 取 8：留 4 核给别的常驻进程，且与 6 的差在噪声内。
OCR_THREADS = 8

# 「看一眼屏幕」的语音意图。命中任一即触发 OCR。
# 注意不要放「看到」这种过宽的词，日常对话里太容易误触。
LOOK_HINTS = (
    "看看", "看一看", "看一眼", "看一下", "瞧瞧", "瞧一眼",
    "屏幕上", "画面上", "屏幕里", "画面里",
    "这是什么", "这是啥", "写的什么", "写了什么", "读一下", "念一下",
    "认得吗", "认识吗", "你看得见", "能看到",
)


def is_look_intent(text: str) -> bool:
    """这句话是不是在要求「看屏幕」。"""
    if not text:
        return False
    return any(h in text for h in LOOK_HINTS)


# ---------------------------------------------------------------- 脱敏
# 屏幕上经常出现 token / 密钥 / 手机号，OCR 全文进 LLM 前先遮掉明显的东西
import re

_SECRET = re.compile(r"\b(sk-|ghp_|mul_|xox[baprs]-|AKIA)[A-Za-z0-9_\-]{8,}")
_LONGKEY = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")


def redact(text: str) -> tuple[str, int]:
    """遮掉疑似密钥/长随机串/手机号，返回 (处理后文本, 命中数)。"""
    n = 0

    def count(_m: "re.Match") -> str:
        nonlocal n
        n += 1
        return _m.group(0)

    for pat in (_SECRET, _LONGKEY, _PHONE):
        text = pat.sub(count, text)
    # 统一替换：保留前缀便于人眼判断是什么类型的密钥
    text = _SECRET.sub(lambda m: m.group(1) + "***", text)
    text = _LONGKEY.sub("***", text)
    text = _PHONE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[-4:], text)
    return text, n


class VisionError(RuntimeError):
    pass


class _Capture:
    """常驻的采集卡句柄：打开一次、预热一次，之后每次只读一两帧。

    为什么不在后台线程里连续解码：1080p MJPG 解码是实打实的 CPU 开销，而「看屏幕」
    一场会议也就用几次。保持「句柄常开、帧按需读」两头都占——省掉 1.2 秒预热和
    设备协商，又不在空闲时烧 CPU。V4L2 自己维护最近帧的队列，读到的仍是当前画面。

    线程安全：内部一把锁串行化，设备本身也是独占的。
    """

    def __init__(self, dev: str, width: int, height: int, warmup: int,
                 *, keep_open: bool = True, idle_release: float = 0.0):
        self.dev = dev
        self.width = width
        self.height = height
        self.warmup = max(1, warmup)
        self.keep_open = keep_open
        self.idle_release = idle_release    # >0 时闲置这么久就松开设备，让别的程序能用
        self._cap = None
        self._lock = threading.RLock()
        self.opens = 0          # 打开次数（正常应为 1，多次说明掉线重连过）
        self.warmups = 0        # 预热轮数（每次 open 一轮；全黑重试会 +1）
        self.reads = 0          # 成功取帧次数
        self.open_s = 0.0       # 最近一次「打开 + 预热」耗时
        self.grab_s = 0.0       # 最近一次取帧耗时
        self.last_read = 0.0
        self.reconnects = 0
        self.last_release = ""

    # ------------------------------------------------------------ 内部
    def _drain_locked(self, n: int):
        """连读 n 帧，返回最后一个成功帧（读不到返回 None）。"""
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
        # 必须显式 MJPG：默认会协商成 YUYV，1080p 在 USB2 上只有 5fps，
        # 读 40 帧预热要 8 秒而不是 1.3 秒
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.opens += 1
        frame = self._drain_locked(self.warmup)     # 一次性丢掉预热黑帧
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

    # ------------------------------------------------------------ 对外
    def read(self, fresh: int = 2):
        """取一帧当前画面。句柄已开就只读 fresh 帧，不重开、不重预热。"""
        with self._lock:
            if self._idle_expired_locked():
                self._release_locked("闲置释放")
            t0 = time.time()
            if self._cap is None:
                self._open_locked()
            frame = self._drain_locked(max(1, fresh))
            if frame is None:
                # 扩展坞那种偶发掉线（曾 166 秒断流）：松开重开一次就回来了
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
        """再丢一轮预热帧。源切分辨率后 MS2109 会重新吐黑帧，用它清掉。

        带时间预算：**无信号时设备会降帧率**（实测 40 帧读到 5 秒），不设上限就会让
        「判无信号」这件事本身变得很慢。预算内没读满 40 帧也无所谓，画面是否恢复由
        调用方复查。
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
    """采集卡取帧 + OCR。线程安全（内部串行化，设备是独占的）。"""

    def __init__(self, dev: str = "/dev/video0", width: int = 1920, height: int = 1080,
                 warmup: int = 40, max_side: int = 1440, min_score: float = 0.35,
                 max_chars: int = 1400, cache_ttl: float = 8.0,
                 keep_open: bool = True, idle_release: float = 0.0,
                 ocr_threads: int = OCR_THREADS, trt_engine: str = ""):
        self.dev = dev
        self.width = width
        self.height = height
        self.warmup = warmup          # 丢弃的前置帧数（MS2109 黑帧预热）
        self.max_side = max_side
        self.min_score = min_score
        self.max_chars = max_chars
        self.cache_ttl = cache_ttl
        self.ocr_threads = ocr_threads
        self.trt_engine = trt_engine  # 非空则把 rec 后端换成 TensorRT（约 2.4×）
        self._trt = None
        self.trt_note = ""            # TensorRT 挂载结果（ok / 失败原因）
        self._lock = threading.Lock()
        self._engine = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self.calls = 0
        self.errors = 0
        self.last: dict | None = None
        self._grab_s = 0.0   # 最近一次取帧耗时；grab 失败时也要有值可读
        self._cap = _Capture(dev, width, height, warmup,
                             keep_open=keep_open, idle_release=idle_release)

    # ------------------------------------------------------------ 引擎
    def _ocr_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR   # 懒加载：0.4s，不用不付
            try:
                self._engine = RapidOCR(intra_op_num_threads=self.ocr_threads)
            except TypeError:
                # 老版本没有这个参数：退回默认（会在 12 核机上超订，慢一倍）
                self._engine = RapidOCR()
            self._attach_trt()
        return self._engine

    def _attach_trt(self) -> None:
        """把 rec 后端换成 TensorRT。失败就留在 CPU，绝不让「看屏幕」整个不可用。

        engine 是跟 TensorRT 版本绑死的（换 JetPack 要重建），也可能是别的机型拷来的，
        所以这里所有异常都吞掉、只记状态——CPU 路径本来就一直能用。
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
        """开机时把 OCR 引擎（含 TensorRT）建好。

        引擎懒加载本来是为了省 0.4 秒启动时间，但 TRT 要反序列化 engine、建 CUDA
        context，第一次前向还要 warmup——这几秒落到用户头上就是「第一次看屏幕特别慢」。
        所以在后台线程里先付掉。
        """
        try:
            self._ocr_engine()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------ 取帧
    def grab(self):
        """返回一帧当前画面。

        设备句柄常驻，所以这里**不再**开关设备、也不再丢 40 帧预热——那两笔只在
        进程启动时付一次（`prewarm()`），取帧本身只剩读 2 帧的时间。
        """
        t0 = time.time()
        frame = self._cap.read()
        self._grab_s = time.time() - t0
        return frame

    def prewarm(self) -> bool:
        """开机时先把设备打开预热好，第一次「看屏幕」不用再等 1.2 秒。

        失败只返回 False，不抛异常：采集卡没插时服务照常启动，只是「看屏幕」不可用。
        """
        try:
            self._cap.read(fresh=1)
            return True
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        """松开采集卡（让别的程序能用这个设备）。"""
        self._cap.close()
        if self._trt is not None:
            try:
                self._trt.close()
            except Exception:  # noqa: BLE001
                pass
            self._trt = None

    def capture_status(self) -> dict:
        return self._cap.status()

    # ------------------------------------------------------------ 帧指纹
    @staticmethod
    def _dhash(img, size: int = 16) -> int:
        """横向梯度哈希。对光标闪烁/时钟跳秒不敏感，又能区分不同版式。

        纯 numpy 实现（不走 cv2）：一个哈希不值得引入依赖，也让单测能在
        没有 opencv 的机器上跑。16×17 个块平均，每次约 1ms。
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

    # ------------------------------------------------------------ 识别
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

    # 无信号的判据是「又暗又平」，不是「纯黑」。
    # 实测：MS2109 在无信号时吐的是 mean=7.0 / std=1.5 的帧（YMAX=7，带传感器噪点），
    # 所以旧判据 mean<3 在真机上永远不会命中，会把无信号当成「一屏没有字的画面」。
    # 而暗色主题的 IDE（mean 15~25）std 有 15 以上，不会被误判。
    BLANK_MEAN, BLANK_STD = 12.0, 6.0

    @classmethod
    def _is_blank(cls, img) -> bool:
        import numpy as np

        a = np.asarray(img, dtype=np.float32)
        return float(a.mean()) < cls.BLANK_MEAN and float(a.std()) < cls.BLANK_STD

    # ------------------------------------------------------------ 对外
    def look(self, force: bool = False) -> dict:
        """看一眼屏幕。同一画面（dHash 相同）在 cache_ttl 内直接返回缓存。"""
        with self._lock:
            self.calls += 1
            t0 = time.time()
            try:
                img = self.grab()
                if self._is_blank(img):
                    # 源切分辨率后 MS2109 会重新吐一轮黑帧。常驻句柄遇不到 open，
                    # 却会遇到这种「流中途变黑」，所以再丢一轮预热帧才判无信号，
                    # 否则会把正常画面误报成「HDMI 没信号」。
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
                # 只留最近 8 个画面指纹，避免常驻进程无限增长
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
    """把 OCR 结果变成喂给 LLM 的上下文块。"""
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
