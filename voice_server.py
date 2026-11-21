#!/usr/bin/env python3
"""voice_server.py —— 浏览器可操作的语音对话测试台。

一条命令起服务，浏览器打开就能对着插在 Jetson 上的 H340 说话，链路是：

    点击 → arecord（VAD 自动收尾）→ SenseVoice → vLLM → Piper → 浏览器播放

三个设计取舍，都是刻意的：

1. **录音在服务器端，不在浏览器**。麦克风插在 Jetson 上；如果让浏览器录音，
   测的就是你笔记本的麦克风，与真机无关。浏览器只负责触发和播放。

2. **录音靠 VAD 收尾，不定长**。固定录 5 秒会有「说完了还得等」的体验，
   而等满 5 秒这件事本身会掩盖真实延迟。这里用能量 VAD 判断你说完没有，
   停顿 0.75 秒即收尾，同时保留 12 秒硬上限。

3. **按句合成、逐句回传**。整段合成要等 LLM 全部生成完，感知延迟等于总和。
   这里 LLM 一吐出完整句子就合成该句并推给浏览器，浏览器边收边播，
   首句听感延迟接近「首字延迟 + 首句合成」。

用法：
    ./asr-venv/bin/python voice_server.py --port 8095
    浏览器打开 http://127.0.0.1:8095

    # LLM 还没起时也能测（ASR→TTS 回声，验证链路）
    ./asr-venv/bin/python voice_server.py --port 8095 --no-llm
"""
# 注意：这里刻意不使用 `from __future__ import annotations`。
# FastAPI 会在模块全局命名空间里解析注解，而 Request 是在 build_app 内导入的局部名字；
# 一旦注解被字符串化，`request: Request` 就会被误判成查询参数并报 422。

import argparse
import asyncio
import difflib
import io
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_probe as vp  # noqa: E402
import vision as vx  # noqa: E402  取帧 + OCR：「看一眼屏幕」

import paths  # noqa: E402  目录约定：MC_DATA 可覆盖，见 paths.py

DATA = paths.DATA
DEFAULT_ASR_DIR = paths.SENSEVOICE_DIR
DEFAULT_TTS = paths.PIPER_JA
DEFAULT_TTS_ZH = paths.PIPER_ZH
DEFAULT_VAD = paths.VAD_MODEL
DEFAULT_KWS = paths.KWS_DIR
WORK = paths.WORK_DIR
AUDIO_DIR = WORK / "audio"
TURN_LOG = WORK / "turns.jsonl"

SR = 16000
FRAME_S = 0.02

# 假名 / 汉字，用来给合成选音色
KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
CJK = re.compile(r"[\u4E00-\u9FFF]")


# ------------------------------------------------------------------ 录音
class RecordError(RuntimeError):
    pass


class VoiceRecorder:
    """录音并按 Silero VAD 自动判停。

    为什么不用能量门限：早期版本用「本底 + 余量」的绝对阈值，实测两种失败都出现过。
      - 增益高时：本底被语音污染，门限抬到 −20dB，语音被自己判成静音（只截到 0.81s）
      - 环境嘈杂时：噪声本身就在门限之上，永远判不出「说完了」，每次都撞 12 秒上限
    Silero VAD 是基于 512 样本窗的语音概率模型，对增益、稳态噪声都不敏感，
    而且它自己维护「说完多久算一段」，不需要我们再调两个时间常数。
    """

    def __init__(self, device: str, vad_model: str, max_s: float = 15.0,
                 start_timeout_s: float = 8.0, silence_s: float = 0.60,
                 min_speech_s: float = 0.25, threshold: float = 0.5,
                 hp: float = 150.0):
        self.device = device
        self.vad_model = vad_model
        self.max_s = max_s
        self.start_timeout_s = start_timeout_s
        self.silence_s = silence_s
        self.min_speech_s = min_speech_s
        self.threshold = threshold
        self.hp = hp

    def _make_vad(self):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = self.vad_model
        cfg.silero_vad.threshold = self.threshold
        cfg.silero_vad.min_silence_duration = self.silence_s
        cfg.silero_vad.min_speech_duration = self.min_speech_s
        cfg.sample_rate = SR
        return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)

    def record(self, source: "queue.Queue | None" = None) -> tuple[np.ndarray, int, dict]:
        """录一段并 VAD 判停。

        source=None：自己起 arecord（点按钮模式）。
        source=队列：从常驻 AudioHub 取帧（唤醒模式）。ALSA 设备同一时刻只能被
        一个进程占用，唤醒线程已经在读了，这里不能再开第二个 arecord。
        """
        vad = self._make_vad()
        frame_len = int(SR * FRAME_S)
        nbytes = frame_len * 2
        proc = None
        if source is None:
            cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE",
                   "-r", str(SR), "-c", "1", "-t", "raw"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        dbs: list[float] = []
        t_start = time.time()
        speech_at: float | None = None
        seg_samples: np.ndarray | None = None
        try:
            while True:
                if source is None:
                    raw = proc.stdout.read(nbytes)
                    if not raw or len(raw) < nbytes:
                        break
                    frame = np.frombuffer(raw, dtype=np.int16)
                else:
                    try:
                        frame = source.get(timeout=self.max_s + 2.0)
                    except queue.Empty:
                        break
                    if frame is None:
                        break
                dbs.append(vp.db_of(frame.astype(np.float64)))
                now = time.time() - t_start

                vad.accept_waveform((frame.astype(np.float32) / vp.FULL_SCALE))
                if vad.is_speech_detected() and speech_at is None:
                    speech_at = now
                # 检测器在有语音段「说完」后才把段吐出来，这正好是我们要的收尾信号
                if not vad.empty():
                    seg = vad.front
                    seg_samples = np.asarray(seg.samples, dtype=np.float32)
                    vad.pop()
                    break

                if speech_at is None and now > self.start_timeout_s:
                    raise RecordError("没有检测到说话（8 秒内 VAD 未触发）")
                if now > self.max_s:
                    break
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if seg_samples is None:
            # 撞到硬上限或提前中断：把已缓冲的段冲出来
            try:
                vad.flush()
                if not vad.empty():
                    seg_samples = np.asarray(vad.front.samples, dtype=np.float32)
            except Exception:  # noqa: BLE001
                pass
        if seg_samples is None or seg_samples.size == 0:
            if not dbs:
                raise RecordError("arecord 没有返回音频（设备被占用或参数不支持）")
            raise RecordError("没有检测到完整语音（可能太短或被噪声打断）")

        audio = np.clip(seg_samples * vp.FULL_SCALE, -32768, 32767).astype(np.int16)
        speech_s = audio.shape[0] / SR
        if speech_s < self.min_speech_s:
            raise RecordError(f"语音只有 {speech_s:.2f}s，太短")

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        info = {
            "speech_s": round(speech_s, 2),
            "peak_db": round(20.0 * np.log10(max(peak, 1.0) / vp.FULL_SCALE), 1),
            "floor_db": round(float(np.percentile(dbs, 10)), 1) if dbs else -999.0,
            "vad": "silero",
        }
        return audio, SR, info


# ------------------------------------------------------------------ 常驻采集
class AudioHub:
    """一个 arecord 进程，多路消费者。

    为什么需要它：唤醒要求麦克风一直开着，而 ALSA 设备同一时刻只能被一个进程占。
    原来 record() 每次自己起 arecord，唤醒线程和录音就会互相抢设备。
    这里让一个线程独占设备、把定长帧分发给订阅者；订阅时先补最近 0.3–0.6 秒的帧，
    这样「小八你好，今天天气怎么样」连说时，唤醒词之后的字不会丢。

    arecord 异常退出（设备被占用、睡眠唤醒）时自动重开，errors 计数供 health 观察。
    """

    def __init__(self, device: str, frame_s: float = FRAME_S, pre_s: float = 0.6):
        self.device = device
        self.frame_len = int(SR * frame_s)
        self.nbytes = self.frame_len * 2
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._recent: deque = deque(maxlen=max(1, int(pre_s / frame_s)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0
        self.last_frame_at = 0.0

    def start(self) -> "AudioHub":
        self._thread = threading.Thread(target=self._run, name="audio-hub", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def rate_s(self) -> float:
        """最近一次收到帧距今的秒数，用来判断采集是否还活着。"""
        return round(time.time() - self.last_frame_at, 1) if self.last_frame_at else -1.0

    def _run(self) -> None:
        cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE",
               "-r", str(SR), "-c", "1", "-t", "raw"]
        while not self._stop.is_set():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
            except OSError:
                self.errors += 1
                time.sleep(1.0)
                continue
            try:
                while not self._stop.is_set():
                    raw = proc.stdout.read(self.nbytes)
                    if not raw or len(raw) < self.nbytes:
                        break
                    self._push(np.frombuffer(raw, dtype=np.int16))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if not self._stop.is_set():
                self.errors += 1
                time.sleep(0.5)

    def _push(self, frame: np.ndarray) -> None:
        self.last_frame_at = time.time()
        with self._lock:
            self._recent.append(frame)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # 慢消费者：丢最旧的一帧，绝不让采集线程阻塞
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except queue.Empty:
                    pass

    def subscribe(self, pre_s: float = 0.3, pre_frames=None) -> queue.Queue:
        """订阅音频帧。

        pre_frames：显式指定「历史帧」，代替从环形缓冲里取。打断交接要用它——
        打断检测线程已经读走了一段人声，交给录音时必须原样带上，否则那句话的
        开头几个字就被吃掉了（环形缓冲只有 0.6 秒，补不回来）。
        """
        q: queue.Queue = queue.Queue(maxsize=int(30 / FRAME_S))
        with self._lock:
            if pre_frames:
                for f in pre_frames:
                    try:
                        q.put_nowait(f)
                    except queue.Full:
                        break
            else:
                n = max(0, int(pre_s / FRAME_S))
                if n:
                    for f in list(self._recent)[-n:]:
                        q.put_nowait(f)
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


# ------------------------------------------------------------------ 唤醒词
class Kws:
    """关键词唤醒：sherpa-onnx zipformer-KWS，中文拼音建模，纯 CPU。

    实测 RTF 0.03（8 秒音频 0.26 秒算完），Jetson 上占单核约 3%，
    和 ASR/TTS 抢核可以忽略。关键词是运行时生成的，改词不用重训模型。
    """

    def __init__(self, model_dir: Path, word: str, keywords_file: Path,
                 threshold: float = 0.25, score: float = 1.0, threads: int = 2):
        import sherpa_onnx
        from sherpa_onnx.utils import text2token

        d = Path(model_dir)
        toks = text2token([word], str(d / "tokens.txt"), "ppinyin")[0]
        Path(keywords_file).write_text(" ".join(toks) + f" @{word}\n", encoding="utf-8")

        def int8(name):
            return str(d / f"{name}-epoch-99-avg-1-chunk-16-left-64.int8.onnx")

        self.word = word
        self.keywords_file = Path(keywords_file)
        self.hits = 0
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"),
            encoder=int8("encoder"), decoder=int8("decoder"), joiner=int8("joiner"),
            keywords_file=str(keywords_file),
            num_threads=threads, keywords_score=score,
            keywords_threshold=threshold, provider="cpu")

    def wait(self, q: queue.Queue, stop: threading.Event | None = None) -> str | None:
        """阻塞等待唤醒词。返回命中的关键词，stop 置位时返回 None。"""
        stream = self.spotter.create_stream()
        while not (stop and stop.is_set()):
            try:
                frame = q.get(timeout=0.2)
            except queue.Empty:
                continue
            stream.accept_waveform(SR, (frame.astype(np.float32) / vp.FULL_SCALE))
            while self.spotter.is_ready(stream):
                self.spotter.decode_stream(stream)
                r = self.spotter.get_result(stream)
                if r:
                    self.spotter.reset_stream(stream)
                    self.hits += 1
                    return r
        return None


class WakeLoop(threading.Thread):
    """常驻监听：等唤醒词 → 跑一轮对话 → 回到监听。"""

    def __init__(self, hub: AudioHub, kws: Kws, runner, history: list[dict],
                 push, turn_lock: threading.Lock, has_listener=None):
        super().__init__(name="wake-loop", daemon=True)
        self.hub = hub
        self.kws = kws
        self.runner = runner
        self.history = history
        self.push = push
        self.turn_lock = turn_lock
        # 没有页面在听时不跑对话：否则关掉页面后每次唤醒都会白跑一轮、
        # 白合成一堆 wav，还占着 turn_lock
        self.has_listener = has_listener or (lambda: True)
        self._stop = threading.Event()
        self.turns = 0
        self.skipped = 0  # 命中了但没有页面在听

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, ev: dict) -> None:
        # 被打断的那轮不进历史：它是不完整的
        if (ev["type"] == "done" and not ev.get("interrupted")
                and ev.get("user") and ev.get("reply")):
            self.history.append({"role": "user", "content": ev["user"]})
            self.history.append({"role": "assistant", "content": ev["reply"]})
        self.push(ev)

    def run(self) -> None:
        while not self._stop.is_set():
            q = self.hub.subscribe(pre_s=0.0)
            try:
                word = self.kws.wait(q, self._stop)
            finally:
                self.hub.unsubscribe(q)
            if not word:
                break

            self.push({"type": "wake", "word": word})
            if not self.has_listener():
                self.skipped += 1
                print(f"[wake] 命中「{word}」但没有页面在听，跳过", flush=True)
                continue
            # 唤醒词之后的音频：pre 只补 0.25 秒，够接上「你好」的尾音又不至于
            # 把整句唤醒词带进 ASR
            q2 = self.hub.subscribe(pre_s=0.25)
            try:
                with self.turn_lock:
                    self.runner.run(self._emit, source=q2,
                                    history=list(self.history), strip_prefix=word)
                self.turns += 1
            except Exception as e:  # noqa: BLE001
                self.push({"type": "error", "message": f"唤醒后处理失败：{e}"})
            finally:
                self.hub.unsubscribe(q2)
            self.push({"type": "state", "state": "waiting"})


# ------------------------------------------------------------------ ASR
class Asr:
    def __init__(self, model_dir: Path, threads: int = 8, language: str = "ja"):
        import sherpa_onnx

        d = Path(model_dir)
        m = d / "model.int8.onnx"
        if not m.exists():
            m = d / "model.onnx"
        if not m.exists():
            raise FileNotFoundError(f"{d} 下没有 SenseVoice 模型")
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(m), tokens=str(d / "tokens.txt"),
            num_threads=threads, use_itn=True, language=language, debug=False)
        self.language = language

    def run(self, audio16: np.ndarray, sr: int) -> tuple[str, float]:
        x = vp.highpass(audio16.astype(np.float64), 150.0, sr, 2)
        t0 = time.time()
        s = self.rec.create_stream()
        s.accept_waveform(sr, (x / vp.FULL_SCALE).astype(np.float32))
        self.rec.decode_stream(s)
        cost = time.time() - t0
        return clean_text(s.result.text), cost


EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")


def clean_text(s: str) -> str:
    """SenseVoice 会在结果里带 <|ja|><|HAPPY|> 这些标签，去掉。"""
    s = re.sub(r"<\|[^|]*\|>", "", s or "")
    s = EMOJI.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def trim_by_vad(audio: np.ndarray, sr: int, hp: float = 150.0) -> tuple[np.ndarray, dict]:
    """把首尾静音裁掉，只保留语音区。

    这一步对 SenseVoice 不是可选的：把带长静音的整段音频直接喂进去，会出现
    重复和错乱（实测 40 秒含两段语音的文件被读成「…確認してくださいこ…確認してださい」）。
    真实录音路径已经由录音端的 VAD 裁过，这里是给 wav 回放用的同一套处理。
    """
    x = vp.highpass(audio.astype(np.float64), hp, sr, 2)
    frames = vp.frame_db(x, sr)
    if frames.size == 0:
        return audio, {"trimmed": False}
    thr = vp.vad_threshold(vp.noise_floor(frames), 8.0, -52.0)
    segs = vp.segment_speech(frames, thr, min_speech_s=0.30, min_gap_s=0.45)
    if not segs:
        return audio, {"trimmed": False, "thr_db": round(thr, 1)}
    # 逐段截取再首尾相接：既去掉静音，也去掉段间长停顿
    parts = []
    for s in segs:
        a = max(0, int((s.start - 0.15) * sr))
        b = min(audio.shape[0], int((s.end + 0.15) * sr))
        if b > a:
            parts.append(audio[a:b])
    if not parts:
        return audio, {"trimmed": False, "thr_db": round(thr, 1)}
    out = np.concatenate(parts)
    return out, {"trimmed": True, "thr_db": round(thr, 1),
                 "speech_s": round(out.shape[0] / sr, 2), "segs": len(segs)}


def events_since(buf, after: int) -> list:
    """短轮询用：取出 seq > after 的事件，保持原顺序。"""
    return [ev for (sq, ev) in buf if sq > after]


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.astype(np.int16).tobytes())


# ------------------------------------------------------------------ TTS
class Tts:
    """多音色合成，按句子的文字自动选引擎。

    为什么必须分引擎：Piper 的日语音色读中文、中文音色读日文都会严重变调
    （假名被当拼音念，汉字读成另一个字的音）。判定规则很可靠：
    有平假名/片假名 → 日语；没有假名但有汉字 → 中文。中日混排的场景已确认不存在，
    所以按句判断就够，不需要按字判断。
    """

    def __init__(self, ja_model: Path, zh_model: Path | None = None,
                 speaker: int | None = None):
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        t0 = time.time()
        self.voices: dict[str, object] = {}
        self.cfgs: dict[str, object] = {}
        self.voices["ja"] = PiperVoice.load(str(ja_model))
        self.cfgs["ja"] = SynthesisConfig(speaker_id=speaker) if speaker is not None else None
        if zh_model and Path(zh_model).exists():
            self.voices["zh"] = PiperVoice.load(str(zh_model))
            self.cfgs["zh"] = None  # 中文音色单说话人，传 speaker_id 会报错
        self.default = "ja" if "ja" in self.voices else next(iter(self.voices))
        self.load_s = time.time() - t0

    def pick(self, text: str, hint: str | None = None) -> str:
        """选音色。hint 是上一句（或本轮回复语言）用过的音色。

        为什么要 hint：一句话可能只有数字或拉丁字母（"1.5"、"OK"、"2026"），
        既没有假名也没有汉字。以前这种句子会落到 default（日语），于是中文回答里
        的「3.14」被日本語読み —— 用户听到的就是「数字怎么是日语」。
        这类句子必须跟随上下文语言，hint 就是上下文。
        """
        if KANA.search(text) and "ja" in self.voices:
            return "ja"
        if CJK.search(text) and "zh" in self.voices:
            return "zh"
        if hint in self.voices:
            return hint
        # 没有上下文时（例如英文首句）用中文：中文模型读英文/数字不会串成日语音
        return "zh" if "zh" in self.voices else self.default

    def _render(self, lang: str, text: str) -> tuple[bytes, int, int]:
        """合成到内存。空音素化时 wave 会在 close 时抛错，这里转成 0 帧返回。"""
        bio = io.BytesIO()
        try:
            with wave.open(bio, "wb") as wf:
                self.voices[lang].synthesize_wav(
                    text, wf, syn_config=self.cfgs.get(lang))
        except wave.Error:
            return b"", 0, 22050
        data = bio.getvalue()
        bio.seek(0)
        with wave.open(bio, "rb") as r:
            return data, r.getnframes(), r.getframerate()

    def synth_to(self, text: str, out: Path, hint: str | None = None) -> dict:
        lang = self.pick(text, hint)
        voices = self.voices
        t0 = time.time()
        data, n, sr = self._render(lang, text)
        if n == 0:  # 理论上已被 strip_for_tts 拦住，兜底换音色再试一次
            alt = next((k for k in voices if k != lang), None)
            if alt:
                data, n, sr = self._render(alt, text)
                if n:
                    lang = alt
        if n == 0:
            raise ValueError(f"文本没有可发音内容（{text!r}）")
        out.write_bytes(data)
        cost = time.time() - t0
        dur = n / sr
        return {"tts_s": round(cost, 3), "audio_s": round(dur, 2), "sr": sr,
                "rtf": round(cost / max(dur, 1e-9), 3), "voice": lang}


# ------------------------------------------------------------------ LLM
class Llm:
    """OpenAI 兼容流式客户端。用 urllib 而不是 httpx，少一个依赖。"""

    def __init__(self, base: str, model: str, timeout: float = 30.0,
                 reply_lang: str = "auto", api_key: str = "",
                 think: bool = False):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.reply_lang = reply_lang
        self.api_key = api_key
        # Qwen3.5 的 chat template 未定义 enable_thinking 时**默认走思考分支**：
        # 160 token 全烧在 reasoning 上，content 恒为空，表现就是「每轮等很久却一个字没有」。
        # 字幕那边踩过一次（README 4.18），这里同理，默认关。
        self.think = think
        self._probe: bool | None = None
        self._probed_at = 0.0
        self.screen_rule = (
            "「[屏幕文字]」で囲まれた部分は、ユーザーの画面をOCRで読み取ったテキストです。"
            "画面について聞かれたらそれを根拠に答え、OCR の誤字は文脈で補正してください。"
            "屏幕文字里没有的信息，不要凭想象补充。"
        )
        self.system_core = (
            "あなたは日本語と中国語のバイリンガル音声アシスタントです。"
            "音声で読み上げられる回答なので、1〜3文・60文字以内で簡潔に答えてください。"
            "箇条書き・記号・Markdown・絵文字は使わないでください。"
            "固有名詞や技術用語は正確に。不確かなことは推測せず「確認が必要です」と言ってください。"
        )
        # 对外保留完整版（含屏幕规则），tools/voice_llm_ab.py 拿它构造带 OCR 的用例
        self.system = self.system_core + self.screen_rule

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _lang_directive(self, text: str) -> str:
        """把「用什么语言回答」变成硬约束。

        只靠 system prompt 里「跟着用户的语言」是不够的：实测前几轮是中文时，
        用户改说日语，模型仍然用中文回答 —— 它跟的是上下文风格，不是当前这一句。
        所以每轮都按当前这句的语言追加一条明确指令。

        **这条指令必须挂在最后一条 user 消息末尾，不能挂在 system 末尾**（2026-11-21 实测）：
            reply-lang=zh，两个场景 × 3 次，判据是输出语言
            system 末尾追加：7B 3/6 中文，35B 0/6 中文（全回日语）
            user 末尾追加：  7B 6/6 中文，35B 6/6 中文
        system 通篇是日语，指令挂在最前面/最后面都会被前面的语境压过去；
        挂在 user 末尾就是紧贴生成位置，两个模型都听。35B 尤其明显——
        它「太懂语境」，用户说日语、屏幕是日语，它就无视远处的中文指令。
        """
        lang = self.reply_lang
        if lang in ("auto", "", None):
            if KANA.search(text):
                lang = "ja"
            elif CJK.search(text):
                lang = "zh"
            else:
                lang = "ja"
        if lang == "ja":
            return "\n今回の回答は必ず日本語で書いてください。中国語は使わないでください。"
        if lang == "zh":
            return "\n本次回答必须使用简体中文，不要使用日语。"
        return ""

    def available(self, ttl: float = 5.0) -> bool:
        """带 5 秒缓存的探活。

        不在启动时一次性决定：vLLM 加载要 90 秒左右，如果本服务比它先起，
        一次性判断会把 LLM 永久降级成 None，之后即使 vLLM 好了也不会恢复。
        """
        now = time.time()
        if self._probe is not None and now - self._probed_at < ttl:
            return self._probe
        try:
            # Magnitude 的 /inference/v1 要 Bearer，vLLM 不校验；带上无害
            req = urllib.request.Request(self.base + "/models",
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=3) as r:
                self._probe = r.status == 200
        except Exception:  # noqa: BLE001
            self._probe = False
        self._probed_at = now
        return self._probe

    def stream(self, text: str, history: list[dict]):
        # system 按「这轮到底看没看屏幕」二选一，而不是写一份常驻的。
        # 为什么：system 里只要出现「[屏幕文字]」这几个字，模型就会引用它 ——
        # 没看图时也答「屏幕文字中没有显示…」，听起来像在推卸（35B 比 7B 更明显，
        # 实测加一条「没有屏幕文字时不要提屏幕」的否定指令也压不住）。
        # 干脆不给它这个词，问题消失。
        sysp = self.system if "[屏幕文字]" in text else self.system_core
        # 语言指令挂两处：system 末尾 + user 末尾。
        # 只挂 user 时，遇到「system 通篇日语 + 用户说日语」35B 仍回日语（3/3 实测）；
        # system 里也得有一句，才能压住整体语境。两处都挂最稳。
        ling = self._lang_directive(text)
        msgs = [{"role": "system", "content": sysp + ling}]
        msgs += history[-6:]
        msgs.append({"role": "user", "content": text + ling})
        body = json.dumps({
            "model": self.model, "messages": msgs, "stream": True,
            "temperature": 0.6, "max_tokens": 160,
            "presence_penalty": 0.3, "frequency_penalty": 0.3,
            "chat_template_kwargs": {"enable_thinking": self.think},
        }).encode()
        req = urllib.request.Request(
            self.base + "/chat/completions", data=body, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                ch = obj.get("choices") or []
                if not ch:
                    continue
                piece = (ch[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield piece


SENT_END = re.compile(r"(?:(?<!\d)\.(?!\d)|[。．!?！？\n])")
MD_NOISE = re.compile(r"^[\s*#>`\-]+|[*#`]+")


# 真正能发音的字符：假名 / 汉字 / 拉丁字母 / 数字。
# 只由标点组成的片段（"。" "……" "「」"）交给 Piper 会得到 0 帧音频，
# wave 模块随后抛 "# channels not specified" —— 那是空音素化的表现，不是音频设备问题。
SPEAKABLE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF0-9A-Za-z]")


def has_speech(s: str) -> bool:
    return bool(SPEAKABLE.search(s))


def strip_for_tts(s: str) -> str:
    s = MD_NOISE.sub("", s)
    s = re.sub(r"[（(]\s*[)）]", "", s)
    s = s.strip()
    return s if has_speech(s) else ""


# ------------------------------------------------------------------ 打断
class SpeakState:
    """播报时钟：估算「前端还在播」到什么时候。

    为什么需要它：合成本身比播放快一个数量级（RTF 0.1），`TurnRunner.run()`
    返回时前面几段还在浏览器里排队播。打断如果按「生成结束」判断，用户会在
    助理还在说话的时候就被当成「已经说完了」，根本没有打断窗口。

    这里用一条虚拟播放时间线：每发一段音频就把结束时刻往后推 audio_s 秒，
    队列排空时它自然归零。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._until = 0.0
        self._texts: list[str] = []
        self._cancel = threading.Event()
        self.turn_id = ""

    def begin(self, turn_id: str) -> None:
        with self._lock:
            self.turn_id = turn_id
            self._until = 0.0
            self._texts = []
        self._cancel.clear()

    def add(self, audio_s: float, text: str) -> None:
        with self._lock:
            now = time.time()
            # 上一段还没播完就接到它后面，否则从现在开始
            self._until = max(now, self._until) + audio_s
            self._texts.append(text)
            del self._texts[:-6]

    def playing(self) -> bool:
        with self._lock:
            return time.time() < self._until

    def remain(self) -> float:
        with self._lock:
            return max(0.0, self._until - time.time())

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            self._until = 0.0

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def said(self) -> list[str]:
        with self._lock:
            return list(self._texts)


class LocalSpeaker:
    """把合成好的 wav 直接播到 Jetson 自己的声卡（耳机）。

    为什么需要它：播报默认交给浏览器播放，而浏览器是「哪台机器开页面，
    声音就从哪台出」。耳机插在 Jetson 上却听不见，正是因为页面开在另一台
    电脑上 —— 麦克风在 Jetson、喇叭在电脑，两端分离。

    这里走 pw-play 而不是 aplay：麦克风被 AudioHub 以 ALSA 独占方式占着，
    播放走 PipeWire 才不会和它抢设备（USB 声卡的播放/录音是两条独立流）。
    """

    def __init__(self, sink: str | None = None, gain: float = 1.0,
                 enabled: bool = False):
        self.sink = sink or None       # None = PipeWire 默认输出
        self.gain = gain
        self.enabled = enabled
        self.plays = 0
        self.errors = 0
        self.last_error = ""
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="local-speak", daemon=True)
        self._t.start()

    def play(self, path, gain: float | None = None) -> bool:
        try:
            self._q.put_nowait((Path(path), self.gain if gain is None else gain))
            return True
        except queue.Full:          # 队列满说明播报严重滞后，丢新的比积压好
            self.errors += 1
            self.last_error = "播放队列已满"
            return False

    def cancel(self) -> None:
        """打断：清掉待播，并掐掉正在播的那一句。"""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass

    def busy(self) -> bool:
        with self._lock:
            p = self._proc
        return bool(p is not None and p.poll() is None) or not self._q.empty()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                path, gain = self._q.get(timeout=0.4)
            except queue.Empty:
                continue
            cmd = ["pw-play", "--volume", f"{max(0.0, min(1.0, gain)):.2f}"]
            if self.sink:
                cmd += ["--target", self.sink]
            cmd.append(str(path))
            try:
                with self._lock:
                    self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.PIPE)
                    p = self._proc
                _, err = p.communicate(timeout=90)
                if p.returncode == 0:
                    self.plays += 1
                elif p.returncode is not None and p.returncode < 0:
                    pass                                # 被 terminate（打断），正常
                else:
                    self.errors += 1
                    self.last_error = (err or b"").decode("utf-8", "replace").strip()[:120]
                    print(f"[local] 播放失败 rc={p.returncode} {self.last_error}", flush=True)
            except FileNotFoundError:
                self.errors += 1
                self.last_error = "找不到 pw-play（需要 pipewire-bin）"
                print("[local] " + self.last_error, flush=True)
                time.sleep(2)
            except subprocess.TimeoutExpired:
                with self._lock:
                    p = self._proc
                if p is not None:
                    p.kill()
                self.errors += 1
                self.last_error = "播放超时"
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                self.last_error = str(e)[:120]


_ECHO_STRIP = re.compile(r"[\s、。，．,.!?！？…「」『』（）()\-—~〜]")


def looks_like_echo(heard: str, spoken, thresh: float = 0.7) -> bool:
    """播报期检测到的「人声」是不是扬声器串音。

    外放时麦克风会听到助理自己的声音，不排除就会自激：一播报就被打断，
    而打断后重新播报又被自己打断。

    判据是**最长公共子串占识别文本的比例**，而不是整句相似度。因为串音窗口
    只有一两秒，识别出来就是三五个字（"今天天"），拿它和十四字的整句算
    SequenceMatcher.ratio 得 0.35，永远够不着阈值。但反过来看，"今天天" 有
    3/3 的字落在播报文本里连成一片，这就非常可疑了。
    """
    h = _ECHO_STRIP.sub("", heard or "")
    if len(h) < 2:
        # 给不出任何统计信息，一律当成人声：宁可多打断一次，
        # 也不要漏掉「停」「等一下」这种短指令
        return False
    cands = [spoken] if isinstance(spoken, str) else list(spoken or [])
    for c in cands:
        s = _ECHO_STRIP.sub("", c or "")
        if len(s) < 2:
            continue
        if h in s:
            return True          # 完整命中，最干脆的情况
        m = difflib.SequenceMatcher(None, h, s).find_longest_match(0, len(h), 0, len(s))
        if m.size >= 3 and m.size >= len(h) * thresh:
            return True
    return False


class BargeMonitor(threading.Thread):
    """播报期打断：助理说话说到一半，用户直接插话就停下听。

    判定链三道，任何一道不过就不打断：
      1) 能量：高于「播报期本底 + margin」。播报期本底含扬声器串音，
         所以外放时门限会自动抬高，避免自激。
      2) 语音概率：Silero VAD 判为人声，过滤键盘、椅子、风扇这类瞬态噪声。
      3) 回声排除：把这一小段送去 ASR，文本与正在播的句子高度重合 = 串音。
         第 3 道要花约 60ms，所以放在能量+VAD 都通过之后，正常情况不付出代价。

    打断后不重新开设备：把已读到的帧原样交给录音，接着从同一条流继续读，
    用户那句话的开头不会丢。
    """

    def __init__(self, hub: AudioHub, asr, speak: SpeakState, runner,
                 history: list[dict], push, turn_lock: threading.Lock,
                 vad_model: str, min_db: float = -42.0, margin: float = 8.0,
                 hold_s: float = 0.30, max_db: float = -18.0):
        super().__init__(name="barge-monitor", daemon=True)
        self.hub = hub
        self.asr = asr
        self.speak = speak
        self.runner = runner
        self.history = history
        self.push = push
        self.turn_lock = turn_lock
        self.vad_model = vad_model
        self.min_db = min_db
        self.margin = margin
        self.hold_s = hold_s
        self.max_db = max_db

        # 最近 1.2 秒的帧：打断时整段交给 ASR/录音，开头不会被切
        self.frames: deque = deque(maxlen=int(1.2 / FRAME_S))
        self._stop = threading.Event()
        self.hits = 0
        self.echo_drops = 0
        self.quiet_floor = -60.0     # 非播报期本底，决定门限
        self.echo_floor = -60.0      # 播报期电平（含串音），只做诊断
        self.last_db = -99.0
        self._cooldown = 0.0         # 判为串音后的静默期，避免每 0.3s 重判一次
        self._in_play = False

    def stop(self) -> None:
        self._stop.set()

    def _make_vad(self):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = self.vad_model
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.25
        cfg.silero_vad.min_speech_duration = 0.10
        cfg.sample_rate = SR
        return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=5)

    def threshold(self) -> float:
        """当前该用的打断门限（dBFS）。

        只跟「静默期本底」走，**不**参考播报期电平。原因：外放时播报期电平就是
        扬声器串音，用它抬门限会让真人插话也打不断；而且串音和真人的音量本来就
        相当，电平上根本分不开。串音交给 ASR 文本比对处理，那才是能分辨的维度。
        """
        return min(max(self.min_db, self.quiet_floor + self.margin), self.max_db)

    def run(self) -> None:
        q = self.hub.subscribe(pre_s=0.0)
        vad = self._make_vad()
        holding = 0.0        # 已连续判为人声的秒数
        voiced: list[np.ndarray] = []

        while not self._stop.is_set():
            try:
                frame = q.get(timeout=0.2)
                if frame is None:
                    break
            except queue.Empty:
                continue

            self.frames.append(frame)
            db = vp.db_of(frame.astype(np.float64))
            self.last_db = db

            if not self.speak.playing():
                # 没在播报：只更新本底，不跑 VAD（几乎零开销）
                self.quiet_floor = min(db, self.quiet_floor + 0.05)
                self.quiet_floor = max(self.quiet_floor, -80.0)
                self.echo_floor = min(db, self.echo_floor + 0.05)
                holding = 0.0
                voiced.clear()
                self._cooldown = 0.0
                self._in_play = False
                continue

            if not self._in_play:
                # 刚进入播报：VAD 在非播报期没被喂帧，带着上一轮的残留状态，
                # 直接复用会让「上一句的尾音」算进这一轮的 holding
                vad = self._make_vad()
                self._in_play = True
                holding = 0.0
                voiced.clear()

            vad.accept_waveform(frame.astype(np.float32) / vp.FULL_SCALE)
            is_voiced = vad.is_speech_detected()
            hot = db > self.threshold()

            if self._cooldown > 0:
                # 刚判为串音：这段时间不再判定，但仍持续收帧（判别要用完整窗口）
                self._cooldown -= FRAME_S
                holding = 0.0
                voiced.clear()
                continue

            if hot and is_voiced:
                holding += FRAME_S
                voiced.append(frame)
            else:
                holding = max(0.0, holding - FRAME_S * 1.5)
                if not hot:
                    self.echo_floor = min(db, self.echo_floor + 0.08)

            if holding >= self.hold_s and voiced:
                is_echo = self._on_barge()
                holding = 0.0
                voiced.clear()
                vad = self._make_vad()   # 状态清干净，避免同一次语音被连判
                if is_echo:
                    self._cooldown = 1.2

        self.hub.unsubscribe(q)

    def _on_barge(self) -> bool:
        """判定并处理一次疑似插话。返回 True 表示判成了串音（需要冷却）。"""
        # 用最近 0.9 秒而不是那 0.3 秒的 voiced：片段太短 ASR 只吐出两个字，
        # 与播报文本做相似度毫无意义。窗口长一点，判别才站得住。
        tail = list(self.frames)[-int(0.9 / FRAME_S):]
        clip = np.concatenate(tail) if tail else np.concatenate(self.frames)
        heard = ""
        try:
            heard, _ = self.asr.run(clip, SR)
        except Exception:  # noqa: BLE001
            pass

        if looks_like_echo(heard, self.speak.said()):
            self.echo_drops += 1
            print(f"[barge] 疑似串音，忽略：「{heard[:24]}」", flush=True)
            return True

        self.hits += 1
        print(f"[barge] 打断：「{heard[:24]}」", flush=True)
        self.speak.cancel()
        # 本地播报也要立刻掐掉，否则耳机里还在自顾自念
        spk = getattr(self.runner, "speaker", None)
        if spk is not None:
            spk.cancel()
        self.push({"type": "interrupt", "heard": heard})
        self._start_turn()
        return False

    def _start_turn(self) -> None:
        # 带上打断检测已经读走的那 1.2 秒，用户那句话的开头不能丢
        q = self.hub.subscribe(pre_s=0.0, pre_frames=list(self.frames))

        def work():
            try:
                with self.turn_lock:
                    self.runner.run(self._emit, source=q, history=list(self.history))
            except Exception as e:  # noqa: BLE001
                self.push({"type": "error", "message": f"打断后处理失败：{e}"})
            finally:
                self.hub.unsubscribe(q)

        threading.Thread(target=work, name="barge-turn", daemon=True).start()

    def _emit(self, ev: dict) -> None:
        # 被打断的那轮不进历史：它是不完整的，留着会污染后续上下文
        if ev["type"] == "done" and not ev.get("interrupted"):
            if ev.get("user") and ev.get("reply"):
                self.history.append({"role": "user", "content": ev["user"]})
                self.history.append({"role": "assistant", "content": ev["reply"]})
        self.push(ev)


# ------------------------------------------------------------------ 单轮编排
class TurnRunner:
    def __init__(self, recorder, asr, tts, llm, audio_dir: Path, hub=None,
                 vision=None, speaker=None):
        self.recorder = recorder
        self.asr = asr
        self.tts = tts
        self.llm = llm
        self.audio_dir = audio_dir
        self.hub = hub  # AudioHub；有它就不能再自己开 arecord，否则抢设备
        self.vision = vision  # Vision；有它才能「看一眼屏幕」
        # 本轮已确定的音色；None 表示还没说过话，由 pick() 自己判断
        self.speak_lang: str | None = None
        # 播报时钟：打断检测靠它判断「现在还在说话」
        self.speak = SpeakState()
        # 本地播报（播到 Jetson 自己的耳机）；None = 只让浏览器播
        self.speaker = speaker

    def run(self, emit, *, wav: str | None = None, text: str | None = None,
            history: list[dict] | None = None,
            source: "queue.Queue | None" = None,
            strip_prefix: str | None = None,
            look: bool = False) -> None:
        history = history or []
        timings: dict[str, float] = {}
        turn_id = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        started = time.time()
        user_text = ""
        reply = ""
        interrupted = False
        lang_hint = getattr(self.llm, "reply_lang", "auto")
        self.speak_lang = lang_hint if lang_hint in ("zh", "ja") else None
        self.speak.begin(turn_id)

        try:
            # 1) 取音频
            if text:
                user_text = text
                emit({"type": "asr", "text": user_text, "skipped": True})
            else:
                if wav:
                    audio, sr = vp.read_wav(wav)
                    audio, info = trim_by_vad(audio, sr)
                    info["source"] = "file"
                    emit({"type": "state", "state": "transcribing", "record": info})
                else:
                    emit({"type": "state", "state": "listening"})
                    t0 = time.time()
                    src, own = source, None
                    if src is None and self.hub is not None:
                        src = own = self.hub.subscribe(pre_s=0.15)
                    try:
                        audio, sr, info = self.recorder.record(src)
                    finally:
                        if own is not None:
                            self.hub.unsubscribe(own)
                    timings["record_s"] = round(time.time() - t0, 2)
                    emit({"type": "state", "state": "transcribing", "record": info})

                t0 = time.time()
                user_text, asr_cost = self.asr.run(audio, sr)
                if strip_prefix and user_text.startswith(strip_prefix):
                    # 唤醒词本身的尾音会被一起录进来，去掉它再送给 LLM
                    user_text = user_text[len(strip_prefix):].lstrip("，,。.、 　")
                timings["asr_s"] = round(time.time() - t0, 2)
                timings["asr_core_s"] = round(asr_cost, 2)
                if not user_text:
                    emit({"type": "error", "message": "识别结果为空，可能没说话或声音太小"})
                    emit({"type": "done", "timings": timings})
                    return
                emit({"type": "asr", "text": user_text, "ms": int(timings["asr_s"] * 1000)})

            # 2) 「看一眼屏幕」：手动按钮，或话里带了「看看屏幕上是什么」这类意图
            if (look or vx.is_look_intent(user_text)) and self.vision is not None:
                emit({"type": "state", "state": "looking"})
                # 先垫一句。取帧+OCR 实测约 8 秒，不能让人干等着不出声
                self._speak(emit, "我看一下。", 0, turn_id, timings)
                t_look = time.time()
                info = self.vision.look(force=True)
                timings["look_s"] = round(time.time() - t_look, 2)
                emit({"type": "look", "info": info, "ms": int(timings["look_s"] * 1000)})
                ctx = vx.format_context(info)
                if ctx:
                    # 屏幕文字放在问题前面：模型先看到「看到什么」，再看到「被问什么」
                    user_text = f"{ctx}\n\n{user_text}"
                else:
                    # 看不到就必须说看不到，否则模型会照着想象编
                    emit({"type": "error",
                          "message": f"看不到屏幕：{info.get('message') or info.get('reason')}"})

            # 3) LLM 流式 → 按句合成
            idx = 0
            buf = ""
            if self.llm is None or not self.llm.available():
                reply = f"（LLM 未就绪，回显）{user_text}"
                emit({"type": "delta", "text": reply})
                buf = reply
            else:
                t_llm = time.time()
                first = True
                try:
                    for piece in self.llm.stream(user_text, history):
                        # 用户插话：立刻停止生成，别把剩下的说完再停
                        if self.speak.cancelled:
                            interrupted = True
                            break
                        if first:
                            timings["llm_first_ms"] = int((time.time() - t_llm) * 1000)
                            first = False
                        reply += piece
                        buf += piece
                        emit({"type": "delta", "text": piece})
                        while True:
                            m = SENT_END.search(buf)
                            if not m:
                                break
                            sent, buf = buf[:m.end()], buf[m.end():]
                            if strip_for_tts(sent):
                                idx += 1
                                self._speak(emit, sent, idx, turn_id, timings)
                    timings["llm_s"] = round(time.time() - t_llm, 2)
                except Exception as e:  # noqa: BLE001
                    emit({"type": "error", "message": f"LLM 调用失败：{e}"})
                    if not reply:
                        reply = f"（LLM 调用失败，回显）{user_text}"
                        buf = reply
                        emit({"type": "delta", "text": reply})

            # 打断后不再补发剩余的尾巴：用户已经不想听了
            if strip_for_tts(buf) and not self.speak.cancelled:
                idx += 1
                self._speak(emit, buf, idx, turn_id, timings)

            timings["total_s"] = round(time.time() - started, 2)
            emit({"type": "done", "timings": timings, "reply": reply,
                  "user": user_text, "chunks": idx, "interrupted": interrupted})
            self._log(turn_id, user_text, reply, timings, idx, interrupted)

        except RecordError as e:
            emit({"type": "error", "message": str(e)})
            emit({"type": "done", "timings": timings})
        except Exception as e:  # noqa: BLE001
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            emit({"type": "done", "timings": timings})

    def _speak(self, emit, sentence: str, idx: int, turn_id: str, timings: dict) -> None:
        text = strip_for_tts(sentence)
        if not text:  # 纯标点/空白：合成会得到 0 帧音频，直接跳过
            return
        if self.speak.cancelled:  # 已经插话了，别再合成
            return
        out = self.audio_dir / f"{turn_id}_{idx:02d}.wav"
        try:
            meta = self.tts.synth_to(text, out, hint=self.speak_lang)
        except Exception as e:  # noqa: BLE001
            emit({"type": "error", "message": f"合成失败：{e}"})
            return
        # 记住这句的音色，供下一句的数字/英文片段继承
        self.speak_lang = meta["voice"]
        if "tts_first_ms" not in timings:
            timings["tts_first_ms"] = int(meta["tts_s"] * 1000)
        # 本地播报开关：开着就直接播到 Jetson 的耳机，浏览器只显示文字
        local = bool(self.speaker and self.speaker.enabled)
        if local:
            self.speaker.play(out)
        emit({"type": "audio", "url": f"/audio/{out.name}", "idx": idx,
              "text": text, "local": local, **meta})
        # 推进播报时钟：打断检测按这个时间线判断「还在说话」
        self.speak.add(meta["audio_s"], text)

    def _log(self, turn_id, user, reply, timings, chunks, interrupted=False) -> None:
        try:
            with TURN_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "turn": turn_id,
                    "user": user, "reply": reply, "timings": timings, "chunks": chunks,
                    "interrupted": interrupted,
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass


# ------------------------------------------------------------------ HTTP
def build_app(cfg) -> "object":
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

    app = FastAPI(title="voice test bench")
    lock = asyncio.Lock()
    runner = TurnRunner(cfg.recorder, cfg.asr, cfg.tts, cfg.llm, AUDIO_DIR,
                        hub=getattr(cfg, "hub", None),
                        vision=getattr(cfg, "vision", None),
                        speaker=getattr(cfg, "speaker", None))
    history: list[dict] = []
    live: set = set()          # (loop, asyncio.Queue) 每个 /api/live 连接一个
    turn_lock = threading.Lock()  # 唤醒轮与手动轮互斥（跨线程）
    wake_ref: dict = {}
    barge_ref: dict = {}
    # 事件环形缓冲：短轮询从这里取增量。长连接（/api/live）在浏览器/代理环境下
    # 可能建不起来——实测就是这种情况，页面一直没连上，唤醒全被判定「没人听」。
    EVENTS: deque = deque(maxlen=600)
    seq_counter = itertools.count(1)
    poll_state = {"at": 0.0, "n": 0}
    visits = {"n": 0, "at": 0.0}  # 页面每次 health 都会 +1

    def push(ev: dict) -> None:
        """记录事件并推给长连接订阅者。"""
        seq = next(seq_counter)
        EVENTS.append((seq, ev))
        for loop, q in list(live):
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:  # 事件循环已关闭
                live.discard((loop, q))

    @app.get("/", response_class=HTMLResponse)
    async def index():
        word = cfg.wake.word if cfg.wake else ""
        return HTMLResponse(HTML.replace("__WAKE_WORD__", word),
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    async def health():
        visits["n"] += 1
        visits["at"] = time.time()
        return {
            "asr": True, "asr_lang": cfg.asr.language,
            "tts": True, "tts_load_s": round(cfg.tts.load_s, 2),
            "tts_voices": list(cfg.tts.voices),
            "llm": bool(cfg.llm and cfg.llm.available()),
            "llm_model": cfg.llm.model if cfg.llm else None,
            "busy": lock.locked(),
            "device": cfg.recorder.device,
            "turns": len(history),
            "wake": cfg.wake.word if cfg.wake else None,
            "wake_hits": cfg.wake.hits if cfg.wake else 0,
            "wake_turns": wake_ref.get("loop").turns if wake_ref.get("loop") else 0,
            "wake_loop": bool(wake_ref.get("loop") and wake_ref["loop"].is_alive()),
            "wake_skipped": getattr(wake_ref.get("loop"), "skipped", 0),
            "live": len(live),
            "poll_s": round(time.time() - poll_state["at"], 1) if poll_state["at"] else -1.0,
            "visits": visits["n"],
            "hub": ({"alive": cfg.hub.alive, "errors": cfg.hub.errors,
                     "rate_s": cfg.hub.rate_s} if cfg.hub else None),
            "speaker": ({"on": cfg.speaker.enabled, "sink": cfg.speaker.sink,
                         "gain": cfg.speaker.gain, "plays": cfg.speaker.plays,
                         "errors": cfg.speaker.errors,
                         "busy": cfg.speaker.busy(),
                         "last_error": cfg.speaker.last_error}
                        if getattr(cfg, "speaker", None) else None),
            "vision": ({"dev": cfg.vision.dev, "calls": cfg.vision.calls,
                        "errors": cfg.vision.errors,
                        "w": cfg.vision.max_side, "threads": cfg.vision.ocr_threads,
                        "trt": cfg.vision.trt_status(),
                        "trt_note": cfg.vision.trt_note or None,
                        "last_ok": (cfg.vision.last or {}).get("ok"),
                        "last_ms": round((cfg.vision.last or {}).get("total_s", 0) * 1000),
                        "capture": cfg.vision.capture_status()}
                       if getattr(cfg, "vision", None) else None),
            "barge": ({"on": True, "hits": barge_ref["m"].hits,
                       "echo": barge_ref["m"].echo_drops,
                       "thr_db": round(barge_ref["m"].threshold(), 1),
                       "last_db": round(barge_ref["m"].last_db, 1),
                       "playing": runner.speak.playing(),
                       "remain_s": round(runner.speak.remain(), 2)}
                      if barge_ref.get("m") else {"on": False}),
        }

    @app.post("/api/reset")
    async def reset():
        history.clear()
        return {"ok": True}

    @app.post("/api/look")
    async def look(request: Request):
        """手动看一眼屏幕：只做取帧 + OCR，不调 LLM。"""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        vis = getattr(cfg, "vision", None)
        if vis is None:
            raise HTTPException(503, "vision 未启用（启动时加了 --no-vision？）")
        force = bool(payload.get("force"))
        # 取帧 + OCR 约 8 秒，必须丢线程池，否则整个事件循环（含轮询）都卡住
        info = await asyncio.get_running_loop().run_in_executor(
            None, lambda: vis.look(force=force))
        push({"type": "look", "info": info})
        return info

    @app.post("/api/say")
    async def say(request: Request):
        """只合成一句、不调 LLM。用来把「没声音」拆成服务端 / 浏览器两半。"""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        text = strip_for_tts(str(payload.get("text") or "声音测试，一二三。")) or "声音测试"
        out = AUDIO_DIR / f"say_{int(time.time() * 1000)}.wav"
        try:
            info = await asyncio.get_running_loop().run_in_executor(
                None, lambda: cfg.tts.synth_to(text, out, hint="zh"))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"合成失败：{e}"}
        # 本地播报开着时，试听也走耳机，这样「试听有没有声」和「实际有没有声」是同一路径
        spk = getattr(cfg, "speaker", None)
        local = bool(spk and spk.enabled)
        if local:
            spk.play(out, gain=float(payload.get("gain") or spk.gain))
        return {"ok": True, "url": f"/audio/{out.name}", "local": local, **info}

    @app.post("/api/route")
    async def route(request: Request):
        """切换播报出口：local=true 播到 Jetson 的耳机，false 交给浏览器。"""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        spk = getattr(cfg, "speaker", None)
        if spk is None:
            raise HTTPException(503, "本地播报未启用（启动时没建 speaker）")
        if "on" in payload or "local" in payload:
            spk.enabled = bool(payload.get("on", payload.get("local")))
            if not spk.enabled:
                spk.cancel()          # 关掉时把还在排队的掐了，免得切完还念一句
        if payload.get("gain") is not None:
            try:
                spk.gain = max(0.0, min(1.0, float(payload["gain"])))
            except (TypeError, ValueError):
                pass
        print(f"[route] 播报出口 = {'耳机(本地)' if spk.enabled else '浏览器'}"
              f" gain={spk.gain:.2f}", flush=True)
        return {"ok": True, "on": spk.enabled, "gain": spk.gain, "sink": spk.sink}

    async def stream_turn(payload: dict, loop):
        q: asyncio.Queue = asyncio.Queue()

        def emit(ev):
            loop.call_soon_threadsafe(q.put_nowait, ev)

        def work():
            try:
                with turn_lock:  # 与唤醒轮互斥：同一时刻只跑一轮
                    runner.run(emit, wav=payload.get("wav"), text=payload.get("text"),
                               history=list(history), look=bool(payload.get("look")))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = await q.get()
            if ev is None:
                break
            if ev["type"] == "done" and payload.get("remember", True):
                if not ev.get("interrupted") and ev.get("user") and ev.get("reply"):
                    history.append({"role": "user", "content": ev["user"]})
                    history.append({"role": "assistant", "content": ev["reply"]})
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    @app.post("/api/turn")
    async def turn(request: Request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        if lock.locked() or turn_lock.locked():
            raise HTTPException(409, "上一轮还在进行中")
        loop = asyncio.get_running_loop()

        async def body():
            async with lock:
                async for chunk in stream_turn(payload, loop):
                    yield chunk

        return StreamingResponse(body(), media_type="application/x-ndjson",
                                 headers={"X-Accel-Buffering": "no",
                                          "Cache-Control": "no-store"})

    @app.get("/audio/{name}")
    async def audio(name: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.wav", name):
            raise HTTPException(400, "bad name")
        p = AUDIO_DIR / name
        if not p.exists():
            raise HTTPException(404, "gone")
        return FileResponse(p, media_type="audio/wav")

    @app.get("/api/poll")
    async def poll(after: int = 0):
        """短轮询：返回 seq > after 的所有事件。

        比 EventSource 笨，但没有长连接可被代理/浏览器掐断的问题，
        400ms 一次的开销可以忽略。after 超过当前最大 seq 时空返回，
        前端用它来「对齐到今天」，不会重放旧事件。
        """
        poll_state["at"] = time.time()
        poll_state["n"] += 1
        evs = events_since(EVENTS, after)
        last = EVENTS[-1][0] if EVENTS else after
        return {"last": last, "events": evs,
                "wake": cfg.wake.word if cfg.wake else None,
                "loop": bool(wake_ref.get("loop") and wake_ref["loop"].is_alive())}

    @app.get("/api/live")
    async def live_stream():
        """常驻监听的事件流：唤醒 → 识别 → 回答 → 音频，一直不断开。

        不用 POST：浏览器端用 EventSource，断线自动重连；15 秒没事件就发心跳，
        免得中间设备把空闲连接掐掉。
        """
        if cfg.wake is None:
            raise HTTPException(503, "唤醒未启用（启动参数 --wake-word）")
        q: asyncio.Queue = asyncio.Queue()
        item = (asyncio.get_running_loop(), q)
        live.add(item)

        async def body():
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield json.dumps({"type": "ping"}) + "\n"
                        continue
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
            finally:
                live.discard(item)

        return StreamingResponse(body(), media_type="application/x-ndjson",
                                 headers={"X-Accel-Buffering": "no",
                                          "Cache-Control": "no-store"})

    @app.post("/api/kws_check")
    async def kws_check():
        """录 3 秒，存成 wav 并跑一次 ASR。

        用来区分「唤醒词没说对/太小」和「KWS 阈值不合适」：ASR 文本对得上，
        说明声音本身没问题，问题在阈值；文本都不对，是发音或拾音问题。
        """
        if cfg.wake is None or cfg.hub is None:
            raise HTTPException(503, "唤醒未启用")
        loop = asyncio.get_running_loop()

        def work():
            q = cfg.hub.subscribe(pre_s=0.0)
            frames: list[np.ndarray] = []
            t0 = time.time()
            try:
                while time.time() - t0 < 3.0:
                    try:
                        frames.append(q.get(timeout=0.5))
                    except queue.Empty:
                        break
            finally:
                cfg.hub.unsubscribe(q)
            if not frames:
                return {"error": "没有采集到音频（麦克风没在出帧）"}
            audio = np.concatenate(frames)
            out = AUDIO_DIR / f"kwscheck_{int(t0)}.wav"
            write_wav(out, audio, SR)
            text, _ = cfg.asr.run(audio, SR)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            return {"text": text, "wav": out.name, "sec": round(audio.size / SR, 2),
                    "peak_db": round(20.0 * np.log10(max(peak, 1.0) / vp.FULL_SCALE), 1)}

        return await loop.run_in_executor(None, work)

    if cfg.wake is not None:
        loop_thread = WakeLoop(cfg.hub, cfg.wake, runner, history, push, turn_lock,
                               has_listener=lambda: bool(live) or (
                                   time.time() - poll_state["at"] < 3.0))
        wake_ref["loop"] = loop_thread
        loop_thread.start()

        # 打断只在常驻模式下可能：手动模式录音设备由 record() 独占，没法旁听
        if getattr(cfg, "barge", False):
            mon = BargeMonitor(cfg.hub, cfg.asr, runner.speak, runner, history, push,
                               turn_lock, str(cfg.vad_model),
                               min_db=cfg.barge_db, margin=cfg.barge_margin)
            mon.start()
            barge_ref["m"] = mon
            print(f"      打断已启用  门限 {cfg.barge_db}dB  余量 {cfg.barge_margin}dB",
                  flush=True)

    return app


HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>语音测试台 · Jetson</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e9ef; --dim:#8b93a7;
    --accent:#5b9cff; --ok:#41c48a; --warn:#e0a34a; --err:#e0645a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:15px/1.6 -apple-system,"Segoe UI",system-ui,"Noto Sans JP",sans-serif;
    display:flex;justify-content:center;padding:24px 16px 48px}
  .wrap{width:100%;max-width:760px}
  h1{font-size:18px;margin:0 0 4px;font-weight:600;letter-spacing:.3px}
  .sub{color:var(--dim);font-size:13px;margin-bottom:18px}
  .pills{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
  .pill{font-size:12px;padding:4px 10px;border-radius:99px;border:1px solid var(--line);
    background:var(--panel);color:var(--dim)}
  .pill b{color:var(--fg);font-weight:600}
  .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
    background:var(--dim);vertical-align:1px}
  .dot.ok{background:var(--ok)} .dot.no{background:var(--err)}
  .micbar{display:flex;align-items:center;gap:16px;margin-bottom:20px}
  button{font:inherit;cursor:pointer;border-radius:12px;border:1px solid var(--line);
    background:var(--panel);color:var(--fg);padding:10px 16px;transition:.15s}
  button:hover:not(:disabled){border-color:#3a4356;background:#1c2029}
  button:disabled{opacity:.45;cursor:not-allowed}
  #talk{display:flex;align-items:center;gap:10px;font-weight:600;padding:14px 22px;
    background:linear-gradient(180deg,#232c3d,#1b2230);border-color:#33405a}
  #talk.rec{background:linear-gradient(180deg,#3a2224,#2a191b);border-color:#6b3438}
  #live.on{background:linear-gradient(180deg,#1b3a2e,#152a22);border-color:#2f6b52;color:#a8e6c8}
  #route.on{background:linear-gradient(180deg,#1b3a2e,#152a22);border-color:#2f6b52;color:#a8e6c8}
  #talk .ring{width:10px;height:10px;border-radius:50%;background:var(--accent)}
  #talk.rec .ring{background:var(--err);animation:pulse 1s infinite}
  @keyframes pulse{50%{opacity:.25}}
  .hint{color:var(--dim);font-size:13px}
  #audiobar{border:1px solid #6b4a2a;background:#2a2119;color:#e8c89a;
    border-radius:12px;padding:10px 14px;margin:-8px 0 18px;font-size:13px;line-height:1.6}
  #audiobar b{color:#ffd9a3}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:16px 18px;margin-bottom:14px}
  .card h2{font-size:12px;letter-spacing:.8px;text-transform:uppercase;color:var(--dim);
    margin:0 0 8px;font-weight:600}
  .txt{white-space:pre-wrap;word-break:break-word;font-size:15px}
  .txt.you{color:#cfd6e6} .txt.bot{color:#dfeaff}
  .txt.ja{font-family:"Hiragino Sans","Noto Sans JP",system-ui,sans-serif}
  .empty{color:#5a6274;font-style:italic}
  .timings{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
  .t{font-size:11.5px;padding:3px 8px;border-radius:6px;background:#12151b;
    border:1px solid var(--line);color:var(--dim);font-variant-numeric:tabular-nums}
  .t b{color:#bcd0f0;font-weight:600}
  .row{display:flex;gap:8px;margin-top:6px}
  input[type=text]{flex:1;font:inherit;background:#12151b;border:1px solid var(--line);
    color:var(--fg);border-radius:10px;padding:10px 13px}
  input[type=text]:focus{outline:none;border-color:#3d5580}
  .log{margin-top:22px;border-top:1px solid var(--line);padding-top:14px}
  .log h3{font-size:12px;color:var(--dim);margin:0 0 10px;letter-spacing:.8px;
    text-transform:uppercase;font-weight:600}
  .logitem{font-size:13px;padding:9px 0;border-bottom:1px dashed #22273180;color:#a8b1c4}
  .logitem .m{color:var(--dim);font-size:11.5px;font-variant-numeric:tabular-nums}
  .err{color:#ffb4ac;background:#2a1719;border:1px solid #5c2a2b;border-radius:10px;
    padding:9px 12px;font-size:13px;margin-top:10px}
  details summary{cursor:pointer;color:var(--dim);font-size:12.5px}
  code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:#a9c0e6}
</style>
</head>
<body>
<div class="wrap">
  <h1>语音测试台</h1>
  <div class="sub">点一下开始说话，或开「常驻监听」后说唤醒词。停顿约 0.7 秒自动收尾；助理说话时直接开口可以打断。</div>

  <div class="pills">
    <span class="pill"><span class="dot" id="d-asr"></span>ASR <b id="v-asr">…</b></span>
    <span class="pill"><span class="dot" id="d-tts"></span>TTS <b id="v-tts">…</b></span>
    <span class="pill"><span class="dot" id="d-llm"></span>LLM <b id="v-llm">…</b></span>
    <span class="pill">唤醒 <b id="v-wake">__WAKE_WORD__</b></span>
    <span class="pill">设备 <b id="v-dev">…</b></span>
  </div>

  <div class="micbar">
    <button id="talk"><span class="ring"></span><span id="talklabel">开始说话</span></button>
    <button id="live">常驻监听</button>
    <button id="probe">测唤醒</button>
    <button id="look">看屏幕</button>
    <button id="aud">声音测试</button>
    <button id="route">声音：耳机</button>
    <span class="hint" id="hint">提示：戴上耳机，麦克风靠近嘴侧</span>
    <span class="hint" id="diag"></span>
  </div>

  <div id="audiobar" style="display:none"></div>

  <div class="card">
    <h2>你说的话</h2>
    <div class="txt you ja" id="you"><span class="empty">尚未录音</span></div>
  </div>

  <div class="card">
    <h2>助理回答</h2>
    <div class="txt bot ja" id="bot"><span class="empty">—</span></div>
    <div class="timings" id="timings"></div>
    <div id="errors"></div>
  </div>

  <div class="card" id="lookcard" style="display:none">
    <h2>屏幕文字 <span class="hint" id="lookmeta"></span></h2>
    <div class="txt" id="looktext" style="white-space:pre-wrap"></div>
  </div>

  <details>
    <summary>不想说话？用文字测同一条链路</summary>
    <div class="row" style="margin-top:10px">
      <input type="text" id="tin" placeholder="この設計書のインターフェース仕様を確認してください。">
      <button id="send">发送</button>
    </div>
  </details>

  <div class="log">
    <h3>历史</h3>
    <div id="log"><span class="hint">还没有记录</span></div>
    <div class="row" style="margin-top:12px">
      <button id="reset">清空上下文</button>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let busy = false, queue = [], playing = false, curAudio = null;
let botText = '', es = null, wakeWord = '__WAKE_WORD__';

async function health(){
  try{
    const r = await fetch('/api/health'); const j = await r.json();
    $('#v-asr').textContent = j.asr ? j.asr_lang : '×';
    $('#d-asr').className = 'dot ' + (j.asr ? 'ok' : 'no');
    $('#v-tts').textContent = j.tts ? 'Piper' : '×';
    $('#d-tts').className = 'dot ' + (j.tts ? 'ok' : 'no');
    $('#v-llm').textContent = j.llm ? j.llm_model : '未就绪';
    $('#d-llm').className = 'dot ' + (j.llm ? 'ok' : 'no');
    wakeWord = j.wake || '';
    $('#v-wake').textContent = wakeWord || '未启用';
    $('#v-dev').textContent = j.device.replace('plughw:CARD=','').replace(',DEV=0','');
    if(!j.llm) $('#hint').textContent = 'LLM 未就绪：当前只会复述你的话（ASR→TTS 通路仍可测）';
  }catch(e){ $('#v-asr').textContent='离线'; }
}

function setBusy(b){
  busy = b;
  $('#talk').disabled = b || !!es;
  $('#send').disabled = b;
  $('#talk').classList.toggle('rec', b && $('#talklabel').textContent.includes('听'));
}

function showAudioBar(msg, clickable){
  const el = $('#audiobar'); if(!el) return;
  el.style.display = 'block';
  el.style.cursor = clickable ? 'pointer' : 'default';
  el.innerHTML = clickable ? '<b>' + msg + '</b>（点这里启用声音）' : msg;
}
function hideAudioBar(){ const el = $('#audiobar'); if(el) el.style.display = 'none'; }

// 浏览器对未交互过的页面会拒绝 play()（NotAllowedError）。
// 旧代码把这个错误静默吞掉了，于是「有文字、没声音」且毫无提示。
function playNext(){
  if(playing || !queue.length) return;
  playing = true;
  const url = queue.shift();
  curAudio = new Audio(url);
  curAudio.onended = () => { playing = false; playNext(); };
  curAudio.onerror = () => {
    diag('音频加载失败：' + url);
    playing = false; playNext();
  };
  const p = curAudio.play();
  if(p && p.catch) p.catch(err => {
    playing = false;
    const name = (err && err.name) || String(err);
    if(name === 'NotAllowedError'){
      queue.unshift(url);                        // 不丢，解锁后接着播
      showAudioBar('浏览器拦截了自动播放，所以这轮没有声音', true);
      diag('自动播放被拦截（NotAllowedError）');
    } else if(name !== 'AbortError'){
      diag('播放失败：' + name);
      playNext();
    }
  });
}

// 点「声音测试」：服务端合成 → 浏览器播放，一次把两半都验掉
async function testSound(){
  const btn = $('#aud');
  if(btn && btn.disabled) return;
  if(btn){ btn.disabled = true; btn.textContent = '合成中…'; }
  try{
    const r = await fetch('/api/say', {method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify({text:'声音测试，一二三。'})});
    const j = await r.json();
    if(!j.ok){
      showAudioBar('服务端合成失败：' + (j.message || '未知错误'), false);
      diag('合成失败');
      return;
    }
    if(j.local){
      // 本地播报模式：这句已经在 Jetson 的耳机里出了，浏览器不要重复播
      hideAudioBar();
      addTiming('声音', '耳机 ' + j.audio_s + 's');
      diag('已在 Jetson 耳机播放：' + j.voice + ' ' + j.audio_s + 's');
      return;
    }
    const a = new Audio(j.url);
    await a.play();                              // 在点击手势内，不会被拦
    hideAudioBar();
    addTiming('声音', '正常 ' + j.audio_s + 's');
    diag('声音测试通过：' + j.voice + ' ' + j.audio_s + 's');
    playing = false; playNext();                 // 把这轮欠下的音频补上
  }catch(e){
    const name = (e && e.name) || String(e);
    showAudioBar('浏览器拒绝播放：' + name + ' —— 检查标签页是否被静音、系统输出设备是否正确', true);
    diag('播放被拒绝：' + name);
  }finally{
    if(btn){ btn.disabled = false; btn.textContent = '声音测试'; }
  }
}

function addTiming(k, v){ $('#timings').insertAdjacentHTML('beforeend',
  `<span class="t">${k} <b>${v}</b></span>`); }

function elapsed(t0){ return ((performance.now()-t0)/1000).toFixed(2)+'s'; }

function diag(s){ const el = $('#diag'); if(el) el.textContent = s ? '· ' + s : ''; }

function setStatus(label, rec){
  $('#talklabel').textContent = label;
  $('#talk').classList.toggle('rec', !!rec);
}

// 唤醒、手动两种入口共用同一个事件渲染
function onEvent(ev){
  if(ev.type === 'state'){
    if(ev.state === 'listening'){ setStatus('聆听中…', true); }
    else if(ev.state === 'transcribing'){ setStatus('识别中…', false);
      if(ev.record) $('#you').innerHTML = '<span class="empty">识别中…（语音 '+ev.record.speech_s+'s）</span>'; }
    else if(ev.state === 'waiting'){ setStatus(es ? '待唤醒：说「'+wakeWord+'」' : '开始说话', false); }
    else if(ev.state === 'looking'){ setStatus('看屏幕…', false); lookPending(); }
  } else if(ev.type === 'asr'){
    $('#you').textContent = ev.text;
    if(ev.ms) addTiming('识别', ev.ms+'ms');
  } else if(ev.type === 'look'){
    showLook(ev.info, ev.ms);
  } else if(ev.type === 'delta'){
    botText += ev.text; $('#bot').textContent = botText;
  } else if(ev.type === 'audio'){
    if(ev.idx === 1) addTiming('首句合成', ev.tts_s + 's');
    // local=true 表示这句已经由服务端播到 Jetson 的耳机了，浏览器别再放一遍
    if(!ev.local){ queue.push(ev.url); playNext(); }
  } else if(ev.type === 'interrupt'){
    // 服务端听到你插话：立刻掐掉正在播的音频，剩下的不再播
    resetTurnUI();                      // 同时把 botText 清零，新回答从头拼
    setStatus('打断，请继续说…', true);
    diag('打断' + (ev.heard ? '「'+ev.heard.slice(0,12)+'」' : ''));
  } else if(ev.type === 'error'){
    $('#errors').insertAdjacentHTML('beforeend', '<div class="err">'+ev.message+'</div>');
  } else if(ev.type === 'done'){
    const t = ev.timings || {};
    if(t.record_s) addTiming('录音', t.record_s+'s');
    if(t.llm_first_ms) addTiming('首字', t.llm_first_ms+'ms');
    if(t.llm_s) addTiming('LLM 全文', t.llm_s+'s');
    if(t.look_s) addTiming('看屏幕', t.look_s+'s');
    addTiming('端到端', (t.total_s ?? 0)+'s');
    if(ev.user) addLog(ev.user, ev.reply || botText, t);
  }
}

function resetTurnUI(){
  botText = '';
  queue = []; playing = false;
  if(curAudio){ try{ curAudio.pause(); }catch(e){} }
  $('#you').innerHTML = '<span class="empty">聆听中…</span>';
  $('#bot').innerHTML = '<span class="empty">等待回应…</span>';
  $('#timings').innerHTML = ''; $('#errors').innerHTML = '';
}

let actx = null;
function beep(){  // 唤醒提示音，不依赖音频文件
  try{
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = 'sine'; o.frequency.value = 880;
    g.gain.setValueAtTime(0.0001, actx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.10, actx.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + 0.20);
    o.connect(g); g.connect(actx.destination);
    o.start(); o.stop(actx.currentTime + 0.22);
  }catch(e){}
}

let pollTimer = null, lastSeq = 0, pollBusy = false;

// 屏幕 OCR 结果：把识别到的文字摊开展示，耗时不藏着
function showLook(info, ms){
  const card = $('#lookcard');
  if(!card) return;
  card.style.display = '';
  const meta = $('#lookmeta'), body = $('#looktext');
  if(!info || !info.ok){
    meta.textContent = '取帧失败';
    body.innerHTML = '<span class="empty">'+((info && (info.message || info.reason)) || '未知错误')+'</span>';
    return;
  }
  const sec = ms ? (ms/1000).toFixed(1) : (info.total_s || 0).toFixed(1);
  meta.textContent = (info.lines||0)+' 行 · 置信 '+(info.score||0)+' · '+sec+'s'
                     + (info.cached ? ' · 缓存命中' : '')
                     + (info.truncated ? ' · 已截断' : '');
  body.textContent = info.text || '（没识别到文字）';
}

// 「看屏幕」按钮：只跑取帧 + OCR，不惊动 LLM
async function lookNow(){
  const btn = $('#look');
  if(btn.disabled) return;
  btn.disabled = true; btn.textContent = '看屏幕…';
  // 取帧 + OCR 要 6~8 秒，必须立刻给出可见反馈，否则看起来像卡死
  lookPending();
  try{
    const r = await fetch('/api/look', {method:'POST', headers:{'Content-Type':'application/json'},
                                        body:'{"force":true}'});
    const info = await r.json();
    showLook(info, info.total_s ? info.total_s*1000 : 0);
    if(info.ok) addTiming('看屏幕', (info.total_s||0)+'s');
  }catch(e){
    showLook({ok:false, message:'请求失败：'+e}, 0);
  }finally{
    btn.disabled = false; btn.textContent = '看屏幕';
  }
}

// 取帧期间的占位状态（语音触发和按钮触发共用）
function lookPending(){
  const card = $('#lookcard');
  if(!card) return;
  card.style.display = '';
  $('#lookmeta').textContent = '取帧 + OCR 中…';
  $('#looktext').innerHTML = '<span class="empty">正在看屏幕，约 6 秒</span>';
}

function handleLiveEvent(ev){
  if(ev.type === 'wake'){ beep(); resetTurnUI(); setStatus('聆听中…', true); return; }
  onEvent(ev);
}

// 用短轮询而不是 EventSource：实测 EventSource 在这个环境里连不上，
// 服务端 live 连接数一直是 0，唤醒全被当成「没人听」跳过了。
// 400ms 一次的轮询在任何浏览器/代理下都能工作，代价可以忽略。
async function startLive(){
  if(es) return;
  if(!wakeWord) await health();
  if(!wakeWord){ setStatus('服务端未启用唤醒词', false); return; }
  es = {};
  $('#live').classList.add('on');
  $('#live').textContent = '监听中（点此停止）';
  $('#talk').disabled = true;
  setStatus('连接中…', false);
  try{
    // after 给一个超大值：只取 last，用于对齐到当前，不重放历史事件
    const j = await (await fetch('/api/poll?after=999999999999', {cache:'no-store'})).json();
    lastSeq = j.last || 0;
    if(!j.loop) diag('服务端唤醒线程没起来');
  }catch(e){ setStatus('连不上服务，检查地址', false); es = null; return; }
  setStatus('待唤醒：说「'+wakeWord+'」', false);
  pollTimer = setInterval(pollOnce, 400);
  pollOnce();
}

async function pollOnce(){
  if(!es || pollBusy) return;
  pollBusy = true;
  try{
    const j = await (await fetch('/api/poll?after='+lastSeq, {cache:'no-store'})).json();
    if(!es) return;
    lastSeq = j.last ?? lastSeq;
    for(const ev of j.events) handleLiveEvent(ev);
    if(!j.events.length) diag('心跳 ' + new Date().toLocaleTimeString().slice(3));
  }catch(e){
    diag('轮询失败');
  }finally{ pollBusy = false; }
}

function stopLive(){
  diag('');
  if(pollTimer){ clearInterval(pollTimer); pollTimer = null; }
  es = null;
  $('#live').classList.remove('on');
  $('#live').textContent = '常驻监听';
  $('#talk').disabled = false;
  setStatus('开始说话', false);
}

async function runTurn(body, btnLabel){
  if(busy) return;
  setBusy(true);
  resetTurnUI();
  $('#you').innerHTML = '<span class="empty">录音中…</span>';
  const t0 = performance.now();

  try{
    const resp = await fetch('/api/turn', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    if(resp.status === 409){ $('#errors').innerHTML = '<div class="err">上一轮还没结束，稍等</div>'; return; }
    const reader = resp.body.getReader(); const dec = new TextDecoder();
    let buf = '';

    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let nl;
      while((nl = buf.indexOf('\n')) >= 0){
        const line = buf.slice(0, nl); buf = buf.slice(nl+1);
        if(!line.trim()) continue;
        let ev; try{ ev = JSON.parse(line); }catch(e){ continue; }
        onEvent(ev);
      }
    }
  }catch(e){
    $('#errors').innerHTML = '<div class="err">'+e+'</div>';
  }finally{
    setBusy(false);
    setStatus('开始说话', false);
  }
}

function addLog(you, bot, t){
  const el = $('#log');
  if(el.querySelector('.hint')) el.innerHTML = '';
  const meta = [t.record_s?('录 '+t.record_s+'s'):null, t.asr_s?('识 '+t.asr_s+'s'):null,
                t.llm_first_ms?('首字 '+t.llm_first_ms+'ms'):null, t.total_s?('总 '+t.total_s+'s'):null]
                .filter(Boolean).join(' · ');
  el.insertAdjacentHTML('afterbegin',
    `<div class="logitem"><div>你：${esc(you)}</div><div>答：${esc(bot)}</div><div class="m">${meta}</div></div>`);
}
function esc(s){ return (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

$('#talk').onclick = () => runTurn({});
$('#live').onclick = () => { if(es) stopLive(); else startLive(); };
$('#look').onclick = () => lookNow();
$('#aud').onclick = () => testSound();
$('#audiobar').onclick = () => testSound();   // 横幅本身就是「点这里启用声音」

// 播报出口：耳机（Jetson 本地播放）/ 浏览器（哪台开页面从哪台出）
let routeOn = null;
function paintRoute(on){
  routeOn = on;
  const b = $('#route');
  b.textContent = on ? '声音：耳机' : '声音：浏览器';
  b.classList.toggle('on', !!on);
}
async function loadRoute(){
  try{
    const j = await (await fetch('/api/health', {cache:'no-store'})).json();
    if(j.speaker) paintRoute(j.speaker.on);
  }catch(e){}
}
$('#route').onclick = async () => {
  const b = $('#route'); b.disabled = true;
  try{
    const j = await (await fetch('/api/route', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({on: !routeOn})})).json();
    paintRoute(j.on);
    diag('播报出口：' + (j.on ? 'Jetson 耳机' : '浏览器'));
  }catch(e){ diag('切换失败：' + e); }
  finally{ b.disabled = false; }
};
loadRoute();

// 任何一次点击都算用户交互；之前被拦下的音频顺手补播
document.addEventListener('click', () => {
  const bar = $('#audiobar');
  if(bar && bar.style.display === 'block' && queue.length){ playing = false; playNext(); }
});
$('#probe').onclick = async () => {
  $('#probe').disabled = true;
  setStatus('说「'+wakeWord+'」（录 3 秒）', true);
  try{
    const j = await (await fetch('/api/kws_check', {method:'POST'})).json();
    if(j.error){ $('#errors').insertAdjacentHTML('beforeend','<div class="err">'+j.error+'</div>'); diag(j.error); }
    else{
      $('#you').textContent = j.text || '(没听清)';
      diag('识别「'+(j.text||'空')+'」 '+j.sec+'s 峰值'+j.peak_db+'dB · '+j.wav);
    }
  }catch(e){ diag('失败 '+e); }
  finally{
    $('#probe').disabled = false;
    setStatus(es ? '待唤醒：说「'+wakeWord+'」' : '开始说话', false);
  }
};
$('#send').onclick = () => {
  const v = $('#tin').value.trim();
  if(!v) return;
  runTurn({text: v});
};
$('#tin').addEventListener('keydown', e => { if(e.key === 'Enter') $('#send').click(); });
$('#reset').onclick = async () => {
  await fetch('/api/reset', {method:'POST'});
  $('#log').innerHTML = '<span class="hint">还没有记录</span>';
  $('#you').innerHTML = '<span class="empty">尚未录音</span>';
  $('#bot').innerHTML = '<span class="empty">—</span>';
  $('#timings').innerHTML = '';
};
health(); setInterval(health, 15000);
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--device", default="plughw:CARD=H340,DEV=0")
    p.add_argument("--asr-dir", default=str(DEFAULT_ASR_DIR))
    p.add_argument("--asr-lang", default="auto",
                   help="auto 由模型判断语种；指定 ja/zh 可提升该语种的稳定性")
    p.add_argument("--asr-threads", type=int, default=8)
    p.add_argument("--tts", default=str(DEFAULT_TTS))
    p.add_argument("--tts-zh", default=str(DEFAULT_TTS_ZH))
    p.add_argument("--vad", default=str(DEFAULT_VAD))
    p.add_argument("--speaker", type=int, default=None)
    p.add_argument("--llm-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--llm-model", default="qwen")
    p.add_argument("--llm-key", default="",
                   help="LLM 的 Bearer token。Magnitude 的 /inference/v1 必填，vLLM 留空")
    p.add_argument("--llm-think", action="store_true",
                   help="打开模型的思考模式。默认关——Qwen3.5 不开这个开关会把 token 全烧在"
                        "推理上，正文恒为空")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--reply-lang", default="auto", choices=["auto", "ja", "zh"],
                   help="auto 跟随每句话的语种；ja/zh 强制固定")
    p.add_argument("--max-s", type=float, default=15.0)
    p.add_argument("--silence", type=float, default=0.60, help="判定说完所需的静音秒数")
    p.add_argument("--wake-word", default="小八你好",
                   help="唤醒词（中文，拼音建模）；空字符串或 --no-wake 关闭")
    p.add_argument("--wake-dir", default=str(DEFAULT_KWS))
    p.add_argument("--wake-threshold", type=float, default=0.18,
                   help="置信阈值，调低更容易唤醒也更易误触发")
    p.add_argument("--wake-score", type=float, default=1.5, help="关键词加分，调高更敏感")
    p.add_argument("--no-wake", action="store_true")
    p.add_argument("--no-barge", dest="barge", action="store_false",
                   help="关闭播报中打断（默认开，仅在常驻模式下生效）")
    p.add_argument("--barge-db", type=float, default=-42.0,
                   help="打断能量门限 dBFS。环境嘈杂就抬高（-38），太钝就降低（-46）")
    p.add_argument("--barge-margin", type=float, default=8.0,
                   help="比播报期本底高多少 dB 才算人声；外放时靠它防自激")
    p.add_argument("--vision-dev", default="/dev/video0",
                   help="采集卡设备；传空字符串关闭「看屏幕」")
    p.add_argument("--vision-side", type=int, default=1440,
                   help="OCR 输入长边像素。1440 实测 5.7s 且最准，960 以下小字会糊")
    p.add_argument("--vision-threads", type=int, default=vx.OCR_THREADS,
                   help="OCR 线程数。默认 8；12 核机上吃满 12 线程反而慢一倍（超订）")
    p.add_argument("--vision-trt", default="",
                   help="rec 的 TensorRT engine 路径。给了就把 OCR 的 rec 搬到 GPU"
                        "（实测 4.10s → 1.69s）。engine 跟 TRT 版本绑定，换 JetPack 要重建")
    p.add_argument("--vision-close", action="store_true",
                   help="每次取帧后释放设备（旧行为）。默认常驻：预热只付一次")
    p.add_argument("--vision-idle-release", type=float, default=0.0,
                   help="闲置多少秒后释放采集卡（0=永不释放）。要让别的程序用设备时调它")
    p.add_argument("--local-speak", action="store_true",
                   help="播报直接播到 Jetson 本机声卡（耳机），而不是让浏览器播")
    p.add_argument("--local-sink", default="",
                   help="本地播报的 PipeWire 目标节点名/id；空=用系统默认输出")
    p.add_argument("--local-gain", type=float, default=1.0,
                   help="本地播报音量 0~1")
    args = p.parse_args(argv)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    # 旧音频清掉，避免累积占盘
    now = time.time()
    for f in AUDIO_DIR.glob("*.wav"):
        try:
            if now - f.stat().st_mtime > 3600:
                f.unlink()
        except OSError:
            pass

    print("[1/4] 加载 ASR …", flush=True)
    asr = Asr(Path(args.asr_dir), args.asr_threads, args.asr_lang)
    print("[2/4] 加载 TTS …", flush=True)
    tts = Tts(Path(args.tts), Path(args.tts_zh), args.speaker)
    print(f"      音色 {'/'.join(tts.voices)}  加载 {tts.load_s:.2f}s", flush=True)
    print("[3/4] 录音设备 …", flush=True)
    if not Path(args.vad).exists():
        print(f"      !! 找不到 VAD 模型 {args.vad}", flush=True)
    rec = VoiceRecorder(args.device, args.vad, max_s=args.max_s, silence_s=args.silence)

    llm = None if args.no_llm else Llm(args.llm_base, args.llm_model,
                                       reply_lang=args.reply_lang,
                                       api_key=args.llm_key,
                                       think=args.llm_think)
    ok = bool(llm and llm.available())
    print(f"[4/4] LLM {'就绪 ' + args.llm_model if ok else '暂未就绪（每轮会重新探活）'}",
          flush=True)
    print(f"      回复语言 {args.reply_lang}", flush=True)

    kws = None
    hub = None
    if args.wake_word and not args.no_wake and Path(args.wake_dir).exists():
        print(f"[5/5] 唤醒词「{args.wake_word}」…", flush=True)
        try:
            kws = Kws(Path(args.wake_dir), args.wake_word, WORK / "wake-keywords.txt",
                      threshold=args.wake_threshold, score=args.wake_score)
            hub = AudioHub(args.device).start()
            time.sleep(1.2)  # 等采集线程吐第一批帧，否则订阅到空缓冲
            if hub.last_frame_at == 0.0:
                raise RuntimeError(f"采集没起来（{args.device} 被占用？）")
            print(f"      关键词 {kws.keywords_file.name}  "
                  f"阈值 {args.wake_threshold}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"      !! 唤醒未启用：{e}", flush=True)
            if hub is not None:
                hub.stop()
            kws = hub = None
    elif not args.no_wake:
        print(f"      !! 找不到唤醒模型 {args.wake_dir}，跳过", flush=True)

    # 视觉：懒加载 OCR 引擎（首次 look 才 import rapidocr 并建引擎）
    vision = None
    if args.vision_dev:
        vision = vx.Vision(args.vision_dev, max_side=args.vision_side,
                           keep_open=not args.vision_close,
                           idle_release=args.vision_idle_release,
                           ocr_threads=args.vision_threads,
                           trt_engine=args.vision_trt)
        if Path(args.vision_dev).exists():
            print(f"[6/6] 看屏幕 {args.vision_dev}  OCR 长边 {args.vision_side}px"
                  f"  线程 {args.vision_threads}"
                  f"  抓帧常驻={'否' if args.vision_close else '是'}"
                  f"  rec 后端={'TensorRT' if args.vision_trt else 'CPU'}", flush=True)
            # 先把句柄打开并预热好（MS2109 前 40 帧是黑的），
            # 这样第一次「看屏幕」不用再等 1.2 秒。失败也不影响启动。
            def _vision_warm():
                vision.prewarm()
                # TRT 的 engine 反序列化 + CUDA context + 首次 warmup 要好几秒，
                # 落到用户头上就是「第一次看屏幕特别慢」，后台先付掉
                if args.vision_trt and vision.warm_ocr():
                    note = vision.trt_note or "ok"
                    print(f"      OCR 引擎就绪  rec 后端={note}"
                          f"  {vision.trt_status() or ''}", flush=True)

            threading.Thread(target=_vision_warm, name="vision-prewarm",
                             daemon=True).start()
        else:
            print(f"      !! 采集卡 {args.vision_dev} 不存在，「看屏幕」会失败", flush=True)

    cfg = argparse.Namespace(recorder=rec, asr=asr, tts=tts, llm=llm,
                             wake=kws, hub=hub, barge=args.barge,
                             vad_model=args.vad, barge_db=args.barge_db,
                             barge_margin=args.barge_margin, vision=vision,
                             speaker=LocalSpeaker(sink=(args.local_sink or None),
                                                  gain=args.local_gain,
                                                  enabled=args.local_speak))
    app = build_app(cfg)
    print(f"\n打开 http://<jetson-ip>:{args.port}   设备 {args.device}\n", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
