#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""voice_server.py — a browser-driven voice conversation test bench.

Start the service with one command, open the page in a browser, and talk into the
H340 microphone plugged into the Jetson. The pipeline is:

    click → arecord (VAD auto-stop) → SenseVoice → vLLM → Piper → browser playback

Three design trade-offs, all deliberate:

1. **Recording lives on the server, not in the browser.** The microphone is
   plugged into the Jetson; if the browser did the recording, you would be
   testing your laptop's microphone, which says nothing about the real device.
   The browser only triggers and plays back.

2. **Recording stops on VAD, not at a fixed length.** A fixed 5-second recording
   feels like "I stopped talking but still have to wait", and that wait itself
   masks the real latency. An energy VAD decides when you are done: 0.75 s of
   silence ends the turn, with a 12 s hard cap.

3. **Synthesize per sentence, stream sentence by sentence.** Synthesizing the
   whole reply means waiting for the LLM to finish, so perceived latency equals
   the sum of everything. Here the LLM emits a complete sentence and we
   synthesize and push it right away; the browser plays as it receives, so the
   perceived first-sentence latency approaches "first token + first sentence".

Usage:
    ./asr-venv/bin/python voice_server.py --port 8095
    then open http://127.0.0.1:8095

    # works before the LLM is up too (ASR→TTS echo, to verify the pipeline)
    ./asr-venv/bin/python voice_server.py --port 8095 --no-llm
"""
# Note: `from __future__ import annotations` is deliberately avoided here.
# FastAPI resolves annotations in module globals, but Request is a local name imported
# inside build_app; stringified annotations turn `request: Request` into a query param (422).

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
import vision as vx  # noqa: E402  frame grab + OCR: "take a look at the screen"

import paths  # noqa: E402  directory conventions: MC_DATA overrides, see paths.py

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

# Kana / CJK ideographs, used to pick a TTS voice
KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
CJK = re.compile(r"[\u4E00-\u9FFF]")


# ------------------------------------------------------------------ recording
class RecordError(RuntimeError):
    pass


class VoiceRecorder:
    """Record and stop automatically via Silero VAD.

    Why not an energy threshold: the early version used an absolute "noise floor +
    headroom" threshold, and both failure modes showed up in practice.
      - With high gain: speech leaked into the noise floor, the threshold rose to
        -20 dB, and speech was judged silence (only 0.81 s captured)
      - In a noisy room: the noise itself sits above the threshold, "done talking"
        is never detected, and every turn hits the 12 s cap
    Silero VAD is a speech-probability model over 512-sample windows; it is
    insensitive to gain and stationary noise, and it tracks "how long a silence
    ends a segment" itself, so we do not have to tune two time constants.
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
        """Record one segment and stop on VAD.

        source=None: spawn our own arecord (button mode).
        source=queue: pull frames from the resident AudioHub (wake-word mode). ALSA
        devices can only be held by one process at a time, and the wake thread is
        already reading, so a second arecord cannot be opened here.
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
                # The detector emits a segment only after speech has "ended", which
                # is exactly the end-of-turn signal we want
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
            # Hit the hard cap or aborted early: flush whatever is still buffered
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


# ------------------------------------------------------------------ capture hub
class AudioHub:
    """One arecord process, many consumers.

    Why this exists: wake-word detection needs the microphone open all the time,
    and an ALSA device can be held by only one process at a time. When record()
    spawned its own arecord on every call, the wake thread and the recorder kept
    fighting over the device. Here a single thread owns the device and hands
    fixed-size frames to subscribers; on subscribe we first top up with the last
    0.3–0.6 s of frames, so when someone says "hey assistant, what's the weather
    today" in one breath, the words after the wake word are not lost.

    If arecord exits unexpectedly (device busy, resume from sleep) it is reopened
    automatically, and the errors counter is there for health to observe.
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
        """Seconds since the last frame arrived, used to tell whether capture is alive."""
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
                # Slow consumer: drop the oldest frame, never block the capture thread
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except queue.Empty:
                    pass

    def subscribe(self, pre_s: float = 0.3, pre_frames=None) -> queue.Queue:
        """Subscribe to audio frames.

        pre_frames: explicitly pass the "history frames" instead of taking them
        from the ring buffer. The barge-in handover needs this — the barge monitor
        has already consumed part of the speech, and the recorder must receive it
        verbatim, otherwise the first few words of that utterance are eaten (the
        ring buffer only holds 0.6 s, so they cannot be recovered).
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


# ------------------------------------------------------------------ wake word
class Kws:
    """Keyword wake word: sherpa-onnx zipformer-KWS, pinyin-modeled, CPU only.

    Measured RTF 0.03 (8 s of audio decoded in 0.26 s), about 3% of one core on
    the Jetson, so competing with ASR/TTS for cores is negligible. Keywords are
    generated at runtime, so changing the phrase needs no retraining.
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
        """Block until the wake word is heard. Returns the matched keyword, or None when stop is set."""
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
    """Resident listener: wait for the wake word → run one turn → back to listening."""

    def __init__(self, hub: AudioHub, kws: Kws, runner, history: list[dict],
                 push, turn_lock: threading.Lock, has_listener=None):
        super().__init__(name="wake-loop", daemon=True)
        self.hub = hub
        self.kws = kws
        self.runner = runner
        self.history = history
        self.push = push
        self.turn_lock = turn_lock
        # Do not run a turn when no page is listening: otherwise every wake after the
        # page is closed burns a pointless turn, synthesizes a pile of wavs, and holds
        # turn_lock while doing it
        self.has_listener = has_listener or (lambda: True)
        self._stop = threading.Event()
        self.turns = 0
        self.skipped = 0  # matched, but no page was listening

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, ev: dict) -> None:
        # An interrupted turn never enters history: it is incomplete
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
            # Audio after the wake word: pre fills only 0.25 s, enough to catch the
            # tail of the greeting without dragging the whole wake phrase into ASR
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
    """SenseVoice decorates its output with tags like <|ja|><|HAPPY|>; strip them."""
    s = re.sub(r"<\|[^|]*\|>", "", s or "")
    s = EMOJI.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def trim_by_vad(audio: np.ndarray, sr: int, hp: float = 150.0) -> tuple[np.ndarray, dict]:
    """Trim leading and trailing silence, keeping only the speech region.

    This is not optional for SenseVoice: feeding it a whole clip padded with long
    silences produces repetitions and garbled output (a measured 40 s file with two
    utterances came back as "…確認してくださいこ…確認してださい").
    The live recording path is already trimmed by the recorder's VAD; this is the
    same processing for wav playback.
    """
    x = vp.highpass(audio.astype(np.float64), hp, sr, 2)
    frames = vp.frame_db(x, sr)
    if frames.size == 0:
        return audio, {"trimmed": False}
    thr = vp.vad_threshold(vp.noise_floor(frames), 8.0, -52.0)
    segs = vp.segment_speech(frames, thr, min_speech_s=0.30, min_gap_s=0.45)
    if not segs:
        return audio, {"trimmed": False, "thr_db": round(thr, 1)}
    # Cut segment by segment, then splice: removes silence and long inter-segment pauses
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
    """For short polling: return events with seq > after, in the original order."""
    return [ev for (sq, ev) in buf if sq > after]


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.astype(np.int16).tobytes())


# ------------------------------------------------------------------ TTS
class Tts:
    """Multi-voice synthesis, picking the engine from each sentence's script.

    Why separate engines are required: a Piper Japanese voice reading Chinese, or a
    Chinese voice reading Japanese, goes badly off-key (kana read as pinyin, kanji
    read with another character's sound). The rule is reliable: hiragana/katakana
    → Japanese; no kana but CJK ideographs → Chinese. Mixed CJK/Japanese lines have
    been confirmed not to occur, so deciding per sentence is enough — no need to
    decide per character.
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
            self.cfgs["zh"] = None  # Chinese voice is single-speaker; passing speaker_id errors
        self.default = "ja" if "ja" in self.voices else next(iter(self.voices))
        self.load_s = time.time() - t0

    def pick(self, text: str, hint: str | None = None) -> str:
        """Pick a voice. hint is the voice used by the previous sentence (or this turn's reply language).

        Why hint: a sentence may contain only digits or Latin letters ("1.5", "OK",
        "2026"), with neither kana nor CJK ideographs. Such sentences used to fall
        back to default (Japanese), so "3.14" inside a Chinese reply was read in
        Japanese — users hear "why is the number in Japanese?".
        These sentences must follow the surrounding language, and hint is that context.
        """
        if KANA.search(text) and "ja" in self.voices:
            return "ja"
        if CJK.search(text) and "zh" in self.voices:
            return "zh"
        if hint in self.voices:
            return hint
        # With no context (e.g. an English first sentence) use Chinese: the Chinese
        # model reads English/digits without slipping into a Japanese accent
        return "zh" if "zh" in self.voices else self.default

    def _render(self, lang: str, text: str) -> tuple[bytes, int, int]:
        """Synthesize into memory. On empty phonemization wave raises at close; return 0 frames instead."""
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
        if n == 0:  # strip_for_tts should already block this; fall back to another voice and retry
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
    """OpenAI-compatible streaming client. Uses urllib instead of httpx, one less dependency."""

    def __init__(self, base: str, model: str, timeout: float = 30.0,
                 reply_lang: str = "auto", api_key: str = "",
                 think: bool = False):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.reply_lang = reply_lang
        self.api_key = api_key
        # Qwen3.5's chat template **defaults to the thinking branch** when
        # enable_thinking is undefined: all 160 tokens burn on reasoning, content
        # stays empty, which looks like "every turn waits forever and says nothing".
        # The caption side hit this once (README 4.18); same here, off by default.
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
        # Keep the full version publicly (screen rule included); tools/voice_llm_ab.py
        # uses it to build cases with OCR
        self.system = self.system_core + self.screen_rule

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _lang_directive(self, text: str) -> str:
        """Turn "which language to answer in" into a hard constraint.

        Relying on the system prompt's "follow the user's language" is not enough:
        measured with the first few turns in Chinese, the user switches to Japanese
        and the model still answers in Chinese — it follows the contextual style,
        not the current sentence. So each turn appends an explicit directive for the
        language of this sentence.

        **This directive must be attached to the end of the last user message, not
        to the end of system** (measured 2026-11-21):
            reply-lang=zh, two scenarios x 3 runs, judged by output language
            appended to system: 7B 3/6 Chinese, 35B 0/6 Chinese (all Japanese)
            appended to user:   7B 6/6 Chinese, 35B 6/6 Chinese
        system is Japanese throughout; a directive at its start or end is overridden
        by the surrounding context. At the end of user it sits right next to the
        generation point, and both models obey. 35B is especially clear — it is
        "too good at context": user speaks Japanese, screen is Japanese, and it
        ignores a distant Chinese directive.
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
        """Liveness probe with a 5-second cache.

        Not decided once at startup: vLLM takes about 90 s to load, so if this
        service comes up first, a one-shot decision would permanently downgrade the
        LLM to None and never recover even after vLLM is ready.
        """
        now = time.time()
        if self._probe is not None and now - self._probed_at < ttl:
            return self._probe
        try:
            # Magnitude's /inference/v1 requires Bearer; vLLM does not check it; harmless to send
            req = urllib.request.Request(self.base + "/models",
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=3) as r:
                self._probe = r.status == 200
        except Exception:  # noqa: BLE001
            self._probe = False
        self._probed_at = now
        return self._probe

    def stream(self, text: str, history: list[dict]):
        # system is picked per turn depending on whether we actually looked at the
        # screen, rather than keeping one resident prompt.
        # Why: as soon as the screen-text marker appears in the system prompt the model
        # quotes it, and answers "the screen text does not show…" even when we never
        # looked, which sounds like an excuse (35B more than 7B; a measured negative
        # directive saying "do not mention the screen when there is no screen text"
        # did not help either).
        # Simply not giving it that word makes the problem disappear.
        sysp = self.system if "[屏幕文字]" in text else self.system_core
        # Attach the language directive twice: end of system + end of user.
        # With user only, when "system is all Japanese + user speaks Japanese" 35B still
        # answered in Japanese (3/3 measured); system needs a line too, to override the
        # overall context. Attaching in both places is the most robust.
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


# Characters that are actually pronounceable: kana / CJK ideographs / Latin letters / digits.
# A fragment made only of punctuation ("。" "……" "「」") gives Piper 0 frames of audio,
# and the wave module then raises "# channels not specified" — the signature of empty
# phonemization, not an audio-device problem.
SPEAKABLE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF0-9A-Za-z]")


def has_speech(s: str) -> bool:
    return bool(SPEAKABLE.search(s))


def strip_for_tts(s: str) -> str:
    s = MD_NOISE.sub("", s)
    s = re.sub(r"[（(]\s*[)）]", "", s)
    s = s.strip()
    return s if has_speech(s) else ""


# ------------------------------------------------------------------ barge-in
class SpeakState:
    """Playback clock: estimates how long the front end will still be playing.

    Why this is needed: synthesis is an order of magnitude faster than playback
    (RTF 0.1), so when `TurnRunner.run()` returns, the earlier chunks are still
    queued in the browser. If barge-in were judged by "generation finished", the
    user would count as "already done talking" while the assistant is still
    speaking, and there would be no barge-in window at all.

    So we keep a virtual playback timeline: each audio chunk pushes the end time
    forward by audio_s seconds, and it naturally returns to zero when the queue drains.
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
            # Append after the previous chunk if it is still playing, else start from now
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
    """Play synthesized wav files straight to the Jetson's own sound card (headphones).

    Why this is needed: playback goes to the browser by default, and a browser plays
    "on whichever machine has the page open". Headphones plugged into the Jetson are
    silent precisely because the page is open on another computer — microphone on the
    Jetson, speaker on the laptop, split across two ends.

    This uses pw-play rather than aplay: the microphone is held exclusively by AudioHub
    through ALSA, and playback through PipeWire does not fight it for the device
    (playback and capture on a USB sound card are two independent streams).
    """

    def __init__(self, sink: str | None = None, gain: float = 1.0,
                 enabled: bool = False):
        self.sink = sink or None       # None = PipeWire default output
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
        except queue.Full:          # Full queue means playback is badly behind; dropping the new one beats piling up
            self.errors += 1
            self.last_error = "播放队列已满"
            return False

    def cancel(self) -> None:
        """Barge-in: clear the pending queue and cut off the sentence currently playing."""
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
                    pass                                # terminated (barge-in), expected
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
    """Is the "speech" detected during playback just speaker bleed?

    On open speakers the microphone hears the assistant itself, and without
    excluding that you get a feedback loop: playback is barged in on immediately,
    and the restarted playback is barged in on again by its own voice.

    The criterion is **the longest common substring as a share of the recognized
    text**, not whole-sentence similarity. The bleed window is only a second or two,
    so recognition yields three to five characters; comparing that against a
    fourteen-character sentence with SequenceMatcher.ratio gives 0.35 and never
    reaches the threshold. But the other way round, all 3/3 characters of that
    fragment fall inside the playback text as one contiguous run — deeply suspicious.
    """
    h = _ECHO_STRIP.sub("", heard or "")
    if len(h) < 2:
        # Too little text for any statistic: always treat it as human speech. Better to
        # barge in once too often than to miss a short command like "stop" or "hold on"
        return False
    cands = [spoken] if isinstance(spoken, str) else list(spoken or [])
    for c in cands:
        s = _ECHO_STRIP.sub("", c or "")
        if len(s) < 2:
            continue
        if h in s:
            return True          # full match, the clearest case
        m = difflib.SequenceMatcher(None, h, s).find_longest_match(0, len(h), 0, len(s))
        if m.size >= 3 and m.size >= len(h) * thresh:
            return True
    return False


class BargeMonitor(threading.Thread):
    """Barge-in during playback: when the user cuts in mid-sentence, stop and listen.

    Three gates, and any failed gate means no barge-in:
      1) Energy: above "playback-period noise floor + margin". The playback-period
         floor includes speaker bleed, so on open speakers the threshold rises
         automatically, preventing a feedback loop.
      2) Speech probability: Silero VAD judges it human voice, filtering transient
         noise such as keyboards, chairs, and fans.
      3) Echo exclusion: send this short clip to ASR; text that heavily overlaps the
         sentence being played is bleed. Gate 3 costs about 60 ms, so it runs only
         after energy and VAD both pass — normally it costs nothing.

    After a barge-in the device is not reopened: the frames already read are handed
    to the recorder verbatim and reading continues on the same stream, so the start
    of the user's utterance is not lost.
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

        # Last 1.2 s of frames: on barge-in the whole stretch goes to ASR/recording,
        # so the start is not cut off
        self.frames: deque = deque(maxlen=int(1.2 / FRAME_S))
        self._stop = threading.Event()
        self.hits = 0
        self.echo_drops = 0
        self.quiet_floor = -60.0     # non-playback floor, decides the threshold
        self.echo_floor = -60.0      # playback-period level (incl. bleed), diagnostics only
        self.last_db = -99.0
        self._cooldown = 0.0         # silence after a bleed verdict, avoids re-judging every 0.3s
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
        """Barge-in threshold to use right now (dBFS).

        Follows only the "silent-period floor", and **not** the playback-period
        level. Reason: on open speakers the playback-period level is speaker bleed,
        and raising the threshold with it would stop a real person from breaking in;
        bleed and a real voice are comparable in loudness anyway, so level alone
        cannot separate them. Bleed is handled by ASR text comparison — that is the
        dimension where they can be told apart.
        """
        return min(max(self.min_db, self.quiet_floor + self.margin), self.max_db)

    def run(self) -> None:
        q = self.hub.subscribe(pre_s=0.0)
        vad = self._make_vad()
        holding = 0.0        # seconds continuously judged as voice
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
                # Not playing: only update the floor, skip VAD (near-zero cost)
                self.quiet_floor = min(db, self.quiet_floor + 0.05)
                self.quiet_floor = max(self.quiet_floor, -80.0)
                self.echo_floor = min(db, self.echo_floor + 0.05)
                holding = 0.0
                voiced.clear()
                self._cooldown = 0.0
                self._in_play = False
                continue

            if not self._in_play:
                # Just entered playback: the VAD received no frames outside playback
                # and carries residue from the previous round; reusing it would count
                # "the tail of the last sentence" toward this round's holding
                vad = self._make_vad()
                self._in_play = True
                holding = 0.0
                voiced.clear()

            vad.accept_waveform(frame.astype(np.float32) / vp.FULL_SCALE)
            is_voiced = vad.is_speech_detected()
            hot = db > self.threshold()

            if self._cooldown > 0:
                # Just judged as bleed: stop judging for a while, but keep collecting
                # frames (the decision needs the full window)
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
                vad = self._make_vad()   # reset state so the same speech is not judged twice
                if is_echo:
                    self._cooldown = 1.2

        self.hub.unsubscribe(q)

    def _on_barge(self) -> bool:
        """Judge and handle one suspected barge-in. Returns True if it was judged bleed (cooldown needed)."""
        # Use the last 0.9 s rather than the 0.3 s voiced buffer: on such a short clip
        # ASR emits only two characters, and comparing that with the playback text is
        # meaningless. A longer window makes the verdict defensible.
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
        # Local playback must be cut off too, or the headphones keep reading to themselves
        spk = getattr(self.runner, "speaker", None)
        if spk is not None:
            spk.cancel()
        self.push({"type": "interrupt", "heard": heard})
        self._start_turn()
        return False

    def _start_turn(self) -> None:
        # Carry the 1.2 s the barge monitor has already consumed; the start of the
        # user's utterance must not be lost
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
        # An interrupted turn never enters history: it is incomplete, and keeping it
        # would pollute the following context
        if ev["type"] == "done" and not ev.get("interrupted"):
            if ev.get("user") and ev.get("reply"):
                self.history.append({"role": "user", "content": ev["user"]})
                self.history.append({"role": "assistant", "content": ev["reply"]})
        self.push(ev)


# ------------------------------------------------------------------ turn orchestration
class TurnRunner:
    def __init__(self, recorder, asr, tts, llm, audio_dir: Path, hub=None,
                 vision=None, speaker=None):
        self.recorder = recorder
        self.asr = asr
        self.tts = tts
        self.llm = llm
        self.audio_dir = audio_dir
        self.hub = hub  # AudioHub; with it we must not spawn our own arecord, or we fight over the device
        self.vision = vision  # Vision; only with it can we "take a look at the screen"
        # Voice fixed for this turn; None means nothing spoken yet, leave it to pick()
        self.speak_lang: str | None = None
        # Playback clock: barge-in detection uses it to know "still speaking"
        self.speak = SpeakState()
        # Local playback (to the Jetson's own headphones); None = browser playback only
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
            # 1) Get the audio
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
                    # The tail of the wake word itself gets recorded too; strip it before the LLM
                    user_text = user_text[len(strip_prefix):].lstrip("，,。.、 　")
                timings["asr_s"] = round(time.time() - t0, 2)
                timings["asr_core_s"] = round(asr_cost, 2)
                if not user_text:
                    emit({"type": "error", "message": "识别结果为空，可能没说话或声音太小"})
                    emit({"type": "done", "timings": timings})
                    return
                emit({"type": "asr", "text": user_text, "ms": int(timings["asr_s"] * 1000)})

            # 2) "Take a look at the screen": manual button, or intent like "what's on screen"
            if (look or vx.is_look_intent(user_text)) and self.vision is not None:
                emit({"type": "state", "state": "looking"})
                # Say a filler first. Frame grab + OCR takes about 8 s measured; the user
                # cannot be left waiting in silence
                self._speak(emit, "我看一下。", 0, turn_id, timings)
                t_look = time.time()
                info = self.vision.look(force=True)
                timings["look_s"] = round(time.time() - t_look, 2)
                emit({"type": "look", "info": info, "ms": int(timings["look_s"] * 1000)})
                ctx = vx.format_context(info)
                if ctx:
                    # Screen text goes before the question: the model sees "what is there"
                    # first, then "what is asked"
                    user_text = f"{ctx}\n\n{user_text}"
                else:
                    # If we cannot see it we must say so, or the model will invent something
                    emit({"type": "error",
                          "message": f"看不到屏幕：{info.get('message') or info.get('reason')}"})

            # 3) LLM streaming → per-sentence synthesis
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
                        # User barged in: stop generating at once, do not finish the rest
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

            # After a barge-in do not flush the remaining tail: the user no longer wants it
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
        if not text:  # pure punctuation/whitespace: synthesis yields 0 frames, skip it
            return
        if self.speak.cancelled:  # already barged in, stop synthesizing
            return
        out = self.audio_dir / f"{turn_id}_{idx:02d}.wav"
        try:
            meta = self.tts.synth_to(text, out, hint=self.speak_lang)
        except Exception as e:  # noqa: BLE001
            emit({"type": "error", "message": f"合成失败：{e}"})
            return
        # Remember this sentence's voice so the next digits/English fragment inherits it
        self.speak_lang = meta["voice"]
        if "tts_first_ms" not in timings:
            timings["tts_first_ms"] = int(meta["tts_s"] * 1000)
        # Local playback switch: when on, play to the Jetson's headphones directly and
        # the browser only shows text
        local = bool(self.speaker and self.speaker.enabled)
        if local:
            self.speaker.play(out)
        emit({"type": "audio", "url": f"/audio/{out.name}", "idx": idx,
              "text": text, "local": local, **meta})
        # Advance the playback clock: barge-in detection reads "still speaking" off it
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
    live: set = set()          # (loop, asyncio.Queue), one per /api/live connection
    turn_lock = threading.Lock()  # wake turns and manual turns are exclusive (across threads)
    wake_ref: dict = {}
    barge_ref: dict = {}
    # Event ring buffer: short polling reads increments from here. Long-lived
    # connections (/api/live) may fail to establish behind browsers/proxies — that is
    # exactly what happened in measurement, the page never connected and every wake
    # was judged "nobody listening".
    EVENTS: deque = deque(maxlen=600)
    seq_counter = itertools.count(1)
    poll_state = {"at": 0.0, "n": 0}
    visits = {"n": 0, "at": 0.0}  # bumped on every health call from the page

    def push(ev: dict) -> None:
        """Record the event and push it to long-lived subscribers."""
        seq = next(seq_counter)
        EVENTS.append((seq, ev))
        for loop, q in list(live):
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:  # event loop already closed
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
        """Manually look at the screen once: frame grab + OCR only, no LLM call."""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        vis = getattr(cfg, "vision", None)
        if vis is None:
            raise HTTPException(503, "vision 未启用（启动时加了 --no-vision？）")
        force = bool(payload.get("force"))
        # Frame grab + OCR takes about 8 s, so it must go to a thread pool; otherwise the
        # whole event loop (polling included) blocks
        info = await asyncio.get_running_loop().run_in_executor(
            None, lambda: vis.look(force=force))
        push({"type": "look", "info": info})
        return info

    @app.post("/api/say")
    async def say(request: Request):
        """Synthesize one sentence only, no LLM call. Splits "no sound" into server / browser halves."""
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
        # With local playback on, the test also goes to the headphones, so "is the test
        # audible" and "is the real thing audible" follow the same path
        spk = getattr(cfg, "speaker", None)
        local = bool(spk and spk.enabled)
        if local:
            spk.play(out, gain=float(payload.get("gain") or spk.gain))
        return {"ok": True, "url": f"/audio/{out.name}", "local": local, **info}

    @app.post("/api/route")
    async def route(request: Request):
        """Switch the playback output: local=true plays to the Jetson's headphones, false goes to the browser."""
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
                spk.cancel()          # on off, cut what is still queued so it does not speak after the switch
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
                with turn_lock:  # exclusive with wake turns: only one turn at a time
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
        """Short polling: return every event with seq > after.

        Dumber than EventSource, but it has no long-lived connection for a proxy or
        browser to cut. One call per 400 ms is negligible cost. When after exceeds the
        current max seq it returns empty, which the front end uses to "align to now"
        without replaying old events.
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
        """Resident event stream: wake → transcribe → answer → audio, never disconnecting.

        Not POST: the browser side uses EventSource and reconnects automatically on
        drop; after 15 s without events it sends a heartbeat, so intermediate devices
        do not reap the idle connection.
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
        """Record 3 s, save it as wav, and run ASR once.

        Separates "the wake word was mispronounced / too quiet" from "the KWS threshold
        is wrong": if the ASR text matches, the sound itself is fine and the threshold
        is the problem; if the text does not match either, it is pronunciation or capture.
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

        # Barge-in is only possible in resident mode: in manual mode record() holds the
        # capture device exclusively, so it cannot be listened to in parallel
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

// Browsers refuse play() on pages the user has not interacted with (NotAllowedError).
// The old code swallowed this error silently, so you got "text but no sound" with no hint.
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
      queue.unshift(url);                        // keep it; play after unlock
      showAudioBar('浏览器拦截了自动播放，所以这轮没有声音', true);
      diag('自动播放被拦截（NotAllowedError）');
    } else if(name !== 'AbortError'){
      diag('播放失败：' + name);
      playNext();
    }
  });
}

// Click "sound test": server synthesis → browser playback, verifying both halves at once
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
      // local playback mode: this line already came out of the Jetson headphones, do not replay it here
      hideAudioBar();
      addTiming('声音', '耳机 ' + j.audio_s + 's');
      diag('已在 Jetson 耳机播放：' + j.voice + ' ' + j.audio_s + 's');
      return;
    }
    const a = new Audio(j.url);
    await a.play();                              // inside the click gesture, so it is not blocked
    hideAudioBar();
    addTiming('声音', '正常 ' + j.audio_s + 's');
    diag('声音测试通过：' + j.voice + ' ' + j.audio_s + 's');
    playing = false; playNext();                 // catch up on the audio this turn owed
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

// Wake and manual entries share the same event rendering
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
    // local=true means the server already played this to the Jetson headphones; do not play it again here
    if(!ev.local){ queue.push(ev.url); playNext(); }
  } else if(ev.type === 'interrupt'){
    // The server heard a barge-in: cut the audio now playing, do not play the rest
    resetTurnUI();                      // also clears botText so the new answer starts from scratch
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
function beep(){  // wake chime, no audio file needed
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

// Screen OCR result: lay out the recognized text plainly and do not hide the timing
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

// "Look at screen" button: runs frame grab + OCR only, does not involve the LLM
async function lookNow(){
  const btn = $('#look');
  if(btn.disabled) return;
  btn.disabled = true; btn.textContent = '看屏幕…';
  // Frame grab + OCR takes 6~8 s; give visible feedback at once or it looks hung
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

// Placeholder state while grabbing frames (shared by voice and button triggers)
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

// Short polling instead of EventSource: measured EventSource cannot connect in this
// environment, the server's live connection count stayed at 0, and every wake was
// skipped as "nobody listening". 400 ms polling works behind any browser/proxy, at
// a negligible cost.
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
    // after is given a huge value: take only last, to align to now without replaying old events
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
$('#audiobar').onclick = () => testSound();   // the banner itself means "click here to enable sound"

// Playback output: headphones (played locally on the Jetson) / browser (whichever machine has the page)
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

// Any click counts as user interaction; opportunistically resume audio that was blocked earlier
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
    # Clear old audio so it does not accumulate on disk
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
            time.sleep(1.2)  # let the capture thread emit its first frames, or we subscribe to an empty buffer
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

    # Vision: lazy-load the OCR engine (the first look imports rapidocr and builds the engine)
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
            # Open the handle and warm it up front (the first 40 frames from an MS2109
            # are black), so the first "look at screen" needs no extra 1.2 s. Failure
            # does not block startup either.
            def _vision_warm():
                vision.prewarm()
                # TRT engine deserialization + CUDA context + first warmup take several
                # seconds; on the user that shows up as "the first look at the screen is
                # very slow", so pay it up front in the background
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
