#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""voice_probe.py — diagnostics for USB mic capture + high-pass + energy VAD + Japanese ASR.

Design goal: unit tests must run on a machine with no mic and no faster-whisper.
All pure logic (filtering, framing, noise estimation, segmentation) is dependency-free numpy,
while recording and transcription live in main and are imported lazily.

Why a high-pass is needed:
    Measured on this machine: the H340 headset mic has a noise floor of -30.6 dBFS, its strongest
    spectral peaks at 60Hz + 120Hz harmonics, and 0-80Hz holds 38.4% of the energy — that is
    mains/ground-loop hum, not room noise. With a 150Hz high-pass the floor falls to -51.7 dBFS
    and SNR rises from 9.8dB to above 18dB. Speech sits at 300Hz-3kHz, so cutting below 150Hz is lossless for ASR.

Usage:
    # process an existing wav (no mic needed)
    python voice_probe.py --wav a.wav --no-asr

    # record 8 seconds from the mic and transcribe
    python voice_probe.py --seconds 8

    # record 30 seconds continuously, segmenting as it goes
    python voice_probe.py --seconds 30 --asr
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

# ---------------------------------------------------------------- Constants
DEFAULT_DEVICE = "plughw:CARD=H340,DEV=0"
DEFAULT_RATE = 16000
FRAME_MS = 20
HOP_MS = 10
FULL_SCALE = 32768.0


# ---------------------------------------------------------------- Filtering
def biquad_highpass(fc: float, sr: int, q: float = 0.70710678) -> tuple[np.ndarray, np.ndarray]:
    """RBJ cookbook 2nd-order high-pass coefficients (already normalised by a0). Pure math, no scipy."""
    if not 0 < fc < sr / 2:
        raise ValueError(f"截止频率必须落在 (0, {sr / 2}) 内，收到 {fc}")
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)
    b = np.array([(1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0])
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    return b / a[0], a / a[0]


def apply_biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Transposed direct form II difference equation. Sample counts are small (hundreds of thousands), a plain loop is fast enough."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for i in range(x.shape[0]):
        xi = float(x[i])
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def highpass(x: np.ndarray, fc: float, sr: int, stages: int = 2) -> np.ndarray:
    """Cascaded multi-stage high-pass. Two stages (4th order) give about -31dB at 60Hz, enough to suppress mains hum."""
    y = np.asarray(x, dtype=np.float64)
    for _ in range(max(1, stages)):
        b, a = biquad_highpass(fc, sr)
        y = apply_biquad(y, b, a)
    return y


# ---------------------------------------------------------------- Framing and level
def frame_db(x: np.ndarray, sr: int, frame_ms: int = FRAME_MS, hop_ms: int = HOP_MS) -> np.ndarray:
    """Per-frame RMS in dBFS. Returns (frame count,)."""
    fl = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if x.shape[0] < fl:
        return np.empty(0)
    n = 1 + (x.shape[0] - fl) // hop
    idx = np.arange(fl)[None, :] + hop * np.arange(n)[:, None]
    frames = np.asarray(x, dtype=np.float64)[idx]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rms, 1e-12) / FULL_SCALE)


def db_of(x: np.ndarray) -> float:
    if x.size == 0:
        return -999.0
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return 20.0 * math.log10(rms / FULL_SCALE) if rms > 0 else -999.0


def noise_floor(frames_db: np.ndarray, pct: float = 10.0) -> float:
    """Use a low percentile as the noise floor. The mean gets pulled up by speech; a percentile does not."""
    if frames_db.size == 0:
        return -999.0
    return float(np.percentile(frames_db, pct))


def vad_threshold(floor_db: float, margin_db: float = 6.0, hard_min: float = -60.0) -> float:
    """Threshold = noise floor + margin, but never below hard_min (prevents too low a threshold in total silence)."""
    return max(floor_db + margin_db, hard_min)


# ---------------------------------------------------------------- Speech segmentation
@dataclass
class Segment:
    start: float
    end: float
    peak_db: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment_speech(
    frames_db: np.ndarray,
    threshold_db: float,
    hop_ms: int = HOP_MS,
    min_speech_s: float = 0.30,
    min_gap_s: float = 0.40,
    floor_db: float | None = None,
    min_peak_margin_db: float = 18.0,
) -> list[Segment]:
    """Speech segmentation with hysteresis.

    Three criteria do different jobs; drop any one and it goes wrong:
      min_gap_s        — silence allowed inside a segment (pauses between words); shorter does not split
      min_speech_s     — lower bound on the net speech length of a segment; shorter is discarded as noise
      min_peak_margin  — segment peak must clear the floor by enough, blocking "small transients near the threshold"

    The third gate came out of measurement: after the capture gain went from full scale to 0dB the floor
    dropped 8dB, and a 0.41s false trigger with a peak of only -41.5dB appeared near the previously
    unreachable threshold. A pure energy threshold drifts with gain, so an extra "peak margin over the floor" gate is needed.
    """
    if frames_db.size == 0:
        return []
    hop = hop_ms / 1000.0
    voiced = frames_db >= threshold_db

    segs: list[Segment] = []
    start_idx: int | None = None
    last_voiced_idx: int | None = None
    gap_frames = max(1, int(round(min_gap_s / hop)))

    def close(i0: int, i1: int) -> None:
        seg = _make_seg(frames_db, i0, i1, hop, min_speech_s, threshold_db)
        if seg is not None and floor_db is not None:
            if seg.peak_db - floor_db < min_peak_margin_db:
                seg = None
        if seg is not None:
            segs.append(seg)

    for i, v in enumerate(voiced):
        if v:
            if start_idx is None:
                start_idx = i
            elif last_voiced_idx is not None and i - last_voiced_idx > gap_frames:
                # gap over limit: close the previous segment and start a new one
                close(start_idx, last_voiced_idx)
                start_idx = i
            last_voiced_idx = i
        elif start_idx is not None and last_voiced_idx is not None:
            if i - last_voiced_idx > gap_frames:
                close(start_idx, last_voiced_idx)
                start_idx = last_voiced_idx = None

    if start_idx is not None and last_voiced_idx is not None:
        close(start_idx, last_voiced_idx)
    return segs


def _make_seg(frames_db: np.ndarray, i0: int, i1: int, hop: float,
              min_speech_s: float, threshold_db: float) -> Segment | None:
    """Turn a frame range back into time, and derive net speech length from frames above the threshold, for filtering."""
    net = float(np.count_nonzero(frames_db[i0:i1 + 1] >= threshold_db)) * hop
    if net < min_speech_s:
        return None
    peak = float(np.max(frames_db[i0:i1 + 1]))
    return Segment(start=round(i0 * hop, 3), end=round((i1 + 1) * hop, 3), peak_db=round(peak, 1))


# ---------------------------------------------------------------- Audio IO
def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read 16-bit PCM wav; returns a float64 mono array and the sample rate."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"只支持 16bit PCM，收到 {w.getsampwidth() * 8}bit")
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64)
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    return raw, sr


def record(device: str, seconds: float, sr: int = DEFAULT_RATE) -> np.ndarray:
    """Capture with arecord. No sounddevice/portaudio — one dependency fewer, one trap fewer on Jetson."""
    if shutil.which("arecord") is None:
        raise RuntimeError("找不到 arecord（apt install alsa-utils）")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(sr),
               "-c", "1", "-d", str(int(round(seconds))), str(tmp)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=seconds + 15)
        data, _ = read_wav(tmp)
        return data
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- Transcription
def transcribe_segments(segments: list[Segment], audio: np.ndarray, sr: int,
                        model_name: str = "large-v3-turbo", language: str = "ja",
                        model_dir: str = str(paths.ASR_DIR),
                        cpu_threads: int = 0) -> list[dict]:
    """Transcribe segment by segment. The model loads lazily — this module imports without faster-whisper.

    With language="auto" the model decides for itself. Forcing a language is double-edged:
    more accurate for a known language, but when the content does not match it, Whisper degrades into
    hallucination (the classic output being 「ご視聴ありがとうございました」).

    cpu_threads=0 hands over to the ctranslate2 default (measured at only 4 threads,
    while Orin has 12 cores; maxing it out gives near-linear speedup).
    """
    from faster_whisper import WhisperModel

    t0 = time.time()
    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         download_root=model_dir,
                         cpu_threads=cpu_threads if cpu_threads > 0 else 0)
    load_s = time.time() - t0
    lang_arg = None if language in ("auto", "", None) else language

    out: list[dict] = []
    for seg in segments:
        a, b = int(seg.start * sr), int(seg.end * sr)
        chunk = (audio[a:b] / FULL_SCALE).astype(np.float32)
        if chunk.size < sr // 10:
            continue
        t1 = time.time()
        segs, info = model.transcribe(
            chunk, language=lang_arg, beam_size=1, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0,
        )
        segs = list(segs)
        text = "".join(s.text for s in segs).strip()
        # Hallucination cue: on non-speech Whisper emits fixed phrasing (「ご視聴ありがとうございました」),
        # and such results share high no_speech_prob with low avg_logprob — a more reliable gate than a blocklist.
        no_speech = max((getattr(s, "no_speech_prob", 0.0) for s in segs), default=0.0)
        logprob = (sum(getattr(s, "avg_logprob", 0.0) for s in segs) / len(segs)) if segs else 0.0
        suspect = no_speech > 0.6 and logprob < -1.0
        cost = time.time() - t1
        out.append({
            "start": seg.start, "end": seg.end, "dur": round(seg.duration, 2),
            "peak_db": seg.peak_db, "text": text,
            "asr_s": round(cost, 2),
            "rtf": round(cost / max(seg.duration, 1e-6), 2),
            "model_load_s": round(load_s, 2),
            "lang": getattr(info, "language", "?"),
            "lang_prob": round(getattr(info, "language_probability", 0.0), 3),
            "no_speech_prob": round(no_speech, 3),
            "avg_logprob": round(logprob, 3),
            "suspect_hallucination": suspect,
        })
    return out


# ---------------------------------------------------------------- Report
def report(audio: np.ndarray, sr: int, raw_db: float, filt_db: float,
           floor_db: float, thr: float, segments: list[Segment], results: list[dict]) -> str:
    lines = []
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clip = int(np.count_nonzero(np.abs(audio) >= 32700))
    dur = audio.shape[0] / sr
    lines.append(f"音频      {dur:.2f}s @ {sr}Hz   峰值 {peak:.0f}/32768   削顶 {clip} 样本")
    lines.append(f"本底      {floor_db:.1f} dBFS   门限 {thr:.1f} dBFS")
    lines.append(f"电平      高通前 {raw_db:.1f} dBFS → 高通后 {filt_db:.1f} dBFS"
                 f"（信噪比 {filt_db - floor_db:.1f} dB）")
    lines.append(f"语音段    {len(segments)} 段")
    for i, s in enumerate(segments, 1):
        r = next((x for x in results if abs(x["start"] - s.start) < 1e-6), None)
        head = (f"  [{i}] {s.start:6.2f}-{s.end:6.2f}s ({s.duration:.2f}s, "
                f"峰值 {s.peak_db}dB / 余量 {s.peak_db - floor_db:.1f}dB)")
        if r:
            flag = "  ⚠ 疑似幻觉" if r.get("suspect_hallucination") else ""
            lines.append(head + f"  RTF {r['rtf']} ({r['asr_s']}s)  语言 {r.get('lang')}"
                               f"({r.get('lang_prob')})  logprob {r.get('avg_logprob')}"
                               f"  no_speech {r.get('no_speech_prob')}{flag}")
            lines.append(f"      {r['text'] or '(空)'}")
        else:
            lines.append(head)
    if clip > 0:
        lines.append(f"提示      有 {clip} 个削顶样本，建议把采集增益从满刻度下调")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="麦克风 → 高通 → VAD → 日文 ASR 诊断")
    p.add_argument("--wav", help="处理已有 wav，而不是现场录音")
    p.add_argument("--seconds", type=float, default=8.0, help="录音秒数（默认 8）")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--hp", type=float, default=150.0, help="高通截止频率 Hz（默认 150，0 关闭）")
    p.add_argument("--stages", type=int, default=2, help="高通级联节数（默认 2 = 4 阶）")
    p.add_argument("--margin", type=float, default=6.0, help="VAD 门限相对本底的余量 dB")
    p.add_argument("--min-gap", type=float, default=0.40, help="段内允许静音秒数")
    p.add_argument("--min-speech", type=float, default=0.30, help="段净语音下限秒数")
    p.add_argument("--min-peak-margin", type=float, default=18.0,
                   help="段峰值须高出本底的 dB 数（挡掉门限附近的小瞬态，默认 18）")
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--model-dir", default=str(paths.ASR_DIR))
    p.add_argument("--lang", default="ja", help="ja / zh / auto")
    p.add_argument("--threads", type=int, default=0, help="ctranslate2 线程数，0=默认（实测默认仅 4）")
    p.add_argument("--no-asr", action="store_true", help="只做电平与切分，不转写")
    p.add_argument("--json", action="store_true", help="输出 JSON 而非文本报告")
    args = p.parse_args(argv)

    if args.wav:
        audio, sr = read_wav(args.wav)
    else:
        print(f"录音 {args.seconds:.0f}s（{args.device}）...", file=sys.stderr)
        audio, sr = record(args.device, args.seconds)

    raw_db = db_of(audio)
    if args.hp > 0:
        filtered = highpass(audio, args.hp, sr, args.stages)
    else:
        filtered = audio
    filt_db = db_of(filtered)

    frames = frame_db(filtered, sr)
    floor_db = noise_floor(frames)
    thr = vad_threshold(floor_db, args.margin)
    segments = segment_speech(frames, thr, min_speech_s=args.min_speech, min_gap_s=args.min_gap,
                              floor_db=floor_db, min_peak_margin_db=args.min_peak_margin)

    results: list[dict] = []
    if not args.no_asr and segments:
        results = transcribe_segments(segments, filtered, sr, args.model, args.lang,
                                      args.model_dir, args.threads)

    if args.json:
        print(json.dumps({
            "duration_s": round(audio.shape[0] / sr, 2), "sr": sr,
            "raw_db": round(raw_db, 1), "filtered_db": round(filt_db, 1),
            "floor_db": round(floor_db, 1), "threshold_db": round(thr, 1),
            "clipped": int(np.count_nonzero(np.abs(audio) >= 32700)),
            "segments": [s.__dict__ for s in segments], "asr": results,
        }, ensure_ascii=False, indent=1))
    else:
        print(report(audio, sr, raw_db, filt_db, floor_db, thr, segments, results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
