#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""bench_sensevoice.py -- run the same clip on CPU with sherpa-onnx's SenseVoice.

Why this exists: faster-whisper has RTF from 0.92 (small) to 2.2 (turbo) on Orin,
while duplex demands "end of speech to result" within 1 second, i.e. RTF <0.5, which whisper on CPU cannot reach.
SenseVoice is a non-autoregressive model built for real-time use; one forward pass yields the whole result, typically in the 0.1 range.

Usage:
    python bench_sensevoice.py --wav a.wav --segments 18.43-21.72,41.95-45.51
    python bench_sensevoice.py --wav a.wav --whole          # no segmentation, feed the whole clip
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voice_probe as vp  # noqa: E402

MODEL_DIR = "/opt/meeting-copilot/data/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


def parse_segments(spec: str) -> list[tuple[float, float]]:
    out = []
    for part in spec.split(","):
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return out


def build_recognizer(model_dir: str, threads: int):
    import sherpa_onnx

    d = Path(model_dir)
    model = d / "model.int8.onnx"
    if not model.exists():
        model = d / "model.onnx"
    if not model.exists():
        raise FileNotFoundError(f"{d} 下找不到 model.int8.onnx / model.onnx")
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model),
        tokens=str(d / "tokens.txt"),
        num_threads=threads,
        use_itn=True,      # number/punctuation normalisation, useful for both captions and dialogue
        language="ja",
        debug=False,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--segments", default="", help="形如 18.4-21.7,41.9-45.5")
    p.add_argument("--whole", action="store_true", help="整段送进去（不切分）")
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--hp", type=float, default=150.0)
    args = p.parse_args(argv)

    audio, sr = vp.read_wav(args.wav)
    if args.hp > 0:
        audio = vp.highpass(audio, args.hp, sr, 2)

    rec = build_recognizer(args.model_dir, args.threads)

    if args.whole:
        spans = [(0.0, audio.shape[0] / sr)]
    else:
        spans = parse_segments(args.segments)
    if not spans:
        print("没有给 --segments 也没给 --whole", file=sys.stderr)
        return 2

    total_audio = 0.0
    total_cost = 0.0
    for i, (a, b) in enumerate(spans, 1):
        chunk = audio[int(a * sr):int(b * sr)].astype(np.float32) / vp.FULL_SCALE
        dur = b - a
        t0 = time.time()
        stream = rec.create_stream()
        stream.accept_waveform(sr, chunk)
        rec.decode_stream(stream)
        cost = time.time() - t0
        total_audio += dur
        total_cost += cost
        print(f"[{i}] {a:.2f}-{b:.2f}s ({dur:.2f}s)  {cost:.2f}s  RTF {cost / dur:.2f}")
        print(f"    {stream.result.text.strip() or '(空)'}")

    if total_audio > 0:
        print(f"\n合计 音频 {total_audio:.2f}s  耗时 {total_cost:.2f}s  "
              f"RTF {total_cost / total_audio:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
