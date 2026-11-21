#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""tts_probe.py — latency and quality diagnostics for Piper Japanese synthesis.

For full duplex, what really matters is not "how fast synthesis finishes" but "how soon the first sound is heard".
So three numbers are measured here:
    load_s      model load (happens only once in a resident service)
    first_ms    latency to the first audio chunk — decides the "responsive" feel
    rtf         (total synthesis time / audio duration) — decides whether it keeps up with speaking speed

Usage:
    python tts_probe.py --text "確認してください。"
    python tts_probe.py --text-file in.txt --out /tmp/a.wav
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

DEFAULT_MODEL = str(paths.PIPER_JA)


def synthesize(model_path: str, text: str, out_path: str | None,
               speaker: int | None = None) -> dict:
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    t0 = time.time()
    voice = PiperVoice.load(model_path)
    load_s = time.time() - t0

    cfg = SynthesisConfig(speaker_id=speaker) if speaker is not None else None

    # Streaming synthesis: record when the first chunk arrives — that is the latency duplex actually perceives
    t1 = time.time()
    first_ms = None
    chunks = []
    for chunk in voice.synthesize(text, syn_config=cfg):
        if first_ms is None:
            first_ms = (time.time() - t1) * 1000.0
        chunks.append(chunk)
    total_s = time.time() - t1

    if not chunks:
        raise RuntimeError("没有合成出任何音频")

    sr = chunks[0].sample_rate
    n_samples = sum(len(c.audio_int16_array) for c in chunks)
    duration = n_samples / sr

    if out_path:
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for c in chunks:
                wf.writeframes(c.audio_int16_bytes)

    return {
        "load_s": round(load_s, 3),
        "first_ms": round(first_ms or 0.0, 1),
        "total_s": round(total_s, 3),
        "duration_s": round(duration, 2),
        "rtf": round(total_s / max(duration, 1e-9), 3),
        "sample_rate": sr,
        "chars": len(text),
        "chars_per_s": round(len(text) / max(duration, 1e-9), 1),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text")
    p.add_argument("--text-file")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out")
    p.add_argument("--speaker", type=int, default=None)
    args = p.parse_args(argv)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()
    if not text:
        print("没有文本", file=sys.stderr)
        return 2

    r = synthesize(args.model, text, args.out, args.speaker)
    print(f"加载      {r['load_s']}s")
    print(f"首块延迟  {r['first_ms']}ms   ← 双工真正感知的延迟")
    print(f"合成      {r['total_s']}s / 音频 {r['duration_s']}s  → RTF {r['rtf']}")
    print(f"采样率    {r['sample_rate']}Hz   语速 {r['chars_per_s']} 字/秒")
    if args.out:
        print(f"输出      {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
