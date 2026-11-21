#!/usr/bin/env python3
"""tts_probe.py —— Piper 日语合成的时延与质量诊断。

对双工来说，真正重要的不是「合成完要多快」，而是「多久能听到第一个音」。
所以这里测三个数：
    load_s      模型加载（常驻服务里只发生一次）
    first_ms    首个音频块返回的时延 —— 决定「回应感」
    rtf         (总合成时间 / 音频时长) —— 决定能否追上说话速度

用法：
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

    # 流式合成：记录首块到达时间，这才是双工真正感知到的延迟
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
