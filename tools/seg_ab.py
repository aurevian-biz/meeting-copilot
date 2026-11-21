#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Offline comparison of segmentation parameters: runs the Segmenter only, never touches ASR / translation, results in seconds.

Usage (run in the deployment directory, the venv has numpy):
    ./asr-venv/bin/python tools/seg_ab.py /tmp/meet60.wav
    ./asr-venv/bin/python tools/seg_ab.py /tmp/meet60.wav --sets "3.0:0.42:0.42:0.42,6.0:0.9:0.3:2.4"

Each --sets entry is written max_s:hold_s:hold_short:hold_long; four comparison sets run by default.

Watch three numbers:
- **silence cut ratio**: the higher, the more cuts land on natural pauses instead of hitting the time cap. This is the only goal of the current rework.
- segment length median / p75: directly sets the text latency (source text only leaves at segment end).
- cap-hit ratio: how much of the rest was forced by max_s.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import caption_server as cs  # noqa: E402
import voice_probe as vp  # noqa: E402

DEFAULT_SETS = [
    # max_s : hold : hold_short : hold_long   -- set 1 is the state before the rework
    (3.0, 0.42, 0.42, 0.42, "改造前 固定3s/固定0.42"),
    (6.0, 0.42, 0.30, 0.90, "分层 上限6s"),
    (8.0, 0.45, 0.30, 0.90, "分层 上限8s"),
    (12.0, 0.45, 0.30, 0.90, "分层 上限12s(近不限)"),
]


def run_set(audio: np.ndarray, max_s: float, hold: float,
            hold_short: float, hold_long: float) -> list[tuple[float, str]]:
    seg = cs.Segmenter(margin_db=8.0, hold_s=hold, max_s=max_s, min_s=0.25,
                       soft_s=0.7, hold_short_s=hold_short, hold_long_s=hold_long,
                       mid_s=1.20, long_s=2.40)
    out: list[tuple[float, str]] = []
    for i in range(0, audio.size - cs.FRAME_N, cs.FRAME_N):
        frame = audio[i:i + cs.FRAME_N]
        db = vp.db_of(frame.astype(np.float64))
        pcm = seg.push(frame, db)
        if pcm is not None and pcm.size:
            out.append((pcm.size / cs.SR, seg.last_cut))
    return out


def summarize(rows: list[tuple[float, str]], total: float) -> str:
    if not rows:
        return "  无输出"
    durs = sorted(d for d, _ in rows)
    n = len(durs)
    sil = sum(1 for _, c in rows if c == "silence")
    hard = n - sil
    return (f"  段数 {n:3d}   静音切 {sil / n * 100:4.0f}%   撞上限 {hard / n * 100:4.0f}%   "
            f"中位 {statistics.median(durs):.2f}s   p75 {durs[n * 3 // 4]:.2f}s   "
            f"max {max(durs):.2f}s   覆盖率 {sum(durs) / total * 100:.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--sets", help="max:hold:hold_short:hold_long,逗号分隔")
    args = ap.parse_args()

    audio, sr = vp.read_wav(args.wav)
    if sr != cs.SR:
        n = int(round(audio.size * cs.SR / sr))
        audio = np.interp(np.linspace(0, audio.size - 1, n),
                          np.arange(audio.size), audio).astype(np.int16)
    total = audio.size / cs.SR
    print(f"[文件] {args.wav}  {total:.1f}s")

    sets = DEFAULT_SETS
    if args.sets:
        sets = []
        for item in args.sets.split(","):
            parts = [float(x) for x in item.split(":")]
            sets.append((parts[0], parts[1], parts[2], parts[3], item))

    for max_s, hold, hs, hl, label in sets:
        rows = run_set(audio, max_s, hold, hs, hl)
        print(f"\n{label}  (max={max_s} hold={hold} short={hs} long={hl})")
        print(summarize(rows, total))
        print("  段长序列 " + " ".join(f"{d:.1f}" for d, _ in rows[:24])
              + (" …" if len(rows) > 24 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
