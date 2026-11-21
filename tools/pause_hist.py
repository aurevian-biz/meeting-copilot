#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Measure the length distribution of pauses in a clip -- it decides how loose the segmentation threshold can be.

Usage:
    ./asr-venv/bin/python tools/pause_hist.py /tmp/meet60.wav

The threshold uses the same convention as the Segmenter: 10th percentile as the baseline + margin 8dB.
Prints the count of pauses above each bucket (0.2/0.3/0.45/0.7/0.9/1.2s), plus "the share of pauses that threshold can use".

Reading it:
- very few pauses ≥0.9s -> dense speech simply has no long pauses, no value of hold_long helps;
the only way out is to drop the "wait for a pause" step (post-segment semantic merge) instead of retuning the threshold.
- plenty ≥0.3s but a sharp drop at ≥0.45s -> the speaker only breathes between words; sentence and word boundaries are indistinguishable by energy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import caption_server as cs  # noqa: E402
import voice_probe as vp  # noqa: E402


def main(path: str) -> int:
    audio, sr = vp.read_wav(path)
    if sr != cs.SR:
        n = int(round(audio.size * cs.SR / sr))
        audio = np.interp(np.linspace(0, audio.size - 1, n),
                          np.arange(audio.size), audio).astype(np.int16)
    total = audio.size / cs.SR

    dbs = []
    for i in range(0, audio.size - cs.FRAME_N, cs.FRAME_N):
        dbs.append(vp.db_of(audio[i:i + cs.FRAME_N].astype(np.float64)))
    dbs_a = np.array(dbs)
    valid = dbs_a[dbs_a > -80.0]
    floor = float(np.percentile(valid, 10)) if valid.size else -70.0
    thr = floor + 8.0
    silent = dbs_a <= thr

    runs: list[float] = []
    cur = 0
    for s in silent:
        if s:
            cur += 1
        elif cur:
            runs.append(cur * cs.FRAME_S)
            cur = 0
    if cur:
        runs.append(cur * cs.FRAME_S)

    speech = 1.0 - silent.mean()
    print(f"[文件] {path}  {total:.1f}s   本底 {floor:.1f}dB  门限 {thr:.1f}dB  有声占比 {speech * 100:.0f}%")
    print(f"停顿总个数 {len(runs)}   停顿总时长 {sum(runs):.1f}s / {total:.1f}s")

    print("\n停顿长度分档（个数 / 占总停顿数 / 累计可用）")
    for t in (0.2, 0.3, 0.45, 0.6, 0.9, 1.2, 2.0):
        n = sum(1 for r in runs if r >= t)
        cum = sum(r for r in runs if r >= t)
        print(f"  ≥{t:3.2f}s  {n:3d} 个  {n / max(1, len(runs)) * 100:3.0f}%   "
              f"共 {cum:5.1f}s")

    # Histogram: 0.2s bins
    buckets: dict[float, int] = {}
    for r in runs:
        b = min(2.0, round(r * 5) / 5)
        buckets[b] = buckets.get(b, 0) + 1
    print("\n分布")
    for b in sorted(buckets):
        bar = "#" * min(60, buckets[b])
        print(f"  {b:4.1f}s {buckets[b]:3d} {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/meet60.wav"))
