#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Tally the segment length distribution of caption segments and the share cut at a sentence boundary versus mid-sentence.

Usage:
    curl -s http://127.0.0.1:8096/api/poll > /tmp/poll.json
    python tools/seg_stats.py /tmp/poll.json

Criteria:
- segment ends on sentence-boundary punctuation (。！？.!?) -> most likely a natural silence segmentation
- segment ends on a comma only, or on no punctuation -> hard cut (max_s hit the ceiling) or a cut at a small inter-word pause
- dur hugs max_s (±0.1s) -> a certain hard cut
"""
from __future__ import annotations

import json
import statistics
import sys

END = "。！？.!?"


def main(path: str, max_s: float | None = None) -> int:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lines = data.get("lines") or data
    if not lines:
        print("没有数据")
        return 1

    durs = [float(l.get("dur") or 0) for l in lines if l.get("dur")]
    lens = [len(l.get("text") or "") for l in lines]
    if not durs:
        print("没有 dur 字段")
        return 1

    n = len(durs)
    durs_sorted = sorted(durs)
    print(f"段数 {n}")
    print(f"段长  min {min(durs):.2f}  p25 {durs_sorted[n // 4]:.2f}  "
          f"中位 {statistics.median(durs):.2f}  p75 {durs_sorted[n * 3 // 4]:.2f}  "
          f"max {max(durs):.2f}")
    print(f"总时长 {sum(durs):.1f}s   平均字数 {statistics.mean(lens):.1f}  "
          f"字/秒 {sum(lens) / max(0.01, sum(durs)):.1f}")

    # Segment length histogram (0.5s bins)
    buckets: dict[float, int] = {}
    for d in durs:
        b = round(d * 2) / 2
        buckets[b] = buckets.get(b, 0) + 1
    print("\n段长分布")
    for b in sorted(buckets):
        bar = "#" * round(buckets[b] / n * 50)
        print(f"  {b:4.1f}s {buckets[b]:3d} {buckets[b] / n * 100:4.0f}% {bar}")

    # Sentence boundary vs mid-sentence
    at_end = sum(1 for l in lines if (l.get("text") or "").rstrip()[-1:] in END)
    print(f"\n段末是句界标点 {at_end}/{n} = {at_end / n * 100:.0f}%")
    print(f"段末不是句界   {n - at_end}/{n} = {(n - at_end) / n * 100:.0f}%  （被切开）")

    # Cap hits: must be judged against the configured max_s; using max(dur) as the baseline misfires on small samples
    # (longest of 7 segments was 4.28s while the config is 4.5, so not a single real cap hit was counted)
    lim = max_s if max_s else max(durs)
    tag = "" if max_s else "（未传 --max-s，用最长段近似，仅供参考）"
    hard = [d for d in durs if d >= lim - 0.15]
    print(f"贴住上限(≥{lim - 0.15:.2f}s) 的段 {len(hard)} 个 = {len(hard) / n * 100:.0f}%"
          f"  ← 撞上限硬切{tag}")

    # Mid-sentence samples: segment ends with no punctuation and the next one starts like a continuation (no subject / numeral-classifier start is hard to auto-detect, so just list samples)
    print("\n段末无标点样例（前 8 条）")
    shown = 0
    for l in lines:
        t = (l.get("text") or "").rstrip()
        if t and t[-1] not in END and shown < 8:
            print(f"  [{l.get('dur')}s] {t}")
            shown += 1
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    ms = None
    if "--max-s" in argv:
        i = argv.index("--max-s")
        ms = float(argv[i + 1])
        del argv[i:i + 2]
    sys.exit(main(argv[0] if argv else "/tmp/poll.json", ms))
