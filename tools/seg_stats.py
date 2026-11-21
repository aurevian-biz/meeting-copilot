#!/usr/bin/env python3
"""统计字幕分段的段长分布与「切在句界还是切在半句」的比例。

用法：
    curl -s http://127.0.0.1:8096/api/poll > /tmp/poll.json
    python tools/seg_stats.py /tmp/poll.json

判据说明：
- 段末有句界标点（。！？.!?）→ 大概率是静音自然断句
- 段末只有逗号或没有标点 → 被硬切（max_s 到顶）或切在词间小停顿
- dur 贴住 max_s（±0.1s）→ 确凿的硬切
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

    # 段长直方（0.5s 一档）
    buckets: dict[float, int] = {}
    for d in durs:
        b = round(d * 2) / 2
        buckets[b] = buckets.get(b, 0) + 1
    print("\n段长分布")
    for b in sorted(buckets):
        bar = "#" * round(buckets[b] / n * 50)
        print(f"  {b:4.1f}s {buckets[b]:3d} {buckets[b] / n * 100:4.0f}% {bar}")

    # 句界 vs 半句
    at_end = sum(1 for l in lines if (l.get("text") or "").rstrip()[-1:] in END)
    print(f"\n段末是句界标点 {at_end}/{n} = {at_end / n * 100:.0f}%")
    print(f"段末不是句界   {n - at_end}/{n} = {(n - at_end) / n * 100:.0f}%  （被切开）")

    # 撞上限：必须用配置的 max_s 判，拿 max(dur) 当基准在样本少时会误判
    # （7 段里最长 4.28s，配置是 4.5，真实撞上限的段一个都没被数出来）
    lim = max_s if max_s else max(durs)
    tag = "" if max_s else "（未传 --max-s，用最长段近似，仅供参考）"
    hard = [d for d in durs if d >= lim - 0.15]
    print(f"贴住上限(≥{lim - 0.15:.2f}s) 的段 {len(hard)} 个 = {len(hard) / n * 100:.0f}%"
          f"  ← 撞上限硬切{tag}")

    # 半句样例：段末无标点，且下一段开头像承接（无主语/以量词数字开头不好自动判，只列样例）
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
