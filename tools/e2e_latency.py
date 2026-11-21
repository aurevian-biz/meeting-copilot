#!/usr/bin/env python3
"""量端到端：从「一段话说完」到「页面看到第一个译文字」。

算法：
  段末时刻 ≈ 收到原文行的时刻 - lag_ms（lag 就是段末到原文出来的时间）
  首字延迟 = 收到该行第一个非空 trans 的时刻 - 段末时刻
对照：整句译完的延迟 = 收到 final 的时刻 - 段末时刻
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8096"
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 40


def main() -> int:
    seen_line: dict[int, float] = {}
    lag: dict[int, float] = {}
    first_word: dict[int, float] = {}
    final_at: dict[int, float] = {}
    fresh: set[int] = set()      # 订阅期间新产生的行（首次收到时还没有译文）

    t0 = time.time()
    with urllib.request.urlopen(f"{BASE}/api/events?after=0", timeout=DUR + 20) as r:
        for raw in r:
            now = time.time()
            if now - t0 > DUR:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                j = json.loads(line[5:].strip())
            except Exception:  # noqa: BLE001
                continue
            for l in j.get("lines") or []:
                seq = l["seq"]
                if seq in seen_line:
                    continue
                seen_line[seq] = now
                lag[seq] = (l.get("lag_ms") or 0) / 1000.0
                if not l.get("trans"):
                    fresh.add(seq)          # 历史补发的行译文已就绪，不能算
                else:
                    first_word.setdefault(seq, now)
            for u in j.get("upd") or []:
                seq = u["seq"]
                if u.get("trans"):
                    first_word.setdefault(seq, now)
                if u.get("final"):
                    final_at[seq] = now

    rows = []
    for seq in sorted(fresh & set(first_word)):
        seg_end = seen_line[seq] - lag[seq]
        fw = (first_word[seq] - seg_end) * 1000
        fa = (final_at[seq] - seg_end) * 1000 if seq in final_at else None
        rows.append((seq, lag[seq] * 1000, fw, fa))
        print(f"#{seq:<3} 段末→原文 {lag[seq]*1000:5.0f}ms   →首字译文 {fw:5.0f}ms"
              f"   →译完 {('%.0fms' % fa) if fa else '—'}")

    if rows:
        print()
        print(f"段末→原文     中位 {statistics.median([r[1] for r in rows]):.0f}ms")
        print(f"段末→首字译文 中位 {statistics.median([r[2] for r in rows]):.0f}ms")
        done = [r[3] for r in rows if r[3]]
        if done:
            print(f"段末→译文全出 中位 {statistics.median(done):.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
