#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Measure end to end: from "the sentence is finished" to "the first translated character shows up on the page".

Method:
  segment end ≈ arrival time of the source line - lag_ms (lag is the time from segment end to source text popping out)
  first token latency = arrival time of the first non-empty trans for that line - segment end
For reference: full-sentence latency = arrival time of final - segment end
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
    fresh: set[int] = set()      # lines created during the subscription (no translation yet on first sight)

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
                    fresh.add(seq)          # backfilled historical lines already have their translation, must not count
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
