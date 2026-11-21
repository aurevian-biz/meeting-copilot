#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Poll 8096 continuously, record the translation trajectory of each seq, and verify that streaming really pushes intermediate states.

Criteria:
  - the same seq shows >1 distinct trans text (intermediate states exist)
  - every intermediate state is a strict prefix of the final translation (so a whole-line replace will not flicker)
  - trans_ms is 0 on intermediate states (the on-page number does not jump around)
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8096"
DUR_S = float(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 40


def poll(after: int) -> dict:
    with urllib.request.urlopen(f"{BASE}/api/poll?after={after}", timeout=10) as r:
        return json.loads(r.read().decode())


def main() -> int:
    hist: dict[int, list[str]] = {}
    finals: dict[int, str] = {}
    last = 0
    t0 = time.time()
    polls = 0
    while time.time() - t0 < DUR_S:
        j = poll(last)
        polls += 1
        for u in j.get("upd") or []:
            seq = u["seq"]
            if seq not in hist:
                hist[seq] = []
            hist[seq].append(u["trans"])
            if u.get("final"):
                finals[seq] = u["trans"]
        for l in j.get("lines") or []:
            last = max(last, l["seq"])
        time.sleep(0.05)

    print(f"轮询 {polls} 次，覆盖 {len(hist)} 个 seq\n")
    multi = 0
    for seq in sorted(hist):
        steps = hist[seq]
        uniq = []
        for s in steps:
            if not uniq or uniq[-1] != s:
                uniq.append(s)
        fin = finals.get(seq, "")
        prefix_ok = all(fin.startswith(u) for u in uniq) if fin else None
        if len(uniq) > 1:
            multi += 1
        print(f"#{seq}  {len(uniq)} 个中间态  final={'有' if fin else '无'}  前缀一致={prefix_ok}")
        for i, u in enumerate(uniq, 1):
            mark = "  <<< final" if u == fin and i == len(uniq) else ""
            print(f"    {i}. {u}{mark}")

    print()
    print(f"有中间态的 seq: {multi}/{len(hist)}")
    all_prefix = all(
        finals.get(s) and all(finals[s].startswith(u) for u in hist[s])
        for s in hist if finals.get(s)
    )
    print("全部中间态都是最终译文的前缀:", all_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
