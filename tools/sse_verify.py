#!/usr/bin/env python3
"""订阅 /api/events，统计收到的中间态与时间间隔。

判据：
  - 同一个 seq 能收到 >1 种 trans（说明服务端把中间态推出来了）
  - 相邻两次推送的间隔明显小于 250ms 轮询周期（说明延迟真的省了）
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8096"
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 40


def main() -> int:
    hist: dict[int, list[str]] = {}
    finals: dict[int, str] = {}
    times: list[float] = []
    req = urllib.request.Request(f"{BASE}/api/events?after=0")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=DUR + 20) as r:
        for raw in r:
            if time.time() - t0 > DUR:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                j = json.loads(line[5:].strip())
            except Exception:  # noqa: BLE001
                continue
            times.append(time.time())
            for u in j.get("upd") or []:
                seq = u["seq"]
                hist.setdefault(seq, []).append(u["trans"])
                if u.get("final"):
                    finals[seq] = u["trans"]

    print(f"收到 {len(times)} 次推送，覆盖 {len(hist)} 个 seq\n")
    multi = 0
    for seq in sorted(hist):
        uniq = []
        for s in hist[seq]:
            if not uniq or uniq[-1] != s:
                uniq.append(s)
        fin = finals.get(seq, "")
        if len(uniq) > 1:
            multi += 1
        print(f"#{seq}  {len(uniq)} 个版本  前缀一致={all(fin.startswith(u) for u in uniq) if fin else None}")
        for u in uniq:
            print(f"    · {u}")

    if len(times) > 1:
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        print(f"\n推送间隔 中位 {statistics.median(gaps):.0f}ms  最快 {min(gaps):.0f}ms")
        print(f"（轮询是固定 250ms；SSE 间隔明显更小说明中间态是即时到达的）")
    print(f"\n有中间态的 seq: {multi}/{len(hist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
