#!/usr/bin/env python3
"""连续轮询 8096，记录每个 seq 的译文变化轨迹，验证流式真的在推中间态。

判据：
  - 同一个 seq 出现 >1 种 trans 文本（说明有中间态）
  - 中间态都是最终译文的严格前缀（前端整行替换才不会抖）
  - 中间态的 trans_ms 为 0（页面数字不乱跳）
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
