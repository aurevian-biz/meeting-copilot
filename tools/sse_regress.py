#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""sse_regress.py -- catch the scene where "the previous sentence's translation gets deleted / regresses".

Symptom (reported by users): the previous line is long and its translation is still streaming when the next line arrives, and the previous stream just disappears.

This does one thing only: subscribe to /api/events, record every translation version seen for each seq, then
judge two things --

  1. **regression**: the new version is not a prefix of the old one (`"abc" → "abd"` / shorter / becoming empty)
  2. **disappearance**: a line's translation goes from non-empty to empty

At the same time it replays the frontend's own merge logic (a Python port of `pushLines` + `applyUpd`),
prints "what will actually end up on the page", and confirms whether this is a frontend merge problem rather than the data itself.

Usage:
    ./asr-venv/bin/python tools/sse_regress.py --sec 90
    ./asr-venv/bin/python tools/sse_regress.py --sec 90 --verbose
"""
import argparse
import json
import time
import urllib.request


def is_prefix(old: str, new: str) -> bool:
    """Whether the new version extends the old one (the normal streaming shape)."""
    return new.startswith(old)


class Probe:
    def __init__(self, verbose: bool = False):
        self.seen: dict[int, str] = {}       # current version pushed by the server
        self.hist: dict[int, list[str]] = {}
        self.regress: list[dict] = []
        self.vanish: list[dict] = []
        self.verbose = verbose
        # Frontend merge state: seq -> line object (only trans matters)
        self.page: dict[int, dict] = {}
        self.page_order: list[int] = []

    def on_payload(self, p: dict, t: float) -> None:
        # ---- raw server pushes ----
        for u in p.get("upd") or []:
            seq, cur = u["seq"], u.get("trans") or ""
            old = self.seen.get(seq)
            if old is not None and cur != old:
                if not is_prefix(old, cur):
                    self.regress.append(dict(t=round(t, 2), seq=seq, old=old,
                                             new=cur, final=u.get("final")))
                    print(f"  !! t+{t:6.2f} #{seq} 回退  {len(old)}字→{len(cur)}字"
                          f" final={u.get('final')}\n     旧: {old[:70]}\n     新: {cur[:70]}",
                          flush=True)
                elif self.verbose:
                    print(f"  .. t+{t:6.2f} #{seq} 增长 {len(old)}→{len(cur)}", flush=True)
            if old and not cur:
                self.vanish.append(dict(t=round(t, 2), seq=seq, old=old))
                print(f"  !! t+{t:6.2f} #{seq} 译文变空（原先 {len(old)} 字）", flush=True)
            self.seen[seq] = cur
            self.hist.setdefault(seq, []).append(cur)

        # ---- frontend merge logic ported: pushLines then applyUpd ----
        for l in p.get("lines") or []:
            if l["seq"] in self.page:
                continue                     # idx.has dedupe
            self.page[l["seq"]] = dict(trans=l.get("trans") or "",
                                       pending=bool(l.get("pending")),
                                       text=(l.get("text") or "")[:20])
            self.page_order.append(l["seq"])
        for u in p.get("upd") or []:
            row = self.page.get(u["seq"])
            if row is not None:
                row["trans"] = u.get("trans") or ""
                if u.get("final"):
                    row["pending"] = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8096/api/events")
    ap.add_argument("--sec", type=float, default=90.0)
    ap.add_argument("--verbose", action="store_true", help="打印每一次正常增长")
    args = ap.parse_args()

    url = f"{args.url}?after=0"
    print(f"订阅 {url}  {args.sec:.0f}s —— 等着看有没有「回退/清空」\n")
    pr = Probe(args.verbose)
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    t0 = time.time()
    events = 0
    try:
        with urllib.request.urlopen(req, timeout=args.sec + 10) as r:
            while time.time() - t0 < args.sec:
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    p = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                events += 1
                pr.on_payload(p, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        print(f"\n异常中断：{type(e).__name__}: {e}")

    print("\n########## 汇总")
    print(f"SSE 事件 {events} 条，涉及 {len(pr.page)} 行")
    print(f"回退 {len(pr.regress)} 次，译文变空 {len(pr.vanish)} 次")
    multi = {k: v for k, v in pr.hist.items() if len(v) > 1}
    print(f"有多次中间态的行 {len(multi)} 行（这些是真正在流的）")
    if pr.regress:
        print("\n回退明细（前 8 条）：")
        for e in pr.regress[:8]:
            print(f"  #{e['seq']} final={e['final']} {len(e['old'])}→{len(e['new'])}字"
                  f"\n    旧: {e['old'][:90]}\n    新: {e['new'][:90]}")
    print("\n页面最终状态（后 6 行）：")
    for seq in pr.page_order[-6:]:
        row = pr.page[seq]
        print(f"  #{seq} pending={row['pending']} trans={row['trans'][:60] or '(空)'}")


if __name__ == "__main__":
    main()
