#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Capture the push rhythm of /api/events for a while: when each event lands and how long its content is.

Meant to answer the hands-on question of whether captions really stream:
- event gap around 80ms -> the translation drips out token by token, streaming works
- event gap in the 1s range with one whole sentence per event -> the frontend is not on the push path, or there is no streaming at all
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8096/api/events")
    ap.add_argument("--sec", type=float, default=14.0)
    args = ap.parse_args()

    url = f"{args.url}?after=0"
    print(f"订阅 {url}  {args.sec}s\n")
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    t0 = time.time()
    last = t0
    n = 0
    gaps = []

    with urllib.request.urlopen(req, timeout=args.sec + 5) as r:
        for raw in r:
            if time.time() - t0 > args.sec:
                break
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            try:
                p = json.loads(s[5:].strip())
            except Exception:  # noqa: BLE001
                continue
            now = time.time()
            gap = (now - last) * 1000
            gaps.append(gap)
            last = now
            n += 1
            seq = p.get("seq")
            upd = p.get("upd") or []
            for u in upd:
                tr = u.get("trans") or ""
                print(f"t+{(now - t0) * 1000:6.0f}ms  gap={gap:5.0f}ms  "
                      f"seq={seq} upd#{u.get('seq')} final={u.get('final')} "
                      f"len={len(tr):3d}  {tr[:60]}")
            d = p.get("draft") or {}
            if d.get("text") and d.get("text") != getattr(main, "_last_draft", None):
                main._last_draft = d["text"]            # type: ignore[attr-defined]
                print(f"t+{(now - t0) * 1000:6.0f}ms  gap={gap:5.0f}ms  "
                      f"[草稿] len={len(d['text']):3d}  {d['text'][:60]}")

    if gaps:
        gaps_sorted = sorted(gaps[1:])
        mid = gaps_sorted[len(gaps_sorted) // 2] if len(gaps_sorted) > 1 else 0
        print(f"\n事件 {n} 个  间隔中位 {mid:.0f}ms  最大 {max(gaps[1:] or [0]):.0f}ms")
    else:
        print("\n14 秒内没有任何事件 —— 说话了吗？或前端链路断了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
