#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Two clients subscribe to /api/events at once: verify each one receives intermediate states (no patch stealing)."""
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8096"
DUR = 40
results: dict[str, dict[int, list[str]]] = {}


def watch(name: str) -> None:
    hist: dict[int, list[str]] = {}
    results[name] = hist
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"{BASE}/api/events?after=0", timeout=DUR + 20) as r:
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
                for u in j.get("upd") or []:
                    hist.setdefault(u["seq"], []).append(u["trans"])
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] 异常: {e}")


ts = [threading.Thread(target=watch, args=(n,)) for n in ("A", "B")]
for t in ts:
    t.start()
for t in ts:
    t.join()

print(f"观测 {DUR}s\n")
seqs_a, seqs_b = set(results["A"]), set(results["B"])
for name in ("A", "B"):
    h = results[name]
    multi = sum(1 for s, v in h.items() if len(set(v)) > 1)
    total = sum(len(set(v)) for v in h.values())
    print(f"客户端 {name}: {len(h)} 个 seq，{multi} 个含中间态，共 {total} 个版本")

print(f"\n两端口径: A={len(seqs_a)} B={len(seqs_b)} 交集={len(seqs_a & seqs_b)}")
both_mid = [s for s in seqs_a & seqs_b
            if len(set(results['A'][s])) > 1 and len(set(results['B'][s])) > 1]
print(f"两端都看到中间态的 seq: {len(both_mid)}  {sorted(both_mid)[:8]}")
print("\n判据：交集里应该也有多版本 —— 说明没有互相抢走补丁")
