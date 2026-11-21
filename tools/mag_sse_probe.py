#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""SSE probe for Magnitude: pull cache_n / prompt_n out of `timings`.

Magnitude attaches `timings` to every chunk, containing
  prompt_n  token count of this prompt
  cache_n   how many of those were **reused from the KV prefix cache**   <- the hard criterion for whether the prefix cache works at all
  prompt_ms actual elapsed prefill time for this call
This is more reliable than "look at first token latency": the first token mixes in decode, scheduling and network, while cache_n is reported by the engine itself.

It also counts characters in content / reasoning_content -- with the Qwen3.5 template and thinking left on,
all 200 tokens burn in reasoning and content is always empty (hit on both vLLM and SGLang).

Usage:
  ./asr-venv/bin/python tools/mag_sse_probe.py            # by default connects to the model loaded on 10100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import caption_server as cs  # noqa: E402


def one(base: str, key: str, model: str, sysp: str, user: str,
        extra: dict, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": user}],
        "stream": True, "temperature": 0.1, "max_tokens": max_tokens,
    }
    body.update(extra)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    first_wall = -1
    content, reasoning = [], []
    timings: dict = {}
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except json.JSONDecodeError:
                continue
            if d.get("timings") and not timings:
                timings = d["timings"]
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                if first_wall < 0:
                    first_wall = int((time.time() - t0) * 1000)
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
    return {
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": round(timings.get("prompt_ms") or 0, 1),
        "predicted_n": timings.get("predicted_n"),
        "ttft_ms": round(timings.get("time_to_first_token_ms") or 0, 1),
        "wall_first_ms": first_wall,
        "total_ms": int((time.time() - t0) * 1000),
        "content": "".join(content).strip(),
        "n_content": len("".join(content).strip()),
        "n_reasoning": len("".join(reasoning).strip()),
    }


VARIANTS = [
    ("plain        ", {}),
    ("think_off(tpl)", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("reason_effort", {"reasoning_effort": "none"}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:10100/inference/v1")
    ap.add_argument("--key", default=json.load(
        open("/opt/meeting-copilot/data/magnitude-data/.magnitude/config.json")
    )["network"]["apiKey"])
    ap.add_argument("--model", default="lfm2.5-8b-a1b:gguf:q4")
    ap.add_argument("--target", default="ja", choices=["zh", "ja"])
    ap.add_argument("--text", default="中方刻意给韩国好处")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="只跑名字里含该串的变体")
    args = ap.parse_args()

    sysp = cs._tr_system(args.target)
    print(f"base={args.base}\nmodel={args.model}\nsystem prompt {len(sysp)} 字\n")

    for label, extra in VARIANTS:
        if args.only and args.only not in label:
            continue
        print(f"-- {label} {json.dumps(extra, ensure_ascii=False) if extra else ''}")
        for i in range(1, args.repeats + 1):
            # identical prefix: mimic production where every sentence uses the same system prompt
            try:
                r = one(args.base, args.key, args.model, sysp, args.text, extra, args.max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"   第{i}发 失败 {type(e).__name__}: {e}")
                break
            print(f"   第{i}发 prompt_n={r['prompt_n']} cache_n={r['cache_n']} "
                  f"prompt={r['prompt_ms']}ms ttft={r['ttft_ms']}ms "
                  f"total={r['total_ms']}ms  content={r['n_content']}字 "
                  f"reasoning={r['n_reasoning']}字")
            print(f"        → {r['content'][:70]!r}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
