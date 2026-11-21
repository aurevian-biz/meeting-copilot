#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Measure the first token only: long vs short prompt × with vs without context, a four-cell comparison.

Why this tool exists: `ttft_ms` from the production `/api/status` sits at 400-800ms for long stretches, while a single offline sentence takes only
89ms. Whether that 3-7x gap is "the system prompt is too long (250-300 tokens)" or "this sentence + context
adds too many new tokens" cannot be seen by just swapping prompts -- the two factors have to be separated.

Four cells:
  full/ctx2   the production shape
  short/ctx2  cut the system prompt only
  full/ctx0   cut the context only
  short/ctx0  cut both, the theoretical lower bound

The criterion is not "how large is the first token" but **the `cache_n` the server reports**:
  · `cache_n` steadily > 0  -> the fixed prefix is reused, those tokens are free, and prompt length barely
    affects the first token; shipping the short version is wasted work
  · `cache_n` = 0      -> every sentence re-pays the whole prefill, and only then is prompt length billed in milliseconds

Mind the field semantics (easy to get backwards): `prompt_n` is **the token count really processed this round**, `cache_n` is
**the part reused from the KV prefix**; only the sum of the two is the length of the whole prompt.

The cases must be **real sentences** (the segment length distribution decides the token count); by default the latest
finalised Japanese segments are pulled from the production `/api/poll`, and context is assembled from neighbouring lines per `--ctx-lines`. Capture one copy before shutting anything down:

    curl -s http://127.0.0.1:8096/api/poll > /tmp/poll.json
    ./asr-venv/bin/python tools/ttft_prompt_ab.py --cases-from /tmp/poll.json --save-cases /tmp/cases.json
    ./asr-venv/bin/python tools/ttft_prompt_ab.py --cases /tmp/cases.json \
        --base http://127.0.0.1:10100/inference/v1 \
        --model "hf:unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" \
        --key mag-xxx --out /tmp/ttft_prompt.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import caption_server as cs  # noqa: E402

CELLS = [("full", 2), ("short", 2), ("full", 0), ("short", 0)]


def cases_from_poll(path: str, ctx_lines: int) -> list[dict]:
    """Turn /api/poll lines into {(target, text, ctx)}. Finalised lines only."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = [l for l in d.get("lines", []) if l.get("text") and not l.get("pending")]
    out = []
    for i, l in enumerate(lines):
        src = l.get("lang", "ja")
        target = "zh" if src == "ja" else "ja"
        ctx = "".join(x["text"] for x in lines[max(0, i - ctx_lines):i]) if ctx_lines else ""
        out.append({"target": target, "text": l["text"], "ctx": ctx,
                    "dur": l.get("dur"), "live_ttft": l.get("ttft_ms")})
    return out


def one(base: str, model: str, key: str, style: str, target: str,
        text: str, ctx: str) -> dict:
    """Send one streaming request. Returns first-token / full-text elapsed + the server-side timings."""
    sysp = cs._tr_system(target, style)
    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    body = {
        "model": model,
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": user}],
        "stream": True, "temperature": 0.1, "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    ttft, parts, tim = -1, [], {}
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
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
                if isinstance(d.get("timings"), dict):
                    tim = d["timings"]
                piece = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if piece:
                    if ttft < 0:
                        ttft = int((time.time() - t0) * 1000)
                    parts.append(piece)
    except Exception as e:  # noqa: BLE001
        return {"ttft": -1, "total": int((time.time() - t0) * 1000),
                "out": f"<ERR {e}>", "tim": {}}
    return {"ttft": ttft, "total": int((time.time() - t0) * 1000),
            "out": "".join(parts).strip(), "tim": tim}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--key", default="")
    ap.add_argument("--cases", help="用例 json（已抓好的）")
    ap.add_argument("--cases-from", help="从 /api/poll 的 dump 造用例")
    ap.add_argument("--save-cases", help="造好的用例存这里")
    ap.add_argument("--ctx-lines", type=int, default=2, help="造用例时带几句上文")
    ap.add_argument("--limit", type=int, default=12, help="最多用几条用例")
    ap.add_argument("--repeat", type=int, default=2, help="每格每句跑几次取中位")
    ap.add_argument("--out", help="结果 json")
    args = ap.parse_args()

    if args.cases_from:
        cases = cases_from_poll(args.cases_from, args.ctx_lines)[:args.limit]
        if args.save_cases:
            Path(args.save_cases).write_text(
                json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"用例 {len(cases)} 条 → {args.save_cases}")
        if not args.cases:
            return 0
    if not args.cases:
        ap.error("给 --cases 或 --cases-from")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))[:args.limit]

    print(f"# base={args.base} model={args.model}")
    print(f"# system 长度：full(ja)={len(cs._tr_system('ja', 'full'))} 字  "
          f"short(ja)={len(cs._tr_system('ja', 'short'))} 字")
    print(f"# 用例 {len(cases)} 条，每格每句 {args.repeat} 次\n")

    res: dict[tuple[str, int], list[dict]] = {c: [] for c in CELLS}
    # Two rounds in opposite order to cancel cache/temperature drift; warm up once at the start of each cell so the prefix cache is hot
    order = [CELLS, list(reversed(CELLS))]
    for rd, cells in enumerate(order, 1):
        for style, ctxn in cells:
            one(args.base, args.model, args.key, style,
                "zh", "熱身", "")          # warm-up, not counted
            for case in cases:
                ctx = case["ctx"] if ctxn else ""
                r = one(args.base, args.model, args.key, style,
                        case["target"], case["text"], ctx)
                r.update({"round": rd, "style": style, "ctxn": ctxn,
                          "src": case["text"], "target": case["target"]})
                res[(style, ctxn)].append(r)

    print(f"{'格':<14}{'prompt_n':>9}{'cache_n':>9}{'首字p50':>9}{'首字p90':>9}"
          f"{'全文p50':>9}{'<200ms':>8}")
    summary = []
    for style, ctxn in CELLS:
        rows = [r for r in res[(style, ctxn)] if r["ttft"] >= 0]
        if not rows:
            print(f"{style}/{ctxn}  全部失败")
            continue
        g = lambda k: [r["tim"].get(k) for r in rows if r["tim"].get(k) is not None]  # noqa: E731
        pn = int(statistics.median(g("prompt_n"))) if g("prompt_n") else -1
        cn = int(statistics.median(g("cache_n"))) if g("cache_n") else -1
        tt = sorted(r["ttft"] for r in rows)
        p50 = int(statistics.median(tt))
        p90 = tt[min(len(tt) - 1, int(len(tt) * 0.9))]
        tot = int(statistics.median([r["total"] for r in rows]))
        hit = sum(1 for t in tt if t < 200)
        print(f"{style + '/ctx' + str(ctxn):<14}{pn:>9}{cn:>9}{p50:>9}{p90:>9}"
              f"{tot:>9}{hit:>5}/{len(tt)}")
        summary.append({"cell": f"{style}/ctx{ctxn}", "prompt_n": pn, "cache_n": cn,
                        "ttft_p50": p50, "ttft_p90": p90, "total_p50": tot,
                        "under200": hit, "n": len(tt),
                        "out": rows[-1]["out"]})

    # Criterion: the cache hit rate, not the first token itself. Note prompt_n is "the token count really processed this round"
    # and cache_n is "the part reused from the KV prefix"; only the sum of the two is the whole prompt.
    print()
    for s in summary:
        if s["prompt_n"] > 0:
            print(f"{s['cell']:<14} 新 token {s['prompt_n']:>3}（复用 cache_n={s['cache_n']}）"
                  f"  首字 {s['ttft_p50']}ms")
    print("\ncache_n > 0 且稳定 ⇒ 固定前缀不要钱，砍它没用；"
          "\ncache_n = 0          ⇒ 每句重付整个 prefill，只有这种时候砍 prompt 才有意义。")

    print("\n-- 译文抽样（看短版有没有掉质量）--")
    for case in cases[:6]:
        print(f"\n[原文] {case['text']}")
        for style, ctxn in CELLS:
            hit = [r for r in res[(style, ctxn)] if r["src"] == case["text"] and r["ttft"] >= 0]
            if hit:
                print(f"  {style}/ctx{ctxn}: {hit[-1]['ttft']:4d}ms {hit[-1]['out'][:60]}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"base": args.base, "model": args.model,
             "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
             "summary": summary,
             "rows": [{k: v for k, v in r.items()} for c in CELLS for r in res[c]]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ 写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
