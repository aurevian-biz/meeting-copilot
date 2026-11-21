#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""A/B of screen hints against the translation: the same batch of real segments, translated once without and once with the screen term hints.

Two questions must both be answered:
  1. **cost**: stuff in N extra characters and how many milliseconds does the first token gain (prefill is billed per token on 35B MoE)
  2. **effect**: did the translation change at all? And when it changed, was it fixed or made dirty?

Measuring only cost is self-deception -- if the screen term hints never influence a single line, they are pure waste.
Measuring only the effect is no good either -- the caption first-token budget is in milliseconds.

Usage (on the Jetson):
  curl -s localhost:8096/api/poll > /tmp/poll.json
  curl -s -X POST localhost:8095/api/look > /tmp/ocr.json
  ../asr-venv/bin/python tools/screen_hint_ab.py --poll /tmp/poll.json \\
      --ocr /tmp/ocr.json --limit 12 --chars 60,120 \\
      --base http://127.0.0.1:10100/inference/v1 \\
      --model 'hf:unsloth/...gguf' --key mag-xxx
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, "/opt/meeting-copilot")
import caption_server as cs  # noqa: E402

# Text from our own caption page. The capture card films "the presenter's screen", and the page is often open right there;
# without filtering it, the previous line's Chinese translation is fed back to the model as screen content -- a feedback loop that drifts further with every line.
UI_WORDS = (
    "会议字幕", "设备", "已识别", "延迟", "首字", "译完", "双语", "正在听",
    "停止", "清屏", "正在说", "运行中", "搜索", "创建", "Chrome", "YouTube",
    "youtube.com", "dB",
)

KATA = re.compile(r"[ァ-ヴー]{3,12}")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9._\-]{1,15}")
NUM = re.compile(r"[0-9]{2,4}")


def ocr_lines(ocr: dict) -> list[str]:
    text = (ocr or {}).get("text") or ""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def filter_lines(lines: list[str], recent: list[str]) -> list[str]:
    """Drop the page UI and the lines that are already our own translation."""
    out = []
    for ln in lines:
        if len(ln) < 3:
            continue
        if any(w in ln for w in UI_WORDS):
            continue
        # 8+ characters in common with the latest translation/source -> assumed copied from our own page
        junk = False
        for r in recent:
            r = re.sub(r"\s+", "", r)
            if len(r) < 8:
                continue
            for i in range(0, len(r) - 7):
                if r[i:i + 8] in ln:
                    junk = True
                    break
            if junk:
                break
        if junk:
            continue
        out.append(ln)
    return out


def build_hint(lines: list[str], max_chars: int) -> str:
    """Squeeze the screen text into one hint block. Picked by "information density", not stacked in source order.

    Priority: lines with Latin letters/digits (proper nouns, model numbers) > lines with katakana > the rest.
    Those are exactly what ASR mishears most and what the screen can best correct.
    """
    scored = []
    for ln in lines:
        s = 0
        if LATIN.search(ln):
            s += 3
        if KATA.search(ln):
            s += 2
        if NUM.search(ln):
            s += 1
        if 6 <= len(ln) <= 40:
            s += 1
        scored.append((s, ln))
    scored.sort(key=lambda kv: -kv[0])

    picked, used = [], 0
    for _s, ln in scored:
        if used + len(ln) > max_chars:
            continue
        picked.append(ln)
        used += len(ln)
    if not picked:
        return ""
    return ("【屏幕文字·仅用于纠正读音相近的专有名词】\n"
            + "\n".join(picked)
            + "\n不要把这段文字当原文翻译，也不要引入本句没有的内容。")


def translate(base: str, model: str, text: str, ctx: str, hint: str) -> dict:
    t0 = time.time()
    out, ttft, total = cs.translate_stream(base, model, text, "zh",
                                           timeout=20.0, ctx=ctx, hint=hint)
    tim = dict(cs.LAST_TIMINGS or {})
    return {"trans": out, "ttft": ttft, "total": total,
            "prompt_n": tim.get("prompt_n"), "cache_n": tim.get("cache_n"),
            "prompt_ms": tim.get("prompt_ms"), "wall": int((time.time() - t0) * 1000)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", default="", help="/api/poll 的 json 文件")
    ap.add_argument("--ocr", default="", help="/api/look 的 json 文件")
    ap.add_argument("--ocr-text", default="", help="直接给屏幕纯文本文件")
    ap.add_argument("--base", default="http://127.0.0.1:10100/inference/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--key", default="")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--chars", default="60,120", help="屏幕词表的字数上限，逗号分隔")
    ap.add_argument("--ctx-lines", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.model:
        sys.exit("--model 必填")
    cs._LLM_KEY = args.key

    lines = json.load(open(args.poll, encoding="utf-8"))["lines"]
    recent = []
    for l in lines[-30:]:
        recent.append(l.get("trans") or "")
        recent.append(l.get("text") or "")

    if args.ocr_text:
        raw_lines = [x.strip() for x in open(args.ocr_text, encoding="utf-8") if x.strip()]
    else:
        raw_lines = ocr_lines(json.load(open(args.ocr, encoding="utf-8")))
    kept = filter_lines(raw_lines, recent)
    print(f"屏幕行：原始 {len(raw_lines)} → 滤掉 UI/自身译文后 {len(kept)}")
    for x in kept[:12]:
        print("   ", x[:70])

    items = []
    for l in lines:
        if not (l.get("trans") or "").strip():
            continue          # segments without a translation do not take part, so "empty" is not treated as a baseline
        items.append(l)
    items = items[-args.limit:]
    print(f"\n参与对比的段：{len(items)} 条")

    caps = [int(x) for x in args.chars.split(",") if x.strip()]
    arms = [("none", 0)] + [(f"h{c}", c) for c in caps]
    hints = {name: (build_hint(kept, c) if c else "") for name, c in arms}
    for name, _c in arms:
        h = hints[name]
        print(f"\n=== {name}: {len(h)} 字 / 约 {len(h) // 2} token")
        if h:
            print(h.replace("\n", " ⏎ ")[:300])

    rows: dict[str, list[dict]] = {n: [] for n, _ in arms}
    for r in range(args.rounds):
        for name, _c in arms:      # run each arm to completion: interleaving keeps invalidating the prefix cache
            for i, l in enumerate(items):
                idx = lines.index(l)
                prev = [lines[j] for j in range(max(0, idx - args.ctx_lines), idx)]
                ctx = "".join(x["text"] for x in prev)
                res = translate(args.base, args.model, l["text"], ctx, hints[name])
                res.update({"i": i, "seq": l["seq"], "src": l["text"],
                            "base_trans": l.get("trans") or "", "arm": name,
                            "round": r})
                rows[name].append(res)
                print(f"  [{name} r{r} {i + 1}/{len(items)}] "
                      f"{res['ttft']:4d}/{res['total']:5d}ms "
                      f"pn={res['prompt_n']} cn={res['cache_n']} "
                      f"| {res['trans'][:40]}", flush=True)

    print("\n########## 汇总")
    print(f"{'臂':<6}{'字数':>6}{'首字p50':>9}{'全文p50':>9}{'prompt_n':>10}{'prompt_ms':>11}")
    for name, _c in arms:
        rs = rows[name]
        t = [x["ttft"] for x in rs]
        print(f"{name:<6}{len(hints[name]):>6}{int(statistics.median(t)):>9}"
              f"{int(statistics.median([x['total'] for x in rs])):>9}"
              f"{int(statistics.median([x['prompt_n'] or 0 for x in rs])):>10}"
              f"{int(statistics.median([x['prompt_ms'] or 0 for x in rs])):>11}")

    # Effect: compare line by line against none
    base_by = {(x["round"], x["i"]): x for x in rows["none"]}
    print("\n########## 效果（每臂 vs none，同一轮同一句）")
    for name, _c in arms[1:]:
        same = diff = 0
        for x in rows[name]:
            b = base_by.get((x["round"], x["i"]))
            if not b:
                continue
            if x["trans"].strip() == b["trans"].strip():
                same += 1
            else:
                diff += 1
                print(f"  [{x['seq']}] 原句 {x['src'][:34]}")
                print(f"      none: {b['trans'][:70]}")
                print(f"      {name:>4}: {x['trans'][:70]}")
        print(f"  {name}: 相同 {same} / 有差异 {diff}")

    if args.out:
        json.dump({"hints": hints,
                   "arms": {n: rows[n] for n, _ in arms}},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n明细写入 {args.out}")


if __name__ == "__main__":
    main()
