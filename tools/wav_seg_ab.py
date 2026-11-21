#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""One real audio clip, one batch of cut segments, several engines compared head to head: first token / full text / quality.

Why not `ab_translate.py`: that uses 14 fixed-length sentences, mostly short ones (8-20 tokens),
so MoE's prefill disadvantage is underestimated and dense's advantage overestimated. Here the segments come from real audio, and the
length distribution is the production distribution, which is what it takes to answer "9B or 35B".

Four design constraints (all learned the hard way):
- **run ASR once only**; every arm reuses the same text and the same [context]. Otherwise it compares ASR jitter.
- **warm up each arm once first**: the first request builds graphs + fills the prefix cache, and skipping the warm-up inflates p50.
- **run each arm through all of its segments consecutively, never interleaved**: switching the prefix cache back and forth
  between two system prompts invalidates all of it (measured: one interleave drops `cache_n` from 290 to 0).
- **judge by `prompt_n - cache_n`** (the token count the engine really processed this round); the first token is merely a function of it.

Usage (run each arm in turn):

    ./asr-venv/bin/python tools/wav_seg_ab.py --wav /tmp/meet60.wav \
        --arm "35B|http://127.0.0.1:10100/inference/v1|hf:unsloth/...gguf|mag-xxx" \
        --arm "9B|http://127.0.0.1:10100/inference/v1|qwen3.5-9b:gguf:q4|mag-xxx" \
        --ctx-lines 2 --out /tmp/wav_ab.json

When only quality matters and speed does not, there is no point turning off `--no-timing` -- there is only one ASR pass anyway.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import caption_server as cs  # noqa: E402
from ab_translate import metrics  # noqa: E402

ASR_DIR = ("/opt/meeting-copilot/data/asr-models/"
           "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")


def segment(wav: str, ns: argparse.Namespace) -> list[np.ndarray]:
    """Cut the wav into segments with the same segmentation parameters as production. Append 1s of silence at the end, otherwise the last sentence never flushes."""
    audio, sr = cs.vp.read_wav(wav)
    if sr != cs.SR:
        n = int(round(audio.size * cs.SR / sr))
        audio = np.interp(np.linspace(0, audio.size - 1, n),
                          np.arange(audio.size), audio)
    seg = cs.Segmenter(margin_db=ns.margin, hold_s=ns.hold, max_s=ns.max_s,
                       min_s=ns.min_s, soft_s=ns.soft_cut, **cs.seg_tuning(ns))
    audio = np.concatenate([audio, np.zeros(cs.SR, dtype=audio.dtype)])
    out = []
    for i in range(0, audio.size - cs.FRAME_N, cs.FRAME_N):
        frame = audio[i:i + cs.FRAME_N]
        db = cs.vp.db_of(frame.astype(np.float64))
        pcm = seg.push(frame, db)
        if pcm is not None and pcm.size:
            out.append(pcm)
    return out


def parse_arms(specs: list[str]) -> list[dict]:
    arms = []
    for s in specs:
        parts = s.split("|")
        if len(parts) < 3:
            raise SystemExit(f"--arm 格式：名字|base|model[|key]，收到 {s!r}")
        name, base, model = parts[0], parts[1], parts[2]
        key = parts[3] if len(parts) > 3 else ""
        arms.append({"name": name, "base": base, "model": model, "key": key})
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="名字|base|model|key")
    ap.add_argument("--ctx-lines", type=int, default=2)
    ap.add_argument("--tr-prompt", default="full", choices=["full", "short"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--max-s", type=float, default=6.0)
    ap.add_argument("--hold", type=float, default=0.45)
    ap.add_argument("--min-s", type=float, default=0.25)
    ap.add_argument("--soft-cut", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=8.0)
    ap.add_argument("--hold-long", type=float, default=0.90)
    ap.add_argument("--hold-short", type=float, default=0.30)
    ap.add_argument("--seg-mid", type=float, default=1.20)
    ap.add_argument("--seg-long", type=float, default=2.40)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="每段之间歇这么久（秒）。模拟线上 3~6 秒来一句的节奏；"
                         "0 = 连着打，量的是连续负载下的表现")
    ap.add_argument("--out")
    ns = ap.parse_args()

    cs.TR_STYLE = ns.tr_prompt
    arms = parse_arms(ns.arm)
    if not arms:
        ap.error("至少给一个 --arm")

    print(f"[1/3] 切段 {ns.wav}", flush=True)
    segs = segment(ns.wav, ns)
    print(f"      {len(segs)} 段，有声总长 {sum(s.size for s in segs) / cs.SR:.1f}s")

    print(f"[2/3] ASR（只跑一次，所有臂共用）", flush=True)
    asr = cs.Asr(Path(ASR_DIR), threads=ns.threads, language=ns.lang)
    cap = cs.Captioner(asr, bilingual=False)
    t0 = time.time()
    items = []
    for pcm in segs:
        text, lang = cap.recognize(pcm)
        if not text:
            continue
        target = "ja" if lang == "zh" else ("zh" if lang == "ja" else None)
        items.append({"text": text, "lang": lang, "target": target,
                      "dur": round(pcm.size / cs.SR, 2)})
    print(f"      {len(items)} 条有内容，ASR 共 {time.time() - t0:.1f}s")
    if not items:
        return 1

    # Context is assembled in ASR text order, identical for every arm
    for i, it in enumerate(items):
        prev = [x["text"] for x in items[max(0, i - ns.ctx_lines):i]] if ns.ctx_lines else []
        it["ctx"] = "".join(prev)

    print(f"[3/3] 翻译：{len(arms)} 臂 × {sum(1 for x in items if x['target'])} 句", flush=True)
    results = {}
    for arm in arms:
        print(f"\n-- {arm['name']}  {arm['model']}", flush=True)
        if arm["key"]:
            cs._LLM_KEY = arm["key"]
        # Warm-up: fire one request with the same system prompt to build graphs + heat the prefix cache
        cs.translate_stream(arm["base"], arm["model"], "熱身", "zh", timeout=120)
        rows = []
        for i, it in enumerate(items, 1):
            if not it["target"]:
                rows.append({**it, "skip": True, "trans": "", "ttft": -1, "total": -1})
                continue
            trans, ttft, total = cs.translate_stream(
                arm["base"], arm["model"], cs.strip_fillers(it["text"]), it["target"],
                timeout=120, ctx=cs.strip_fillers(it["ctx"]))
            tim = dict(cs.LAST_TIMINGS or {})
            rows.append({**it, "skip": False, "trans": trans, "ttft": ttft,
                         "total": total,
                         "prompt_n": tim.get("prompt_n"), "cache_n": tim.get("cache_n"),
                         "chars": len(it["text"].replace(" ", ""))})
            print(f"  [{i:02d}/{it['target']}] {len(it['text'].replace(' ', '')):3d}字 "
                  f"新token={str(tim.get('prompt_n')):>4} "
                  f"{ttft:4d}/{total:5d}ms  {trans[:44]}", flush=True)
            if ns.delay:
                time.sleep(ns.delay)
        ok = [r for r in rows if not r.get("skip") and r["ttft"] >= 0]
        tt = sorted(r["ttft"] for r in ok)
        tot = sorted(r["total"] for r in ok)
        m = [metrics(r["trans"], r["target"]) for r in ok]
        summ = {
            "arm": arm["name"], "model": arm["model"], "n": len(ok),
            "ttft_p50": int(statistics.median(tt)) if tt else -1,
            "ttft_p90": tt[min(len(tt) - 1, int(len(tt) * 0.9))] if tt else -1,
            "total_p50": int(statistics.median(tot)) if tot else -1,
            "chars_p50": int(statistics.median([r["chars"] for r in ok])) if ok else 0,
            # For Magnitude, prompt_n **is the token count really processed this round** (cache_n is the reused part,
            # and only the sum of the two is the whole prompt). So new tokens = prompt_n; do not subtract again.
            "new_tok_p50": int(statistics.median([r["prompt_n"] for r in ok])) if ok else -1,
            "suspect_copy": sum(x["suspect_copy"] for x in m),
            "kana_left": sum(x["kana_left"] for x in m),
            "empty": sum(1 for r in ok if not r["trans"]),
        }
        results[arm["name"]] = {"summary": summ, "rows": rows}
        print(f"  首字 p50 {summ['ttft_p50']}ms / p90 {summ['ttft_p90']}ms  "
              f"全文 p50 {summ['total_p50']}ms  新token p50 {summ['new_tok_p50']}  "
              f"照抄 {summ['suspect_copy']} 残假名 {summ['kana_left']} 空 {summ['empty']}",
              flush=True)

    print(f"\n{'臂':<10}{'首字p50':>9}{'首字p90':>9}{'全文p50':>9}{'新token':>9}"
          f"{'照抄':>6}{'残假名':>7}{'空':>4}")
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<10}{s['ttft_p50']:>9}{s['ttft_p90']:>9}{s['total_p50']:>9}"
              f"{s['new_tok_p50']:>9}{s['suspect_copy']:>6}{s['kana_left']:>7}{s['empty']:>4}")

    # Line-by-line comparison: only the same line reveals a quality difference
    if len(results) > 1:
        print(f"\n-- 逐句对照 --")
        names = list(results)
        for i, it in enumerate(items, 1):
            if not it["target"]:
                continue
            print(f"\n[{i:02d}] {it['text']}")
            for nm in names:
                r = results[nm]["rows"][i - 1]
                print(f"  {nm:<6}{r['ttft']:4d}/{r['total']:5d}ms {r['trans'][:70]}")

    if ns.out:
        Path(ns.out).write_text(json.dumps(
            {"wav": ns.wav, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
             "ctx_lines": ns.ctx_lines, "tr_prompt": ns.tr_prompt,
             "arms": results}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ 写入 {ns.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
