#!/usr/bin/env python3
"""同一段真实音频、同一批切段，多个引擎正面比：首字 / 全文 / 质量。

为什么不用 `ab_translate.py`：那个用 14 条定长句子，短句（8~20 token）占多数，
MoE 的 prefill 劣势被低估、dense 的优势被高估。这里用真实音频切出来的段，
长度分布就是线上分布，才能回答「9B 还是 35B」。

四条设计约束（都是被坑出来的）：
- **ASR 只跑一次**，多臂共用同一份文本和同一份【上文】。否则比的是 ASR 抖动。
- **每臂先热身一发**：首次请求要建图 + 填前缀缓存，不热身会把 p50 拉高。
- **每臂连续跑完自己的全部段，不要交替**：前缀缓存在两套 system prompt 之间
  来回切换会全部失效（实测一交替 `cache_n` 就从 290 掉到 0）。
- **判据看 `prompt_n - cache_n`**（引擎本轮真处理的 token 数），首字只是它的函数。

用法（多臂各跑一遍）：

    ./asr-venv/bin/python tools/wav_seg_ab.py --wav /tmp/meet60.wav \
        --arm "35B|http://127.0.0.1:10100/inference/v1|hf:unsloth/...gguf|mag-xxx" \
        --arm "9B|http://127.0.0.1:10100/inference/v1|qwen3.5-9b:gguf:q4|mag-xxx" \
        --ctx-lines 2 --out /tmp/wav_ab.json

只看质量不比速度时，把 `--no-timing` 关掉没必要——反正只有一次 ASR。
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
    """按线上同一套断句参数把 wav 切成段。末尾补 1 秒静音，否则最后一句不 flush。"""
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

    # 上文按 ASR 文本顺序拼，所有臂完全一致
    for i, it in enumerate(items):
        prev = [x["text"] for x in items[max(0, i - ns.ctx_lines):i]] if ns.ctx_lines else []
        it["ctx"] = "".join(prev)

    print(f"[3/3] 翻译：{len(arms)} 臂 × {sum(1 for x in items if x['target'])} 句", flush=True)
    results = {}
    for arm in arms:
        print(f"\n-- {arm['name']}  {arm['model']}", flush=True)
        if arm["key"]:
            cs._LLM_KEY = arm["key"]
        # 热身：同一份 system prompt 打一发，建图 + 让前缀缓存热起来
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
            # Magnitude 的 prompt_n **就是本轮真处理的 token 数**（cache_n 是复用的那部分，
            # 两者相加才是总 prompt）。所以新 token = prompt_n，别再相减。
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

    # 逐句对照：只有同一句才看得出质量差
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
