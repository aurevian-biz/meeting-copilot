#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Translation A/B of two LLMs: the same real sentences, the same production prompt, the same request shape.

This compares **models**, not prompts, so it calls the production `caption_server._tr_system()` directly
together with the production request body (stream=True / temperature 0.1 / max_tokens 200). For prompt comparisons
use `tools/tr_quality.py`; do not mix the two tools.

Usage (in the deployment directory, the LLM must be up):

    # Capture a baseline while vLLM 7B is still running
    ./asr-venv/bin/python tools/ab_translate.py --tag A --out /tmp/ab_A.json
    # After the model switch
    ./asr-venv/bin/python tools/ab_translate.py --tag B --out /tmp/ab_B.json
    # Put them side by side
    ./asr-venv/bin/python tools/ab_translate.py --compare /tmp/ab_A.json /tmp/ab_B.json

Only two automatic criteria are **reliable**; the rest is left to the eye:
  · target Japanese and "not a single kana" -> the whole sentence was copied (particles/endings always carry kana, zero false positives)
  · target Chinese and "kana shows up" -> source language leftover (proper nouns such as 深セン false-positive, flagged separately)
Kanji share is reference only: a normal Japanese translation runs about 35-50% kanji, while copying goes above 85%.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caption_server as cs  # noqa: E402

KANA = re.compile(r"[\u3040-\u30FF]")
CJK = re.compile(r"[\u4e00-\u9FFF]")
LATIN = re.compile(r"[A-Za-z]{2,}")

# (target language, source, context) -- the first 10 are bad outputs caught in past measurements, the last 4 come from 62s of real audio
CASES = [
    ("zh", "身体面にアメリカは坚持するべきです", ""),
    ("zh", "深zのアプリで何点给深zに評価するか", ""),
    ("zh", "定め好了、王毅は中国大陸の外務である", ""),
    ("zh", "台湾問題を取引材料にするのか", ""),
    ("zh", "中方科意、韓国に良いところを与える", "日本は冷遇されている"),
    ("ja", "我是主持人谭英文", ""),
    ("ja", "中方刻意给韩国好处", ""),
    ("ja", "所以去扯什么台湾会不会进行交易筹码？", ""),
    ("ja", "这个都没有啊，怎么可能第一次就有结论，那不可能的事。", ""),
    ("ja", "三强会晤，三角的峰会我不认为。", "中方刻意的给了韩国好处"),
    ("ja", "他比较晚了，有开了等一长会议又开了开了会之后才有。", ""),
    ("ja", "这就投扯太远的了，根本不可能在这种地方提出了。", ""),
    ("ja", "了不起在中美会晤中国大陆再讲一下啊，我认为就是这样了。", ""),
    ("ja", "因为他的很容易跟韩国做对比是。", ""),
]


def call_once(base: str, model: str, text: str, target: str, ctx: str,
              think: bool, key: str = "") -> tuple[str, int, int]:
    """Send one streaming request; returns (translation, first token ms, total ms). The shape matches caption_server."""
    sysp = cs._tr_system(target)
    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    body = {
        "model": model,
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": user}],
        "stream": True, "temperature": 0.1, "max_tokens": 200,
    }
    if not think:
        # In the Qwen3.5 template enable_thinking defaults to on; only turning it off keeps the 200 tokens
        # from burning entirely in thinking and leaving the translation blank. The Qwen2.5 template ignores this variable, so passing it is harmless.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"  # Magnitude's /inference/v1 requires auth
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers=headers)

    parts: list[str] = []
    t0 = time.time()
    ttft = -1
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if ttft < 0:
                        ttft = int((time.time() - t0) * 1000)
                    parts.append(piece)
    except Exception as e:  # noqa: BLE001
        return f"<ERROR {e}>", -1, int((time.time() - t0) * 1000)
    return "".join(parts).strip().strip('"“”「」 '), ttft, int((time.time() - t0) * 1000)


def metrics(out: str, target: str) -> dict:
    kana = len(KANA.findall(out))
    cjk = len(CJK.findall(out))
    latin = len(LATIN.findall(out))
    chars = len(re.sub(r"\s", "", out))
    return {
        "kana": kana, "cjk": cjk, "latin": latin, "chars": chars,
        "han_ratio": round(cjk / chars, 2) if chars else 0.0,
        # Whole sentence copied: not a single kana in the Japanese (proper nouns may false-positive, but very rarely)
        "suspect_copy": target == "ja" and kana == 0 and cjk >= 6,
        # Source language leftover: kana surfacing in the Chinese translation (proper nouns like 深セン false-positive)
        "kana_left": target == "zh" and kana > 0,
    }


def run(base: str, model: str, tag: str, think: bool, repeat: int, out: str,
        key: str = "") -> int:
    print(f"# tag={tag} model={model} think={'on' if think else 'off'} repeat={repeat}")
    # Warm-up: the first call builds graphs and prefills, so it is untimed; it also heats the prefix cache
    for target in ("zh", "ja"):
        call_once(base, model, "テスト" if target == "zh" else "测试", target, "", think, key)

    rows = []
    for i, (target, text, ctx) in enumerate(CASES, 1):
        runs = [call_once(base, model, text, target, ctx, think, key) for _ in range(repeat)]
        ok = [r for r in runs if r[1] >= 0]
        text_out = runs[-1][0]
        ttfts = [r[1] for r in ok] or [-1]
        totals = [r[2] for r in ok] or [-1]
        m = metrics(text_out, target)
        row = {"i": i, "target": target, "src": text, "ctx": ctx, "out": text_out,
               "ttft_ms": int(statistics.median(ttfts)),
               "total_ms": int(statistics.median(totals)), **m}
        rows.append(row)
        flag = "抄?" if m["suspect_copy"] else ("残假名" if m["kana_left"] else "ok")
        print(f"[{i:02d}/{target}] {row['ttft_ms']:4d}/{row['total_ms']:5d}ms {flag:6s} "
              f"汉{m['han_ratio']:.2f}  {text_out}")

    payload = {"tag": tag, "model": model, "think": think, "base": base,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    ja = [r for r in rows if r["target"] == "ja"]
    zh = [r for r in rows if r["target"] == "zh"]
    print(f"\n-- {tag} 汇总 --")
    print(f"日语组 {len(ja)} 句：疑似整句照抄 {sum(r['suspect_copy'] for r in ja)} 句，"
          f"汉字率中位 {statistics.median([r['han_ratio'] for r in ja]):.2f}")
    print(f"中文组 {len(zh)} 句：残留假名 {sum(r['kana_left'] for r in zh)} 句")
    print(f"首字 中位 {int(statistics.median([r['ttft_ms'] for r in rows if r['ttft_ms'] >= 0]))}ms"
          f"   全文 中位 {int(statistics.median([r['total_ms'] for r in rows]))}ms")
    print(f"→ 写入 {out}")
    return 0


def compare(files: list[str]) -> int:
    data = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    print(f"\n{'=' * 78}")
    for i, d in enumerate(data):
        print(f"{d['tag']}: model={d['model']} think={d['think']} @ {d['ts']}")

    for idx in range(len(data[0]["rows"])):
        rows = [d["rows"][idx] for d in data]
        src = rows[0]["src"]
        ctx = rows[0]["ctx"]
        print(f"\n{'-' * 78}\n[{idx + 1:02d}/{rows[0]['target']}] {src}"
              + (f"    上文：{ctx}" if ctx else ""))
        for d, r in zip(data, rows):
            flag = "抄?" if r["suspect_copy"] else ("残假名" if r["kana_left"] else "ok")
            print(f"  {d['tag']:3s} {r['ttft_ms']:4d}/{r['total_ms']:5d}ms 汉{r['han_ratio']:.2f} "
                  f"{flag:6s} {r['out']}")

    print(f"\n{'=' * 78}\n{'指标':<22}" + "".join(f"{d['tag']:>16}" for d in data))
    for name, fn in [
        ("日语照抄句数", lambda rs: sum(r["suspect_copy"] for r in rs if r["target"] == "ja")),
        ("中文残留假名句数", lambda rs: sum(r["kana_left"] for r in rs if r["target"] == "zh")),
        ("日语汉字率中位", lambda rs: round(statistics.median(
            [r["han_ratio"] for r in rs if r["target"] == "ja"]), 2)),
        ("首字中位(ms)", lambda rs: int(statistics.median(
            [r["ttft_ms"] for r in rs if r["ttft_ms"] >= 0]))),
        ("全文中位(ms)", lambda rs: int(statistics.median([r["total_ms"] for r in rs]))),
        ("输出字数中位", lambda rs: int(statistics.median([r["chars"] for r in rs]))),
    ]:
        print(f"{name:<22}" + "".join(f"{fn(d['rows']):>16}" for d in data))
    print("\n汉字率：正常日译 0.35~0.50，照抄会到 0.85 以上；两臂都高于 0.8 说明都在抄。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--tag", help="这一臂的名字，如 A / B / B-think")
    ap.add_argument("--out", help="结果 json 路径")
    ap.add_argument("--repeat", type=int, default=3, help="每句重复几次取中位（默认 3）")
    ap.add_argument("--think", choices=["on", "off"], default="off")
    ap.add_argument("--key", default="", help="Bearer token（Magnitude 必填）")
    ap.add_argument("--compare", nargs="+", metavar="JSON")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare)
    if not args.tag:
        ap.error("要么给 --tag 抓一臂，要么给 --compare 比较")
    out = args.out or f"/tmp/ab_{args.tag}.json"
    return run(args.base, args.model, args.tag, args.think == "on", args.repeat, out,
               args.key)


if __name__ == "__main__":
    sys.exit(main())
