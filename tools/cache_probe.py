#!/usr/bin/env python3
"""测前缀缓存到底有没有生效：同一个固定 system prompt 连发 N 次，看首字。

判据（同一个模型、同一个 prompt）：
  · 命中缓存 → 第 2 发起首字明显下降，落在一个「每 token 解码时间」量级
  · 没命中   → 每次都差不多，等于每句都重新 prefill 那几百 token

实测参考：vLLM + 7B 从 533ms 掉到 94ms；vLLM + 35B-A3B 因为 mamba page
把复用粒度顶到 1056 token，命中率恒为 0%，首字一直 400ms。

两个模式：
  warm  前缀完全相同的连续请求（线上真实形态：同一份 system prompt）
  cold  每次在 system prompt 末尾加一个随机串，强制不命中，量「纯 prefill 成本」

用法：
    ./asr-venv/bin/python tools/cache_probe.py                        # 默认 vLLM 8000
    ./asr-venv/bin/python tools/cache_probe.py --mode cold
    ./asr-venv/bin/python tools/cache_probe.py \
        --base http://127.0.0.1:10100/inference/v1 --model <id> \
        --key mag-xxx --n 6 --compare
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import caption_server as cs  # noqa: E402


def one(base: str, model: str, key: str, sysp: str, text: str,
        target: str, max_tokens: int) -> tuple[int, int, str, dict]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": text}],
        "stream": True, "temperature": 0.1, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    ttft, parts, tim = -1, [], {}
    with urllib.request.urlopen(req, timeout=120) as r:
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
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                if ttft < 0:
                    ttft = int((time.time() - t0) * 1000)
                parts.append(delta["content"])
    return ttft, int((time.time() - t0) * 1000), "".join(parts).strip(), tim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--key", default="", help="Bearer token（Magnitude 必填）")
    ap.add_argument("--target", default="ja", choices=["zh", "ja"])
    ap.add_argument("--text", default="中方刻意给韩国好处")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--style", default="full", choices=["full", "short"],
                    help="用哪一档 prompt。短版的固定前缀更短，用来看「前缀太短还会不会命中」")
    ap.add_argument("--mode", default="warm", choices=["warm", "cold", "both"])
    args = ap.parse_args()

    sysp = cs._tr_system(args.target, args.style)
    print(f"base={args.base} model={args.model} system prompt {len(sysp)} 字")

    modes = ["warm", "cold"] if args.mode == "both" else [args.mode]
    med: dict[str, int] = {}
    for mode in modes:
        print(f"\n-- {mode} --")
        ttfts: list[int] = []
        for i in range(1, args.n + 1):
            # cold：每次在末尾塞随机串，前缀必然不命中
            prompt = sysp if mode == "warm" else f"{sysp}\n[{uuid.uuid4().hex}]"
            ttft, total, out, tim = one(args.base, args.model, args.key, prompt,
                                        args.text, args.target, args.max_tokens)
            cn, pn = tim.get("cache_n", -1), tim.get("prompt_n", -1)
            if ttft >= 0:
                ttfts.append(ttft)
            print(f"  第{i}发  ttft={ttft:4d}ms  total={total:5d}ms  "
                  f"cache={cn}/{pn}  {out[:40]}")
        if ttfts:
            # 第 1 发常含建图/加载，中位数去掉它更接近稳态
            body = ttfts[1:] or ttfts
            med[mode] = int(statistics.median(body))
            print(f"  首字中位（跳过第1发）={med[mode]}ms")

    if len(med) == 2:
        print(f"\n前缀复用收益：cold {med['cold']}ms → warm {med['warm']}ms "
              f"（省 {med['cold'] - med['warm']}ms）")
        print("warm 与 cold 差不多 ⇒ 该运行时/模型没有复用固定 system prompt。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
