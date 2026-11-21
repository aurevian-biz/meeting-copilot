#!/usr/bin/env python3
"""语音场景的模型复测：首字延迟 + 单请求 decode + 首句出声。

为什么不看聚合吞吐：个人助理是 1 路、2~4K 上下文、一次一句话。agent run 那套
「40 次调用、30K 上下文、比聚合 tok/s」的口径在这里会得出相反结论（27B 在
批处理上很强，但首句要 3 秒才出声，语音里就是「没反应」）。

三个判据（README 4.14 的硬线）：
  前缀命中 TTFT      < 300 ms
  单请求 decode      >= 20 tok/s
  首句出声（实测）    < 600 ms

注意：不要用「20 token」去折算首句延迟。中文分词约 1.3 字/token，Qwen2.5-7B
实测 26 tok/s 折出来是 767 ms，但真实首句（到第一个句号或 12 字）只要 284 ms。
这里直接测首句，20 token 只作为参考值打印。

用法：
  python tools/llm_probe.py --base http://127.0.0.1:8000/v1 --model qwen
  python tools/llm_probe.py --base http://127.0.0.1:8080/v1 --model local --runs 5

只依赖标准库，任意 OpenAI 兼容端点都能测（vLLM / llama-server / Magnitude）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import uuid

# 接近真实助理的人设 + 工具说明长度（约 600~900 token）
SYSTEM = (
    "你是「小八」，运行在 Jetson AGX Orin 上的个人语音助理。"
    "你通过麦克风听到用户说话，回答会被直接念出来，所以："
    "1) 先说结论，一到两句，不要分点罗列；"
    "2) 不要念 Markdown、括号、链接；"
    "3) 数字和专有名词照原样说，不确定就说不确定；"
    "4) 用户问「屏幕上是什么」时，你会拿到 OCR 文本，只挑与问题相关的部分回答。"
    "你的知识截止到 2026 年，工具包括：查时间、查天气、看屏幕、开关会议字幕、"
    "把 HDMI 音频回放到耳机、记录待办。调用工具前先用一句话说明你要做什么，"
    "不要连续调用超过两个工具。" * 4
)

QUESTION = "用两句话说明你现在能做什么，以及你为什么会慢。"

STOP = "。！？!?\n"


def post_stream(base: str, model: str, messages: list[dict], max_tokens: int,
                usage_out: dict | None = None):
    """流式请求，返回 (ttft_ms, first_sent_ms, decode_tps, tokens, text)。

    usage_out 传入 dict 时会填入 prompt_tokens / completion_tokens（调用方想看上下文大小时用）。
    """
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    t_first = None
    t_sent = None
    text = ""
    tokens = 0
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage") or {}
            if usage.get("completion_tokens"):
                usage_tokens = usage["completion_tokens"]
            if usage_out is not None:
                if usage.get("prompt_tokens"):
                    usage_out["prompt_tokens"] = usage["prompt_tokens"]
                if usage.get("completion_tokens"):
                    usage_out["completion_tokens"] = usage["completion_tokens"]
            for ch in obj.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content") or ""
                if not delta:
                    continue
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                text += delta
                tokens += 1
                if t_sent is None and (any(c in text for c in STOP) or len(text) >= 12):
                    t_sent = now
    t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("没有收到任何 content，检查 base/model")
    ttft_ms = (t_first - t0) * 1000
    first_sent_ms = ((t_sent or t_end) - t0) * 1000
    n = usage_tokens or tokens
    # 首 token 之前的时间不算进 decode（否则 TTFT 会被重复计一次）
    gen_s = max(t_end - t_first, 1e-6)
    decode_tps = max(n - 1, 1) / gen_s
    return ttft_ms, first_sent_ms, decode_tps, n, text


def median(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows]
    return statistics.median(vals), (max(vals) - min(vals))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen")
    p.add_argument("--runs", type=int, default=3, help="每档次数，取中位数（>=3）")
    p.add_argument("--max-tokens", type=int, default=80, help="约等于一句半话")
    p.add_argument("--json", action="store_true", help="只输出 JSON")
    args = p.parse_args()
    runs = max(3, args.runs)

    base_msgs = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": QUESTION}]

    def one(cold: bool) -> dict:
        msgs = json.loads(json.dumps(base_msgs))
        if cold:
            # 前缀加唯一 nonce，逼它重新 prefill（cache_n=0）
            msgs[0]["content"] = f"[req {uuid.uuid4().hex[:8]}]\n" + msgs[0]["content"]
        ttft, sent, tps, n, _ = post_stream(args.base, args.model, msgs, args.max_tokens)
        return {"ttft_ms": ttft, "first_sent_ms": sent, "decode_tps": tps, "tokens": n}

    try:
        one(cold=True)  # 预热：加载模型 / 触发 kernel 调优，不计入结果
    except Exception as e:  # noqa: BLE001
        print(f"预热失败：{e}", file=sys.stderr)
        return 2

    warm, cold = [], []
    for i in range(runs):
        w = one(cold=False)
        warm.append(w)
        print(f"  warm #{i + 1}  ttft {w['ttft_ms']:.0f}ms  "
              f"首句 {w['first_sent_ms']:.0f}ms  {w['decode_tps']:.1f} tok/s  "
              f"({w['tokens']} tok)", file=sys.stderr)
    for i in range(runs):
        c = one(cold=True)
        cold.append(c)
        print(f"  cold #{i + 1}  ttft {c['ttft_ms']:.0f}ms  "
              f"首句 {c['first_sent_ms']:.0f}ms  {c['decode_tps']:.1f} tok/s", file=sys.stderr)

    w_ttft, w_rng = median(warm, "ttft_ms")
    w_sent, s_rng = median(warm, "first_sent_ms")
    w_tps, t_rng = median(warm, "decode_tps")
    c_ttft, _ = median(cold, "ttft_ms")
    first20 = 20 / w_tps * 1000 if w_tps else float("inf")

    verdict = {
        "ttft_ok": w_ttft < 300,
        "decode_ok": w_tps >= 20,
        "first_sentence_ok": w_sent < 600,
    }
    out = {
        "base": args.base,
        "model": args.model,
        "warm": {"ttft_ms": round(w_ttft), "ttft_range_ms": round(w_rng),
                 "first_sentence_ms": round(w_sent),
                 "first_sentence_range_ms": round(s_rng),
                 "decode_tps": round(w_tps, 2), "decode_range": round(t_rng, 2),
                 "first20_tokens_ms": round(first20)},
        "cold": {"ttft_ms": round(c_ttft)},
        "verdict": verdict,
        "pass": all(verdict.values()),
        "runs": {"warm": warm, "cold": cold},
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0 if out["pass"] else 1

    print()
    print(f"模型 {args.model} @ {args.base}（{runs} 次中位数）")
    print(f"  前缀命中 TTFT   {w_ttft:6.0f} ms   (极差 {w_rng:.0f})   判据 <300   "
          f"{'PASS' if verdict['ttft_ok'] else 'FAIL'}")
    print(f"  冷 prefill TTFT {c_ttft:6.0f} ms")
    print(f"  单请求 decode   {w_tps:6.1f} tok/s (极差 {t_rng:.2f})   判据 >=20   "
          f"{'PASS' if verdict['decode_ok'] else 'FAIL'}")
    print(f"  首句出声         {w_sent:6.0f} ms   (极差 {s_rng:.0f}，到首个句号或 12 字)   "
          f"判据 <600   {'PASS' if verdict['first_sentence_ok'] else 'FAIL'}")
    print(f"  参考：20 token   {first20:6.0f} ms   （不是判据，token≠字，别拿它折算）")
    print(f"\n结论：{'可以进语音热路径' if out['pass'] else '不适合热路径，降档或只用于冷路径'}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
