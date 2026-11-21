#!/usr/bin/env python3
"""对比 vLLM 翻译的「非流式整句」与「流式首字」延迟。

用字幕服务里的真实句子和真实 system prompt，测三种指标：
  total    非流式：从发出到拿到整句
  ttft     流式：从发出到第一个 token
  t_all    流式：从发出到流结束

判据：流式相对非流式的收益 = total - ttft（用户看到第一个字提前了多少）。
"""
import argparse
import json
import re
import sys
import time
import urllib.request

sys.path.insert(0, "/opt/meeting-copilot")

# 真实句子（取自 8096 的 /api/poll，seq 911~916）
CASES = [
    ("短句", "元首和平宣言。", ""),
    ("中句", "巨头的会面哦，看起来呢。", "元首和平宣言。"),
    ("中句2", "他们有达成一些共识，包含是设。", "巨头的会面哦，看起来呢。 / 元首和平宣言。"),
    ("长句", "及了乌克兰跟伊朗的议题，那也。", "他们有达成一些共识，包含是设。"),
    ("长句2", "发表了和平宣言承诺和。", "及了乌克兰跟伊朗的议题，那也。"),
    ("偏长", "对话好，那我们看到中二美。", "发表了和平宣言承诺和。"),
]

SYS = ("你是会议同传。把用户给的这句话翻译成日语。\n"
       "硬性要求：\n"
       "1. 只输出译文本身，不要引号、不要解释、不要补充，不要输出原文；\n"
       "2. 译文必须是纯日语，一个原文词都不许留（专有名词除外）；\n"
       "3. 原文是语音识别结果，可能有错字、断句不完整，请按最合理的语义翻译，"
       "半句就译成半句，不要自行补全；\n"
       "4. 如果给了【上文】，译文要与上文的人称、术语、语气保持一致；\n"
       "5. 原文里的语气词（嗯、呃、えーと、あの、um 等）和说到一半的重来直接忽略；"
       "但有实义的词要保留。")


def call(base, model, text, ctx, stream, timeout=30):
    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": user}],
        "stream": stream, "temperature": 0.1, "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        out = (d["choices"][0]["message"]["content"] or "").strip()
        return (time.time() - t0) * 1000, None, out

    ttft = None
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            delta = ((ev.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                if ttft is None:
                    ttft = (time.time() - t0) * 1000
                parts.append(delta)
    return (time.time() - t0) * 1000, ttft, "".join(parts).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    print(f"{'用例':6} {'非流式':>8} {'流式首字':>9} {'流式全':>8} {'提前':>7}  译文")
    print("-" * 92)
    rows = []
    for name, text, ctx in CASES:
        for _ in range(args.rounds):
            t_total, _, out_ns = call(args.base, args.model, text, ctx, False)
            t_all, t_ttft, out_s = call(args.base, args.model, text, ctx, True)
        gain = t_total - t_ttft
        rows.append((name, t_total, t_ttft, t_all, gain))
        print(f"{name:6} {t_total:7.0f}ms {t_ttft:8.0f}ms {t_all:7.0f}ms {gain:+6.0f}ms  {out_s[:44]}")

    print("-" * 92)
    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"{'均值':6} {avg([r[1] for r in rows]):7.0f}ms "
          f"{avg([r[2] for r in rows]):8.0f}ms {avg([r[3] for r in rows]):7.0f}ms "
          f"{avg([r[4] for r in rows]):+6.0f}ms")
    # 与生产实测对照
    print("\n生产 8096 实测（916 段）：asr 116~174ms，trans 296~1210ms，lag 117ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
