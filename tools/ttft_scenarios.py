#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Sweep first token latency (TTFT) across the prompt shapes a real assistant sees.

How it differs from llm_probe.py: that one uses a fixed persona and answers "can this model be used at all";
this one feeds voice_server's real system prompt and sweeps four realistic contexts,
answering "under what conditions does the first token get slower, and does it cross 300ms".

Four scenarios:
  short   system + 1 user sentence (about the everyday one-question exchange)
  hist6   system + 6 turns of history (voice_server takes history[-6:])
  screen  system + 2000 chars of OCR text (reading the screen)
  screen4k system + 4000 chars of OCR text (long document / dense small print)

Each runs two ways:
  warm  repeat the same prompt, prefix hits (the real shape of day-to-day multi-turn)
  cold  prepend a unique nonce to the prompt, forcing a full recompute (worst case)

Usage:
  python tools/ttft_scenarios.py --base http://127.0.0.1:8000/v1 --model qwen
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_probe import post_stream  # noqa: E402  reuse the same streaming measurement

OCR_SAMPLE = (
    "会議資料 2026年度 第3四半期 進捗報告 議題 1 開発状況 2 品質指標 3 リスクと対策 "
    "添付資料あり 次回会議 11月27日 15時 会議室B 参加者 田中 鈴木 佐藤 高橋 "
    "画面右下 通知 3件 Slack メール カレンダー 予定 16時30分 レビュー "
)


def real_system() -> str:
    """Fetch the real system prompt from voice_server so the two copies of the text cannot drift apart."""
    from voice_server import Llm

    return Llm("http://127.0.0.1:8000/v1", "qwen").system


def build(scene: str, system: str) -> list[dict]:
    if scene == "short":
        return [{"role": "system", "content": system},
                {"role": "user", "content": "今日の天気は？"}]
    if scene == "hist6":
        hist = []
        for i in range(3):
            hist.append({"role": "user", "content": f"{i + 1}点目の質問です。字幕の遅れはどれくらい？"})
            hist.append({"role": "assistant", "content": "だいたい1秒から2秒です。原文が先に出て、訳文が少し遅れて付きます。"})
        hist.append({"role": "user", "content": "じゃあ会議で使える？"})
        return [{"role": "system", "content": system}] + hist
    if scene == "screen":
        ocr = (OCR_SAMPLE * 12)[:2000]
        return [{"role": "system", "content": system},
                {"role": "user", "content": "[屏幕文字]\n" + ocr + "\n画面について教えて"}]
    if scene == "screen4k":
        ocr = (OCR_SAMPLE * 25)[:4000]
        return [{"role": "system", "content": system},
                {"role": "user", "content": "[屏幕文字]\n" + ocr + "\n画面について教えて"}]
    raise SystemExit("未知场景：" + scene)


def run(base: str, model: str, msgs: list[dict], cold: bool) -> dict:
    if cold:
        msgs = json.loads(json.dumps(msgs))
        import uuid
        msgs[0]["content"] = f"[req {uuid.uuid4().hex[:8]}]\n" + msgs[0]["content"]
    usage: dict = {}
    ttft, sent, tps, n, text = post_stream(base, model, msgs, 160, usage)
    return {"ttft_ms": ttft, "first_sent_ms": sent, "decode_tps": tps, "tokens": n,
            "chars": len(text), "prompt_tokens": usage.get("prompt_tokens", 0)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    runs = max(3, args.runs)

    system = real_system()
    print(f"真实 system prompt：{len(system)} 字", file=sys.stderr)
    scenes = ["short", "hist6", "screen", "screen4k"]
    rows = {}
    for scene in scenes:
        msgs = build(scene, system)
        try:
            run(args.base, args.model, msgs, cold=True)  # warm-up, not counted
        except Exception as e:  # noqa: BLE001
            print(f"{scene} 预热失败：{e}", file=sys.stderr)
            continue
        warm = [run(args.base, args.model, msgs, cold=False) for _ in range(runs)]
        cold = [run(args.base, args.model, msgs, cold=True) for _ in range(runs)]
        rows[scene] = {
            "prompt_tokens": warm[0]["prompt_tokens"],
            "warm_ttft_ms": round(statistics.median(r["ttft_ms"] for r in warm)),
            "warm_first_sent_ms": round(statistics.median(r["first_sent_ms"] for r in warm)),
            "warm_decode_tps": round(statistics.median(r["decode_tps"] for r in warm), 1),
            "cold_ttft_ms": round(statistics.median(r["ttft_ms"] for r in cold)),
            "reply_chars": round(statistics.median(r["chars"] for r in warm)),
        }
        print(f"  {scene:<8} ctx {rows[scene]['prompt_tokens']:>4} tok  "
              f"warm {rows[scene]['warm_ttft_ms']}ms  "
              f"cold {rows[scene]['cold_ttft_ms']}ms  "
              f"首句 {rows[scene]['warm_first_sent_ms']}ms  "
              f"{rows[scene]['warm_decode_tps']} tok/s", file=sys.stderr)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0

    print()
    print("| 场景 | 上下文 | 首字（warm） | 首字（cold） | 首句出声 | decode |")
    print("|---|---|---|---|---|---|")
    label = {"short": "一问一答", "hist6": "6 轮历史",
             "screen": "看屏幕 2K 字", "screen4k": "看屏幕 4K 字"}
    for k, v in rows.items():
        print(f"| {label.get(k, k)} | {v['prompt_tokens']} tok | {v['warm_ttft_ms']} ms | "
              f"{v['cold_ttft_ms']} ms | {v['warm_first_sent_ms']} ms | "
              f"{v['warm_decode_tps']} tok/s |")
    worst = max((v["cold_ttft_ms"] for v in rows.values()), default=0)
    print(f"\n最坏冷 prefill 首字 {worst} ms —— "
          f"{'全部在 300 ms 内' if worst < 300 else '有场景越过 300 ms，见上表'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
