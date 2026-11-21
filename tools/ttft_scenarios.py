#!/usr/bin/env python3
"""按真实助理的 prompt 形态，扫一遍首字延迟（TTFT）。

与 llm_probe.py 的区别：那个用固定人设，回答「这个模型能不能用」；
这个直接用 voice_server 里真实的 system prompt，按四种真实上下文扫，
回答「在什么情况下首字会变慢、会不会越过 300ms」。

四种场景：
  short   system + 1 句用户（≈日常一问一答）
  hist6   system + 6 轮历史（voice_server 取 history[-6:]）
  screen  system + 2000 字 OCR 文本（看屏幕）
  screen4k system + 4000 字 OCR 文本（长文档 / 小字密排）

每档跑两种：
  warm  同一 prompt 重复，前缀命中（日常多轮真实形态）
  cold  prompt 最前面加唯一 nonce，逼它整段重算（最坏情况）

用法：
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

from llm_probe import post_stream  # noqa: E402  复用同一套流式测量

OCR_SAMPLE = (
    "会議資料 2026年度 第3四半期 進捗報告 議題 1 開発状況 2 品質指標 3 リスクと対策 "
    "添付資料あり 次回会議 11月27日 15時 会議室B 参加者 田中 鈴木 佐藤 高橋 "
    "画面右下 通知 3件 Slack メール カレンダー 予定 16時30分 レビュー "
)


def real_system() -> str:
    """从 voice_server 取真实的 system prompt，避免两处文案走偏。"""
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
            run(args.base, args.model, msgs, cold=True)  # 预热，不计
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
