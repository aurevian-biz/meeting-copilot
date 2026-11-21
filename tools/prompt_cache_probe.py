#!/usr/bin/env python3
"""前缀缓存到底能救回多少：改动位置 × 首字延迟。

背景：想用「只把变化的部分送进 prompt」给看屏幕提速。但 vLLM 的前缀缓存
按块（16 token）沿前缀链匹配，所以收益完全取决于**变化落在什么位置**：

  - 变化落在末尾 → 前面整段还命中，只重算尾巴
  - 变化落在中间 → 从变化处往后全部失效
  - 变化落在开头 → 等于冷 prefill

测法很关键：**必须用「原始文本」预热，再发「修改后的文本」**。如果先用修改版
预热再测修改版，测到的是「同一内容第二次」，永远 100% 命中（第一版就栽在这）。

用法：
  python tools/prompt_cache_probe.py --base http://127.0.0.1:8000/v1 --model qwen
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_probe import post_stream  # noqa: E402

LINE = "第{0:02d}行 会議資料 2026年度 第3四半期 進捗報告 項目 担当 田中 状態 完了"
N_LINES = 55


def ocr_lines() -> list[str]:
    return [LINE.format(i + 1) for i in range(N_LINES)]


def build(system: str, lines: list[str], question: str = "画面について教えて") -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": "[屏幕文字]\n" + "\n".join(lines) + "\n" + question}]


def measure(base: str, model: str, msgs: list[dict]) -> tuple[float, int]:
    """单次发送（不预热本内容），返回 (ttft_ms, prompt_tokens)。"""
    u: dict = {}
    ttft, *_ = post_stream(base, model, msgs, 8, u)
    return ttft, u.get("prompt_tokens", 0)


def prime(base: str, model: str, msgs: list[dict], times: int = 2) -> None:
    for _ in range(times):
        post_stream(base, model, msgs, 8)


def cold_ref(base: str, model: str, msgs: list[dict], runs: int = 2) -> float:
    vals = []
    for _ in range(runs):
        m = json.loads(json.dumps(msgs))
        m[0]["content"] = f"[req {uuid.uuid4().hex[:8]}]\n" + m[0]["content"]
        ttft, *_ = post_stream(base, model, m, 8)
        vals.append(ttft)
    return statistics.median(vals)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen")
    args = p.parse_args()

    from voice_server import Llm
    system = Llm(args.base, args.model).system
    base_lines = ocr_lines()
    base = build(system, base_lines)

    prime(args.base, args.model, base)
    warm, pts = measure(args.base, args.model, base)

    variants: list[tuple[str, list[dict]]] = []
    variants.append(("末尾追加一行", build(system, base_lines + ["第56行 追加的一行"])))

    lines = list(base_lines)
    lines[-1] = "第55行 光标位置变了"
    variants.append(("改最后一行", build(system, lines)))

    lines = list(base_lines)
    lines[N_LINES // 2] = "第28行 通知弹窗 3件"
    variants.append(("改中间一行", build(system, lines)))

    lines = list(base_lines)
    lines[0] = "第01行 窗口标题变了"
    variants.append(("改第一行", build(system, lines)))

    # 对照：改掉前 30 行（若按块位置无关复用，应只重算这 30 行）
    lines = list(base_lines)
    for i in range(30):
        lines[i] = f"第{i + 1:02d}行 完全不同的内容 変わりました"
    variants.append(("改前 30 行", build(system, lines)))

    # 对照：等长但内容全换（验证缓存到底是前缀链式还是块级任意复用）
    lines = [f"第{i + 1:02d}行 別の資料 別の内容 担当 佐藤 状態 未着手 メモ なし"
             for i in range(N_LINES)]
    variants.append(("全部换掉（等长）", build(system, lines)))

    rows = [("完全相同（缓存态基准）", pts, warm)]
    for name, msgs in variants:
        t, pt = measure(args.base, args.model, msgs)
        rows.append((name, pt, t))

    cold = cold_ref(args.base, args.model, base)

    # 重排实验：稳定段在前、只换尾部
    stable = base_lines[:12]
    vol_a = [f"易变区 时钟 23:2{i} 通知 {i} 件" for i in range(6)]
    vol_b = [f"易变区 时钟 23:5{i} 通知 {i} 件" for i in range(6)]
    prime(args.base, args.model, build(system, stable + vol_a))
    t, pt = measure(args.base, args.model, build(system, stable + vol_b))
    rows.append(("重排：稳定在前只换尾部", pt, t))

    print("| 场景 | 上下文 | 首字 | 相对冷态省 |")
    print("|---|---|---|---|")
    for name, pt, t in rows:
        print(f"| {name} | {pt} tok | **{t:.0f} ms** | {1 - t / cold:.0%} |")
    print(f"\n冷 prefill 基准（同一 55 行上下文，加 nonce）：{cold:.0f} ms")
    print(f"缓存全命中基准：{warm:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
