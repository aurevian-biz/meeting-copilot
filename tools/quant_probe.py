#!/usr/bin/env python3
"""量一个 ONNX ASR 模型「还能压多小」：按消费它的算子分类统计权重字节。

    python tools/quant_probe.py /path/to/model.onnx
    python tools/quant_probe.py model.int8.onnx --target 80

为什么需要它：把 234MB 的 SenseVoice 压到 80MB 听起来是「再省 2/3」，但只有
MatMul/Gemm 的权重能降到 4-bit（1/4）；Conv、LayerNorm、embedding(Gather) 不行。
所以先看它们各占多少字节，才知道 4-bit 到底能到哪个数、够不够目标。

分类依据是「谁消费这个初始器」而不是名字猜：
  MatMul/Gemm  → 4-bit 可压，乘 0.28（4-bit 权重 + 每 32 元素一个 fp16 scale）
  Conv/ConvTranspose → 保持 8-bit（ONNX Runtime 没有 4-bit Conv 内核）
  Gather/Take  → 8-bit（embedding 行访问，压了也不省带宽）
  其余（Norm/Bias/位置编码等）→ 保持 8-bit，通常很小

退出码：0 = 4-bit 后能达到 --target；1 = 达不到（说明要砍参数，不是砍位宽）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import onnx

# 消费某初始器的算子 → 归到哪一类
CATEGORY = {
    "MatMul": "matmul", "Gemm": "matmul", "MatMulInteger": "matmul",
    "Conv": "conv", "ConvTranspose": "conv", "ConvInteger": "conv",
    "Gather": "embed", "GatherElements": "embed", "EmbedLayerNormalization": "embed",
}
# 每类的压缩系数，都以「1 字节/参数」为基准
# MatMul: 4-bit 权重 + 每 32 元素一个 fp16 scale ≈ 0.5 + 0.03125 → 0.53
KEEP_4BIT = {"matmul": 0.53, "conv": 1.0, "embed": 1.0, "other": 1.0}
KEEP_8BIT = {"matmul": 1.0, "conv": 1.0, "embed": 1.0, "other": 1.0}

# TensorProto 枚举 → 单元素字节数（float32 是 4，别写反）
DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 16: 2}


def elems_of(t) -> int:
    n = 1
    for d in t.dims:
        n *= int(d)
    return n


def bytes_of(t) -> int:
    return elems_of(t) * DTYPE_BYTES.get(t.data_type, 4)


def analyze(path: Path) -> dict:
    m = onnx.load(str(path))
    consumers: dict[str, set[str]] = collections.defaultdict(set)
    for node in m.graph.node:
        for i in node.input:
            consumers[i].add(node.op_type)

    cat_bytes: dict[str, int] = collections.Counter()
    cat_elems: dict[str, int] = collections.Counter()
    fp32_bytes = 0
    biggest: list[tuple[int, str, str, str]] = []
    for init in m.graph.initializer:
        cats = {CATEGORY.get(op, "other") for op in consumers.get(init.name, ())}
        # 一个初始器被多类算子共用时，按最保守（不可压）的类记账
        cat = "other" if not cats else max(cats, key=lambda c: KEEP_8BIT[c])
        n = bytes_of(init)
        cat_bytes[cat] += n
        cat_elems[cat] += elems_of(init)
        if init.data_type == 1:                 # FLOAT
            fp32_bytes += n
        biggest.append((n, init.name, str(init.dims), cat))

    total = sum(cat_bytes.values()) or 1
    est8 = sum(cat_elems[k] * KEEP_8BIT[k] for k in cat_elems)
    est4 = sum(cat_elems[k] * KEEP_4BIT[k] for k in cat_elems)
    biggest.sort(reverse=True)
    return {
        "file_bytes": path.stat().st_size,
        "weights_bytes": total,
        "fp32_bytes": fp32_bytes,
        "params": sum(cat_elems.values()),
        "by_category": dict(cat_bytes),
        "elems_by_category": dict(cat_elems),
        "est_4bit_bytes": int(est4),
        "est_8bit_bytes": int(est8),
        "top": biggest[:8],
        "op_types": dict(collections.Counter(
            n.op_type for n in m.graph.node).most_common(10)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ONNX 权重压缩可行性探针")
    p.add_argument("model")
    p.add_argument("--target", type=float, default=80.0, help="目标体积（MB），默认 80")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    path = Path(a.model)
    if not path.exists():
        print(f"找不到 {path}", file=sys.stderr)
        return 2
    r = analyze(path)

    mw = r["weights_bytes"] / 2**20
    print(f"文件 {r['file_bytes']/2**20:.0f} MB，权重合计 {mw:.0f} MB，"
          f"参数 {r['params']/1e6:.0f}M（其中 fp32 部分 {r['fp32_bytes']/2**20:.0f} MB）")
    print()
    print(f"{'类别':<8}{'字节':>12}  {'占比':>7}  说明")
    note = {"matmul": "可降到 4-bit", "conv": "只能 8-bit（无 4-bit 内核）",
            "embed": "Gather 行访问，压了不省带宽", "other": "Norm/Bias/位置编码"}
    for k, v in sorted(r["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"{k:<8}{v/2**20:>10.1f}MB  {v/r['weights_bytes']*100:>6.1f}%  {note.get(k,'')}")
    print()
    print(f"全 int8（1 字节/参数）估 {r['est_8bit_bytes']/2**20:.0f} MB")
    print(f"MatMul 改 4-bit 后估   {r['est_4bit_bytes']/2**20:.0f} MB "
          f"（目标 {a.target:.0f} MB，比 int8 只省 "
          f"{(r['est_8bit_bytes']-r['est_4bit_bytes'])/2**20:.0f} MB）")
    print()
    print("最大的几个权重：")
    for n, name, shape, cat in r["top"]:
        print(f"  {n/2**20:>7.1f}MB  {cat:<7}{name}  {shape}")

    ok = r["est_4bit_bytes"] / 2**20 <= a.target
    print()
    if ok:
        print("结论：4-bit 能达到目标。下一步验证 ORT 能否加载（MatMulNBits 需要 "
              "ORT>=1.17 且 CPU EP；Jetson 的 CUDA EP 不一定支持）。")
    else:
        short = r["est_4bit_bytes"] / 2**20 - a.target
        print(f"结论：4-bit 只能到 {r['est_4bit_bytes']/2**20:.0f} MB，离目标还差 "
              f"{short:.0f} MB。差距只能靠砍参数（蒸馏/剪枝），不是砍位宽。")
    if a.json:
        print(json.dumps(r, ensure_ascii=False, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
