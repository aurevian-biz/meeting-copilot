#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trt_rec.py — 把 OCR 的 rec 后端换成 TensorRT

## 为什么需要它

这台 Jetson 是 JetPack 7.2（L4T r39.2.1）+ CUDA **13.2**，而 onnxruntime 官方只发
**cu12** 轮子。`pip` 上没有 cu13 包，`apt` 里也没有 `cuda-cudart-12-*`，所以
`onnxruntime-gpu` 虽然能装、`get_available_providers()` 也会列出 CUDAExecutionProvider，
但建 session 时会报 `No registered plugin EP device found`——EP 的 .so dlopen 失败。
结论：**CUDA EP 这条路在这套系统上是死的**，别在这上面浪费时间。

TensorRT 10.16 倒是随系统就有（`libnvinfer.so.10` + `trtexec` + python `tensorrt`），
所以直接绕过 onnxruntime，用 TRT 跑 rec。

## 效果（真机 132 行画面）

| 后端 | rec 耗时 | OCR 合计 | 行数 | 均分 |
|---|---|---|---|---|
| CPU (onnxruntime) | 3.4s | 4.10s | 132 | 0.951 |
| **TensorRT FP16** | — | **1.69s** | 132 | 0.949 |

## 为什么只换 rec，不换 det/cls

rec 占 OCR 的 88%，换它性价比最高。det（0.30s）和 cls（0.15s）以后再说，
而且 det 的输入尺寸随画面变化，profile 更难定。

## 为什么会有 CPU 回退

rec 的批输入宽度 = 48 × 批内最大宽高比。屏幕上偶尔有一行特别长（路径、URL、
一整段代码），会把整批撑到 1600+ 宽。实测 75 批里：**76% 就是基准 320**，
超 1536 的只占 4%。与其为大到 3000+ 的极值重建 engine 拖慢常见路径，
不如让超限的那几批退回 CPU——4% 的批走慢路，比 100% 的批用差 kernel 划算。

## engine 是跟 TensorRT 版本绑死的

换 JetPack / 升级 TRT 后必须重建，重建命令见 README 4.8。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class TrtRecSession:
    """把 TRT engine 伪装成 RapidOCR 的 OrtInferSession。

    `TextRecognizer` 只用到 session 的三件事：
      `__call__(batch) -> [logits]`、`have_key()`、`get_character_list()`。
    后两个（读字符表）委托给原 ORT session，这里只接管前向。
    """

    def __init__(self, engine_path: str | Path, inner=None, verbose: bool = False):
        import tensorrt as trt
        from cuda.bindings import runtime as rt

        p = Path(engine_path)
        if not p.exists():
            raise FileNotFoundError(f"engine 不存在：{p}")

        self._rt = rt
        self._inner = inner
        self.engine_path = str(p)
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(p.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"反序列化失败（TRT 版本不匹配？）：{p}")
        self.ctx = self.engine.create_execution_context()

        self.in_name = self.out_name = ""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        if not self.in_name or not self.out_name:
            raise RuntimeError("engine 缺少输入/输出张量")

        err, stream = rt.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")
        self.stream = stream
        self._mem: dict[tuple, tuple] = {}    # (in_shape,out_shape) -> (d_in,d_out,bytes_in,bytes_out)

        # 统计：GPU 走了多少批、回退多少批、每次前向花了多久
        self.calls = 0
        self.fallbacks = 0
        self.total_ms = 0.0

    # ---------------------------------------------------------------- 工具
    @staticmethod
    def _check(err, what: str):
        if int(err) != 0:
            raise RuntimeError(f"{what} 失败：{err}")

    def _buffers(self, in_shape, out_shape):
        """按形状缓存显存。统一内存上每次 malloc 都不便宜，而这里是每批都调。"""
        key = (in_shape, out_shape)
        hit = self._mem.get(key)
        if hit is None:
            rt = self._rt
            n_in = int(np.prod(in_shape)) * 4
            n_out = int(np.prod(out_shape)) * 4
            e1, d_in = rt.cudaMalloc(n_in)
            e2, d_out = rt.cudaMalloc(n_out)
            self._check(e1, "cudaMalloc(in)")
            self._check(e2, "cudaMalloc(out)")
            hit = (d_in, d_out, n_in, n_out)
            self._mem[key] = hit
        return hit

    # ---------------------------------------------------------------- 前向
    def __call__(self, x: np.ndarray):
        t0 = time.perf_counter()
        x = np.ascontiguousarray(x, dtype=np.float32)
        shape = tuple(int(v) for v in x.shape)

        if not self.ctx.set_input_shape(self.in_name, shape):
            # 超出 optimization profile：这一批退回 CPU，不影响正确性
            if self._inner is None:
                raise RuntimeError(f"输入形状 {shape} 超出 profile，且没有 CPU 回退")
            self.fallbacks += 1
            return self._inner(x)

        out_shape = tuple(int(v) for v in self.ctx.get_tensor_shape(self.out_name))
        d_in, d_out, n_in, n_out = self._buffers(shape, out_shape)
        rt = self._rt

        # 注意：set_tensor_address 返回 bool，不是 cudaError_t
        if not self.ctx.set_tensor_address(self.in_name, int(d_in)):
            raise RuntimeError("set_tensor_address(in) 失败")
        if not self.ctx.set_tensor_address(self.out_name, int(d_out)):
            raise RuntimeError("set_tensor_address(out) 失败")

        e, = rt.cudaMemcpyAsync(d_in, x.ctypes.data, n_in,
                                rt.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self._check(e, "H2D")
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 失败")
        host = np.empty(out_shape, dtype=np.float32)
        e, = rt.cudaMemcpyAsync(host.ctypes.data, d_out, n_out,
                                rt.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        self._check(e, "D2H")
        e, = rt.cudaStreamSynchronize(self.stream)
        self._check(e, "stream sync")

        self.calls += 1
        self.total_ms += (time.perf_counter() - t0) * 1000
        return [host]

    # ------------------------------------------------ 委托给原 ORT session
    def have_key(self):
        return bool(self._inner and self._inner.have_key())

    def get_character_list(self):
        return self._inner.get_character_list() if self._inner else None

    # ---------------------------------------------------------------- 其他
    def status(self) -> dict:
        n = self.calls + self.fallbacks
        return {
            "engine": Path(self.engine_path).name,
            "calls": self.calls,
            "fallbacks": self.fallbacks,
            "fallback_pct": round(100.0 * self.fallbacks / n, 1) if n else 0.0,
            "avg_ms": round(self.total_ms / self.calls, 1) if self.calls else 0.0,
            "shapes": len(self._mem),
        }

    def close(self) -> None:
        for d_in, d_out, _a, _b in self._mem.values():
            self._rt.cudaFree(d_in)
            self._rt.cudaFree(d_out)
        self._mem.clear()
        try:
            self._rt.cudaStreamDestroy(self.stream)
        except Exception:  # noqa: BLE001
            pass


def attach(ocr_engine, engine_path: str | Path) -> TrtRecSession:
    """把 TensorRT 挂到已经建好的 RapidOCR 实例上（原地替换 rec 的 session）。"""
    sess = TrtRecSession(engine_path, inner=ocr_engine.text_rec.session)
    ocr_engine.text_rec.session = sess
    return sess
