#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""trt_rec.py — swap OCR's rec backend for TensorRT

## Why this exists

This Jetson is JetPack 7.2 (L4T r39.2.1) + CUDA **13.2**, while onnxruntime officially ships only
**cu12** wheels. `pip` has no cu13 package and `apt` has no `cuda-cudart-12-*`, so although
`onnxruntime-gpu` installs and `get_available_providers()` lists CUDAExecutionProvider, building a
session fails with `No registered plugin EP device found` — the EP's .so fails to dlopen.
Conclusion: **the CUDA EP path is dead on this system**; do not waste time on it.

TensorRT 10.16 does ship with the system (`libnvinfer.so.10` + `trtexec` + python `tensorrt`),
so bypass onnxruntime entirely and run rec on TRT.

## Results (132-line screen, real hardware)

| Backend | rec time | OCR total | Lines | Mean score |
|---|---|---|---|---|
| CPU (onnxruntime) | 3.4s | 4.10s | 132 | 0.951 |
| **TensorRT FP16** | — | **1.69s** | 132 | 0.949 |

## Why only rec, not det/cls

rec is 88% of OCR, so swapping it has the best payoff. det (0.30s) and cls (0.15s) can wait,
and det's input size varies with the picture, which makes the profile harder to pin down.

## Why there is a CPU fallback

rec's batch input width = 48 × the largest aspect ratio in the batch. Screens occasionally hold one very
long line (a path, a URL, a whole block of code) that stretches the whole batch to 1600+ wide. Over 75
measured batches, **76% are exactly the 320 baseline** and only 4% exceed 1536. Rather than rebuild the
engine for 3000+ outliers and slow the common path, 4% of batches on the slow path beats 100% on a worse kernel.

## The engine is pinned to the TensorRT version

Rebuild it after switching JetPack / upgrading TRT; the rebuild command is in README 4.8.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class TrtRecSession:
    """Disguise a TRT engine as RapidOCR's OrtInferSession.

    `TextRecognizer` uses only three things from the session:
      `__call__(batch) -> [logits]`, `have_key()`, `get_character_list()`.
    The latter two (reading the character list) delegate to the original ORT session; only the forward pass is taken over.
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

        # Stats: how many batches ran on GPU, how many fell back, how long each forward pass took
        self.calls = 0
        self.fallbacks = 0
        self.total_ms = 0.0

    # ---------------------------------------------------------------- Helpers
    @staticmethod
    def _check(err, what: str):
        if int(err) != 0:
            raise RuntimeError(f"{what} 失败：{err}")

    def _buffers(self, in_shape, out_shape):
        """Cache device memory by shape. Every malloc on unified memory costs, and this runs per batch."""
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

    # ---------------------------------------------------------------- Forward
    def __call__(self, x: np.ndarray):
        t0 = time.perf_counter()
        x = np.ascontiguousarray(x, dtype=np.float32)
        shape = tuple(int(v) for v in x.shape)

        if not self.ctx.set_input_shape(self.in_name, shape):
            # Outside the optimization profile: this batch falls back to CPU, correctness unaffected
            if self._inner is None:
                raise RuntimeError(f"输入形状 {shape} 超出 profile，且没有 CPU 回退")
            self.fallbacks += 1
            return self._inner(x)

        out_shape = tuple(int(v) for v in self.ctx.get_tensor_shape(self.out_name))
        d_in, d_out, n_in, n_out = self._buffers(shape, out_shape)
        rt = self._rt

        # Note: set_tensor_address returns bool, not cudaError_t
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

    # ------------------------------------------------ Delegated to the original ORT session
    def have_key(self):
        return bool(self._inner and self._inner.have_key())

    def get_character_list(self):
        return self._inner.get_character_list() if self._inner else None

    # ---------------------------------------------------------------- Misc
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
    """Attach TensorRT to an already-built RapidOCR instance (replaces rec's session in place)."""
    sess = TrtRecSession(engine_path, inner=ocr_engine.text_rec.session)
    ocr_engine.text_rec.session = sess
    return sess
