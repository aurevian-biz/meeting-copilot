#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""tools/trt_rec_probe.py — move the OCR rec stage onto TensorRT and measure the real gain

Why TensorRT rather than onnxruntime's CUDA EP:
  JetPack 7.2 ships CUDA 13.2, while onnxruntime publishes cu12 wheels only (no cu13 on pip,
  no cuda-cudart-12-* on apt), so the CUDA EP fails its dlopen at load time and reports
  "No registered plugin EP device found". TensorRT 10.16, on the other hand, is already on the system.

Approach: no changes to RapidOCR sources, just swap `TextRecognizer.session` for a TensorRT wrapper.
That way resize_norm_img / CTC decoding / the charset all reuse the original implementation, and the only difference between the two sides is "the inference backend",
so what gets measured is the pure backend delta.

Usage:
  python tools/trt_rec_probe.py --engine /tmp/rec_fp16.engine --image /tmp/vision-bench/frame0.png
  python tools/trt_rec_probe.py --engine ... --image ... --side 1440 --threads 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TrtSession:
    """Disguise a TensorRT engine as an OrtInferSession.

    RapidOCR's TextRecognizer only uses three things from the session:
      `__call__(batch) -> [logits]`, `have_key()`, `get_character_list()`.
    The last two are delegated to the original ORT session (it reads the keys file); we only take over the forward pass.
    """

    def __init__(self, engine_path: str, inner=None, verbose: bool = False):
        import tensorrt as trt
        from cuda.bindings import runtime as rt

        self._rt = rt
        self._inner = inner
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            blob = f.read()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"反序列化失败：{engine_path}")
        self.ctx = self.engine.create_execution_context()

        self.in_names, self.out_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_names.append(name)
            else:
                self.out_names.append(name)
        self.in_name = self.in_names[0]
        self.out_name = self.out_names[0]

        err, stream = rt.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")
        self.stream = stream
        self._mem: dict[tuple, tuple[int, int]] = {}   # shape -> (d_in, d_out)
        self.calls = 0
        self.fallbacks = 0     # batches sent back to CPU because the shape exceeded the profile
        self.last_ms = 0.0

    # ---------------------------------------------------------------- helpers
    def _check(self, err, what: str):
        if int(err) != 0:
            raise RuntimeError(f"{what} 失败：{err}")

    def _buffers(self, in_shape, out_shape):
        """Cache device memory per shape to avoid a malloc every time (especially costly on unified memory)."""
        key = (in_shape, out_shape)
        if key not in self._mem:
            rt = self._rt
            n_in = int(np.prod(in_shape)) * 4
            n_out = int(np.prod(out_shape)) * 4
            e1, d_in = rt.cudaMalloc(n_in)
            e2, d_out = rt.cudaMalloc(n_out)
            self._check(e1, "cudaMalloc(in)")
            self._check(e2, "cudaMalloc(out)")
            self._mem[key] = (d_in, d_out, n_in, n_out)
        return self._mem[key]

    # ---------------------------------------------------------------- forward
    def __call__(self, x: np.ndarray):
        rt = self._rt
        x = np.ascontiguousarray(x, dtype=np.float32)
        shape = tuple(int(v) for v in x.shape)
        t0 = time.perf_counter()

        if not self.ctx.set_input_shape(self.in_name, shape):
            # Beyond the optimization profile (the screen occasionally has one very long line, and max_wh_ratio widens the whole batch).
            # Rather than rebuild a big engine covering the extremes and slowing the common path, send these few batches back to CPU:
            # the measured width distribution is 76% at the 320 baseline and only 4% above 1536.
            if self._inner is not None:
                self.fallbacks += 1
                self.last_ms = (time.perf_counter() - t0) * 1000
                return self._inner(x)
            raise RuntimeError(f"set_input_shape({shape}) 失败（无 CPU 回退）")
        out_shape = tuple(int(v) for v in self.ctx.get_tensor_shape(self.out_name))
        d_in, d_out, n_in, n_out = self._buffers(shape, out_shape)

        # Note: set_tensor_address returns bool (True=success), not cudaError_t
        if not self.ctx.set_tensor_address(self.in_name, int(d_in)):
            raise RuntimeError("set_tensor_address(in) 失败")
        if not self.ctx.set_tensor_address(self.out_name, int(d_out)):
            raise RuntimeError("set_tensor_address(out) 失败")

        e, = rt.cudaMemcpyAsync(d_in, x.ctypes.data, n_in,
                                rt.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self._check(e, "H2D")
        ok = self.ctx.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("execute_async_v3 失败")
        host = np.empty(out_shape, dtype=np.float32)
        e, = rt.cudaMemcpyAsync(host.ctypes.data, d_out, n_out,
                                rt.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        self._check(e, "D2H")
        e, = rt.cudaStreamSynchronize(self.stream)
        self._check(e, "sync")

        self.calls += 1
        self.last_ms = (time.perf_counter() - t0) * 1000
        return [host]

    # -------------------------------------------------- delegate to the original ORT session
    def have_key(self):
        return bool(self._inner and self._inner.have_key())

    def get_character_list(self):
        return self._inner.get_character_list() if self._inner else None

    def close(self):
        for d_in, d_out, _a, _b in self._mem.values():
            self._rt.cudaFree(d_in)
            self._rt.cudaFree(d_out)
        self._mem.clear()
        self._rt.cudaStreamDestroy(self.stream)


def _load_image(path: str, side: int):
    import cv2
    im = cv2.imread(path)
    if im is None:
        raise SystemExit(f"读不到图片：{path}")
    h, w = im.shape[:2]
    s = side / max(w, h)
    if s < 1.0:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return im


def _stats(res):
    good = [(r[1], float(r[2])) for r in (res or []) if float(r[2]) >= 0.35 and r[1].strip()]
    return {
        "rows": len(good),
        "score": sum(v for _, v in good) / len(good) if good else 0.0,
        "chars": sum(len(t) for t, _ in good),
        "text": [t for t, _ in good],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TensorRT engine 路径")
    ap.add_argument("--image", required=True, help="测试图（1080p 原始帧）")
    ap.add_argument("--side", type=int, default=1440, help="OCR 输入长边")
    ap.add_argument("--threads", type=int, default=8, help="CPU 侧线程数（对照组）")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args(argv)

    from rapidocr_onnxruntime import RapidOCR

    im = _load_image(args.image, args.side)

    # ---------------------------------------------------------- control group: CPU
    print("[1/3] CPU (onnxruntime)", flush=True)
    cpu = RapidOCR(intra_op_num_threads=args.threads)
    cpu(im)                                  # warm-up
    t_cpu = []
    for _ in range(args.runs):
        t = time.perf_counter()
        res, _ = cpu(im)
        t_cpu.append(time.perf_counter() - t)
    st_cpu = _stats(res)
    print(f"      {['%.2f' % x for x in t_cpu]}  均 {sum(t_cpu)/len(t_cpu):.2f}s"
          f"  {st_cpu['rows']} 行 / 均分 {st_cpu['score']:.3f} / {st_cpu['chars']} 字符")

    # ------------------------------------------------------- test group: TensorRT
    print("[2/3] 换成 TensorRT rec（预处理与解码不变）", flush=True)
    trt_eng = RapidOCR(intra_op_num_threads=args.threads)
    trt_sess = TrtSession(args.engine, inner=trt_eng.text_rec.session)
    trt_eng.text_rec.session = trt_sess
    trt_eng(im)                              # warm-up: CUDA context + shape cache
    t_gpu = []
    for _ in range(args.runs):
        t = time.perf_counter()
        res_trt, _ = trt_eng(im)
        t_gpu.append(time.perf_counter() - t)
    st_trt = _stats(res_trt)
    print(f"      {['%.2f' % x for x in t_gpu]}  均 {sum(t_gpu)/len(t_gpu):.2f}s"
          f"  {st_trt['rows']} 行 / 均分 {st_trt['score']:.3f} / {st_trt['chars']} 字符")
    print(f"      GPU 批 {trt_sess.calls} 次，超 profile 回退 CPU {trt_sess.fallbacks} 次")

    # ------------------------------------------------------------ consistency
    print("[3/3] 结果一致性", flush=True)
    same = sum(1 for a, b in zip(st_cpu["text"], st_trt["text"]) if a == b)
    diff = [(a, b) for a, b in zip(st_cpu["text"], st_trt["text"]) if a != b]
    print(f"      行数 {st_cpu['rows']} vs {st_trt['rows']}；逐行完全一致 {same}/"
          f"{min(st_cpu['rows'], st_trt['rows'])}")
    for a, b in diff[:5]:
        print(f"        CPU: {a}\n        TRT: {b}")

    sp_cpu, sp_trt = sum(t_cpu) / len(t_cpu), sum(t_gpu) / len(t_gpu)
    print(f"\n合计 {sp_cpu:.2f}s → {sp_trt:.2f}s（{sp_cpu/sp_trt:.2f}×）")

    trt_sess.close()

    # Criterion: same line count + mean score delta < 0.01, i.e. "accuracy has not collapsed"
    ok_rows = st_cpu["rows"] == st_trt["rows"]
    ok_score = abs(st_cpu["score"] - st_trt["score"]) < 0.01
    print(f"精度判定：行数{'一致' if ok_rows else '不一致'}、均分差 "
          f"{abs(st_cpu['score']-st_trt['score']):.3f} → "
          f"{'通过' if ok_rows and ok_score else '不通过'}")
    return 0 if ok_rows and ok_score else 1


if __name__ == "__main__":
    sys.exit(main())
