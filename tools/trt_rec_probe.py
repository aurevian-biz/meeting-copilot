#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/trt_rec_probe.py — 把 OCR 的 rec 换到 TensorRT，量真实收益

为什么走 TensorRT 而不是 onnxruntime 的 CUDA EP：
  JetPack 7.2 带 CUDA 13.2，而 onnxruntime 官方只发 cu12 轮子（pip 无 cu13、
  apt 无 cuda-cudart-12-*），CUDA EP 加载时 dlopen 失败、报
  「No registered plugin EP device found」。TensorRT 10.16 倒是随系统就有。

做法：不改 RapidOCR 源码，只把 `TextRecognizer.session` 换成一个 TensorRT 包装。
这样 resize_norm_img / CTC 解码 / 字符表全部复用原实现，两边只差「推理后端」，
测出来的才是纯粹的后端差异。

用法：
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
    """把 TensorRT engine 伪装成 OrtInferSession。

    RapidOCR 的 TextRecognizer 只用到 session 的三件事：
      `__call__(batch) -> [logits]`、`have_key()`、`get_character_list()`。
    后两个委托给原 ORT session（它负责读 keys 文件），我们只接管前向。
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
        self.fallbacks = 0     # 形状超 profile 而退回 CPU 的批次数
        self.last_ms = 0.0

    # ---------------------------------------------------------------- 工具
    def _check(self, err, what: str):
        if int(err) != 0:
            raise RuntimeError(f"{what} 失败：{err}")

    def _buffers(self, in_shape, out_shape):
        """按形状缓存显存，避免每次 malloc（在统一内存上尤其贵）。"""
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

    # ---------------------------------------------------------------- 前向
    def __call__(self, x: np.ndarray):
        rt = self._rt
        x = np.ascontiguousarray(x, dtype=np.float32)
        shape = tuple(int(v) for v in x.shape)
        t0 = time.perf_counter()

        if not self.ctx.set_input_shape(self.in_name, shape):
            # 超出 optimization profile（屏幕上偶尔有一行特别长，整批被 max_wh_ratio 撑宽）。
            # 与其重建一个覆盖极值的大 engine 拖慢常见路径，不如让这几批退回 CPU：
            # 实测宽度分布 76% 就是基准 320，超 1536 的只占 4%。
            if self._inner is not None:
                self.fallbacks += 1
                self.last_ms = (time.perf_counter() - t0) * 1000
                return self._inner(x)
            raise RuntimeError(f"set_input_shape({shape}) 失败（无 CPU 回退）")
        out_shape = tuple(int(v) for v in self.ctx.get_tensor_shape(self.out_name))
        d_in, d_out, n_in, n_out = self._buffers(shape, out_shape)

        # 注意：set_tensor_address 返回 bool（True=成功），不是 cudaError_t
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

    # -------------------------------------------------- 委托给原 ORT session
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

    # ---------------------------------------------------------- 对照组：CPU
    print("[1/3] CPU (onnxruntime)", flush=True)
    cpu = RapidOCR(intra_op_num_threads=args.threads)
    cpu(im)                                  # 暖机
    t_cpu = []
    for _ in range(args.runs):
        t = time.perf_counter()
        res, _ = cpu(im)
        t_cpu.append(time.perf_counter() - t)
    st_cpu = _stats(res)
    print(f"      {['%.2f' % x for x in t_cpu]}  均 {sum(t_cpu)/len(t_cpu):.2f}s"
          f"  {st_cpu['rows']} 行 / 均分 {st_cpu['score']:.3f} / {st_cpu['chars']} 字符")

    # ------------------------------------------------------- 实验组：TensorRT
    print("[2/3] 换成 TensorRT rec（预处理与解码不变）", flush=True)
    trt_eng = RapidOCR(intra_op_num_threads=args.threads)
    trt_sess = TrtSession(args.engine, inner=trt_eng.text_rec.session)
    trt_eng.text_rec.session = trt_sess
    trt_eng(im)                              # 暖机：CUDA context + 形状缓存
    t_gpu = []
    for _ in range(args.runs):
        t = time.perf_counter()
        res_trt, _ = trt_eng(im)
        t_gpu.append(time.perf_counter() - t)
    st_trt = _stats(res_trt)
    print(f"      {['%.2f' % x for x in t_gpu]}  均 {sum(t_gpu)/len(t_gpu):.2f}s"
          f"  {st_trt['rows']} 行 / 均分 {st_trt['score']:.3f} / {st_trt['chars']} 字符")
    print(f"      GPU 批 {trt_sess.calls} 次，超 profile 回退 CPU {trt_sess.fallbacks} 次")

    # ------------------------------------------------------------ 一致性
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

    # 判据：行数一致 + 均分差 < 0.01，才算「精度没塌」
    ok_rows = st_cpu["rows"] == st_trt["rows"]
    ok_score = abs(st_cpu["score"] - st_trt["score"]) < 0.01
    print(f"精度判定：行数{'一致' if ok_rows else '不一致'}、均分差 "
          f"{abs(st_cpu['score']-st_trt['score']):.3f} → "
          f"{'通过' if ok_rows and ok_score else '不通过'}")
    return 0 if ok_rows and ok_score else 1


if __name__ == "__main__":
    sys.exit(main())
