#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Check whether sherpa-onnx's provider argument can really switch to GPU.

Background: the from_sense_voice(provider='cpu') signature does take provider,
but the wheel is a pure CPU build (ldd finds no libcudart / libnvinfer).
This measures three values to see whether it errors out or silently degrades to cpu.
"""
import time
from pathlib import Path

D = Path("/opt/meeting-copilot/data/asr-models/"
         "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
model = D / "model.int8.onnx"
if not model.exists():
    model = D / "model.onnx"
tokens = D / "tokens.txt"
print("模型:", model.name, model.exists(), "| tokens:", tokens.exists())

import numpy as np
import sherpa_onnx

audio = (np.zeros(16000 * 2, dtype=np.float32))

for prov in ("cpu", "cuda", "trt"):
    t0 = time.time()
    try:
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(tokens), num_threads=4,
            use_itn=True, language="auto", provider=prov)
        build = time.time() - t0
        t1 = time.time()
        s = rec.create_stream()
        s.accept_waveform(16000, audio)
        rec.decode_stream(s)
        dec = time.time() - t1
        print(f"provider={prov:5} 建 {build:.2f}s  解码 {dec*1000:.0f}ms  "
              f"结果={s.result.text!r}   ← 没报错")
    except Exception as e:  # noqa: BLE001
        print(f"provider={prov:5} 失败: {type(e).__name__}: {e}")

print("\n注意：不报错 ≠ 真的用了 GPU。要证明得上 nvidia-smi 看显存/利用率。")
