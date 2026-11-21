#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Inspect sherpa-onnx provider capabilities: whether the CUDA / TensorRT EP is present."""
import inspect
import json
import subprocess


def section(t):
    print("\n" + "=" * 8, t, "=" * 8)


section("sherpa-onnx 版本与编译信息")
try:
    import sherpa_onnx
    print("version:", getattr(sherpa_onnx, "__version__", "?"))
    so = sherpa_onnx.__file__
    print("so:", so)
    # Build options
    try:
        print("build:", sherpa_onnx.get_build_info() if hasattr(sherpa_onnx, "get_build_info") else "n/a")
    except Exception as e:  # noqa: BLE001
        print("build info err:", e)
    print("providers (module attr):", getattr(sherpa_onnx, "PROVIDERS", "n/a"))
except Exception as e:  # noqa: BLE001
    print("import 失败:", e)

section("from_sense_voice 签名")
try:
    import sherpa_onnx
    print(inspect.signature(sherpa_onnx.OfflineRecognizer.from_sense_voice))
except Exception as e:  # noqa: BLE001
    print("err:", e)

section("so 里链接的 EP 库（ldd / strings）")
try:
    import sherpa_onnx
    lib = sherpa_onnx.__file__
    out = subprocess.run(["bash", "-lc", f"ldd {lib} 2>/dev/null | grep -iE 'cuda|nvinfer|cudnn|cublas' | head"],
                         capture_output=True, text=True, timeout=60)
    print("ldd:", out.stdout.strip() or "(无 CUDA 相关依赖 → 纯 CPU 编译)")
    out2 = subprocess.run(["bash", "-lc",
                           f"strings {lib} 2>/dev/null | grep -oiE 'TensorrtExecutionProvider|CUDAExecutionProvider|CoreMLExecutionProvider' | sort -u"],
                          capture_output=True, text=True, timeout=120)
    print("strings 里的 EP 名:", out2.stdout.strip() or "(无 → 不支持 GPU provider)")
except Exception as e:  # noqa: BLE001
    print("err:", e)

section("onnxruntime 里的 EP")
try:
    import onnxruntime as ort
    print("ort version:", ort.__version__)
    print("providers:", ort.get_available_providers())
except Exception as e:  # noqa: BLE001
    print("err:", e)

section("模型文件")
out = subprocess.run(["bash", "-lc",
                      "ls -lh /opt/meeting-copilot/asr-models/*/ 2>/dev/null | head -12"],
                     capture_output=True, text=True, timeout=60)
print(out.stdout or "(路径不对)")
