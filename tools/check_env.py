#!/usr/bin/env python3
"""装完自查：依赖能不能 import、模型在不在、设备认不认。

为什么单独一个脚本：这套系统最坑的不是报错，而是**静默降级**——
  · cuda-python 版本和 JetPack 不配套 → TensorRT 不可用 → OCR 悄悄跑在 CPU 上，
    从 300ms 变 3s，日志里一个字都不提
  · pyopenjtalk 装成了官方源码版但编译失败 → 日语音色直接不可用
  · 采集卡有 /dev/video0 但读出来是全黑（预热帧没丢）
这些在「服务起来了」这一层完全看不出来，只能逐项探。
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

OK, BAD, WARN = "✓", "✗", "!"


def line(mark: str, msg: str) -> None:
    print(f"  {mark} {msg}")


def check_imports() -> int:
    eggs = 0
    mods = [
        ("numpy", "必须"), ("scipy", "VAD 之后的处理"), ("sherpa_onnx", "ASR / VAD / 唤醒"),
        ("cv2", "抓帧"), ("rapidocr_onnxruntime", "看屏幕"), ("fastapi", "服务"),
        ("uvicorn", "服务"), ("piper", "TTS"),
    ]
    for mod, why in mods:
        try:
            m = importlib.import_module(mod)
            line(OK, f"{mod:<22} {getattr(m, '__version__', '')}  ({why})")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"{mod:<22} 装不上：{e}")
            eggs += 1
    try:
        import pyopenjtalk  # noqa: F401
        line(OK, "pyopenjtalk            日语分词 / 屏幕词读音")
    except Exception as e:  # noqa: BLE001
        line(WARN, f"pyopenjtalk 不可用（日语 TTS 与屏幕词表会退化）：{e}")
    return eggs


def check_models() -> int:
    eggs = 0
    for p in [paths.SENSEVOICE_DIR / "model.int8.onnx", paths.VAD_MODEL,
              paths.PIPER_JA, paths.PIPER_ZH]:
        if p.is_file():
            line(OK, f"{p.relative_to(paths.DATA)}  {p.stat().st_size / 1e6:.0f}MB")
        else:
            line(BAD, f"缺 {p}（跑 scripts/download-models.sh）")
            eggs += 1
    return eggs


def check_gpu() -> int:
    """TensorRT 是可选加速；不可用时明确告警，因为它是**静默**降级的。"""
    try:
        import tensorrt as trt
        line(OK, f"TensorRT {trt.__version__}（OCR 可走 GPU）")
    except Exception as e:  # noqa: BLE001
        line(WARN, f"TensorRT 不可用，OCR 会跑 CPU（慢 5~10 倍）：{e}")
    try:
        from cuda.bindings import runtime as rt
        err, _ = rt.cudaMalloc(16)
        line(OK if int(err) == 0 else WARN, f"cuda-python cudaMalloc → {err}")
    except Exception as e:  # noqa: BLE001
        line(WARN, f"cuda-python 不可用：{e}")
    return 0


def check_audio() -> int:
    if not shutil.which("pactl"):
        line(WARN, "没有 pactl（pipewire-pulse 没装？）")
        return 0
    try:
        src = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True,
                             text=True, timeout=5).stdout
        hits = [l.split("\t")[1] for l in src.splitlines() if "Capture" in l or "capture" in l]
        if hits:
            for h in hits[:4]:
                line(OK, f"音源 {h}")
        else:
            line(BAD, "没有采集源：采集卡插了吗 / PipeWire 起了吗")
    except Exception as e:  # noqa: BLE001
        line(WARN, f"pactl 失败：{e}")
    vids = sorted(str(p) for p in Path("/dev").glob("video*"))
    line(OK if vids else WARN, f"视频节点 {vids or '无（看屏幕不可用）'}")
    return 0


def main() -> int:
    print(f"代码 {paths.HOME}\n数据 {paths.DATA}\n")
    print("== Python 依赖"); n = check_imports()
    print("\n== 模型文件"); n += check_models()
    print("\n== GPU / TensorRT"); check_gpu()
    print("\n== 音频与视频设备"); check_audio()
    print()
    if n:
        print(f"{n} 项缺失，先补齐再往下调")
        return 1
    print("依赖齐了。下一步：scripts/llm-magnitude.sh 或 scripts/llm-vllm.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
