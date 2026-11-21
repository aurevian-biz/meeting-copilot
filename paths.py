"""目录约定：所有默认路径都从这里来，别在代码里写死绝对路径。

安装脚本会把模型放到数据目录下。如果你把模型放在外置盘（推荐：ASR 模型 3.2G、
TTS 137M、TensorRT engine 和会话语料都别占根盘），只要设一个环境变量就行：

    export MC_DATA=/media/ssd/meeting-copilot-data

布局（括号里是默认值）：

    MC_HOME  代码目录（= 本文件所在目录）
    MC_DATA  数据根（= $MC_HOME/data）
        asr-models/        SenseVoice / Silero VAD / KWS zipformer
        tts-models/        Piper 日语、中文音色
        models/            rec_fp16.engine 之类的 TensorRT 产物
        work-voice/        voice_server 的临时录音
        magnitude-data/    Magnitude 的 dockerd data-root（35B 权重在这里）
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MC_DATA") or HOME / "data")

ASR_DIR = DATA / "asr-models"
TTS_DIR = DATA / "tts-models"
ENGINE_DIR = DATA / "models"
WORK_DIR = DATA / "work-voice"

SENSEVOICE_DIR = ASR_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
VAD_MODEL = ASR_DIR / "vad" / "silero_vad.onnx"
KWS_DIR = ASR_DIR / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
PIPER_JA = TTS_DIR / "ja_JP-hi_fi_captain-medium.onnx"
PIPER_ZH = TTS_DIR / "zh_CN-huayan-medium.onnx"

__all__ = ["HOME", "DATA", "ASR_DIR", "TTS_DIR", "ENGINE_DIR", "WORK_DIR",
           "SENSEVOICE_DIR", "VAD_MODEL", "KWS_DIR", "PIPER_JA", "PIPER_ZH"]
