# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Directory conventions: every default path comes from here — never hard-code absolute paths in code.

The install script puts models under the data directory. If you keep models on an external disk (recommended:
ASR 3.2G, TTS 137M, TensorRT engines and meeting corpora should not eat the root disk), one env var is enough:

    export MC_DATA=/mnt/mc-data

Layout (defaults in parentheses):

    MC_HOME  code directory (= the directory holding this file)
    MC_DATA  data root (= $MC_HOME/data)
        asr-models/        SenseVoice / Silero VAD / KWS zipformer
        tts-models/        Piper Japanese and Chinese voices
        models/            TensorRT artifacts such as rec_fp16.engine
        work-voice/        voice_server's temporary recordings
        magnitude-data/    Magnitude's dockerd data-root (the 35B weights live here)
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
