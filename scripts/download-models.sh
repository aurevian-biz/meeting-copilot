#!/bin/bash
# 下载全部离线模型到 $MC_DATA（默认 /opt/meeting-copilot/data），可重复执行。
#
# 为什么要单独一个脚本：一共 3.4G，其中 SenseVoice 1.0G、KWS 30M、Piper 78M+64M。
# 都放在外置盘上，根盘别动（JetPack 根盘通常只剩几个 G）。
set -euo pipefail

MC_DATA="${MC_DATA:-/opt/meeting-copilot/data}"
ASR="$MC_DATA/asr-models"
TTS="$MC_DATA/tts-models"
mkdir -p "$ASR" "$TTS"

say() { printf '\n== %s\n' "$*"; }
get() { [ -e "$2" ] && { echo "跳过（已存在）$2"; return; }; curl -fL --retry 3 --progress-bar -o "$2" "$1"; }

say "SenseVoice（中日英韩粤，离线 ASR，1.0G）"
if [ ! -d "$ASR/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" ]; then
    get https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2 \
        /tmp/sensevoice.tar.bz2
    tar -xjf /tmp/sensevoice.tar.bz2 -C "$ASR" && rm -f /tmp/sensevoice.tar.bz2
else
    echo "跳过（已存在）SenseVoice"
fi

say "Silero VAD（643KB，判停用）"
mkdir -p "$ASR/vad"
get https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx "$ASR/vad/silero_vad.onnx"

say "KWS zipformer（3.3M，唤醒词「小八你好」）"
if [ ! -d "$ASR/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01" ]; then
    get https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2 \
        /tmp/kws.tar.bz2
    tar -xjf /tmp/kws.tar.bz2 -C "$ASR" && rm -f /tmp/kws.tar.bz2
else
    echo "跳过（已存在）KWS"
fi

say "Piper 音色：日语 hi_fi_captain（2 说话人）+ 中文 huayan"
HF=https://huggingface.co/rhasspy/piper-voices/resolve/main
get "$HF/ja/ja_JP/hi_fi_captain/medium/ja_JP-hi_fi_captain-medium.onnx" \
    "$TTS/ja_JP-hi_fi_captain-medium.onnx"
get "$HF/ja/ja_JP/hi_fi_captain/medium/ja_JP-hi_fi_captain-medium.onnx.json" \
    "$TTS/ja_JP-hi_fi_captain-medium.onnx.json"
get "$HF/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx" \
    "$TTS/zh_CN-huayan-medium.onnx"
get "$HF/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json" \
    "$TTS/zh_CN-huayan-medium.onnx.json"

say "完成"
du -sh "$ASR" "$TTS"
echo
echo "LLM 不在这里下载：35B-A3B 由 Magnitude 自己拉（见 docs/llm-serving.md），"
echo "或者用 vLLM 挂任何 OpenAI 兼容的小模型（--llm-model qwen 走 8000）。"
