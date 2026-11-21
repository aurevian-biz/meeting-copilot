#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

# One-shot install: system deps -> venv -> Python deps -> directories -> self-check.
# Usage:
#   sudo bash scripts/install.sh            # code lives in /opt/meeting-copilot, data too
#   MC_DATA=/media/ssd/mc-data sudo -E bash scripts/install.sh   # data on an external disk
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MC_HOME="${MC_HOME:-/opt/meeting-copilot}"
MC_DATA="${MC_DATA:-$MC_HOME/data}"
PY="${PYTHON:-python3}"

say() { printf '\n== %s\n' "$*"; }

say "代码位置 $HERE → $MC_HOME"
if [ "$(readlink -f "$HERE")" != "$(readlink -f "$MC_HOME")" ]; then
    mkdir -p "$(dirname "$MC_HOME")"
    rm -rf "$MC_HOME"
    cp -a "$HERE" "$MC_HOME"
fi
mkdir -p "$MC_DATA"/{asr-models,tts-models,models,work-voice}

say "系统依赖（PipeWire 工具 / 构建 pyopenjtalk 需要的 cmake）"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3-venv python3-dev build-essential cmake \
    pipewire pipewire-bin pulseaudio-utils alsa-utils \
    curl bzip2 ffmpeg

say "Python venv（$PY）"
if [ ! -x "$MC_HOME/.venv/bin/python" ]; then
    "$PY" -m venv "$MC_HOME/.venv"
fi
"$MC_HOME/.venv/bin/pip" install -q --upgrade pip wheel
# numpy 2.x has no legacy ABI issues on aarch64, but sherpa-onnx 2.0.1 requires <3
"$MC_HOME/.venv/bin/pip" install -q -r "$MC_HOME/requirements.txt"

say "模型"
bash "$MC_HOME/scripts/download-models.sh"

say "自检"
MC_DATA="$MC_DATA" "$MC_HOME/.venv/bin/python" "$MC_HOME/tools/check_env.py" || true

cat <<EOF

安装完成。接下来：

  1) 下载/启动 LLM（二选一）
       bash scripts/llm-magnitude.sh      # 35B-A3B，质量优先
       bash scripts/llm-vllm.sh           # 7B AWQ，延迟优先
  2) 装 systemd 单元
       sudo MC_HOME=$MC_HOME MC_DATA=$MC_DATA bash deploy/install-units.sh
  3) 打开界面
       http://<本机IP>:8095  语音助手
       http://<本机IP>:8096  会议字幕

第一次跑之前请务必看 docs/hardware.md 里的两件事：60Hz 工频哼声要高通、
USB 采集卡要丢预热帧。这两条不做，后面所有调参都是白费。
EOF
