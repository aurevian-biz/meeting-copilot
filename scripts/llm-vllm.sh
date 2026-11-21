#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

# Alternative path: vLLM + Qwen2.5-7B-AWQ (faster first token, less VRAM, lower quality than 35B).
#
# When to use it:
#   · the box has only 32G of RAM / or the GPU must be shared with other services
#   · you only do captions, not long answers such as "read the screen + summarize"
# Measured comparison in docs/latency.md: 35B-A3B (MoE) is 170~370ms slower on short answers but 25~35% faster on long ones.
set -euo pipefail

MC_DATA="${MC_DATA:-/opt/meeting-copilot/data}"
NAME="${NAME:-vllm-qwen}"
PORT="${PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/models/qwen}"

mkdir -p "$MC_DATA/models"
if [ ! -d "$MC_DATA/models/qwen" ]; then
    echo "先把 Qwen2.5-7B-Instruct-AWQ 放到 $MC_DATA/models/qwen"
    echo "（huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir $MC_DATA/models/qwen）"
    exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --network host --runtime nvidia \
    -v "$MC_DATA/models:/models" \
    ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
    vllm serve "$MODEL_DIR" --served-model-name qwen \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len 4096 --gpu-memory-utilization 0.25 --enforce-eager

echo "起好了。等 /health 通过后，用这一组参数跑服务："
echo "  --llm-base http://127.0.0.1:$PORT/v1 --llm-model qwen   （--llm-key 留空）"
