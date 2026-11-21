#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

# Swap the LLM on port 8000: vLLM(Qwen2.5-7B-AWQ) <-> SGLang(Qwen3.5-35B-A3B-GPTQ-Int4).
#
# Both serve as served-model-name=qwen, so voice_server / caption_server need no changes at all.
#
#   ./tools/llm_switch.sh status              # who is running now and how long it takes to come up
#   ./tools/llm_switch.sh stop                # stop the current LLM (start of the switch)
#   ./tools/llm_switch.sh start-sgl           # start 35B-A3B
#   ./tools/llm_switch.sh start-vllm          # roll back to 7B
#   ./tools/llm_switch.sh wait                # wait until /v1/models is ready
#
# Tunables (environment variables):
#   MEM=0.62   VRAM fraction. Jetson uses unified memory and 61G must still hold OCR/ASR/system,
#              0.8 would drain the box; 0.62 ~ 38G leaves a KV pool far beyond 8K context needs
#   CTX=8192   context limit. Production prompts are a few hundred tokens; 131072 is pure waste
#   SPEC=0     1 = enable NEXTN speculative decoding (combination verified in the readme; get it working first)
#
set -uo pipefail

PORT=${PORT:-8000}
VLLM_NAME=${VLLM_NAME:-vllm-qwen}
SGL_NAME=${SGL_NAME:-sgl-qwen35b}
SGL_IMAGE=${SGL_IMAGE:-mitakad/sglang:0.6.0-r36.5.tegra-aarch64-cp310-cu126-22.04-at-commit-8c8a487}
MEM=${MEM:-0.62}
CTX=${CTX:-8192}
SPEC=${SPEC:-0}
MODELS=/opt/meeting-copilot/data/models
CACHE=/opt/meeting-copilot/data/vllm-cache
LOG_DIR=/opt/meeting-copilot/logs

say() { printf '%s\n' "$*"; }

wait_ready() {
    local limit=${1:-300} t0 elapsed
    t0=$(date +%s)
    while :; do
        if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            elapsed=$(( $(date +%s) - t0 ))
            say "READY ${elapsed}s  http://127.0.0.1:${PORT}/v1/models"
            curl -s --max-time 3 "http://127.0.0.1:${PORT}/v1/models" | head -c 300; say
            return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$limit" ]; then
            say "TIMEOUT ${elapsed}s 未就绪，看日志：docker logs --tail 60 ${2:-}"
            return 1
        fi
        sleep 3
    done
}

case "${1:-}" in
status)
    say "== 容器 =="
    docker ps -a --filter "name=${VLLM_NAME}" --filter "name=${SGL_NAME}" \
        --format '{{.Names}}\t{{.Status}}'
    say "== 端口 =="
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 200; say
    else
        say "8000 无响应（停机中）"
    fi
    say "== 内存 =="
    free -g | sed -n '2p'
    ;;

stop)
    # Stop the container without removing it: on rollback docker start skips re-loading image layers, much faster
    docker stop "${VLLM_NAME}" >/dev/null 2>&1 && say "stopped ${VLLM_NAME}"
    docker stop "${SGL_NAME}"  >/dev/null 2>&1 && say "stopped ${SGL_NAME}"
    say "8000 已释放"
    ;;

start-vllm)
    # VLLM_MEM defaults to 0.40. Squeeze it to 0.25 when co-existing with Magnitude's 35B (27.9G):
    # the 7B KV pool used to be 19.94G/373k tokens while we only ever use a few thousand — pure waste.
    VLLM_MEM=${VLLM_MEM:-0.40}
    if docker inspect "${VLLM_NAME}" >/dev/null 2>&1; then
        docker start "${VLLM_NAME}" >/dev/null && say "docker start ${VLLM_NAME}（复用原容器）"
    else
        docker run -d --name "${VLLM_NAME}" --runtime nvidia --network host --gpus all \
            -v "${MODELS}/qwen2.5-7b-awq:/models/qwen:ro" \
            ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
            vllm serve /models/qwen --served-model-name qwen \
            --host 127.0.0.1 --port "${PORT}" \
            --max-model-len 4096 --gpu-memory-utilization "${VLLM_MEM}" --enforce-eager >/dev/null \
            && say "新建 ${VLLM_NAME}（mem=${VLLM_MEM}）"
    fi
    wait_ready 240 "${VLLM_NAME}"
    ;;

start-vllm-35b)
    # The two commands from the doc do not run as-is: gpu-memory-utilization 0.8/0.9 + max-model-len
    # 128000/262144 starves OCR/ASR on 61G of unified memory. Capped here at 0.55 / 8192.
    docker rm -f "${SGL_NAME}" >/dev/null 2>&1
    docker rm -f vllm-qwen35b >/dev/null 2>&1
    docker run -d --name vllm-qwen35b --runtime nvidia --network host --gpus all \
        --shm-size 16g \
        -v "${MODELS}:/models" \
        -v "${CACHE}/qwen35b:/root/.cache" \
        -e HF_HOME=/root/.cache \
        ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
        vllm serve /models/qwen35b-w4a16 \
        --served-model-name qwen --host 127.0.0.1 --port "${PORT}" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "${VLLM_MEM:-0.55}" \
        --max-model-len "${CTX}" \
        --quantization compressed-tensors \
        --max-num-seqs 8 \
        --enable-prefix-caching --enable-chunked-prefill \
        --trust-remote-code --enforce-eager \
        --limit-mm-per-prompt '{"image": 0, "video": 0}' >/dev/null \
        || { say "docker run 失败"; exit 1; }
    say "vLLM 35B 已起，等待加载（21G 权重）"
    wait_ready 600 vllm-qwen35b
    ;;

start-sgl)
    mkdir -p "${LOG_DIR}" "${CACHE}/${SGL_NAME}"
    docker rm -f "${SGL_NAME}" >/dev/null 2>&1
    args=(--host 127.0.0.1 --port "${PORT}"
          --model-path /models/qwen35a3b
          --served-model-name qwen
          --tp-size 1
          --mem-fraction-static "${MEM}"
          --context-length "${CTX}"
          --reasoning-parser qwen3
          --quantization moe_wna16
          --mamba-scheduler-strategy extra_buffer)
    envs=(-e SGLANG_DISABLE_CUDNN_CHECK=1)
    if [ "${SPEC}" = "1" ]; then
        args+=(--speculative-algorithm NEXTN
               --speculative-num-steps 3
               --speculative-eagle-topk 1
               --speculative-num-draft-tokens 4)
        # Set this only when speculation is really on: even "0" counts as "set" and turns on the v2 path
        envs+=(-e SGLANG_ENABLE_SPEC_V2=1)
    fi
    say "启动：MEM=${MEM} CTX=${CTX} SPEC=${SPEC}"
    docker run -d --name "${SGL_NAME}" --runtime nvidia --network host --gpus all \
        --shm-size 16g \
        "${envs[@]}" \
        -v "${MODELS}:/models" \
        -v "${CACHE}/${SGL_NAME}:/root/.cache" \
        "${SGL_IMAGE}" \
        python -m sglang.launch_server "${args[@]}" >/dev/null || { say "docker run 失败"; exit 1; }
    say "容器已起，加载 23G 权重通常要 1~3 分钟（首次 JIT 更久），日志：docker logs -f ${SGL_NAME}"
    wait_ready 480 "${SGL_NAME}"
    ;;

wait)
    wait_ready "${2:-480}" "${SGL_NAME}"
    ;;

*)
    sed -n '2,22p' "$0"
    exit 1
    ;;
esac
