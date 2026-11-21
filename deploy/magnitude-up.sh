#!/bin/bash
# 开机把 Magnitude 拉起来并加载 35B-A3B。
# 只做三件事：起容器（缺了就建）→ 等 /health → 等模型 Ready。
# 由 /etc/systemd/system/magnitude.service 调用，不手工跑。
#
# 依赖环境变量（都在 /etc/meeting-copilot/env 里）：
#   MC_DATA       数据根，默认 /opt/meeting-copilot/data
#   MC_MODEL_ID   模型标识，默认 Qwen3.5-35B-A3B 的 GGUF
#   MC_UID/GID    容器里跑的 uid/gid（要和宿主上存档文件的人一致）
set -uo pipefail

R="${MC_DATA:-/opt/meeting-copilot/data}/magnitude-data"
ID="${MC_MODEL_ID:-hf:unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
UID_GID="${MC_UID:-$(id -u)}:${MC_GID:-$(id -g)}"
export DOCKER_HOST="unix://${R}/docker.sock"

say() { echo "[magnitude-up] $*"; }

# apiKey 是 magnitude 首次启动时自己生成的，不是我们配的 —— 每次现读
KEY=$(python3 -c "import json;print(json.load(open('${R}/.magnitude/config.json'))['network']['apiKey'])" 2>/dev/null)

if ! docker inspect magnitude-bench >/dev/null 2>&1; then
    say "容器不存在，新建"
    docker run -d --name magnitude-bench --restart unless-stopped --network host \
        --runtime nvidia --user "${UID_GID}" -e NVIDIA_VISIBLE_DEVICES=all \
        -v "${R}:/data" magnitudedev/magnitude:latest || exit 1
else
    docker start magnitude-bench >/dev/null 2>&1 || true
fi

for _ in $(seq 1 90); do
    curl -sf -m 2 -H "Authorization: Bearer ${KEY}" http://127.0.0.1:10100/health >/dev/null && break
    sleep 2
done
curl -sf -m 2 -H "Authorization: Bearer ${KEY}" http://127.0.0.1:10100/health >/dev/null \
    || { say "health 超时"; exit 1; }
say "服务已就绪"

# 模型不会自动加载，必须显式 load；22.9G 权重冷加载约 150s
docker exec magnitude-bench magnitude models load "${ID}" >/dev/null 2>&1 || true
for _ in $(seq 1 120); do
    if docker exec magnitude-bench magnitude models status "${ID}" 2>/dev/null | grep -q "Ready"; then
        say "模型 Ready"
        exit 0
    fi
    sleep 3
done
say "模型加载超时"
exit 1
