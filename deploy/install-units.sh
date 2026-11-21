#!/bin/bash
# 安装四个 systemd unit，并把本机相关的占位符填掉。
#
#   sudo MC_DATA=/opt/meeting-copilot/data bash deploy/install-units.sh
#   # 或者：export MC_DATA=... 后用 sudo -E bash deploy/install-units.sh
#
# 会做三件事：
#   1. 生成 /etc/meeting-copilot/env（权限 0640）：模型地址、API key、音频出口
#   2. 把 unit 里的 __HOME__ / __USER__ / __UID__ / __DATA__ 换成真实值
#   3. systemctl daemon-reload + enable（不自动 start，免得半夜把 35B 拉起来）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC_HOME="${MC_HOME:-$(dirname "$HERE")}"
MC_DATA="${MC_DATA:-$MC_HOME/data}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
RUN_UID="$(id -u "$RUN_USER")"
RUN_GID="$(id -g "$RUN_USER")"
CONF_DIR=/etc/meeting-copilot

[ "$(id -u)" -eq 0 ] || { echo "要 root：sudo bash $0"; exit 1; }

# ---- 1. env 文件
mkdir -p "$CONF_DIR"
KEY=""
if [ -f "$MC_DATA/magnitude-data/.magnitude/config.json" ]; then
    KEY="$(python3 -c "import json;print(json.load(open('$MC_DATA/magnitude-data/.magnitude/config.json'))['network']['apiKey'])" 2>/dev/null || true)"
fi
SINK="${MC_SINK:-$(pactl list short sinks 2>/dev/null | head -1 | cut -f2)}"

if [ -f "$CONF_DIR/env" ]; then
    echo "存在 $CONF_DIR/env，保留原值（要改请直接编辑）"
else
    cat > "$CONF_DIR/env" <<EOF
# meeting-copilot 运行时配置。这个文件含 LLM 的 Bearer token，别提交到 git。
MC_DATA=$MC_DATA
# Magnitude 的 OpenAI 兼容端点；用 vLLM 就改成 http://127.0.0.1:8000/v1
LLM_BASE=http://127.0.0.1:10100/inference/v1
LLM_MODEL=hf:unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
LLM_KEY=$KEY
# 「听 HDMI」和本地播报往哪个 sink 送。pactl list short sinks 里挑一个
MC_SINK=$SINK
MC_UID=$RUN_UID
MC_GID=$RUN_GID
EOF
    chmod 0640 "$CONF_DIR/env"
    chgrp "$RUN_USER" "$CONF_DIR/env" 2>/dev/null || true
fi

# ---- 2. 装 unit
GH_REPO="$(git -C "$MC_HOME" remote get-url origin 2>/dev/null | sed 's|.*github.com/||; s|\.git$||')"
[ -n "$GH_REPO" ] || GH_REPO="your-name/meeting-copilot"
for f in caption-bench voice-bench magnitude magnitude-dockerd; do
    sed -e "s|__HOME__|$MC_HOME|g" \
        -e "s|__USER__|$RUN_USER|g" \
        -e "s|__UID__|$RUN_UID|g" \
        -e "s|__DATA__|$MC_DATA|g" \
        -e "s|__GH_REPO__|$GH_REPO|g" \
        "$HERE/$f.service" > "/etc/systemd/system/$f.service"
    echo "装好 /etc/systemd/system/$f.service"
done

install -m 0755 "$HERE/magnitude-up.sh" /usr/local/bin/magnitude-up.sh

# ---- 3. 生效
systemctl daemon-reload
systemctl enable magnitude-dockerd.service magnitude.service voice-bench.service caption-bench.service >/dev/null

cat <<EOF

装好了。启动顺序（Magnitude 冷启约 3 分钟，35B 权重 22.9G）：

    sudo systemctl start magnitude          # 会连带起 magnitude-dockerd
    sudo systemctl start voice-bench        # 8095 语音助手
    sudo systemctl start caption-bench      # 8096 会议字幕

看状态：bash scripts/healthcheck.sh
EOF
