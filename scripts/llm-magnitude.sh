#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

# Run a dedicated dockerd for Magnitude (own data-root / socket, leaves the system daemon alone).
#
# Why not the main daemon: Magnitude needs --runtime nvidia, needs a data-root for 22.9G of
# weights, and must truly release memory on stop. The separate daemon means "tearing it down
# doesn't disturb other containers"; the cost is one extra systemd unit. To skip it, use the
# main daemon instead: remove DOCKER_HOST from deploy/magnitude-up.sh.
set -euo pipefail

MC_DATA="${MC_DATA:-/opt/meeting-copilot/data}"
R="$MC_DATA/magnitude-data"
mkdir -p "$R"

cat > "$R/daemon.json" <<EOF
{
  "data-root": "$R/docker",
  "exec-root": "$R/docker-exec",
  "pidfile": "$R/dockerd.pid",
  "hosts": ["unix://$R/docker.sock"],
  "containerd-namespace": "magnitude-bench",
  "iptables": false,
  "ip6tables": false,
  "bridge": "none",
  "default-runtime": "nvidia",
  "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "args": [] } },
  "storage-driver": "overlay2",
  "log-level": "info"
}
EOF

echo "写好 $R/daemon.json"
echo "接着：sudo bash deploy/install-units.sh   # 会装 magnitude-dockerd / magnitude 两个 unit"
echo
echo "不装 systemd 也能手工跑（调试用）："
echo "  sudo dockerd --config-file $R/daemon.json &"
echo "  DOCKER_HOST=unix://$R/docker.sock docker run -d --name magnitude-bench \\"
echo "      --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \\"
echo "      -v $R:/data magnitudedev/magnitude:latest"
