#!/bin/bash
# 给 Magnitude 单独起一个 dockerd（独立 data-root / socket，不碰系统主 daemon）。
#
# 为什么不让主 daemon 跑：Magnitude 要挂 --runtime nvidia、要独占 22.9G 权重的
# data-root，还要能在停服务时把内存真正放掉。独立 daemon 的好处是「拔掉它不影响
# 机器上其他容器」，代价是多一个 systemd unit。不想要这套就直接用主 daemon：
# 把 deploy/magnitude-up.sh 里的 DOCKER_HOST 去掉即可。
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
