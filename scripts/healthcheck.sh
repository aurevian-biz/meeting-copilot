#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

# One command to see whether the whole stack is alive. Run this first when debugging, before digging through logs.
set -uo pipefail

MC_DATA="${MC_DATA:-/opt/meeting-copilot/data}"
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
no() { printf '  \033[31m✗\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }

echo "== 服务"
for u in magnitude-dockerd magnitude voice-bench caption-bench; do
    if systemctl is-active --quiet "$u" 2>/dev/null; then
        n=$(systemctl show "$u" -p NRestarts --value 2>/dev/null)
        ok "$u active（重启 $n 次）"
    else
        warn "$u 未运行"
    fi
done

echo "== 端口"
for p in 10100 8095 8096 8000; do
    if curl -sf -m 2 -o /dev/null "http://127.0.0.1:$p/" || curl -sf -m 2 -o /dev/null "http://127.0.0.1:$p/health"; then
        ok "$p 有响应"
    else
        warn "$p 无响应"
    fi
done

echo "== 采集设备"
if command -v pw-record >/dev/null; then
    src=$(pactl list short sources 2>/dev/null | grep -i -m1 capture | cut -f2)
    [ -n "$src" ] && ok "PipeWire 源：$src" || no "找不到采集源（采集卡插了吗）"
fi
ls /dev/video* >/dev/null 2>&1 && ok "视频节点：$(ls /dev/video* | tr '\n' ' ')" || warn "没有 /dev/video*（看屏幕功能不可用）"

echo "== 模型"
for p in "$MC_DATA/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx" \
         "$MC_DATA/asr-models/vad/silero_vad.onnx" \
         "$MC_DATA/tts-models/ja_JP-hi_fi_captain-medium.onnx" \
         "$MC_DATA/tts-models/zh_CN-huayan-medium.onnx"; do
    [ -f "$p" ] && ok "$(basename "$p")" || no "缺 $(basename "$p")（跑 scripts/download-models.sh）"
done

echo "== 字幕链路的关键指标"
st=$(curl -sf -m 3 http://127.0.0.1:8096/api/status || true)
if [ -n "$st" ]; then
    python3 - "$st" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
print(f"  段 {s.get('segments')}  错 {s.get('errors')}  "
      f"q_asr={s.get('q_asr')} q_tr={s.get('q_tr')} dropped={s.get('dropped')}")
print(f"  首字 {s.get('ttft_ms')}ms  译文 {s.get('trans_ms')}ms  延迟 {s.get('lag_ms')}ms")
sc = s.get("screen")
if sc:
    print(f"  屏幕 {sc.get('terms')} 词 命中 {sc.get('hits')} 次  "
          f"每次 {sc.get('last_ms')}ms  错误 {sc.get('last_err')}")
if s.get("q_tr"):
    print("  ! q_tr 涨说明翻译跟不上：先降 --ctx-lines，再看 LLM 是否在抢显存")
if s.get("dropped"):
    print("  ! dropped 非 0 说明队列在丢段：延迟已经超过实时，务必处理")
PY
else
    no "8096 没有 /api/status"
fi
