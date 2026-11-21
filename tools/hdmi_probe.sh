#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

# HDMI input probe: grab frames repeatedly and tell "no signal / pure black / picture" apart
# Usage: ./hdmi_probe.sh [count] [interval seconds]
#   on a picture, save /tmp/hdmi_ok_<time>.jpg immediately and exit 0
#   exit 1 if there is no signal the whole time
#
# Key point: after the MS2109 opens the device or switches resolution, the first ~33-36 frames (~1.2s) are black.
#       So each open here discards WARMUP frames before sampling; otherwise the sample is always a warm-up black frame
#       and you draw the wrong conclusion "no signal" (this project has hit that trap).
set -u
DEV=${DEV:-/dev/video0}
SIZE=${SIZE:-1920x1080}
WARMUP=${WARMUP:-40}          # number of leading frames to discard (~1.3s @30fps)
N=${1:-20}
GAP=${2:-1}

probe_once() {
  rm -f /tmp/_probe.jpg
  # single open: capture 2.5s, take 1 frame starting at frame WARMUP
  ffmpeg -hide_banner -loglevel quiet -f v4l2 -input_format mjpeg \
         -video_size "$SIZE" -framerate 30 -i "$DEV" -t 2.5 \
         -vf "select=gte(n\,$WARMUP)" -frames:v 1 -y /tmp/_probe.jpg 2>/dev/null
  if [ ! -s /tmp/_probe.jpg ]; then
    echo "坏帧|未写出文件"; return
  fi
  # metadata=print writes to stderr, so 2>&1 is required
  stats=$(ffmpeg -hide_banner -i /tmp/_probe.jpg -vf signalstats,metadata=print -f null - 2>&1)
  ymin=$(printf '%s' "$stats" | grep -o 'YMIN=[0-9.]*' | head -1 | cut -d= -f2)
  ymax=$(printf '%s' "$stats" | grep -o 'YMAX=[0-9.]*' | head -1 | cut -d= -f2)
  yavg=$(printf '%s' "$stats" | grep -o 'YAVG=[0-9.]*' | head -1 | cut -d= -f2)
  sz=$(stat -c%s /tmp/_probe.jpg)
  if [ -z "${ymax:-}" ]; then
    echo "坏帧|无亮度统计 size=$sz"; return
  fi
  if [ "$(printf '%.0f' "$ymax")" -le 10 ]; then
    echo "无信号(纯黑)|YMIN=$ymin YAVG=$yavg YMAX=$ymax size=$sz"
  else
    echo "有画面|YMIN=$ymin YAVG=$yavg YMAX=$ymax size=$sz"
  fi
}

echo "探针 $DEV $SIZE  预热丢弃 ${WARMUP} 帧  x$N  间隔 ${GAP}s   $(date '+%H:%M:%S')"
for i in $(seq 1 "$N"); do
  res=$(probe_once)
  state=${res%%|*}
  detail=${res#*|}
  printf '[%02d] %s  %s  %s\n' "$i" "$(date +%H:%M:%S)" "$state" "$detail"
  if [ "$state" = "有画面" ]; then
    out="/tmp/hdmi_ok_$(date +%H%M%S).jpg"
    cp /tmp/_probe.jpg "$out"
    echo "==> 已保存 $out"
    exit 0
  fi
  [ "$i" -lt "$N" ] && sleep "$GAP"
done
echo "==> 全程无信号：检查 HDMI 线是否插在 IN 口、源设备是否在向该口输出"
exit 1
