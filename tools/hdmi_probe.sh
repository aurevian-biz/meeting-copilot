#!/usr/bin/env bash
# HDMI 输入探针：连续抓帧，区分「无信号 / 纯黑 / 有画面」
# 用法：./hdmi_probe.sh [次数] [间隔秒]
#   有画面时立刻保存 /tmp/hdmi_ok_<时间>.jpg 并退出 0
#   全程无信号退出 1
#
# 关键：MS2109 每次打开设备/切换分辨率后，前 ~33-36 帧（约 1.2s）是黑帧。
#       因此这里单次打开后先丢弃 WARMUP 帧再取样，否则抓到的一定是预热黑帧，
#       会得出「无信号」的错误结论（本项目踩过这个坑）。
set -u
DEV=${DEV:-/dev/video0}
SIZE=${SIZE:-1920x1080}
WARMUP=${WARMUP:-40}          # 丢弃的前置帧数（约 1.3s @30fps）
N=${1:-20}
GAP=${2:-1}

probe_once() {
  rm -f /tmp/_probe.jpg
  # 单次打开：抓 2.5 秒，从第 WARMUP 帧起取 1 帧
  ffmpeg -hide_banner -loglevel quiet -f v4l2 -input_format mjpeg \
         -video_size "$SIZE" -framerate 30 -i "$DEV" -t 2.5 \
         -vf "select=gte(n\,$WARMUP)" -frames:v 1 -y /tmp/_probe.jpg 2>/dev/null
  if [ ! -s /tmp/_probe.jpg ]; then
    echo "坏帧|未写出文件"; return
  fi
  # metadata=print 走 stderr，必须 2>&1
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
