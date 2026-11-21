# 硬件：买什么、怎么接、先修哪三件事

[English](../en/hardware.md) · [**中文**](hardware.md) · [日本語](../ja/hardware.md)

整套东西的硬件只有四样，加起来不到一台笔记本的钱（不含 Orin 本体）。但**有三件事不修，
后面所有软件调参都是白费**——它们不报错，只让你的 VAD 门限失真、ASR 断句错乱、
采集卡黑屏。

## 1. 清单

| 件 | 型号 / 规格 | 作用 | 备注 |
|---|---|---|---|
| 计算单元 | Jetson AGX Orin 64GB 开发套件 | 跑 ASR / OCR / TTS / 35B-A3B | 32GB 版跑得动 35B，但会和系统抢内存 |
| HDMI 采集卡 | MacroSilicon **MS2109**（`534d:2109`） | 把演示机的画面变成 UVC 摄像头 | 免驱、几十块；1080p30 稳定 |
| USB 耳麦 | Logitech **H340** | 拾音 + 本地监听 | 单声道麦 + 3.5mm 耳机口，一条 USB 全解决 |
| 外置 SSD | 任意 USB3 SSD，≥128G | 放模型（3.4G）+ 35B 权重（22.9G） | 根盘通常只剩几个 G，别硬塞 |
| HDMI 线 ×2 / 分配器 | — | 演示机 → 采集卡 → 投影 | 有分配器才能「边投屏边采集」 |

系统：**JetPack 7.2.1 / Ubuntu 24.04 / Python 3.12**。JetPack 6.x（22.04 / Python 3.10）
也能跑，但 `pyopenjtalk-plus` 和 `tensorrt` 的 wheel 要换版本，见 [models.md](models.md)。

## 2. 接线

```
                ┌──────────────┐
   演示机 HDMI ─┤ HDMI 分配器  ├─→ 投影 / 大屏
                └──────┬───────┘
                       │ HDMI
                ┌──────┴───────┐   USB   ┌──────────────────┐
                │ MS2109 采集卡 ├────────→│                  │
                └──────────────┘         │  AGX Orin 64GB   │
                ┌──────────────┐   USB   │                  │
                │ H340 耳麦     ├────────→│  :8095 语音助手  │
                └──────────────┘         │  :8096 会议字幕  │
                                         └──────────────────┘
```

- 采集卡**直插机身 USB**，不要走扩展坞。踩过一次：扩展坞供电不足，长跑几小时后
  `/dev/video0` 掉线，`v4l2` 还在、读出来全是黑帧
- H340 的 3.5mm 口可以插耳机做监听；不插耳机时**外放会被自己的麦克风拾到**，
  可能自己把自己唤醒（见 [troubleshooting.md](troubleshooting.md#5-自己把自己唤醒)）

## 3. 三件必须先做的事

### 3.1 60Hz 工频哼声（不做，VAD 门限全废）

实测本底 **−30.6 dBFS**，频谱最强点 60Hz 且有 120Hz 谐波，0–80Hz 占 38% 能量。
这不是房间噪声，是工频/地环路。加 150Hz 4 阶高通后：

| | 高通前 | 高通后 |
|---|---|---|
| 本底 | −30.6 dBFS | **−48.8 ~ −56.6 dBFS** |
| 语音段信噪比 | 9.8 dB | **34 dB** |

语音主区在 300Hz–3kHz，切掉 150Hz 以下对 ASR 完全无损。代码里默认就带高通
（`voice_probe.py` 的 `highpass()`），**自己另写抓音代码时必须自己加**。

### 3.2 采集增益（H340 出厂是满刻度）

H340 的采集增益出厂 `44/44 = +8dB`，实测削顶 33 个样本。

```bash
amixer -c 0 cset numid=4 32     # 44 → 32，即 +8dB → 0dB
amixer -c 0 cset numid=4 44     # 恢复原值
```

坑：这个控件**没有** `Playback/Capture` 前缀，`amixer sget 'Mic Capture Volume'`
会报找不到，必须用 `numid`。用 `amixer -c 0 contents | grep -A2 numid=4` 先确认。

### 3.3 采集卡预热黑帧（不做，前 1.2 秒是黑的）

MS2109 每次 `open()` 之后有一段时间输出**全黑或噪点帧**。只丢几帧不够：
噪点帧的像素标准差 ≠ 0，用「std 判断是否全黑」会误判成有画面。

正确做法（`vision.py` 里已实现）：**连续丢 40 帧**再开始判。1080p30 下约 1.3 秒。
所以「看一眼屏幕」的耗时里固定含约 1.2 秒预热——**让它常驻抓帧**（预热的钱只付一次）
是 2026-11-21 的主要优化之一，见 [engineering-log.md](engineering-log.md)。

## 4. 音频通路（PipeWire）

采集与播报都走 PipeWire，原因很实际：**PipeWire 允许多个客户端同时读同一个 source**，
`arecord` 直连会报「设备忙」。所以：

```bash
# 抓音（不独占）
pw-record --target <source> --rate 16000 --channels 1 - > /dev/null

# 播报（指定出口，别让声音跑到 HDMI 去）
pw-play --target <sink> out.wav
```

看当前有哪些节点：

```bash
pactl list short sources    # 采集：找 USB3.0 Capture / H340
pactl list short sinks      # 播报：找 H340 的 analog-stereo
```

节点名很长（`alsa_output.usb-...H340-00.analog-stereo`），服务里用 `--local-sink`
传进去。**踩过的坑**：不给 `--target` 时，PipeWire 会把声音送到默认 sink，
而默认 sink 可能是 HDMI——于是「播报没声音」，其实声音跑到显示器去了。

> 服务由 systemd 拉起时，`pw-play` / `pw-record` 需要 `XDG_RUNTIME_DIR=/run/user/<uid>`，
> 否则**静默失效**（不报错，就是没声音）。unit 里已经写了，别删。
