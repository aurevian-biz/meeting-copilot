# meeting-copilot

**一台边缘设备上跑完的日中双语会议字幕 + 语音助手，全本地、零云端推理。**

[日本語](README.md) · [English](README.en.md) · [MIT](LICENSE)

语音识别、翻译、语音合成、屏幕 OCR、大模型推理全在本机完成。会议的声音和画面都不出机器，
适合内网会议、涉密内容，以及「没网也要能用」的现场。

```
:8095  语音助手   唤醒 → ASR → LLM → TTS → 本机耳机播报
:8096  会议字幕   HDMI 音频 → 断句 → ASR → 流式翻译 → SSE → 浏览器
:10100 LLM        Qwen3.5-35B-A3B（Magnitude），或任何 OpenAI 兼容端点
```

## 能做什么

- **日语发言出中文字幕**（反向亦可）。译文逐 token 流式显示，首字约 400~470ms，
  端到端感知延迟约 290ms；`q_asr=q_tr=dropped=0` 时表示跟得上实时
- **唤醒词「小八你好」**常驻监听，播报中插话即停（barge-in），串音不误触
- **「看一眼屏幕」**：HDMI 采集卡 OCR。屏幕上的专有名词只在「读音相似、写法不同」
  命中时才递给翻译纠正 ASR 同音错字（命中 +65ms，不命中 0 token）
- **浏览器只负责显示**：录音和播报都在设备上，不受笔记本麦克风影响
- 两个页面同时开不会互相抢译文补丁；最新一条在最上，增量 DOM 渲染
- 断网可用。模型全部本地，唯一的密钥是本机推理服务自己的 Bearer token

## 快速开始

刷好 JetPack 的 AGX Orin（64GB）上插好 HDMI 采集卡和 USB 耳麦：

```bash
git clone https://github.com/<you>/meeting-copilot.git /opt/meeting-copilot
cd /opt/meeting-copilot
export MC_DATA=/media/ssd/mc-data          # 模型与权重放外置盘
sudo -E bash scripts/install.sh             # 依赖 + venv + 3.4G 模型

bash scripts/llm-magnitude.sh               # 生成 35B 用的独立 dockerd 配置
sudo -E bash deploy/install-units.sh        # 装 4 个 unit + /etc/meeting-copilot/env
sudo systemctl start magnitude              # 冷启约 3 分钟（22.9G 权重）
sudo systemctl start voice-bench caption-bench

bash scripts/healthcheck.sh
```

浏览器打开 `http://<设备IP>:8095`（助手）和 `:8096`（字幕）。

> **动手前务必先看 [hardware.md](docs/hardware.md) 的三件事**：60Hz 工频哼声要高通、
> H340 采集增益要从 +8dB 降到 0dB、采集卡要丢 40 帧预热黑帧。
> 这三条不修，后面所有调参都是白费，而且它们**都不报错**。

完整步骤见 [install.md](docs/install.md)，卡住先看
[troubleshooting.md](docs/troubleshooting.md)。

## 硬件

| 部件 | 型号 | 用途 |
|---|---|---|
| 计算单元 | Jetson AGX Orin 64GB（JetPack 7.2 / Ubuntu 24.04 / Python 3.12） | 全部推理 |
| HDMI 采集卡 | MacroSilicon MS2109（`534d:2109`） | 屏幕 OCR + HDMI 音频 |
| USB 耳麦 | Logitech H340 | 拾音 + 本地监听 |
| 外置 SSD | ≥128G | 模型 3.4G + 35B 权重 22.9G |

x86 + NVIDIA GPU 也能跑；CUDA 13 下 `onnxruntime-gpu` 用不了，OCR 加速改走
TensorRT（见 [ocr-gpu.md](docs/ocr-gpu.md)）。

## 架构要点

```
HDMI ─→ 采集卡 ─┬─→ vision.py（常驻抓帧 / OCR / TensorRT rec）─→ /api/look
                └─→ 音频 ─→ Segmenter ─→ SenseVoice ─→ 流式翻译 ─→ SSE
USB 麦 ─→ KWS ─→ ASR ─→ LLM ─→ 按句 TTS ─→ pw-play ─→ 耳机
                          │
                Qwen3.5-35B-A3B（Magnitude :10100），可换成 vLLM 7B
```

- **字幕和助手分两个进程**：一个要不断流、一个要一问一答，延迟预算和线程模型完全不同。
  OCR 通过 `/api/look` 共享，因为 `/dev/video0` 是独占设备，只有 voice-bench 常驻持有
- **采集走 PipeWire**：允许多个客户端读同一个 source；`arecord` 会报「设备忙」
- **译文 SSE + 增量 DOM**：早期全量 `innerHTML` + 像素 `scrollY` 回填，
  长句逐字变高时锚点漂移，表现为「下一句一来上一句整行消失」

详见 [architecture.md](docs/architecture.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [hardware.md](docs/hardware.md) | BOM、接线、必须先修的三个物理问题 |
| [install.md](docs/install.md) | 从零装一遍 |
| [models.md](docs/models.md) | 模型清单、为什么选它、许可证提醒 |
| [llm-serving.md](docs/llm-serving.md) | Magnitude 35B vs vLLM 7B、prefill / 前缀缓存的坑 |
| [ocr-gpu.md](docs/ocr-gpu.md) | TensorRT 把 OCR 提速 3.5 倍 |
| [usage.md](docs/usage.md) | 界面操作、全部参数、状态字段 |
| [latency.md](docs/latency.md) | 延迟预算、量法、哪些是硬地板 |
| [testing.md](docs/testing.md) | 离线回归与 A/B 工具 |
| [troubleshooting.md](docs/troubleshooting.md) | 静默失效类故障合集 |
| [engineering-log.md](docs/engineering-log.md) | 全部实测数据与被否掉的方案（约 65KB） |

## 测试

不需要麦克风、不需要模型在线，全部打桩：

```bash
.venv/bin/python test_caption_pipe.py        # 16 组（含页面 JS 语法）
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py
node tools/dom_render_test.js                # 渲染层 26 项
```

> `tools/sse_regress.py` 是**在线长跑自检**：会连到正在跑的 8096 订阅 90 秒，
> 检查译文是否 0 回退、0 变空，不属于离线测试。

## 配置与安全

- 所有目录约定集中在 [paths.py](paths.py)，**代码里不写死绝对路径**；
  模型放外置盘只需设一个 `MC_DATA`
- LLM 的 Bearer token 放在 `/etc/meeting-copilot/env`（0640），
  **仓库不含任何密钥**。CI 每次都会扫 `mag-*` / 本机路径 / 内网 IP
- 首次部署可先复制 `conf/env.example` 核对参数

## 许可证

代码 **MIT**；**模型权重不包含在仓库里**，`scripts/download-models.sh` 只从上游官方地址下载。
SenseVoice / Piper 各音色 / PP-OCR / Qwen 等各有自己的许可证，商用前请逐个核对模型卡。

致谢：[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、
[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)、
[RapidOCR](https://github.com/RapidAI/RapidOCR)、
[Piper](https://github.com/rhasspy/piper)、[Qwen](https://github.com/QwenLM)、
[Magnitude](https://magnitude.dev)。
