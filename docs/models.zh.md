# 模型清单：装了什么、为什么是它、能不能换

[English](models.md) · [**中文**](models.zh.md) · [日本語](models.ja.md)

全部离线，一共 3.4G。表里的「替代方案」都是**真试过并否掉**的，不是没考虑。

## 1. 总表

| 用途 | 模型 | 大小 | 来源 | 延迟（Orin） |
|---|---|---|---|---|
| ASR | **SenseVoice-Small**（sherpa-onnx 打包，int8） | 1.1G（int8 权重仅 239M） | [k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | RTF 0.04，一句 3.3s 音频约 290ms |
| 判停 VAD | **Silero VAD** | 636K | 同上 | 每帧微秒级 |
| 唤醒词 | **KWS zipformer 3.3M**（wenetspeech） | 36M | [kws-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) | RTF 0.03，纯 CPU |
| TTS 日语 | **Piper `ja_JP-hi_fi_captain-medium`** | 74M | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | 首块 636ms，RTF 0.135 |
| TTS 中文 | **Piper `zh_CN-huayan-medium`** | 61M | 同上 | RTF 0.08~0.19 |
| OCR | **RapidOCR (PP-OCRv4 det/cls/rec)** | 16M | pip 包自带 | CPU 4.1s/140 行；TensorRT rec 1.7s |
| LLM | **Qwen3.5-35B-A3B**（UD-Q4_K_XL GGUF） | 22.9G | Magnitude 拉取 | 首字 300~470ms，见 [llm-serving.md](llm-serving.zh.md) |

`scripts/download-models.sh` 一行下载前六个；LLM 由推理运行时自己拉。

## 2. 为什么是这些

### ASR：SenseVoice 而不是 Whisper

同一段日语（3.29s），SenseVoice 与 faster-whisper 对比：

| | SenseVoice | Whisper large-v3-turbo |
|---|---|---|
| 中文场景 | 准确 | 会把日语听成中文（**直接翻错语言**） |
| 延迟 | RTF 0.04 | 慢一个量级 |
| 中日英韩粤 | 一个模型全包 | 要 large 才勉强 |

**致命问题不是速度而是语言判定**：会议是日语，Whisper 在小段音频上经常判成中文，
后面再好的翻译也没救。所以 `--lang auto` 交给 SenseVoice。

### TTS：Piper 而不是 VOICEVOX / MeloTTS

Piper 是唯一在 aarch64 上「pip 装完就能用」的：CPU 跑、RTF 0.135、首块 636ms。
代价是日语只有**一个音色**（`hi_fi_captain`，2 个说话人）。想要更好音质只能换
VOICEVOX（要跑 GPU/HTTP 服务）或修 MeloTTS（aarch64 编译麻烦）。

三个安装坑（都踩过）：

1. `piper-phonemize` 在 aarch64 **没有 wheel** —— 但 `piper-tts>=1.8.0` 已经不依赖它
2. 日语要 `pyopenjtalk`，官方包只有源码 —— 用 **`pyopenjtalk-plus`**（cp312 aarch64 有 wheel，模块名仍是 `pyopenjtalk`）
3. CLI 写 wav 头会报 `# channels not specified` —— 用 Python API `PiperVoice.synthesize_wav()`

### 唤醒词：KWS zipformer 而不是「一直开 ASR」

常驻 ASR 要一直吃 CPU/GPU，且会把电视声音、咳嗽都当成话。KWS 模型只有 3.3M，
纯 CPU RTF 0.03，只在命中「小八你好」时才唤醒 ASR。

自定义唤醒词：KWS 是**拼音/音素**级别的，改词要重新准备 `keywords.txt`
（见 sherpa-onnx 官方文档），不是改一行字符串。

### OCR：RapidOCR 而不是 PaddleOCR

PaddleOCR 在 aarch64 + CUDA 13 上装不动（paddlepaddle 没有对应轮子）。
RapidOCR 是同一套 PP-OCRv4 模型的 onnxruntime 封装，pip 装完就能跑。

## 3. 换模型前先量这三个数

换 ASR / LLM / TTS 都别凭感觉，先量：

| 指标 | 在哪看 | 什么算「能接受」 |
|---|---|---|
| 首字延迟 ttft | `/api/status` 的 `ttft_ms` 或页面「首字」 | 字幕 < 600ms，助手 < 1s |
| 单句总耗时 | `trans_ms` / `asr_ms` | 字幕 < 1.5× 句长，否则会积压（`q_tr` 涨） |
| 断句完整性 | ASR 文本是否成句 | 半句率 < 15%，否则译文必然断章 |

现成的 A/B 工具（都能打桩跑，不需要麦克风）：

```bash
.venv/bin/python tools/wav_seg_ab.py        --wav x.wav --a 0.45 --b 0.7
.venv/bin/python tools/voice_llm_ab.py      --base http://127.0.0.1:10100/inference/v1
.venv/bin/python tools/ttft_prompt_ab.py    --prompt short --prompt full
```

## 4. 许可证

**本仓库代码是 MIT，但模型不是**。各模型有自己的许可证，商用前请核对上游：

- SenseVoice / PP-OCR / Silero / Piper 各音色：见其仓库或 HuggingFace 模型卡的 License 字段
  （Piper 各音色来源不同，**逐个看**，别默认全是 CC0）
- Qwen2.5 / Qwen3.5：见阿里官方模型卡
- Magnitude 运行时：见 magnitude.dev

我们不打包任何权重，`scripts/download-models.sh` 只是从上游官方地址下载。
如果你的场景要求「模型也必须宽松许可」，优先挑 Apache-2.0 的权重。
