# Model list: what is installed, why that one, and whether you can swap it

[**English**](models.md) · [中文](models.zh.md) · [日本語](models.ja.md)

Everything is offline — 3.4G in total. The "alternatives" in the table are ones we
**actually tried and rejected**, not ones we never considered.

## 1. Overview

| Purpose | Model | Size | Source | Latency (Orin) |
|---|---|---|---|---|
| ASR | **SenseVoice-Small** (packaged by sherpa-onnx, int8) | 1.1G (int8 weights only 239M) | [k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | RTF 0.04, about 290ms for one 3.3s utterance |
| Endpointing VAD | **Silero VAD** | 636K | same as above | microseconds per frame |
| Wake word | **KWS zipformer 3.3M** (wenetspeech) | 36M | [kws-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) | RTF 0.03, CPU only |
| TTS Japanese | **Piper `ja_JP-hi_fi_captain-medium`** | 74M | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | first chunk 636ms, RTF 0.135 |
| TTS Chinese | **Piper `zh_CN-huayan-medium`** | 61M | same as above | RTF 0.08~0.19 |
| OCR | **RapidOCR (PP-OCRv4 det/cls/rec)** | 16M | bundled with the pip package | CPU 4.1s/140 lines; TensorRT rec 1.7s |
| LLM | **Qwen3.5-35B-A3B** (UD-Q4_K_XL GGUF) | 22.9G | pulled by Magnitude | first token 300~470ms, see [llm-serving.md](llm-serving.md) |

`scripts/download-models.sh` fetches the first six in one line; the LLM is pulled by the
inference runtime itself.

## 2. Why these

### ASR: SenseVoice instead of Whisper

Same Japanese clip (3.29s), SenseVoice against faster-whisper:

| | SenseVoice | Whisper large-v3-turbo |
|---|---|---|
| Chinese scenarios | accurate | hears Japanese as Chinese (**gets the language flat wrong**) |
| Latency | RTF 0.04 | an order of magnitude slower |
| JA/ZH/EN/KO/YUE | one model covers all | needs large to just barely cope |

**The fatal problem is not speed but language identification**: the meeting is in
Japanese, and Whisper often decides short audio segments are Chinese — no amount of good
downstream translation can fix that. So `--lang auto` is left to SenseVoice.

### TTS: Piper instead of VOICEVOX / MeloTTS

Piper is the only one that is "pip install and go" on aarch64: runs on CPU, RTF 0.135,
first chunk 636ms. The price is that Japanese has only **one voice** (`hi_fi_captain`,
2 speakers). For better audio quality you have to move to VOICEVOX (needs a GPU/HTTP
service) or fix MeloTTS (painful to build on aarch64).

Three install traps (we hit all of them):

1. `piper-phonemize` has **no wheel** for aarch64 — but `piper-tts>=1.8.0` no longer depends on it
2. Japanese needs `pyopenjtalk`, and the official package ships source only — use **`pyopenjtalk-plus`** (there is a cp312 aarch64 wheel; the module name is still `pyopenjtalk`)
3. The CLI fails with `# channels not specified` when writing the wav header — use the Python API `PiperVoice.synthesize_wav()`

### Wake word: KWS zipformer instead of "keep ASR always on"

An always-on ASR eats CPU/GPU continuously and treats TV audio and coughs as speech. The
KWS model is only 3.3M, runs on CPU at RTF 0.03, and wakes the ASR only when it hears
「小八你好」.

Custom wake words: KWS works at the **pinyin/phoneme** level, so changing the word means
preparing a new `keywords.txt` (see the sherpa-onnx official docs), not editing one
string.

### OCR: RapidOCR instead of PaddleOCR

PaddleOCR will not install on aarch64 + CUDA 13 (no matching paddlepaddle wheels).
RapidOCR is an onnxruntime wrapper around the same PP-OCRv4 models; pip install and it
runs.

## 3. Measure these three numbers before swapping models

Whether you swap ASR / LLM / TTS, don't go by feel — measure first:

| Metric | Where to look | What counts as acceptable |
|---|---|---|
| First-token latency ttft | `ttft_ms` in `/api/status`, or 「首字」 on the page | captions < 600ms, assistant < 1s |
| Total time per utterance | `trans_ms` / `asr_ms` | captions < 1.5× utterance length, otherwise it backs up (`q_tr` climbs) |
| Segmentation completeness | whether the ASR text forms sentences | half-sentence rate < 15%, otherwise the translation is bound to be cut off |

Ready-made A/B tools (all run stubbed, no microphone needed):

```bash
.venv/bin/python tools/wav_seg_ab.py        --wav x.wav --a 0.45 --b 0.7
.venv/bin/python tools/voice_llm_ab.py      --base http://127.0.0.1:10100/inference/v1
.venv/bin/python tools/ttft_prompt_ab.py    --prompt short --prompt full
```

## 4. Licenses

**This repository's code is MIT, but the models are not.** Each model carries its own
license — check upstream before commercial use:

- SenseVoice / PP-OCR / Silero / each Piper voice: see the License field on the respective
  repository or HuggingFace model card
  (Piper voices come from different sources, **check each one**; don't assume they are all CC0)
- Qwen2.5 / Qwen3.5: see Alibaba's official model cards
- Magnitude runtime: see magnitude.dev

We do not bundle any weights; `scripts/download-models.sh` only downloads from upstream
official URLs. If your situation requires that "the models must be permissively licensed
too", prefer Apache-2.0 weights.
