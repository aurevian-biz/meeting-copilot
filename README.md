# meeting-copilot

**Fully local, Japanese↔Chinese meeting captions and a voice assistant on a single edge box.**

[**English**](README.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [MIT](LICENSE)

No cloud inference. ASR, translation, TTS, screen OCR and the LLM all run on the device,
so meeting audio and screen content never leave the machine.

```
:8095  voice assistant   wake word → ASR → LLM → TTS → local playback
:8096  meeting captions  HDMI audio → segmentation → ASR → streaming MT → SSE → browser
:10100 LLM               Qwen3.5-35B-A3B (Magnitude), or any OpenAI-compatible endpoint
```

## Highlights

- **Japanese speech → Chinese subtitles** (and vice versa). Tokens stream in as they are
  produced: first token in ~400–470 ms, perceived end-to-end lag ~290 ms.
- **Wake word** plus barge-in (interrupt playback by speaking); crosstalk does not
  false-trigger.
- **"Look at the screen"**: OCR over an HDMI capture card, used to correct ASR
  homophone mistakes for proper nouns — only when a term actually matches
  (+65 ms on a hit, zero extra tokens otherwise).
- The browser only renders. Recording and playback happen on the device, so your
  laptop's microphone is irrelevant.
- SSE shared safely between multiple open pages; newest line on top, incremental DOM
  updates (no full `innerHTML` repaints).
- Works offline. The only credential is the Bearer token of your own local LLM server.

## Quick start

On a Jetson AGX Orin 64GB (JetPack 7.2) with an HDMI capture card and USB headset:

```bash
git clone https://github.com/<you>/meeting-copilot.git /opt/meeting-copilot
cd /opt/meeting-copilot
export MC_DATA=/mnt/mc-data          # models + weights on an external SSD
sudo -E bash scripts/install.sh             # deps + venv + 3.4G of models

bash scripts/llm-magnitude.sh               # dedicated dockerd config for 35B
sudo -E bash deploy/install-units.sh        # 4 systemd units + /etc/meeting-copilot/env
sudo systemctl start magnitude              # ~3 min cold start (22.9G weights)
sudo systemctl start voice-bench caption-bench

bash scripts/healthcheck.sh
```

Open `http://<device-ip>:8095` (assistant) and `:8096` (captions).

> **Read [hardware.md](docs/hardware.md) first**: 60 Hz mains hum needs a high-pass filter,
> the H340 capture gain must be lowered from +8 dB to 0 dB, and the capture card emits
> ~40 warm-up frames that must be discarded. None of these produce an error — they just
> silently invalidate every later tuning step.

Full steps: [install.md](docs/install.md). Stuck? [troubleshooting.md](docs/troubleshooting.md).

## Hardware

| Part | Example | Role |
|---|---|---|
| Edge board | Jetson AGX Orin 64GB (JetPack 7.2 / Ubuntu 24.04) | all inference |
| HDMI capture | MacroSilicon MS2109 (`534d:2109`) | screen OCR + HDMI audio |
| USB headset | Logitech H340 | mic + local monitoring |
| External SSD | ≥128 GB | 3.4G models + 22.9G LLM weights |

Also runs on x86 + NVIDIA GPU. On CUDA 13, `onnxruntime-gpu` is unusable, so OCR
acceleration goes through TensorRT ([ocr-gpu.md](docs/ocr-gpu.md)).

## Design notes

- Captions and the assistant are **separate processes**: a continuous stream and a
  turn-based dialogue have different latency budgets and thread models. OCR is shared
  via `/api/look` because `/dev/video0` is exclusive — only `voice-bench` holds it.
- Audio uses **PipeWire**, which allows multiple readers of one source (`arecord` gets
  "device busy").
- Translation streams over **SSE** with **incremental DOM** updates. The earlier
  full-`innerHTML` + pixel-`scrollY` approach drifted while a long line grew, making the
  previous line appear to vanish.

See [architecture.md](docs/architecture.md).

## Documentation

Every document exists in three languages: this table links the English files, and the
same name with `.zh.md` / `.ja.md` is the Chinese / Japanese version
(for example [hardware.zh.md](docs/hardware.zh.md), [hardware.ja.md](docs/hardware.ja.md)).
Each page carries a language switcher at the top.

| Doc | Contents |
|---|---|
| [hardware.md](docs/hardware.md) | BOM, wiring, the three physical issues to fix first |
| [install.md](docs/install.md) | Zero-to-running |
| [models.md](docs/models.md) | Model list, why each, license caveats |
| [llm-serving.md](docs/llm-serving.md) | Magnitude 35B vs vLLM 7B; prefill / prefix-cache traps |
| [ocr-gpu.md](docs/ocr-gpu.md) | 3.5× OCR with TensorRT |
| [usage.md](docs/usage.md) | UI, all flags, status fields |
| [latency.md](docs/latency.md) | Budget, measurement, hard floors |
| [testing.md](docs/testing.md) | Offline regression + A/B tools |
| [troubleshooting.md](docs/troubleshooting.md) | Silent-failure playbook |
| [engineering-log.md](docs/engineering-log.md) | All measurements and rejected approaches |

## Tests

No microphone and no live model needed — everything is stubbed:

```bash
.venv/bin/python test_caption_pipe.py        # 16 groups, incl. page JS syntax
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py
node tools/dom_render_test.js                # 26 render-layer checks
```

`tools/sse_regress.py` is a **live check**: it subscribes to a running :8096 for 90 s
and asserts no translation regressions or blanking. It is not an offline test.

## Configuration & secrets

- All paths live in [paths.py](paths.py); **no absolute paths in code**. Point `MC_DATA`
  at an external SSD and everything follows.
- The LLM Bearer token lives in `/etc/meeting-copilot/env` (mode 0640).
  **No secrets in this repo** — CI scans for `mag-*`, local paths and private IPs.
- Nothing in the repo identifies the development machine: no hostnames, usernames,
  e-mail addresses, serial numbers or LAN addresses. See [SECURITY.md](SECURITY.md).

## License

Code is **MIT** — see [LICENSE](LICENSE). Copyright (c) 2026 meeting-copilot contributors.

**Model weights are not included**; `scripts/download-models.sh` only downloads them from
upstream official URLs. SenseVoice, Piper voices, PP-OCR and Qwen each carry their own
licenses — check the model cards before commercial use.
Third-party components and their notices are listed in [NOTICE](NOTICE).

Thanks to [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
[SenseVoice](https://github.com/FunAudioLLM/SenseVoice),
[RapidOCR](https://github.com/RapidAI/RapidOCR),
[Piper](https://github.com/rhasspy/piper), [Qwen](https://github.com/QwenLM),
and [Magnitude](https://magnitude.dev).
