# Install from scratch

[**English**](install.md) · [中文](install.zh.md) · [日本語](install.ja.md)

This assumes you already have an AGX Orin with JetPack flashed, can ssh into it, and have the capture card and headset plugged in. No network is needed for inference at any point — **ASR / TTS / OCR / LLM all run locally**, so the box keeps working offline once installed.

## 0. Pre-flight check

```bash
python3 -V                    # 3.10+ (JetPack 7 is 3.12)
nvidia-smi | head -5          # driver / CUDA healthy
pactl list short sources      # capture card and headset visible
ls /dev/video*                # /dev/video0 present
df -h /                       # ideally 5G+ free on the root disk
```

## 1. Fetch the code + install dependencies

```bash
git clone https://github.com/<you>/meeting-copilot.git /opt/meeting-copilot
cd /opt/meeting-copilot

# data (3.4G model + 22.9G of weights later) belongs on the external drive
export MC_DATA=/mnt/mc-data

sudo -E bash scripts/install.sh
```

`install.sh` does four things: apt-install PipeWire/build tools → create `.venv` → `pip install -r requirements.txt` → call `scripts/download-models.sh` to fetch the models → finally run `tools/check_env.py` as a self-check.

**It does not install TensorRT / cuda-python**: those two must match JetPack's CUDA exactly, and a wrong pip version **silently** falls back to CPU (OCR goes from 300ms to 3s, and the log says nothing). To get GPU-accelerated OCR, first confirm `python -c "import tensorrt, cuda"` passes, then build the engine as described in [ocr-gpu.md](ocr-gpu.md).

Installing by hand works too:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download-models.sh
```

## 2. Start a large model

Two routes, **pick one first**:

| | Magnitude + Qwen3.5-35B-A3B | vLLM + Qwen2.5-7B-AWQ |
|---|---|---|
| VRAM / RAM | 22.9G of weights, ~26G resident | ~6G |
| First token | 300~470ms (short sentences) | **90ms** |
| Long answers | **25~35% faster** | slower |
| Mixed Chinese/Japanese, hallucination | clean | will emit 「会議的时间确认 needed。」 |
| Starting the service | ~3 min cold start | ~40 s |

```bash
# A. 35B-A3B (recommended: quality first; used for both captions and the assistant)
bash scripts/llm-magnitude.sh            # generate the dockerd config
sudo bash deploy/install-units.sh        # install the magnitude / magnitude-dockerd units
sudo systemctl start magnitude           # ~3 min cold start (22.9G of weights)

# B. 7B (latency first / tight memory)
bash scripts/llm-vllm.sh
# then change /etc/meeting-copilot/env to:
#   LLM_BASE=http://127.0.0.1:8000/v1
#   LLM_MODEL=qwen
#   LLM_KEY=
```

For the details (why a 35B MoE is the better deal, why the prefix cache never hits), see [llm-serving.md](llm-serving.md).

## 3. Install the services

```bash
sudo MC_DATA=/mnt/mc-data bash deploy/install-units.sh
sudo systemctl start voice-bench      # 8095 voice assistant
sudo systemctl start caption-bench    # 8096 meeting captions
```

`install-units.sh` generates `/etc/meeting-copilot/env` (0640, holds the LLM Bearer token) and substitutes the real values for `__HOME__` / `__USER__` / `__UID__` in the units. **The token is not in the repository**; Magnitude generates it and the script reads and fills it in at install time.

## 4. Verify

```bash
bash scripts/healthcheck.sh
```

Expected output:

```
== 服务
  ✓ magnitude active（重启 0 次）
  ✓ voice-bench active
  ✓ caption-bench active
== 字幕链路的关键指标
  段 12  错 0  q_asr=0 q_tr=0 dropped=0
  首字 443ms  译文 1005ms  延迟 290ms
```

Then one offline regression (no microphone, no models online, runs stubbed):

```bash
.venv/bin/python test_caption_pipe.py      # 16 groups
node tools/dom_render_test.js              # 26 checks in the render layer
```

## 5. Open the UI

```
http://<LAN-IP>:8095   voice assistant (talk into the headset; wake word 「小八你好」)
http://<LAN-IP>:8096   meeting captions (press 「开始」; the status light in the top-left should turn 「运行中」)
```

**If the page is stuck on 「连接中…」 (connecting…), look here first**: that means the JS never ran (it is not a network problem). Checks like `curl -s localhost:8096/ | node --check` are already in CI; locally you can just run `.venv/bin/python test_caption_pipe.py`, where `test_page_js_syntax` runs `node --check` over the page.

## 6. Three ways the install fails

| Symptom | Cause | Fix |
|---|---|---|
| `piper-phonemize` won't install | old piper versions depend on it and there is no aarch64 wheel | use `piper-tts>=1.8.0` (already required by this repo) |
| Japanese TTS reports `pyopenjtalk` not found | the official package is source-only and fails to compile | switch to `pyopenjtalk-plus` (module name unchanged) |
| The service comes up but there is no sound | missing `XDG_RUNTIME_DIR` | don't delete that line in the unit; export it yourself when running by hand |

More in [troubleshooting.md](troubleshooting.md).
