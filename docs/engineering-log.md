# Engineering log: where every number came from

[**English**](engineering-log.md) · [中文](engineering-log.zh.md) · [日本語](engineering-log.ja.md)

> This document is the raw record of the project's development process, kept in chronological order:
> "what we tried, what number we measured, which path we finally took".
> It is not introductory material — to get running quickly see [../README.md](../README.md) and [install.md](install.md).
> Failed approaches and measured data are kept here because every default value in this system has a set of
> rejected alternatives behind it, and changing one number degrades it (for example, moving `--max-s` from 6.0
> to 7.0 makes hallucinations start appearing on the same piece of audio).

Voice dialogue pipeline on a Jetson AGX Orin 64GB. Currently in a **usable stage**: capture → ASR → LLM → TTS is closed into one loop, and there is a browser test bench you can talk to directly.

## 1. Current state

| Component | Status | Location |
|---|---|---|
| Microphone | ✅ Logitech H340 USB headset (card 0) | `plughw:CARD=H340,DEV=0` |
| ASR · SenseVoice | ✅ RTF 0.04, can switch zh/ja | `asr-models/sherpa-onnx-sense-voice-.../model.int8.onnx` |
| VAD · Silero | ✅ model 643KB, decides end of speech | `asr-models/vad/silero_vad.onnx` |
| TTS · Piper Japanese | ✅ first chunk 636ms, RTF 0.135 | `tts-models/ja_JP-hi_fi_captain-medium.onnx` |
| TTS · Piper Chinese | ✅ RTF 0.08–0.19 | `tts-models/zh_CN-huayan-medium.onnx` |
| LLM · Qwen2.5-7B AWQ | ✅ 25 tok/s, first token 90ms | `models/qwen2.5-7b-awq` (5.2G) served through vLLM 0.19 |
| Wake · KWS zipformer | ✅ wake word 「小八你好」, RTF 0.03, CPU only | `asr-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01` |
| Barge-in | ✅ speaking during playback stops it; crosstalk does not false-trigger | see 4.6, `BargeMonitor` |
| Echo discrimination | ✅ crosstalk during playback excluded by ASR text overlap, no AEC | see 4.6; this is what prevents self-oscillation on speaker |
| HDMI capture · MS2109 | ✅ 1080p30 stable, must discard 1.2s of warm-up frames | see 4.7, `/dev/video0` |
| See screen · capture-card OCR | ✅ RapidOCR 5.9s / 54 lines, triggered by voice or button | see 4.8, `vision.py` |
| Resident service | ✅ systemd `voice-bench` | port 8095, restarted automatically |

Disk: `asr-models` 3.1G + `tts-models` 137M + weights 5.2G, venvs all on an external disk, root disk has 5.9G free.

## 1.1 Three pitfalls installing TTS (all stepped on)

1. **`piper-phonemize` has no aarch64 wheel** and cannot be installed — but `piper-tts` 1.8.0 no longer depends on it and installs directly (`piper_tts-1.8.0-cp39-abi3-manylinux_2_28_aarch64.whl`)
2. **Japanese needs `pyopenjtalk`**, and the official `pyopenjtalk` only ships a source package (you have to compile OpenJTalk). Switched to **`pyopenjtalk-plus`**, which has a cp312 aarch64 wheel; the module name is still `pyopenjtalk`, so it is a drop-in replacement
3. **The CLI reports `# channels not specified` when writing the wav header** — switched to the Python API `PiperVoice.synthesize_wav()`, see `tts_probe.py`

Piper's Japanese has only **one voice**: `ja_JP-hi_fi_captain-medium` (2 speakers). For better audio quality the only options are to switch to VOICEVOX or to fix MeloTTS.

## 2. Capture: two problems that must be handled first

### 2.1 60Hz mains hum (measured)

Raw noise floor **−30.6 dBFS**, the strongest spectral peak at 60Hz with a 120Hz harmonic, 0–80Hz carrying 38.4% of the energy. This is not room noise, it is mains/ground loop.

After adding a 150Hz, 4th-order high-pass:

| | Before high-pass | After high-pass |
|---|---|---|
| Noise floor | −30.6 dBFS | **−48.8 ~ −56.6 dBFS** |
| Speech SNR | 9.8 dB | **34 dB** |

The main speech region is 300Hz–3kHz, so cutting everything below 150Hz is lossless for ASR. **This step cannot be skipped**, otherwise the VAD threshold gets pushed up by the hum and speech never gets cut out.

### 2.2 Capture gain (adjusted, recoverable)

The H340's capture gain ships at full scale, `44/44 = +8dB`, and we measured 33 clipped samples.

```bash
amixer -c 0 cset numid=4 32     # 44 → 32, i.e. +8dB → 0dB
amixer -c 0 cset numid=4 44     # restore the original value
```

Note: this control has **no** Playback/Capture prefix, so `amixer sget 'Mic Capture Volume'` reports not found; you must use `numid`.

## 3. Engine comparison (same Japanese clip, 3.29s / 3.56s)

What the user said (presumed): 「このセクションのインターフェース仕様を確認してください」 ("please check the interface spec of this section")

| Engine | Clip 1 | Clip 2 | RTF | Error type |
|---|---|---|---|---|
| whisper large-v3-turbo | この**セクション**の…**使用**を | この**セクッション**の…**使用**を | 2.2 | kanji semantic substitution |
| whisper small | このセクションの…**使用**を | この**セキューション**の**インタフェース**を | 0.92 | katakana degradation + dropped words |
| **SenseVoice int8** | この**セシ**の…**仕様**を | この**石晶**の**インタフェース** **仕様**を | **0.04** | katakana collapse |

The two engines' failure points **do not overlap**: whisper reads the katakana correctly but replaces `仕様` (spec) with `使用` (use) — a semantic-level error that cannot be discovered after the fact; SenseVoice reads `仕様` correctly but collapses `セクション` (section) into `セシ`/`石晶` — a glyph-level error that context can repair.

**Conclusion**: use SenseVoice for ASR (50× faster, the only viable option for duplex), and hand katakana errors to the local LLM for correction. This is the same paradigm as the OCR rule in `work-memory`: "deterministic extraction + LLM correction".

### Why whisper cannot be used

`ctranslate2`'s aarch64 wheel is **CPU only**: `get_cuda_device_count() == 0`, so faster-whisper cannot use the GPU on this machine. 12 threads are already maxed out (the default is only 4 threads; after raising it RTF went 3.83 → 2.08). The GPU has to be reserved for the LLM.

## 3.1 TTS measurements and closed-loop verification

```
load            1.803s
first chunk     636.1ms      ← the latency duplex actually perceives, far more important than total RTF
synthesis       1.053s / audio 7.83s  → RTF 0.135
sample rate     22050Hz   speaking rate 6.6 chars/s
```

Closed-loop verification (synthesise → read back, original text known):

| | Text |
|---|---|
| Original | この**設計書**のインターフェース仕様を確認してください。データベース接続には**コネクションプール**を使用します。 |
| SenseVoice read-back | この**設計者**のインターフェース仕様を確認してください、データベース接続にはコネクションプールを使用します。 |

Only one error: `設計書→設計者` (しょ→しゃ, "design document" → "designer"); the long katakana terms are all correct. **Note that this is a score on synthetic speech**, far easier than a real microphone recording, so it must not be taken as real-device accuracy.

## 4. Web test bench (`voice_server.py`)

Open **http://127.0.0.1:8095** in a browser, click once to start talking; the server records, recognises, answers and synthesises, and the browser plays as it receives.

Playback by default comes out of **the machine that opened the page**; for the headset plugged into the Jetson to make sound you need to turn on "Sound: headset" (see 4.11).

The service is registered with systemd and recovers automatically on reboot:

```bash
sudo systemctl status voice-bench      # status
sudo systemctl restart voice-bench     # restart after changing code
journalctl -u voice-bench -f           # watch the log
```

### 4.1 Three design trade-offs

| Decision | Reason |
|---|---|
| **Recording on the server side** | The microphone is plugged into the Jetson. Recording in the browser would only use your laptop's microphone and would never exercise the real device's capture |
| **Silero VAD decides the end, no fixed length** | Recording a fixed N seconds makes "you still have to wait after you finish speaking" mask the real latency. VAD closes the turn 0.6 seconds after a pause |
| **Synthesise per sentence, send back per sentence** | Synthesising the whole thing would wait for the LLM to generate everything. Here the LLM emits one sentence and it is synthesised and pushed to the browser |

### 4.2 Why energy VAD was abandoned

At first we used an absolute threshold of "noise floor + margin", and both failure modes showed up in measurement:

- **When the gain is high**: the noise floor gets contaminated by speech → the threshold is pushed to −20dB → speech is judged as silence by itself, and only 0.81s is captured (`thr_db` exactly equal to `peak_db` is the fingerprint of this disease)
- **In a noisy environment**: the noise itself sits above the threshold → "finished speaking" is never detected → every recording hits the 12-second cap (both records the user measured were `record_s: 12.04`)

Silero VAD is based on speech probability over 512-sample windows, insensitive to both gain and stationary noise, and it comes with "how long after you stop counts as one segment". After the swap, a 7.59-second clip closed as soon as playback ended and no longer hit the cap.

### 4.3 Automatic Chinese / Japanese voice switching

Piper's Japanese voice reading Chinese, and its Chinese voice reading Japanese, both distort badly. Decision rule: **kana present → Japanese; no kana but kanji present → Chinese** (mixed Chinese-Japanese has been confirmed not to occur, so a per-sentence judgement is enough).

Also, "what language to answer in" is not left to the model to decide: we measured that when the previous turns were in Chinese and the user switched to Japanese, the model still answered in Chinese, because it follows the style of the context. So each turn appends a hard instruction based on the language of the current sentence. To pin the language use `--reply-lang zh|ja`, default `auto`.

#### Two real faults (both fixed)

**① `合成失败：# channels not specified`** ("synthesis failed: # channels not specified") — it is neither a recording nor a sound-card problem. When the text contains no pronounceable character (`。`, `……`, `「」`, pure whitespace), Piper's phonemisation result is empty, `synthesize_wav` writes not a single frame, and `wave` finds the channel count is still 0 when closing, so it throws this line. The corresponding real signal in the log is `JPCommonLabel_make(): No phoneme`.

There are now three lines of defence: `strip_for_tts()` uses `SPEAKABLE` (kana/kanji/letters/digits) to decide, and text with no pronounceable characters is skipped and never sent to synthesis; `synth_to()` synthesises into memory and checks the frame count first, retrying with the other voice if it is 0 frames; only if it is still 0 does it raise a readable `ValueError`.

**② Numbers in Chinese answers were read out in Japanese** — pure-number sentences (`1.5`, `2026`, `OK`) have neither kana nor kanji, so they used to fall through to `default` (Japanese), and "3.14" became 「さんてんいちよん」 (san-ten-ichi-yon). Now `pick()` accepts a `hint`: such sentences follow the voice already decided for the previous sentence, `TurnRunner` initialises it from `--reply-lang` each turn and updates it after each sentence is synthesised; only when there is no context does it fall back to Chinese.

Sentence splitting was fixed along the way: `.` in `SENT_END` used to cut `3.14` and `v2.0` in half; now `(?<!\d)\.(?!\d)` requires that the two sides of the decimal point are not both digits.

### 4.4 Measured latency (this machine)

| Stage | Chinese | Japanese |
|---|---|---|
| ASR (SenseVoice) | 0.3–0.5s | 0.23–0.37s |
| LLM first token | 90–440ms | 90ms |
| TTS first sentence | 0.14–0.31s | 0.18s |
| **End to end (no recording wait)** | **0.55–1.19s** | **0.44s** |

### 4.5 Wake word (always-on listening)

Click "常驻监听" (always-on listening) on the page, then say **「小八你好」** ("hi Xiaoba") and a dialogue turn starts without touching the mouse.

It uses `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`: Chinese pinyin modelling, 3.3M parameters, CPU only.

| Metric | Measured |
|---|---|
| RTF | 0.030 (8 seconds of audio computed in 0.26 seconds, about 3% of one core) |
| False triggers | 0 times across 14 segments of ordinary Chinese speech (news/interviews) |
| Custom words | kanji → pinyin generated at runtime by `text2token`; changing the word needs no retraining |

**Why `AudioHub` is mandatory**: the wake word needs the microphone open all the time, and an ALSA device can only be held by one process at a time. Originally `record()` started its own `arecord` every time, so as soon as the wake thread started it fought for the device. Now a single thread owns the device and distributes fixed 20ms frames to subscribers:

- On subscribe, backfill the last 0.3–0.6 seconds of frames → nothing is lost when you say 「小八你好，今天天气怎么样」 ("hi Xiaoba, what's the weather like today") in one breath
- When the queue is full, drop the oldest frames; the capture thread is never blocked by a slow consumer
- `arecord` exiting abnormally reopens automatically (observable as `hub.errors` in `/api/health`)

On the browser side **no long connection: 400ms short polling** (`GET /api/poll?after=<seq>`). Previously it was an NDJSON long connection on `GET /api/live`, which in this access environment could not be established at all (server-side `live` was always 0), and all 7 wake hits were judged "no page listening" and skipped. Short polling uses ordinary fetch, works through any proxy environment, at the cost of up to 400ms delay from wake to the cue tone.

The server keeps a 600-event ring buffer plus a monotonic sequence number; when the page opens it aligns to the present with `after=` set to a very large value, without replaying historical events. Wake turns and manual turns are mutually excluded by the same `turn_lock`, so they never fight over the microphone at the same time.

> Chinese wake words only: the official KWS model is trained on wenetspeech (Chinese pinyin), and Japanese pronunciation does not match. If Japanese wake-up is wanted in future, the only fallback is "always-on SenseVoice recognition + text matching", which raises CPU usage from 3% to around 10%.

### 4.6 Barge-in (interjecting during playback)

If you simply start talking while the assistant is still speaking, it stops immediately and listens, without waiting for it to finish.

**Why a playback clock is needed**: synthesis is an order of magnitude faster than playback (RTF 0.1), so when `TurnRunner.run()` returns, the earlier segments are still queued in the browser waiting to play. If we judged by "generation finished", the user would already be treated as "done speaking" while the assistant is talking, and there would be no barge-in window at all. So the server maintains a virtual playback timeline (`SpeakState`): each time a segment of audio is sent, the end time is pushed back by `audio_s` seconds, and it naturally returns to zero when the queue drains.

The decision chain has three stages; if any one of them fails there is no barge-in:

| Criterion | Role | Default |
|---|---|---|
| Energy | above silent noise floor + margin | `--barge-db -42`, `--barge-margin 8` |
| Speech probability | Silero VAD, filters keyboard/chair/fan | threshold 0.5, requires 0.30s continuous |
| Text overlap | excludes the assistant's own crosstalk on speaker | longest common substring ≥3 chars and ratio ≥0.7 |

**The third stage is the key one.** On speaker the microphone hears the assistant's own voice, and if you do not exclude it you get self-oscillation: it starts playing, is immediately interrupted, replays, and is interrupted by itself again. The crosstalk window is only one or two seconds, so what gets recognised is three to five characters (measured: `今天天`); computing `SequenceMatcher.ratio` against the full fourteen-character sentence gives only 0.35, which never reaches the threshold; switching to "longest common substring ratio" gives `3/3 = 1.0`, a clean decision.

A few implementation details:

- **The threshold follows only the silent noise floor**, it does not reference the level during playback — crosstalk and a real person's volume are comparable, so they simply cannot be separated by level, and using it to raise the threshold would only make a real person's interjection unable to barge in either
- **After being judged as crosstalk, cool down for 1.2 seconds**, otherwise the same sentence is re-judged every 0.3 seconds and a dozen ASR runs are wasted
- **Frame handover**: the 1.2 seconds of frames already consumed by barge-in detection are handed to the recorder as-is via `hub.subscribe(pre_frames=...)`, so the beginning of the user's sentence is not lost (the ring buffer only holds 0.6 seconds, not enough to backfill)
- **An interrupted turn does not enter history** (`done.interrupted`): it is incomplete, and keeping it would pollute the following context
- When playback ends and we return to the non-playing period, rebuild the VAD, otherwise the leftover state of the previous sentence skews the next turn's holding computation

Measured (real VAD + real ASR + real TTS, synthetic audio fed as frames instead of the microphone):

| Scenario | Result |
|---|---|
| Speaker crosstalk | 2 of 2 judged crosstalk, 0 barge-ins |
| Someone interjects 「等一下，我还想问个问题」 ("wait, I want to ask something else") | 1 barge-in, a new turn started, 1 event and 1 LLM call each |
| 2 seconds of silence | no reaction at all |
| Speaking after playback has finished | not judged (playback clock at zero) |

Reproduce: `../asr-venv/bin/python tools/barge_e2e.py` (no microphone needed, about 8 seconds).

**Limitations**: barge-in only takes effect in always-on mode. When you click "开始说话" (start talking) manually, the recording device is exclusively held by `record()` and there is no listening channel alongside. Also, headphones work best — on speaker, crosstalk is excluded, but barge-in is impossible during the cooldown.

```bash
--no-barge              # disable barge-in
--barge-db -38          # raise the threshold in a noisy environment
--barge-margin 12       # add margin when the noise floor fluctuates a lot
```

Observation: `barge.hits` (barge-in count), `barge.echo` (excluded crosstalk count), `barge.thr_db` (current threshold), `barge.playing` / `remain_s` (playback clock) in `/api/health`. Both hits and crosstalk are written to the journal: `[barge] 打断：「…」` (barge-in) / `[barge] 疑似串音，忽略：「…」` (suspected crosstalk, ignored).

## 4.7 HDMI capture card (MacroSilicon MS2109)

Link: **signal source → 4K USB KVM → capture card HDMI IN → USB → Jetson**. Working since 2026-11-19, stable image at 1080p30.

| Item | Value |
|---|---|
| Device | `534d:2109 MacroSilicon USB3.0 Capture`, UVC 1.00 |
| Nodes | `/dev/video0` (video), `/dev/video1` (metadata) |
| Current port | 4-port dock port **4** → system `1-4.4` (Bus 001 / 480Mbps) |
| Formats | MJPEG + YUYV, 1920×1080 ~ 640×480, 11 modes in total |
| Actual capability | HDMI input up to 4K@30Hz; UVC output capped at 1080p30 (the MS2109 is a USB2.0 chip, "USB3.0" refers to the HDMI side) |

### Biggest pitfall: about 1.2 seconds of warm-up black frames every time the device is opened

**Every time the MS2109 device is opened, and after every resolution switch, the first 33~36 frames are solid black (YMAX=7).**

Measured (640×480, 40 consecutive frames): the first 33 frames have `YMAX=7`, and from frame 34 on `YMAX=255`.

Anyone who grabs a single frame to decide whether there is a signal (me included) will inevitably conclude "no signal" — what gets grabbed is always a warm-up black frame. More confusingly, this makes the symptom look exactly opposite to bandwidth: `640×480` is all black, while `1920×1080` actually produces an image 9/10 times, because that round at the low resolution had just switched and had not warmed up enough.

The correct sampling method (discard warm-up frames within a single open):

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 30 -i /dev/video0 \
       -t 2.5 -vf "select=gte(n\,40)" -frames:v 1 -y shot.jpg
```

### Stability after warm-up

Continuous capture for 10 seconds / 300 frames: the first 36 frames black, then **all 264 frames normal, zero dropouts**. Occasional `EOI missing, emulating` / `overread 8` (corrupt JPEGs at the edge of hub bandwidth, which ffmpeg recovers from) do not affect the picture.

### Troubleshooting order (use this next time)

1. `lsusb | grep 534d` — if it is missing it is an enumeration problem (check power, change port, do not use the flashing USB-C port)
2. Whether `/dev/video*` exists — a card without nodes usually means uvcvideo did not bind
3. Run `tools/hdmi_probe.sh` (warm-up discarding built in) — only when it says there is no signal is there really no signal
4. If there is still no signal, check further: is the HDMI cable in the IN port, is the source switched on, is the KVM switched to that channel, does 4K@60 exceed the limit

### Power pitfalls stepped on

With the same card in port 3 of the dock, enumeration succeeded but **it disconnected itself after 166 seconds**; after moving to port 4 there were no dropouts for 8 continuous minutes. An unpowered hub carrying headset + mouse + capture card is tight on power; for long captures it is better to plug straight into the machine's Type-A port (`1-1` / `1-2`).

## 4.8 See screen (frame grab → OCR → into the dialogue)

Lets the assistant answer "what is written on the screen". Link: **capture card grabs a frame → discard warm-up frames → downscale to 1440 → RapidOCR → splice into the user message**.

```bash
python vision.py --image shot.jpg        # no hardware, test OCR directly
python vision.py --dev /dev/video0       # grab one frame from the real device and recognise it
curl -X POST -d '{"force":true}' localhost:8095/api/look   # go through the service
```

### Two triggers

| Trigger | Behaviour |
|---|---|
| Say 「看看屏幕上是什么」 ("look at what's on the screen") or similar | `is_look_intent()` matches → plays 「我看一下。」 ("let me take a look") first → OCR → asks the LLM together with the screen text |
| Click the "看屏幕" (look at screen) button on the page | only runs frame grab + OCR and lays the result out in the "屏幕文字" (screen text) card, without disturbing the LLM |

### Why RapidOCR and not PaddleOCR

The same 1080p Chinese UI (50 lines) measured on the Jetson:

| Engine | Time | Mean confidence |
|---|---|---|
| **RapidOCR (onnxruntime)** | **6.2s** | 0.894 |
| PaddleOCR default server model | 17.0s | 0.98 |
| PaddleOCR mobile | 24.6s | 0.749 |

The mobile model is actually slower on the Jetson (the CPU is unfriendly to large convolutions) and the confidence dropped as well. RapidOCR also saves the several hundred MB of paddle dependencies — it is already there in asr-venv.

### Why the input is downscaled to 1440 first

| Input | Time | Result |
|---|---|---|
| 1920 | 7.3s | 「窗口」 read as 「官口」 ("window" → "official opening") |
| **1440** | **5.7s** | 「窗口」 read correctly |
| 1280 | 4.6s | errors start |
| 960 | 4.3s | small text blurred (「编辑」→「编冀」, "edit" → nonsense) |

RapidOCR internally squeezes the long edge to a fixed size; downscaling with INTER_AREA ourselves first saves one resampling and is actually more accurate.

On 2026-11-21 we re-measured the downscaling path with a real-device frame (132 lines of dense text), **and the conclusion is: do not do it**:

| Long edge | Time (4 threads, alternating under the same conditions) | Lines | Confidence |
|---|---|---|---|
| **1440** | 6.13s | 132 | **0.951** |
| 1200 | 5.94s | 134 | 0.942 |
| 1080 | 5.70s | 132 | 0.922 |

Saves only 7%, confidence drops 0.03, and the line count is not even stable. **With many lines the time is determined by "lines × per-line decode", not by the number of input pixels** — the gain computed from area simply cannot be cashed in.

This rule was later nailed down by two data sets (same machine, same 8 threads, same 1440 width):

| Screen lines | OCR time | Per line |
|---|---|---|
| 132 lines (IDE full screen) | 4.24s | 32 ms |
| 53 lines (switched to another window) | 1.61s | 30 ms |

**So "make looking at the screen faster" = "put fewer lines on the screen"**, and it has nothing to do with resolution or with engine parameters.

### Measured (including the voice path, on the machine 2026-11-21, HDMI has a picture)

```
LOOK: ok=True 行=130 置信=0.949 4.45s
DONE: look_s=4.45 llm_first_ms=1001 llm_s=2.0 total_s=6.56
```

After the resident frame grab took effect, two rounds re-measured **end to end 6.56s / 7.30s** (the earlier estimate of 6.3s basically matches, slightly optimistic):

| Stage | Round 1 | Round 2 | Share |
|---|---|---|---|
| Frame grab | 0.08s | 0.06s | ~1% |
| **OCR** | **4.34s** | **5.55s** | **68%~77%** |
| LLM first token | 1.00s | 0.45s | cold / warm (prefix cache hit) |
| LLM full | 2.00s | 1.59s | — |
| First sentence TTS synthesis | 0.10s | 0.07s | ~1% |

Conclusion: **frame grabbing has been pushed down to negligible; OCR is the only bottleneck, and it jitters by ±1.2s itself**. The LLM first token is 1 second cold and 0.45 seconds warm, because the 130-line OCR text changes every time and the prefix cache only covers the system prompt part.

> These are **pure CPU** numbers. After rec moved to TensorRT the end-to-end dropped to **3.4~3.9s**, see "Moving rec to the GPU" below.

### Caching cannot save it: the picture is always changing

`look()` has a dHash cache (the same picture within 8 seconds is returned directly). But testing `force=false` three times in a row on the real device, `cached` was **always False**, and the line count jumped between 130/133/132 — the IDE cursor, timestamps and terminal output are always moving.

So the assumption "looking at the same picture repeatedly saves 4 seconds" **does not hold**, and continuing to use `force=True` for voice triggers is correct.

### OCR thread count: must be pinned explicitly (2026-11-21)

When `RapidOCR()` is not given a thread count it takes the onnxruntime default. Same real-device frame, same process, serial measurement:

| Threads | Time | Lines | Confidence |
|---|---|---|---|
| 6 | **4.34s** | 132 | 0.951 |
| **8** | **4.49s** | 132 | 0.951 |
| 10 | 5.11s | 132 | 0.951 |
| 12 (= core count) | **9.10s** | 132 | 0.951 |

**12 threads is twice as slow as 8** — this machine runs vLLM, ASR and the caption service at the same time, and 12 threads oversubscribe all 12 cores; the cost of thread migration and core contention far exceeds the parallel gain. The flat region is 6~8, degradation starts at 10, and 12 collapses outright.

Now `vision.OCR_THREADS = 8` is hard-coded, adjustable via `--vision-threads`. Note: **the original production default already landed near 8** (measurements were always 4.0~4.5s, not 9s), so this change is **hardening against regression**, not a speed-up — it means that changing the machine model, the onnxruntime version or the core count will not suddenly make it twice as slow.

> Counter-example lesson: in the first comparison round all three settings ran together, and 1200/1080 came out twice as slow as 1440. The reason is that the benchmark's own onnxruntime also started 12 threads, so what was being measured was "two processes fighting each other", not resolution. **When measuring anything concurrency-sensitive you must limit threads first and leave a gap between settings.**

### Why rec cannot be pushed down on CPU (seven paths, all tried)

First correct an easy assumption: **RapidOCR's rec is already batched**, it is not "one call per line". `TextRecognizer.__call__` takes a **list** of crops and assembles them into one `norm_img_batch` by `rec_batch_num` (default 6) before sending to the session. So the optimisation direction "merge multiple lines into one batch" never existed in the first place.

Time breakdown (132 real-device lines, 8 threads):

| Stage | Time | Share |
|---|---|---|
| det | 0.297s | 7.6% |
| cls | 0.153s | 3.9% |
| **rec** | **3.437s** (25 batches × batch 6, 147 boxes) | **88.4%** |

Then tried one by one:

| Attempt | Result | Verdict |
|---|---|---|
| `rec_batch_num` 6 → 12 / 24 / 48 | 4.03 → 4.13 / 4.01 / 4.85s | ❌ no gain; 48 actually collapses (`max_wh_ratio` takes the max within the batch, so bigger batches mean wider padding) |
| Turn off `use_cls` | -0.15s | ⚠️ only 4%, and it loses the protection for inverted text; not worth it |
| int8 full quantisation | 10.8MB → 2.9MB | ❌ **accuracy collapses**: 132 lines → 9 lines, mean score 0.951 → 0.563 (CTC is extremely sensitive to logits) |
| int8 on MatMul only | accuracy 132 lines / 0.951 / 1846 characters, **completely lossless** | ❌ speed 3.98s, not faster at all |
| Switch to PP-OCRv6_small | 4.68s, accuracy actually higher (135 lines / 0.971 / 1976 characters) | ❌ 15% slower |
| `unclip_ratio` 1.6 → 1.2 / 1.0 | 4.69 / 5.14s | ❌ tighter boxes are slower (counter-intuitive) |
| Constant folding `Constant` → `initializer` | 3.90s (-4%, accuracy identical character by character) | ➖ gain ≈ measurement noise (±0.2s), not worth maintaining a custom model file |
| CUDA EP | system CUDA **13.2**, ORT only ships **cu12** wheels; `pip` has no cu13 package and `apt-cache` has no `cuda-cudart-12-*` either | ❌ this path requires building ORT yourself |

Conclusion: **there is no room for lossless speed-up of rec on CPU**. The only real way out is the GPU — see below.

#### Also noting a Paddle2ONNX pitfall

In `ch_PP-OCRv4_rec_infer.onnx` the **initializer count = 0; all 420 weights are `Constant` nodes**. ORT's `quantize_dynamic` only recognises initializers, and running it directly reports `Expected conv2d_157.w_0 to be an initializer`. You must do constant folding first (move the `Constant` tensors into `graph.initializer` and then delete the nodes) for quantisation to actually touch the weights — otherwise you get a fake success where "the file size is exactly the same".

### Moving rec to the GPU: TensorRT (2026-11-21)

**Do not use onnxruntime's CUDA EP.** This machine is JetPack 7.2 + CUDA **13.2**, while ORT officially only ships **cu12** wheels: there is no cu13 package on pip, and no `cuda-cudart-12-*` in `apt-cache` either. After installing `onnxruntime-gpu`, `get_available_providers()` will **list** `CUDAExecutionProvider` as usual (deceptive), but creating a session reports `No registered plugin EP device found` — that is the EP's .so failing to dlopen. Do not waste time on it.

TensorRT 10.16 ships with the system (`libnvinfer.so.10` + `trtexec` + the python `tensorrt`), so we bypass onnxruntime and run rec directly.

**Approach**: do not touch RapidOCR's source, only replace `TextRecognizer.session` with `trt_rec.TrtRecSession`. That way `resize_norm_img`, CTC decoding and the character table all reuse the original implementation, and the only difference between the two sides is the inference backend — so what gets measured is a pure backend difference.

| Backend | OCR total (132 lines) | Lines | Mean score | Per-line agreement |
|---|---|---|---|---|
| CPU (onnxruntime) | 4.12s | 132 | 0.951 | baseline |
| **TensorRT FP16** | **1.17s (3.5×)** | 132 | 0.952 | 128/132 |

End to end (voice question 「屏幕上是什么」, "what's on the screen", 140 lines): **6.56s / 7.30s → 3.87s / 3.41s**. The remaining time is mostly spent on LLM generation (2.1s); OCR is no longer the bottleneck.

The only per-line differences are single-character jitter such as `θ.98 → 8.98`, `—a → 一a`, `误。 → 误触`; that is the normal difference between FP16 and CPU FP32 and does not affect "understanding the screen".

#### Where the engine comes from

```bash
trtexec --onnx=<rapidocr>/models/ch_PP-OCRv4_rec_infer.onnx \
        --saveEngine=models/rec_fp16.engine --fp16 \
        --minShapes=x:1x3x48x256 --optShapes=x:6x3x48x320 --maxShapes=x:32x3x48x2048 \
        --memPoolSize=workspace:1024 --timingCacheFile=/tmp/trt_cache.bin
```

On Orin it takes about **11~12 minutes** (tactic search for dynamic shapes is very slow; the log stops updating for long stretches, do not assume it has hung).

Setting `optShapes` to 320 has a basis: we measured the width distribution over 75 batches and **76% are exactly the baseline 320** (when there is no extra-wide line in the batch, `max_wh_ratio` takes the default 6.67), and only 4% exceed 1536. So setting opt to a small value makes the common path fastest, rather than picking a "middle value".

> The first version failed with `max` set to 1024 and to 1536 — there was one particularly long line on screen (a path, a URL, a long line of code) and `max_wh_ratio` stretched the whole batch to 1036, 1656. **This number has to be measured before it is set**; guessing means a rebuild (12 minutes each time).

#### Batches beyond the profile fall back to CPU

In `TrtRecSession`, if `set_input_shape` returns False it hands over to the original ORT session. The reason: rebuilding an engine with max 3000+ for the extreme case would slow down the 96% common batches, so it is better to let that 4% take the slow path. `vision.trt.fallback_pct` in `/api/health` is that ratio, and **normally it should be close to 0** (after the profile was set correctly we measured 0 fallbacks).

#### Runtime dependency: `cuda-python` (do not uninstall it as exploration leftovers)

`TrtRecSession` allocates GPU memory and sets up streams itself, relying on `cuda-python`:

```bash
asr-venv/bin/pip install cuda-python
# self-check (about 1 second)
asr-venv/bin/python -c "from cuda.bindings import runtime as rt; e,p = rt.cudaMalloc(16); print(e,p)"
# → (<cudaError.cudaSuccess: 0>, <cuda.bindings.runtime.cudaDevicePtr...>)
```

**It has nothing to do with `onnxruntime-gpu`**, do not conflate the two. ORT's CUDA EP is dead on this system (CUDA 13.2 vs wheels only shipped for cu12), while the TensorRT path works — precisely because of that, `cuda-python` is currently the only dependency that puts rec on the GPU.

Uninstalling it **does not raise an error**, it only **silently** falls back to CPU: `vision.trt.backend` in `/api/health` becomes `cpu`, and OCR quietly goes from 1.17s back to 4.12s. Remember this when cleaning up the environment.

Two API details (both stepped on): `cudaMalloc` returns an `(err, ptr)` tuple so you must unpack two values; `cudaFree` returns only `err`.

#### Enabling

```bash
# put models/rec_fp16.engine in the deployment directory, then
voice_server.py ... --vision-trt /path/to/models/rec_fp16.engine
```

Without `--vision-trt` it uses CPU, with behaviour exactly as before. Engine load failure, TRT version mismatch, missing file — all are swallowed and fall back to CPU; "look at screen" is never made completely unavailable. At startup a background thread builds the engine first (measured 3 seconds), otherwise the first look-at-screen has to wait extra for the CUDA context.

**The engine is tied to the TensorRT version**: after changing JetPack / upgrading TRT it must be rebuilt, otherwise deserialisation fails (at which point it falls back to CPU automatically and does not crash).

> Coexisting with vLLM is no problem: the Orin has unified memory, vLLM resident takes about 7GB and TRT inference uses only a few hundred MB. Measured with both running at once, OCR and the LLM both work normally. But **building the engine consumes 30GB+** for profiling, so during those few minutes do not let other tasks compete for GPU memory.

### Resident frame grab: pay the warm-up only once (2026-11-21)

The old code called `cv2.VideoCapture()` on every `grab()`, opened the device and threw away 40 warm-up frames, **paying a fixed 1.2 seconds on every look-at-screen**, and `look()` grabbed the frame before checking the cache — so it paid even when the screen had not changed. Now `_Capture` keeps the handle resident:

| | Old (open/close every time) | New (resident) |
|---|---|---|
| Device open + negotiation | every time | **once at process start** (`prewarm()` in a background thread) |
| 40-frame warm-up | 1.2s every time | one round per open; normally exactly 1 round in total |
| Per frame grab | 2.87s | **read 2 frames, about 0.07s** |

Not doing "continuous decoding in a background thread" is deliberate: 1080p MJPG decoding is a continuous CPU cost, while look-at-screen is used only a few times per meeting. `BUFFERSIZE=1` makes the driver always hold exactly one buffer that is continuously overwritten, so what you read is the current picture — there is no need to rely on continuous reading to stay fresh.

Three self-healing rules:

1. **Frame read fails → release and reopen once** (`reconnects` counter). The 166-second dropout with the dock is recovered by this rule, without restarting the service.
2. **All black → do one more warm-up round before judging**. After the source switches resolution the MS2109 emits black frames again; a resident handle never sees `open` but does see this "stream turning black mid-way", and without the extra round it would falsely report "HDMI has no signal".
3. **Release the device after `--vision-idle-release` seconds idle** (default 0 = never release). Set it to something like 300 when you want another program to use this capture card.

In `vision.capture` of `GET /api/status` you can see `opens / warmups / reads / reconnects / open_s / grab_s` directly: **`opens` should stay 1**; if it grows, the connection dropped and reconnected at some point.

### Three protections

1. **Warm-up frames**: `_Capture` always discards 40 frames on each open (see 4.7), otherwise what you get is a black image and OCR comes back empty
2. **No signal → judge blank**: the criterion is "dark and flat" — `mean < 12 and std < 6`. The old criterion `mean < 3` never matched on the real device: with no signal the MS2109 emits frames with **mean 7.0 / std 1.5** (YMAX=7 with sensor noise), so they were treated as "a screen with no text" and returned `ok=True, lines=0`, a fake pass. Before judging blank it does one extra warm-up round and re-checks, to avoid a false report when the source switches resolution
3. **If it cannot see, say so**: when OCR gets no text it does not stuff empty context in, it reports an explicit error, so the model does not make things up from imagination

### Known limitations

- **3.4~3.9 seconds** to a result (rec on TensorRT, measured). The accounting is: **frame grab 0.08s + OCR 0.76~1.56s + LLM first token 0.2~0.6s + generation 2.1s**. The bottleneck has moved to LLM generation, no longer OCR
- Without `--vision-trt` it falls back to CPU, 6.5~7.3 seconds end to end
- To compress the LLM part, **cutting the number of characters of OCR text sent into the prompt** is more useful than changing models: cold prefill is about 0.58 ms/token, so 1.6K characters costs 1 second and 3.3K characters close to 2 seconds; switching to 27B turns this item from 1 second into 8 seconds (see 4.14)
- **Barge-in does not work while waiting for OCR**: there is no playback at that time, so barge-in listening is not running, and a user interjection has to wait for the turn to end
- OCR has stable misrecognitions (帧 → 顿, "frame" → "meal"; Jetson → letson; lost quotation marks); we let the LLM "correct by context" via the system prompt and do not build a correction model
- Screen text longer than 1400 characters is truncated (`max_chars`)

## 4.9 Text but no sound (browser autoplay blocking)

Symptom: the textual answer appears normally on the page, but **there is no sound at all and no error either**.

The root cause is not on the server. One round of measurements:

| Check | Result |
|---|---|
| Server-side TTS | ✅ synthesised a 2.38s wav, `mean -15.1dB / max 0dB`, not silence |
| `GET /audio/xxx.wav` | ✅ 200, `audio/wav`, 105004 bytes |
| Browser | ❌ `play()` returns `NotAllowedError`, and the old code `catch(()=>{ playNext() })` **silently swallowed this error** |

Chrome / Edge policy: when the page has never had any user interaction, a `play()` with sound is rejected outright. In continuous dialogue via the wake word there is no click at all, so every turn is blocked — text renders as usual, the sound is entirely lost, and nothing looks abnormal in the UI.

### Current handling

1. **No longer silent**: when `play()` fails and `name === 'NotAllowedError'`, an orange banner appears at the top of the page: 「浏览器拦截了自动播放，所以这轮没有声音」 ("the browser blocked autoplay, so this turn has no sound")
2. **Do not drop the audio**: the blocked URL goes back to the head of the queue and plays once unlocked (previously it was simply dropped)
3. **One-click self-proof**: new "声音测试" (sound test) button → `POST /api/say` synthesises 「声音测试，一二三。」 → the browser plays it. On success it shows the duration and voice; on failure it shows the specific error name
4. **Any click counts as unlocking**: the banner itself, and any click on the page, will try to play back the owed audio

```bash
curl -X POST -H 'Content-Type: application/json' \
     -d '{"text":"声音测试"}' localhost:8095/api/say
# {"ok":true,"url":"/audio/say_xxx.wav","tts_s":0.15,"audio_s":2.04,"voice":"zh"}
```

### Troubleshooting order

1. Click "声音测试" — sound → it is the autoplay policy, clicking the page once unlocks it; no sound → continue
2. Look at the error name shown in the banner: `NotAllowedError` = policy block; `NotSupportedError` = cannot fetch the wav (check the `/audio/` route)
3. If neither, check **whether the tab is muted** (a speaker with a slash on the tab), the system output device, and whether the headset is plugged into the Jetson rather than the computer

### The sound went to HDMI (actually hit this)

The browser follows Windows' default output device. This machine's default output was once `UHD-SPLITTER` / `HD TO USB` (a Display Audio endpoint), so the playback all went into the HDMI link and the speakers made no sound at all.

Two approaches, ordered by size of change:

| Goal | How | Cost |
|---|---|---|
| **Only the browser through the speakers** (recommended) | Win11 設定 → システム → サウンド → 音量ミキサー → find Chrome → change output to `スピーカー (SC-HC410)` ("Speakers") | none, nothing to install |
| The same audio out of both at once | Voicemeeter (A1/A2 dual output); or Stereo Mix + Listen | installs a virtual sound card / 100ms+ latency |

**One hidden hazard along the way**: the capture card MS2109 has USB Audio, and the Jetson now shows an extra `card 4: MS2109`. That means the computer's sound goes through HDMI → KVM → capture card back into the Jetson. Right now playback is in the browser so it is unaffected; **as soon as playback is changed to local `aplay` on the Jetson, a loop forms**, and barge-in would treat its own playback as a user interjection.

## 4.10 Meeting captions (caption_server.py)

During a meeting, turn the meeting audio into real-time Chinese-Japanese bilingual captions. A separate process on a separate port (8096), **completely decoupled** from the voice assistant, each taking a different audio source so they do not interfere.

```bash
sudo systemctl start caption-bench       # resident service (already enabled)
journalctl -u caption-bench -f           # live log
python caption_server.py --wav x.wav     # run one file offline, no hardware needed
```

Link: `capture card audio → energy segmentation → SenseVoice(auto) → zh/ja mutual translation → scrolling large captions on the web page`

### Three pitfalls (all stepped on)

1. **PipeWire seizing the device** — the most important one. After PipeWire recognised the MS2109 as the default recording source it held `pcmC4D0c` long-term, so `arecord -D plughw:CARD=MS2109` kept saying "device or resource busy", and `pgrep arecord` showed no process at all (the holder was `pipewire`). Switched to `pw-record --target <node>` to take the stream, which does not hold the device exclusively and can coexist with other audio clients. The node name is found automatically from `pw-dump` by `MACROSILICON`.
2. **pw-record cannot connect from inside systemd** — a system service has no `XDG_RUNTIME_DIR`, cannot find the user session's PipeWire socket, and the process exits as soon as it starts. The unit must contain `Environment=XDG_RUNTIME_DIR=/run/user/<uid>`.
3. **Initial noise-floor value contaminating everything** — we used to estimate the floor with `min(floor, db)`, and the first digital-silence sample (-91dB) pinned floor at -999, so the threshold became -991dB, equivalent to "no sound at all counts as speech". Changed to take the 10th percentile over the first 1.2 seconds, then fast-down slow-up, clamped to `[-85, -32]`.

### Start/stop on the page

| Button | Behaviour |
|---|---|
| **Stop** | really terminates the capture process and releases the sound card; the button turns green "Start" and the level meter goes to zero |
| **Start** | rebuilds the Segmenter (re-estimating the noise floor for the room at that moment) and resumes capture |
| Clear | only clears captions already shown, does not stop capture |

Clicking Start immediately after Stop may hit the USB sound card's release latency; internally it retries 4 times automatically (interval 0.35~1.1 seconds).

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--backend` | `pw` | `pw`=PipeWire (recommended); `alsa`=arecord direct |
| `--device` | `plughw:CARD=MS2109,DEV=0` | only used with `--backend alsa` |
| `--margin` | `8.0` | how many dB above the floor counts as speech; raise it in noisy environments |
| `--hold` | `0.45` | baseline silence threshold (used when the segment length is between `--seg-mid` and `--seg-long`) |
| `--hold-long` | `0.90` | how long to wait for silence before cutting while the segment is still short, see 4.16 |
| `--hold-short` | `0.30` | cut threshold when the segment is already long, see 4.16 |
| `--seg-mid` / `--seg-long` | `1.20` / `2.40` | switching points between the two thresholds above (seconds) |
| `--max-s` | `6.0` | maximum length of one segment in seconds, hard cut as the fallback (basis for the value in 4.16: 7.0 produces hallucinations) |
| `--soft-cut` | `0.7` | search window for backtracking to an energy trough before a hard cut (seconds) |
| `--min-s` | `0.25` | drop anything whose speech is shorter than this |
| `--no-bilingual` | off | original text only, no LLM translation |

### Known limitations

- **Music gets "heard" as words**: SenseVoice produces hallucinated text for lyrics and pure music (measured: a song recognised as 「风从楼间穿过去…」, "wind passing between the buildings…"). Only real human speech is accurate
- Latency = segmentation wait (0.55s) + ASR (0.1~0.4s) + translation (0.3~0.8s), about 1~2 seconds — enough for meeting notes, on the low side for simultaneous interpretation
- Translation goes through local qwen (7B); long sentences and proper nouns drift; when the original text has misrecognised characters the translation may tilt along with it

### 4.11 Playback output: browser vs local headset

**Symptom**: the headset is plugged into the Jetson's USB port and the microphone is recognised (`plughw:CARD=H340`), but the assistant never makes sound from the headset.

**Root cause**: playback is done by the **browser**. `_speak()` only does `emit({"type":"audio","url":...})`, and the machine that opened the page fetches the wav and plays it. So "where the sound comes out" equals "which machine the page is open on", regardless of where the microphone is plugged in. Measured with the page open on the computer (`127.0.0.1`), there is no playback stream at all on the `voice-bench` machine.

**Solution**: add a local playback path `LocalSpeaker`, which `pw-play`s straight to the local sound card after synthesis.

```
synthesise wav ─┬─ emit audio{local:true} → browser only displays text, does not play
                └─ LocalSpeaker.play()    → pw-play --target <H340> → headset
```

**Three design points**

| Question | Approach |
|---|---|
| Why `pw-play` and not `aplay` | The microphone is held exclusively by `AudioHub` through ALSA, so `arecord -D plughw:CARD=H340` reports "device or resource busy". Playback goes through PipeWire: playback and capture are two independent streams on the USB sound card, so they do not fight |
| How barge-in cuts it off | `LocalSpeaker.cancel()` clears the queue + `terminate()`s the current `pw-play`; called together with `SpeakState.cancel()` inside `BargeMonitor._on_barge()` |
| Why there is no sound from systemd | Same pitfall as the caption service: missing `XDG_RUNTIME_DIR=/run/user/<uid>`, so `pw-play` cannot connect to the user session's PipeWire and fails outright |

**Switching**: new "声音：耳机 / 声音：浏览器" (sound: headset / sound: browser) button on the page → `POST /api/route {on:bool}`, switchable at runtime without a restart. The "声音测试" (sound test) in `/api/say` also follows this switch, so that "does the test produce sound" and "does the real thing produce sound" are the same path.

**Startup parameters**

| Parameter | Default | Description |
|---|---|---|
| `--local-speak` | off | enable local playback |
| `--local-sink` | empty (system default) | PipeWire target node name/id, e.g. `alsa_output.usb-Logitech_Inc._Logi_USB_Headset_H340-00.analog-stereo` |
| `--local-gain` | `1.0` | stream volume 0~1 |

The `speaker` field in `/api/health` shows `on / sink / plays / errors / busy / last_error`: `plays` only counts playback with **exit code 0**, which distinguishes "played but no sound" from "never played at all".

**Troubleshooting order**: `speaker.plays` growing → the playback chain works, go check the headset itself (the H340 inline volume wheel, whether the plug is fully inserted); `plays` not moving while `errors` grows → look at `last_error` (most likely `XDG_RUNTIME_DIR` or a misspelled sink name).

### 4.12 Play HDMI audio back to the headset

**Scenario**: the computer's HDMI output all goes into the capture card (both audio and video into the Jetson) and the person cannot hear it. Hanging another headset on the Jetson and playing that stream back as-is effectively adds a monitor port to the HDMI link.

```
capture card source ──┬─ pw-record 16k  → segmentation → ASR → captions
                      └─ pw-record 44.1k → pw-cat → headset (monitor)
```

Two capture streams coexist: PipeWire allows multiple clients to read the same source, and measurement showed the 16k caption stream was completely unaffected.

**Implementation points**

| Question | Approach |
|---|---|
| How to move the data | `pw-record \| pw-cat`, two subprocesses wired directly together, sample data goes only through the kernel pipe, Python is not involved, CPU usage is close to 0 |
| Why not use `pw-link` to connect the nodes directly | A direct link has lower latency, but the link breaks without any notice when the device is replugged; with the subprocess approach a break is immediately visible from `poll()`, which makes reconnection easy |
| Sample rate | The headset node is **44100** and the capture card is 48000. Use 44100 uniformly to save one resampling |
| Buffering | `--latency 20ms`, 20ms at each end, about 60~100ms end to end. Measured 20/40/default, none underran; 20ms feels the most responsive |
| Switch | "听 HDMI" (listen to HDMI) button on the page → `POST /api/monitor {on}`; `--monitor` makes the service start with it on |

**Verification method** (for sound you cannot just check that the process is alive)

```bash
# check the ports are really connected: only [active] means it is transferring
wpctl status | sed -n '/Streams:/,$p' | grep -A2 pw-cat
# check it reads at real-time rate: rchar should grow by about 700KB after 4 seconds
grep rchar /proc/$(pgrep -x pw-cat)/io
```

**Pitfall**: in `pw-top` the `pw-cat` and headset nodes show `???` (quantum 0, rate 0) and look like they are not running, yet the `[active]` link and the growing `rchar` both prove it is transferring. `???` only means that node is not the driver in the graph and gets no timestamps — **it cannot be taken as evidence of "no sound"**; the real criteria are `[active]` in `wpctl status` and the process IO counters.

**Side effect (worth noting)**: the meeting sound coming out of the headset is picked up a little by the H340 microphone and may trigger the wake word or barge-in. If self-question-and-answer happens during a meeting, turn "听 HDMI" off first, or switch to earbuds that are not a headset.

### 4.13 Meetings need to be "real time": squeeze 8 seconds down to 1~2

**The original latency accounting** (measured): every segment was a flat 6.0 seconds (`max_s` hard cut) → on average 3 seconds of dead wait inside the segment → ASR 0.38 seconds → **translation 0.3~1 second running serially in the recognition thread** (meanwhile captured frames wait in the queue) → front-end 600ms polling. From finishing speaking to seeing text: **7~8 seconds**, and simultaneous interpretation can no longer keep up with the listening experience.

**Four changes**

| Change | Before | Now | Effect |
|---|---|---|---|
| Segment cap `--max-s` | 6.0s | **3.0s** | hard upper bound on latency, halved |
| Silence cut `--hold` | 0.55s | **0.42s** | one sentence closes faster after it ends |
| How text appears | wait for the translation, then output the whole line | **original text instantly, translation arrives as a patch** | seeing text only waits for ASR |
| Thread model | synchronous ASR + translation in the capture thread | **three independent queues: capture / ASR / translation** | any stage slowing down does not block capture |
| Front-end polling | 600ms | **250ms** | feels more responsive |

**Measured (continuous audio, music playing)**: `lag_ms` (segment end → original text visible) **96~190ms**, with `asr_ms` accounting for all of it; `trans_ms` 217~756ms follows; `q_asr=0 / q_tr=0` shows there is no backlog anywhere. From finishing speaking to seeing the original text is about **0.6~1 second** (0.42s silence decision + ASR + polling), with the translation another 0.2~0.8 seconds later.

**Implementation points of the two-stage design**

- `Captioner.submit()` only enqueues; the capture loop never waits for recognition; when the queue is full **the oldest segment is dropped** rather than letting the captions fall further and further behind (a caption 10 seconds late is as good as none).
- As soon as recognition finishes, append a caption line with `trans="" / pending=true`; the front end shows a 「译文中…」 ("translating…") placeholder.
- When the translation thread finishes it backfills `line["trans"]` in place (history and newly opened pages are complete) and at the same time pushes `{seq, trans}` into the patch queue, which is delivered once as the `upd` field of `/api/poll` and then cleared.
- The front end merges patches by `seq` and preserves the scroll position when redrawing (stuck to the bottom → follow; reading in the middle → do not move).

**Two quality problems fixed along the way**

1. **Hard cut landing in the middle of a syllable**: `天真好` cut into `天真` gets translated as 「天空」 ("sky"). Now when it reaches the `--soft-cut` limit it **backs off to the frame with the lowest energy** within the last 0.7 seconds to make the cut, leaving the remaining frames for the next segment (segment length now fluctuates 2.3~3.0 seconds, but far fewer misrecognitions).
2. **Short sentences' translations mix languages**: after cutting shorter, a single sentence is often half a sentence and the model leaves 「的/了」 or mixes Chinese/English/Japanese. The translation prompt was changed to a hard requirement that "the translation must be purely in the target language, not a single source word may remain", and the **previous two sentences' source text is sent in as 【上文】** (previous context), so that person and terminology line up.
3. **Filler words treated as content words** (`strip_fillers()`): meetings are full of 「嗯、呃、えーと、あの」 ("uh, um, eto, ano") and the model translates them into 「はい」「ええ」「那个」, dragging the whole sentence's meaning off course. Now **before sending to translation** pure fillers are removed, while **the original text shown on the page stays verbatim** (a verbatim transcript has value in itself).

The cleaning rules are deliberately conservative, relying on "must be preceded/followed by punctuation or a sentence boundary" rather than guessing from a word list: the 「那个」 in `那个项目` ("that project") and the 「あの」 in `あの人` ("that person") are followed by content words, so they naturally do not match, and therefore do no damage even though they are not on the list (same for the `er` in `that is her eraser`). When the whole sentence is filler (e.g. 「哦。」) it **falls back to the original text** and never returns an empty string.

**Parameters**

```bash
--max-s 3.0        # hard latency cap; do not exceed 4 seconds in a meeting
--hold 0.42        # how much silence before cutting a segment; any smaller splits words in two
--min-s 0.25       # drop anything shorter than this (clicks)
--soft-cut 0.7     # search window for backtracking to a trough before a hard cut
--asr-workers 1    # ASR takes only 0.1~0.2 seconds per segment, 1 is enough
--tr-workers 1     # 1 in production: measured no first-token difference between 1 and 2 (4.21), one queueing stage less
--ctx-lines 1      # 1 in production: carry one sentence of context, first token saves ~90ms versus 2 (4.21)
                   # translation streams by default (first token 260ms); --no-stream-trans reverts to whole sentences
--tr-prompt full   # system prompt tier; short is the 35B short version, measured slower (see 4.20)
```

The status now also has `lag_ms / asr_ms / trans_ms / ttft_ms / q_asr / q_tr / dropped`, and the top of the page shows 「延迟 x.x s · 首字 xx ms · 译完 x.x s」 (latency x.x s · first token xx ms · translated x.x s) directly. To judge whether it keeps up, look at only two numbers: **queues permanently at 0 means real time**; the moment `q_tr` keeps rising the translation cannot keep up, so add `--tr-workers`.

> To see 「译文中」 ("translating…") turn into text appearing character by character, see **4.15** (streaming translation + SSE push, first token 260ms).

**Regression test**: `test_caption_pipe.py` (ASR and translation stubbed, does not depend on models), 7 cases covering two-stage text output, patch de-duplication, context passing, non-blocking delivery, queue overflow dropping the oldest, short-segment cap, trough cut point, clear screen.

### 4.14 Three numbers to measure before changing models (criteria for the voice scenario)

We did a round of inference-engine comparison on the Jetson (llama.cpp vs Magnitude, 1B / 27B × 8K / 32K); the conclusions are in memmy. But **those numbers are in agent-run terms**, and taking them directly to choose a model for a voice assistant gives the opposite answer:

| Dimension | agent run (benchmark terms) | Personal voice assistant |
|---|---|---|
| Concurrency | dozens of calls chained together | **1 user, 1 stream** |
| Single output | 200–800 tokens | one sentence, 20–80 tokens |
| Context | 20K–60K, only grows | 2–4K (persona + last few turns) |
| Key metric | aggregate throughput, unit cost | **prefix-hit TTFT**, time to first spoken sentence |
| Consequence of being slow | it runs a bit longer | the user thinks it did not hear them |

So the two most eye-catching numbers in the benchmark — Magnitude's 12.17 tok/s aggregate at N=8, and the batch sweet spot of 4 streams — **are almost useless for an assistant**, because there simply are no 8 concurrent requests.

#### Three hard lines

| Criterion | Line | Reason |
|---|---|---|
| Prefix-hit TTFT | **< 300 ms** | the persona system prompt is fixed and hits the cache every turn; this is the real starting point |
| Single-request decode | **≥ 20 tok/s** | TTS is about 4~5 characters/s; below 20 you get "thinking while speaking" breath pauses |
| First sentence spoken (measured) | **< 600 ms** | up to the first full stop or 12 characters; this is the moment the user perceives "it understood me" |

**Do not convert first-sentence latency from tokens**: Chinese tokenisation is about 1.3 characters/token, and at 26 tok/s for 7B "20 tokens" works out to 769 ms, yet the real first sentence takes only 279 ms. The first version of the criteria died exactly here (falsely judged as failing).

#### Measured on the same machine (Jetson AGX Orin 64GB, single request, temperature=0, median of 3 runs)

| Model | decode tok/s | Prefix-hit TTFT | Cold prefill TTFT | Process cold start | Peak memory | First sentence spoken |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-AWQ (vLLM, **currently in use**) | **26.0** | **88 ms** | 438 ms | — (resident) | — | **279 ms** ✅ |
| 27B Q4 (llama.cpp) | 7.74 | 309 ms | 7.85 s | 15.4 s | 29.1 GB | ≈2.6 s ❌ |
| 27B Q4 (Magnitude) | 6.17 | 294 ms | 7.06 s | 148.4 s | 33.6 GB | ≈3.2 s ❌ |
| 27B Q4 (Magnitude + DFlash2) | 20.19 | 280 ms | 7.50 s | 97 s | 34.3 GB | ≈1.0 s △ |
| 1B Q4 (llama.cpp) | 111.8 | 26 ms | — | 2.1 s | 11.9 GB | ≈0.2 s ✅ |

**Cold prefill and process cold start are two different things**: the former can happen every turn (any change in context needs recomputation), the latter happens only once when the service restarts. The 27B's 7.85-second cold prefill is enough to condemn it on its own — the user finishes speaking and just recomputing the prompt takes 8 seconds.

#### Scanning with the real prompt: only "look at screen" breaks the line

The table above used a fixed persona. Switching to the real 242-character system prompt in `voice_server.py` and running each of four real contexts once (`tools/ttft_scenarios.py`, median of 3 runs):

| Scenario | Context | First token (warm) | First token (cold) | First sentence spoken | decode |
|---|---|---|---|---|---|
| One question, one answer | 172 tok | **86 ms** | 130 ms | 393 ms | 26.0 tok/s |
| 6 turns of history | 335 tok | **86 ms** | 224 ms | 319 ms | 26.0 tok/s |
| Look at screen, 2K chars | 1661 tok | 91 ms | **1008 ms** | 435 ms | 26.2 tok/s |
| Look at screen, 4K chars | 3273 tok | 104 ms | **1944 ms** | 449 ms | 26.0 tok/s |

Two conclusions:

1. **The warm first token is almost independent of context** (86→104 ms; 172→3273 tokens adds only 18 ms). The prefix cache really is working, and everyday multi-turn dialogue stays at this magnitude forever — ten more turns will not make it slower.
2. **The cold first token grows linearly with context**, with a slope of about **0.58 ms/token** (i.e. cold prefill ≈ 1700 tok/s). So **look-at-screen is the only scenario that breaks 300 ms**: the OCR text changes every frame, so everything from the OCR block onwards has to be recomputed, and 1.6K characters means 1 second while 3.3K characters is close to 2 seconds.

This also explains the time composition of look-at-screen (CPU version: frame grab 0.07 + OCR 4.3~5.6 + **LLM first token 0.45~1.0** + generation). To compress it further, prioritise cutting the number of characters of OCR sent into the prompt (region-based OCR / only taking the main body of the window) rather than changing models — switching to 27B turns this item from 1 second into 8 seconds.

**Verdict**: 27B running bare cannot get into the voice hot path — it is not "a bit slower", it takes 3 seconds to speak the first sentence, which rolls back the whole 1~2 second result of 4.13. And in the 7B's 279 ms the LLM is only a small part; the bottleneck is still segmentation (0.42s) and ASR (0.19s), which is the healthy structure.

#### Tiers and engines

- **Hot path (wake response, commands, small talk, transitions): 1B–4B or 7B**. 7B is the current sweet spot, measured as passing every item.
- **Cold path (reasoning after OCR look-at-screen, meeting-minutes post-processing, long-document translation): 9B–14B / 27B**, where the user is willing to wait 5–10 seconds.
- But 1B can only be a router: measured, when asked to talk about "speculative decoding" it answered with financial terminology. Personified dialogue exposes it.
- **Use llama.cpp as a resident engine, not Magnitude**: cold start 15.4s vs 148.4s (after a crash you need to self-heal within 15 seconds); Magnitude **has no `n_ctx` switch**, so even 1B allocates KV for a fixed 131072 and wastes 12.8GB, while llama.cpp with `-c 4096` squeezes it to just over 3GB — and with 64GB of unified memory there is ASR / RapidOCR / vLLM competing for it; prefix hit 26ms vs 141ms for free.
- The only scenario where Magnitude is worth it is **when you insist on using 27B in the hot path**, using DFlash2 to pull decode up to 20.19 (2.6×).
- **Leave multi-stream concurrency to vLLM, do not introduce a third stack**: the assistant's real concurrency is "voice Q&A + caption translation + background summarisation", and continuous batching is vLLM's strength.

#### Re-measurement protocol (to prevent drift in the protocol)

```bash
# any OpenAI-compatible endpoint: vLLM / llama-server / Magnitude
../asr-venv/bin/python tools/llm_probe.py --base http://127.0.0.1:8000/v1 --model qwen --runs 3
# add --json for structured output; exit code 0 = all items pass, 1 = some item fails, so it can go straight into CI
```

What the script does: warm up once (model loading / kernel tuning is not counted) → N runs with the same prompt to measure the prefix-hit state → N runs with a unique nonce prepended to the prompt to measure cold prefill → print the median and the range. All the judging is in the script, you do not compute anything yourself.

Protocol requirements (these matter more than the numbers):

1. **The same GGUF, with its sha256**; if you change the quantisation, do not compare across.
2. `temperature=0`, `seed=42`, **≥3 runs each taking the median**, record the range as the **noise floor**; a difference must exceed the range to count as real.
3. **Measure the prefix-hit state and cold prefill separately**: for cold prefill, prepend a unique nonce to the front of the prompt and confirm `cache_n=0`.
4. **Align speculative decoding on both sides**: both on or both off, never one on while comparing (llama.cpp has shipped `--spec-type` since 7d7f028, so the earlier "llama has no equivalent implementation" no longer holds).
5. **Measure single-request (`-np 1`)**, and start a separate group for concurrency; do not mix them in one table.
6. Measure speed only, not quality — **changing the engine does not buy intelligence, only throughput**.

### 4.15 Streaming translation + SSE: from "appearing all at once after 0.8 seconds" to "starting to appear character by character at 0.26 seconds"

The 4.13 work pushed the original text down to 147ms, but **the translation still appeared as a whole sentence at once** (`trans 296~1210ms`). The user asked "can it be made streaming" — yes, and the gain is bigger than expected.

**First measure exactly where it is slow**: vLLM's system prompt is prefix-cached, so

| | Non-streaming (whole sentence) | Streaming first token | Streaming full |
|---|---|---|---|
| Short sentence 「元首和平宣言。」 | 368ms | **93ms** | 370ms |
| Long sentence (20 chars) | 756ms | **88ms** | 759ms |
| Longest sentence | 1309ms | **88ms** | 1309ms |
| Mean of 6 cases | 701ms | **96ms** | 690ms |

**The key point**: the first token takes only ~96ms while the whole sentence takes 700ms — what is slow is **per-token generation**, not prefill. So streaming is not merely "it looks faster": it makes **the time until the user sees the first character 275~1221ms earlier**, with the total duration completely unchanged.

#### Measured end to end (real HDMI meeting audio, 810 segments in a running session)

| Metric | Before the change | After the change |
|---|---|---|
| Segment end → original text | 147ms | 147ms |
| Segment end → **first translated token** | 830~1360ms (= waiting for the whole sentence) | **260ms** |
| Segment end → full translation | 830~1360ms | 808ms |

Perceived latency dropped **3~5×**, while GPU usage did not change (the same request, it just no longer blocks waiting for it to finish writing).

#### Implementation points

1. **`translate_stream()`**: parses vLLM's `data:` SSE stream and calls back on every delta received. Returns `(full_text, ttft_ms, total_ms)`. On an exception it **keeps the part already received** — half a sentence of translation is more useful than a blank.
2. **Push the full accumulated text, not the delta**: the front end's `applyUpd` was already replacing the whole line with `l.trans = u.trans`, so pushing the accumulated string means no change to the rendering logic. Measured: for 13/13 segments every intermediate state was a **strict prefix** of the final translation, so replacement does not flicker.
3. **Merge patches for the same seq** (`_push_update`): the front end polls every 250ms and streaming produces a delta every 80ms, so without merging one poll would receive 3 intermediate states of the same line and waste 2 redraws.
4. **Intermediate states carry no `trans_ms`, only `final` does**: otherwise the 「译 x.x s」 (translated in x.x s) on the page would climb all the way up.
5. **`--no-stream-trans` reverts to whole sentences**, used only when debugging.

#### Output side: 250ms polling → SSE push

Polling itself contributes 125ms on average. After switching to `/api/events` (`text/event-stream`) intermediate states are drawn the moment they arrive, and the measured push interval reaches **51ms** at its fastest.

**Hit a multi-client pitfall**: `take_updates()` is "take and clear", so two open pages fight over the patches and the second one stays at 「译文中…」 forever. So the SSE endpoint **does not use** `take_updates()`; instead each connection keeps its own `{seq: trans}` snapshot and diffs against it. Verification: with two clients subscribing at the same time, each received **97 versions / all 15 segments with intermediate states**, completely symmetric.

The front end keeps polling as a fallback: if `EventSource` errors it closes itself and falls back to `tick()`.

#### Also fixed: `pushLines` did not de-duplicate

When SSE reconnects the browser uses the original URL (`after` unchanged) and the server sends the old lines again. `pushLines` used to do `lines.push()` directly, producing duplicate lines. Now it de-duplicates by `idx`.

#### Why ASR was not put on the GPU (evaluated, rejected)

Translation already runs on the GPU (vLLM); that is the only place that should use it. ASR was worth evaluating, but:

- `sherpa-onnx`'s wheel is **compiled for pure CPU** (`ldd` finds no `libcudart`/`libnvinfer`)
- `from_sense_voice(provider="cuda")` **silently falls back**: `Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available providers: CPUExecutionProvider. Fallback to cpu!`
- `provider="trt"` is rejected outright: `Tensorrt support for Online models only`

Putting it on the GPU would mean rebuilding sherpa-onnx yourself (matching onnxruntime for CUDA 13.2, **and that wheel does not exist**, see 4.8), or writing TRT inference for fbank + CTC yourself as was done for rec. **And ASR only takes 147ms, so the ceiling on the gain is that 147ms** — compared with the 600ms saved by streaming, it is not worth it at all.

**Conclusion: the GPU is already being used in the right place.** The next penny in the caption pipeline should be spent on ASR accuracy, not speed.

### 4.16 Segmentation: from "fixed 3-second cut" to "cut at natural pauses" (2026-11-21)

**The problem**: watching the captions, the user felt "it hard-cuts a segment every 3 seconds". Reading the code shows it was in fact always driven by natural pauses — `--hold 0.42` decides silence, and `--max-s 3.0` is only a fallback cap. But **measurement proves the user's feeling was right**:

Taking 62 seconds of real meeting audio (recorded straight from the capture card with `pw-record`) and running offline segmentation, statistics over 129 segments:

| Parameter set | Segments | Silence cut | Hit cap | Median segment length | p75 |
|---|---|---|---|---|---|
| max=3.0 hold=0.42 (before the change) | 25 | **44%** | **56%** | 2.54s | 2.86s |
| max=4.0 | 19 | 68% | 32% | 3.22s | 3.62s |
| max=4.5 | 19 | 74% | 26% | 2.92s | 4.20s |
| max=5.0 | 17 | 76% | 24% | 3.08s | 4.72s |
| max=6.0 | 15 | 80% | 20% | 3.86s | 5.78s |
| max=8.0 | 15 | **100%** | **0%** | 3.20s | 5.84s |

**56% of the segments were hard-cut by hitting the cap** — so "fixed 3 seconds" is literally true for more than half of them.

**The root cause is not "using time", it is "there is only one fixed value, 0.42s, for the pause threshold"**: in news-style speech the small pauses between words often reach 0.3~0.4s, just enough to trigger a cut, so it cuts out half a sentence; while the long pause that comes when a sentence really ends (0.6~1.2s) often has not arrived yet and the 3-second cap gets there first. With both thresholds fixed, it necessarily fails at both ends.

**The change: layer the threshold by segment length + relax the fallback cap.**

```
voiced < seg_mid(1.2s)  → must wait for a long pause of hold_long(0.90s) to cut   # rather merge with the next sentence than cut it into pieces
voiced < seg_long(2.4s) → use the baseline hold(0.45s)
voiced ≥ seg_long(2.4s) → cut at hold_short(0.30s)                              # already dragged long enough, latency must not grow further
max_s = 6.0                                                                     # fallback
```

#### Pitfall stepped on: using "silence-cut share" as the criterion gives the wrong conclusion

By that metric, `max_s 7.0 + hold_short 0.25 + seg_long 1.8` achieves **100% silence cut with a p75 of only 3.88s**, which looks perfect. But listening to it on real audio, the ASR text is **more fragmented** than with the 4.5 setting:

| 4.5 setting (silence cut 74%) | 7.0 + 0.25/1.8 (silence cut 100%) |
|---|---|
| 所以去**扯**什么台湾会不会进行交易筹码？ | 所以去**者**什么，台湾会不会进行交易筹码？ |
| 了不起在中美会晤中国大陆**再讲一下**。 | 了不起在中美会晤。＋做好到路再讲一下啊 |

(`扯` "talk nonsense" turned into `者` "person"; `再讲一下` "talk a bit more" became the nonsense `做好到路再讲一下啊`.)

The reason is obvious: in speech, a 0.25-second pause is mostly **a breath between words**, and cutting there produces half a sentence. **"Did not hit the cap" and "cut at a good place" are two different things**, and the silence-cut share only measures the former. The only real criterion is whether the ASR text reads as complete, and that takes a human to look at.

#### The 6.0 cap was decided the same way (same audio run through `--wav`, comparing the text)

| max_s | Median | p75 | Text |
|---|---|---|---|
| 4.5 | 2.92s | 4.20s | 「有开了等一长会又拍了开了我会」; 「大会」 cut into 「大」 + 「会议」 |
| 5.5 | 3.86s | 4.72s | still two split sentences: 「…我会嗯。」 + 「之后才有亚尔达会议。」 |
| **6.0** | 3.86s | 5.78s | 「了不起在中美会晤中国大陆再讲一下啊，我认为就是这样了」 merged into one sentence |
| 7.0 | 3.86s | 5.78s | **hallucination appears**: the same spot 「等一长会议」 becomes 「德拉兰会议」 |

The 7.0 row is the most important negative evidence: **when the segment is too long the ASR starts making things up — longer is not better**. 6.0 is the optimum for text quality.

**Effect: the completeness of the ASR original text clearly improves** (same audio, offline run with `--wav`):

| Before the change (3s) | After the change (6s, layered) |
|---|---|
| 「这个都没有啊，怎么可能第一次。」+「结论那不可能的事嘛啊。」 | 「这个都没有啊，怎么可能第一次就有结论，那不可能的事。」 |
| 「呃，三强会物三强的峰会我。」+「不认为。」 | 「三角的峰会我不认为。」（「我不认为」 is no longer split apart） |
| 「所以去什麼台不会行交易？」 | 「所以去扯什么台湾会不会进行交易筹码？」 |
| 「了不起在中美会晤中国大陆在。」+「一下啊，我认为就是这样了。」 | 「了不起在中美会晤中国大陆再讲一下啊，我认为就是这样了。」 |

Row 3 illustrates it best: with too short a segment the ASR cannot use context and 「什么台不会行交易」 is gibberish; once the segment is stretched to a complete sentence it is recognised as 「什么台湾会不会进行交易筹码」. **Bad segmentation → bad ASR → even worse translation**, the chain is linked.

**Cost**: median time-to-text goes 2.54→3.86s and p75 2.86→5.78s (offline 62s sample). Online measurement gives a median of 4.1s. This is what "complete sentences" costs; change back to `--max-s 4.5` to revert to the intermediate state.

**Also fixed**: `--min-s 0.25` let pure noise segments through, SenseVoice returned `" 。"`, and the page showed an empty caption line while still burning a translation. Now `recognize()` uses the `SUBSTANCE` regex to filter substantial characters (CJK/kana/letters/digits) and discards anything that is only punctuation.

**How to tune it yourself**: `tools/seg_ab.py` runs only the Segmenter (it does not touch ASR/translation) and gives a result in seconds; `--sets` takes arbitrary sets of parameters:

```bash
./asr-venv/bin/python tools/seg_ab.py /tmp/meet60.wav
./asr-venv/bin/python tools/seg_ab.py /tmp/meet60.wav --sets "4.5:0.45:0.30:0.90,6.0:0.45:0.30:0.90"
```

The criteria are the **silence-cut share** (the higher it is, the more cuts land on natural pauses) and the **p75 segment length** (which is exactly the tail of the time-to-text).

**What is still unsolved**: the energy method cannot tell "a pause between words" from "a pause between sentences"; it can only guess from duration. Real sentence boundaries need falling intonation or a punctuation-level semantic model, which an energy threshold cannot do. 26% of segments are still hard cuts, caught by `soft-cut` backing off to an energy trough.

#### But "natural pauses" itself has a physical ceiling

`tools/pause_hist.py` measured the pause distribution of the same 62-second audio (same threshold convention as the Segmenter):

```
total pauses 59   total duration 17.3s / 62.0s
≥0.20s  35  59%
≥0.30s  26  44%
≥0.45s  17  29%
≥0.60s  10  17%
≥0.90s   0   0%     ← not a single one
```

**Chinese speech simply has no pauses of 0.9 seconds or more.** Even the longest ones are only just over 0.8 seconds (4 in the 0.8s bucket). Therefore:

- `--hold-long 0.9` is **almost never satisfied** in real speech. Its job is not "often cut at a long pause" but **preventing the beginning of a sentence from being cut into 0.5-second fragments** — a short segment would rather wait until it has accumulated `--seg-mid` and switched to the 0.45 threshold.
- What really lengthens the segments is `--max-s 3.0 → 4.5`, not the layered thresholds. Layering solves "do not cut into pieces", relaxing the cap solves "do not hard cut".
- Key corollary: **if a 4-second segment still hits the cap, it is not that the parameters are badly tuned, it is that this sentence itself is 5~6 seconds long**. Raising the cap further only buys a longer wait.

**To really solve it you have to replace the premise of "wait until a segment finishes"**: use streaming ASR (sherpa-onnx's online models — the fact that `provider="trt"` only supports online models is corroborating evidence) to produce text while speaking, decoupling latency from segment length, with segmentation used only to cut sentence boundaries for translation. The cost is that the Chinese accuracy of streaming models is generally not as good as the offline non-streaming SenseVoice, and that needs measurement before deciding.

### 4.17 The translation cannot hold "pure target language": prompt rewrite + residue detection (2026-11-21)

**Symptom**: in Chinese-Japanese translation the 7B copies the source text a lot. Outputs actually captured:

```
原文 我是主持人谭英文      → 我是主持人田英和      （整句中文照抄，等于没翻译）
原文 深zのアプリで何点给…  → 在深Z的应用中给深Z评分多少
原文 …中国大陆再讲一下啊，我认为就是这样了。
     → 了 pesticid在中国と米国の会見で…私はこれが正しいと认为。
原文 因为他的很容易跟韩国做对比是。
     → Because it is easy for him to compare with South Korea.   （整句变英文）
```

(The first case is copied wholesale from the Chinese, i.e. not translated at all; the third mixes in a raw English word `pesticid` and leaves the Chinese 「认为」 ("think") in the Japanese; the fourth turns into a whole English sentence.)

**Why the old prompt could not hold**: it only said "the translation must be pure Chinese / pure Japanese, not a single word of the source may remain" — all abstract rules, no counter-examples. And the peculiarity that **Chinese and Japanese share kanji** gives the model a lot of room to be lazy in a way that "looks right" — outputting the Chinese as-is into Japanese, where every character is a legitimate kanji, leaves the model with no incentive to change anything.

**Changes** (`_tr_system()`, also merging the two completely duplicated prompts in `translate` and `translate_stream`):

1. **Correct errors before translating**: explicitly authorise the model to fix homophone misrecognitions by context (`科意`→`刻意`, `三强贵物`→`三强会晤`). Without saying so it faithfully translates the wrong characters.
2. **Give real counter-examples**: using the bad outputs actually captured as negative teaching material is more effective than abstract rules — the 7B's tendency to imitate is far stronger than its tendency to follow rules.
3. **Require a self-check after translating**: explicitly "if any kana/Chinese grammar remains, rewrite the whole sentence".

**Measured comparison** (`tools/tr_quality.py`, 14 real cases, same vLLM):

| Source | Old prompt | New prompt |
|---|---|---|
| 我是主持人谭英文 | 我是主持人田英和 ← copied | 私は司会の谭英文です |
| 深zのアプリで何点给深zに評価するか | 在深Z的应用中给深Z评分多少 | 该怎么评价深z这款应用？ |
| 中方刻意给韩国好处 | 中国は韓国に特別な利益を与えた | 中国側は意図的に韓国に便宜を図った。 |
| 中方科意、韓国に良いところを与える | 中方希望给韩国好的印象 ← the misrecognised character was not corrected | 中方有意给予韩国一些好处。 |
| 身体面にアメリカは坚持するべきです | 美国在身体方面应该坚持 ← 「坚持」 copied | 在美国方面，美国应当坚持立场。 |

**But long sentences still leak**: `…之后才有` (the "才有" left in), `…交易筹码` (the "筹码" left in), `…我认为就是这样了` (the "认为" left in). This part is not fixed; the reason is below.

#### Added residue detection + a single retry, but essentially ineffective on long sentences

`lang_clean()` has two criteria:
- **Japanese→Chinese checks for kana** — zero false positives, reliable.
- **Chinese→Japanese checks for simplified-only glyphs** (`给/这/说/对/时/个`…, Japanese uses traditional or different glyphs) + **no kana in the whole sentence and ≥6 characters** (a sentence-length Japanese sentence cannot lack particle kana).

If dirt is detected it retries once with a reinforced instruction; if still dirty it keeps the first version (a half-sentence of mixed language is better than a blank, and it cannot retry endlessly and hold up the translation thread). The counters are `tr.retry / tr.fixed` in `/api/status`.

**Blind spot (measured on 14 real cases: not a single dirty translation was detected)**: 「才有」「筹码」「认为」 are all **words with identical forms in Chinese and Japanese** — Japanese can be written the same way, so nothing can be found at the character level. To detect it reliably you would have to do "does the translation contain a fragment of ≥2 characters from the source + a whitelist of same-form words", and the whitelist is extremely fragile; the cost of a false positive is an extra translation run per sentence.

**Conclusion: this is the capability ceiling of 7B, not a badly written prompt.** Even for the most obvious error, whole-sentence copying, changing the prompt only brings it from about 50% down to about 20%. Really solving it needs a different model: on this machine there are `qwen3627b` (19G), `Qwen3.8-27B` (16G), `qwen35a3b` (35B-A3B MoE, 23G). **The MoE one is the best** — only 3B active parameters, speed close to 7B, quality close to 35B. But vLLM currently occupies 34G/61G, leaving only 26G of headroom to start a new instance, which needs a maintenance window to switch, and would affect the voice assistant at the same time (8095 also uses the same `qwen`).

### 4.18 A/B on switching to 35B-A3B: quality won, latency lost (2026-11-21)

The previous section attributed "7B cannot hold pure target language" to a capability ceiling, so we took downtime and actually measured switching models. **All three arms were run on the same day**, with the same set of 14 sentences (including 10 measured bad cases) + 62 seconds of real audio.

| Arm | Runtime | Model | Median first token | Median full text | Known residue |
|---|---|---|---|---|---|
| **A** | vLLM | Qwen2.5-7B-AWQ | **88ms** | **571ms** | 7 places (`才有`/`筹码`/`SUCH`/`pesticid`/`认为`/whole-sentence English) |
| **B** | SGLang 0.6.0 | Qwen3.5-35B-A3B GPTQ-Int4 | 419ms | 933ms | **0 places** |
| **C** | vLLM | Qwen3.5-35B-A3B w4a16 | 400ms | 2300ms | not measured (same model) |

**B's quality really turned around**; all 7 known failures disappeared at once:

| Source | A (7B) | B (35B-A3B) |
|---|---|---|
| 这就投扯太远的了… | ここでは **SUCH** 問題を提起する可能性は低い | そもそもこのような場で提案されることはあり得ません |
| 了不起在中美会晤…我认为就是这样了 | **了 pesticid**在中国と米国の会見で…と**认为** | せいぜい米中首脳会談で中国側が改めて触れる程度でしょう |
| 因为他的很容易跟韩国做对比是 | **Because it is easy for him to compare with South Korea.** | それは韓国との対比が容易だからです |
| Love冷落日本 | 日本の愛は冷たくなった | アメリカが日本を冷遇している |

#### Why the 35B's first token cannot be pushed down: the prefix cache is unreachable

This is the most valuable finding of this round, more important than the latency numbers themselves.

Every caption sentence uses the **same 390-character system prompt**, and 7B turns it into something free through the prefix cache (533ms → 94ms). 35B cannot get it:

- **vLLM**: the log says it itself
  `Mamba cache mode is set to 'align' when prefix caching is enabled`
  `Prefix caching in Mamba cache 'align' mode ... support for Mamba layers is experimental`
  `Setting attention block size to 1056 tokens to ensure attention page size >= mamba page size`
  → the cache-reuse granularity is pushed up to **1056 tokens** by the mamba page, while our prompt is only ~300 tokens, so it **can never hit**. Measured `Prefix cache hit rate: 0.0%`, every sentence has to re-prefill 300 tokens, and the first token is pushed to 400ms.
- **SGLang**: it does cache (1022ms → 375ms), but 375ms is its floor and it still cannot get it back.

**So the latency is not "35B computes slowly", it is "paying the prompt bill again for every sentence".** Corollary: if you must use 35B, the most effective move is not tuning kernels but **cutting `_tr_system` from 390 characters to around 100** — 35B does not need 7B's few-shot examples. This move has not been tried yet.

#### Three pitfalls (all stepped on this time)

1. **The chat template enables thinking by default** (with `enable_thinking` undefined it takes the thinking branch). `max_tokens=200` all burns on thinking and `content` stays an empty string — the online symptom is "9~10 seconds between lines and not one character of translation". Fix: put `chat_template_kwargs: {"enable_thinking": false}` in the request body; `caption_server` now has a `--llm-think` switch (off by default). The Qwen2.5 template does not recognise this variable, and passing it does no harm.
2. **The vLLM command in the file cannot be run as-is**: `--gpu-memory-utilization 0.9` + `--max-model-len 262144` on 61G of unified memory squeezes OCR/ASR to death; `--disable-log-requests` has been removed in the current image (it exits directly with `unrecognized arguments`).
3. **First request takes 50 seconds**: MoE kernel auto-tuning. The log has `Using default MoE kernel config. Performance might be sub-optimal!` — it is missing `E=256,N=256,device_name=Orin,dtype=int4_w4a16.json` (the qwen36b version of the command in the file copies it from `/root/.cache` into the image). The smoke-test script's timeout must be set to 60 seconds, otherwise the first shot always fails falsely.

**The conclusion of this section was later overturned by 4.19:** changing the runtime (Magnitude) gives token-granularity prefix caching, and 35B can come up without changing the prompt. The judgement and evidence of the time are kept below, because the root-cause analysis of "why 35B is slow on vLLM" still holds.

### 4.19 Loading 35B-A3B with Magnitude: changing the runtime on the same machine squares the books (2026-11-21)

The conclusion of 4.18 was "35B wins on quality, loses on latency", and it lost on **not being able to get the prefix cache**. So the thing to do is not tune kernels but change the runtime — to **Magnitude** (YC S25, a local inference engine written in Rust, with kernel auto-tuning built in, GGUF route, OpenAI compatible).

**One command settles the difference**: sending the same 390-character system prompt repeatedly, Magnitude carries `timings` in every chunk and reports directly

```
第1发 prompt_n=296 cache_n=0   prompt=1130ms
第2发 prompt_n=1   cache_n=295 prompt=25ms   ← 前缀按 token 粒度复用
```

`cache_n` is far more reliable than "look at the first token": it tells you explicitly **how many tokens were reused from the KV prefix cache**. On the vLLM side that number is 0 (a 1056-token reuse granularity); on Magnitude it is 295/296.

#### Five-arm measurement (same set of 14 sentences, same `_tr_system`, same request shape)

| Arm | Runtime | Model | Median first token | Median full text | Known residue |
|---|---|---|---|---|---|
| A (old production) | vLLM | Qwen2.5-7B-AWQ | 88ms | 571ms | 7 places |
| B | SGLang 0.6.0 | 35B-A3B GPTQ | 419ms | 933ms | 0 |
| C | vLLM | 35B-A3B w4a16 | 400ms | 2300ms | not measured |
| E9B | Magnitude | Qwen3.5-9B Q4 | 157ms | 929ms | 0 |
| **D (current production)** | **Magnitude** | **Qwen3.5-35B-A3B UD-Q4_K_XL** | **89ms** | **487ms** | **0** |

Arm D is even faster than A on **short sentences**, and its quality equals B. On 62 seconds of real audio, 17 lines, all 7 of A's residues (`才有`/`筹码`/`SUCH`/`pesticid`/`认为`/whole-sentence English/`日本の愛は冷たくなった`) disappeared, leaving only one place where the ASR misrecognition `Love` (a mishearing of "美国") was kept as a proper noun.

#### But the online first token is ~490ms, not 89ms — this rule must be remembered

Only after going live did we find out: median `ttft` 490ms. Printing the server-reported `timings` with `CAP_DEBUG=1` made it clear:

```
[tr] hdr=531ms ttft=531ms | srv prompt_n=49 cache_n=290 prompt_ms=491
```

The cache hit (`cache_n=290`) is fine; what is slow is the **prefill of the new tokens**. The measured relationship is:

```
TTFT ≈ 25ms + 6~8ms × number of new tokens
```

- Short sentence (8~20 tokens) → 90~150ms, which is why the 14-sentence test in A/B looked so good
- **Real online segments are 5~6 second Japanese sentences (60~80 tokens) → 450~700ms**
- Plus the few dozen tokens of 【上文】 (which change every sentence and do not enter the cache) → that is where the 490ms comes from

The root cause is that MoE prefill is extremely unfriendly to **small batches**: 60 tokens make it read almost all the expert weights across 40 layers (22.9G), whereas at decode time it reads only the 3B that are active. So **MoE is "cheap to decode, expensive to prefill", and dense is the opposite**. This also explains why the 35B's first token on vLLM is also 400ms — it is not vLLM's fault, it is the physical characteristic of this class of model.

**Net effect: the time to translate the whole thing is on par with 7B (1071ms vs 1085ms), with only the first token moving later by about 300ms.** A caption lag of 0.5 seconds is perfectly acceptable, so we switched.

#### Three new pitfalls

1. **If thinking cannot be turned off the translation gets swallowed** (the same disease as item 1 in 4.18). Qwen3.5 recognises `chat_template_kwargs: {"enable_thinking": false}` (once off, reasoning goes from 640 characters to 0 and `content` is a normal 18 characters); but **LFM2.5 does not recognise this variable** and still emits 600-plus characters of reasoning first with `content` permanently empty — the first time we tried LFM we almost concluded wrongly that "Magnitude does not support turning thinking off". `reasoning_effort: "none"` gives LFM a straight 400, while Qwen accepts it.
2. **`--ctx-lines 0` becomes "carry all context"**: `prev[-self.ctx_lines:]` equals `[0:]` when it is 0, stuffing the whole history into the prompt (`prompt_n` rose from 80 to 138, first token costing 300ms more). Now explicitly blocked in `_ctx_before()` with a unit test added.
3. **Magnitude's own downloader is 3~5× slower than a direct connection**: `catalog pull` is about 10MB/s, while `curl` straight to HF reaches 15MB/s for the same file, so 23GB took 41 minutes. It writes weights in `blobs/lfs-sha256-<hash>` format — **do not hand-build an HF cache directory to stuff them in**, it will not recognise them.

#### Deployment shape: only 35B runs now (vLLM is stopped)

35B (27.9G) + system ≈ 33G/61G, leaving 28G. **vLLM 7B was stopped on 2026-11-21 at 11:53** (`docker stop vllm-qwen`; its restart policy is `no`, so it will not come back by itself, and there is no corresponding systemd unit). The container was not deleted; rolling back only needs `docker start vllm-qwen`. Configuration from the coexistence period, for the record: vLLM squeezed to `--gpu-memory-utilization 0.25` (about 15G) + 35B + system ≈ 48G/61G; the 7B's KV pool was originally 19.94G/373k tokens and we only used a few thousand, so squeezing it down cost nothing.

```
magnitude-dockerd.service   separate dockerd (separate data-root, does not touch the main daemon's three production containers)
magnitude.service           start the container → wait for /health → explicitly load 35B → wait for Ready (cold start ~150s)
caption-bench.service       --llm-base http://127.0.0.1:10100/inference/v1 --llm-model <id> --llm-key <key>
```

Both units are enabled, and **on 2026-11-21 11:54 they were genuinely handed over to systemd**: stop the manual dockerd → `systemctl start magnitude-dockerd` → `systemctl start magnitude`, cold start **77 seconds** to Ready; after the two business services reconnected, the caption first token was p50 474ms and `NRestarts` stopped growing. `/opt/meeting-copilot` is in `/etc/fstab`, and `local-fs.target` comes before dockerd, so the boot order is fine.

Migration steps (copy and use):

```bash
# 1 quiet the business services
sudo systemctl stop caption-bench voice-bench
# 2 stop the container (the data-root is untouched, container state is preserved)
DOCKER_HOST=unix:///opt/meeting-copilot/data/magnitude-data/docker.sock \
  docker stop -t 20 magnitude-bench
# 3 stop the manual dockerd
sudo pkill -TERM -f 'magnitude-data/daemo[n].json'
# 4 hand over to systemd (load the model after dockerd is up)
sudo systemctl start magnitude-dockerd && sudo systemctl start magnitude
# 5 reconnect the business services
sudo systemctl start caption-bench voice-bench
```

> ⚠️ **Pitfall 1 stepped on: `pkill -f` kills itself**. When you write `pkill -f 'dockerd --config-file .../daemon.json'`,
> the command line of this ssh remote shell also contains the same string, so `pkill` kills the shell as well, which
> shows up as `Exit 255` and a sudden disconnect.
> Use a character-class form like `daemo[n].json` so the regex cannot match its own command line.
>
> ⚠️ **Pitfall 2 stepped on: a manually started dockerd makes the unit fail and restart endlessly** (4631 cumulative retries
> that morning, the log flooded and CPU burned for nothing).
> The root cause is that `dockerd.pid` already exists and systemd is turned back every time by `process with PID xxx is still running`,
> with `Restart=on-failure` coming back every 5 seconds.
> `NRestarts` is a cumulative counter and does not reset after the migration, so **the only way to tell whether it is still
> looping is whether it keeps growing + whether `ActiveEnterTimestamp` changes**.

**Rolling back to 7B**: `docker start vllm-qwen` → wait for `/v1/models` (about 66s, possibly 250s the first time) → change `--llm-*` in `caption-bench` / `voice-bench` back to `--llm-model qwen` (base defaults to 8000), and restart those two services.

### 4.20 Cutting the system prompt short to squeeze the first token under 200ms: the short version is slower (2026-11-21)

4.19 attributed the online first token of 490ms to "prefill of the new tokens". The most obvious guess was: `_tr_system` is 390 characters (≈290 tokens), the largest block of fixed text in every request, so cutting it to ~95 characters (≈70 tokens) should save a lot.

That is how `--tr-prompt short` was added (`_tr_system_short()`, keeping only three hard rules — pure target language / correct homophone misrecognitions / half a sentence translates as half a sentence — with all examples and filler-word rules cut). Took downtime and ran four cells; the conclusion is the **opposite**:

| Cell | prompt_n (actually processed this round) | cache_n (reused) | First token p50 | First token p90 | Full text p50 | <200ms |
|---|---|---|---|---|---|---|
| full/ctx2 | 61 | 290 | 300ms | 345ms | 648ms | 0/24 |
| short/ctx2 | 124 | **0** | 367ms | 416ms | 749ms | 0/24 |
| full/ctx0 | 26 | 290 | 244ms | 264ms | 553ms | 0/24 |
| short/ctx0 | 89 | **0** | 340ms | 615ms | 687ms | 0/24 |

(12 real Japanese segments × 2 rounds per cell, quiet machine, `tools/ttft_prompt_ab.py`)

**The short version's `cache_n` is always 0.** The 290-token prefix hits every sentence, and the 70-token prefix never hits even once; so although the short version has 220 fewer tokens of system text, it re-prefills the whole prompt (including context), costing 60~70ms more than the long version.

It is not that "too short a prefix cannot be cached". When the same prompt is sent repeatedly both settings hit cleanly:

```
full   第1发 1223ms cache=0/298 → 第2发 72ms cache=297/1
short  第1发  457ms cache=0/92  → 第2发 67ms cache=91/1
```

The difference appears only when **the user content changes every sentence** (the real online shape): the long prefix reuses 290 tokens stably, the short prefix drops to zero outright. We did not find the mechanism (Magnitude is closed), but the behaviour is very stable: 24 out of 24 requests were 0. **If you can use the long one, use the long one — it costs nothing.**

#### The first token cannot get under 200ms; this is a hard floor

Putting `prompt_n` and the first token side by side, the model is

```
TTFT ≈ 65ms + 3.5~6.5ms × new tokens
```

- new tokens = 1 (resend as-is) → 72ms
- new tokens = 26 (drop the context, keep only this sentence) → 244ms
- new tokens = 61 (the current online shape: this sentence + two sentences of context) → 300ms

**To get under 200ms the new tokens have to be squeezed to within 20~30**, i.e. a sentence of about 15 characters and no context. A real meeting segment is 5~6 seconds and 60~80 characters, so physically it cannot be reached. So the target of "first token 200ms" should be abandoned and attention turned elsewhere:

- The only thing still reducible is the context: `ctx0` saves 56ms (300→244), at the cost of quality collapsing (`だ。` → `的。`; a standalone half sentence loses its subject and translates into garbage) — not worth it
- ~~The 190ms between the online 490ms and the offline 300ms is a scheduling problem~~ → **4.21 proves it is not**: that 190ms is just "online sentences are longer and the context is longer", and after converting to tokens online and offline match exactly (65ms + 4.5~6.7ms/token)

**Landed as**: `--tr-prompt` still defaults to `full`; the short version stays in the code for dense models — dense prefill is billed linearly per token, so the "cut the prompt" path only holds for 9B/7B, not for MoE.

### 4.21 ctx-lines cut to 1 (-90ms), workers cut to 1 (no difference), 9B loses to 35B on long sentences (2026-11-21 morning)

**First pin down the field semantics** (I got this backwards once and it made every later derivation wrong): Magnitude's `prompt_n` is **the number of tokens actually processed this round**, `cache_n` is **what was reused from the KV prefix**, and only their sum is the length of the whole prompt. So "new tokens = prompt_n − cache_n" is wrong — **new tokens are simply `prompt_n`**.

#### 1. `--ctx-lines 2 → 1`: measured ~90ms saved, not 200ms

Turning on `Environment=CAP_DEBUG=1` online and reading the engine-reported `prompt_ms` directly, on the same news stream with one round captured before and one after (48 vs 52 items):

| Configuration | New tokens p50 | Engine prompt_ms p50 | First token p50 | First token p90 |
|---|---|---|---|---|
| `--ctx-lines 1` | 53 | 403ms | **435ms** | 811ms |
| `--ctx-lines 2` | 73 | 485ms | **524ms** | 761ms |

20 fewer tokens, 89ms saved ⇒ **slope about 4.5ms/token**. The "cut one sentence of context saves 200ms" estimated at 6.7ms/token is optimistic: to really save 200ms the previous sentence would have to be more than 45 characters. For short news sentences (the previous sentence around 20 characters) it is on the order of 90ms.

#### 2. `--tr-workers 1 vs 2`: no difference

Under ctx1, workers=1 → 435ms, workers=2 → 475ms, but the `prompt_n` of the two rounds also differ (53 vs 61, the news content changed). After normalising at 4.5ms/token **the two are equal**. A GGUF engine does not use worker count to push down the first token; `q_tr=0` and `lag≈300ms` throughout, one translation thread is enough. Settled on 1 (one queueing stage less).

#### 3. 9B vs 35B head to head on 62 seconds of real audio (`tools/wav_seg_ab.py`)

16 real segments, the same ASR text, the same context, ctx-lines 1:

| Arm | First token p50 | First token p90 | Full text p50 | Hard defect |
|---|---|---|---|---|
| **35B (current production)** | **293ms** | 407ms | **693ms** | — |
| 9B | 326ms | 378ms | 1085ms | sentence 6 directly repeats sentence 5's translation |

9B **loses on both metrics**, 56% slower on the full text, and it also produced a repetition hallucination. The earlier flattering 157ms for 9B came from short sentences (8~20 tokens); on long sentences "dense prefill is cheaper" did not cash in (and it decodes more slowly too). **Keep 35B; the 9B line can be closed.**

> The 35B first token of 293ms here was measured while caption-bench was stopped; online with the same parameters it is 435ms. The 140ms difference is capture / ASR / draft threads competing for CPU, not the engine — so when "it looks good offline and bad online", look at who is competing rather than doubting the model.

**Quality did not drop**: comparing ctx2 and ctx1 sentence by sentence over the same 16 sentences, 5 are identical and all 11 differences are synonymous substitutions (`凝縮`→`必要となる`, `対比`→`比較`), with no semantic drift and no language dropping. The `Love冷落日本` sentence got the subject wrong in both settings (ASR heard "美国" as `Love`), unrelated to ctx.

**Landed as**: `caption-bench` settled on `--tr-workers 1 --ctx-lines 1`; `CAP_DEBUG` is turned off after sampling (one line per entry, tens of MB a day if left on).

### 4.22 The voice assistant also switches to 35B-A3B (2026-11-21 midday)

**Why it was 7B before the switch**: not because it was chosen, but because of historical order. `voice-bench` initially only served voice dialogue, and at that time the machine only had vLLM's Qwen2.5-7B-AWQ (port 8000); later the captions were also connected to 8000 first, and only in 4.19 moved to Magnitude's 35B. The voice assistant just stayed on 7B and nobody touched it. What really blocked the switch were two **fears that had not been verified at the time**: MoE prefill is expensive (afraid the first token would get slower), and the GGUF engine has only one slot (afraid the two services would queue behind each other).

#### 1. Measure four real scenarios (`tools/voice_llm_ab.py`)

The voice assistant and the captions have completely different load shapes: captions have a long system (290 tokens, cache hit) + short content + short output; the assistant has a short system + **possibly several hundred characters of OCR** + long output (160 tokens). So the caption conclusion cannot be carried over directly.

`reply-lang=zh`, 3 runs per scenario, **changing the question each time** (sending the same sentence repeatedly makes the whole prompt hit the cache, `prompt_n` drops straight to 1 and the first token is a fake 78ms):

| Scenario | 7B first token | 7B full text | 35B first token | 35B full text |
|---|---|---|---|---|
| chat-short (pure Q&A) | 98ms | 461ms | 272ms | 887ms |
| chat-multi (with 3 turns of history) | 104ms | 1070ms | 426ms | 905ms |
| ocr-mid (~100 chars on screen) | 102ms | 1899ms | 314ms | **1488ms** |
| ocr-long (a full spoken script on screen) | 102ms | 2925ms | 472ms | **2127ms** |

- **First token 170~370ms slower** (a fixed cost: 35B has more weights to warm up), but **long answers are 25~35% faster in full text** — MoE decode activates only 3B, while 7B dense has to run the whole model for every token.
- Perceived latency = first token + first-sentence synthesis. For a first sentence of about 45 characters: 35B about 1.2s, 7B about 0.95s. A 250ms difference, audible, but what it buys is the next item.
- **The quality gap is overwhelming**: with a Japanese system and a Chinese answer, 7B produces `会議的时间确认 needed。` (Chinese-Japanese mixture), `優先度が一番高いのは…-topic1-と判断します` (garbage token), `necesita дополнительной информации` (Spanish). 35B had none of that in 12 cases.

#### 2. Concurrency is not a problem (this overturned the fear held before the switch)

`magnitude serve --help` shows only one `-h`, which looks like a single slot. Measured: while making the assistant repeatedly run long ocr-long outputs, at the same time typing short caption requests, **the caption first token only went from 204ms to 233ms (+29ms)**. The engine can schedule concurrently, and the two services sharing 10100 do not queue behind each other. Online the caption first token p50 stayed at 435~474ms, with `q_tr=0` and `dropped=0`.

#### 3. The pitfall that got stuck: where the language instruction is attached matters more than what it says

The voice assistant's value is "announce in Chinese during a Japanese meeting" (`--reply-lang zh`). But after switching to 35B, **all 12 answers were Japanese (0/12)**; 7B was 8/12. `tools/voice_lang_fix.py` tried four formulations, two scenarios × 3 runs:

| Formulation | 35B Chinese rate | 7B Chinese rate |
|---|---|---|
| Instruction at the end of the system (**original implementation**) | **0/6** | 3/6 |
| Instruction at the end of the user message | **6/6** | 6/6 |
| Hard declaration at the start of the system | 0/6 | 5/6 |
| Both at the end of the user message and the start of the system | 5/6 | 6/6 |

Putting it at the end of the user message fixed it immediately — **the instruction has to sit next to the generation position**. It is especially pronounced with 35B: it "understands the context too well", so when the user speaks Japanese and the screen is Japanese it ignores a distant Chinese instruction; 7B just mechanically follows the last sentence.

But attaching to the user message alone is not enough: after splitting the system into a "pure dialogue version" (removing the screen rules), the system becomes **entirely Japanese with not one Chinese character**, and the language instruction gets pushed back down again (3/3 all Japanese). So the final approach is **attach both**: one at the end of the system + one at the end of the user message. After the change, 4/4 were all Chinese.

#### 4. Also fixed a bad habit specific to 35B

When `look` was not passed (no image was looked at), 35B would answer 「屏幕文字中没有显示今天的会议时间」 ("the screen text does not show today's meeting time") — even though this turn grabbed no frame at all, which sounded like shirking. Adding a **negative instruction** "do not mention the screen when there is no [屏幕文字]" **could not suppress it** (2/2 unchanged).

The root cause is that the word 「[屏幕文字]」 appears repeatedly in the system, so the model refers to it. The final fix is a **dynamic system**: `voice_server.Llm` is split into `system_core` (pure dialogue) and `system + screen_rule` (with screen rules), and `stream()` picks one by `"[屏幕文字]" in text`. **Do not give it the word and the problem is gone.**

#### 5. Changes and rollback

- `voice_server.py`: `Llm(api_key, think)` (Magnitude needs Bearer, Qwen3.5 must have thinking off), `available()` with an auth header, `system_core`/`screen_rule` dynamic system, language instruction attached in both places, new `--llm-key` / `--llm-think`
- `deploy/voice-bench.service`: `--llm-base .../10100/inference/v1` + the 35B model id + key
- New tools: `tools/voice_llm_ab.py` (four-scenario multi-arm comparison + `--concurrent` preemption test + language compliance statistics), `tools/voice_lang_fix.py` (comparison of four instruction formulations)
- **Rollback**: `docker start vllm-qwen` → change the three `--llm-*` lines in the unit back to `--llm-model qwen` (base defaults to 8000) → restart voice-bench. **The vLLM container was stopped at 11:53** (idling and occupying 15G) but not deleted; rollback follows the two steps above.

### 4.23 "As soon as the next sentence arrives, the previous whole line disappears": scroll anchor drift from full redraw (2026-11-21 midday)

**Symptom** (reported by the user): when the previous sentence is long text and its translation is still streaming character by character, as soon as the next sentence arrives the previous sentence's **whole line (original + translation) disappears**, and it is only obvious with long sentences.

#### 1. First prove it is not a data problem

Two independent pieces of evidence, both produced online:

- `tools/sse_regress.py` subscribing to `/api/events` for 120 seconds: **407 events, 206 lines, 0 regressions, 0 becoming empty**. Every seq's translation grew as a strict prefix.
- `tr: {retry: 1, fixed: 0}` in `/api/status`: out of 219 segments only 1 triggered "residue detection → synchronous retranslation", and it was not recovered (`fixed=0` means no whole-line replacement). So it is not retranslation replacing the long sentence.

And there is genuinely no path on the server that clears `trans` (only `cap.clear()`, which is attached only to the "清屏" (clear screen) button on `/api/control`), and `cap.lines` is a `deque(maxlen=400)`, with the current 219 lines not reaching the limit.

#### 2. Root cause: full `innerHTML` rebuild + pixel `scrollY` restore

The old `render()` did this on every push received (about every 80ms when streaming):

```js
const y = window.scrollY;
const atBottom = (window.innerHeight + y + 80) >= document.body.scrollHeight;
m.innerHTML = lines.slice(-30).map(...)   // rebuild all 30 lines
window.scrollTo(0, atBottom ? document.body.scrollHeight : y);
```

The long sentence's translation is growing line by line, so after the rebuild the **total height of `#main` has changed but `y` is still that same pixel value** — the same `y` now points at different content, and the line the user is reading is pushed out of the viewport. The longer the sentence, the bigger the height change, which is why only long sentences show it. On top of that, rebuilding 30 lines of DOM every 80ms makes the browser reflow the whole tree, and dropped frames amplify the impression.

**Note that this is not "the data was deleted", it is "the viewport anchor drifted"** — so refreshing the page, or scrolling back a little, shows that the line is still there. To judge which class a "content disappeared" bug belongs to, check this first: if the data layer can reproduce a regression it is the backend; if it cannot, it is rendering.

#### 3. The fix: switch to incremental rendering

Give each line a **resident DOM node** (`dom: seq -> {root, head, o, tr}`), and only write the text nodes that change:

- When the translation streams and grows, **only the current line's text node gets longer**; the 29 lines above are not touched at all and their geometry does not move
- New lines are appended with `appendChild`; ordering is only rearranged when it is genuinely out of order (`appendChild` on an existing node is a "move", not a rebuild, and does not lose content)
- Scrolling only happens when "a new line really arrived" and "we were already stuck to the bottom"; **the translation growing inside an existing line no longer triggers a scroll** — that is exactly what was shoving the view away
- The draft line is also a resident node, `remove()`d when `draft` is empty, no more string concatenation

#### 4. How it was verified: extract the page JS and run it against a DOM mock

`tools/dom_render_test.js`: extracts the `<script>` from `caption_server.py`, injects a minimal DOM (`createElement` / `appendChild` / `remove` / `querySelector` / `nextElementSibling`), and asserts on **node identity** directly, not just the text:

- After 20 streamed updates of a long sentence, line 30's DOM node is **still the same object** (`===`), and its text has not changed by one character
- After a new line arrives the preceding line nodes are not rebuilt, and the previous sentence's translation is preserved in full
- When two sentences push interleaved they do not step on each other; after `final` the translation is kept and the cursor class name disappears
- Inserting/removing the draft line does not affect existing lines; clearing returns to 「等待翻译…」 ("waiting for translation…")

All 18 items passed. **And it was re-extracted from the online `caption_server.py` and run again on the Jetson** — verifying the online file, not a local copy:

```bash
python3 -c "import re,pathlib;s=pathlib.Path('caption_server.py').read_text();\
pathlib.Path('tools/_page_check.js').write_text(re.search(r'<script>(.*?)</script>',s,re.S).group(1))"
node tools/dom_render_test.js
```

> Pitfall: after changing the page JS you must re-extract `tools/_page_check.js`, otherwise you are testing the old code. The local script compares sha1 and warns when they differ.

**Cost**: this change is front-end only, but `PAGE` is a module-level constant, so **caption-bench must be restarted for it to take effect**, and a restart clears the lines accumulated so far.

### 4.24 Captions changed to "newest at the top, older lines scroll down" (2026-11-21 afternoon)

The user's requirement: the newest line is displayed at the top, and when new text appears the whole screen pushes down. The version from midday 2026-11-21 was "append at the bottom + scroll to the bottom".

The change is only in the page rendering layer (`PAGE` in `caption_server.py`):

| Location | Old (append at the bottom) | New (newest at the top) |
|---|---|---|
| Node creation in `syncLine()` | `appendChild` | `insertBefore(node, firstElementChild)` |
| `syncDraft()` | attached at the end | attached at the front (the draft is the newest line) |
| Order check | ascending by seq | `orderOk()`: draft first, the rest **seq descending** |
| Reclaim | delete the earliest when the window scrolls | same, only now it sits at the **bottom** |
| Scrolling | `scrollTo(bottom)` when stuck to the bottom | **do nothing when stuck to the top** (inserting a new line naturally pushes content down); when scrolled down, no compensation means the view gets shoved, see below |

**The scroll anchor has to be flipped together with it**, otherwise the 4.23 bug gets committed again in reverse:

- Stuck to the top (`scrollY <= 24`): do nothing; the new line is inserted at the front → the content shifts down a whole screen, which is the "scrolls down" impression.
- Scrolled down reading older lines: anchor on "the original first line", and after insertion compensate with `scrollBy` using the **difference** in `getBoundingClientRect().top`.
  It is easy to get the direction backwards: when the content is pushed down by `h` pixels, `top` **increases** by `h`, so you need `scrollBy(0, +h)` (scroll `h` more);
  writing `anchorTop - topNow` makes the scroll go the wrong way and the view jump even more.
- `#main` gets `overflow-anchor:none`: Chrome's built-in scroll anchoring secretly changes `scrollY` when inserting at the top, and it drifts when stacked on top of our hand-computed compensation.

The test `tools/dom_render_test.js` was changed to 26 items in step, with these key new ones:

```
[ok] 最新的一条在最上（35） / 最老的一条沉到底部（6）
[ok] 新行插到最上，原来的 35 退到第二位
[ok] 草稿行在最上 / 草稿移除后 36 回到最上
[ok] 新行把上面的内容往下推了 100px
[ok] 滚动补偿到位，scrollY=400（期望 400）
[ok] 贴顶时不做补偿，新行自然把内容往下推
```

For this the DOM mock gained `insertBefore` / `getBoundingClientRect` (simulating layout as "which child × 100px") and `window.scrollBy`.
Again it was **re-extracted from the online file on the Jetson and run**, and all 26 items passed; the backend `test_caption_pipe.py` 14 groups also all passed.

> Note: this order is "reverse order". If some day you want to continue downwards and need to read forwards, flip the four places above back (the core is changing the `orderOk` traversal direction and the scroll anchor together).

### 4.25 Using on-screen text to correct ASR proper nouns: hint on demand, no full injection (2026-11-21 afternoon)

The request: "produce captions while watching the screen, and fix errors with the words on the screen". This can be done, but **you must not stuff the whole block of screen text into the translation request** — measurement shows a negative return.

#### First measure the cost and the effect (`tools/screen_hint_ab.py`, 6 real segments × three arms)

| Arm | Screen characters | First token p50 | prompt_n | prompt_ms |
|---|---|---|---|---|
| none | 0 | 87ms\* | 1 | 25 |
| h60 | 113 | 423ms | 134 | 352 |
| h120 | 176 | 417ms | 160 | 361 |

\* The `none` arm hit the cache for the whole prompt because the same sentence had just been translated online (`cache_n=351`), so its speed is not credible; converted to the ctx1 online shape, 113 characters ≈ +72 tokens ≈ **+250~320ms per sentence**.

**Worse still is the effect**: three of h60's five sentences got worse — 「三社の会談」 ("the three-company talks") translated as 「这三家公司的首次会谈」 ("the first meeting of these three companies"), 「この後ちょっと…」 ("a bit after this…") sprouted 「这位先生在这三家公司此次会谈中」 ("this gentleman in these three companies' talks") out of nowhere, and 「特朗普那边则是特朗普那边的情况」 ("on Trump's side it is Trump's side's situation") repeated. There are two root causes:
1. The capture card films the presenter's whole screen and **our own caption page is open on it**, so the screen text contains the previous sentence's Chinese translation → feeding the translation back in as reference is a self-loop;
2. The bottom third of the screen, the title bar and the mail list have nothing to do with this sentence, and the model goes to "make it fit".

#### So only "hint on a hit" is done

`ScreenHints` (`caption_server.py`):
- In the background, every 15 seconds it asks voice-bench's `/api/look` for one OCR (**the caption service does not touch `/dev/video0`**, that device is exclusive and voice-bench holds it resident; internally it caches by picture fingerprint, about 0.5s for the same picture and about 1.3s for a new one)
- Filters three kinds of junk: page UI words, email/URL lines, and **lines sharing a common substring of 8+ characters with the last 30 captions** (to prevent the self-loop)
- Extracts only three kinds of candidates: numbers of 4+ digits, katakana of 4~10 mora, kanji runs of 3~8 characters; takes the reading of each (pyopenjtalk `run_frontend`, 3ms per sentence), and drops readings outside 4~14 mora
- Before translating, splits this sentence into morphemes, takes reading windows of 1~4 morphemes, and computes the edit distance against the candidate readings: only **≤1 and a different spelling** counts as a hit, at most one word hinted per segment
- On a hit it appends one sentence to the end of the request body: `【屏幕词】「畔蒜泰助」——本句的「阪井泰介」若指同一事物，请改用屏幕写法。` ("[screen word] ... — if this sentence's ... refers to the same thing, please use the spelling on screen", 37 characters ≈ 25 tokens, about +100ms, **paid only on this segment**); with no hit not a single token is added → zero cost, zero pollution

**Why Latin letters are not handled**: OpenJTalk reads `CloudflareObservability` letter by letter as 50 mora of 「しーえるおー…」, which can never match a Japanese reading, and measurement showed all 40 candidates were email/URL fragments. If the English word on screen is also correctly written by the ASR there is no need to hint anyway; the case where it is heard as katakana should go through **ASR hotwords**, not this path.

#### Current state and how to verify
- With the unit off, `screen` in `/api/status` is `null`; with it on it gives `polls/hits/terms/sample/last_ms/last_hint`
- The page title bar gains a 「屏幕 N 词 · 命中 M」 (N screen words · M hits); hovering shows the term sample, the most recent hit and the timing — **if you cannot see it, it is not in effect**, which is why it is out in the open
- Online measurement (`--screen-hints --screen-every 15`): `q_tr=0`, `dropped=0`, lag≈290ms, first token p50 473→443ms (within noise, **no measurable cost**); `terms` 14~40 entries, changing with the screen
- **A real hit has not happened yet**: during this period the screen showed an Outlook mail list, web pages and chat windows, unrelated to the meeting content, and `hits` stayed at 0. The logic is covered by `test_screen_hints` (15 groups, including a stubbed reader)

#### Manually create one hit to confirm the path really works (measured on 35B)

Taking one real bad sentence from production and artificially giving it a screen word:
```
原文（ASR）：同小委部同じベットネって違う夢見ているのか、見ないといけない。
无提示   → 同小委部是否抱着同样的期待，还是必须去确认？          ttft 299ms / pn=33
带提示   → 同小委员会是否做着不同的梦，必须加以确认。            ttft 364ms / pn=62
提示内容： 【屏幕词】「小委員会」——本句的「同小委部」若指同一事物，请改用屏幕写法。
```
**+29 tokens ⇒ +65ms**, and the proper noun really was corrected. So the mechanism holds; what is missing is only "that word actually appearing on screen".

> Verdict: the **ceiling of this path is "homophone misrecognition of proper nouns"**, and it cannot save mishearing of ordinary words (`つぶさに` → `つぶさんに` and the like are hopeless). In everyday meetings 90% of the misrecognitions are the latter, so do not expect this to raise overall quality.

### 4.26 The page stuck at "连接中…" (connecting…) with all buttons dead: PAGE not being a raw string broke the JS (2026-11-21 afternoon)

Symptom (the first refresh after 4.25 went live): the page had only the header and 「等待声音…」 ("waiting for sound…"), with `已识别 0 段` (0 segments recognised), `延迟 —` (latency —), `-99 dB`, `连接中…` — and clicking the three buttons did nothing; **while the server side was completely fine** — `systemctl is-active` was active, `/api/status` had 77 segments, `level_db=-29`, and `/api/poll` returned 30KB of JSON directly.

#### Locating it

The four groups of values in the screenshot are exactly the **initial DOM hard-coded in PAGE** (`<span id="db">-99 dB</span>`, `<span id="live">连接中…</span>`). Since even `paint(true)` (at the end of the script, which turns the status light to 「运行中」 "running") had not executed, it was not "the network is down / SSE cannot connect" but **the whole `<script>` had not run**. Grabbing the page the server emitted and handing it to node:

```
node --check /tmp/page.js
SyntaxError: Invalid or unexpected token   ← line 270
```
The culprit was the tooltip added in 4.25:
```python
sc.title = ... + '\n错误：' + ...      # looks fine in the source
```
`PAGE` was an ordinary `"""` string at the time, so Python ate the `'\n'` into a **real newline** first, and the JS the browser received was
```js
sc.title = (... + '
错误：' + ...)
```
The single-quoted string was cut by the newline → script syntax error → not a single character executed. **The HTML is static, so the screen "looked normal"; only the script was dead.**

#### The fix (two layers)
1. Change `PAGE = r"""..."""` to a raw string, so `\n`/`\u0001` in the JS reach the browser as-is (`HTML = r"""` in `voice_server.py` has always been raw, which is why that side never stepped on it)
2. `test_page_js_syntax`: extracts the rendered `<script>` and runs `node --check` on it, plus a fallback check that "single quotes must be paired on each line" (skipped on machines without node)

#### Lessons
- A healthy server ≠ a live page. **To judge "did the page really run its JS", look at the initial values that only JS changes** (status light, dB, segment count), do not just look at the server log.
- Whenever you write JS escapes into `PAGE`, first confirm it is raw; after changing it you must run `node --check` once, do not wait until a refresh to find out.
- Front-end changes still need a **manual F5** (`PAGE` is a constant, and restarting the server does not refresh the browser for you).

## 5. Command-line tools


```bash
# record 8 seconds, print level, noise floor, speech segments, transcription
../asr-venv/bin/python voice_probe.py --seconds 8 --threads 12

# process an existing wav, look only at segmentation, no transcription
../asr-venv/bin/python voice_probe.py --wav a.wav --no-asr

# SenseVoice benchmark (same audio, specified segments)
../asr-venv/bin/python tools/bench_sensevoice.py --wav a.wav --segments 18.43-21.72,41.95-45.51

# TTS: synthesise and report first-chunk latency
../asr-venv/bin/python tts_probe.py --text "確認してください。" --out /tmp/a.wav

# run this before changing models: first-token latency / single-request decode / first sentence spoken (criteria in 4.14)
../asr-venv/bin/python tools/llm_probe.py --base http://127.0.0.1:8000/v1 --model qwen --runs 3

# scan four context tiers with the real system prompt (one Q&A / 6 turns of history / look at screen 2K·4K chars)
../asr-venv/bin/python tools/ttft_scenarios.py --base http://127.0.0.1:8000/v1 --model qwen

# look at the screen once (frame grab + OCR, no LLM call); keep threads at 8 or below, 12 threads measured twice as slow
curl -s -X POST -d '{"force":true}' localhost:8095/api/look

# CPU vs TensorRT rec backend A/B (criteria: same line count + mean-score difference < 0.01)
../asr-venv/bin/python tools/trt_rec_probe.py --engine models/rec_fp16.engine --image /tmp/frame0.png

# rebuild the TRT engine (mandatory after changing JetPack, about 11~12 minutes; how to set the profile, see 4.8)
trtexec --onnx=<rapidocr>/models/ch_PP-OCRv4_rec_infer.onnx \
        --saveEngine=models/rec_fp16.engine --fp16 \
        --minShapes=x:1x3x48x256 --optShapes=x:6x3x48x320 --maxShapes=x:32x3x48x2048

# streaming vs whole sentence: comparison of translated first-token latency (criteria in 4.15)
../asr-venv/bin/python tools/trans_stream_probe.py --base http://127.0.0.1:8000/v1 --model qwen

# verify caption streaming really pushes intermediate states (check that patches are increasing prefixes of the final translation)
../asr-venv/bin/python tools/stream_verify.py 40

# SSE push verification: number of intermediate states + push interval (fastest should reach ~50ms, not limited by the 250ms poll)
../asr-venv/bin/python tools/sse_verify.py 45

# two clients subscribing at once, confirming multiple clients do not fight over patches
../asr-venv/bin/python tools/sse_multi_client.py

# end to end: segment end → original text / first translated token / full translation
../asr-venv/bin/python tools/e2e_latency.py 40

# ---- changing models / downtime (criteria in 4.18) ----
# capture one arm before and after changing the LLM: 14 real sentences × N runs, recording first token / full length / kanji ratio
../asr-venv/bin/python tools/ab_translate.py --tag A --out /tmp/ab_A.json
../asr-venv/bin/python tools/ab_translate.py --compare /tmp/ab_A.json /tmp/ab_B.json

# smoke test after changing models (timeout must be ≥60s: the first request is running MoE kernel tuning)
../asr-venv/bin/python tools/smoke_llm.py

# is the prefix cache actually working (send the same 390-character system prompt 4 times, see whether the first token drops)
../asr-venv/bin/python tools/cache_probe.py
../asr-venv/bin/python tools/cache_probe.py --style short   # does a short prefix still hit (4.20)

# first token only: prompt long/short × with/without context, four-cell comparison (judge by cache_n, not by the first token)
curl -s http://127.0.0.1:8096/api/poll > /tmp/poll.json
../asr-venv/bin/python tools/ttft_prompt_ab.py --cases-from /tmp/poll.json --save-cases /tmp/cases.json
../asr-venv/bin/python tools/ttft_prompt_ab.py --cases /tmp/cases.json \
    --base http://127.0.0.1:10100/inference/v1 --model '<gguf id>' --key mag-xxx \
    --out /tmp/ttft_prompt.json

# head-to-head multi-engine comparison on the same real audio (ASR runs once, all arms share the text and context; criteria in 4.21)
../asr-venv/bin/python tools/wav_seg_ab.py --wav /tmp/meet60.wav --ctx-lines 1 \
    --arm '35B|http://127.0.0.1:10100/inference/v1|hf:...gguf|mag-xxx' \
    --arm '9B|http://127.0.0.1:10100/inference/v1|qwen3.5-9b:gguf:q4|mag-xxx' \
    --out /tmp/wav_ab.json

# to compare only ctx-lines, change the number and run again (0 / 1 / 2, one round each, same 16 segments)

# is stuffing screen text into the translation request worth it (criteria: first-token cost + whether the translation actually changed)
curl -s localhost:8096/api/poll > /tmp/poll.json
curl -s -X POST localhost:8095/api/look > /tmp/ocr.json
../asr-venv/bin/python tools/screen_hint_ab.py --poll /tmp/poll.json --ocr /tmp/ocr.json \
    --limit 6 --chars 60,120 --base http://127.0.0.1:10100/inference/v1 \
    --model '<gguf id>' --key mag-xxx --out /tmp/screen_ab.json

# can the screen term list collide with ASR homophone misrecognitions (0 hits = the input for this path does not exist at all)
../asr-venv/bin/python tools/screen_terms_probe.py --poll /tmp/poll.json --ocr /tmp/ocr.json

# Magnitude only: read the engine-reported cache_n / prompt_ms / ttft directly, and verify the thinking switch
../asr-venv/bin/python tools/mag_sse_probe.py --model 'qwen3.5-9b:gguf:q4' --repeats 3

# why is the online first token slower than the engine? add Environment=CAP_DEBUG=1 to the unit and look at these two segments:
#   hdr=response headers received  ttft=first character  | srv prompt_n/cache_n/prompt_ms is computed by the engine itself
sudo journalctl -u caption-bench --since '-2min' | grep '\[tr\]'

# Magnitude's container and model (cold start ~150s, idempotent, can be run repeatedly)
sudo /usr/local/bin/magnitude-up.sh

# SSE protocol diagnosis: non-streaming vs streaming, whether the connection is closed after [DONE]
../asr-venv/bin/python tools/sse_probe.py

# switch the runtime on port 8000 (vLLM 7B / SGLang 35B / vLLM 35B), with readiness polling and rollback
./tools/llm_switch.sh status
MEM=0.62 CTX=8192 SPEC=0 ./tools/llm_switch.sh start-sgl
./tools/llm_switch.sh start-vllm          # rollback: reuse the original container, measured 66s

# ---- changing the voice assistant's model (criteria in 4.22) ----
# four scenarios × multiple arms: pure Q&A / with history / 100 characters on screen / a spoken script on screen
../asr-venv/bin/python tools/voice_llm_ab.py \
    --arm '7B|http://127.0.0.1:8000/v1|qwen|' \
    --arm '35B|http://127.0.0.1:10100/inference/v1|<gguf id>|mag-xxx' \
    --reps 3 --reply-lang zh
# add --concurrent: how much the caption first token is dragged when the assistant's long output saturates (measured only +29ms)

# how the language instruction must be written for the model to obey (at the end of system 0/6, at the end of user 6/6)
../asr-venv/bin/python tools/voice_lang_fix.py \
    --arm '35B|http://127.0.0.1:10100/inference/v1|<gguf id>|mag-xxx'

# one end-to-end turn (including TTS playback), look at llm_first_ms and the reply language
curl -s -N -X POST localhost:8095/api/turn \
     -H 'Content-Type: application/json' -d '{"text":"今日の会議は何時からですか？"}'

# ---- caption page rendering (criteria in 4.23) ----
# does the data layer regress/become empty: subscribe to SSE and compare seq by seq whether it grows as a strict prefix
../asr-venv/bin/python tools/sse_regress.py --sec 120
# does the page rendering rebuild/delete lines: a DOM mock asserts node identity directly (running on the server is harder evidence)
python3 -c "import re,pathlib;s=pathlib.Path('caption_server.py').read_text();\
pathlib.Path('tools/_page_check.js').write_text(re.search(r'<script>(.*?)</script>',s,re.S).group(1))"
node tools/dom_render_test.js

# playback
aplay -D plughw:CARD=H340,DEV=0 /tmp/a.wav
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `--hp` | 150 | high-pass cutoff in Hz, 0 disables |
| `--margin` | 6 | VAD threshold = noise floor + margin |
| `--min-peak-margin` | 18 | how many dB the segment peak must be above the floor, to block small transients near the threshold |
| `--min-gap` | 0.40 | silence allowed inside a segment; shorter than this does not cut |
| `--threads` | 0 | ctranslate2 threads, 0=default (measured as only 4) |

## 6. Verification

```bash
python -m unittest test_voice_probe    # 33 items, no microphone or whisper needed
python -m unittest test_tts_routing    # 15 items, no model weights needed
python -m unittest test_wake           # 12 items, no microphone needed
python -m unittest test_barge          # 24 items, no microphone needed
python -m unittest test_vision         # 22 items, no capture card needed (intent/redaction/fingerprint/protection/resident frame grab)

# barge-in decision chain end to end (real VAD + real ASR + real TTS, synthetic audio fed as frames instead of the microphone)
../asr-venv/bin/python tools/barge_e2e.py

# HDMI capture card probe (warm-up discarding built in; with a picture it saves /tmp/hdmi_ok_*.jpg)
./tools/hdmi_probe.sh 10 1
```

`test_voice_probe` covers: 60Hz suppression (above −24dB), 1kHz essentially lossless, frame-level levels aligned with theoretical RMS, hysteresis segmentation (a pause between words does not cut / a long gap does cut / short sharp sounds dropped), peak margin boundaries, wav mono/stereo and bit-depth validation.

`test_tts_routing` covers: pure punctuation must be intercepted into an empty string, pure numbers pick the voice by hint (a Chinese context no longer falls through to Japanese), decimal points are not broken by sentence splitting.

`test_barge` covers: playback clock accumulation and expiry, returning to zero after cancel, crosstalk discrimination (whole-sentence hit / truncated fragment / real interjection / short command / punctuation difference), frame handover not mixing in the old ring buffer, the threshold not being dragged off by crosstalk.

## 7. TODO

1. **Barge-in** — ✅ speaking during playback stops it (see 4.6), crosstalk does not false-trigger
2. **Always-on listening** — ✅ wake word 「小八你好」 is wired in (see 4.5), and after a turn it returns to waiting for the wake word automatically
3. **vLLM autostart at boot** — right now it is only a **rollback backup** (both services go through Magnitude's 35B) and does not start at boot; to roll back, first `docker start vllm-qwen`. The server probes liveness every turn, so the order does not affect correctness, but it will degrade to echo first
4. **First-chunk latency optimisation** — a good part of the 636ms is pyopenjtalk phonemisation; consider cutting the first sentence shorter
5. **Filling in unit tests** — `test_tts_routing.py` already covers voice routing and sentence splitting, but `voice_server.py`'s Silero end-of-speech is still missing
6. **Calibrating barge-in on real human voice** — the decision chain is currently verified with synthetic audio; the thresholds under real speaking rate/volume still need one round of tuning against actual recordings
7. **Connecting the capture card back into the business flow** — ✅ the 1080p30 link works (see 4.7), ✅ frame grab → OCR → into the dialogue is connected (see 4.8); the next step could be region OCR (recognising only the main body of the window) to press 6 seconds under 3
8. **Power verification for long captures** — it currently hangs off the dock (which once dropped out at 166s); before running hour-scale continuous capture it is better to plug straight into the machine's port
9. **The 4B–9B sweet spot tier** — ✅ measured and closed: on 62s of real audio 9B loses on both first token and full text (326/1085 vs 35B's 293/693) and also repeats, and the earlier 157ms was only from short sentences (see 4.21)
10. **Handing the hand-started dockerd over to systemd** — needs a ~3-minute window (stop dockerd → stop the container → magnitude cold start 150s, both services down). The failing restart loop has already been stopped; wait for the next machine reboot to take over naturally (see the warning in 4.19)
11. **Pushing the voice assistant's first token down further** — after switching to 35B it is 343~627ms, 170~370ms slower than 7B. Could try: cutting `history[-6:]` to `[-4:]` (fewer new tokens), or giving 35B a shorter system (rejected on MoE, see 4.20, but needs re-measuring for the voice scenario)
