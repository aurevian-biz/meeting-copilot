# Usage

[**English**](usage.md) · [中文](usage.zh.md) · [日本語](usage.ja.md)

## Meeting captions (8096)

Open `http://<device-ip>:8096` and click **开始 / 停止 / 清屏** ("Start / Stop / Clear") in the top right:

- **开始 / 停止**: controls capture and segmentation; `clear` wipes the lines and resets seq (the frontend also clears its local DOM)
- **听 HDMI** ("Listen to HDMI"): plays the audio the capture card receives to your headphones, for monitoring during the meeting
- **音量条 / dB** ("volume bar / dB"): live level and threshold; when the bar turns green it is above the threshold and sound is coming through
- **状态灯** ("status light"): `连接中…` ("connecting…") → `运行中` / `已停止` / `断开`
- **屏幕 N 词 · 命中 M** ("screen N terms · M hits"): hover to see term samples, recent hits, and how long one screen look takes
- **最新一条在最上** ("newest on top"): the draft line occupies the very top, so scrolling back through history is never pushed away by new lines

### What to check when the page doesn't respond

The status light stays at 「连接中…」, dB reads `-99`, segment count is `0`, clicking the buttons does nothing — all of these are initial values hard-coded in the HTML, which means **the script never ran**; it is not a network problem. Check in this order:

1. Does `curl -s localhost:8096/api/status` return any data? (server alive ≠ page alive)
2. Any `SyntaxError` in the browser console?
3. `.venv/bin/python test_caption_pipe.py` (`test_page_js_syntax` among them runs `node --check`)
4. Changed the frontend? **F5**: `PAGE` is a module-level constant, so restarting the service won't refresh the HTML in your browser

## Voice assistant (8095)

Open `http://<device-ip>:8095`:

- Press the button and speak, or say the wake word **「小八你好」** ("Xiao Ba, hello") to trigger the always-on listener
- Speaking while it is playing back interrupts it (`barge-in`); crosstalk does not false-trigger
- The **「看屏幕」** ("Look at the screen") button (or a spoken question) fires one OCR pass and feeds the screen text to the LLM
- The browser needs **one manual click** the first time before it will play audio: autoplay policy blocks it. Playback goes through the local `pw-play` (`--local-speak`) and does not depend on browser autoplay

## Common flags

### Caption segmentation (the defaults were tuned against measurements — don't change them casually)

| Flag | Default | Notes |
|---|---|---|
| `--max-s` | 6.0 | Hard-cut ceiling. At 4.5s, 26% of segments get cut (the word 「大会」, "conference", is sliced in two); **past 7s you get hallucinations** |
| `--hold` | 0.45 | Baseline silence threshold; tiered pauses also have `--hold-short 0.3` / `--hold-long 0.9` |
| `--seg-mid` / `--seg-long` | 1.2 / 2.4 | Short, medium and long sentences use different pause thresholds |
| `--soft-cut` | 0.7 | Before a hard cut, the search window (seconds) for backtracking into the tail to find an energy trough |
| `--min-s` | 0.25 | Anything shorter is dropped outright (clicks, coughs) |
| `--ctx-lines` | 1 (in unit) | Number of context lines; it changes every sentence, so cutting it saves about 90ms |
| `--tr-workers` | 1 (in unit) | Extra workers make no difference to single-sentence latency, they only fight over VRAM |
| `--tr-prompt` | full | The short version is actually slower, see [llm-serving.md](llm-serving.md) |
| `--screen-hints` / `--screen-every` | off / 15s | On-demand screen term hints, see below |
| `--wav` | — | No capture; run a single wav file instead (reproduce segmentation problems offline) |

### Screen term hints (optional)

With `--screen-hints` on, every `--screen-every` seconds the server asks voice-bench's `/api/look` for one OCR pass and extracts candidate terms. Only when **a word with similar pronunciation but a different spelling appears in this sentence's ASR text** does it pass one hint to translation:

```
【屏幕词】「畔蒜泰助」——本句的「阪井泰介」若指同一事物，请改用屏幕写法。
```

(The hint text itself is emitted in Chinese.)

- A hit costs about +25 tokens (+65ms), paid only on that segment; a miss adds not a single token
- **Don't stuff the whole screen text into translation**: measured at +70 tokens / +250ms, and 3 of 6 sentences got worse (our caption page is itself on the screen, so it loops on itself)
- The ceiling is "homophone misspellings in proper nouns"; it cannot rescue ordinary mishearing (`つぶさに` → `つぶさんに`)
- Keep Latin words out of the term list: OpenJTalk reads them letter by letter as 50 morae and they never match

## How to read status

```bash
curl -s localhost:8096/api/status | python -m json.tool
```

Key fields:

| Field | Meaning | Healthy value |
|---|---|---|
| `q_asr` | segments waiting for ASR | 0 |
| `q_tr` | segments waiting for translation | 0; **rising = translation can't keep up** |
| `dropped` | segments dropped from the queue | 0 |
| `ttft_ms` | first-token latency of the translation | 400~600ms |
| `trans_ms` | whole sentence translated | ~1000ms |
| `lag_ms` | end-to-end latency | ~290ms |
| `screen` | term list state | `terms / hits / polls / last_ms / last_err` |
| `monitor` | playback state | `on / starts / error` |

One command for the whole picture: `bash scripts/healthcheck.sh`.

## Offline reproduction

When you don't want to keep talking into the microphone, run the same segmentation chain on a wav file:

```bash
.venv/bin/python caption_server.py --wav sample.wav --no-draft
.venv/bin/python tools/wav_seg_ab.py --wav sample.wav --a 0.45 --b 0.7
.venv/bin/python tools/sse_regress.py     # data layer: 0 regressions, 0 empty
```
