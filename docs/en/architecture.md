# Architecture: two pipelines, one LLM, three processes

[**English**](architecture.md) · [中文](../zh/architecture.md) · [日本語](../ja/architecture.md)

```
                               ┌───────────────────────────────────────────────────────┐
   HDMI capture card ─ UVC ───→│  vision.py (resident frame grab / OCR / TensorRT rec) │──→ /api/look
                               └───────────────────────────────────────────────────────┘    │
                                                                                            │
   ┌──────────── voice-bench :8095 (voice assistant) ──────────────────────────────────┐
   │  H340 mic → KWS 「小八你好」→ pw-record → Silero VAD → SenseVoice (ja)            │    │
   │       → LLM(35B) → sentence split → Piper(ja/zh) → pw-play → local headset        │←────┘
   │  Browser triggers and displays only; recording and playback run on the Jetson     │
   └───────────────────────────────────────────────────────────────────────────────────┘
                                     │ shared OpenAI-compatible endpoint
                          ┌──────────┴──────────┐
                          │  Magnitude :10100   │  (or vLLM :8000)
                          │  Qwen3.5-35B-A3B    │
                          └──────────┬──────────┘
   ┌──────────── caption-bench :8096 (meeting captions) ───────────────────────────────┐
   │  capture audio → pw-record (non-exclusive) → Segmenter (tiered pause + soft-cut)  │
   │     → SenseVoice → translate_stream (streaming, on-demand screen-term hint)       │
   │     → SSE /api/events → browser incremental DOM (newest line on top)              │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

> 「小八你好」, the wake-word phrase the KWS listens for, is kept verbatim in the diagram.

**Why two processes**: the voice assistant is turn-based Q&A that has to speak out loud, while
captions are a continuous stream that has to stay real time. Their thread models and latency budgets
are completely different. Forcing them into one process makes them fight over ASR and the LLM. They
share OCR results through `/api/look` in `voice_server.py` — **the capture card is an exclusive
device, and only voice-bench holds `/dev/video0` permanently**, so captions ask it for screen terms
instead of opening the device themselves.

## Processes and ports

| Process | Port | Role | Key environment variables |
|---|---|---|---|
| `magnitude-dockerd` | socket (not TCP) | standalone dockerd | — |
| `magnitude` | 10100 | 35B-A3B inference | `MC_DATA` / `MC_MODEL_ID` |
| `voice-bench` | 8095 | voice dialogue + looking at the screen | `XDG_RUNTIME_DIR`, `LLM_*`, `MC_SINK` |
| `caption-bench` | 8096 | bilingual captions + monitor playback | `XDG_RUNTIME_DIR`, `LLM_*` |

## HTTP API

**8095 voice assistant**

| Method | Path | Notes |
|---|---|---|
| GET | `/` | Test bench page |
| GET | `/api/health` | Liveness (includes whether the LLM is available) |
| POST | `/api/turn` | One full turn: record → ASR → LLM → TTS (event stream) |
| POST | `/api/say` | Synthesize one sentence directly (debugging TTS) |
| POST | `/api/look` | Grab a frame → OCR → return screen text; the caption service gets its vocabulary from here |
| POST | `/api/route` | Run only the routing / end-of-utterance logic |
| POST | `/api/reset` | Clear the dialogue context |
| GET | `/api/live` `/api/poll` | State push / fallback polling |
| POST | `/api/kws_check` | Wake-word self-check |

**8096 meeting captions**

| Method | Path | Notes |
|---|---|---|
| GET | `/` | Caption page |
| GET | `/api/events` | SSE, one translation increment every ~80ms; the front end falls back to `/api/poll` on disconnect |
| GET | `/api/poll?after=N` | Polling fallback (250ms) |
| GET | `/api/status` | Full state, including `q_asr/q_tr/dropped/screen/monitor` |
| POST | `/api/control` | `{"action": "start" \| "stop" \| "clear"}` |
| POST | `/api/monitor` | 「听 HDMI」 (Listen to HDMI): play capture-card audio to the headset |

## Why captions are "streaming + SSE + incremental DOM"

1. **Streaming LLM**: waiting until a whole sentence is translated costs 0.8~1.3s; pushing token by
   token starts filling in at 0.26s
2. **SSE**: the server checks for updates every 50ms and pushes as soon as an intermediate state
   arrives; polling would wait up to 250ms longer. The server does not use `take_updates()` (that is
   "take and clear", so two open pages steal patches from each other) — each connection compares
   against its own snapshot of the translation
3. **Incremental DOM**: early on it was a full `innerHTML` rewrite plus pixel `scrollY` backfill, so
   the anchor drifted while a long translation was growing: after the rebuild the same y pointed at
   different content → "the moment the next line arrives, the previous line disappears". Now every
   line is a persistent node and only the text nodes that changed are written;
   `#main{overflow-anchor:none}`

The order is **newest line on top**, old lines pushed down. The visual anchor moves from the bottom to
the top; when you scroll down to read old lines, `scrollY` is compensated by the `top` delta (with
reverse order the sign of the compensation is `+`).

## File map

| File | Responsibility |
|---|---|
| `voice_server.py` | wake / record / ASR / LLM / TTS / web test bench / `/api/look` |
| `caption_server.py` | capture / segmentation / ASR / streaming translation / SSE / caption page |
| `voice_probe.py` | audio front end: high-pass, VAD, segmentation, SenseVoice wrapper |
| `vision.py` | frame grab (drops 40 warm-up frames), OCR, screen-term candidates |
| `trt_rec.py` | replaces RapidOCR's rec backend with TensorRT FP16 |
| `paths.py` | the **only** directory convention; `MC_DATA` can override it |
| `deploy/` | 4 systemd units + install script |
| `tools/` | A/B measurement, self-checks, probes (none of them run at runtime) |
| `test_*.py` | stubbed regression, no microphone or live model needed |
| `docs/en/engineering-log.md` | the measurements and rejected approaches behind every default |

## Threads / real time

On the caption path, `q_asr` (segments waiting for ASR) and `q_tr` (segments waiting for
translation) are the **only reliable real-time diagnostic signal**:

```
q_asr = 0, q_tr = 0, dropped = 0   → keeping up
q_tr keeps rising                  → translation cannot keep up; lower --ctx-lines first, then check whether the LLM is hogging VRAM
dropped > 0                        → the queue has started dropping segments; latency is past real time and must be dealt with
```

`--tr-workers 1` is the settled value: adding workers makes no difference to single-sentence latency,
it only takes more VRAM. `--ctx-lines 1` drops the context that changes every sentence; measured, it
saves about 90ms.
