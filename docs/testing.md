# Testing: layered regression, no microphone required

[**English**](testing.md) · [中文](testing.zh.md) · [日本語](testing.ja.md)

Changing one segmentation parameter, swapping a model, or touching the front end can break some
other stage. These tests are split by layer, and **every one of them runs offline** (ASR / LLM / OCR
are stubbed), so no model has to be online and nobody has to talk into a microphone.

## Run everything

```bash
cd /opt/meeting-copilot

# Data layer (script style, ASR/translation stubbed, 16 groups)
.venv/bin/python test_caption_pipe.py

# The other 5 unittest files
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py

# Render layer (26 checks, needs node)
node tools/dom_render_test.js

# Data-layer self-check: 0 translation regressions, 0 blanks (**live**, connects to :8096 for 90 s)
.venv/bin/python tools/sse_regress.py
```

> `test_caption_pipe.py` is a **script**: the stubs are installed inside `__main__`, so don't run it
> through pytest on its own (the stubs would not be in place and you get false failures). The other
> 5 are standard `unittest`, so either pytest or `python -m unittest` works.
>
> `tools/sse_regress.py` is **not an offline test**: it connects to a running :8096 and subscribes to
> the live stream for 90 s, checking whether any translation regressed or went blank. Run
> `systemctl start caption-bench` first.

## What each layer covers

| Layer | File | Covers |
|---|---|---|
| Caption data | `test_caption_pipe.py` | Two-stage output (source before translation), queue decoupling, context lines, dropping short segments, soft cut, tiered pauses, drafts, clear screen, both `--tr-prompt` levels, screen-vocabulary hits, **page JS syntax** |
| Render | `tools/dom_render_test.js` | 26 checks: node identity, order (newest on top), scroll compensation, reverse insertion, draft row position |
| Data self-check | `tools/sse_regress.py` | In the SSE stream the translation must be a growing prefix: **0 regressions, 0 blanks** |
| Barge-in | `test_barge.py` | Speaking during playback stops it; crosstalk does not false-trigger |
| TTS routing | `test_tts_routing.py` | JA/ZH voice switches automatically; numbers inside a Chinese answer are not read with a Japanese voice |
| Vision | `test_vision.py` | Warm-up black frames, OCR redaction (`sk-***`), TensorRT attachment |
| Audio front end | `test_voice_probe.py` | 150 Hz high-pass, VAD, segmentation |
| Wake | `test_wake.py` | KWS triggering and debounce |

### Why `test_page_js_syntax` exists

There was an incident on 2026-11-21: a tooltip was added to the page with `'\n'` in it, and `PAGE`
was not a raw string at the time, so Python turned the JS escape into a real newline, the
single-quoted string was cut in half → **the entire `<script>` block had a syntax error, the page
sat at 「连接中…」 (Connecting…), every button was dead, and the server was perfectly healthy**.

This case pulls the rendered `<script>` out and hands it to `node --check`, plus a fallback check
that single quotes are paired on every line (skipped on machines without node). CI runs it too.
**Change `PAGE` without running this and you have not really changed anything.**

## A/B and probes (`tools/`)

These are not regression tests. They are "measure before you swap" tools, run individually as needed:

| Tool | Purpose |
|---|---|
| `wav_seg_ab.py` | Head-to-head comparison of two segmentation parameter sets on the same wav |
| `voice_llm_ab.py` | Score an LLM swap against the diagnostic signals (captions / assistant) |
| `ttft_prompt_ab.py` | First token across prompt level × prefix cache × new tokens |
| `ttft_scenarios.py` | First-token scenarios by sentence length / number of context lines |
| `screen_hint_ab.py` / `screen_terms_probe.py` | Whether screen-term hints pay off / whether they land on ASR misspellings |
| `trans_stream_probe.py` / `stream_verify.py` | Streaming vs whole-sentence, whether the translation grows as a prefix |
| `sse_verify.py` / `sse_multi_client.py` | SSE push interval; whether two open pages steal patches from each other |
| `bench_sensevoice.py` / `trt_rec_probe.py` | ASR and TensorRT backend benchmarks |
| `check_env.py` | Post-install self-check: deps / models / devices / silent fallback to CPU |

## Acceptance checklist for a model or parameter change

1. `.venv/bin/python test_caption_pipe.py` → `ALL PASS`
2. `.venv/bin/python tools/wav_seg_ab.py --wav <62 s of real audio>` → segmentation completeness (**look only at whether the ASR text forms sentences**, not at the share of silence cuts)
3. `voice_llm_ab.py` → language purity, hallucination, latency of long answers
4. After deploy, `bash scripts/healthcheck.sh` → `q_asr=0 q_tr=0 dropped=0`
5. Front-end change → restart the service + **F5** + `test_page_js_syntax`

## CI

`.github/workflows/ci.yml` does only three things that need no hardware:

1. `python -m compileall -q .` (syntax across the whole repo)
2. Extract the `<script>` from `caption_server.py` and run `node --check`
3. Scan the repo and confirm there are no keys / local paths / private IPs

Anything that needs the real device (ASR, OCR, TensorRT, GPU) can only be run on the box. Do not
expect CI to cover it.
