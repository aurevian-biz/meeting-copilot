# Contributing

[**English**](CONTRIBUTING.md) · [中文](CONTRIBUTING.zh.md) · [日本語](CONTRIBUTING.ja.md)

Issues and pull requests are welcome. Half the value of this project is *why* it is built
this way, so **when you change a default, bring the measurement with it**.

## Before you push

```bash
python -m compileall -q .
.venv/bin/python test_caption_pipe.py        # 16 groups, incl. page JS syntax
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py
node tools/dom_render_test.js
bash scripts/healthcheck.sh                  # if you have the hardware
```

Add a case at the layer you touched:

| Change | Test you must add |
|---|---|
| Segmentation / queue / translation request | `test_caption_pipe.py` |
| Caption page JS (`PAGE`) | `test_page_js_syntax` + `tools/dom_render_test.js`; reload the page afterwards |
| SSE / data format | `tools/sse_regress.py` |
| Audio front end / VAD | `test_voice_probe.py` |
| OCR / screen term hints | `test_vision.py` / `test_screen_hints` (inside `test_caption_pipe.py`) |

## Conventions

1. **No absolute paths in code.** Directories go through `paths.py`; override with `MC_DATA`.
2. **No secrets.** `conf/env.example` is only a template; real values live in
   `/etc/meeting-copilot/env`.
3. **`PAGE` must stay `r"""`.** A normal Python string eats the `\n` inside the embedded JS,
   which breaks the whole `<script>`: the page freezes on 「连接中…」 while the server looks
   perfectly healthy. This has already happened once.
4. **State the criterion when swapping models or parameters** — `q_asr/q_tr/dropped`,
   `ttft_ms`, segmentation completeness, language purity. "Feels faster" is not a criterion.
5. **Keep the three languages in sync.** Every document and README exists as
   `x.md` (English), `x.zh.md` (Chinese) and `x.ja.md` (Japanese). A change to one is a
   change to all three.
6. **Comments in code are English.** User-facing strings stay in their original language —
   the UI and the prompts are part of the product, not documentation.

## Maintainer note (redaction)

The public repository is derived from a private working tree that also contains debug
screenshots, runtime artefacts, and units with machine-specific paths, LAN addresses and
an LLM bearer token. Regenerate with the packaging script (kept in the private tree only):

```bash
python tools/package_opensource.py --dest /path/to/meeting-copilot --force
```

It picks only the files the runtime needs, rewrites sensitive strings by rule, and self-checks
"no secrets / no local paths / no private IPs". It never deletes `docs/` or the READMEs.
CI (`.github/workflows/ci.yml`) runs the same scan to catch a slip before it is merged.
