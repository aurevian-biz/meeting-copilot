# Changelog

[**English**](CHANGELOG.md) · [中文](CHANGELOG.zh.md) · [日本語](CHANGELOG.ja.md)

This repository is the cleaned-up result of five days of on-device development
(2026-11-17 → 11-21); the first open-source release squashes that process into one version.
The measurements behind each decision, and the approaches that were rejected, are in
[docs/en/engineering-log.md](docs/en/engineering-log.md).

## v0.1.0 — 2026-11-21

### Added
- Voice assistant (`:8095`): wake word, VAD segmentation, SenseVoice, LLM, per-sentence
  Piper synthesis, barge-in, screen OCR, browser test bench
- Meeting captions (`:8096`): PipeWire capture, layered-pause segmentation, streaming
  translation, SSE, incremental DOM, newest line on top, monitor playback
- Screen term hints: on-demand hint only when the reading is similar *and* one character
  is shared — tokens are added only on a hit
- TensorRT FP16 backend replacing RapidOCR's rec stage (OCR 4.12 s → 1.17 s / 132 lines)
- Full deployment: 4 systemd units, one-shot install, model downloader, health check,
  offline regression tests
- Trilingual README and 11 documents in Chinese / Japanese / English, plus the engineering log

### Frozen defaults
- `--tr-workers 1 --ctx-lines 1 --tr-prompt full`
- `--max-s 6.0` + layered pauses + `--soft-cut 0.7` + `--min-s 0.25`
- `--screen-hints --screen-every 15`

### Key fixes
- Scroll anchor drift caused by full re-render (a long line pushed the previous one away)
- Scroll compensation sign inverted in the newest-on-top layout
- `PAGE` not a raw string broke the embedded JS: page stuck on 「连接中…」, buttons dead
- Language directive only in `system` made 35B answer in the wrong language → attach to
  both `system` and `user`
- Qwen3.5 enables thinking by default and eats `max_tokens` → explicit
  `enable_thinking=false`
- Injecting the whole screen text caused a feedback loop → hint only on a hit
- `pkill -f` killing its own ssh session; dockerd pidfile causing an endless restart loop
- Latin words in the screen term list read letter-by-letter by OpenJTalk → kept out

### Known limitations
- The screen term hints have not yet hit a real term in a real meeting (logic is covered
  by stubbed cases)
- Front-end changes still require a manual F5
- 35B weights take ~150 s to load after a cold start
- A true powered-off cold restart has not been re-verified end to end
