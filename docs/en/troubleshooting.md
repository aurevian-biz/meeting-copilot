# Troubleshooting: look up by symptom, not by module

[**English**](troubleshooting.md) · [中文](../zh/troubleshooting.md) · [日本語](../ja/troubleshooting.md)

Every item here was actually hit, and **most of them produce no error at all** — a silent failure
costs more time than a crash.

## 1. Page stuck at 「连接中…」 (Connecting…) / dB reads -99 / all buttons dead

Start with `/api/status`: if the server has data and the segment count is climbing, then **the page
script never ran at all**. Those values are hard-coded initial values in the HTML, not "SSE cannot
connect".

```bash
curl -s localhost:8096/api/status          # is the server alive?
.venv/bin/python test_caption_pipe.py      # test_page_js_syntax runs node --check
```

The usual root cause is a JS syntax error inside `PAGE` (a non-raw Python string swallowed the
`\n`). After a front-end change: **restart the service + F5 in the browser**. `PAGE` is a constant;
restarting the service does not refresh the HTML in the browser.

## 2. Service is up, but there is no sound / no recording

| Check | Cause |
|---|---|
| Does the unit contain `Environment=XDG_RUNTIME_DIR=/run/user/<uid>` | Without it `pw-play` / `pw-record` **fail silently** |
| `pactl list short sinks` — is the default sink HDMI | Audio is going to the monitor; point `--local-sink` at the H340 |
| `arecord` reports 「设备忙」 (device busy) | Switch to `--backend pw`; PipeWire lets several clients read the same source |
| No sound in the browser the first time | Autoplay policy blocks it; playback already goes through the local `pw-play`, so click once in the page |

## 3. systemd

| Symptom | Action |
|---|---|
| It becomes `failed` after `stop` | `caption-bench` needs `TimeoutStopSec=20`, otherwise the capture thread drags on to 90s and gets SIGKILLed |
| `NRestarts` keeps climbing | Check whether `ActiveEnterTimestamp` shows repeated restarts; a `pkill -f` that kills your own ssh also disguises itself as a restart |
| Magnitude restarts endlessly (thousands of times) | A manually started dockerd pidfile is in systemd's way; stop the manual process, clear the socket/pid, then hand it back to systemd |
| `pkill -f 'dockerd --config-file ...'` killed ssh | Use the character class `daemo[n].json` so the pattern does not match your own command line |

## 4. OCR / looking at the screen

| Symptom | Cause |
|---|---|
| First 1.2 s all black / noise | MS2109 warm-up frames; you must **drop 40 frames** before judging (noise makes `std ≠ 0`, dropping only a few frames is not enough) |
| One screen look takes 3.4~3.9s | Normal: 1.2s warm-up + OCR ≈ lines × 30ms |
| OCR suddenly 5~10× slower | The `cuda-python` version does not match JetPack → TensorRT silently falls back to CPU, and the log says not a word about it |
| The picture changed but the text did not | `/api/look` caches by fingerprint internally, so the same frame is not recomputed |
| Our own captions leak into the vocabulary | The caption page is on that screen; we now drop a line if it shares an 8-character substring with any of the last 30 captions |

## 5. Waking itself up

The headset's **speaker output is picked up by the microphone** and can trigger the wake word,
creating a feedback loop. Either wear the headset, or turn local monitoring off (「听 HDMI」 /
Listen to HDMI on the page), and do not put the speaker next to the microphone.

## 6. Translation quality

| Symptom | Action |
|---|---|
| Output mixes Chinese and Japanese / Spanish words show up | Switch to 35B-A3B; 7B does this every time with this system prompt |
| 35B output is all reasoning, `content` is an empty string | The Qwen3.5 template enables thinking by default; you must set `enable_thinking=false` |
| Chinese / Japanese language instructions have no effect | The language instruction must be **attached in both places, system and user** (system only: 0/6, user only: 6/6) |
| The model invents references to 「屏幕文字」 (screen text) | Without OCR, use the dynamic system prompt that never mentions the word |
| Half a sentence translated as a full sentence / a subject added | Segmentation is off; run `wav_seg_ab.py` first and look at ASR text completeness |
| Only half of the translation appears | Check whether the SSE stream is a growing prefix; `tools/sse_regress.py` is built to catch 0 regressions / 0 blanks |

## 7. A long sentence pushes the previous line away

Early on we used a full `innerHTML` rewrite plus pixel `scrollY` backfill, so the anchor drifted while
a long translation grew character by character and the same y ended up pointing at different content.
Now it is **persistent DOM with incremental updates**, `#main{overflow-anchor:none}`.
In reverse order (newest on top) the scroll-compensation sign is `+`; we got it backwards once, and
`node tools/dom_render_test.js` covers it.

## 8. Do not chase these as bugs

- Only a rising `q_tr` means it really cannot keep up; a single high `lag_ms` may just be the tail of a segmentation
- Segmentation quality: **look only at whether the ASR text forms sentences**, not at the "share of silence cuts"
- 200ms to first token is the hard floor; `TTFT ≈ 65ms + 4.5ms × new tokens`
- For short prefixes `cache_n` stays 0 — not a broken cache, the user content changes every sentence
- vLLM cannot hit the 35B prefix cache (1056-token granularity)
- The capture card once dropped out on a docking station; for long runs plug it straight into the device's USB
- `hits=0` for screen-term hints usually means the screen has nothing to do with the meeting, not that the feature is broken

## The whole picture in one line

```bash
bash scripts/healthcheck.sh
```

It reports, in order, the services, the ports, the devices, the models, and
`q_asr / q_tr / dropped / ttft / screen`. **Run it first, then read the logs.**
