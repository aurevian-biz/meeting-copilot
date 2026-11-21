# Latency: the budget, how to measure it, and what is a hard floor

[**English**](latency.md) · [中文](latency.zh.md) · [日本語](latency.ja.md)

There is little tuning headroom in this system, because **most of the time is spent in physics**: ASR has to hear the end of the utterance, MoE has to do prefill, the GPU has to do OCR. This document separates "what can be saved" from "what to stop chasing".

## Latency budget of one caption (measured in production)

```
speech ends
  ├─ segmentation tail (tiered pause / soft-cut)   0.30~0.90s   ← set by speaking rhythm, not waste
  ├─ SenseVoice ASR                                ~290ms
  ├─ LLM prefill → first token                     400~470ms   ← ttft_ms
  ├─ streaming translation, token by token         ~80ms/increment
  └─ whole sentence translated                     ~1000ms      ← trans_ms

perceived latency lag_ms ≈ 290ms (from audio to the first line of text)
```

**Key point**: display starts as soon as the first token arrives, so the user never waits for `trans_ms`. That is why you must never turn on `--no-stream-trans`: showing only the finished sentence pushes perceived latency from 0.26s to 0.8~1.3s.

## Rule of thumb for the first token (35B-A3B)

```
TTFT ≈ 65ms + 3.5~6.5ms × new tokens        measured on-line slope 4.5ms/token
```

"New tokens" is the `prompt_n` returned by the engine (**the tokens actually processed this round**), not `prompt_n - cache_n`; `cache_n` is the reused part, and the two together make up the whole prompt. **A 200ms first token is a hard floor**; stop pushing below it.

This yields three counter-intuitive conclusions:

| Move | Result | Why |
|---|---|---|
| Cut the system prompt from 250 tokens to 70 | **Slower** | User content changes every sentence, so the short prefix has `cache_n` = 0 (24/24) and you pay prefill for the whole thing |
| `--ctx-lines 2 → 1` | **Saves about 90ms** | The context part changes every sentence, so it is genuinely new tokens |
| `--tr-workers 2 → 1` | No difference | Single-sentence latency doesn't depend on concurrency, you just fight less over VRAM |

**A long prefix is free**: a stable long system prompt is reused from cache (about 290 tokens), so cutting it actually breaks the reuse.

## 7B vs 35B (same 62-second clip, head-to-head on long sentences)

| | 7B-AWQ (dense, vLLM) | 35B-A3B (MoE, Magnitude) |
|---|---|---|
| Short-sentence first token | **90ms** | 300~470ms |
| Long answer, full text | slow | **25~35% faster** |
| Japanese-Chinese mixed text / hallucinations | makes mistakes | clean |

MoE activates only about 3B per token, so decoding is cheap; but prefill has to read nearly all the experts, so it is expensive. **Dense wins short answers, MoE wins long answers** — don't apply 7B intuition to 35B.

## Decision criteria (look at only these three numbers)

```bash
curl -s localhost:8096/api/status
```

| Field | Healthy | What to do first when it isn't |
|---|---|---|
| `q_asr` | 0 | rising → not enough ASR threads / capture is taken |
| `q_tr` | 0 | **rising = translation can't keep up**: try `--ctx-lines 1` first, then check whether the LLM is fighting for VRAM |
| `dropped` | 0 | >0 → the queue has started dropping segments; latency is already past real time and must be dealt with |

Read `NRestarts` together with `ActiveEnterTimestamp` to tell whether it is stuck in an endless restart loop.

## How to measure your own machine

```bash
.venv/bin/python tools/ttft_prompt_ab.py     # prompt variants × prefix cache
.venv/bin/python tools/ttft_scenarios.py     # first token across sentence lengths / context lines
.venv/bin/python tools/trans_stream_probe.py # streaming vs whole sentence
.venv/bin/python tools/sse_verify.py 45      # number and spacing of SSE intermediate states
.venv/bin/python tools/e2e_latency.py        # end-to-end
.venv/bin/python tools/wav_seg_ab.py --wav x.wav --a 0.45 --b 0.7
```

## Don't investigate these as bugs

| Symptom | Truth |
|---|---|
| One screen look takes 3.4~3.9s | The capture card drops 40 warm-up frames (about 1.2s) + OCR ≈ lines × 30ms |
| Frontend change doesn't show up | `PAGE` is a constant: restart the service + **user presses F5** |
| `cache_n` is always 0 | User content changes every sentence, so the prefix is invalidated; the cache is not broken |
| vLLM can't hit the 35B prefix cache | 1056-token granularity, a physical property of the model + engine |
| First token won't get below 200ms | 65ms fixed overhead + prefill; it is a hard floor |
