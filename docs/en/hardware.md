# Hardware: what to buy, how to wire it, and the three fixes to do first

[**English**](hardware.md) · [中文](../zh/hardware.md) · [日本語](../ja/hardware.md)

The whole rig comes down to four parts, and together they cost less than a laptop (not counting the Orin itself). But **if you skip three of the fixes below, every later round of software tuning is wasted** — they raise no error; they only skew your VAD threshold, garble ASR segmentation, and black out the capture card.

## 1. Bill of materials

| Item | Model / spec | Role | Notes |
|---|---|---|---|
| Compute unit | Jetson AGX Orin 64GB Developer Kit | Runs ASR / OCR / TTS / 35B-A3B | The 32GB version can run 35B, but it competes with the system for memory |
| HDMI capture card | MacroSilicon **MS2109** (`534d:2109`) | Turns the demo machine's picture into a UVC camera | Driver-free, a few dozen yuan; stable at 1080p30 |
| USB headset | Logitech **H340** | Capture + local monitoring | Mono mic + 3.5mm headphone jack; one USB cable covers both |
| External SSD | Any USB3 SSD, ≥128G | Holds the model (3.4G) + 35B weights (22.9G) | The root disk usually has only a few G left; don't force it in |
| HDMI cable ×2 / splitter | — | Demo machine → capture card → projector | Only with a splitter can you "project and capture at the same time" |

System: **JetPack 7.2.1 / Ubuntu 24.04 / Python 3.12**. JetPack 6.x (22.04 / Python 3.10) works as well, but the wheels for `pyopenjtalk-plus` and `tensorrt` have to be swapped for other versions; see [models.md](models.md).

## 2. Wiring

```
                 ┌────────────────────┐
   demo PC HDMI ─┤ HDMI splitter      ├─→ projector / screen
                 └──────────┬─────────┘
                            │ HDMI
                 ┌──────────┴─────────┐   USB   ┌──────────────────────┐
                 │ MS2109 capture card├────────→│                      │
                 └────────────────────┘         │  AGX Orin 64GB       │
                 ┌────────────────────┐   USB   │                      │
                 │ H340 headset       ├────────→│  :8095 assistant     │
                 └────────────────────┘         │  :8096 captions      │
                                                └──────────────────────┘
```

- Plug the capture card **straight into the body's USB ports**, not through a dock. We hit this once: the dock could not supply enough power, and after a few hours of running `/dev/video0` dropped off — `v4l2` still listed it, but every frame read back was black.
- The H340's 3.5mm jack takes headphones for monitoring. With no headphones plugged in, **the speaker output is picked up by its own microphone**, and the device can wake itself up (see [troubleshooting.md](troubleshooting.md)).

## 3. The three things you must do first

### 3.1 60Hz mains hum (skip it and every VAD threshold is wrong)

Measured noise floor **−30.6 dBFS**, the strongest spectral peak at 60Hz with a 120Hz harmonic, and 0–80Hz holds 38% of the energy. This is not room noise; it is mains hum / a ground loop. After a 150Hz 4th-order high-pass filter:

| | Before high-pass | After high-pass |
|---|---|---|
| Noise floor | −30.6 dBFS | **−48.8 ~ −56.6 dBFS** |
| Speech SNR | 9.8 dB | **34 dB** |

Speech sits mainly in 300Hz–3kHz, so cutting everything below 150Hz costs ASR nothing. The code applies the high-pass by default (`highpass()` in `voice_probe.py`), but **if you write your own capture code you must add it yourself**.

### 3.2 Capture gain (the H340 ships at full scale)

The H340's capture gain ships at `44/44 = +8dB`, and we measured 33 clipped samples.

```bash
amixer -c 0 cset numid=4 32     # 44 → 32, i.e. +8dB → 0dB
amixer -c 0 cset numid=4 44     # restore the original value
```

Trap: this control has **no** `Playback/Capture` prefix, so `amixer sget 'Mic Capture Volume'` reports that it cannot find it; you must use `numid`. Confirm first with `amixer -c 0 contents | grep -A2 numid=4`.

### 3.3 Capture card warm-up black frames (skip it and the first 1.2 seconds are black)

Every time the MS2109 is `open()`ed, it outputs **all-black or noisy frames** for a while. Dropping a few frames is not enough: noisy frames have a non-zero pixel standard deviation, so "is std zero → is it all black" misclassifies them as real picture.

The right way (already implemented in `vision.py`): **drop 40 consecutive frames** before you start judging. At 1080p30 that is about 1.3 seconds. So a "take a look at the screen" call always carries about 1.2 seconds of warm-up — **keeping the capture resident** (paying the warm-up cost once) was one of the main optimizations of 2026-11-21, see [engineering-log.md](engineering-log.md).

## 4. Audio path (PipeWire)

Both capture and playback go through PipeWire, for a practical reason: **PipeWire lets several clients read the same source at the same time**, whereas `arecord` connected directly reports "device busy". So:

```bash
# capture (non-exclusive)
pw-record --target <source> --rate 16000 --channels 1 - > /dev/null

# playback (pin the output, or the sound runs off to HDMI)
pw-play --target <sink> out.wav
```

To see which nodes exist right now:

```bash
pactl list short sources    # capture: look for USB3.0 Capture / H340
pactl list short sinks      # playback: look for the H340's analog-stereo
```

Node names are long (`alsa_output.usb-...H340-00.analog-stereo`); pass them into the service with `--local-sink`. **Trap we hit**: without `--target`, PipeWire sends the sound to the default sink, and the default sink may be HDMI — so "playback is silent" when the sound actually went to the monitor.

> When systemd starts the service, `pw-play` / `pw-record` need `XDG_RUNTIME_DIR=/run/user/<uid>`, otherwise they **fail silently** (no error, just no sound). The unit already sets it; don't delete it.
