# Moving the OCR recognition backend to TensorRT (optional, but worth a lot)

[**English**](ocr-gpu.md) · [中文](ocr-gpu.zh.md) · [日本語](ocr-gpu.ja.md)

Only needed on the 「看一眼屏幕」 path. It runs without it too, at the cost of OCR going
from ~1.7s to ~4.1s.

## Why not onnxruntime-gpu

This machine is JetPack 7.2 (L4T r39.2.1) + **CUDA 13.2**, while onnxruntime only ships
**cu12** wheels. `pip install onnxruntime-gpu` succeeds and `get_available_providers()`
even lists `CUDAExecutionProvider`, but creating a session fails with
`No registered plugin EP device found` — the EP's `.so` fails to dlopen. **This road is
dead on CUDA 13**; don't spend time on it.

TensorRT 10.16 ships with the system (`libnvinfer.so.10` + `trtexec` + python `tensorrt`),
so we bypass onnxruntime and run TRT directly.

## Why only rec

rec accounts for 88% of OCR, so it buys the most. det (0.30s) and cls (0.15s) stay as they
are, and det's input size varies with the screen, which makes its profile harder to pin
down.

## Building the engine (about 11~12 minutes)

```bash
python -c "import rapidocr_onnxruntime, pathlib; print(pathlib.Path(rapidocr_onnxruntime.__file__).parent)"
# note this directory; below, <rapidocr> stands for it

trtexec --onnx=<rapidocr>/models/ch_PP-OCRv4_rec_infer.onnx \
        --saveEngine=$MC_DATA/models/rec_fp16.engine --fp16 \
        --minShapes=x:1x3x48x256 --optShapes=x:6x3x48x320 --maxShapes=x:32x3x48x2048 \
        --memPoolSize=workspace:1024 --timingCacheFile=/tmp/trt_cache.bin
```

All three numbers have a reason — don't change them casually:

- `optShapes` set to **320**: measured, 76% of 75 batches are exactly the 320 baseline
  (when there is no extra-wide line in the batch, `max_wh_ratio` takes the default 6.67).
  Setting opt to a small value makes the **common path** fastest; it is not about taking
  the middle value
- `maxShapes` set to **2048**: the first version set 1024/1536 and both failed — when an
  extra-long line (path, URL, long code line) appears on screen, the whole batch gets
  stretched to 1036, 1656. This number must be **measured** before you set it; guessing
  always means a rebuild (12 minutes each time)
- Tactic search for dynamic shapes is slow and the log stays unchanged for a long time —
  **don't assume it has hung**

The engine is tied to the TensorRT version: after changing JetPack or upgrading TRT you
must rebuild it.

## Results (132-line screen on real hardware)

| Backend | rec | OCR total | Mean score | Per-line agreement |
|---|---|---|---|---|
| CPU (onnxruntime) | 3.4s | 4.12s | 0.951 | baseline |
| **TensorRT FP16** | — | **1.17s (3.5×)** | 0.952 | 128/132 |

End to end (voice asks 「屏幕上是什么」, 140 lines): **6.56s → 3.4~3.9s**. The bottleneck is
no longer OCR at this point, but LLM generation (about 2.1s).

## Batches beyond the profile fall back to CPU

rec's batch input width = 48 × the largest width/height ratio in the batch. Every so often
an extra-long line (URL, path) appears on screen and pushes the whole batch outside the
profile. Measured, only **4%** of 75 batches exceed 1536, so we let those few batches fall
back to CPU — 4% of batches taking the slow path beats 100% of batches using a worse
kernel. Watch the fallback ratio at `vision.trt.fallback_pct` in `/api/status`.

## Wiring it in

Just pass the engine path when starting the service; no RapidOCR source changes:

```bash
--vision-trt $MC_DATA/models/rec_fp16.engine
```

Implementation (`trt_rec.py`): only `TextRecognizer.session` is swapped for
`TrtRecSession`; `resize_norm_img`, CTC decoding and the character table all reuse the
original implementation — the two sides differ only in the inference backend, so what you
measure is purely the backend difference.
