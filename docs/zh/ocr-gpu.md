# 把 OCR 的识别后端换成 TensorRT（可选，但很值）

[English](../en/ocr-gpu.md) · [**中文**](ocr-gpu.md) · [日本語](../ja/ocr-gpu.md)

只在「看一眼屏幕」这条链路上需要。不做也能跑，代价是 OCR 从 ~1.7s 变 ~4.1s。

## 为什么不用 onnxruntime-gpu

这台机器是 JetPack 7.2（L4T r39.2.1）+ **CUDA 13.2**，而 onnxruntime 官方只发 **cu12** 轮子。
`pip install onnxruntime-gpu` 能过、`get_available_providers()` 也会列出
`CUDAExecutionProvider`，但建 session 时报 `No registered plugin EP device found`——
EP 的 `.so` dlopen 失败。**这条路在 CUDA 13 上是死的**，别在上面耗时间。

TensorRT 10.16 随系统就有（`libnvinfer.so.10` + `trtexec` + python `tensorrt`），
所以绕过 onnxruntime 直接跑 TRT。

## 为什么只换 rec

rec 占 OCR 的 88%，换它性价比最高。det（0.30s）和 cls（0.15s）先不动，
而且 det 的输入尺寸随画面变化，profile 更难定。

## 建 engine（约 11~12 分钟）

```bash
python -c "import rapidocr_onnxruntime, pathlib; print(pathlib.Path(rapidocr_onnxruntime.__file__).parent)"
# 记下这个目录，下面用 <rapidocr> 表示

trtexec --onnx=<rapidocr>/models/ch_PP-OCRv4_rec_infer.onnx \
        --saveEngine=$MC_DATA/models/rec_fp16.engine --fp16 \
        --minShapes=x:1x3x48x256 --optShapes=x:6x3x48x320 --maxShapes=x:32x3x48x2048 \
        --memPoolSize=workspace:1024 --timingCacheFile=/tmp/trt_cache.bin
```

三个数都是有来由的，别随手改：

- `optShapes` 定 **320**：实测 75 个批次里 76% 就是基准 320（批内没有超宽行时
  `max_wh_ratio` 取默认 6.67）。opt 定小值是让**常见路径**最快，不是取中间值
- `maxShapes` 定 **2048**：第一版设 1024/1536 都失败过——屏幕上出现一行超长
  （路径、URL、长代码行）时，整批会被撑到 1036、1656。这个数必须**量出来**再定，
  拍脑袋必然要重建（每次 12 分钟）
- 动态 shape 的 tactic 搜索很慢，日志会长时间不更新，**别以为卡死**

engine 和 TensorRT 版本绑死：换 JetPack / 升级 TRT 后必须重建。

## 效果（真机 132 行画面）

| 后端 | rec | OCR 合计 | 均分 | 逐行一致 |
|---|---|---|---|---|
| CPU (onnxruntime) | 3.4s | 4.12s | 0.951 | 基准 |
| **TensorRT FP16** | — | **1.17s（3.5×）** | 0.952 | 128/132 |

端到端（语音问「屏幕上是什么」，140 行）：**6.56s → 3.4~3.9s**。此时瓶颈已经不是 OCR，
而是 LLM 生成（约 2.1s）。

## 超 profile 的批次退回 CPU

rec 的批输入宽度 = 48 × 批内最大宽高比。屏幕偶尔出现超长行（URL、路径），
会把整批撑到 profile 之外。实测 75 批里超 1536 的只占 **4%**，所以让这几批退回 CPU
——4% 的批走慢路，比 100% 的批用差 kernel 划算。回退比例看 `/api/status` 的
`vision.trt.fallback_pct`。

## 挂上去

服务启动时传 engine 路径即可，不需要改 RapidOCR 源码：

```bash
--vision-trt $MC_DATA/models/rec_fp16.engine
```

实现（`trt_rec.py`）：只把 `TextRecognizer.session` 换成 `TrtRecSession`，
`resize_norm_img`、CTC 解码、字符表全部复用原实现——两边只差推理后端，
测出来的才是纯后端差异。
