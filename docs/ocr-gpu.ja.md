# OCR の認識バックエンドを TensorRT に替える（任意だが、効果は大きい）

[English](ocr-gpu.md) · [中文](ocr-gpu.zh.md) · [**日本語**](ocr-gpu.ja.md)

「看一眼屏幕」の経路でのみ必要です。入れなくても動きますが、その場合 OCR は約 1.7s から
約 4.1s になります。

## onnxruntime-gpu を使わない理由

このマシンは JetPack 7.2（L4T r39.2.1）+ **CUDA 13.2** ですが、onnxruntime の公式 wheel は
**cu12** までしか出ていません。`pip install onnxruntime-gpu` は通り、
`get_available_providers()` も `CUDAExecutionProvider` を列挙しますが、セッション作成時に
`No registered plugin EP device found` で失敗します —— EP の `.so` の dlopen に失敗して
いるためです。**この道は CUDA 13 では閉ざされています**。ここに時間を費やさないでください。

TensorRT 10.16 はシステムに同梱されています（`libnvinfer.so.10` + `trtexec` + python
`tensorrt`）。そこで onnxruntime を迂回し、TRT を直接動かします。

## rec だけを替える理由

rec は OCR の 88% を占めるので、替える費用対効果が最も高いです。det（0.30s）と cls
（0.15s）はそのままにします。また det の入力サイズは画面によって変わるため、profile を
決めるのがより難しくなります。

## engine のビルド（約 11〜12 分）

```bash
python -c "import rapidocr_onnxruntime, pathlib; print(pathlib.Path(rapidocr_onnxruntime.__file__).parent)"
# このディレクトリを控えておきます。以下では <rapidocr> と表記します

trtexec --onnx=<rapidocr>/models/ch_PP-OCRv4_rec_infer.onnx \
        --saveEngine=$MC_DATA/models/rec_fp16.engine --fp16 \
        --minShapes=x:1x3x48x256 --optShapes=x:6x3x48x320 --maxShapes=x:32x3x48x2048 \
        --memPoolSize=workspace:1024 --timingCacheFile=/tmp/trt_cache.bin
```

3 つの数値にはいずれも根拠があります。気軽に変えないでください：

- `optShapes` を **320** にする：実測では 75 バッチのうち 76% が基準の 320 です（バッチ内に
  極端に横長の行がなければ `max_wh_ratio` は既定の 6.67 になります）。opt を小さい値に
  するのは**よく通る経路**を最速にするためで、中間値を取るためではありません
- `maxShapes` を **2048** にする：最初の版では 1024/1536 にしてどちらも失敗しました ——
  画面に極端に長い行（パス、URL、長いコード行）が 1 行現れると、バッチ全体が 1036、1656
  まで引き伸ばされます。この値は**実測して**決める必要があり、当て推量では必ず再ビルドに
  なります（1 回 12 分）
- 動的 shape の tactic 探索は非常に遅く、ログが長時間更新されません。**ハングしたと
  決めつけないでください**

engine は TensorRT のバージョンに固定されます。JetPack を変える／TRT をアップグレード
したら再ビルドが必要です。

## 効果（実機 132 行の画面）

| バックエンド | rec | OCR 合計 | 平均スコア | 行単位の一致 |
|---|---|---|---|---|
| CPU (onnxruntime) | 3.4s | 4.12s | 0.951 | 基準 |
| **TensorRT FP16** | — | **1.17s（3.5 倍）** | 0.952 | 128/132 |

エンドツーエンド（「屏幕上是什么」と音声で質問、140 行）：**6.56s → 3.4〜3.9s**。この時点で
ボトルネックはもはや OCR ではなく、LLM の生成（約 2.1s）です。

## profile を超えるバッチは CPU に戻す

rec のバッチ入力幅 = 48 × バッチ内の最大アスペクト比です。画面にときどき極端に長い行
（URL、パス）が現れ、バッチ全体が profile の外に押し出されます。実測では 75 バッチのうち
1536 を超えるのは **4%** だけなので、その数バッチだけ CPU に戻します —— 4% のバッチが
遅い道を通るほうが、100% のバッチが劣った kernel を使うより得です。フォールバック率は
`/api/status` の `vision.trt.fallback_pct` で確認できます。

## 組み込み方

サービス起動時に engine のパスを渡すだけです。RapidOCR のソースを変更する必要は
ありません：

```bash
--vision-trt $MC_DATA/models/rec_fp16.engine
```

実装（`trt_rec.py`）：`TextRecognizer.session` を `TrtRecSession` に差し替えるだけで、
`resize_norm_img`、CTC デコード、文字表はすべて元の実装を再利用します —— 両者の違いは
推論バックエンドだけなので、測れるのは純粋なバックエンド差だけです。
