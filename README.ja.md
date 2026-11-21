# meeting-copilot

**1 台のエッジデバイスで完結する、日中バイリンガル会議字幕 ＋ 音声アシスタント。**

[English](README.md) · [中文](README.zh.md) · [**日本語**](README.ja.md) · [License: MIT](LICENSE)

クラウド API は一切使いません。音声認識・翻訳・音声合成・画面 OCR・LLM 推論のすべてを
ローカルで動かします。会議の音声も画面も外に出ないので、社内会議や機微な内容にも使えます。

```
:8095  音声アシスタント   ウェイクワード → ASR → LLM → TTS → ローカル再生
:8096  会議字幕           HDMI 音声 → 分割 → ASR → ストリーミング翻訳 → SSE → ブラウザ
:10100 LLM               Qwen3.5-35B-A3B（Magnitude）／ 任意の OpenAI 互換エンドポイント
```

## できること

- **日本語で話すと中国語の字幕が流れる**（逆方向も可）。訳文は 1 トークンずつ逐次表示、
  最初の 1 文字は約 400〜470ms、知覚レイテンシは約 290ms
- **ウェイクワード「小八你好」**で常駐起動。発話中に割り込むと即座に停止（barge-in）
- **「画面を見て」と頼むと HDMI キャプチャを OCR**。画面上の固有名詞を使って
  ASR の同音異字を訂正する仕組み付き（当たった文だけ +65ms、外れれば 0 トークン）
- **ブラウザは表示だけ**。録音も再生もデバイス側なので、PC のマイク性能に左右されません
- **2 つの画面を同時に開いても壊れない** SSE 設計、最新行が上に来る増分 DOM レンダリング
- 100% オフライン。モデルはすべてローカル、API キーは自前の推論サーバー用だけ

## クイックスタート

JetPack を焼いた Jetson AGX Orin（64GB）に、HDMI キャプチャカードと USB ヘッドセットを
挿してから：

```bash
git clone https://github.com/<you>/meeting-copilot.git /opt/meeting-copilot
cd /opt/meeting-copilot
export MC_DATA=/media/ssd/mc-data          # モデル・重みは外付け SSD 推奨
sudo -E bash scripts/install.sh             # 依存 + venv + モデル 3.4G

bash scripts/llm-magnitude.sh               # 35B-A3B 用の dockerd 設定
sudo -E bash deploy/install-units.sh        # systemd unit と /etc/meeting-copilot/env
sudo systemctl start magnitude              # コールドスタート約 3 分
sudo systemctl start voice-bench caption-bench

bash scripts/healthcheck.sh
```

ブラウザで `http://<デバイスIP>:8095`（音声）/ `:8096`（字幕）を開きます。
**最初に [hardware.ja.md](docs/hardware.ja.md) の 3 項目（60Hz ハム・ゲイン・ウォームアップフレーム）
を必ず確認してください** — ここを怠ると以降のチューニングが全部無効になります。

詳しい手順は [install.ja.md](docs/install.ja.md)、詰まったら
[troubleshooting.ja.md](docs/troubleshooting.ja.md)。

## ハードウェア

| パーツ | 例 | 用途 |
|---|---|---|
| エッジボード | Jetson AGX Orin 64GB（JetPack 7.2 / Ubuntu 24.04） | 全推論 |
| HDMI キャプチャ | MacroSilicon MS2109（`534d:2109`） | 画面 OCR ＋ HDMI 音声 |
| USB ヘッドセット | Logitech H340 | 集音 ＋ ローカルモニタ |
| 外付け SSD | 128GB 以上 | モデル 3.4G ＋ 35B の重み 22.9G |

Orin 以外でも動きます（x86 + NVIDIA GPU でも可）。ただし CUDA 13 系では
onnxruntime-gpu が使えないため、OCR の高速化は TensorRT 経由になります
（[ocr-gpu.ja.md](docs/ocr-gpu.ja.md)）。

## 構成

```
HDMI ─→ キャプチャカード ─┬─→ vision.py（常駐フレーム取得 / OCR / TensorRT rec）─→ /api/look
                          └─→ 音声 ─→ Segmenter ─→ SenseVoice ─→ ストリーミング翻訳 ─→ SSE
USB マイク ─→ KWS ─→ ASR ─→ LLM ─→ 文単位 TTS ─→ pw-play ─→ ヘッドセット
                              │
                    Qwen3.5-35B-A3B（Magnitude :10100）※ vLLM 7B にも差し替え可
```

重要な設計判断：

- **字幕と音声アシスタントは別プロセス**。レイテンシ予算もスレッド模型も違うため、
  1 つにまとめると ASR と LLM を取り合います。OCR は `/api/look` 経由で共有
  （`/dev/video0` は排他デバイスなので片側だけが保持）
- **録音は PipeWire**。複数クライアントが同じ source を読めます（`arecord` は「device busy」）
- **訳文は SSE ＋ 増分 DOM**。全量 `innerHTML` ＋ ピクセル `scrollY` 復元は、
  長い文が伸びている最中に錨がズレて前の行が消えるバグを生みました

詳細は [architecture.ja.md](docs/architecture.ja.md)。

## ドキュメント

各ドキュメントは日本語 / 中文 / English の 3 版があります。下表は日本語版へのリンクで、
ファイル名を `.md` / `.zh.md` に変えれば英語版 / 中国語版になります
（例：[hardware.md](docs/hardware.md)、[hardware.zh.md](docs/hardware.zh.md)）。
各ページの先頭に言語切り替えがあります。

| 文書 | 内容 |
|---|---|
| [hardware.ja.md](docs/hardware.ja.md) | 部品・配線・最初に直すべき 3 つの物理問題 |
| [install.ja.md](docs/install.ja.md) | ゼロから起動まで |
| [models.ja.md](docs/models.ja.md) | モデル一覧・選定理由・ライセンス注意 |
| [llm-serving.ja.md](docs/llm-serving.ja.md) | Magnitude 35B vs vLLM 7B、prefill / キャッシュの落とし穴 |
| [ocr-gpu.ja.md](docs/ocr-gpu.ja.md) | TensorRT で OCR を 3.5 倍にする |
| [usage.ja.md](docs/usage.ja.md) | 画面の使い方・全パラメータ・状態の読み方 |
| [latency.ja.md](docs/latency.ja.md) | レイテンシ予算・計測方法・「これ以上追うな」という硬い床 |
| [testing.ja.md](docs/testing.ja.md) | オフライン回帰と A/B ツール |
| [troubleshooting.ja.md](docs/troubleshooting.ja.md) | 静かに効かなくなる系のトラブル集 |
| [engineering-log.ja.md](docs/engineering-log.ja.md) | 全実測データと没案の記録 |

## テスト

マイクもモデル也不要、すべてスタブで動きます：

```bash
.venv/bin/python test_caption_pipe.py        # 16 グループ（ページ JS 構文チェック含む）
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py
node tools/dom_render_test.js                # レンダリング 26 項目
```

※ `tools/sse_regress.py` は**稼働中の 8096 に 90 秒 subscribe する実運用チェック**
（オフラインではありません）。訳文の後戻り・空化が 0 かを実ストリームで確認します。

## 設定とセキュリティ

- ディレクトリは [paths.py](paths.py) に集約。**コード内に絶対パスは書きません**。
  外付け SSD を使うときは `MC_DATA` を 1 つ設定するだけです
- LLM の Bearer トークンなどは `/etc/meeting-copilot/env`（権限 0640）に置きます。
  **リポジトリには秘密情報を含めません**。CI で `mag-*` / ローカルパス / プライベート IP を
  毎回スキャンしています
- 初回起動前に `conf/env.example` をコピーして中身を確認してください
- リポジトリには開発機を特定できる情報（ホスト名・ユーザー名・メールアドレス・
  シリアル番号・LAN アドレス）を一切含めていません。詳細は [SECURITY.md](SECURITY.md)

## ライセンスと注意

コードは **MIT** です（[LICENSE](LICENSE)：Copyright (c) 2026 meeting-copilot contributors）。
ただし**モデル本体は含まれません**。
`scripts/download-models.sh` が上流の公式配布先からダウンロードするだけです。
各モデル（SenseVoice / Piper 各音色 / PP-OCR / Qwen など）はそれぞれ別のライセンスです。
商用利用の前に必ず上流のモデルカードを確認してください。
サードパーティ製コンポーネントとその告知は [NOTICE](NOTICE) にまとめています。

謝辞：[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、
[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)、
[RapidOCR](https://github.com/RapidAI/RapidOCR)、
[Piper](https://github.com/rhasspy/piper)、[piper-voices](https://huggingface.co/rhasspy/piper-voices)、
[Qwen](https://github.com/QwenLM)、[Magnitude](https://magnitude.dev)。
