# モデル一覧：何を入れたか、なぜそれか、差し替えられるか

[English](../en/models.md) · [中文](../zh/models.md) · [**日本語**](models.md)

すべてオフライン、合計 3.4G です。表の「代替案」は**実際に試して却下した**もので、
検討していないわけではありません。

## 1. 一覧

| 用途 | モデル | サイズ | 取得元 | レイテンシ（Orin） |
|---|---|---|---|---|
| ASR | **SenseVoice-Small**（sherpa-onnx 同梱、int8） | 1.1G（int8 の重みは 239M のみ） | [k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | RTF 0.04、3.3 秒の発話 1 本で約 290ms |
| 発話終端 VAD | **Silero VAD** | 636K | 同上 | 1 フレームあたりマイクロ秒単位 |
| ウェイクワード | **KWS zipformer 3.3M**（wenetspeech） | 36M | [kws-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) | RTF 0.03、CPU のみ |
| TTS 日本語 | **Piper `ja_JP-hi_fi_captain-medium`** | 74M | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | 最初のチャンク 636ms、RTF 0.135 |
| TTS 中国語 | **Piper `zh_CN-huayan-medium`** | 61M | 同上 | RTF 0.08〜0.19 |
| OCR | **RapidOCR（PP-OCRv4 det/cls/rec）** | 16M | pip パッケージに同梱 | CPU 4.1s/140 行、TensorRT rec 1.7s |
| LLM | **Qwen3.5-35B-A3B**（UD-Q4_K_XL GGUF） | 22.9G | Magnitude が取得 | 最初のトークン 300〜470ms、[llm-serving.md](llm-serving.md) 参照 |

`scripts/download-models.sh` が最初の 6 つを 1 行で取得します。LLM は推論ランタイムが
自分で取得します。

## 2. なぜこれらなのか

### ASR：Whisper ではなく SenseVoice

同じ日本語（3.29 秒）で SenseVoice と faster-whisper を比べました：

| | SenseVoice | Whisper large-v3-turbo |
|---|---|---|
| 中国語の場面 | 正確 | 日本語を中国語として聞き取る（**言語判定そのものを誤る**） |
| レイテンシ | RTF 0.04 | 1 桁遅い |
| 日・中・英・韓・広東語 | 1 モデルで全部 | large でようやく |

**致命的なのは速度ではなく言語判定**です。会議は日本語なのに、Whisper は短い音声区間を
しばしば中国語と判定し、後段の翻訳がどんなに良くても救えません。そこで `--lang auto` は
SenseVoice に任せています。

### TTS：VOICEVOX / MeloTTS ではなく Piper

Piper は aarch64 で「pip を入れたらそのまま動く」唯一の選択肢です。CPU で動き、
RTF 0.135、最初のチャンク 636ms。代償は日本語の声が**1 種類だけ**（`hi_fi_captain`、
話者 2 人）であることです。音質を上げたければ VOICEVOX（GPU/HTTP サービスが必要）に
替えるか、MeloTTS を直す（aarch64 でのビルドが面倒）しかありません。

導入時の落とし穴が 3 つ（いずれも踏みました）：

1. `piper-phonemize` は aarch64 用の **wheel がありません** —— ただし `piper-tts>=1.8.0` はすでにこれに依存していません
2. 日本語には `pyopenjtalk` が必要ですが、公式パッケージはソースのみ —— **`pyopenjtalk-plus`** を使ってください（cp312 の aarch64 wheel があり、モジュール名は `pyopenjtalk` のままです）
3. CLI で wav ヘッダーを書くと `# channels not specified` で失敗します —— Python API の `PiperVoice.synthesize_wav()` を使ってください

### ウェイクワード：ASR を常時起動せず KWS zipformer

ASR を常駐させると CPU/GPU を常に食い、テレビの音や咳まで発話として扱います。KWS モデルは
3.3M しかなく、CPU のみで RTF 0.03、「小八你好」に反応したときだけ ASR を起動します。

ウェイクワードの変更：KWS は**ピンイン／音素**単位なので、語を変えるには `keywords.txt` を
用意し直す必要があります（sherpa-onnx の公式ドキュメント参照）。1 行の文字列を書き換える
だけでは済みません。

### OCR：PaddleOCR ではなく RapidOCR

PaddleOCR は aarch64 + CUDA 13 ではインストールできません（paddlepaddle に対応する wheel が
ありません）。RapidOCR は同じ PP-OCRv4 モデルを onnxruntime で包んだもので、pip を入れたら
そのまま動きます。

## 3. モデルを替える前に測るべき 3 つの数値

ASR / LLM / TTS を替えるときも感覚で決めず、まず測ってください：

| 指標 | どこで見るか | 何なら「許容」か |
|---|---|---|
| 最初のトークンまでの遅延 ttft | `/api/status` の `ttft_ms`、または画面の「首字」 | 字幕 < 600ms、アシスタント < 1s |
| 1 発話の合計時間 | `trans_ms` / `asr_ms` | 字幕 < 発話長の 1.5 倍。超えると滞留します（`q_tr` が上がる） |
| 文分割の完全性 | ASR テキストが文になっているか | 半文率 < 15%。超えると訳文は必ず途中で切れる |

すぐ使える A/B ツール（すべてスタブで動くのでマイクは不要です）：

```bash
.venv/bin/python tools/wav_seg_ab.py        --wav x.wav --a 0.45 --b 0.7
.venv/bin/python tools/voice_llm_ab.py      --base http://127.0.0.1:10100/inference/v1
.venv/bin/python tools/ttft_prompt_ab.py    --prompt short --prompt full
```

## 4. ライセンス

**このリポジトリのコードは MIT ですが、モデルは違います**。各モデルには独自のライセンスが
あるので、商用利用の前に上流を確認してください：

- SenseVoice / PP-OCR / Silero / Piper の各音声：各リポジトリまたは HuggingFace のモデルカードの License 欄を参照
  （Piper の音声は出所がそれぞれ異なるので、**1 つずつ確認**してください。全部 CC0 と決めつけないこと）
- Qwen2.5 / Qwen3.5：Alibaba 公式のモデルカードを参照
- Magnitude ランタイム：magnitude.dev を参照

当リポジトリは重みを一切同梱していません。`scripts/download-models.sh` は上流の公式 URL
からダウンロードするだけです。「モデルも寛容なライセンスでなければならない」という要件が
ある場合は、Apache-2.0 の重みを優先してください。
