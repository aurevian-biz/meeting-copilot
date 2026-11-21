# 大規模言語モデル：2 つの路線と、知っておくべきエンジンの詳細 3 点

[English](llm-serving.md) · [中文](llm-serving.zh.md) · [**日本語**](llm-serving.ja.md)

2 つのサービス（`voice_server.py` / `caption_server.py`）が扱うエンドポイントは 1 種類だけです：
**OpenAI 互換の `/chat/completions`**。したがってバックエンドを替えても業務
コードは変更不要で、`--llm-base` / `--llm-model` / `--llm-key` の 3 つを差し替えるだけです。

## 1. 2 つの路線

| | Magnitude + Qwen3.5-35B-A3B | vLLM + Qwen2.5-7B-AWQ |
|---|---|---|
| エンドポイント | `http://127.0.0.1:10100/inference/v1` | `http://127.0.0.1:8000/v1` |
| モデル id | `hf:unsloth/Qwen3.5-35B-A3B-GGUF/...Q4_K_XL.gguf` | `qwen` |
| 認証 | Bearer（Magnitude 自身が生成、後述） | なし |
| 重み | 22.9G GGUF | 約 5.2G AWQ |
| 最初のトークン（短文） | 300〜470ms | **90ms** |
| 長い回答（画面読み取り／要約） | **25〜35% 速い** | 遅い |
| 日中混在／ハルシネーション | きれい | 「会議的时间确认 needed。」、`-topic1-`、スペイン語の語が出る |
| コールドスタート | 約 3 分（重みの読み込み 150s を含む） | 約 40 秒 |

**結論**：このシステムは「日本語で話し、中国語で読み上げ・字幕表示する」ものなので、
言語指示は一般的な場面より厳しくなります。35B を導入してから既知の翻訳失敗 7 件はすべて
消えました。代償は短文での最初のトークンが 170〜370ms 遅いことです。メモリが厳しい場合や
短文の字幕だけであれば 7B でも構いません（`scripts/llm-vllm.sh` 参照）。

### なぜ MoE は長い回答でむしろ速いのか

35B-A3B は MoE で、トークンあたり約 3B のパラメータしか活性化しません。dense な 7B は
トークンごとに全パラメータを通します。コストは **prefill**（ほぼすべてのエキスパートを
読む必要がある）にあり、利得は**デコード**にあります。したがって短文（数十文字）は 7B、
長文（数百文字）は 35B が 25〜35% 勝ちます。

## 2. 起動とパラメータ

### Magnitude

```bash
bash scripts/llm-magnitude.sh                 # 専用 dockerd 用の daemon.json を生成
sudo bash deploy/install-units.sh             # magnitude / magnitude-dockerd をインストール
sudo systemctl start magnitude                # コンテナ起動 → health 待ち → モデルを明示的に load
```

Magnitude の apiKey は**初回起動時に自身で生成**されるもので、こちらが設定するものでは
ありません：

```bash
python3 -c "import json; print(json.load(open('$MC_DATA/magnitude-data/.magnitude/config.json'))['network']['apiKey'])"
```

`deploy/install-units.sh` はこれをその場で読み取り、`/etc/meeting-copilot/env`（0640）に
書き込みます。**このキーをリポジトリにコミットしないでください**。

dockerd を分けている理由：Magnitude は `--runtime nvidia` を必要とし、22.9G の data-root を
占有します。デーモンを分ければ「これを止めてもマシン上の他のコンテナに影響しない」ように
できます。この構成が不要なら、`deploy/magnitude-up.sh` の `DOCKER_HOST` を外してください。

### vLLM

```bash
bash scripts/llm-vllm.sh
# その後 /etc/meeting-copilot/env を次のように変更します：
#   LLM_BASE=http://127.0.0.1:8000/v1
#   LLM_MODEL=qwen
#   LLM_KEY=
```

### 業務側のパラメータ

```bash
--llm-base   http://127.0.0.1:10100/inference/v1
--llm-model  hf:unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
--llm-key    <Bearer>
```

## 3. 知っておくべきエンジンの詳細 3 点

### 3.1 `prompt_n` は「新規トークン − キャッシュ」ではない

エンジンが返す `prompt_n` は**今回実際に処理したトークン数**、`cache_n` は再利用した
トークン数で、両者を足したものがプロンプト全体です。**新規トークンは `prompt_n`** であり、
`prompt_n - cache_n` ではありません（初期に逆に書いていて、傾きの推定が 2 倍ずれました）。

そこから得られる経験式（35B-A3B）：

```
TTFT ≈ 65ms + 3.5~6.5ms × 新規トークン       # 本番実測の傾き 4.5ms/token
```

最初のトークン 200ms はハードフロアです。これ以上追わないでください。

### 3.2 短いプロンプトのほうが遅い

直感的には「system を 70 トークンまで削れば prefill が速くなる」ですが、実測は**逆**でした：

- 長いプレフィックス（full、約 250 トークン）：290 トークンを安定して再利用し、`cache_n` が高い
- 短いプレフィックス（short、約 70 トークン）：ユーザー内容が毎回変わると `cache_n` は**常に 0**（24/24）で、全体を prefill し直すため、最初のトークンはむしろ遅くなる
- 同じプロンプトを 2 回続けて送る：どちらもヒットする（`297/1` 対 `91/1`）→「短いとキャッシュできない」のではなく、**毎回変わる内容がプレフィックスを無効化する**

したがって既定は `--tr-prompt full` です。`short` は対照実験用に残していて、本番用途には
推奨しません。同様に `--ctx-lines 1` で直前の文脈 30 トークンを削るのは本当に効きます
（約 −90ms）。その部分が**毎回変わる**からです。`--tr-workers 1` はレイテンシには差がなく、
VRAM の取り合いが減るだけです。

### 3.3 言語指示は user の末尾に置き、system にも二重に置く

35B は指示の位置に非常に敏感です（日本語会議 → 中国語出力）：

| 指示の位置 | 中国語出力の成功率 |
|---|---|
| system 末尾のみ | **0/6** |
| user 末尾のみ | **6/6** |
| system + user の二重 | 4/4 |

実装は `voice_server.py` の `_lang_directive()` にあります。user 内容の末尾に 1 つ追加し、
system にも 1 つ入れます。字幕側も同様で、`caption_server.py` の `_tr_system()` を参照して
ください。

さらに 35B 固有の落とし穴が 2 つあります：

- Qwen3.5 の chat template は**既定で thinking が有効**です。`max_tokens=200` が思考で
  使い切られ、`content` が空文字列になり、「訳文がない」という症状になります。
  `chat_template_kwargs={"enable_thinking": false}` を渡す必要があります
- **動的な system**：system に「[屏幕文字]」という語が一度でも入ると、モデルはそれを
  引用しようとします。したがって OCR 内容がないときは別の system を使い、「屏幕文字」に
  一切触れないようにします

## 4. バグとして追わないこと

- vLLM は 35B のプレフィックスキャッシュ（1056 トークン粒度）にヒットできません。これは
  **モデル + エンジンの物理特性**です
- 35B は prefill が高コストで decode が低コスト、dense はその逆です。7B の直感を MoE に
  当てはめないでください
- コールドスタート時の最初の `models load` は約 150s かかります。ログが動かないのは
  フリーズではありません
