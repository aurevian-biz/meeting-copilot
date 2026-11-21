# テスト：マイクに頼らない層別リグレッション

[English](../en/testing.md) · [中文](../zh/testing.md) · [**日本語**](testing.md)

断句パラメータを 1 つ変える、モデルを差し替える、フロントエンドに手を入れる——どれも
別の層を壊し得ます。このテスト群は層ごとに分かれていて、**すべてオフラインで実行できます**
（ASR / LLM / OCR はスタブ）。モデルを起動しておく必要も、マイクに向かって話す必要もありません。

## まとめて実行する

```bash
cd /opt/meeting-copilot

# データ層（スクリプト形式、ASR/翻訳はスタブ、16 グループ）
.venv/bin/python test_caption_pipe.py

# 残り 5 本の unittest ファイル
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py

# 描画層（26 項目、node が必要）
node tools/dom_render_test.js

# データ層の自己チェック：訳文の後戻り 0・空化 0（**オンライン**、:8096 に 90 秒接続して購読）
.venv/bin/python tools/sse_regress.py
```

> `test_caption_pipe.py` は**スクリプト**です。スタブは `__main__` の中で組み込まれるので、
> pytest で単体実行しないでください（スタブが入らず誤検知になります）。残りの 5 本は標準の
> `unittest` なので、pytest でも `python -m unittest` でも構いません。
>
> `tools/sse_regress.py` は**オフラインテストではありません**。稼働中の 8096 に接続し、実ストリームを
> 90 秒購読して、訳文の後戻りや空化がないかを確認します。実行する前に
> `systemctl start caption-bench` を済ませてください。

## 各層が担保するもの

| 層 | ファイル | 担保する内容 |
|---|---|---|
| 字幕データ | `test_caption_pipe.py` | 二段階表示（原文が訳文より先に出る）、キューの分離、上文の引き継ぎ、短い段の破棄、ソフトカット、段階的ポーズ、下書き、画面クリア、`--tr-prompt` の 2 段階、画面語彙のヒット、**ページ JS の構文** |
| 描画層 | `tools/dom_render_test.js` | 26 項目：ノードの同一性、並び順（最新が最上部）、スクロール補正、逆順挿入、下書き行の位置 |
| データ自己チェック | `tools/sse_regress.py` | SSE ストリーム内の訳文が単調に伸びる前方一致であること。**後戻り 0・空化 0** |
| 割り込み | `test_barge.py` | 読み上げ中の発話で停止する、回り込みでは誤発火しない |
| TTS ルーティング | `test_tts_routing.py` | 日中音声の自動切り替え、中国語応答内の数字を日本語音声で読まない |
| 視覚 | `test_vision.py` | ウォームアップの黒フレーム、OCR のマスキング（`sk-***`）、TensorRT への接続 |
| 音声フロントエンド | `test_voice_probe.py` | 150Hz ハイパス、VAD、分割 |
| ウェイク | `test_wake.py` | KWS の発火とデバウンス |

### `test_page_js_syntax` がある理由

2026-11-21 に事故がありました。ページにツールチップを追加する際に `'\n'` と書いたのですが、
当時 `PAGE` は raw 文字列ではなかったため、Python が JS 側の改行エスケープを本物の改行に変換し、
シングルクォート文字列が途中で切れました → **`<script>` 全体が構文エラーになり、ページは
「连接中…」で止まったまま、ボタンはすべて無反応、それでもサーバー側はまったく正常**という状態です。

このテストは描画後の `<script>` を抜き出して `node --check` に渡し、さらに「シングルクォートは
各行で必ず対になる」という保険の検査も行います（node のない環境ではスキップ）。CI でも実行します。
**`PAGE` を変更したら、これを必ず実行してください。実行していなければ、変更していないのと同じです。**

## A/B とプローブ（`tools/`）

これらはリグレッションテストではなく、「入れ替える前に測る」ためのツールです。必要に応じて個別に実行します。

| ツール | 用途 |
|---|---|
| `wav_seg_ab.py` | 2 組の断句パラメータを同じ wav 上で直接比較する |
| `voice_llm_ab.py` | LLM を差し替えるときに判定指標で採点する（字幕 / アシスタントの 2 系統） |
| `ttft_prompt_ab.py` | prompt 段階 × プレフィックスキャッシュ × 新規トークン別に、最初の文字が出るまでを測る |
| `ttft_scenarios.py` | 文の長さ / 上文行数ごとの、最初の文字が出るまでのシナリオ |
| `screen_hint_ab.py` / `screen_terms_probe.py` | 画面語ヒントが割に合うか / ASR の誤字に当たるか |
| `trans_stream_probe.py` / `stream_verify.py` | ストリーミングか一括か、訳文が前方一致で伸びるか |
| `sse_verify.py` / `sse_multi_client.py` | SSE の push 間隔、2 画面を同時に開いてもパッチを奪い合わないか |
| `bench_sensevoice.py` / `trt_rec_probe.py` | ASR と TensorRT バックエンドのベンチマーク |
| `check_env.py` | インストール後の自己診断：依存 / モデル / デバイス / CPU への静かなフォールバックがないか |

## モデル／パラメータを変えるときの受け入れチェック

1. `.venv/bin/python test_caption_pipe.py` → `ALL PASS`
2. `.venv/bin/python tools/wav_seg_ab.py --wav <62 秒の実音声>` → 断句の完全性（**ASR テキストが文になっているかだけを見る**。無音カットの割合は見ない）
3. `voice_llm_ab.py` → 言語の純度、ハルシネーション、長い応答の所要時間
4. デプロイ後に `bash scripts/healthcheck.sh` → `q_asr=0 q_tr=0 dropped=0`
5. フロントエンドの変更 → サービス再起動 + **F5** + `test_page_js_syntax`

## CI

`.github/workflows/ci.yml` は、ハードウェアに依存しない 3 つのことだけを行います。

1. `python -m compileall -q .`（リポジトリ全体の構文）
2. `caption_server.py` から `<script>` を抜き出して `node --check`
3. リポジトリ全体を走査し、鍵 / ローカルパス / プライベート IP がないことを確認

実機が要るもの（ASR、OCR、TensorRT、GPU）は実機でしか実行できません。CI でカバーできると
期待しないでください。
