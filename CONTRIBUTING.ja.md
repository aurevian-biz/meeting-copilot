# コントリビューション

[English](CONTRIBUTING.md) · [中文](CONTRIBUTING.zh.md) · [**日本語**](CONTRIBUTING.ja.md)

issue も pull request も歓迎します。このプロジェクトの価値の半分は「なぜそう作ったか」に
あるので、**デフォルト値を変えるときは実測データを添えてください**。

## push する前に

```bash
python -m compileall -q .
.venv/bin/python test_caption_pipe.py        # 16 グループ（ページ JS 構文チェック含む）
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py
node tools/dom_render_test.js
bash scripts/healthcheck.sh                  # 実機がある場合
```

触った層に対応するテストを追加してください：

| 変更内容 | 必須のテスト |
|---|---|
| セグメンテーション / キュー / 翻訳リクエスト | `test_caption_pipe.py` |
| 字幕ページの JS（`PAGE`） | `test_page_js_syntax` + `tools/dom_render_test.js`、変更後は F5 で再読込 |
| SSE / データ形式 | `tools/sse_regress.py` |
| 音声フロントエンド / VAD | `test_voice_probe.py` |
| OCR / 画面用語ヒント | `test_vision.py` / `test_screen_hints`（`test_caption_pipe.py` 内） |

## 约定事

1. **コードに絶対パスを書かない。** ディレクトリは `paths.py` を経由し、`MC_DATA` で上書きします。
2. **秘密情報をコミットしない。** `conf/env.example` はテンプレートで、実値は
   `/etc/meeting-copilot/env` に置きます。
3. **`PAGE` は `r"""` のまま。** 通常の文字列にすると埋め込み JS の `\n` が Python に
   食われ、`<script>` 全体が構文エラーになります。ページは「连接中…」で止まり、
   サーバー側はいたって正常に見えます。これは一度実際に起きた事故です。
4. **モデルやパラメータを差し替えるときは判定基準を書く** — `q_asr/q_tr/dropped`、
   `ttft_ms`、セグメンテーションの完全性、言語の純度。「速くなった気がする」は基準になりません。
5. **3 言語を同期させる。** すべてのドキュメントと README は `x.md`（英語）/
   `x.zh.md`（中国語）/ `x.ja.md`（日本語）の 3 版があります。1 つ直したら 3 つ直します。
6. **コード内のコメントは英語。** 画面文言とプロンプトは原文のままにします。
   それらは製品の一部であって、ドキュメントではありません。

## メンテナ向けメモ（秘匿情報の除去）

公開リポジトリは、デバッグ用スクリーンショット・実行時生成物・開発機固有のパスや
LAN アドレス・LLM の Bearer トークンを含む unit が混在した非公開ツリーから生成しています。
再生成にはパッケージ用スクリプトを使います（**非公開ツリーにのみ存在**）。

```bash
python tools/package_opensource.py --dest /path/to/meeting-copilot --force
```

実行時に必要なファイルだけを選び、規則に従って機微な文字列を置換し、最後に
「秘密情報なし / ローカルパスなし / プライベート IP なし」を自己検査します。
`docs/` と README は削除しません。CI（`.github/workflows/ci.yml`）でも同じスキャンを
回して、混入を merge 前に止めています。
