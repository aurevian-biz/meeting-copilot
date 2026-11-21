# ゼロからインストールする

[English](install.md) · [中文](install.zh.md) · [**日本語**](install.ja.md)

JetPack を書き込んだ AGX Orin がすでに手元にあり、ssh で入れて、キャプチャカードとヘッドセットを接続済みであることを前提とします。推論にネットワークは一切不要です。**ASR / TTS / OCR / LLM はすべてローカル**で動くため、インストール後はオフラインでも使えます。

## 0. 事前チェック

```bash
python3 -V                    # 3.10 以上（JetPack 7 は 3.12）
nvidia-smi | head -5          # ドライバ / CUDA が正常
pactl list short sources      # キャプチャカードとヘッドセットが見える
ls /dev/video*                # /dev/video0 がある
df -h /                       # ルートディスクは 5G 以上空いているのが望ましい
```

## 1. コード取得 + 依存関係のインストール

```bash
git clone https://github.com/<you>/meeting-copilot.git /opt/meeting-copilot
cd /opt/meeting-copilot

# データ（3.4G のモデル + 後で 22.9G の重み）は外付けディスクに置くことを推奨
export MC_DATA=/media/ssd/mc-data

sudo -E bash scripts/install.sh
```

`install.sh` は 4 つのことをします：apt で PipeWire / ビルドツールを入れる → `.venv` を作る → `pip install -r requirements.txt` → `scripts/download-models.sh` を呼んでモデルを取得 → 最後に `tools/check_env.py` で自己診断する。

**TensorRT / cuda-python はインストールしません**。この 2 つは JetPack の CUDA と厳密に一致させる必要があり、pip で誤ったバージョンを入れると**静かに** CPU にフォールバックします（OCR が 300ms から 3s になり、ログには何も出ません）。OCR を GPU で高速化するには、まず `python -c "import tensorrt, cuda"` が通ることを確認し、そのうえで [ocr-gpu.md](ocr-gpu.ja.md) の手順に従って engine を生成してください。

手動で入れることもできます：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download-models.sh
```

## 2. 大規模モデルを起動する

2 通りの方法があります。**まずどちらか一方を選びます**：

| | Magnitude + Qwen3.5-35B-A3B | vLLM + Qwen2.5-7B-AWQ |
|---|---|---|
| VRAM / メモリ | 重み 22.9G、常駐約 26G | 約 6G |
| 最初のトークン | 300〜470ms（短文） | **90ms** |
| 長い回答 | **25〜35% 高速** | 遅い |
| 中国語・日本語の混在 / ハルシネーション | きれい | 「会議的时间确认 needed。」が出る |
| サービスの起動 | コールドスタート約 3 分 | 約 40 秒 |

```bash
# A. 35B-A3B（推奨：品質優先。字幕もアシスタントもこれを使う）
bash scripts/llm-magnitude.sh            # dockerd の設定を生成
sudo bash deploy/install-units.sh        # magnitude / magnitude-dockerd の unit をインストール
sudo systemctl start magnitude           # コールドスタート約 3 分（重み 22.9G）

# B. 7B（レイテンシ優先 / メモリが厳しい場合）
bash scripts/llm-vllm.sh
# そのうえで /etc/meeting-copilot/env を次のように変更します：
#   LLM_BASE=http://127.0.0.1:8000/v1
#   LLM_MODEL=qwen
#   LLM_KEY=
```

詳細（35B の MoE のほうがかえって割安な理由、prefix cache が当たらない理由）は [llm-serving.md](llm-serving.ja.md) を参照してください。

## 3. サービスをインストールする

```bash
sudo MC_DATA=/media/ssd/mc-data bash deploy/install-units.sh
sudo systemctl start voice-bench      # 8095 音声アシスタント
sudo systemctl start caption-bench    # 8096 会議字幕
```

`install-units.sh` は `/etc/meeting-copilot/env`（0640、LLM の Bearer トークンを含む）を生成し、unit 内の `__HOME__` / `__USER__` / `__UID__` を実際の値に置き換えます。**トークンはリポジトリに入っていません**。Magnitude が生成したものを、スクリプトがその場で読んで埋めます。

## 4. 検証

```bash
bash scripts/healthcheck.sh
```

次のように表示されれば正常です：

```
== 服务
  ✓ magnitude active（重启 0 次）
  ✓ voice-bench active
  ✓ caption-bench active
== 字幕链路的关键指标
  段 12  错 0  q_asr=0 q_tr=0 dropped=0
  首字 443ms  译文 1005ms  延迟 290ms
```

さらにオフライン回帰テストを 1 本追加します（マイク不要、モデルがオンラインでなくても、スタブで動きます）：

```bash
.venv/bin/python test_caption_pipe.py      # 16 グループ
node tools/dom_render_test.js              # レンダリング層 26 項目
```

## 5. 画面を開く

```
http://<LAN-IP>:8095   音声アシスタント（ヘッドセットに向かって話す。ウェイクワードは「小八你好」）
http://<LAN-IP>:8096   会議字幕（「开始」を押すと、左上のステータスランプが「运行中」に変わる）
```

**ページが「连接中…」（接続中…）で止まる場合は、まずここを見てください**：それは JS が動いていないという意味です（ネットワークの問題ではありません）。`curl -s localhost:8096/ | node --check` のようなチェックはすでに CI に入っています。ローカルでは `.venv/bin/python test_caption_pipe.py` を直接実行すればよく、この中の `test_page_js_syntax` が `node --check` を一通り実行します。

## 6. インストールできない 3 つのケース

| 現象 | 原因 | 対処 |
|---|---|---|
| `piper-phonemize` が入らない | 古いバージョンの piper が依存しているが、aarch64 用の wheel がない | `piper-tts>=1.8.0` を使う（本リポジトリはすでにこれを要求） |
| 日本語 TTS で `pyopenjtalk` が見つからないと出る | 公式パッケージはソースのみで、ビルドに失敗する | `pyopenjtalk-plus` に切り替える（モジュール名は変わらない） |
| サービスは起動するが音が出ない | `XDG_RUNTIME_DIR` がない | unit のその行を消さない。手動で実行するときは自分で export する |

詳しくは [troubleshooting.md](troubleshooting.ja.md) を参照してください。
