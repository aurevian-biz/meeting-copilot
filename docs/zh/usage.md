# 怎么用

[English](../en/usage.md) · [**中文**](usage.md) · [日本語](../ja/usage.md)

## 会议字幕（8096）

打开 `http://<本机IP>:8096`，点右上角 **开始 / 停止 / 清屏**：

- **开始 / 停止**：控制采集与断句；`clear` 会清掉行并重置 seq（前端也会清空本地 DOM）
- **听 HDMI**：把采集卡收到的声音回放到耳机，开会时当监听口用
- **音量条 / dB**：实时电平与门限，条变绿说明超过门限、正在出声
- **状态灯**：`连接中…` → `运行中` / `已停止` / `断开`
- **屏幕 N 词 · 命中 M**：鼠标悬停看词表样本、最近命中、看一次屏幕的耗时
- **最新一条在最上**：草稿行占最顶，往下翻看历史时不会被新行顶跑

### 页面没反应时的判断顺序

状态灯停在「连接中…」、dB 是 `-99`、段数 `0`、按钮点了没反应 —— 这几个值全是
HTML 里写死的初始值，说明**脚本根本没执行**，不是网络问题。按顺序查：

1. `curl -s localhost:8096/api/status` 有没有数据（服务端活着 ≠ 页面活着）
2. 浏览器控制台有没有 `SyntaxError`
3. `.venv/bin/python test_caption_pipe.py`（其中 `test_page_js_syntax` 会 `node --check`）
4. 改过前端就 **F5**：`PAGE` 是模块级常量，重启服务不会帮浏览器刷新 HTML

## 语音助手（8095）

打开 `http://<本机IP>:8095`：

- 直接点按钮说话，或说唤醒词 **「小八你好」** 触发常驻监听
- 播报中插话会打断（`barge-in`），串音不会误触
- 「看屏幕」按钮（或语音提问）触发一次 OCR，把屏幕文字喂给 LLM
- 浏览器第一次要**手动点一下**才能播放声音：浏览器自动播放策略会拦住，
  播报走本机 `pw-play`（`--local-speak`），不依赖浏览器自动播放

## 常用参数

### 字幕断句（默认值是实测调出来的，别随手改）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-s` | 6.0 | 硬切上限。4.5s 时 26% 的段被切断（「大会」切两半）；**超过 7s 会出现幻觉** |
| `--hold` | 0.45 | 基准静音门限；分层停顿另有 `--hold-short 0.3` / `--hold-long 0.9` |
| `--seg-mid` / `--seg-long` | 1.2 / 2.4 | 短句、中句、长句用不同的停顿门限 |
| `--soft-cut` | 0.7 | 硬切前在尾巴里回退找能量低谷的搜索窗（秒） |
| `--min-s` | 0.25 | 短于此直接丢（咔哒声、咳嗽） |
| `--ctx-lines` | 1（unit 里） | 上文行数；每句都变，砍掉省约 90ms |
| `--tr-workers` | 1（unit 里） | 加 worker 对单句延迟无差别，只多抢显存 |
| `--tr-prompt` | full | 短版反而更慢，见 [llm-serving.md](llm-serving.md) |
| `--screen-hints` / `--screen-every` | 关 / 15s | 按需屏幕词提示，见下 |
| `--wav` | — | 不采集，直接跑一个 wav 文件（离线复现断句问题） |

### 屏幕词提示（可选）

`--screen-hints` 打开后，每 `--screen-every` 秒向 voice-bench 的 `/api/look` 要一次 OCR，
抽候选词，只有**本句 ASR 文本里出现「读音相似、写法不同」的词**时才给翻译递一句提示：

```
【屏幕词】「畔蒜泰助」——本句的「阪井泰介」若指同一事物，请改用屏幕写法。
```

- 命中约 +25 token（+65ms），只在这一段付；没命中一个 token 都不加
- **别把整段屏幕文字塞进翻译**：实测 +70 token / +250ms，且 6 句里 3 句变差
  （我们的字幕页面本身就在屏幕上，会自我循环）
- 上限是「专有名词同音错字」，救不了普通误听（`つぶさに` → `つぶさんに`）
- 别让拉丁词进词表：OpenJTalk 会逐字母读成 50 拍，永远撞不上

## 状态怎么看

```bash
curl -s localhost:8096/api/status | python -m json.tool
```

重点字段：

| 字段 | 含义 | 健康值 |
|---|---|---|
| `q_asr` | 等 ASR 的段数 | 0 |
| `q_tr` | 等翻译的段数 | 0；**涨 = 翻译跟不上** |
| `dropped` | 队列丢段数 | 0 |
| `ttft_ms` | 译文首字延迟 | 400~600ms |
| `trans_ms` | 整句译完 | ~1000ms |
| `lag_ms` | 端到端延迟 | ~290ms |
| `screen` | 词表状态 | `terms / hits / polls / last_ms / last_err` |
| `monitor` | 回放状态 | `on / starts / error` |

一条命令看全貌：`bash scripts/healthcheck.sh`。

## 离线复现

不想对着麦克风反复说话时，用 wav 文件跑同一条断句链：

```bash
.venv/bin/python caption_server.py --wav sample.wav --no-draft
.venv/bin/python tools/wav_seg_ab.py --wav sample.wav --a 0.45 --b 0.7
.venv/bin/python tools/sse_regress.py     # 数据层：0 回退、0 变空
```
