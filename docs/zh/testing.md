# 测试：分层回归，不靠麦克风

[English](../en/testing.md) · [**中文**](testing.md) · [日本語](../ja/testing.md)

改一行断句参数、换一个模型、动一次前端，都可能把别的环节带坏。这套测试按层分开，
**全部离线可跑**（ASR / LLM / OCR 都打桩），不需要模型在线，也不需要人对着麦克风说话。

## 快速全跑

```bash
cd /opt/meeting-copilot

# 数据层（脚本式，ASR/翻译打桩，16 组）
.venv/bin/python test_caption_pipe.py

# 其余 5 个 unittest 文件
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py

# 渲染层（26 项，需要 node）
node tools/dom_render_test.js

# 数据层自检：译文 0 回退、0 变空（**在线**，会连 8096 订阅 90 秒）
.venv/bin/python tools/sse_regress.py
```

> `test_caption_pipe.py` 是**脚本**，打桩在 `__main__` 里装，别用 pytest 单独跑它
> （那样桩没装上会误报）。其余 5 个是标准 `unittest`，pytest 和
> `python -m unittest` 都行。
>
> `tools/sse_regress.py` **不是离线测试**：它连正在运行的 8096 订阅 90 秒实流，
> 检查译文有没有回退 / 变空。要跑它先 `systemctl start caption-bench`。

## 各层覆盖什么

| 层 | 文件 | 覆盖 |
|---|---|---|
| 字幕数据 | `test_caption_pipe.py` | 两段式出字（原文先于译文）、队列解耦、上头文、短段丢弃、软切、分层停顿、草稿、清屏、`--tr-prompt` 两档、屏幕词表命中、**页面 JS 语法** |
| 渲染层 | `tools/dom_render_test.js` | 26 项：节点身份、顺序（最新在最上）、滚动补偿、倒序插入、草稿行位置 |
| 数据自检 | `tools/sse_regress.py` | SSE 流里译文必须是递增前缀，**0 回退、0 变空** |
| 打断 | `test_barge.py` | 播报中插话停、串音不误触 |
| TTS 路由 | `test_tts_routing.py` | 中日音色自动切、中文回答里的数字不被日语音读 |
| 视觉 | `test_vision.py` | 预热黑帧、OCR 脱敏（`sk-***`）、TensorRT 挂接 |
| 音频前端 | `test_voice_probe.py` | 150Hz 高通、VAD、切分 |
| 唤醒 | `test_wake.py` | KWS 触发与去抖 |

### 为什么有 `test_page_js_syntax`

2026-11-21 出过一次事故：给页面加 tooltip 时写了 `'\n'`，而 `PAGE` 当时不是 raw 字符串，
Python 把 JS 里的换行转义变成真换行，单引号字符串被截断 → **整段 `<script>` 语法错误，
页面停在「连接中…」、按钮全失效，服务端却完全正常**。

这个用例直接把渲染后的 `<script>` 抽出来交给 `node --check`，外加一条
「单引号每行必须成对」的兜底检查（没有 node 的机器跳过）。CI 里也跑一遍。
**改 `PAGE` 之后不跑它，等于没改。**

## A/B 与探针（`tools/`）

这些不是回归测试，是「换东西之前先量」的工具，可以按需单独跑：

| 工具 | 用途 |
|---|---|
| `wav_seg_ab.py` | 两段断句参数在同一段 wav 上正面对比 |
| `voice_llm_ab.py` | 换 LLM 时按判据打分（字幕 / 助手两套） |
| `ttft_prompt_ab.py` | prompt 档位 × 前缀缓存 × 新 token 的首字 |
| `ttft_scenarios.py` | 不同句长 / 上文行数的首字场景 |
| `screen_hint_ab.py` / `screen_terms_probe.py` | 屏幕词提示值不值 / 能否撞上 ASR 错字 |
| `trans_stream_probe.py` / `stream_verify.py` | 流式 vs 整句、译文是否递增前缀 |
| `sse_verify.py` / `sse_multi_client.py` | SSE 推送间隔、两个页面同时开会不会互相抢补丁 |
| `bench_sensevoice.py` / `trt_rec_probe.py` | ASR 与 TensorRT 后端基准 |
| `check_env.py` | 装完自查：依赖 / 模型 / 设备 / 有没有静默退回 CPU |

## 换模型 / 换参数的验收清单

1. `.venv/bin/python test_caption_pipe.py` → `ALL PASS`
2. `.venv/bin/python tools/wav_seg_ab.py --wav <62 秒真实音频>` → 断句完整度（**只看 ASR 文本是否成句**，别看静音切占比）
3. `voice_llm_ab.py` → 语言纯净度、幻觉、长回答耗时
4. 上线后 `bash scripts/healthcheck.sh` → `q_asr=0 q_tr=0 dropped=0`
5. 前端改动 → 重启服务 + **F5** + `test_page_js_syntax`

## CI

`.github/workflows/ci.yml` 只做三件不依赖硬件的事：

1. `python -m compileall -q .`（全仓语法）
2. 从 `caption_server.py` 抽出 `<script>` 跑 `node --check`
3. 扫一遍仓库，确认没有密钥 / 本机路径 / 内网 IP

真机相关（ASR、OCR、TensorRT、GPU）只能在上机跑，别指望 CI 覆盖。
