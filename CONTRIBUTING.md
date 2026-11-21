# Contributing / 贡献指南

欢迎提 issue 和 PR。这个项目的价值有一半在「为什么这么设计」，所以**改默认值时请带上实测数据**。

## 提交前

```bash
python -m compileall -q .
.venv/bin/python test_caption_pipe.py        # 16 组，含页面 JS 语法
.venv/bin/python -m pytest -q test_barge.py test_tts_routing.py test_vision.py \
                              test_voice_probe.py test_wake.py
node tools/dom_render_test.js
bash scripts/healthcheck.sh                  # 上机的话
```

改动落在哪一层，就补哪层的用例：

| 改动 | 必须补 |
|---|---|
| 断句 / 队列 / 翻译请求 | `test_caption_pipe.py` |
| 字幕页面 JS（`PAGE`） | `test_page_js_syntax` + `tools/dom_render_test.js`，改完记得 F5 |
| SSE / 数据格式 | `tools/sse_regress.py` |
| 音频前端 / VAD | `test_voice_probe.py` |
| OCR / 屏幕词表 | `test_vision.py` / `test_screen_hints`（在 `test_caption_pipe.py` 内） |

## 几条约定

1. **不在代码里写绝对路径**。目录一律走 `paths.py`，用 `MC_DATA` 覆盖。
2. **不提交密钥**。`conf/env.example` 只是模板；真值在 `/etc/meeting-copilot/env`。
3. **`PAGE` 必须是 `r"""`**。普通字符串会把 JS 的 `\n` 吃掉，整段脚本语法错误，
   页面卡「连接中…」而服务端全好——已经出过一次事故。
4. **换模型/换参数要写清判据**：`q_asr/q_tr/dropped`、`ttft_ms`、断句完整度、
   语言纯净度，别只写「感觉更快」。
5. 中文/日文/英文 README 三份保持一致；细节文档写中文即可，不必三语同步。

## 脱敏规则（维护者备忘）

公开仓是从内部工作区整理出来的：那边混着调试截图、运行时产物，以及写死了本机路径 /
内网 IP / LLM Bearer token 的 unit。重新生成时用维护脚本（**只在内部工作区，不在公开仓**）：

```bash
# 在内部工作区执行，目标是公开仓目录
python tools/package_opensource.py --dest /path/to/meeting-copilot --force
```

它只挑运行时真正需要的文件、按规则替换敏感字符串，并在结束时自检
「无密钥 / 无本机路径 / 无内网 IP」；`docs/` 和 README 不会被删。
CI（`.github/workflows/ci.yml`）里还有一道同样的扫描，防止误提交。
