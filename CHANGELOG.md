# Changelog

本仓库是 2026-11-17 ~ 11-21 五天实机开发的整理结果，首次开源发布时把过程压缩成一个版本。
每个技术决策背后的实测数据与被否掉的方案，见
[docs/engineering-log.md](docs/engineering-log.md)。

## v0.1.0 — 2026-11-21

### 新增
- 语音助手（`:8095`）：唤醒词、VAD 断句、SenseVoice、LLM、Piper 按句合成、
  barge-in、看屏幕 OCR、浏览器测试台
- 会议字幕（`:8096`）：PipeWire 采集、分层停顿断句、流式翻译、SSE、增量 DOM、
  最新一条在最上、监听回放
- 屏幕词表：读音相似 + 共用一字的按需提示，命中才加 token
- TensorRT FP16 替换 RapidOCR 的 rec 后端（OCR 4.12s → 1.17s / 132 行）
- 完整部署：4 个 systemd unit、一键安装、模型下载、健康检查、离线回归测试
- 三语 README 与 10 份专题文档、工程日志

### 定稿的默认参数
- `--tr-workers 1 --ctx-lines 1 --tr-prompt full`
- `--max-s 6.0` + 分层停顿 + `--soft-cut 0.7` + `--min-s 0.25`
- `--screen-hints --screen-every 15`

### 关键修复
- 全量重绘导致的滚动锚点漂移（长句把上一行顶没了）
- 倒序渲染时滚动补偿符号写反
- `PAGE` 非 raw 字符串把内嵌 JS 打断，页面停「连接中…」且按钮失效
- 语言指令只挂 system 导致 35B 输出错误语言 → 改 system + user 双挂
- Qwen3.5 默认开 thinking 吃光 `max_tokens` → 显式 `enable_thinking=false`
- 屏幕文字整段注入造成自我循环 → 改为命中才提示
- `pkill -f` 杀掉自己的 ssh、dockerd pidfile 导致无限重启

### 已知限制
- 屏幕词表的真实命中还未在真实会议里出现（逻辑由打桩用例覆盖）
- 改前端仍需手动 F5
- 冷启动后 35B 权重加载约 150s
- 真·断电冷重启的完整链路尚未复验
