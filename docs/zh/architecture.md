# 架构：两条链路、一个 LLM、三个进程

[English](../en/architecture.md) · [**中文**](architecture.md) · [日本語](../ja/architecture.md)

```
                          ┌─────────────────────────────────────────────┐
   HDMI 采集卡 ─ UVC ───→│  vision.py（常驻抓帧 / OCR / TensorRT rec）  │──→ /api/look
                          └─────────────────────────────────────────────┘        │
                                                                                  │
   ┌──────────── voice-bench :8095（语音助手）──────────────────────────────┐    │
   │  H340 麦 → KWS「小八你好」→ pw-record → Silero VAD → SenseVoice (ja)  │    │
   │       → LLM(35B) → 按句切 → Piper(ja/zh) → pw-play → 本地耳机         │←───┘
   │  浏览器只负责触发与显示；录音和播报都在 Jetson 上，不在浏览器里        │
   └────────────────────────────────────────────────────────────────────────┘
                                     │ 共用同一个 OpenAI 兼容端点
                          ┌──────────┴──────────┐
                          │  Magnitude :10100   │  （或 vLLM :8000）
                          │  Qwen3.5-35B-A3B    │
                          └──────────┬──────────┘
   ┌──────────── caption-bench :8096（会议字幕）────────────────────────────┐
   │  采集卡音频 → pw-record（不独占）→ Segmenter（分层停顿 + soft-cut）   │
   │      → SenseVoice → translate_stream（流式、按需屏幕词提示）           │
   │      → SSE /api/events → 浏览器增量 DOM 渲染（最新一条在最上）         │
   └────────────────────────────────────────────────────────────────────────┘
```

**为什么分成两个进程**：语音助手是「一问一答、要播报」，字幕是「不断流、要实时」，
两者的线程模型和延迟预算完全不一样。强行合在一起会互相抢 ASR 与 LLM。
它们通过 `voice_server.py` 的 `/api/look` 共享 OCR 结果——**采集卡是独占设备，
只有 voice-bench 常驻持有 `/dev/video0`**，字幕要屏幕词时去问它，不自己开设备。

## 进程与端口

| 进程 | 端口 | 角色 | 关键环境变量 |
|---|---|---|---|
| `magnitude-dockerd` | socket（非 TCP） | 独立 dockerd | — |
| `magnitude` | 10100 | 35B-A3B 推理 | `MC_DATA` / `MC_MODEL_ID` |
| `voice-bench` | 8095 | 语音对话 + 看屏幕 | `XDG_RUNTIME_DIR`、`LLM_*`、`MC_SINK` |
| `caption-bench` | 8096 | 双语字幕 + 监听回放 | `XDG_RUNTIME_DIR`、`LLM_*` |

## HTTP 接口

**8095 语音助手**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 测试台页面 |
| GET | `/api/health` | 探活（含 LLM 是否可用） |
| POST | `/api/turn` | 完整一轮：录音 → ASR → LLM → TTS（事件流） |
| POST | `/api/say` | 直接合成一句话（调试 TTS） |
| POST | `/api/look` | 抓一帧 → OCR → 返回屏幕文字；字幕服务从这里取词表 |
| POST | `/api/route` | 只跑路由/判停逻辑 |
| POST | `/api/reset` | 清空对话上下文 |
| GET | `/api/live` `/api/poll` | 状态推流 / 兜底轮询 |
| POST | `/api/kws_check` | 唤醒词自检 |

**8096 会议字幕**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 字幕页面 |
| GET | `/api/events` | SSE，译文每 ~80ms 一个增量；前端断线自动回落 `/api/poll` |
| GET | `/api/poll?after=N` | 轮询兜底（250ms） |
| GET | `/api/status` | 全量状态，含 `q_asr/q_tr/dropped/screen/monitor` |
| POST | `/api/control` | `{"action": "start" \| "stop" \| "clear"}` |
| POST | `/api/monitor` | 「听 HDMI」：把采集卡收到的声音回放到耳机 |

## 字幕为什么要「流式 + SSE + 增量 DOM」

1. **流式 LLM**：整句译完才显示要等 0.8~1.3s；改成逐 token 推送后 0.26s 就开始冒字
2. **SSE**：服务端每 50ms 检查一次更新，中间态一到就推；轮询最多多等 250ms。
   服务端不用 `take_updates()`（那是「取走即清空」，两个页面会互相抢补丁），
   每个连接自己比对译文快照
3. **增量 DOM**：早期用全量 `innerHTML` + 像素 `scrollY` 回填，长句译文正在变高时
   重建后同一个 y 指向别的内容 → 「下一句一来，上一句整行消失」。
   现在每行一个常驻节点，只有变化的文本节点被写；`#main{overflow-anchor:none}`

排列方向是**最新一条在最上**，老行往下推。视线锚点从底部变成顶部，
翻到下面看旧行时按 `top` 差值补偿 `scrollY`（倒序时补偿符号是 `+`）。

## 文件地图

| 文件 | 职责 |
|---|---|
| `voice_server.py` | 唤醒 / 录音 / ASR / LLM / TTS / Web 测试台 / `/api/look` |
| `caption_server.py` | 采集 / 断句 / ASR / 流式翻译 / SSE / 字幕页面 |
| `voice_probe.py` | 音频前端：高通、VAD、切分、SenseVoice 封装 |
| `vision.py` | 抓帧（丢 40 帧预热）、OCR、屏幕词候选 |
| `trt_rec.py` | 把 RapidOCR 的 rec 后端换成 TensorRT FP16 |
| `paths.py` | **唯一**的目录约定，`MC_DATA` 可覆盖 |
| `deploy/` | 4 个 systemd unit + 安装脚本 |
| `tools/` | A/B 量测、自检、探针（都不参与运行时） |
| `test_*.py` | 打桩回归，不需要麦克风和模型在线 |
| `docs/zh/engineering-log.md` | 每个默认值背后的实测数据与失败方案 |

## 线程 / 实时性

字幕链路上 `q_asr`（等 ASR 的段）和 `q_tr`（等翻译的段）是**唯一可靠的实时性判据**：

```
q_asr = 0, q_tr = 0, dropped = 0   → 跟得上
q_tr 持续上涨                     → 翻译跟不上，先降 --ctx-lines，再看 LLM 是否在抢显存
dropped > 0                       → 队列开始丢段，延迟已超过实时，必须处理
```

`--tr-workers 1` 是定稿值：加 worker 对单句延迟无差别，只是多抢显存。
`--ctx-lines 1` 砍掉每句都变的上文，实测省约 90ms。
