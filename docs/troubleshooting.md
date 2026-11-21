# 排障：按现象查，不按模块猜

这里的每一条都是真踩过的，且**大部分不报错**——静默失效比报错更花时间。

## 1. 页面停在「连接中…」/ dB 是 -99 / 按钮全失效

先看 `/api/status`：如果服务端有数据、段数在涨，那就是**页面脚本根本没跑**。
那几个值全是 HTML 里写死的初始值，不是「SSE 连不上」。

```bash
curl -s localhost:8096/api/status          # 服务端活着？
.venv/bin/python test_caption_pipe.py      # test_page_js_syntax 会 node --check
```

常见根因是 `PAGE` 里的 JS 语法错（Python 非 raw 字符串吃掉了 `\n`）。
改完前端：**重启服务 + 浏览器 F5**，`PAGE` 是常量，重启不会帮浏览器刷新 HTML。

## 2. 服务起来了，但没声音 / 没录音

| 检查 | 原因 |
|---|---|
| unit 里有没有 `Environment=XDG_RUNTIME_DIR=/run/user/<uid>` | 缺了 `pw-play` / `pw-record` **静默失效** |
| `pactl list short sinks`，默认 sink 是不是 HDMI | 声音跑到显示器去了，给 `--local-sink` 指定 H340 |
| `arecord` 报「设备忙」 | 换 `--backend pw`；PipeWire 允许多客户端读同一 source |
| 浏览器第一次没声 | 自动播放策略拦截；播报已走本机 `pw-play`，先在页面点一下 |

## 3. systemd

| 现象 | 处理 |
|---|---|
| `stop` 之后变 `failed` | `caption-bench` 必须 `TimeoutStopSec=20`，否则采集线程拖到 90s 被 SIGKILL |
| `NRestarts` 一直涨 | 看 `ActiveEnterTimestamp` 是不是反复重启；`pkill -f` 杀自己的 ssh 也会伪装成重启 |
| Magnitude 无限重启（几千次） | 手工 dockerd 的 pidfile 挡住了 systemd；停手工进程、清 socket/pid 再交 systemd |
| `pkill -f 'dockerd --config-file ...'` 把 ssh 杀了 | 用字符类 `daemo[n].json`，避免匹配到自己的命令行 |

## 4. OCR / 看屏幕

| 现象 | 原因 |
|---|---|
| 前 1.2 秒全黑/噪点 | MS2109 预热帧；必须**丢 40 帧**再判（噪点会让 `std ≠ 0`，只丢几帧不够） |
| 看一次屏幕 3.4~3.9s | 正常：预热 1.2s + OCR ≈ 行数 × 30ms |
| OCR 突然慢 5~10 倍 | `cuda-python` 版本和 JetPack 不配套 → TensorRT 静默退回 CPU，日志一个字不提 |
| 画面变了但字没更新 | `/api/look` 内部按指纹缓存，同一画面不重复算 |
| 词表里混进自己的字幕 | 字幕页面就在屏幕上；已加「与最近 30 条字幕有 8 字共同子串就丢行」 |

## 5. 自己把自己唤醒

耳机的**外放会被麦克风拾到**，可能触发唤醒词形成自激。要么戴耳机，
要么关掉本地监听（页面上的「听 HDMI」），别把音箱摆在麦克风旁边。

## 6. 翻译质量

| 现象 | 处理 |
|---|---|
| 输出中日混杂 / 冒出西语词 | 换 35B-A3B；7B 在这套 system 下必出 |
| 35B 输出全是思考、`content` 空串 | Qwen3.5 模板默认开 thinking，必须 `enable_thinking=false` |
| 中文/日文指令不生效 | 语言指令要 **system + user 双挂**（只挂 system 是 0/6，只挂 user 是 6/6） |
| 模型凭空引用「屏幕文字」 | 没有 OCR 时用动态 system，压根别提这个词 |
| 半句被译成完整句 / 补主语 | 断句没调好，先 `wav_seg_ab.py` 看 ASR 文本完整度 |
| 译文只出了一半 | 看 SSE 流是不是递增前缀；`tools/sse_regress.py` 专门查 0 回退 / 0 变空 |

## 7. 长句把上一行顶没了

早期用全量 `innerHTML` + 像素 `scrollY` 回填，长句译文逐字变高时同一个 y 指向别的内容。
现在改成**常驻 DOM 增量更新**，`#main{overflow-anchor:none}`。
倒序（最新在最上）时滚动补偿符号是 `+`，写反过一次，用
`node tools/dom_render_test.js` 覆盖。

## 8. 别当 bug 查

- `q_tr` 涨才是真跟不上；`lag_ms` 单独高一点可能只是断句收尾
- 断句好坏**只看 ASR 文本是否成句**，别看「静音切占比」
- 首字 200ms 是硬地板；`TTFT ≈ 65ms + 4.5ms × 新 token`
- 短前缀 `cache_n` 恒 0，不是缓存坏了，是用户内容每句都变
- vLLM 命中不了 35B 前缀缓存（1056 token 粒度）
- 采集卡在扩展坞上曾掉线，长跑请直插机身 USB
- 屏幕词提示 `hits=0` 多半是屏幕上内容与会议无关，不是功能坏了

## 一行看全貌

```bash
bash scripts/healthcheck.sh
```

它会依次报服务、端口、设备、模型、以及 `q_asr / q_tr / dropped / ttft / screen`。
**先跑它，再翻日志。**
