# 延迟：预算、怎么量、哪些是硬地板

[English](../en/latency.md) · [**中文**](latency.md) · [日本語](../ja/latency.md)

这套系统的调参空间不大，因为**大部分时间花在物理上**：ASR 要听完、MoE 要做 prefill、
GPU 要做 OCR。这份文档把「能省的」和「别再追的」分开。

## 一次字幕的延迟预算（线上实测）

```
说话结束
  ├─ 断句收尾（分层停顿 / soft-cut）     0.30~0.90s   ← 由说话节奏决定，不是浪费
  ├─ SenseVoice ASR                      ~290ms
  ├─ LLM prefill → 首字                  400~470ms   ← ttft_ms
  ├─ 流式译文逐字推                      ~80ms/增量
  └─ 整句译完                            ~1000ms      ← trans_ms

感知延迟 lag_ms ≈ 290ms（从音频到第一行字）
```

**关键**：首字一到就开始显示，用户不需要等 `trans_ms`。这也是为什么
`--no-stream-trans` 千万别开——整句译完才显示会把感知延迟从 0.26s 推到 0.8~1.3s。

## 首字的经验公式（35B-A3B）

```
TTFT ≈ 65ms + 3.5~6.5ms × 新 token        线上斜率实测 4.5ms/token
```

「新 token」= 引擎返回的 `prompt_n`（**本轮真正处理的 token**），
不是 `prompt_n - cache_n`；`cache_n` 是复用的，两者相加才是整条 prompt。
**首字 200ms 是硬地板**，别再往下追。

由此得到三条反直觉结论：

| 做法 | 结果 | 原因 |
|---|---|---|
| 把 system 从 250 token 砍到 70 | **更慢** | 用户内容每句都变，短前缀 `cache_n` 恒 0（24/24），整份重付 prefill |
| `--ctx-lines 2 → 1` | **省约 90ms** | 上文部分每句都变，是真正的新 token |
| `--tr-workers 2 → 1` | 无差别 | 单句延迟不受并发数影响，只是少抢显存 |

**长前缀不要钱**：稳定的长 system 会被缓存复用（约 290 token），砍它反而破坏了复用。

## 7B vs 35B（同一段 62 秒音频，长句正面对比）

| | 7B-AWQ（dense，vLLM） | 35B-A3B（MoE，Magnitude） |
|---|---|---|
| 短句首字 | **90ms** | 300~470ms |
| 长回答全文 | 慢 | **快 25~35%** |
| 中日混排 / 幻觉 | 会出错 | 干净 |

MoE 每 token 只激活约 3B，解码便宜；但 prefill 要读几乎全部专家，所以贵。
**短回答 dense 赢，长回答 MoE 赢** —— 别拿 7B 的直觉套 35B。

## 判据（只看这三个数）

```bash
curl -s localhost:8096/api/status
```

| 字段 | 健康 | 不正常时先做什么 |
|---|---|---|
| `q_asr` | 0 | 涨 → ASR 线程不够 / 采集被占 |
| `q_tr` | 0 | **涨 = 翻译跟不上**：先 `--ctx-lines 1`，再看 LLM 是否在抢显存 |
| `dropped` | 0 | >0 → 队列开始丢段，延迟已超实时，必须处理 |

`NRestarts` 和 `ActiveEnterTimestamp` 一起看，能判断是不是在无限重启循环。

## 怎么量自己的机器

```bash
.venv/bin/python tools/ttft_prompt_ab.py     # prompt 档位 × 前缀缓存
.venv/bin/python tools/ttft_scenarios.py     # 不同句长 / 上文行数的首字
.venv/bin/python tools/trans_stream_probe.py # 流式 vs 整句
.venv/bin/python tools/sse_verify.py 45      # SSE 中间态数量与间隔
.venv/bin/python tools/e2e_latency.py        # 端到端
.venv/bin/python tools/wav_seg_ab.py --wav x.wav --a 0.45 --b 0.7
```

## 别当 bug 查

| 现象 | 真相 |
|---|---|
| 看一次屏幕要 3.4~3.9s | 采集卡要丢 40 帧预热（约 1.2s）+ OCR ≈ 行数 × 30ms |
| 改了页面前端没变化 | `PAGE` 是常量：重启服务 + **用户 F5** |
| `cache_n` 一直是 0 | 用户内容每句都变，前缀失效；不是缓存坏了 |
| vLLM 命中不了 35B 前缀缓存 | 1056 token 粒度，模型 + 引擎的物理特性 |
| 首字压不到 200ms 以内 | 65ms 固定开销 + prefill，是硬地板 |
