# 大模型：两条路线、三个必须知道的引擎细节

两个服务（`voice_server.py` / `caption_server.py`）都只认一个东西：**OpenAI 兼容的
`/chat/completions` 端点**。所以换后端不用改业务代码，只要换 `--llm-base` / `--llm-model` /
`--llm-key` 三个参数。

## 1. 两条路线

| | Magnitude + Qwen3.5-35B-A3B | vLLM + Qwen2.5-7B-AWQ |
|---|---|---|
| 端点 | `http://127.0.0.1:10100/inference/v1` | `http://127.0.0.1:8000/v1` |
| 模型 id | `hf:unsloth/Qwen3.5-35B-A3B-GGUF/...Q4_K_XL.gguf` | `qwen` |
| 鉴权 | Bearer（Magnitude 自己生成，见下） | 无 |
| 权重 | 22.9G GGUF | 约 5.2G AWQ |
| 首字（短句） | 300~470ms | **90ms** |
| 长回答（读屏幕/总结） | **快 25~35%** | 慢 |
| 中日混排 / 幻觉 | 干净 | 会出「会議的时间确认 needed。」、`-topic1-`、西语词 |
| 冷启动 | 约 3 分钟（含权重加载 150s） | 约 40 秒 |

**结论**：这套系统是「日语说话、中文播报/字幕」，语言指令比一般场景更苛刻。
35B 上线后 7 处已知翻译失败全部消失；代价是短句首字慢 170~370ms。
内存紧张或只做短句字幕时用 7B 也可以，见 `scripts/llm-vllm.sh`。

### 为什么 MoE 长回答反而快

35B-A3B 是 MoE，每 token 只激活约 3B 参数；dense 7B 每个 token 都要过全部参数。
代价在 **prefill**（要读几乎全部专家），收益在**解码**。所以：
短回答（几十字）7B 赢，长回答（几百字）35B 赢 25~35%。

## 2. 启动与参数

### Magnitude

```bash
bash scripts/llm-magnitude.sh                 # 生成独立 dockerd 的 daemon.json
sudo bash deploy/install-units.sh             # 装 magnitude / magnitude-dockerd
sudo systemctl start magnitude                # 起容器 → 等 health → 显式 load 模型
```

Magnitude 的 apiKey 是**首次启动时自己生成**的，不是我们配的：

```bash
python3 -c "import json; print(json.load(open('$MC_DATA/magnitude-data/.magnitude/config.json'))['network']['apiKey'])"
```

`deploy/install-units.sh` 会现读现写进 `/etc/meeting-copilot/env`（0640）。
**不要把这个 key 提交到仓库**。

为什么单独一个 dockerd：Magnitude 要 `--runtime nvidia`、要独占 22.9G 的 data-root，
独立 daemon 让「停它不影响机器上别的容器」。不想要这套就把
`deploy/magnitude-up.sh` 里的 `DOCKER_HOST` 去掉。

### vLLM

```bash
bash scripts/llm-vllm.sh
# 然后 /etc/meeting-copilot/env 改成：
#   LLM_BASE=http://127.0.0.1:8000/v1
#   LLM_MODEL=qwen
#   LLM_KEY=
```

### 业务侧参数

```bash
--llm-base   http://127.0.0.1:10100/inference/v1
--llm-model  hf:unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
--llm-key    <Bearer>
```

## 3. 三个必须知道的引擎细节

### 3.1 `prompt_n` 不是「新增 token 减去缓存」

引擎返回的 `prompt_n` = **本轮真正处理的 token 数**，`cache_n` = 复用的 token 数，
两者相加才是整条 prompt。**新 token 就是 `prompt_n`**，
不是 `prompt_n - cache_n`（早期我们写反过，导致斜率估计错了两倍）。

由此得到的经验公式（35B-A3B）：

```
TTFT ≈ 65ms + 3.5~6.5ms × 新 token       # 线上实测斜率 4.5ms/token
```

首字 200ms 是硬地板，别再往下追。

### 3.2 短 prompt 反而更慢

直觉是「把 system 砍到 70 token，prefill 就快」。实测 **否**：

- 长前缀（full，约 250 token）：稳定复用 290 token，`cache_n` 高
- 短前缀（short，约 70 token）：用户内容每句都变时 `cache_n` **恒 0**（24/24），
  整份重付 prefill，首字反而更慢
- 同一 prompt 连发两遍：两档都命中（`297/1` vs `91/1`）→ 不是「短了不能缓存」，
  是**每句都变的内容让前缀失效**

所以默认 `--tr-prompt full`。`short` 保留作对照实验，不建议生产用。
同理 `--ctx-lines 1` 砍掉上文那 30 个 token 是真的省（约 −90ms），
因为那部分**每句都变**；`--tr-workers 1` 则对延迟无差别，只是少抢显存。

### 3.3 语言指令挂在 user 末尾，system 里也要双挂

35B 对指令位置极敏感（日语会议 → 中文输出）：

| 指令位置 | 中文输出成功率 |
|---|---|
| 只挂 system 末尾 | **0/6** |
| 只挂 user 末尾 | **6/6** |
| system + user 双挂 | 4/4 |

实现在 `voice_server.py` 的 `_lang_directive()`：user 内容末尾追加一份，
system 里也带一份。字幕那边同理，见 `caption_server.py` 的 `_tr_system()`。

另外两个 35B 专属坑：

- Qwen3.5 的 chat template **默认开 thinking**：`max_tokens=200` 全烧在思考里、
  `content` 是空串，表现为「没有译文」。必须传
  `chat_template_kwargs={"enable_thinking": false}`
- **动态 system**：system 里一旦出现「[屏幕文字]」这个词，模型就会去引用它。
  所以没有 OCR 内容时用另一份 system，压根不提「屏幕文字」

## 4. 别当 bug 查

- vLLM 命中不了 35B 的前缀缓存（1056 token 粒度），这是**模型 + 引擎的物理特性**
- 35B 的 prefill 贵、decode 便宜；dense 反过来。别拿 7B 的直觉套 MoE
- 冷启动第一次 `models load` 约 150s，日志不动不是卡死
