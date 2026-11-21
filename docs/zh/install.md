# 从零装一遍

[English](../en/install.md) · [**中文**](install.md) · [日本語](../ja/install.md)

假设你已经有一台刷好 JetPack 的 AGX Orin，能 ssh 进去，插好了采集卡和耳麦。
全程不需要联网推理——**ASR / TTS / OCR / LLM 全在本地**，装完之后断网也能用。

## 0. 前置检查

```bash
python3 -V                    # 3.10+（JetPack 7 是 3.12）
nvidia-smi | head -5          # 驱动 / CUDA 正常
pactl list short sources      # 能看到采集卡和耳麦
ls /dev/video*                # 有 /dev/video0
df -h /                       # 根盘最好留 5G 以上
```

## 1. 取代码 + 装依赖

```bash
git clone https://github.com/<you>/meeting-copilot.git /opt/meeting-copilot
cd /opt/meeting-copilot

# 数据（3.4G 模型 + 之后的 22.9G 权重）建议放外置盘
export MC_DATA=/mnt/mc-data

sudo -E bash scripts/install.sh
```

`install.sh` 做四件事：apt 装 PipeWire/构建工具 → 建 `.venv` → `pip install -r requirements.txt`
→ 调 `scripts/download-models.sh` 下模型 → 最后跑 `tools/check_env.py` 自查。

**它不会装 TensorRT / cuda-python**：这两个必须和 JetPack 的 CUDA 严格配套，
pip 装错版本会**静默**退回 CPU（OCR 从 300ms 变 3s，日志一个字不提）。
要 GPU 加速 OCR，先确认 `python -c "import tensorrt, cuda"` 能过，再按
[ocr-gpu.md](ocr-gpu.md) 生成 engine。

想手工装也行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download-models.sh
```

## 2. 起一个大模型

两条路，**先选一条**：

| | Magnitude + Qwen3.5-35B-A3B | vLLM + Qwen2.5-7B-AWQ |
|---|---|---|
| 显存/内存 | 22.9G 权重，常驻约 26G | 约 6G |
| 首字 | 300~470ms（短句） | **90ms** |
| 长回答 | **快 25~35%** | 慢 |
| 中日混排 / 幻觉 | 干净 | 会出「会議的时间确认 needed。」 |
| 起服务 | 冷启约 3 分钟 | 约 40 秒 |

```bash
# A. 35B-A3B（推荐：质量优先，字幕与助手都用它）
bash scripts/llm-magnitude.sh            # 生成 dockerd 配置
sudo bash deploy/install-units.sh        # 装 magnitude / magnitude-dockerd unit
sudo systemctl start magnitude           # 冷启约 3 分钟（22.9G 权重）

# B. 7B（延迟优先 / 内存紧张）
bash scripts/llm-vllm.sh
# 然后把 /etc/meeting-copilot/env 改成：
#   LLM_BASE=http://127.0.0.1:8000/v1
#   LLM_MODEL=qwen
#   LLM_KEY=
```

细节（为什么 35B 的 MoE 反而更划算、prompt 缓存为什么命中不了）见
[llm-serving.md](llm-serving.md)。

## 3. 装服务

```bash
sudo MC_DATA=/mnt/mc-data bash deploy/install-units.sh
sudo systemctl start voice-bench      # 8095 语音助手
sudo systemctl start caption-bench    # 8096 会议字幕
```

`install-units.sh` 会生成 `/etc/meeting-copilot/env`（0640，含 LLM 的 Bearer token）
并把 unit 里的 `__HOME__` / `__USER__` / `__UID__` 替换成真实值。
**token 不在仓库里**，Magnitude 自己生成、脚本现读现填。

## 4. 验证

```bash
bash scripts/healthcheck.sh
```

期望看到：

```
== 服务
  ✓ magnitude active（重启 0 次）
  ✓ voice-bench active
  ✓ caption-bench active
== 字幕链路的关键指标
  段 12  错 0  q_asr=0 q_tr=0 dropped=0
  首字 443ms  译文 1005ms  延迟 290ms
```

再加一条离线回归（不需要麦克风、不需要模型在线，打桩跑）：

```bash
.venv/bin/python test_caption_pipe.py      # 16 组
node tools/dom_render_test.js              # 渲染层 26 项
```

## 5. 打开界面

```
http://<本机IP>:8095   语音助手（对着耳麦说话；唤醒词「小八你好」）
http://<本机IP>:8096   会议字幕（点「开始」，左上角状态灯应变成「运行中」）
```

**页面卡在「连接中…」先看这里**：那说明 JS 没跑起来（不是网络问题）。
`curl -s localhost:8096/ | node --check` 这种检查已进 CI；本地可直接
`.venv/bin/python test_caption_pipe.py`，其中 `test_page_js_syntax` 会过一遍 `node --check`。

## 6. 装不动的三种情况

| 现象 | 原因 | 处理 |
|---|---|---|
| `piper-phonemize` 装不上 | 老版本 piper 依赖它，aarch64 没 wheel | 用 `piper-tts>=1.8.0`（本仓库已要求） |
| 日语 TTS 报 `pyopenjtalk` 找不到 | 官方包只有源码，编译失败 | 换成 `pyopenjtalk-plus`（模块名不变） |
| 服务起来但没声音 | 缺 `XDG_RUNTIME_DIR` | unit 里的那行别删；手工跑时自己 export |

更多见 [troubleshooting.md](troubleshooting.md)。
