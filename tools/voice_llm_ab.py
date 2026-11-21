#!/usr/bin/env python3
"""voice_llm_ab.py —— 语音助手换 35B 值不值：拿真实的 system prompt 和四种真实场景比。

为什么单独写：字幕和语音助手的负载形态**完全不同**
  · 字幕：system 长（290 token，命中缓存）→ 新内容短（60~80 token）、输出短（30~60）
  · 助手：system 短（~200 字日语）→ 新内容可能很长（OCR 屏幕文字几百字）、输出长（160 token）
所以「字幕上 35B 首字 300ms」这个结论不能直接搬过来：OCR 场景下 35B 的 prefill
（4.5ms/token）可能把它拖到 1 秒以上，而 7B 的 prefill 便宜一个量级。

用法：
    # 四场景 × 两臂，各 3 次
    ./asr-venv/bin/python tools/voice_llm_ab.py \
        --arm '7B|http://127.0.0.1:8000/v1|qwen|' \
        --arm '35B|http://127.0.0.1:10100/inference/v1|<gguf id>|mag-xxx'

    # 只看 OCR 场景
    ./asr-venv/bin/python tools/voice_llm_ab.py --only ocr-long --arm ...

判据不是「首字多少」，而是**带 OCR 的场景有没有越过 1 秒**——语音助手的体感线。
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 直接复用线上那段 system prompt，别在这里抄一份（抄了就会漂）
try:
    import voice_server as vs
    _L = vs.Llm("", "")
    # 与语音助手一致：带 OCR 的用例用含屏幕规则的 system，不带的用纯对话版
    SYS_BASE = _L.system_core
    SYS_FULL = _L.system
except Exception:  # noqa: BLE001
    SYS_BASE = SYS_FULL = ""


def lang_directive(code: str) -> str:
    """线上就是 `system + _lang_directive(text)`，这里必须走同一个函数。

    为什么单列成函数：35B 对这句末尾的中文指令的遵循明显弱于 7B（README 4.22），
    所以「用什么语言回答」本身就是一个要量的指标，不能写成模块级常量一次性定死。
    """
    if not SYS_BASE or code in ("", "none"):
        return ""
    return vs.Llm("", "", reply_lang=code)._lang_directive("x")


def lang_of(text: str) -> str:
    """粗判输出语言：有假名就是日语；纯汉字无假名按中文算。"""
    if not text:
        return "?"
    if vs.KANA.search(text):
        return "ja"
    if vs.CJK.search(text):
        return "zh"
    return "other"

# 一小段真实的会议 OCR 文本：从字幕日志里摘的，不是编的
OCR_MID = (
    "[屏幕文字] 会議の議題は三つあります。第一に、来期の販売計画について。"
    "第二に、台湾向けの取引条件の見直し。第三に、新しい物流拠点の選定です。"
    "まず販売計画から説明します。前年比で十五パーセントの増加を見込んでいます。[/屏幕文字]"
)
OCR_LONG = OCR_MID.rstrip("[/屏幕文字]") + (
    " 具体的には、四月から六月までの第一四半期で、既存顧客への販売を八パーセント伸ばし、"
    "新規顧客の開拓で残りの七パーセントを確保する計画です。営業体制については、"
    "現在の東西二ブロック制を三ブロック制に改め、西日本の担当者を三人増員します。"
    "物流拠点は、大阪と名古屋の二候補に絞り込み、六月末までに最終決定します。"
    "コスト面では、拠点統合により年間で約二億円の削減効果を見込んでいますが、"
    "初期投資として三億円程度が必要になる見通しです。ご質問があればお願いします。[/屏幕文字]"
)

# asks 是同一场景的多种问法。为什么必须换：同一句重复发，整条 prompt 会被前缀缓存
# 全命中（35B 的 prompt_n 直接掉到 1），首字被低估到 78ms —— 那是假的。真实使用
# 每轮问句都不一样，缓存只能命中 system + OCR 那一段。
SCENES = {
    # 无 OCR 的纯对话：最短路径，history 也短
    "chat-short": dict(hist=0, ocr=None, asks=[
        "今日の会議は何時からですか？",
        "会議の開始時間を教えてください。",
        "今日の打ち合わせ、何時スタート？",
    ]),
    # 对话中段：6 轮历史（voice_server 就是取 history[-6:]）
    "chat-multi": dict(hist=3, ocr=None, asks=[
        "じゃあ、その三つの議題のうち、いちばん重要なのはどれ？",
        "三つの議題で、優先度がいちばん高いのはどれだと思いますか？",
        "その中で最初に議論すべきものは何でしょう？",
    ]),
    # OCR 中等：屏幕上约 100 字的日文
    "ocr-mid": dict(hist=0, ocr=OCR_MID, asks=[
        "画面に書いてある議題を教えて",
        "画面の内容を三つに整理して",
        "モニターに表示されている議題は何？",
    ]),
    # OCR 长：屏幕上一段完整口播稿，这才是「让它读屏幕」的真实形态
    "ocr-long": dict(hist=0, ocr=OCR_LONG, asks=[
        "画面の内容を一分で要約して",
        "画面の話を短くまとめてください。",
        "表示されている内容の要点を教えて",
    ]),
}

HIST_POOL = [
    ("おはようございます。今日の予定を教えてください",
     "おはようございます。今日は午前十時から定例会議、午後二時に客先訪問があります。"),
    ("会議の資料はもう配られましたか？",
     "はい、資料は昨日の夕方にメールで配信されています。"),
    ("参加者は何人くらいですか？",
     "今回は社内から八名、取引先から三名の合計十一名の予定です。"),
]


def build_messages(scene: dict, lang: str, rep: int = 0) -> list[dict]:
    sysp = (SYS_FULL if scene.get("ocr") else SYS_BASE) + lang_directive(lang)
    msgs = [{"role": "system", "content": sysp}]
    for i in range(scene["hist"]):
        u, a = HIST_POOL[(i + rep) % len(HIST_POOL)]
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    asks = scene.get("asks") or [scene.get("text", "")]
    text = asks[rep % len(asks)]
    if scene["ocr"]:
        text = f"{scene['ocr']}\n\n{text}"
    msgs.append({"role": "user", "content": text})
    return msgs


def one(base: str, model: str, key: str, msgs: list[dict], max_tokens: int
        ) -> dict:
    body = {
        "model": model, "messages": msgs, "stream": True,
        "temperature": 0.6, "max_tokens": max_tokens,
        "presence_penalty": 0.3, "frequency_penalty": 0.3,
        # 语音助手必须关思考：Qwen3.5 模板未定义该变量就走 thinking 分支，
        # 160 token 全烧在推理上，content 恒空（字幕那边踩过，见 README 4.18）
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    ttft, parts, tim = -1, [], {}
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    d = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if d.get("timings"):
                    tim = d["timings"]
                ch = d.get("choices") or []
                if not ch:
                    continue
                piece = (ch[0].get("delta") or {}).get("content") or ""
                if piece:
                    if ttft < 0:
                        ttft = int((time.time() - t0) * 1000)
                    parts.append(piece)
    except Exception as e:  # noqa: BLE001
        return dict(ttft=-1, total=-1, text=f"<ERR {type(e).__name__}: {e}>", tim={})
    out = "".join(parts).strip()
    return dict(ttft=ttft, total=int((time.time() - t0) * 1000),
                text=out, tim=tim, chars=len(out.replace(" ", "")))


def run_arm(name: str, base: str, model: str, key: str, scenes: list[str],
            reps: int, max_tokens: int, gap: float, lang: str = "zh") -> dict:
    print(f"\n########## {name}  {base}  {model}   reply-lang={lang}")
    rows = {}
    for sc in scenes:
        scene = SCENES[sc]
        got = []
        for i in range(reps):
            msgs = build_messages(scene, lang=lang, rep=i)
            r = one(base, model, key, msgs, max_tokens)
            r["lang"] = lang_of(r["text"])
            got.append(r)
            pn = (r.get("tim") or {}).get("prompt_n")
            extra = f" prompt_n={pn}" if pn is not None else ""
            print(f"  [{sc:<10} {i + 1}/{reps}] 首字 {r['ttft']:>5}ms "
                  f"全文 {r['total']:>5}ms {r.get('chars', 0):>3}字"
                  f" [{r['lang']}]{extra}  {r['text'][:44]}", flush=True)
            if gap:
                time.sleep(gap)
        ok = [g for g in got if g["ttft"] >= 0]
        rows[sc] = dict(
            n=len(ok),
            ttft_p50=int(statistics.median([g["ttft"] for g in ok])) if ok else -1,
            ttft_max=max((g["ttft"] for g in ok), default=-1),
            ttft_first=(ok[0]["ttft"] if ok else -1),
            total_p50=int(statistics.median([g["total"] for g in ok])) if ok else -1,
            chars_p50=int(statistics.median([g["chars"] for g in ok])) if ok else 0,
            prompt_n=(ok[0]["tim"].get("prompt_n") if ok and ok[0]["tim"] else None),
            over_1s=sum(1 for g in ok if g["ttft"] > 1000),
            lang_ok=sum(1 for g in ok if g["lang"] == lang),
            sample=ok[-1]["text"] if ok else "",
        )
    return rows


def concurrency_probe(base: str, model: str, key: str, sub_base: str,
                      sub_model: str, sub_key: str, n: int = 4,
                      lang: str = "zh") -> None:
    """最关键的一项：语音助手在生成时，字幕的首字会被拖到多少。

    GGUF 引擎大概率单槽（`magnitude serve --help` 只输出一个 -h，没有并发参数）。
    如果确实是单槽，两个服务共用 10100 就会互相排队。
    """
    print("\n########## 并发：助手长输出打满时，字幕请求的首字")

    def sub_caption(i: int, out: list) -> None:
        msgs = [
            {"role": "system", "content": "你是中日会议同传。把用户给的日文译成中文，只输出译文。"},
            {"role": "user", "content": "営業体制については、現在の東西二ブロック制を"
                                       "三ブロック制に改め、西日本の担当者を三人増員します。"},
        ]
        r = one(sub_base, sub_model, sub_key, msgs, 80)
        out.append((i, r["ttft"], r["total"]))

    # 1) 安静基线
    base_out = []
    for i in range(n):
        sub_caption(i, base_out)
    print(f"  安静时字幕首字 {[t for _, t, _ in base_out]}")

    # 2) 一边跑语音助手的长输出，一边打字幕请求
    stop = threading.Event()

    def noise() -> None:
        scene = SCENES["ocr-long"]
        while not stop.is_set():
            msgs = build_messages(scene, lang=lang, rep=0)
            one(base, model, key, msgs, 160)

    th = threading.Thread(target=noise, daemon=True)
    th.start()
    time.sleep(2)
    busy_out = []
    for i in range(n):
        sub_caption(i, busy_out)
    stop.set()
    th.join(timeout=30)
    print(f"  并发时字幕首字 {[t for _, t, _ in busy_out]}")
    a = [t for _, t, _ in base_out if t >= 0]
    b = [t for _, t, _ in busy_out if t >= 0]
    if a and b:
        print(f"  → 基线 p50 {int(statistics.median(a))}ms  "
              f"并发 p50 {int(statistics.median(b))}ms  "
              f"劣化 {int(statistics.median(b)) - int(statistics.median(a))}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="名字|base|model|key，可重复")
    ap.add_argument("--only", action="append", default=[],
                    help="只跑某个场景，可重复；默认全跑")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=160,
                    help="与 voice_server 的 max_tokens 一致")
    ap.add_argument("--gap", type=float, default=0.6,
                    help="同场景两次之间的间隔，避免首轮冷启动混进中位数")
    ap.add_argument("--out", default="")
    ap.add_argument("--reply-lang", default="zh", choices=["zh", "ja", "auto", "none"],
                    help="与 voice_server --reply-lang 一致；线上是 zh。"
                         "none 表示不加语言指令，用来看模型自己的倾向")
    ap.add_argument("--concurrent", action="store_true",
                    help="额外做「助手打满时字幕首字劣化」测试")
    ap.add_argument("--sub-base", default="http://127.0.0.1:10100/inference/v1")
    ap.add_argument("--sub-model", default="")
    ap.add_argument("--sub-key", default="")
    args = ap.parse_args()

    if not args.arm:
        print(__doc__)
        return
    scenes = args.only or list(SCENES)

    result = {"scenes": scenes, "reply_lang": args.reply_lang, "arms": {}}
    for spec in args.arm:
        parts = (spec.split("|") + ["", "", ""])[:4]
        name, base, model, key = parts
        result["arms"][name] = run_arm(name, base, model, key, scenes,
                                       args.reps, args.max_tokens, args.gap,
                                       args.reply_lang)

    print("\n########## 汇总（首字 p50 / 全文 p50 / 输出字数 p50 / 语言遵循）")
    hdr = f"{'场景':<12}" + "".join(f"{n:>26}" for n in result["arms"])
    print(hdr)
    for sc in scenes:
        line = f"{sc:<12}"
        for n, rows in result["arms"].items():
            r = rows[sc]
            flag = "!" if r["over_1s"] else " "
            line += f"{r['ttft_p50']:>8}ms{r['total_p50']:>7}ms{r['chars_p50']:>4}字{flag}"
            line += f"{r['lang_ok']:>2}/{r['n']} 语言"
        print(line)
    print("\n! = 该场景有超过 1 秒的首字（语音助手体感线）")
    print(f"语言遵循 = 输出语言等于 reply-lang({args.reply_lang}) 的条数")

    if args.concurrent and args.arm:
        parts = (args.arm[0].split("|") + ["", "", ""])[:4]
        concurrency_probe(parts[1], parts[2], parts[3],
                          args.sub_base, args.sub_model or parts[2],
                          args.sub_key or parts[3], lang=args.reply_lang)

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
