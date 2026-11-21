#!/usr/bin/env python3
"""voice_lang_fix.py —— 语言指令到底该怎么写，模型才真的听。

背景：voice_server 的 system 通篇是日语（"あなたは日本語と中国語のバイリンガル…"），
只在末尾追加一句「本次回答必须使用简体中文」。实测（README 4.22）：

    reply-lang=zh   7B：8/12 中文      35B：0/12 中文（全回日语）

7B 是机械跟随最后一句指令，35B 反而「太懂语境」——用户说日语、屏幕是日语，
它就无视指令回日语。语音助手的价值就在「日语会议里用中文播报」，所以这条不通
就不能换模型。

这里只测「怎么把话说明白」，不测速度。四种写法：
    sys-tail   现状：system 末尾追加中文指令
    user-tail  追加到最后一条 user 消息末尾（离生成位置最近，指令权重最高）
    sys-hard   system 开头就声明语言，末尾再强调一次
    both       user-tail + sys-hard 一起上

判据只有一个：输出语言正确率。速度都是几百毫秒的差别，语言错了就全废。
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import voice_llm_ab as W  # noqa: E402

ZH_SOFT = "\n本次回答必须使用简体中文，不要使用日语。"
ZH_USER = "\n\n（回答语言：简体中文。不要用日语。）"
ZH_HARD_PREFIX = "重要：无论用户用哪种语言提问、屏幕内容是什么语言，你的回答必须始终使用简体中文。\n"


def build(mode: str, scene: dict, rep: int) -> list[dict]:
    msgs = W.build_messages(scene, lang="none", rep=rep)  # 先不加任何语言指令
    if mode in ("sys-tail", "sys-hard", "both"):
        msgs[0]["content"] = msgs[0]["content"] + ZH_SOFT
    if mode == "sys-hard":
        msgs[0]["content"] = ZH_HARD_PREFIX + msgs[0]["content"]
    if mode in ("user-tail", "both"):
        msgs[-1]["content"] = msgs[-1]["content"] + ZH_USER
    if mode == "both":
        msgs[0]["content"] = ZH_HARD_PREFIX + msgs[0]["content"]
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="名字|base|model|key，可重复")
    ap.add_argument("--scene", action="append", default=["chat-multi", "ocr-long"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--modes", default="sys-tail,user-tail,sys-hard,both")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    modes = args.modes.split(",")
    summary = {}
    for spec in args.arm:
        name, base, model, key = (spec.split("|") + ["", "", ""])[:4]
        print(f"\n########## {name}   {model}")
        got = {}
        for mode in modes:
            ok = tot = 0
            samples = []
            for sc in args.scene:
                for i in range(args.reps):
                    msgs = build(mode, W.SCENES[sc], i)
                    r = W.one(base, model, key, msgs, args.max_tokens)
                    lg = W.lang_of(r["text"])
                    tot += 1
                    ok += lg == "zh"
                    samples.append(f"[{lg}]{r['text'][:30]}")
                    time.sleep(0.4)
            got[mode] = dict(ok=ok, tot=tot, samples=samples)
            print(f"  {mode:<10} 中文 {ok}/{tot}   " + " | ".join(samples[:2]))
        summary[name] = got

    print("\n########## 中文正确率")
    print(f"{'写法':<12}" + "".join(f"{n:>14}" for n in summary))
    for mode in modes:
        line = f"{mode:<12}"
        for name, got in summary.items():
            g = got[mode]
            line += f"{g['ok']:>7}/{g['tot']:<6}"
        print(line)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
