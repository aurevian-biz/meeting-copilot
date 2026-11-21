#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""voice_lang_fix.py -- how the language directive should be worded so the model actually obeys.

Background: voice_server's system prompt is Japanese throughout ("あなたは日本語と中国語のバイリンガル…"),
with a single line appended at the end: "this answer must be in Simplified Chinese". Measured (README 4.22):

    reply-lang=zh   7B: 8/12 in Chinese      35B: 0/12 in Chinese (all answered in Japanese)

7B mechanically follows the last instruction, while 35B is "too good at context" -- the user speaks Japanese and the screen is Japanese,
so it ignores the directive and answers in Japanese. The whole value of the voice assistant is "broadcast in Chinese during a Japanese meeting", so if this fails
the model cannot be swapped in.

This only measures "how to say it clearly", not speed. Four wordings:
    sys-tail   current state: append the Chinese directive at the end of system
    user-tail  append it to the end of the last user message (closest to the generation point, highest instruction weight)
    sys-hard   declare the language at the start of system and stress it once more at the end
    both       user-tail + sys-hard together

There is only one criterion: output language accuracy. The speed differences are all a few hundred ms; getting the language wrong ruins everything.
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
    msgs = W.build_messages(scene, lang="none", rep=rep)  # start with no language directive at all
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
