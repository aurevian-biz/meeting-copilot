#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Translation quality comparison: old prompt vs new prompt on the same set of real bad cases.

Usage (run in the deployment directory, vLLM must be up):
    ./asr-venv/bin/python tools/tr_quality.py
    ./asr-venv/bin/python tools/tr_quality.py --only new

Every case comes from a bad output caught in measurement (Chinese bleeding in, verbatim copying, uncorrected typos);
the criterion is "lang_clean passes + a human can read it", not speed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caption_server as cs  # noqa: E402

OLD_SYS = {
    "zh": ("你是会议同传。把用户给的这句话翻译成中文。\n硬性要求：\n"
           "1. 只输出译文本身，不要引号、不要解释、不要补充，不要输出原文；\n"
           "2. 译文必须是纯中文，一个原文词都不许留（专有名词除外）；\n"
           "3. 原文是语音识别结果，可能有错字、断句不完整，请按最合理的语义翻译，"
           "半句就译成半句，不要自行补完；\n"
           "4. 如果给了【上文】，译文要与上文的人称、术语、语气保持一致；\n"
           "5. 原文里的语气词（嗯、呃、えーと、あの、um 等）和说到一半的重来直接忽略，"
           "不要把语气词译成实词；但有实义的词要保留。"),
    "ja": ("你是会议同传。把用户给的这句话翻译成日语。\n硬性要求：\n"
           "1. 只输出译文本身，不要引号、不要解释、不要补充，不要输出原文；\n"
           "2. 译文必须是纯日语，一个原文词都不许留（专有名词除外）；\n"
           "3. 原文是语音识别结果，可能有错字、断句不完整，请按最合理的语义翻译，"
           "半句就译成半句，不要自行补完；\n"
           "4. 如果给了【上文】，译文要与上文的人称、术语、语气保持一致；\n"
           "5. 原文里的语气词（嗯、呃、えーと、あの、um 等）和说到一半的重来直接忽略，"
           "不要把语气词译成实词；但有实义的词要保留。"),
}

# (target language, source, context)  -- the first 3 are literal leftovers caught in measurement, the rest are normal sentences for regression
CASES = [
    ("zh", "身体面にアメリカは坚持するべきです", ""),
    ("zh", "深zのアプリで何点给深zに評価するか", ""),
    ("zh", "定め好了、王毅は中国大陸の外務である", ""),
    ("zh", "台湾問題を取引材料にするのか", ""),
    ("zh", "中方科意、韓国に良いところを与える", "日本は冷遇されている"),
    ("ja", "我是主持人谭英文", ""),
    ("ja", "中方刻意给韩国好处", ""),
    ("ja", "所以去扯什么台湾会不会进行交易筹码？", ""),
    ("ja", "这个都没有啊，怎么可能第一次就有结论，那不可能的事。", ""),
    ("ja", "三强会晤，三角的峰会我不认为。", "中方刻意的给了韩国好处"),
    # The 4 below come from end-to-end output on 62 seconds of real audio: verbatim copies that 7B fails to suppress
    ("ja", "他比较晚了，有开了等一长会议又开了开了会之后才有。", ""),
    ("ja", "这就投扯太远的了，根本不可能在这种地方提出了。", ""),
    ("ja", "了不起在中美会晤中国大陆再讲一下啊，我认为就是这样了。", ""),
    ("ja", "因为他的很容易跟韩国做对比是。", ""),
]

# Repeat the most critical line at the end of the user message: closest to the generation point, and small models are very sensitive to instruction position
TAIL = {
    "ja": "\n\n只输出日语译文。禁止出现任何中文词、中文语气词或英文字句。",
    "zh": "\n\n只输出中文译文。禁止出现任何日文词或假名。",
}


def call(base: str, model: str, sysp: str, text: str, target: str, ctx: str,
         tail: str = "") -> tuple[str, int]:
    import json
    import urllib.request

    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    if tail:
        user += tail
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": user}],
        "stream": False, "temperature": 0.1, "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read())
        out = (d["choices"][0]["message"]["content"] or "").strip()
        return out.strip('"“”「」 '), int((time.time() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        return f"<ERROR {e}>", int((time.time() - t0) * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--variant", choices=["old", "new", "tail"],
                    help="只跑某一侧；不传则跑 new / tail 两组")
    args = ap.parse_args()

    sides = [args.variant] if args.variant else ["new", "tail"]
    dirty = {s: 0 for s in sides}
    for target, text, ctx in CASES:
        print(f"\n{'=' * 66}\n[{target}] {text}" + (f"    上文：{ctx}" if ctx else ""))
        for side in sides:
            if side == "old":
                sysp, tail = OLD_SYS[target], ""
            elif side == "new":
                sysp, tail = cs._tr_system(target), ""
            else:
                sysp, tail = cs._tr_system(target), TAIL[target]
            out, ms = call(args.base, args.model, sysp, text, target, ctx, tail)
            clean = cs.lang_clean(out, target)
            if not clean:
                dirty[side] += 1
            print(f"  {side:4s} {'OK  ' if clean else '脏  '}{ms:5d}ms  {out}")

    print(f"\n{'=' * 66}")
    print("残留源语言的句子数：" + "   ".join(f"{s} {dirty[s]}/{len(CASES)}" for s in sides))
    return 0


if __name__ == "__main__":
    sys.exit(main())
