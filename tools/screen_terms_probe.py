#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Can the screen term hints match the ASR's misrecognitions? -- measure the hit rate and false positives first, then decide whether the feature is worth building.

Idea: if a proper noun on screen (person/institution/model number/digit) gets heard by ASR as a homophone,
then **the two readings are highly similar while the spellings differ**. So:

  1. Extract candidate terms from the OCR (Latin/digits/katakana/kanji runs), filtering out our own page and the existing translation
  2. Take the reading of each candidate (pyopenjtalk)
  3. Chop each ASR segment into morphemes and take reading windows of 1-4 morphemes
  4. Reading edit distance <=1 with a different spelling -> candidate hit

A low hit rate or many false positives means the "inject screen term hints" route does not hold in this scenario,
so stop spending time on prompt engineering (measured cost 250-320ms per sentence, see tools/screen_hint_ab.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

KATA = re.compile(r"[ァ-ヴー]{4,12}")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9.&_\-]{1,20}")
NUM = re.compile(r"[0-9][0-9,.]{1,10}[兆億万%]?")
KANJI = re.compile(r"[一-龥]{3,10}")

UI_WORDS = ("会议字幕", "设备", "已识别", "延迟", "首字", "译完", "双语", "正在听",
            "停止", "清屏", "正在说", "运行中", "搜索", "创建", "Chrome",
            "youtube.com", "dB")


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", required=True)
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    import pyopenjtalk
    import time

    lines = json.load(open(args.poll, encoding="utf-8"))["lines"]
    ocr_text = (json.load(open(args.ocr, encoding="utf-8")).get("text") or "")
    recent = []
    for l in lines[-30:]:
        recent += [l.get("text") or "", l.get("trans") or ""]

    # ---------- extract screen candidates ----------
    raw = [x.strip() for x in ocr_text.splitlines() if x.strip()]
    kept = []
    for ln in raw:
        if len(ln) < 3 or any(w in ln for w in UI_WORDS):
            continue
        if any(len(r) >= 8 and r[:8] in ln for r in recent if r):
            continue
        kept.append(ln)
    print(f"屏幕行：{len(raw)} → 过滤后 {len(kept)}")
    for x in kept[:10]:
        print("   ", x[:72])

    cands: dict[str, str] = {}
    for ln in kept:
        for m in list(LATIN.finditer(ln)) + list(NUM.finditer(ln)) \
                + list(KATA.finditer(ln)) + list(KANJI.finditer(ln)):
            w = m.group()
            if w in cands or len(w) < 3:
                continue
            try:
                rd = kata2hira(pyopenjtalk.g2p(w).replace(" ", ""))
                rd = re.sub(r"[^ぁ-ゖー]", "", rd)
            except Exception:  # noqa: BLE001
                continue
            if len(rd) < 4:
                continue
            cands[w] = rd
    print(f"\n候选词 {len(cands)} 个（读音≥4 拍）：")
    for w, rd in list(cands.items())[:24]:
        print(f"   {w:<16} {rd}")

    # ---------- match segment by segment ----------
    print("\n########## 命中检查")
    t_all = time.time()
    n_seg = hits = 0
    shown = 0
    for l in lines[-args.limit:]:
        text = l.get("text") or ""
        if len(text) < 6:
            continue
        n_seg += 1
        t0 = time.time()
        nodes = pyopenjtalk.run_frontend(text)
        fe_ms = int((time.time() - t0) * 1000)
        toks = [(n["string"], kata2hira(n["read"] or "")) for n in nodes]
        # merge window: 1-4 morphemes
        wins = []
        for i in range(len(toks)):
            surf, rd = "", ""
            for j in range(i, min(i + 4, len(toks))):
                surf += toks[j][0]
                rd += toks[j][1]
                if len(rd) >= 12:
                    break
                if len(rd) >= 4:
                    wins.append((surf, rd))
        for w, rd in cands.items():
            for surf, wrd in wins:
                if surf == w:
                    continue
                if abs(len(wrd) - len(rd)) > 1:
                    continue
                if lev(wrd, rd) <= 1:
                    hits += 1
                    if shown < args.show:
                        shown += 1
                        print(f"  [{l['seq']}] 屏幕「{w}」({rd})  ≈  ASR「{surf}」({wrd})"
                              f"  frontend={fe_ms}ms")
                    break
    print(f"\n段数 {n_seg}，命中 {hits} 条；frontend 总耗时 {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
