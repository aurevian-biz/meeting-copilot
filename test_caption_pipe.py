#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Regression tests for the caption latency rework: two-stage output, decoupled queue,
context carry-over, short segments.

ASR and translation are both stubbed, so neither a model nor vLLM is needed — runs on a laptop or a Jetson:

    ./asr-venv/bin/python test_caption_pipe.py
    python test_caption_pipe.py            # local machine (requires numpy)

What is verified is the "structure": the source text must appear before the translation, and
the capture thread must never be blocked by ASR/translation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_server as cs  # noqa: E402

SR = cs.SR
FRAME_N = cs.FRAME_N
calls: list[dict] = []          # call log of fake_translate, shared between cases


# ---------------------------------------------------------------- stubs
class _Result:
    text = "<|zh|>测试一句话。"


class _Stream:
    result = _Result()

    def accept_waveform(self, sr, x):
        pass


class _Rec:
    def create_stream(self):
        return _Stream()

    def decode_stream(self, s):
        return None


class FakeAsr:
    rec = _Rec()


def fake_translate(base, model, text, target, timeout=12.0, ctx=""):
    calls.append({"ctx": ctx, "text": text, "target": target})
    time.sleep(0.25)                       # simulate a 250ms translation
    return "テスト文"


stream_calls: list[dict] = []


def fake_translate_stream(base, model, text, target, timeout=12.0, ctx="",
                          on_delta=None, screen_hint=""):
    """Simulates streaming: emits two deltas first, then the full text. Fast first token, slow full sentence."""
    stream_calls.append({"ctx": ctx, "text": text, "target": target,
                         "screen_hint": screen_hint})
    t0 = time.time()
    pieces = ["テスト", "文"]
    for i, p in enumerate(pieces, 1):
        time.sleep(0.09)     # > 80ms throttle window, so both deltas are really pushed out
        if on_delta is not None:
            on_delta("".join(pieces[:i]))
    ttft = int(90)
    return "".join(pieces), ttft, int((time.time() - t0) * 1000)


def ok(cond, msg, detail=""):
    if not cond:
        print(f"[FAIL] {msg}  {detail}")
        raise SystemExit(1)
    print(f"[ok]   {msg}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- cases
def test_two_stage():
    """Source text comes out first, translation is added later, and it backfills the same line."""
    cap = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=2, ctx_lines=2)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)

    t0 = time.time()
    cap.submit(pcm, 1.0, end_ts=t0)
    for _ in range(300):
        if cap.lines:
            break
        time.sleep(0.005)
    t_text = (time.time() - t0) * 1000
    ok(bool(cap.lines), "识别线程产出原文行")
    line = cap.lines[0]
    ok(line["trans"] == "" and line["pending"] is True,
       "原文行先出来时译文还是空的", f"pending={line['pending']}")
    ok(t_text < 150, "原文不用等翻译", f"{t_text:.0f}ms")

    seen, final = [], None
    t1 = time.time()
    while (time.time() - t1) < 3:
        for u in cap.take_updates():
            seen.append(u)
            if u.get("final"):
                final = u
        if final:
            break
        time.sleep(0.005)
    ok(final is not None, "译文以 final 补丁形式到达", f"共 {len(seen)} 条补丁")
    ok(final["seq"] == line["seq"] and final["trans"] == "テスト文",
       "补丁指向同一 seq", json.dumps(final, ensure_ascii=False))
    ok(line["trans"] == "テスト文" and line["pending"] is False,
       "行对象被就地回填（历史/新开页面也完整）")
    ok(cap.take_updates() == [], "补丁取走即清空，不重复下发")


def test_stream_trans():
    """Streaming: repeated pushes for the same seq merge, intermediate states are prefixes of the final
    translation, and only final carries timings."""
    cap = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1)
    cap._push_update(7, "テ", False)
    cap._push_update(7, "テスト", False)
    cap._push_update(7, "テスト文", True, 300, 96)
    upd = cap.take_updates()
    ok(len(upd) == 1, "同一 seq 多次推送合并成一条（250ms 轮询才不会白重绘）",
       f"{len(upd)} 条")
    ok(upd[0]["trans"] == "テスト文" and upd[0]["final"] and upd[0]["ttft_ms"] == 96,
       "合并后留下的是 final 版本，带首字耗时")

    cap2 = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1)
    cap2.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    cap2.submit(pcm, 1.0, end_ts=time.time())
    seen: list[dict] = []
    t0 = time.time()
    while (time.time() - t0) < 4:
        seen.extend(cap2.take_updates())
        if seen and seen[-1].get("final"):
            break
        time.sleep(0.005)
    ok(bool(seen), "流式补丁有下发")
    ok(all("テスト文".startswith(u["trans"]) for u in seen),
       "每条补丁都是最终译文的递增前缀（前端可以直接整行替换）",
       " | ".join(u["trans"] for u in seen))
    mids = [u for u in seen if not u["final"]]
    ok(all(u["trans_ms"] == 0 for u in mids), "中间态不带耗时，避免页面上数字乱跳")
    ok(seen[-1].get("final") and seen[-1]["ttft_ms"] > 0, "final 带首字耗时")
    ok(cap2.lines[0]["pending"] is False and cap2.lines[0]["trans"] == "テスト文",
       "行对象最终态正确")

    cap3 = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1, stream_trans=False)
    ok(cap3.stream_trans is False, "stream_trans=False 可退回整句出")


def test_ctx_and_order():
    """The second sentence carries the first as context; when translations back up, they are filled in order."""
    cap = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1, ctx_lines=2)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    for _ in range(3):
        cap.submit(pcm, 1.0, end_ts=time.time())
    t1 = time.time()
    while (time.time() - t1) < 6 and sum(1 for l in cap.lines if l["trans"]) < 3:
        time.sleep(0.01)
    ok(len(cap.lines) == 3, "三段都进了队列并被识别", f"{len(cap.lines)} 段")
    ok(all(l["trans"] for l in cap.lines), "译文全部补齐")
    # the streaming path records into stream_calls, the whole-sentence path into calls; take whichever ran
    last = (stream_calls or calls)[-1]
    ok(last["ctx"].startswith("测试一句话"), "翻译带上了上文",
       repr(last["ctx"][:20]))


def test_submit_not_blocked():
    """Submitting a segment from the capture thread must not be blocked by ASR/translation."""
    cap = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    t = time.time()
    for _ in range(20):
        cap.submit(pcm, 1.0, end_ts=time.time())
    dt = (time.time() - t) * 1000
    ok(dt < 100, "连投 20 段立刻返回", f"{dt:.1f}ms")


def test_queue_overflow_drops_oldest():
    """A full queue drops the oldest segment rather than growing without bound."""
    a = cs.Captioner(FakeAsr(), bilingual=False)
    a._started = True                       # don't start worker threads, so it just accumulates
    pcm = np.zeros(FRAME_N, dtype=np.int16)
    for _ in range(a._q_asr.maxsize + 5):
        a.submit(pcm, 0.02, end_ts=time.time())
    ok(a._q_asr.qsize() == a._q_asr.maxsize, "队列封顶", f"qsize={a._q_asr.qsize()}")
    ok(a.dropped == 5, "丢弃计数", f"dropped={a.dropped}")


def test_segment_length():
    """max_s=3.0 takes effect: continuous speech is capped at 3 seconds per segment."""
    seg = cs.Segmenter(margin_db=8.0, hold_s=0.42, max_s=3.0, min_s=0.25)
    frame = (np.random.randn(FRAME_N) * 400).astype(np.int16)
    lens = []
    for i in range(400):                    # 8 seconds of continuous voice activity
        db = -20.0
        out = seg.push(frame, db)
        if out is not None:
            lens.append(out.size / SR)
        # the first 1.2 seconds are noise-floor warm-up and are not judged as speech
    ok(len(lens) >= 2, "连续说话被强制切段", f"{[round(x,2) for x in lens]}")
    ok(max(lens) <= 3.0 + 1e-6, "单段不超过 3 秒", f"max={max(lens):.2f}s")
    ok(all(x >= 0.25 for x in lens), "不产出过短的碎段")


def test_soft_cut_at_dip():
    """Before a hard cut, back off to an energy dip: segment length is no longer always exactly max_s."""
    seg = cs.Segmenter(margin_db=8.0, hold_s=0.42, max_s=3.0, min_s=0.25, soft_s=0.7)
    frame = (np.random.randn(FRAME_N) * 400).astype(np.int16)
    lens = []
    for i in range(400):
        # place 2 deep-dip frames every 0.5s; that never reaches the 21-frame silence hang, so only a hard cut fires
        db = -62.0 if (i % 25) < 2 else -20.0
        out = seg.push(frame, db)
        if out is not None:
            lens.append(out.size / SR)
    ok(len(lens) >= 2, "连续有声被切成多段", f"{[round(x,2) for x in lens][:6]}")
    ok(max(lens) <= 3.0 + 1e-6, "不超过硬上限 3 秒", f"max={max(lens):.2f}s")
    ok(min(lens) < 2.95, "切点回退到了低谷附近", f"min={min(lens):.2f}s")


def _pump(seg, frame, voiced_n, silent_n, db_v=-20.0, db_s=-70.0):
    """Speak for voiced_n frames, then stay silent for silent_n frames; collect every segment produced meanwhile."""
    outs = []
    for _ in range(voiced_n):
        r = seg.push(frame, db_v)
        if r is not None:
            outs.append(r)
    for _ in range(silent_n):
        r = seg.push(frame, db_s)
        if r is not None:
            outs.append(r)
    return outs


def test_hold_layering():
    """Layered pause thresholds: while a segment is short only long pauses count;
    once it is long enough, a short pause already cuts."""
    frame = (np.random.randn(FRAME_N) * 400).astype(np.int16)

    def fresh():
        seg = cs.Segmenter(margin_db=8.0, hold_s=0.45, max_s=4.5, min_s=0.25,
                           soft_s=0.7, hold_long_s=0.90, hold_short_s=0.30,
                           mid_s=1.20, long_s=2.40)
        for _ in range(int(1.2 / cs.FRAME_S) + 2):      # warm-up, to estimate the noise floor
            seg.push(frame, -70.0)
        return seg

    # 0.6s utterance + 0.5s pause: 0.5 < hold_long(0.9), so no cut allowed.
    # This is exactly what used to split "…况且140。" + "亿美元对台均售…" at the digits.
    seg = fresh()
    outs = _pump(seg, frame, 30, 25)
    ok(not outs, "短段遇 0.5s 停顿不该断句", f"切出了 {len(outs)} 段")

    # same segment; once silence holds past 0.9s it must cut
    outs = _pump(seg, frame, 0, 25)
    ok(len(outs) == 1, "静音够 0.9s 就断句", f"{len(outs)} 段")
    if outs:
        dur = outs[0].size / SR
        ok(1.2 < dur < 2.2, "段长≈0.6 有声 + 0.9 静音 + 预补", f"{dur:.2f}s")

    # 3-second utterance + 0.32s pause: the segment already exceeds seg_long, so cut at once per hold_short
    seg = fresh()
    outs = _pump(seg, frame, 150, 16)
    ok(len(outs) == 1, "长段遇短停顿就切，延迟不再往上堆", f"{len(outs)} 段")
    ok(seg.last_cut == "silence", "记录成因是自然停顿，不是撞上限", seg.last_cut)

    # 3-second utterance + 0.2s pause (below hold_short): still no cut
    seg = fresh()
    outs = _pump(seg, frame, 150, 10)
    ok(not outs, "0.2s 停顿短于 hold_short，不切", f"{len(outs)} 段")

    # when the silence threshold is set shorter than the base, the long-segment threshold must not
    # end up below the base either (fallback for swapped parameters)
    seg = cs.Segmenter(margin_db=8.0, hold_s=0.45, max_s=4.5, min_s=0.25,
                       hold_long_s=0.10, hold_short_s=2.0)
    ok(seg.hold_long_frames >= seg.hold_frames, "hold_long 不小于基准")
    ok(seg.hold_short_frames <= seg.hold_frames, "hold_short 不大于基准")


def test_noise_segment_dropped():
    """An ASR result that is only punctuation is not a sentence and must not occupy a caption line."""
    ok(not cs.SUBSTANCE.search(" 。"), "纯标点丢弃")
    ok(not cs.SUBSTANCE.search("  "), "空白丢弃")
    ok(bool(cs.SUBSTANCE.search("嗯。")), "单个实词保留")
    ok(bool(cs.SUBSTANCE.search("140")), "纯数字保留")
    ok(bool(cs.SUBSTANCE.search("はい")), "假名保留")


def test_lang_clean():
    """Translation residue detection: kana must always hit; Han characters shared by Chinese and
    Japanese must never be false positives."""
    ok(not cs.lang_clean("美国应该坚持するべきです", "zh"), "日译中残留假名被检出")
    ok(cs.lang_clean("在实质层面，美国应当坚持立场。", "zh"), "干净中文放过")
    # proper nouns may keep their Japanese spelling; a few kana must not trigger a retry
    ok(cs.lang_clean("特别是深圳（深セン）被称为中国的硅谷。", "zh"), "专有名词少量假名放过")
    ok(not cs.lang_clean("美国应该坚持するべきです", "zh"), "整句假名仍判脏")
    ok(not cs.lang_clean("中国側は故意に韓国に给好处", "ja"), "中译日残留简体被检出")
    ok(cs.lang_clean("中国側は意図的に韓国に便宜を図った。", "ja"), "干净日语放过")
    # these three Han characters are shared by Chinese and Japanese; adding them to the list would
    # flag healthy translations as dirty and waste a whole extra translation call
    ok(cs.lang_clean("会議は都会で開かれる。", "ja"), "「会」「都」不误伤")
    # when a whole sentence is copied from Chinese, every character is also a Han character used in
    # Japanese, so only "does it contain kana" can decide
    ok(not cs.lang_clean("我是主持人田英和", "ja"), "纯汉字整句判脏（没翻译）")
    ok(cs.lang_clean("私は司会の谭英文です。", "ja"), "带假名的正常译文放过")
    ok(cs.lang_clean("了解", "ja"), "短句不受无假名规则影响")
    ok(cs.lang_clean("その刹那の話。", "ja"), "「那」不误伤")
    ok(cs.lang_clean("anything goes", "en"), "没有判据的语言直接放过")


def asr_saying(text: str):
    """ASR stand-in that always emits one sentence, used to verify "source stays verbatim, cleaning happens before translation"."""
    result = type("R", (), {"text": text})()
    stream = type("S", (), {"result": result,
                            "accept_waveform": lambda self, sr, x: None})()
    rec = type("Rec", (), {"create_stream": lambda self: stream,
                           "decode_stream": lambda self, s: None})()
    return type("A", (), {"rec": rec})()


def test_strip_fillers():
    """Filler-removal boundaries: only pure interjections are deleted; content words and empty results must survive."""
    ok(cs.strip_fillers("嗯，那个项目怎么样了") == "那个项目怎么样了", "句首语气词删掉")
    ok(cs.strip_fillers("我觉得那个不行") == "我觉得那个不行", "「那个」有实义，不动")
    ok(cs.strip_fillers("嗯嗯，呃，好的") == "好的", "连续语气词全清，标点不残留")
    ok(cs.strip_fillers("嗯。") == "嗯。", "整句都是语气词时退回原文，不返回空")
    ok(cs.strip_fillers("") == "", "空串安全")
    ok(cs.strip_fillers("えーと、それでですね") == "それでですね", "日语填充词")
    ok(cs.strip_fillers("あの人は誰ですか") == "あの人は誰ですか", "「あの+名词」不动")
    ok(cs.strip_fillers("That is her eraser") == "That is her eraser", "英文不误伤词内字母")
    ok(cs.strip_fillers("嗯，今天天气不错，呃，我们出门吧") == "今天天气不错，我们出门吧",
       "句中语气词也清，句义完整")


def test_fillers_only_for_translation():
    """The source text displayed on the page stays verbatim; only what is sent to translation gets cleaned."""
    calls.clear()                       # threads from earlier cases are still running, so clear the log first
    stream_calls.clear()
    cap = cs.Captioner(asr_saying("<|zh|>嗯，那个项目怎么样了。"), bilingual=True,
                       tr_workers=1)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    cap.submit(pcm, 1.0, end_ts=time.time())
    t1 = time.time()
    while (time.time() - t1) < 3 and not (cap.lines and cap.lines[0]["trans"]):
        time.sleep(0.01)
    ok(bool(cap.lines), "产出一行字幕")
    ok(cap.lines[0]["text"] == "嗯，那个项目怎么样了。",
       "显示用的原文仍带语气词", repr(cap.lines[0]["text"]))
    rec = stream_calls or calls          # streaming is the default path
    ok(any(c["text"] == "那个项目怎么样了。" for c in rec),
       "翻译拿到的是清洗后的文本", repr([c["text"] for c in rec][-3:]))
    ok(not any(c["text"].startswith("嗯") for c in rec),
       "没有任何一次翻译收到带语气词的文本")


def test_clear_resets():
    cap = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    cap.submit(pcm, 1.0, end_ts=time.time())
    for _ in range(200):
        if cap.lines and cap.lines[0]["trans"]:
            break
        time.sleep(0.01)
    cap.clear()
    ok(len(cap.lines) == 0 and cap.take_updates() == [], "清屏连补丁一起清")


def test_segmenter_draft():
    """The segmenter must be able to hand out "the current segment that is not cut yet" —
    the raw material for a draft caption."""
    seg = cs.Segmenter()
    ok(seg.draft() is None, "没声音时没有草稿")
    frame = (np.random.randn(FRAME_N) * 300).astype(np.int16)
    for _ in range(100):                     # 2 seconds: the first 60 frames are warm-up, segments start after that
        seg.push(frame, -20.0)
    d = seg.draft()
    ok(d is not None and d[0].size > 0 and d[1] > 0.5, "有声时能拿到草稿音频",
       f"{d[1]:.2f}s" if d else "None")


def test_draft_caption():
    """Grey text appears first while speaking; once the segment is finalized the draft is cleared
    immediately (the real line takes over)."""
    cap = cs.Captioner(FakeAsr(), bilingual=True, draft_on=True)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)

    cap.push_draft(pcm, 1.0)
    for _ in range(100):
        if cap.draft_state()["text"]:
            break
        time.sleep(0.01)
    st = cap.draft_state()
    ok(st["text"] != "", "草稿能先出字", st["text"])

    ep = st["epoch"]
    cap.submit(pcm, 1.0, end_ts=time.time())
    time.sleep(0.02)
    st2 = cap.draft_state()
    ok(st2["text"] == "" and st2["epoch"] == ep + 1,
       "定稿后草稿清空、epoch 前进（前端据此收起灰行）")


def test_draft_stale_dropped():
    """A draft slower than finalization must be dropped, otherwise the grey text would be drawn
    over content that is already final."""

    class SlowAsr:
        class rec:
            @staticmethod
            def create_stream():
                time.sleep(0.25)             # simulate recognition slower than segmentation
                return _Stream()

            @staticmethod
            def decode_stream(s):
                return None

    cap = cs.Captioner(SlowAsr(), bilingual=True, draft_on=True)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)

    cap.push_draft(pcm, 1.0)
    time.sleep(0.05)                         # the draft thread is already inside recognize (needs 0.25s)
    cap.submit(pcm, 1.0, end_ts=time.time())
    time.sleep(0.6)
    ok(cap.draft_state()["text"] == "", "过期草稿不落到屏幕上")


def test_no_draft():
    cap = cs.Captioner(FakeAsr(), bilingual=True, draft_on=False)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    cap.push_draft(pcm, 1.0)
    time.sleep(0.2)
    ok(cap.draft_state()["text"] == "" and cap.drafts == 0,
       "--no-draft 完全不跑草稿识别")


def test_ctx_lines_zero():
    """ctx_lines=0 must truly carry no context.

    When the slice is written as `prev[-self.ctx_lines:]`, 0 turns into `[0:]` and treats **the
    whole history** as context; measured on the 35B model this grew the prompt from 80 to 138
    tokens and cost 300ms extra on the first token.
    """
    cap = cs.Captioner(FakeAsr(), bilingual=True, ctx_lines=0)
    for i in range(1, 5):
        cap.lines.append({"seq": i, "text": f"第{i}句"})
    ok(cap._ctx_before(5) == "", "ctx_lines=0 不带上文", cap._ctx_before(5))

    cap2 = cs.Captioner(FakeAsr(), bilingual=True, ctx_lines=2)
    for i in range(1, 5):
        cap2.lines.append({"seq": i, "text": f"第{i}句"})
    got = cap2._ctx_before(5)
    ok(got == "第3句 / 第4句", "ctx_lines=2 只取最近两句", got)


def test_tr_prompt_styles():
    """Both prompt styles must exist and be selectable via --tr-prompt / TR_STYLE.

    The short version was added in 4.20: it keeps only the three hard rules and drops the examples,
    so its length must really shrink (otherwise nothing changed), but it must not shrink so far
    that the "target language only" rule is lost.
    """
    for target in ("zh", "ja"):
        full = cs._tr_system(target, "full")
        short = cs._tr_system(target, "short")
        ok(short != full, f"{target} 两档不同", short[:20])
        ok(len(short) < 140, f"{target} short 确实短（{len(short)} 字）", len(short))
        ok(len(full) > len(short) * 2, f"{target} full 至少两倍长", len(full))
        # not a single keyword of the three hard rules may be missing
        for kw in ("【本句】", "【上文】", "半句"):
            ok(kw in short, f"{target} short 含 {kw}", short)

    # module-level switch: when style is not passed, TR_STYLE decides
    old = cs.TR_STYLE
    try:
        cs.TR_STYLE = "short"
        ok(cs._tr_system("ja") == cs._tr_system("ja", "short"), "TR_STYLE 生效")
        cs.TR_STYLE = "full"
        ok(cs._tr_system("ja") == cs._tr_system("ja", "full"), "TR_STYLE 复原")
    finally:
        cs.TR_STYLE = old


def test_screen_hints():
    """Screen term hints: only triggered by "similar reading, different spelling"; a hit passes a hint, a miss costs nothing."""
    # ---- reading helpers
    ok(cs._hira("ハンズイ") == "はんずい", "_hira 片假名转平假名")
    ok(cs._hira("ABC 12、") == "", "_hira 丢掉拉丁/数字/标点")
    ok(cs._lev("はんずい", "はんずい") == 0, "_lev 相同为 0")
    ok(cs._lev("はんずい", "はんすい") == 1, "_lev 差一个音为 1")
    ok(cs._lev("はんずい", "はん") == 9, "_lev 长度差过大直接判 9")

    # ---- fake reading engine: no pyopenjtalk dependency, runs on a laptop too
    lex = {
        "畔蒜泰助": "はんずいたいすけ",
        "阪井泰介": "はんずいたいすけ",     # ASR's homophone with different characters
        "首脳会談": "しゅのうかいだん",
        "三社": "さんしゃ", "さんしゃ": "さんしゃ",
        "会議": "かいぎ",
        "さん": "さん", "が": "が", "来": "き", "まし": "まし", "た": "た",
        "この": "この", "は": "は", "。": "",
    }

    def reader(text: str) -> list[tuple[str, str]]:
        out, i = [], 0
        words = sorted(lex, key=len, reverse=True)
        while i < len(text):
            for w in words:
                if text.startswith(w, i):
                    out.append((w, lex[w]))
                    i += len(w)
                    break
            else:
                out.append((text[i], ""))
                i += 1
        return out

    h = cs.ScreenHints(reader=reader, recent=lambda: ["这是上一句已经译好的中文内容"])
    # screen text: UI lines must go, self-feeding lines must go, only the rest enters the term list
    h.set_text("会议字幕 已识别206段\n这是上一句已经译好的中文内容\n"
               "这样字幕可以根据屏幕内容修改\n"
               "畔蒜泰助 東京財団政策研究所\n三社の会談")
    words = [w for w, _r in h.terms]
    ok("畔蒜泰助" in words, "屏幕上的汉字人名进词表", str(words))
    ok(all("会议字幕" not in w for w in words), "页面 UI 行被丢掉", str(words))
    ok(all("中文内容" not in w for w in words), "我们自己的译文行被丢掉（防自我循环）")
    ok(all("这样" not in w for w in words), "中文界面/聊天窗口的行被丢掉（简体外挂字）")
    ok(h.status()["lines"] == 5, "记录原始屏幕行数", h.status()["lines"])

    # hit: identical reading, different spelling
    hit = h.suggest("阪井泰介さんが来ました")
    ok(hit.startswith("【屏幕词】") and "畔蒜泰助" in hit, "同音异字被命中并提示", hit)
    ok(len(hit) < 80, "提示很短（只多几十个 token）", f"{len(hit)} 字")
    ok(h.hits == 1, "命中计数 +1", h.hits)

    # miss: this sentence is already spelled correctly, so no hint (otherwise pure waste plus noise)
    ok(h.suggest("畔蒜泰助さんが来ました") == "", "写法已正确时不提示")
    # miss: the readings are far apart
    ok(h.suggest("この会議は長いです") == "", "读音不相似时不提示")
    # miss: similar reading but not a single character in common (pure coincidence, usually a Chinese UI word force-read with Japanese readings)
    h3 = cs.ScreenHints(reader=reader)
    h3.terms = [("三社", "さんしゃ")]
    ok(h3.suggest("さんしゃです") == "", "读音巧合但无共同字时不提示")

    # empty term list
    h2 = cs.ScreenHints(reader=reader)
    ok(h2.suggest("阪井泰介さんが来ました") == "", "没有屏幕词表时恒为空串")

    # ---- wired into Captioner: the hint appears in the request body only on a hit
    cap = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1, screen=h)
    cap.start_workers()
    stream_calls.clear()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    cap.submit(pcm, 1.0, end_ts=time.time())
    t0 = time.time()
    while time.time() - t0 < 2 and not cap.lines:
        time.sleep(0.005)
    ok(cap.status()["screen"] is not None, "status 里带屏幕状态",
       json.dumps(cap.status().get("screen"), ensure_ascii=False)[:60])
    ok(stream_calls and stream_calls[-1]["screen_hint"] == "",
       "没命中时请求体不带任何屏幕提示（零代价）",
       repr(stream_calls[-1]["screen_hint"])[:40])

    # the stubbed ASR says "阪井泰介" while the screen shows "畔蒜泰助" → a hint is expected
    cap2 = cs.Captioner(asr_saying("阪井泰介さんが来ました"), bilingual=True,
                        tr_workers=1, screen=h)
    cap2.start_workers()
    cap2.submit(pcm, 1.0, end_ts=time.time())
    t0 = time.time()
    while time.time() - t0 < 2 and not cap2.lines:
        time.sleep(0.005)
    while time.time() - t0 < 2 and not cap2.lines[0].get("screen_hint"):
        time.sleep(0.005)
    calls = [c for c in stream_calls if "阪井" in c["text"]]
    ok(bool(calls), "打桩 ASR 的句子进了翻译", str(len(calls)))
    ok("畔蒜泰助" in calls[-1]["screen_hint"], "命中时提示随请求体一起发出",
       calls[-1]["screen_hint"][:50])
    ok(cap2.lines[0].get("screen_hint", "").startswith("【屏幕词】"),
       "行对象上标了 screen_hint，便于事后核对")

    # off by default: without screen, status is None and the request body must carry no screen_hint
    cap3 = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1)
    ok(cap3.status()["screen"] is None, "不传 screen 时 status.screen 为 None")


def test_page_js_syntax():
    """The embedded page must parse in the browser.

    Hit on 2026-11-21: adding a tooltip to #scr, we wrote '\\n', and PAGE was not a raw string
    at the time, so the Python side turned the newline escape inside the JS into a real newline;
    the single-quoted string was truncated → the whole <script> block had a syntax error → the
    page stuck on the initial DOM (connecting… / -99 dB / 0 segments) with every button dead,
    while the server was perfectly fine (/api/status and /api/poll both had data), which sent the
    debugging effort in the wrong direction. So here the <script> inside PAGE is handed straight
    to node --check: a syntax error cannot slip through.
    """
    import re
    import shutil
    import subprocess
    import tempfile

    html = cs.PAGE.replace("__DEV__", "USB3.0 Capture (PipeWire)")
    ok("__DEV__" not in html, "设备名占位符被替换")
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    ok(m is not None, "页面里有 <script> 块")
    js = m.group(1)
    ok("applyPayload" in js and "startSSE" in js, "关键函数还在", js[:20])

    # a single-quoted string must never contain a real newline (this is the first thing to blow up
    # if the raw string is reverted). Comments are stripped first: an English comment containing
    # an apostrophe ("don't") is not a broken string, and counting it made this check cry wolf.
    def _strip_comments(text):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        out = []
        for line in text.split("\n"):
            pos = line.find("//")
            while pos != -1:
                # only a comment if the quotes before it are balanced (skip URLs in strings)
                if line[:pos].count("'") % 2 == 0:
                    line = line[:pos]
                    break
                pos = line.find("//", pos + 1)
            out.append(line)
        return out

    bad = [i + 1 for i, ln in enumerate(_strip_comments(js)) if ln.count("'") % 2 == 1]
    ok(not bad, "没有跨行断掉的单引号字符串", f"可疑行 {bad[:5]}")

    node = shutil.which("node")
    if not node:
        print("    (跳过 node --check：本机没有 node)")
        return
    p = Path(tempfile.gettempdir()) / "_page_js_check.js"
    p.write_text(js, encoding="utf-8")
    r = subprocess.run([node, "--check", str(p)], capture_output=True, text=True)
    ok(r.returncode == 0, "node --check 通过", (r.stderr or "").strip().split("\n")[0])


if __name__ == "__main__":
    cs.translate = fake_translate
    cs.translate_stream = fake_translate_stream
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name}")
            fn()
    print("ALL PASS")
