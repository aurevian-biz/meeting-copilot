#!/usr/bin/env python3
"""字幕延迟改造的回归测试：两段式出字、队列解耦、上头文、短段。

ASR 和翻译都打桩，所以不依赖模型和 vLLM，本机/Jetson 都能跑：

    ./asr-venv/bin/python test_caption_pipe.py
    python test_caption_pipe.py            # 本机（需 numpy）

验证的是"结构"：原文必须比译文先出来，而且采集线程不能被识别/翻译卡住。
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
calls: list[dict] = []          # fake_translate 的调用记录，用例之间共享


# ---------------------------------------------------------------- 打桩
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
    time.sleep(0.25)                       # 模拟 250ms 译文耗时
    return "テスト文"


stream_calls: list[dict] = []


def fake_translate_stream(base, model, text, target, timeout=12.0, ctx="",
                          on_delta=None, screen_hint=""):
    """模拟流式：先吐两个增量，再给全文。首字快、整句慢。"""
    stream_calls.append({"ctx": ctx, "text": text, "target": target,
                         "screen_hint": screen_hint})
    t0 = time.time()
    pieces = ["テスト", "文"]
    for i, p in enumerate(pieces, 1):
        time.sleep(0.09)     # > 80ms 节流窗口，保证两次增量都真的推出去
        if on_delta is not None:
            on_delta("".join(pieces[:i]))
    ttft = int(90)
    return "".join(pieces), ttft, int((time.time() - t0) * 1000)


def ok(cond, msg, detail=""):
    if not cond:
        print(f"[FAIL] {msg}  {detail}")
        raise SystemExit(1)
    print(f"[ok]   {msg}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- 用例
def test_two_stage():
    """原文先出，译文后补，且回填同一行。"""
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
    """流式：同一 seq 合并、中间态是最终译文的前缀、只有 final 带耗时。"""
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
    """第二句带第一句做上文；积压时译文按序补。"""
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
    # 流式路径记在 stream_calls，整句路径记在 calls；按实际走的取
    last = (stream_calls or calls)[-1]
    ok(last["ctx"].startswith("测试一句话"), "翻译带上了上文",
       repr(last["ctx"][:20]))


def test_submit_not_blocked():
    """采集线程投递段不能被 ASR/翻译阻塞。"""
    cap = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)
    t = time.time()
    for _ in range(20):
        cap.submit(pcm, 1.0, end_ts=time.time())
    dt = (time.time() - t) * 1000
    ok(dt < 100, "连投 20 段立刻返回", f"{dt:.1f}ms")


def test_queue_overflow_drops_oldest():
    """队列满了丢最老的一段，而不是无限堆积。"""
    a = cs.Captioner(FakeAsr(), bilingual=False)
    a._started = True                       # 不启工作线程，让它纯堆积
    pcm = np.zeros(FRAME_N, dtype=np.int16)
    for _ in range(a._q_asr.maxsize + 5):
        a.submit(pcm, 0.02, end_ts=time.time())
    ok(a._q_asr.qsize() == a._q_asr.maxsize, "队列封顶", f"qsize={a._q_asr.qsize()}")
    ok(a.dropped == 5, "丢弃计数", f"dropped={a.dropped}")


def test_segment_length():
    """max_s=3.0 生效：连续说话最多 3 秒一段。"""
    seg = cs.Segmenter(margin_db=8.0, hold_s=0.42, max_s=3.0, min_s=0.25)
    frame = (np.random.randn(FRAME_N) * 400).astype(np.int16)
    lens = []
    for i in range(400):                    # 8 秒连续有声
        db = -20.0
        out = seg.push(frame, db)
        if out is not None:
            lens.append(out.size / SR)
        # 前 1.2 秒是本底预热，不判语音
    ok(len(lens) >= 2, "连续说话被强制切段", f"{[round(x,2) for x in lens]}")
    ok(max(lens) <= 3.0 + 1e-6, "单段不超过 3 秒", f"max={max(lens):.2f}s")
    ok(all(x >= 0.25 for x in lens), "不产出过短的碎段")


def test_soft_cut_at_dip():
    """硬切前回退到能量低谷：段长不再一律等于 max_s。"""
    seg = cs.Segmenter(margin_db=8.0, hold_s=0.42, max_s=3.0, min_s=0.25, soft_s=0.7)
    frame = (np.random.randn(FRAME_N) * 400).astype(np.int16)
    lens = []
    for i in range(400):
        # 每 0.5 秒安排 2 帧深低谷；够不上 21 帧的静音挂起，只能靠硬切
        db = -62.0 if (i % 25) < 2 else -20.0
        out = seg.push(frame, db)
        if out is not None:
            lens.append(out.size / SR)
    ok(len(lens) >= 2, "连续有声被切成多段", f"{[round(x,2) for x in lens][:6]}")
    ok(max(lens) <= 3.0 + 1e-6, "不超过硬上限 3 秒", f"max={max(lens):.2f}s")
    ok(min(lens) < 2.95, "切点回退到了低谷附近", f"min={min(lens):.2f}s")


def _pump(seg, frame, voiced_n, silent_n, db_v=-20.0, db_s=-70.0):
    """先说话 voiced_n 帧、再静音 silent_n 帧，收集期间产出的所有段。"""
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
    """停顿门限分层：段还短时只认长停顿，段够长时短停顿就切。"""
    frame = (np.random.randn(FRAME_N) * 400).astype(np.int16)

    def fresh():
        seg = cs.Segmenter(margin_db=8.0, hold_s=0.45, max_s=4.5, min_s=0.25,
                           soft_s=0.7, hold_long_s=0.90, hold_short_s=0.30,
                           mid_s=1.20, long_s=2.40)
        for _ in range(int(1.2 / cs.FRAME_S) + 2):      # 预热，估本底
            seg.push(frame, -70.0)
        return seg

    # 0.6 秒短句 + 0.5 秒停顿：0.5 < hold_long(0.9)，不许切。
    # 这正是「…况且140。」+「亿美元对台均售…」被数字切断的成因。
    seg = fresh()
    outs = _pump(seg, frame, 30, 25)
    ok(not outs, "短段遇 0.5s 停顿不该断句", f"切出了 {len(outs)} 段")

    # 同一个段，静音继续撑到 0.9 秒以上就该断
    outs = _pump(seg, frame, 0, 25)
    ok(len(outs) == 1, "静音够 0.9s 就断句", f"{len(outs)} 段")
    if outs:
        dur = outs[0].size / SR
        ok(1.2 < dur < 2.2, "段长≈0.6 有声 + 0.9 静音 + 预补", f"{dur:.2f}s")

    # 3 秒长句 + 0.32 秒停顿：段已超 seg_long，按 hold_short 立刻切
    seg = fresh()
    outs = _pump(seg, frame, 150, 16)
    ok(len(outs) == 1, "长段遇短停顿就切，延迟不再往上堆", f"{len(outs)} 段")
    ok(seg.last_cut == "silence", "记录成因是自然停顿，不是撞上限", seg.last_cut)

    # 3 秒长句 + 0.2 秒停顿（不足 hold_short）：还是不许切
    seg = fresh()
    outs = _pump(seg, frame, 150, 10)
    ok(not outs, "0.2s 停顿短于 hold_short，不切", f"{len(outs)} 段")

    # 静音门限调到比基准还短时，不能反过来让长段门槛低于基准（参数传反的兜底）
    seg = cs.Segmenter(margin_db=8.0, hold_s=0.45, max_s=4.5, min_s=0.25,
                       hold_long_s=0.10, hold_short_s=2.0)
    ok(seg.hold_long_frames >= seg.hold_frames, "hold_long 不小于基准")
    ok(seg.hold_short_frames <= seg.hold_frames, "hold_short 不大于基准")


def test_noise_segment_dropped():
    """只有标点的识别结果不算一句话，不能占一行字幕。"""
    ok(not cs.SUBSTANCE.search(" 。"), "纯标点丢弃")
    ok(not cs.SUBSTANCE.search("  "), "空白丢弃")
    ok(bool(cs.SUBSTANCE.search("嗯。")), "单个实词保留")
    ok(bool(cs.SUBSTANCE.search("140")), "纯数字保留")
    ok(bool(cs.SUBSTANCE.search("はい")), "假名保留")


def test_lang_clean():
    """译文残留检测：假名必中；中日共用汉字绝不能误伤。"""
    ok(not cs.lang_clean("美国应该坚持するべきです", "zh"), "日译中残留假名被检出")
    ok(cs.lang_clean("在实质层面，美国应当坚持立场。", "zh"), "干净中文放过")
    # 专有名词允许保留日文写法，少量假名不该触发重试
    ok(cs.lang_clean("特别是深圳（深セン）被称为中国的硅谷。", "zh"), "专有名词少量假名放过")
    ok(not cs.lang_clean("美国应该坚持するべきです", "zh"), "整句假名仍判脏")
    ok(not cs.lang_clean("中国側は故意に韓国に给好处", "ja"), "中译日残留简体被检出")
    ok(cs.lang_clean("中国側は意図的に韓国に便宜を図った。", "ja"), "干净日语放过")
    # 这三个汉字中日共用，误放进名单就会把正常译文判脏、白白多跑一次翻译
    ok(cs.lang_clean("会議は都会で開かれる。", "ja"), "「会」「都」不误伤")
    # 整句照抄中文时每个字都是日语也有的汉字，只能靠「有没有假名」判
    ok(not cs.lang_clean("我是主持人田英和", "ja"), "纯汉字整句判脏（没翻译）")
    ok(cs.lang_clean("私は司会の谭英文です。", "ja"), "带假名的正常译文放过")
    ok(cs.lang_clean("了解", "ja"), "短句不受无假名规则影响")
    ok(cs.lang_clean("その刹那の話。", "ja"), "「那」不误伤")
    ok(cs.lang_clean("anything goes", "en"), "没有判据的语言直接放过")


def asr_saying(text: str):
    """固定输出一句话的 ASR 替身，用来验证「原文逐字、翻译前清洗」。"""
    result = type("R", (), {"text": text})()
    stream = type("S", (), {"result": result,
                            "accept_waveform": lambda self, sr, x: None})()
    rec = type("Rec", (), {"create_stream": lambda self: stream,
                           "decode_stream": lambda self, s: None})()
    return type("A", (), {"rec": rec})()


def test_strip_fillers():
    """去口癖的边界：只删纯语气词，实义词与空结果都必须保住。"""
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
    """页面显示的原文保持逐字，只有送进翻译的文本被清洗。"""
    calls.clear()                       # 前面的用例留下的线程还在跑，先清空记录
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
    rec = stream_calls or calls          # 流式是默认路径
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
    """分段器要能交出「还没断句的当前段」——草稿字幕的原料。"""
    seg = cs.Segmenter()
    ok(seg.draft() is None, "没声音时没有草稿")
    frame = (np.random.randn(FRAME_N) * 300).astype(np.int16)
    for _ in range(100):                     # 2 秒：前 60 帧是预热期，之后才建段
        seg.push(frame, -20.0)
    d = seg.draft()
    ok(d is not None and d[0].size > 0 and d[1] > 0.5, "有声时能拿到草稿音频",
       f"{d[1]:.2f}s" if d else "None")


def test_draft_caption():
    """说话过程中先出灰字，段定稿后草稿立即清空（正式行接手）。"""
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
    """草稿识别慢于定稿时必须丢弃，否则灰字会盖到已经定稿的内容上。"""

    class SlowAsr:
        class rec:
            @staticmethod
            def create_stream():
                time.sleep(0.25)             # 模拟识别比断句还慢
                return _Stream()

            @staticmethod
            def decode_stream(s):
                return None

    cap = cs.Captioner(SlowAsr(), bilingual=True, draft_on=True)
    cap.start_workers()
    pcm = (np.random.randn(SR) * 100).astype(np.int16)

    cap.push_draft(pcm, 1.0)
    time.sleep(0.05)                         # 草稿线程已进 recognize（要 0.25s）
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
    """ctx_lines=0 必须真的不带上文。

    切片写成 `prev[-self.ctx_lines:]` 时，0 会变成 `[0:]` 把**全部历史**都当上文，
    实测把 35B 的 prompt 从 80 token 涨到 138、首字多花 300ms。
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
    """两档 prompt 都要在、且能按 --tr-prompt / TR_STYLE 切换。

    short 版是 4.20 加的：short 只留三条硬规则，砍掉示例，长度必须真的短下来
    （否则等于没改），但也不能短到把「纯目标语言」这条丢掉。
    """
    for target in ("zh", "ja"):
        full = cs._tr_system(target, "full")
        short = cs._tr_system(target, "short")
        ok(short != full, f"{target} 两档不同", short[:20])
        ok(len(short) < 140, f"{target} short 确实短（{len(short)} 字）", len(short))
        ok(len(full) > len(short) * 2, f"{target} full 至少两倍长", len(full))
        # 三条硬规则的关键词一个都不能少
        for kw in ("【本句】", "【上文】", "半句"):
            ok(kw in short, f"{target} short 含 {kw}", short)

    # 模块级开关：不传 style 时看 TR_STYLE
    old = cs.TR_STYLE
    try:
        cs.TR_STYLE = "short"
        ok(cs._tr_system("ja") == cs._tr_system("ja", "short"), "TR_STYLE 生效")
        cs.TR_STYLE = "full"
        ok(cs._tr_system("ja") == cs._tr_system("ja", "full"), "TR_STYLE 复原")
    finally:
        cs.TR_STYLE = old


def test_screen_hints():
    """屏幕词表：只按「读音相似、写法不同」命中；命中才递提示，不命中零代价。"""
    # ---- 读音工具
    ok(cs._hira("ハンズイ") == "はんずい", "_hira 片假名转平假名")
    ok(cs._hira("ABC 12、") == "", "_hira 丢掉拉丁/数字/标点")
    ok(cs._lev("はんずい", "はんずい") == 0, "_lev 相同为 0")
    ok(cs._lev("はんずい", "はんすい") == 1, "_lev 差一个音为 1")
    ok(cs._lev("はんずい", "はん") == 9, "_lev 长度差过大直接判 9")

    # ---- 假读音器：不依赖 pyopenjtalk，本机也能跑
    lex = {
        "畔蒜泰助": "はんずいたいすけ",
        "阪井泰介": "はんずいたいすけ",     # ASR 的同音异字
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
    # 屏幕文本：UI 行要丢、自我循环行要丢、剩下的才进词表
    h.set_text("会议字幕 已识别206段\n这是上一句已经译好的中文内容\n"
               "这样字幕可以根据屏幕内容修改\n"
               "畔蒜泰助 東京財団政策研究所\n三社の会談")
    words = [w for w, _r in h.terms]
    ok("畔蒜泰助" in words, "屏幕上的汉字人名进词表", str(words))
    ok(all("会议字幕" not in w for w in words), "页面 UI 行被丢掉", str(words))
    ok(all("中文内容" not in w for w in words), "我们自己的译文行被丢掉（防自我循环）")
    ok(all("这样" not in w for w in words), "中文界面/聊天窗口的行被丢掉（简体外挂字）")
    ok(h.status()["lines"] == 5, "记录原始屏幕行数", h.status()["lines"])

    # 命中：读音完全相同、写法不同
    hit = h.suggest("阪井泰介さんが来ました")
    ok(hit.startswith("【屏幕词】") and "畔蒜泰助" in hit, "同音异字被命中并提示", hit)
    ok(len(hit) < 80, "提示很短（只多几十个 token）", f"{len(hit)} 字")
    ok(h.hits == 1, "命中计数 +1", h.hits)

    # 不命中：本句已经写对了，不该再提示（否则纯属浪费 + 干扰）
    ok(h.suggest("畔蒜泰助さんが来ました") == "", "写法已正确时不提示")
    # 不命中：读音差得远
    ok(h.suggest("この会議は長いです") == "", "读音不相似时不提示")
    # 不命中：读音相似但一个字都不共用（纯巧合，多半是中文界面词被硬读出日语读音）
    h3 = cs.ScreenHints(reader=reader)
    h3.terms = [("三社", "さんしゃ")]
    ok(h3.suggest("さんしゃです") == "", "读音巧合但无共同字时不提示")

    # 空词表
    h2 = cs.ScreenHints(reader=reader)
    ok(h2.suggest("阪井泰介さんが来ました") == "", "没有屏幕词表时恒为空串")

    # ---- 接到 Captioner：命中才出现在请求体里
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

    # 打桩 ASR 说的是「阪井泰介」，屏幕上是「畔蒜泰助」→ 该带提示
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

    # 默认关闭：不传 screen 时 status 是 None、请求体也不该有 screen_hint
    cap3 = cs.Captioner(FakeAsr(), bilingual=True, tr_workers=1)
    ok(cap3.status()["screen"] is None, "不传 screen 时 status.screen 为 None")


def test_page_js_syntax():
    """内嵌页面必须能被浏览器解析。

    2026-11-21 踩过：给 #scr 加 tooltip 时写了 '\\n'，PAGE 当时不是 raw 字符串，
    Python 端把 JS 里的换行转义变成了真换行，单引号字符串被截断 → 整段 <script>
    语法错误 → 页面停在初始 DOM（连接中… / -99 dB / 0 段），按钮全失效，
    而服务端一切正常（/api/status、/api/poll 都有数据），排查方向全被带偏。
    这里直接把 PAGE 里的 <script> 交给 node --check，语法错就跑不过。
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

    # 单引号字符串里绝不能出现真换行（raw 字符串被改回去时最先炸的就是这个）
    bad = [i + 1 for i, ln in enumerate(js.split("\n")) if ln.count("'") % 2 == 1]
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
