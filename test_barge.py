"""打断（barge-in）的纯逻辑测试：不需要麦克风、模型权重、LLM。

覆盖三块容易错的地方：
  1. SpeakState 的播报时间线 —— 合成比播放快，时钟必须按音频时长累加
  2. 回声判别 —— 外放时不能自己打断自己
  3. 音频交接 —— 打断后第一句的开头几个字不能被吃掉
"""

import queue
import time
import unittest

import numpy as np

import voice_server as vs


class TestSpeakState(unittest.TestCase):
    def setUp(self):
        self.sp = vs.SpeakState()
        self.sp.begin("t1")

    def test_idle_before_any_audio(self):
        self.assertFalse(self.sp.playing())
        self.assertEqual(self.sp.remain(), 0.0)

    def test_playing_after_add(self):
        self.sp.add(1.0, "第一句。")
        self.assertTrue(self.sp.playing())
        self.assertGreater(self.sp.remain(), 0.5)

    def test_queue_accumulates(self):
        # 两段各 1 秒：第二段接在第一段后面，总窗口 ≈2 秒而不是 1 秒
        self.sp.add(1.0, "第一句。")
        self.sp.add(1.0, "第二句。")
        self.assertGreater(self.sp.remain(), 1.5)

    def test_expires(self):
        self.sp.add(0.05, "短句。")
        time.sleep(0.12)
        self.assertFalse(self.sp.playing())

    def test_cancel_zeroes_clock(self):
        self.sp.add(3.0, "长句。")
        self.sp.cancel()
        self.assertFalse(self.sp.playing())
        self.assertTrue(self.sp.cancelled)

    def test_begin_clears_cancel(self):
        self.sp.cancel()
        self.sp.begin("t2")
        self.assertFalse(self.sp.cancelled)
        self.assertFalse(self.sp.playing())

    def test_said_keeps_recent_only(self):
        for i in range(10):
            self.sp.add(0.1, f"第{i}句。")
        said = self.sp.said()
        self.assertEqual(len(said), 6)          # 只留最近几句
        self.assertEqual(said[-1], "第9句。")


class TestEcho(unittest.TestCase):
    """外放时麦克风听到 TTS，必须判成串音，否则一播报就被打断。"""

    def test_exact_echo(self):
        self.assertTrue(vs.looks_like_echo(
            "今天天气不错，适合出门。", ["今天天气不错，适合出门。"]))

    def test_partial_echo(self):
        # 串音常被截到半句
        self.assertTrue(vs.looks_like_echo(
            "今天天气不错", ["今天天气不错，适合出门散步。"]))

    def test_truncated_fragment(self):
        # 端到端实测的真实输出：0.9 秒串音只识别出三个字。
        # 用整句相似度会得到 0.35 而漏判，所以判据是「最长公共子串占比」
        self.assertTrue(vs.looks_like_echo(
            "今天天", ["今天天气不错，我们出去走走吧。"]))

    def test_truncated_interjection(self):
        self.assertFalse(vs.looks_like_echo(
            "等一下午", ["今天天气不错，我们出去走走吧。"]))

    def test_real_interjection(self):
        self.assertFalse(vs.looks_like_echo(
            "等一下我有别的问题", ["今天天气不错，适合出门散步。"]))

    def test_short_command_is_speech(self):
        # 「停」这种短指令给不出可靠相似度，宁可打断也不要漏
        self.assertFalse(vs.looks_like_echo("停", ["停下来听我说。"]))
        self.assertFalse(vs.looks_like_echo("等一下", ["停下来听我说。"]))

    def test_empty_heard(self):
        self.assertFalse(vs.looks_like_echo("", ["任何句子。"]))

    def test_punctuation_ignored(self):
        self.assertTrue(vs.looks_like_echo(
            "今天天气不错。", ["今天天气不错"]))

    def test_similar_but_different_topic(self):
        # 讲同一件事但措辞不同：用户插话，不是串音
        self.assertFalse(vs.looks_like_echo(
            "那明天会不会下雨", ["今天多云转晴，最高气温二十度。"]))

    def test_said_list_input(self):
        self.assertTrue(vs.looks_like_echo(
            "第二句话的内容在这里", ["第一句。", "第二句话的内容在这里。"]))


class TestFrameHandoff(unittest.TestCase):
    """打断交接：检测线程读走的帧必须原样进录音队列。"""

    def setUp(self):
        self.hub = vs.AudioHub("plughw:NONE", pre_s=0.6)  # 不 start，纯逻辑

    def test_pre_frames_replace_ring(self):
        hub = self.hub
        hub._recent.append(np.full(320, 7, dtype=np.int16))   # 环形缓冲里的旧帧
        frames = [np.full(320, 9, dtype=np.int16) for _ in range(5)]
        q = hub.subscribe(pre_s=0.6, pre_frames=frames)
        got = [q.get_nowait() for _ in range(5)]
        self.assertTrue(all(f[0] == 9 for f in got))
        self.assertTrue(q.empty())      # 没把环形缓冲的旧帧混进来

    def test_ring_buffer_still_used_by_default(self):
        hub = self.hub
        hub._recent.clear()
        hub._recent.append(np.full(320, 5, dtype=np.int16))
        q = hub.subscribe(pre_s=0.02)
        self.assertEqual(q.get_nowait()[0], 5)

    def test_no_duplicate_on_subscribe(self):
        hub = self.hub
        q1 = hub.subscribe(pre_s=0.0)
        q2 = hub.subscribe(pre_s=0.0)
        self.assertEqual(len(hub._subs), 2)
        hub.unsubscribe(q1)
        hub.unsubscribe(q2)
        self.assertEqual(len(hub._subs), 0)


class TestBargeThreshold(unittest.TestCase):
    """门限必须跟静默本底走，且不能被播报期的串音带偏。"""

    def make(self):
        return vs.BargeMonitor(
            hub=None, asr=None, speak=vs.SpeakState(), runner=None,
            history=[], push=lambda ev: None, turn_lock=None,
            vad_model="unused", min_db=-42.0, margin=8.0, max_db=-18.0)

    def test_quiet_room(self):
        m = self.make()
        m.quiet_floor = -56.0
        self.assertAlmostEqual(m.threshold(), -42.0)   # 落在 min_db

    def test_noisy_room_raises(self):
        m = self.make()
        m.quiet_floor = -30.0
        self.assertAlmostEqual(m.threshold(), -22.0)   # 本底 +8

    def test_echo_does_not_raise(self):
        # 回归测试：曾经把播报期电平算进门限，结果外放时串音把门限抬到
        # 人声之上，真人插话永远打不断
        m = self.make()
        m.quiet_floor = -56.0
        m.echo_floor = -20.0        # 外放，串音很大
        self.assertAlmostEqual(m.threshold(), -42.0)

    def test_capped(self):
        m = self.make()
        m.quiet_floor = -10.0
        self.assertAlmostEqual(m.threshold(), -18.0)   # max_db 封顶


if __name__ == "__main__":
    unittest.main(verbosity=2)
