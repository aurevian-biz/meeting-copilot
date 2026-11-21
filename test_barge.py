# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Pure-logic tests for barge-in: no microphone, model weights or LLM needed.

Covers three areas that are easy to get wrong:
  1. SpeakState's playback timeline — synthesis is faster than playback, so the clock must accumulate by audio duration
  2. Echo detection — with the speaker on, it must not interrupt itself
  3. Audio handoff — after a barge-in, the first few words of the next sentence must not be eaten
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
        # two pieces of 1 second each: the second follows the first, so the window is ≈2s, not 1s
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
        self.assertEqual(len(said), 6)          # only the most recent ones are kept
        self.assertEqual(said[-1], "第9句。")


class TestEcho(unittest.TestCase):
    """With the speaker on, the mic picks up the TTS; that must be judged as echo, otherwise every
    announcement gets interrupted."""

    def test_exact_echo(self):
        self.assertTrue(vs.looks_like_echo(
            "今天天气不错，适合出门。", ["今天天气不错，适合出门。"]))

    def test_partial_echo(self):
        # echo usually gets truncated to half a sentence
        self.assertTrue(vs.looks_like_echo(
            "今天天气不错", ["今天天气不错，适合出门散步。"]))

    def test_truncated_fragment(self):
        # real output measured end to end: 0.9 seconds of echo recognized as only three characters.
        # whole-sentence similarity yields 0.35 and misses it, so the criterion is "longest common substring ratio"
        self.assertTrue(vs.looks_like_echo(
            "今天天", ["今天天气不错，我们出去走走吧。"]))

    def test_truncated_interjection(self):
        self.assertFalse(vs.looks_like_echo(
            "等一下午", ["今天天气不错，我们出去走走吧。"]))

    def test_real_interjection(self):
        self.assertFalse(vs.looks_like_echo(
            "等一下我有别的问题", ["今天天气不错，适合出门散步。"]))

    def test_short_command_is_speech(self):
        # a short command like "停" gives no reliable similarity; better to barge in than to miss it
        self.assertFalse(vs.looks_like_echo("停", ["停下来听我说。"]))
        self.assertFalse(vs.looks_like_echo("等一下", ["停下来听我说。"]))

    def test_empty_heard(self):
        self.assertFalse(vs.looks_like_echo("", ["任何句子。"]))

    def test_punctuation_ignored(self):
        self.assertTrue(vs.looks_like_echo(
            "今天天气不错。", ["今天天气不错"]))

    def test_similar_but_different_topic(self):
        # same topic but different wording: the user is interjecting, not echo
        self.assertFalse(vs.looks_like_echo(
            "那明天会不会下雨", ["今天多云转晴，最高气温二十度。"]))

    def test_said_list_input(self):
        self.assertTrue(vs.looks_like_echo(
            "第二句话的内容在这里", ["第一句。", "第二句话的内容在这里。"]))


class TestFrameHandoff(unittest.TestCase):
    """Barge-in handoff: frames consumed by the detection thread must enter the recording queue unchanged."""

    def setUp(self):
        self.hub = vs.AudioHub("plughw:NONE", pre_s=0.6)  # not started, pure logic

    def test_pre_frames_replace_ring(self):
        hub = self.hub
        hub._recent.append(np.full(320, 7, dtype=np.int16))   # an old frame sitting in the ring buffer
        frames = [np.full(320, 9, dtype=np.int16) for _ in range(5)]
        q = hub.subscribe(pre_s=0.6, pre_frames=frames)
        got = [q.get_nowait() for _ in range(5)]
        self.assertTrue(all(f[0] == 9 for f in got))
        self.assertTrue(q.empty())      # the old ring-buffer frame was not mixed in

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
    """The threshold must track the silence floor and must not be thrown off by echo while playing."""

    def make(self):
        return vs.BargeMonitor(
            hub=None, asr=None, speak=vs.SpeakState(), runner=None,
            history=[], push=lambda ev: None, turn_lock=None,
            vad_model="unused", min_db=-42.0, margin=8.0, max_db=-18.0)

    def test_quiet_room(self):
        m = self.make()
        m.quiet_floor = -56.0
        self.assertAlmostEqual(m.threshold(), -42.0)   # lands on min_db

    def test_noisy_room_raises(self):
        m = self.make()
        m.quiet_floor = -30.0
        self.assertAlmostEqual(m.threshold(), -22.0)   # floor +8

    def test_echo_does_not_raise(self):
        # regression test: playback-level audio used to be folded into the threshold, so with the
        # speaker on the echo pushed the threshold above speech and a real interjection never got through
        m = self.make()
        m.quiet_floor = -56.0
        m.echo_floor = -20.0        # speaker on, lots of echo
        self.assertAlmostEqual(m.threshold(), -42.0)

    def test_capped(self):
        m = self.make()
        m.quiet_floor = -10.0
        self.assertAlmostEqual(m.threshold(), -18.0)   # capped at max_db


if __name__ == "__main__":
    unittest.main(verbosity=2)
