#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Dispatch-semantics tests for the always-on capture (AudioHub); no microphone or model needed.

The hard part of wake word is not recognition but the multiple consumers: the wake thread, the VAD
recorder, and the time continuity when someone keeps talking right after the wake word. All of it
can be verified without any device.
"""
import sys
import unittest
from pathlib import Path

from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_server as vs  # noqa: E402


def hub(pre_s=0.6):
    h = vs.AudioHub("null", frame_s=0.02, pre_s=pre_s)
    return h


def frame(h, value=1):
    return np.full(h.frame_len, value, dtype=np.int16)


class TestSubscribe(unittest.TestCase):
    def test_prefill_recent(self):
        h = hub(pre_s=0.2)          # 10 frames
        for i in range(25):
            h._push(frame(h, i))
        q = h.subscribe(pre_s=0.2)
        self.assertEqual(q.qsize(), 10)
        got = [q.get_nowait()[0] for _ in range(10)]
        self.assertEqual(got, list(range(15, 25)))  # the 10 most recent frames, and they are the new ones

    def test_no_prefill(self):
        h = hub()
        for i in range(5):
            h._push(frame(h))
        q = h.subscribe(pre_s=0.0)
        self.assertEqual(q.qsize(), 0)

    def test_prefill_capped_by_buffer(self):
        h = hub(pre_s=0.2)          # the buffer holds only 10 frames
        for i in range(40):
            h._push(frame(h, i))
        q = h.subscribe(pre_s=5.0)  # asks for 5 seconds, but only 10 frames can be given
        self.assertEqual(q.qsize(), 10)

    def test_live_frames_arrive(self):
        h = hub()
        q = h.subscribe(pre_s=0.0)
        h._push(frame(h, 7))
        self.assertEqual(q.get_nowait()[0], 7)

    def test_unsubscribe_stops_delivery(self):
        h = hub()
        q = h.subscribe(pre_s=0.0)
        h.unsubscribe(q)
        h._push(frame(h, 9))
        self.assertEqual(q.qsize(), 0)

    def test_two_subscribers_both_get_frames(self):
        h = hub()
        a = h.subscribe(pre_s=0.0)
        b = h.subscribe(pre_s=0.0)
        h._push(frame(h, 3))
        self.assertEqual(a.get_nowait()[0], 3)
        self.assertEqual(b.get_nowait()[0], 3)

    def test_slow_consumer_drops_oldest(self):
        h = hub()
        q = h.subscribe(pre_s=0.0)
        q.maxsize = 2
        for i in range(5):
            h._push(frame(h, i))
        got = [q.get_nowait()[0] for _ in range(2)]
        self.assertEqual(got, [3, 4])  # only the newest two frames are kept; the capture thread is not blocked

    def test_rate_s_without_frames(self):
        self.assertEqual(hub().rate_s, -1.0)


class TestEventsSince(unittest.TestCase):
    """Incremental slicing for short polling: wake events reach the page entirely through this."""

    def buf(self, n):
        return deque(((i, {"i": i}) for i in range(1, n + 1)), maxlen=600)

    def test_tail(self):
        got = vs.events_since(self.buf(5), 3)
        self.assertEqual([e["i"] for e in got], [4, 5])

    def test_none_new(self):
        self.assertEqual(vs.events_since(self.buf(5), 5), [])

    def test_future_after_is_empty(self):
        # the frontend aligns with after=1e12 when it opens; history must not be replayed
        self.assertEqual(vs.events_since(self.buf(5), 10**12), [])

    def test_empty_buffer(self):
        self.assertEqual(vs.events_since(deque(), 0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
