#!/usr/bin/env python3
"""常驻采集（AudioHub）的分发语义测试，不需要麦克风和模型。

唤醒的难点不在识别，而在多路消费者：唤醒线程、VAD 录音、以及「唤醒词刚说完就接着说」
时的时间连续性。这些都能在没有设备的情况下验证。
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
        h = hub(pre_s=0.2)          # 10 帧
        for i in range(25):
            h._push(frame(h, i))
        q = h.subscribe(pre_s=0.2)
        self.assertEqual(q.qsize(), 10)
        got = [q.get_nowait()[0] for _ in range(10)]
        self.assertEqual(got, list(range(15, 25)))  # 最近 10 帧、且是新的

    def test_no_prefill(self):
        h = hub()
        for i in range(5):
            h._push(frame(h))
        q = h.subscribe(pre_s=0.0)
        self.assertEqual(q.qsize(), 0)

    def test_prefill_capped_by_buffer(self):
        h = hub(pre_s=0.2)          # 缓冲只有 10 帧
        for i in range(40):
            h._push(frame(h, i))
        q = h.subscribe(pre_s=5.0)  # 要 5 秒，但只有 10 帧可给
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
        self.assertEqual(got, [3, 4])  # 只保留最新两帧，采集线程不被阻塞

    def test_rate_s_without_frames(self):
        self.assertEqual(hub().rate_s, -1.0)


class TestEventsSince(unittest.TestCase):
    """短轮询的增量切片：唤醒事件全靠它送到页面。"""

    def buf(self, n):
        return deque(((i, {"i": i}) for i in range(1, n + 1)), maxlen=600)

    def test_tail(self):
        got = vs.events_since(self.buf(5), 3)
        self.assertEqual([e["i"] for e in got], [4, 5])

    def test_none_new(self):
        self.assertEqual(vs.events_since(self.buf(5), 5), [])

    def test_future_after_is_empty(self):
        # 前端打开时用 after=1e12 对齐，不应重放历史
        self.assertEqual(vs.events_since(self.buf(5), 10**12), [])

    def test_empty_buffer(self):
        self.assertEqual(vs.events_since(deque(), 0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
