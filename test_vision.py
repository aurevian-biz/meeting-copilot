#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_vision.py — 「看一眼屏幕」的纯逻辑单测

不碰采集卡、不调 OCR 引擎：只测意图判定、脱敏、上下文拼装、
帧指纹、以及 look() 的「全黑即无信号」保护。
"""

import os
import sys
import time
import types
import unittest

import numpy as np

import vision as vx


# ---------------------------------------------------------------- 假 cv2
class _FakeCapture:
    """假采集卡。用类属性记全局状态，方便断言「只开一次」。"""

    instances = 0
    reads = 0
    released = 0
    dark_until = 0      # 前 N 次 read 返回全黑（模拟 MS2109 预热黑帧）
    fail_next = 0       # 接下来 N 次 read 失败（模拟扩展坞掉线）

    def __init__(self, dev=None, api=None):
        type(self).instances += 1

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        cls = type(self)
        cls.reads += 1
        if cls.fail_next > 0:
            cls.fail_next -= 1
            return False, None
        if cls.reads <= cls.dark_until:
            return True, np.zeros((8, 8, 3), dtype=np.uint8)
        return True, np.full((8, 8, 3), 120, dtype=np.uint8)

    def release(self):
        type(self).released += 1


def _fake_cv2():
    m = types.ModuleType("cv2")
    m.CAP_V4L2 = 200
    m.CAP_PROP_FOURCC = 6
    m.CAP_PROP_FRAME_WIDTH = 3
    m.CAP_PROP_FRAME_HEIGHT = 4
    m.CAP_PROP_FPS = 5
    m.CAP_PROP_BUFFERSIZE = 38
    m.INTER_AREA = 3
    m.VideoWriter_fourcc = lambda *a: 0
    m.VideoCapture = _FakeCapture
    return m


class TestCaptureReuse(unittest.TestCase):
    """抓帧常驻：预热只付一次，掉线能自愈，闲置能松开。"""

    def setUp(self):
        self._real = sys.modules.get("cv2")
        sys.modules["cv2"] = _fake_cv2()
        _FakeCapture.instances = 0
        _FakeCapture.reads = 0
        _FakeCapture.released = 0
        _FakeCapture.dark_until = 0
        _FakeCapture.fail_next = 0

    def tearDown(self):
        if self._real is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = self._real

    def test_warmup_paid_once(self):
        v = vx.Vision(warmup=40, keep_open=True)
        v.grab()
        after_first = _FakeCapture.reads
        v.grab()
        v.grab()
        self.assertEqual(_FakeCapture.instances, 1, "只打开一次设备")
        self.assertEqual(v._cap.warmups, 1, "预热只做一轮")
        self.assertEqual(after_first, 40 + 2, "首次 = 40 帧预热 + 2 帧取新")
        self.assertEqual(_FakeCapture.reads, 40 + 6, "后续每次只读 2 帧")

    def test_reconnect_after_dropout(self):
        v = vx.Vision(warmup=40)
        v.grab()
        _FakeCapture.fail_next = 2          # 恰好吃掉这次取帧的 2 帧
        frame = v.grab()                    # 应自动重开一次并成功
        self.assertIsNotNone(frame)
        self.assertEqual(_FakeCapture.instances, 2, "掉线后重开了一次")
        self.assertEqual(v._cap.reconnects, 1)

    def test_keep_open_false_reopens(self):
        v = vx.Vision(warmup=40, keep_open=False)
        v.grab()
        v.grab()
        self.assertEqual(_FakeCapture.instances, 2, "每次取帧都重开（旧行为）")
        self.assertEqual(v._cap.warmups, 2)

    def test_idle_release(self):
        v = vx.Vision(warmup=40, idle_release=0.01)
        v.grab()
        time.sleep(0.05)
        v.grab()
        self.assertEqual(_FakeCapture.instances, 2, "闲置超时后重新打开")

    def test_blank_rewarms_then_recovers(self):
        v = vx.Vision(warmup=40, cache_ttl=0)
        _FakeCapture.dark_until = 40 + 2     # 首次取帧正好落在一轮黑帧里
        v._ocr = lambda img: {"text": "字", "lines": 1, "score": 0.9,  # type: ignore
                              "ocr_s": 0.1, "truncated": False}
        out = v.look(force=True)
        self.assertTrue(out["ok"], f"重预热后应拿到正常画面：{out}")
        self.assertEqual(v._cap.warmups, 2, "全黑时补了一轮预热")

    def test_still_blank_reports_blank(self):
        v = vx.Vision(warmup=40)
        _FakeCapture.dark_until = 10 ** 9    # 永远黑
        out = v.look(force=True)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "blank")
        self.assertEqual(v.errors, 1)


class TestBlankJudge(unittest.TestCase):
    """无信号判据：真机是「又暗又平」，不是纯黑。"""

    def test_real_device_nosignal_frame(self):
        # 实测数值：无信号帧 mean=7.0 / std=1.5，旧判据 mean<3 永远不命中
        frame = np.random.normal(7.0, 1.5, (1080, 1920, 3)).clip(0, 255).astype(np.uint8)
        self.assertLess(abs(float(frame.mean()) - 7.0), 1.0)   # 裁剪到 0 会让均值歪一点
        self.assertTrue(vx.Vision._is_blank(frame))

    def test_pure_black(self):
        self.assertTrue(vx.Vision._is_blank(np.zeros((1080, 1920, 3), np.uint8)))

    def test_dark_theme_is_not_blank(self):
        # 暗色 IDE：mean 低但纹理在，std 有十几，不能判成没信号
        frame = np.random.randint(0, 70, (1080, 1920, 3)).astype(np.uint8)
        self.assertGreater(float(frame.std()), 6.0)
        self.assertFalse(vx.Vision._is_blank(frame))


class TestLookIntent(unittest.TestCase):
    def test_hits(self):
        for s in ("看看屏幕上写的是什么", "你看一眼这个", "画面上有什么",
                  "这是什么", "屏幕上是啥", "帮我读一下这里的字",
                  "你看得见吗", "你能看到吗"):
            self.assertTrue(vx.is_look_intent(s), s)

    def test_misses(self):
        # 日常对话里不该被误判成「要看屏幕」
        for s in ("今天天气怎么样", "帮我算一下三加五", "你好",
                  "明天几点开会", "把音量调小一点"):
            self.assertFalse(vx.is_look_intent(s), s)

    def test_empty(self):
        self.assertFalse(vx.is_look_intent(""))
        self.assertFalse(vx.is_look_intent(None))  # type: ignore[arg-type]


class TestRedact(unittest.TestCase):
    def test_long_key(self):
        out, n = vx.redact("token是 " + "a1b2c3d4" * 5)
        self.assertGreaterEqual(n, 1)
        self.assertIn("***", out)
        self.assertNotIn("a1b2c3d4a1b2c3d4", out)

    def test_sk_key_keeps_prefix(self):
        out, n = vx.redact("sk-abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(n, 1)
        self.assertIn("sk-***", out)

    def test_phone_masked(self):
        out, _ = vx.redact("电话 13812345678")
        self.assertIn("138****5678", out)
        self.assertNotIn("13812345678", out)

    def test_plain_text_untouched(self):
        s = "基本設計書 1. 目的 本書は業務システムの刷新に伴う方針を定める。"
        out, n = vx.redact(s)
        self.assertEqual(n, 0)
        self.assertEqual(out, s)


class TestFormatContext(unittest.TestCase):
    def test_wraps_lines(self):
        ctx = vx.format_context({"text": "第一行\n\n第二行\n"})
        self.assertTrue(ctx.startswith("[屏幕文字]"))
        self.assertIn("第一行", ctx)
        self.assertIn("第二行", ctx)
        self.assertNotIn("\n\n", ctx)          # 空行被丢掉
        self.assertTrue(ctx.endswith("[/屏幕文字]"))

    def test_empty(self):
        self.assertEqual(vx.format_context({"text": ""}), "")
        self.assertEqual(vx.format_context({}), "")
        self.assertEqual(vx.format_context(None), "")  # type: ignore[arg-type]

    def test_truncates_lines(self):
        ctx = vx.format_context({"text": "\n".join(f"L{i}" for i in range(100))},
                                max_lines=10)
        self.assertIn("L9", ctx)
        self.assertNotIn("L10\n", ctx)


class TestOcrThreads(unittest.TestCase):
    """OCR 线程数必须显式传下去。

    踩过的坑：onnxruntime 默认吃满核数，12 核机上 12 线程实测 9.10s，
    显式 8 线程 4.36s——超订把速度打回一半。所以这里守住「参数真的传到了引擎」。
    """

    def setUp(self):
        self._real = sys.modules.get("rapidocr_onnxruntime")
        self.got: list[dict] = []

    def tearDown(self):
        if self._real is None:
            sys.modules.pop("rapidocr_onnxruntime", None)
        else:
            sys.modules["rapidocr_onnxruntime"] = self._real

    def _install(self, accepts_threads: bool):
        got = self.got

        if accepts_threads:
            class Fake:      # 新版签名：收 intra_op_num_threads
                def __init__(self, intra_op_num_threads=None):
                    got.append({"intra_op_num_threads": intra_op_num_threads})

                def __call__(self, _img):
                    return [], 0.0
        else:
            class Fake:      # type: ignore[no-redef]  老版签名：不收，传了就 TypeError
                def __init__(self):
                    got.append({})

                def __call__(self, _img):
                    return [], 0.0

        m = types.ModuleType("rapidocr_onnxruntime")
        m.RapidOCR = Fake
        sys.modules["rapidocr_onnxruntime"] = m

    def test_threads_passed_through(self):
        self._install(accepts_threads=True)
        v = vx.Vision(dev="", ocr_threads=6)
        v._ocr_engine()
        self.assertEqual(self.got, [{"intra_op_num_threads": 6}])

    def test_default_is_not_auto(self):
        """默认值必须是写死的 8，不能是 0/None（那等于让 onnxruntime 吃满核）。"""
        v = vx.Vision(dev="")
        self.assertEqual(v.ocr_threads, vx.OCR_THREADS)
        self.assertGreater(v.ocr_threads, 0)

    def test_old_version_falls_back(self):
        """老版本不收这个参数：退回默认构造，而不是抛异常让「看屏幕」整个不可用。"""
        self._install(accepts_threads=False)
        v = vx.Vision(dev="", ocr_threads=6)
        self.assertIsNotNone(v._ocr_engine())      # 没炸
        self.assertEqual(self.got, [{}])           # 用无参构造建了引擎


class TestTrtFallback(unittest.TestCase):
    """TensorRT 是「加分项」，任何情况下都不能让「看屏幕」整体不可用。

    真机上 engine 会因换 JetPack / 升级 TRT 而失效，也可能是别的机型拷来的，
    所以「坏 engine → 静默退回 CPU」这条必须有测试兜着。
    """

    def setUp(self):
        self._real = sys.modules.get("rapidocr_onnxruntime")
        self._tmp: list[str] = []

    def tearDown(self):
        for p in self._tmp:
            try:
                os.unlink(p)
            except OSError:
                pass
        if self._real is None:
            sys.modules.pop("rapidocr_onnxruntime", None)
        else:
            sys.modules["rapidocr_onnxruntime"] = self._real

    def _install_fake_ocr(self):
        class FakeRec:
            session = None

        class Fake:
            def __init__(self, **_kw):
                self.text_rec = FakeRec()

        m = types.ModuleType("rapidocr_onnxruntime")
        m.RapidOCR = Fake
        sys.modules["rapidocr_onnxruntime"] = m

    def test_missing_engine_keeps_cpu(self):
        self._install_fake_ocr()
        v = vx.Vision(dev="", trt_engine="/nonexistent/rec.engine")
        self.assertIsNotNone(v._ocr_engine())      # 引擎照建
        self.assertIsNone(v.trt_status())          # 但没有 TRT
        self.assertIn("不存在", v.trt_note)

    def test_broken_engine_falls_back(self):
        self._install_fake_ocr()
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=".engine", delete=False)
        f.write(b"not a tensorrt engine")
        f.close()
        self._tmp.append(f.name)

        v = vx.Vision(dev="", trt_engine=f.name)
        self.assertIsNotNone(v._ocr_engine())
        self.assertIsNone(v.trt_status())
        self.assertNotEqual(v.trt_note, "ok")      # 记下失败原因，但不抛

    def test_no_engine_means_pure_cpu(self):
        self._install_fake_ocr()
        v = vx.Vision(dev="")
        v._ocr_engine()
        self.assertIsNone(v.trt_status())
        self.assertEqual(v.trt_note, "")           # 没配就不该有噪音


class TestDhash(unittest.TestCase):
    def _frame(self, seed: int):
        rng = np.random.default_rng(seed)
        return (rng.random((108, 192, 3)) * 255).astype(np.uint8)

    def test_same_frame_same_hash(self):
        a = vx.Vision._dhash(self._frame(1))
        b = vx.Vision._dhash(self._frame(1).copy())
        self.assertEqual(a, b)

    def test_different_frame_differs(self):
        a = vx.Vision._dhash(self._frame(1))
        b = vx.Vision._dhash(self._frame(2))
        self.assertNotEqual(a, b)

    def test_gradient_not_absolute(self):
        """整体变亮不该改变哈希（dHash 看梯度，不看绝对值）。"""
        img = self._frame(3)
        brighter = np.clip(img.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        # 加常量会顶到 255，梯度在饱和处被打平，只要求大部分位不变
        h1, h2 = vx.Vision._dhash(img), vx.Vision._dhash(brighter)
        same = bin(h1 ^ h2).count("0")
        self.assertGreater(same, 200)


class TestLookGuards(unittest.TestCase):
    """look() 的两道保护：全黑图不当有效帧、异常不抛到调用方。"""

    def test_blank_frame_rejected(self):
        v = vx.Vision()
        v.grab = lambda: np.zeros((1080, 1920, 3), dtype=np.uint8)  # type: ignore
        out = v.look(force=True)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "blank")
        self.assertEqual(v.errors, 1)

    def test_grab_failure_is_soft(self):
        v = vx.Vision()

        def boom():
            raise vx.VisionError("设备被占用")

        v.grab = boom  # type: ignore
        out = v.look(force=True)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "error")
        self.assertIn("设备被占用", out["message"])

    def test_ocr_result_cached(self):
        v = vx.Vision(cache_ttl=30.0)
        frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        frame[::40, :, :] = 30          # 造点纹理，避免全白被当无效帧
        v.grab = lambda: frame  # type: ignore
        v._ocr = lambda img: {"text": "屏幕上的字", "lines": 1, "score": 0.9,  # type: ignore
                              "ocr_s": 1.0, "truncated": False}
        first = v.look(force=True)
        self.assertTrue(first["ok"])
        self.assertFalse(first["cached"])
        second = v.look(force=False)      # 同一画面 → 命中缓存
        self.assertTrue(second["cached"])
        third = v.look(force=True)        # force 绕过缓存
        self.assertFalse(third["cached"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
