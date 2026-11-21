#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Unit tests for voice_probe. No microphone, faster-whisper or scipy required."""
from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import voice_probe as vp

SR = 16000


def tone(freq: float, amp: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return amp * np.sin(2 * math.pi * freq * t)


def gain_db(x: np.ndarray, y: np.ndarray) -> float:
    return vp.db_of(y) - vp.db_of(x)


# ---------------------------------------------------------------- filtering
class TestHighpass(unittest.TestCase):
    def test_rejects_bad_cutoff(self):
        with self.assertRaises(ValueError):
            vp.biquad_highpass(0, SR)
        with self.assertRaises(ValueError):
            vp.biquad_highpass(SR / 2, SR)

    def test_coeffs_normalized(self):
        b, a = vp.biquad_highpass(150, SR)
        # a highpass has unity gain at Nyquist: a coefficient sum of 0 means DC is fully blocked
        self.assertAlmostEqual(float(b.sum()), 0.0, places=6)
        self.assertAlmostEqual(float(a[0]), 1.0, places=9)

    def test_kills_50_60hz_hum(self):
        """Measured on this machine, the worst noise peaks are 60Hz + 120Hz; this is the core protection."""
        x = tone(60, 3000)
        y = vp.highpass(x, 150, SR, stages=2)
        # a 4th-order filter is theoretically about -31dB at 60Hz
        self.assertLess(gain_db(x, y), -24.0, "60Hz 应被压 24dB 以上")

    def test_passes_speech_band(self):
        x = tone(1000, 3000)
        y = vp.highpass(x, 150, SR, stages=2)
        self.assertGreater(gain_db(x, y), -2.0, "1kHz 应基本无损")

    def test_more_stages_more_attenuation(self):
        x = tone(60, 3000)
        g2 = gain_db(x, vp.highpass(x, 150, SR, stages=2))
        g4 = gain_db(x, vp.highpass(x, 150, SR, stages=4))
        self.assertLess(g4, g2 - 15.0)

    def test_no_cutoff_returns_input(self):
        x = tone(1000, 3000, 0.1)
        # stages<1 is clamped to 1 stage, but main skips the call when hp<=0; this checks the clamp does not error
        y = vp.highpass(x, 150, SR, stages=0)
        self.assertEqual(y.shape, x.shape)

    def test_output_is_finite(self):
        y = vp.highpass(np.zeros(1000), 150, SR)
        self.assertTrue(np.all(np.isfinite(y)))


# ---------------------------------------------------------------- level
class TestFrameDb(unittest.TestCase):
    def test_sine_rms_matches_theory(self):
        amp = 3000.0
        f = vp.frame_db(tone(1000, amp, 1.0), SR)
        expected = 20 * math.log10((amp / math.sqrt(2)) / 32768)
        self.assertAlmostEqual(float(np.median(f)), expected, delta=1.0)

    def test_silence_is_very_low(self):
        f = vp.frame_db(np.zeros(SR), SR)
        self.assertLess(float(f.max()), -100)

    def test_short_input_returns_empty(self):
        self.assertEqual(vp.frame_db(np.zeros(10), SR).size, 0)

    def test_frame_count(self):
        # 1s, 20ms frames, 10ms hop → 1 + (16000-320)/160 = 99
        f = vp.frame_db(np.zeros(SR), SR)
        self.assertEqual(f.size, 99)

    def test_db_of_empty(self):
        self.assertEqual(vp.db_of(np.zeros(0)), -999.0)


class TestNoiseFloor(unittest.TestCase):
    def test_percentile_ignores_loud_tail(self):
        frames = np.array([-50.0] * 90 + [-10.0] * 10)
        self.assertAlmostEqual(vp.noise_floor(frames), -50.0, places=3)

    def test_empty(self):
        self.assertEqual(vp.noise_floor(np.zeros(0)), -999.0)

    def test_threshold_margin(self):
        self.assertAlmostEqual(vp.vad_threshold(-50, 6), -44.0)

    def test_threshold_has_floor(self):
        # in total silence the floor is -120; the threshold must not follow it down to -114, it has to fall back to -60
        self.assertAlmostEqual(vp.vad_threshold(-120, 6), -60.0)


# ---------------------------------------------------------------- segmentation
def frames_from(levels: list[float]) -> np.ndarray:
    return np.array(levels, dtype=float)


class TestSegmentSpeech(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(vp.segment_speech(np.zeros(0), -40), [])

    def test_all_silence(self):
        self.assertEqual(vp.segment_speech(np.full(200, -60.0), -40), [])

    def test_single_utterance(self):
        # 10ms per frame, 50 frames = 0.5s of speech
        f = frames_from([-60.0] * 20 + [-20.0] * 50 + [-60.0] * 20)
        segs = vp.segment_speech(f, -40)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0].start, 0.20, places=3)
        self.assertAlmostEqual(segs[0].duration, 0.50, places=3)

    def test_word_gap_is_bridged(self):
        """An inter-word pause shorter than min_gap must not cause segmentation, or one sentence shatters into pieces."""
        f = frames_from([-20.0] * 30 + [-60.0] * 20 + [-20.0] * 30)
        segs = vp.segment_speech(f, -40, min_gap_s=0.40)
        self.assertEqual(len(segs), 1)

    def test_long_gap_splits(self):
        # 100 silent frames in the middle = 1.0s > 0.4s
        f = frames_from([-20.0] * 30 + [-60.0] * 100 + [-20.0] * 30)
        segs = vp.segment_speech(f, -40, min_gap_s=0.40)
        self.assertEqual(len(segs), 2)

    def test_short_blip_dropped(self):
        """A 20-frame = 0.2s burst (knocking on the desk) must be dropped."""
        f = frames_from([-60.0] * 10 + [-20.0] * 20 + [-60.0] * 10)
        self.assertEqual(vp.segment_speech(f, -40, min_speech_s=0.30), [])

    def test_peak_recorded(self):
        f = frames_from([-30.0] * 40 + [-15.0] * 10 + [-30.0] * 40)
        segs = vp.segment_speech(f, -40)
        self.assertEqual(segs[0].peak_db, -15.0)

    def test_utterance_at_end_is_closed(self):
        f = frames_from([-60.0] * 10 + [-20.0] * 40)
        segs = vp.segment_speech(f, -40)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0].end, 0.50, places=3)

    def test_multiple_utterances_timestamps_ordered(self):
        f = frames_from([-20.0] * 40 + [-60.0] * 60 + [-20.0] * 40 + [-60.0] * 60 + [-20.0] * 40)
        segs = vp.segment_speech(f, -40)
        self.assertEqual(len(segs), 3)
        self.assertLess(segs[0].end, segs[1].start)
        self.assertLess(segs[1].end, segs[2].start)

    def test_reported_noise_floor_case(self):
        """Reproduces a local measurement: floor -30.8, speech -21, threshold -24.8 — can a segment still be cut?"""
        f = frames_from([-30.8] * 30 + [-21.0] * 50 + [-30.8] * 30)
        thr = vp.vad_threshold(vp.noise_floor(f), 6.0)
        segs = vp.segment_speech(f, thr)
        self.assertEqual(len(segs), 1, "实测电平下必须能切出语音段")


class TestPeakMargin(unittest.TestCase):
    """A false trigger measured after the gain was lowered: 0.41s, peak -41.5, floor -56.6."""

    FLOOR = -56.6
    BLIP = [-56.6] * 20 + [-41.5] * 41 + [-56.6] * 20
    SPEECH = [-56.6] * 20 + [-14.0] * 41 + [-56.6] * 20

    def test_low_peak_transient_rejected(self):
        thr = vp.vad_threshold(self.FLOOR, 6.0)
        segs = vp.segment_speech(frames_from(self.BLIP), thr,
                                 floor_db=self.FLOOR, min_peak_margin_db=18.0)
        self.assertEqual(segs, [], "门限附近的小瞬态不能当语音")

    def test_real_speech_accepted(self):
        thr = vp.vad_threshold(self.FLOOR, 6.0)
        segs = vp.segment_speech(frames_from(self.SPEECH), thr,
                                 floor_db=self.FLOOR, min_peak_margin_db=18.0)
        self.assertEqual(len(segs), 1)

    def test_without_floor_no_peak_filter(self):
        """With no floor passed, keep the old behavior so existing callers are not broken."""
        thr = vp.vad_threshold(self.FLOOR, 6.0)
        segs = vp.segment_speech(frames_from(self.BLIP), thr, floor_db=None)
        self.assertEqual(len(segs), 1)

    def test_margin_boundary(self):
        # a margin of exactly 18dB must be kept (only < margin is dropped)
        f = [-56.6] * 20 + [-38.6] * 41 + [-56.6] * 20
        thr = vp.vad_threshold(self.FLOOR, 6.0)
        segs = vp.segment_speech(frames_from(f), thr,
                                 floor_db=self.FLOOR, min_peak_margin_db=18.0)
        self.assertEqual(len(segs), 1)


# ---------------------------------------------------------------- audio IO
class TestReadWav(unittest.TestCase):
    def _write(self, data: np.ndarray, sr: int = SR, ch: int = 1) -> Path:
        p = Path(tempfile.mkdtemp()) / "t.wav"
        arr = np.clip(data, -32768, 32767).astype(np.int16)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(arr.tobytes())
        return p

    def test_roundtrip_mono(self):
        p = self._write(tone(440, 1000, 0.1))
        d, sr = vp.read_wav(p)
        self.assertEqual(sr, SR)
        self.assertEqual(d.size, 1600)

    def test_stereo_downmix(self):
        mono = tone(440, 1000, 0.1)
        stereo = np.repeat(mono, 2)
        p = self._write(stereo, ch=2)
        d, _ = vp.read_wav(p)
        self.assertEqual(d.size, 1600)
        self.assertAlmostEqual(float(np.max(np.abs(d))), 1000, delta=5)

    def test_8bit_rejected(self):
        p = Path(tempfile.mkdtemp()) / "t.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)
            w.setframerate(SR)
            w.writeframes(b"\x80" * 100)
        with self.assertRaises(ValueError):
            vp.read_wav(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
