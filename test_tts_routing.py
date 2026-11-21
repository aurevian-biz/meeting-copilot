#!/usr/bin/env python3
"""TTS 文本清洗 / 音色路由 / 切句的纯逻辑测试（不加载模型）。

覆盖两个真实故障：
  1. 纯标点片段进 Piper → 0 帧音频 → wave 抛 "# channels not specified"
  2. 纯数字句子（"1.5"）没有假名也没有汉字 → 落到默认音色（日语），
     中文回答里的数字被日本語読み
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_server as vs  # noqa: E402


def fake_tts(voices=("ja", "zh")):
    """绕过模型加载，只要 pick() 需要的三个属性。"""
    t = vs.Tts.__new__(vs.Tts)
    t.voices = {k: object() for k in voices}
    t.default = "ja" if "ja" in t.voices else voices[0]
    return t


class TestStrip(unittest.TestCase):
    def test_punct_only_is_empty(self):
        for s in ["。", "？", "……", "「」", " - ", "：", "\n\n"]:
            self.assertEqual(vs.strip_for_tts(s), "", s)

    def test_keeps_speakable(self):
        for s in ["3.14", "OK", "2026年", "セクション", "50%"]:
            self.assertTrue(vs.strip_for_tts(s), s)

    def test_empty_paren(self):
        self.assertEqual(vs.strip_for_tts("（ ）"), "")

    def test_markdown_noise(self):
        self.assertEqual(vs.strip_for_tts("**重要**"), "重要")


class TestPick(unittest.TestCase):
    def setUp(self):
        self.t = fake_tts()

    def test_kana_is_ja(self):
        self.assertEqual(self.t.pick("セクションを確認します。"), "ja")

    def test_han_only_is_zh(self):
        self.assertEqual(self.t.pick("今天天气很好。"), "zh")

    def test_digits_follow_hint(self):
        # 故障 2：中文上下文里 "1.5" 曾经被判成 ja
        self.assertEqual(self.t.pick("1.5", hint="zh"), "zh")
        self.assertEqual(self.t.pick("2026", hint="zh"), "zh")

    def test_digits_follow_ja_hint(self):
        self.assertEqual(self.t.pick("1.5", hint="ja"), "ja")

    def test_digits_default_zh(self):
        # 无上下文的英文/数字首句：中文模型不会读成日语音
        self.assertEqual(self.t.pick("OK"), "zh")

    def test_latin_follows_hint(self):
        self.assertEqual(self.t.pick("vLLM 就绪", hint="zh"), "zh")

    def test_ja_only_voices(self):
        t = fake_tts(("ja",))
        self.assertEqual(t.pick("1.5"), "ja")


class TestSentenceSplit(unittest.TestCase):
    def split(self, s):
        out = []
        buf = s
        while True:
            m = vs.SENT_END.search(buf)
            if not m:
                break
            out.append(buf[:m.end()])
            buf = buf[m.end():]
        if buf:
            out.append(buf)
        return out

    def test_decimal_not_split(self):
        self.assertEqual(self.split("耗时 3.14 秒。"), ["耗时 3.14 秒。"])

    def test_english_period_split(self):
        self.assertEqual(self.split("OK. Next")[0], "OK.")

    def test_chinese_period(self):
        self.assertEqual(self.split("第一句。第二句。"), ["第一句。", "第二句。"])

    def test_version_number(self):
        self.assertEqual(len([p for p in self.split("版本 2.0 已发布。") if p]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
