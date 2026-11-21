"""打断判定链的端到端验证：真 VAD + 真 ASR + 真 TTS。

只有 hub 和 LLM 是替身：hub 用合成语音喂帧，代替麦克风；LLM 计数代替真实生成。
这样不用人对着麦克风喊，也能验证「串音不打断、真人插话打断、静默无反应」。

在 Jetson 上跑（需要部署目录里的 voice_server 和模型）：
    /opt/meeting-copilot/data/asr-venv/bin/python tools/barge_e2e.py
"""

import io
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice

APP = "/opt/meeting-copilot"
sys.path.insert(0, APP)
import voice_server as vs  # noqa: E402

voice = PiperVoice.load("/opt/meeting-copilot/data/tts-models/zh_CN-huayan-medium.onnx")
LINE = "今天天气不错，我们出去走走吧。"


def tts16k(text):
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        voice.synthesize_wav(text, wf, syn_config=None)
    bio.seek(0)
    with wave.open(bio, "rb") as r:
        sr = r.getframerate()
        x = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32)
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)


class FakeHub:
    def __init__(self):
        self.subs = []
        self.lock = threading.Lock()

    def subscribe(self, pre_s=0.0, pre_frames=None):
        q = queue.Queue()
        for f in (pre_frames or []):
            q.put_nowait(f)
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def feed(self, audio, pad_s=1.0):
        fl = 320
        for i in range(0, len(audio) - fl, fl):
            for q in list(self.subs):
                q.put_nowait(audio[i:i + fl])
        for _ in range(int(pad_s / 0.02)):
            for q in list(self.subs):
                q.put_nowait(np.zeros(fl, dtype=np.int16))


class StubRunner:
    def __init__(self):
        self.calls = 0

    def run(self, emit, **kw):
        self.calls += 1


def main():
    asr = vs.Asr(Path(vs.DEFAULT_ASR_DIR), 8, "auto")
    hub, speak, runner = FakeHub(), vs.SpeakState(), StubRunner()
    evs = []
    mon = vs.BargeMonitor(hub, asr, speak, runner, [], evs.append,
                          threading.Lock(), str(vs.DEFAULT_VAD),
                          min_db=-42.0, margin=8.0)
    mon.start()
    fails = []

    def check(name, cond, detail):
        print(f"{name:10} {detail:44} {'PASS' if cond else 'FAIL'}", flush=True)
        if not cond:
            fails.append(name)

    # 1) 外放：麦克风听到的就是助理自己的声音，绝不能打断
    speak.begin("t1")
    speak.add(8.0, LINE)
    hub.feed(tts16k(LINE))
    time.sleep(1.4)
    check("外放串音", mon.hits == 0 and mon.echo_drops > 0 and not speak.cancelled,
          f"hits={mon.hits} echo={mon.echo_drops} cancelled={speak.cancelled}")

    # 2) 真人插话：打断 + 起新一轮
    h0 = mon.hits
    speak.begin("t2")
    speak.add(8.0, LINE)
    evs.clear()
    hub.feed(tts16k("等一下，我还想问个问题。"))
    time.sleep(1.4)
    check("真插话", mon.hits == h0 + 1 and speak.cancelled and len(evs) == 1,
          f"hits={mon.hits} cancelled={speak.cancelled} 事件={len(evs)} llm={runner.calls}")

    # 3) 静默：不该有任何反应
    h1, e1 = mon.hits, mon.echo_drops
    speak.begin("t3")
    speak.add(8.0, LINE)
    hub.feed(np.zeros(32000, dtype=np.int16))
    time.sleep(1.0)
    check("静默", mon.hits == h1 and mon.echo_drops == e1,
          f"hits={mon.hits} echo={mon.echo_drops}")

    # 4) 已经播完：不再判定（播报时钟归零）
    speak.begin("t4")
    speak.add(0.1, LINE)
    time.sleep(0.3)
    h2 = mon.hits
    hub.feed(tts16k("等一下，我还想问个问题。"))
    time.sleep(0.8)
    check("播完不判", mon.hits == h2,
          f"playing={speak.playing()} hits={mon.hits}")

    mon.stop()
    print("RESULT", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
