#!/usr/bin/env python3
"""会议字幕：采集卡音频 → 断句 → SenseVoice → 中日双语 → 网页滚动字幕。

与 voice_server 完全解耦，各占一个音频设备（这里用 MS2109 采集卡），互不干扰。

    ./asr-venv/bin/python caption_server.py                       # 采集卡 → 8096
    ./asr-venv/bin/python caption_server.py --wav test.wav        # 离线跑一个文件
    ./asr-venv/bin/python caption_server.py --device plughw:CARD=H340,DEV=0

设计取舍：
- 断句用能量门限而不是 Silero：会议音频直接从 HDMI 取，是没有回声和噪声的干净信号，
  能量法足够，且没有 VAD 状态机跨段残留的问题。
- 每段最长 3 秒强制切：发言人不换气时不至于憋一大段一次蹦出来。实测 ASR 对 6 秒音频
  只要 0.3 秒，真正的延迟在"等段结束"，所以段越短出字越快。
- **原文先出，译文后补**（两段式）：识别完立刻推一行只有原文的字幕，译文由独立线程
  算完再以补丁形式回填同一个 seq。看字延迟从"ASR+翻译"降到"只有 ASR"。
- 三段流水线解耦：采集线程只断句，ASR 线程只识别，翻译线程只翻译。任何一环变慢都
  只在自己队列里堆积，不会把采集卡住（旧版翻译 1 秒期间采集帧在队列里干等）。
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

# 必须放在模块级：本文件开了 `from __future__ import annotations`，
# 注解是字符串，FastAPI 用 get_type_hints 在模块全局里找类型；
# 只写在函数内部的话，它认不出 Request，会把它当 query 参数。
from fastapi import Request  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  目录约定：MC_DATA 可覆盖，见 paths.py
import voice_probe as vp  # noqa: E402
from voice_server import Asr, clean_text  # noqa: E402

SR = 16000
FRAME_S = 0.02
FRAME_N = int(SR * FRAME_S)

LANG_TAG = re.compile(r"<\|(zh|ja|en|ko|yue)\|>")
KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
CJK = re.compile(r"[\u4e00-\u9FFF]")
# 有实质内容的字符。噪声段被 min_s=0.25 放过时，SenseVoice 会返回 " 。" 这种
# 只有标点的结果——它在页面上是一行空字幕，还白白占一次翻译调用和一行 seq。
SUBSTANCE = re.compile(r"[0-9A-Za-z\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")


# ------------------------------------------------------------------ 采集
def resolve_pw_target(hint: str = "MACROSILICON") -> str:
    """从 pw-dump 里找采集卡的 source 节点名。

    PipeWire 把这台机器上的采集卡认成了默认录音源并长期占着 PCM，
    ALSA 直连(arecord -D plughw)会一直「设备或资源忙」；走 PipeWire 取流则不冲突。
    """
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=8).stdout
        for node in json.loads(out):
            props = (node.get("info") or {}).get("props") or {}
            name = props.get("node.name") or ""
            if props.get("media.class") == "Audio/Source" and hint.lower() in name.lower():
                return name
    except Exception:  # noqa: BLE001
        pass
    return "@DEFAULT_AUDIO_SOURCE@"


def resolve_pw_source(hint: str = "MACROSILICON") -> str:
    return resolve_pw_target(hint or "MACROSILICON")


def resolve_pw_sink(hint: str = "H340") -> str:
    """找输出节点（耳机）。默认按名字找 H340，找不到就用系统默认输出。"""
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=8).stdout
        for node in json.loads(out):
            props = (node.get("info") or {}).get("props") or {}
            name = props.get("node.name") or ""
            if props.get("media.class") == "Audio/Sink" and hint.lower() in name.lower():
                return name
    except Exception:  # noqa: BLE001
        pass
    return "@DEFAULT_AUDIO_SINK@"


class Capture:
    """定长 20ms 帧。读线程独占设备，别处不能再开同一个 card。

    backend="pw" 走 PipeWire（不独占，和其他音频客户端共存）；
    backend="alsa" 走 arecord 直连（PipeWire 没抓这个设备时可用）。
    """

    def __init__(self, device: str, sr: int = SR, *, backend: str = "pw",
                 target: str | None = None):
        self.device = device
        self.sr = sr
        self.backend = backend
        self.target = target
        self.frames: queue.Queue = queue.Queue(maxsize=600)   # 12 秒，满了丢最旧
        self.proc: subprocess.Popen | None = None
        self.errors = 0
        self.last_error = ""
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _cmd(self) -> list[str]:
        if self.backend == "pw":
            tgt = self.target or resolve_pw_target()
            self.target = tgt
            return ["pw-record", "--target", tgt, "--rate", str(self.sr),
                    "--channels", "1", "--format", "s16", "-"]
        return ["arecord", "-q", "-D", self.device, "-f", "S16_LE",
                "-r", str(self.sr), "-c", "1", "-t", "raw"]

    def start(self) -> None:
        self.proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self._t = threading.Thread(target=self._run, name="capture", daemon=True)
        self._t.start()

    def _run(self) -> None:
        assert self.proc and self.proc.stdout
        nbytes = FRAME_N * 2
        buf = b""
        while not self._stop.is_set():
            chunk = self.proc.stdout.read(nbytes - len(buf))
            if not chunk:
                # 进程退了，把它最后那句 stderr 留下来 —— 否则只有"设备忙"三个字可猜
                err = ""
                if self.proc.poll() is not None and self.proc.stderr:
                    try:
                        err = (self.proc.stderr.read() or b"").decode("utf-8", "replace").strip()
                    except Exception:  # noqa: BLE001
                        pass
                self.last_error = err.splitlines()[-1][:160] if err else "无输出"
                self.errors += 1
                break
            buf += chunk
            if len(buf) < nbytes:
                continue
            frame = np.frombuffer(buf[:nbytes], dtype=np.int16)
            buf = buf[nbytes:]
            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MonitorPipe:
    """把 HDMI 进来的声音原样回放到本机耳机（监听 / 回放）。

    用途：电脑的 HDMI 输出全部进了采集卡，人反而听不到。这里在 Jetson 上
    把采集卡的音频接出来播到耳机，等于给 HDMI 链路挂了一个监听口。

    实现是 pw-record | pw-cat 两条子进程直接对接，Python 不参与搬运：
    采样数据在内核管道里过，延迟只有两端的缓冲（各 20ms），不占 CPU。

    为什么能同时跑：PipeWire 允许多个客户端读同一个 source，字幕服务自己
    那条 16k 采集流不受影响（实测两路并存无冲突）。
    """

    def __init__(self, src: str, sink: str, rate: int = 44100,
                 latency_ms: int = 20):
        self.src = src
        self.sink = sink
        self.rate = rate
        self.latency_ms = latency_ms
        self.rec: subprocess.Popen | None = None
        self.play: subprocess.Popen | None = None
        self.started_at = 0.0
        self.starts = 0
        self.last_error = ""
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self.rec and self.play
                        and self.rec.poll() is None and self.play.poll() is None)

    def start(self) -> bool:
        with self._lock:
            if self.rec and self.play and self.rec.poll() is None:
                return False
            lat = f"{self.latency_ms}ms"
            rec_cmd = ["pw-record", "--target", self.src, "--rate", str(self.rate),
                       "--channels", "2", "--format", "s16", "--latency", lat, "-"]
            play_cmd = ["pw-cat", "--playback", "--target", self.sink,
                        "--rate", str(self.rate), "--channels", "2",
                        "--format", "s16", "--latency", lat, "-"]
            try:
                self.rec = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, bufsize=0)
                self.play = subprocess.Popen(play_cmd, stdin=self.rec.stdout,
                                             stderr=subprocess.PIPE)
                if self.rec.stdout:
                    self.rec.stdout.close()     # 让 play 独占读端，避免 EOF 收不到
            except FileNotFoundError as e:
                self.last_error = f"找不到 {e.filename}（需要 pipewire-bin）"
                self._kill()
                return False
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)[:120]
                self._kill()
                return False
            self.started_at = time.time()
            self.starts += 1
            self.last_error = ""
        return True

    def stop(self) -> None:
        with self._lock:
            self._kill()
        print("[monitor] 停止回放", flush=True)

    def _kill(self) -> None:
        for p in (self.play, self.rec):
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
        for p in (self.play, self.rec):
            if p is not None:
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        # 子进程被 kill 时 drain 掉 stderr，否则僵尸管道会攒一堆符号
        for p in (self.play, self.rec):
            if p is not None and p.stderr:
                try:
                    err = p.stderr.read() or b""
                    if err and not self.last_error:
                        self.last_error = err.decode("utf-8", "replace").strip()[:120]
                except (OSError, ValueError):
                    pass
        self.rec = None
        self.play = None

    def status(self) -> dict:
        return {"on": self.running, "src": self.src, "sink": self.sink,
                "rate": self.rate, "latency_ms": self.latency_ms,
                "starts": self.starts,
                "up_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
                "error": self.last_error}


# ------------------------------------------------------------------ 断句
class Segmenter:
    """能量门限 + 静音挂起。本底用开头的安静样本估，之后快降慢升。

    硬切（到 max_s）时会回头找能量最低的一帧下刀：切在音节中间会让 ASR 输出半截词，
    "天真好"被切成"天真"就会译成"天空"。往回退到音节间隙只需几百毫秒，但错字少很多。

    停顿门限随段长分层（2026-11-21）：
    固定 0.42s 门限有个两难——设紧了，说话人词间的小停顿就被切碎
    （实测 "140 亿美元…" 被切成 "…况且140。" + "亿美元对台均售…"，数字和单位分家）；
    设松了，一句话说完要干等近 1 秒才出字幕，延迟全砸在等待上。
    所以按"这一段已经攒了多久"来定门槛：
      有声 < mid_s  → 要等 hold_long_s 的长停顿才切。段还短，宁可跟下一句并一起，也不切碎。
      有声 < long_s → 用基准 hold_s，0.4s 上下大致对应正常句间停顿。
      有声 ≥ long_s → hold_short_s 就切。已经拖够长，短停顿也认，延迟不能再涨。
    max_s 仍是兜底上限，但有了分层，撞上限从"常态"退化成"少数情况"。
    """

    FLOOR_MIN, FLOOR_MAX = -70.0, -32.0

    def __init__(self, margin_db: float = 8.0, hold_s: float = 0.42,
                 min_s: float = 0.25, max_s: float = 3.0, pre_s: float = 0.25,
                 warm_s: float = 1.2, soft_s: float = 0.7,
                 hold_long_s: float = 0.90, hold_short_s: float = 0.30,
                 mid_s: float = 1.20, long_s: float = 2.40):
        self.margin_db = margin_db
        self.hold_frames = max(1, int(hold_s / FRAME_S))
        self.min_frames = max(1, int(min_s / FRAME_S))
        self.max_frames = max(1, int(max_s / FRAME_S))
        self.pre_frames = max(0, int(pre_s / FRAME_S))
        self.warm_frames = max(1, int(warm_s / FRAME_S))
        self.soft_frames = max(0, int(soft_s / FRAME_S))
        # 分层门限：长段门槛不小于基准，短段门槛不大于基准，防止有人传反了把逻辑搞死
        self.hold_long_frames = max(self.hold_frames, int(hold_long_s / FRAME_S))
        self.hold_short_frames = min(self.hold_frames,
                                     max(1, int(hold_short_s / FRAME_S)))
        self.mid_frames = max(1, int(mid_s / FRAME_S))
        self.long_frames = max(self.mid_frames + 1, int(long_s / FRAME_S))

        self.floor = -70.0
        self._warm = 0
        self._samples: list[float] = []
        self._pre: collections.deque = collections.deque(maxlen=max(1, self.pre_frames))
        self._pre_db: collections.deque = collections.deque(maxlen=max(1, self.pre_frames))
        self._cur: list[np.ndarray] = []
        self._cur_db: list[float] = []
        self._sil = 0
        self._voiced_run = 0
        self.last_db = -99.0
        self.level_db = -99.0
        self.last_cut = ""          # "silence"=自然停顿切 / "hard"=撞上限硬切

    def threshold(self) -> float:
        return self.floor + self.margin_db

    def hold_need(self) -> int:
        """当前这一段需要多久静音才允许断句（帧数）。段越长，要求越松。"""
        voiced = len(self._cur) - self._sil
        if voiced < self.mid_frames:
            return self.hold_long_frames
        if voiced < self.long_frames:
            return self.hold_frames
        return self.hold_short_frames

    def _clamp(self, v: float) -> float:
        return min(self.FLOOR_MAX, max(self.FLOOR_MIN, v))

    def push(self, frame: np.ndarray, db: float) -> np.ndarray | None:
        self.last_db = db
        # 电平显示用慢衰减包络，别让它闪；下限 -99 免得数字静音时一路掉到 -290
        self.level_db = max(-99.0, db, self.level_db - 1.2)

        # 开头 1.2 秒只估本底、不判语音；数字静音(-91)不是有效样本
        if self._warm < self.warm_frames:
            self._warm += 1
            if db > -80.0:
                self._samples.append(db)
            if self._warm >= self.warm_frames and self._samples:
                self.floor = self._clamp(float(np.percentile(self._samples, 10)))
            self._pre.append(frame)             # 预热期也要留预补，别丢了第一句开头
            self._pre_db.append(db)
            return None

        # 快降慢升：安静时迅速贴住本底，说话时几乎不动。
        # 数字静音（-80 以下，HDMI 无音频流时的底噪）不是"房间本底"，不参与更新，
        # 否则 floor 会被一路拖到下限，门限烂掉，连空调声都当成人说话。
        if db > -80.0:
            if db < self.floor:
                self.floor = self._clamp(0.70 * self.floor + 0.30 * db)
            elif db < self.floor + 4.0:
                self.floor = self._clamp(0.995 * self.floor + 0.005 * db)

        thr = self.threshold()
        voiced = db > thr

        if not self._cur:
            self._pre.append(frame)
            self._pre_db.append(db)
            if voiced:
                self._voiced_run += 1
                if self._voiced_run >= 3:        # 连续 60ms 才认，挡咔哒声
                    self._cur = list(self._pre)
                    self._cur_db = list(self._pre_db)
                    self._pre.clear()
                    self._pre_db.clear()
                    self._sil = 0
                    self._voiced_run = 0
            else:
                self._voiced_run = 0
            return None

        self._cur.append(frame)
        self._cur_db.append(db)
        self._sil = 0 if voiced else self._sil + 1

        if self._sil >= self.hold_need() and len(self._cur) - self._sil >= self.min_frames:
            self.last_cut = "silence"
            return self._flush()
        if len(self._cur) >= self.max_frames:
            self.last_cut = "hard"
            return self._flush(look_back=self.soft_frames)
        return None

    def draft(self) -> tuple[np.ndarray, float] | None:
        """当前这一段的草稿音频（还没断句）。

        离线 ASR 必须等整段说完才能出字，一段最长 max_s 秒，屏幕上就是
        「动两秒、空四五秒」。草稿让屏幕在说话过程中先出灰字，段结束再换成正式结果。
        """
        if not self._cur:
            return None
        return np.concatenate(self._cur), len(self._cur) * FRAME_S

    def _flush(self, look_back: int = 0) -> np.ndarray:
        """收尾。look_back>0 时在尾巴里找能量最低处下刀，剩下的留给下一段。"""
        cur, cur_db = self._cur, self._cur_db
        if look_back and len(cur) > look_back + self.min_frames:
            tail = cur_db[-look_back:]
            cut = len(cur) - look_back + int(np.argmin(tail))
            if cut >= self.min_frames and len(cur) - cut >= 1:
                seg = np.concatenate(cur[:cut]) if cut else np.zeros(0, dtype=np.int16)
                self._cur = cur[cut:]
                self._cur_db = cur_db[cut:]
                self._sil = 0
                self._pre.clear()
                self._pre_db.clear()
                return seg
        seg = np.concatenate(cur) if cur else np.zeros(0, dtype=np.int16)
        self._cur = []
        self._cur_db = []
        self._pre.clear()
        self._pre_db.clear()
        self._sil = 0
        return seg


def seg_tuning(args) -> dict:
    """分层停顿的四个调参项。集中一处，实时采集和 --wav 离线两条路径不会跑偏。"""
    return {"hold_long_s": getattr(args, "hold_long", 0.90),
            "hold_short_s": getattr(args, "hold_short", 0.30),
            "mid_s": getattr(args, "seg_mid", 1.20),
            "long_s": getattr(args, "seg_long", 2.40)}


# ------------------------------------------------------------------ 翻译
# ------------------------------------------------------------------ 去口癖
# 只清洗「送进翻译的文本」，不改页面显示的原文：会议里逐字稿本身有价值，
# 但语气词会让翻译模型把「嗯…那个」当成实词译出来（实测会把「嗯」译成
# 「はい/ええ」，把「あの」译成「那个」，整句语义被带偏）。
# 名单刻意保守：只收纯语气词。「那个项目」的「那个」、「あの人」的「あの」
# 后面跟的是实词，靠"前后必须是标点或句界"这条规则天然排除，不必进名单猜。
_FILLERS = (
    "嗯嗯", "嗯", "呃", "额", "唔", "唉", "哎",
    "えーと", "えっと", "えー", "あのー", "そのー", "んー", "あの", "その",
    "umm", "um", "uhh", "uh", "erm", "er", "hmm", "ah",
)
_SEP = r"[\s，,、。．.!！?？…~～]"
_FILLER_RE = re.compile(
    rf"(?<![^{_SEP[1:-1]}])"                       # 前面是句首或标点
    rf"(?:{'|'.join(map(re.escape, _FILLERS))})"   # 一个语气词
    rf"(?![^{_SEP[1:-1]}])"                        # 后面也是标点或句尾
    rf"{_SEP}*",                                   # 顺手吃掉后面的分隔符
    re.I)


def strip_fillers(text: str) -> str:
    """删掉纯语气词，给翻译用。全句都是语气词时原样返回，绝不返回空串。

    「嗯，那个项目怎么样了」→「那个项目怎么样了」
    「あの人は誰」→ 不动（あの 后面是实词）
    """
    if not text:
        return text
    out = _FILLER_RE.sub("", text)
    out = re.sub(rf"^{_SEP}+", "", out)            # 残留的行首标点
    out = re.sub(r"([，,、])[，,、\s]+", r"\1", out)  # 「，，」→「，」
    out = re.sub(r"[，,、]\s*([。！？!?])", r"\1", out)
    out = out.strip()
    if not re.search(r"[^\s，,、。．.!！?？…~～]", out):
        return text                                # 删完了就退回原文
    return out


# 译完自查：源语言的特征还留在译文里 = 没译干净。
# 中日互译防不住照抄，prompt 只能降低概率，必须有个确定性的检查兜底。
# 日译中查假名是**零误报**的；中译日只能查简体专用字形（日语用繁体或不同字形：
# 說/説、對/対、這/这、時/时、個/个）。
# 千万别把「会」「都」「那」放进来——它们在日语里是常用汉字，会误伤正常译文。
_LEFTOVER = {
    "zh": re.compile(r"[\u3040-\u30FF\uFF66-\uFF9F]"),
    "ja": re.compile(r"[们给这说对很没让从把么吗呢吧啊嗯呃时个]"),
}
_RETRY_HINT = {
    "zh": "【注意】上一版译文混入了日文。这次必须从头到尾都是中文，"
          "一个日文词、一个假名都不许留（人名机构名除外）。",
    "ja": "【注意】上一版译文混入了中文。这次必须从头到尾都是日语，"
          "一个中文词都不许留（简体字「们给这说对很没让从把么吗呢吧」都是错的）。",
}
TR_STATS = {"retry": 0, "fixed": 0}         # 供 /api/status 观察重试命中率


def lang_clean(trans: str, target: str) -> bool:
    """译文里有没有源语言残留。没有可判据的语言直接放过。

    中译日的字符名单只能抓「混写」，抓不住**整句照抄**——「我是主持人田英和」
    每个字都是日语也有的汉字，按字符查是干净的，可它就是没翻译。
    所以补一条结构性判据：一句话长度的日语不可能一个假名都没有
    （助词 は/が/を/に/の、语尾 です/ます 必然带假名）。纯汉字 ≥6 字判脏。

    日译中反过来要防误伤：专有名词按规则可以保留原文写法（「深セン」是深圳的
    日文写法），少量假名夹在大段中文里是正常的，不该为此白跑一次重试。
    """
    if target == "ja" and len(trans) >= 6 and not re.search(r"[\u3040-\u30FF]", trans):
        return False
    if target == "zh":
        kana = len(re.findall(r"[\u3040-\u30FF]", trans))
        if kana == 0:
            return True
        cjk = len(re.findall(r"[\u4e00-\u9FFF]", trans))
        return kana <= 3 and cjk >= kana * 2      # 少量假名 + 大段汉字 = 专有名词
    pat = _LEFTOVER.get(target)
    return True if pat is None else pat.search(trans) is None


# Magnitude 的 /inference/v1 要 Bearer 鉴权，vLLM 不需要。由 --llm-key 设置。
_LLM_KEY = ""

# 翻译 prompt 的档位：full（带反例的长版，7B 时代调出来的）/ short（35B 短版）。
# 实测见 README 4.19：短版把 system 从 ~250 token 压到 ~70 token。
TR_STYLE = "full"

# 最近一次 translate_stream 拿到的引擎侧 timings（prompt_n / cache_n / prompt_ms）。
# 只给离线工具做判据用，不参与业务逻辑。
LAST_TIMINGS: dict = {}


def _llm_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _LLM_KEY:
        h["Authorization"] = f"Bearer {_LLM_KEY}"
    return h


def _tr_system_short(target: str) -> str:
    """35B 专用短版（~100 字）。只保留三条硬规则，示例和语气词规则全砍。

    为什么可以砍：完整的 `_tr_system` 是靠「给反例 + 译完自查」压住中日汉字互抄，
    但这份 prompt 有 250~300 token，是每句请求里最大的一块固定成本。35B-A3B 的
    prefill 要读几乎全部专家，token 数直接换算成毫秒，所以先量「砍掉示例还守不守得住」。
    三条不能砍的：
    - 纯目标语言 + 中日共用汉字的陷阱（不给反例时最弱的一环，靠措辞硬顶）
    - 按上文纠同音错字（否则忠实翻译错字）
    - 半句译半句（段被切短后大量是半句）
    """
    zh = (
        "你是中日会议同传。把【本句】译成纯中文，只输出译文一行，不加解释。\n"
        "按【上文】纠正识别错字。中日共用汉字，禁止把日语原样当中文；人名可留原文。半句译半句。"
    )
    ja = (
        "你是中日会议同传。把【本句】译成纯日语，只输出译文一行，不加解释。\n"
        "按【上文】纠正识别错字。中日共用汉字，禁止把中文原样当日语"
        "（不得出现「们」「给」「这」「了」）；人名可留原文。半句译半句。"
    )
    if target == "zh":
        return zh
    if target == "ja":
        return ja
    name = target
    return (f"你是会议同传。把【本句】译成纯{name}，只输出译文一行；"
            f"按上文纠正识别错字；半句译半句。")


def _tr_system(target: str, style: str | None = None) -> str:
    """翻译用 system prompt。两种目标语言各一份，前缀固定才能命中 vLLM 缓存。

    为什么写成这样（都是实测踩出来的）：
    - **中日共用汉字是最容易翻车的地方**。7B 模型遇到「坚持」「好处」「主持人」
      这类词会直接照抄，输出一句汉字完全正确、语法全错的"日语"。光说
      "必须是纯目标语言"守不住，必须给具体反例，并且要求译完自查。
    - **ASR 同音错字要显式授权它改**。不说的话模型会忠实翻译错字
      （「中方科意」→「中国側は科意」，一个不存在的词）。
    - 反例用实测抓到的真实坏输出，模型的模仿倾向比抽象规则强。
    """
    if (style or TR_STYLE) == "short":
        return _tr_system_short(target)
    zh = (
        "你是中日会议同传。把【本句】译成中文。\n"
        "第一步：按【上文】纠正语音识别的同音错字，译纠正后的意思而不是错字本身"
        "（如「科意」应为「刻意」、「三强贵物」应为「三强会晤」）。\n"
        "输出规则：\n"
        "1. 只输出译文一行，不要引号、不要解释、不要附原文。\n"
        "2. 译文必须 100% 是中文。中日共用汉字，最容易犯的错是把日语句子原样"
        "当中文输出。译完自查：只要有一处还是日语语法或残留假名，整句重写。\n"
        "3. 人名、机构名、缩写可保留原文写法，其余不得保留任何日文词。\n"
        "4. 原文是半句就译成半句，不要补主语、不要自行补完。\n"
        "5. 语气词（えーと、あの、ね、かな）略去不译；有实义的词要译。\n"
        "示例：\n"
        "原文：身体面にアメリカは坚持するべきです\n"
        "错误：在实质层面，美国应该坚持するべきです\n"
        "正确：在实质层面，美国应当坚持立场。\n"
        "\n"
        "原文：深zのアプリをどう評価するか\n"
        "错误：深zのアプリで何点给深zに評価するか\n"
        "正确：该怎么评价深z这款应用？"
    )
    ja = (
        "你是中日会议同传。把【本句】译成日语。\n"
        "第一步：按【上文】纠正语音识别的同音错字，译纠正后的意思而不是错字本身"
        "（如「科意」应为「刻意」、「三强贵物」应为「三强会晤」）。\n"
        "输出规则：\n"
        "1. 只输出译文一行，不要引号、不要解释、不要附原文。\n"
        "2. 译文必须 100% 是日语。中日共用汉字，最容易犯的错是把中文句子原样"
        "当日语输出。译完自查：只要有一处还是中文语法（如「们」「给」「这」「很」"
        "「了」），整句重写。\n"
        "3. 人名、机构名、缩写可保留原文写法，其余不得保留任何中文词。\n"
        "4. 原文是半句就译成半句，不要补主语、不要自行补完。\n"
        "5. 语气词（嗯、呃、那个）略去不译；有实义的词要译。\n"
        "示例：\n"
        "原文：我是主持人谭英文\n"
        "错误：私は主持人谭英文です\n"
        "正确：私は司会の谭英文です。\n"
        "\n"
        "原文：中方刻意给韩国好处\n"
        "错误：中国側は故意に韓国に给好处\n"
        "正确：中国側は意図的に韓国に便宜を図った。"
    )
    if target == "zh":
        return zh
    if target == "ja":
        return ja
    name = target
    return (f"你是会议同传。把【本句】译成{name}。只输出译文一行，不要引号或解释；"
            f"译文必须是纯{name}，不得保留任何原文词（专有名词除外）；"
            f"原文可能有语音识别错字，按上下文纠正后翻译；半句就译成半句，不要补完。")


# Qwen3.5 系的 chat template 只要 enable_thinking 没定义就走 thinking 分支，
# 于是 max_tokens=200 全烧在思考里、content 一直是空串，字幕表现为「没有译文」，
# 而且每句都要等满 timeout 才放弃（实测每行间隔 9~10 秒）。所以默认显式关掉。
# Qwen2.5 的模板不认这个变量，传了也无害。要跑思考模式加 --llm-think。
LLM_THINK = False


def _think_kw() -> dict:
    return {} if LLM_THINK else {"chat_template_kwargs": {"enable_thinking": False}}



def translate(base: str, model: str, text: str, target: str, timeout: float = 12.0,
              ctx: str = "", hint: str = "") -> str:
    """只输出译文，短句直译。失败返回空串，不阻塞字幕。

    ctx 是前一两句原文：段被切短后单句常常是半句，带上文能让术语、人称、语气对上，
    也是模型能纠正同音错字的唯一依据（单看半句「中方科意」无从判断是「刻意」）。
    hint 是重试时追加的强化指令，只在 lang_clean() 判脏之后用。
    """
    sys = _tr_system(target)
    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    if hint:
        user += "\n" + hint
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        "stream": False, "temperature": 0.1, "max_tokens": 200,
        **_think_kw(),
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers=_llm_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        out = (d["choices"][0]["message"]["content"] or "").strip()
        return out.strip('"“”「」 ')
    except Exception:  # noqa: BLE001
        return ""


def translate_stream(base: str, model: str, text: str, target: str,
                     timeout: float = 12.0, ctx: str = "",
                     on_delta=None, hint: str = "",
                     screen_hint: str = "") -> tuple[str, int, int]:
    """流式版翻译：边收边回调，返回 (全文, 首字ms, 总ms)。

    为什么要流式：vLLM 的 system prompt 被前缀缓存，**首字只要 ~96ms**，
    但整句要 300~1300ms（慢的是逐 token 生成，不是 prefill）。所以让译文
    逐字浮现，用户看到第一个字的时间提前 3~7 倍，而总耗时完全不变。

    中间态不做 strip 清理（会闪烁），只在结束时清一次。
    """
    sys = _tr_system(target)
    user = f"【上文】{ctx}\n【本句】{text}" if ctx else text
    # 屏幕提示只在本句有「同音异字候选」时才非空：平时一个 token 都不加
    if screen_hint:
        user += "\n" + screen_hint
    if hint:
        user += "\n" + hint
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        "stream": True, "temperature": 0.1, "max_tokens": 200,
        **_think_kw(),
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers=_llm_headers())
    parts: list[str] = []
    t0 = time.time()
    ttft_ms = 0
    t_hdr = -1
    dbg: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            t_hdr = int((time.time() - t0) * 1000)   # 连上并拿到响应头
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                if "timings" in ev and not dbg:
                    dbg = ev["timings"]
                ch = (ev.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content") or ""
                if not piece:
                    continue
                if not ttft_ms:
                    ttft_ms = int((time.time() - t0) * 1000)
                parts.append(piece)
                if on_delta is not None:
                    try:
                        on_delta("".join(parts))
                    except Exception:  # noqa: BLE001
                        pass          # 推送失败绝不能让翻译线程崩
    except Exception:  # noqa: BLE001
        # 已经收到的部分仍然返回：半句译文比空白有用
        pass
    total_ms = int((time.time() - t0) * 1000)
    # 最近一次的引擎侧 timings。离线工具（wav_seg_ab / ttft_prompt_ab）要拿
    # prompt_n / cache_n 做判据，又不想自己重写一份请求体，就从这里取。
    global LAST_TIMINGS
    LAST_TIMINGS = dbg or {}
    if os.environ.get("CAP_DEBUG"):
        # 分段定位：hdr=连上并拿到响应头，ttft=第一个字，total=整句
        print(f"[tr] hdr={t_hdr}ms ttft={ttft_ms}ms total={total_ms}ms "
              f"| srv prompt_n={dbg.get('prompt_n')} cache_n={dbg.get('cache_n')} "
              f"prompt_ms={dbg.get('prompt_ms')} ttft_ms={dbg.get('time_to_first_token_ms')}",
              flush=True)
    final = "".join(parts).strip().strip('"“”「」 ')
    # 检出源语言残留就重试一次（非流式，带强化指令）。仍脏则保留首版——
    # 半句混语言的译文也比空白强，而且重试不能无限循环拖住翻译线程。
    if final and not lang_clean(final, target):
        TR_STATS["retry"] += 1
        alt = translate(base, model, text, target, timeout=timeout, ctx=ctx,
                        hint=_RETRY_HINT.get(target, ""))
        if alt and lang_clean(alt, target):
            TR_STATS["fixed"] += 1
            total_ms = int((time.time() - t0) * 1000)
            if on_delta is not None:
                try:
                    on_delta(alt)          # 整行替换成干净版本，前端 applyUpd 本就整行替换
                except Exception:  # noqa: BLE001
                    pass
            return alt, ttft_ms, total_ms
    return final, ttft_ms, total_ms


# ------------------------------------------------------------------ 屏幕词表
# 屏幕上现成的字（幻灯片标题、下三分之一人名条、窗口标题）是纠正 ASR 专有名词
# 最便宜的来源——它不花一个 token，只要判断「ASR 是不是把屏幕上的词听成了同音异字」。
#
# 但**不能每句都塞一段屏幕文字**（2026-11-21 实测否定）：
#   - 代价：60 字 → prompt 多 72 token → 首字 490→740ms；120 字更贵
#   - 效果：6 句里 5 句被改写，其中 3 句变差（「三社」→「首次会谈」、凭空冒出
#     「三家公司」、重复「特朗普那边」），因为屏幕里既有我们自己的页面（自我循环，
#     把上一句译文当屏幕内容喂回去），又有与本句毫无关系的图文
# 所以这里只做「按需提示」：屏幕词与本句某段**读音高度相似但写法不同**时才给一句话，
# 每段最多一个词；没有命中就一个字都不加（零代价、零污染）。
_SCREEN_UI = ("会议字幕", "设备", "已识别", "延迟", "首字", "译完", "双语", "正在听",
              "停止", "清屏", "正在说", "运行中", "搜索", "创建", "Chrome",
              "youtube.com", "dB")

_SCREEN_RES = (
    re.compile(r"[0-9]{4,}"),                            # 年份/金额/期数
    re.compile(r"[ァ-ヴー]{4,10}"),                      # 片假名名词
    re.compile(r"[一-龥]{3,8}"),                         # 汉字串（人名、机构、术语）
)
# 为什么没有拉丁字母：OpenJTalk 会把 "CloudflareObservability" 逐字母读成 50 拍
# 「しーえるおー…」，永远撞不上日语读音，只会把词表挤满邮箱地址与网址
# （实测 40 条候选全是这类垃圾）。屏幕上的英文词若 ASR 也写对了，本就不需要提示；
# 若被听成片假名，靠读音也救不回来 —— 那种情况该走 ASR 热词，不是这条路。
_SCREEN_JUNK = ("@", "://", ".com", ".cn", ".jp", ".biz", ".net", "www.")
# 简体外挂字：聊天窗口/中文界面上的字混进日语会议的词表只会制造假命中。
# 出现两个以上才算中文行（日语里「的」偶尔出现，单个不能判）。
_SCREEN_CN = "这们为说时个还就没都很地对吗呢吧啊从给过应该"


def _hira(s: str) -> str:
    """片假名转平假名，只留能比读音的字符（标点、拉丁、数字都不参与比较）。"""
    out = []
    for ch in s or "":
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        elif 0x3041 <= o <= 0x3096 or ch == "ー":
            out.append(ch)
    return "".join(out)


def _lev(a: str, b: str) -> int:
    """编辑距离。只关心 ≤1，长度差超过 1 直接判 9（省掉整张表）。"""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class ScreenHints:
    """后台隔几秒看一眼屏幕，压成一张词表，按需给翻译递一句提示。

    为什么不自己开采集卡：`/dev/video0` 是独占的，voice-bench 常驻持有它。
    所以这里走 voice-bench 的 `/api/look`（它内部有 dHash 缓存，同一画面近乎免费），
    字幕服务不碰设备、也不引入 opencv/OCR 依赖。
    """

    def __init__(self, api: str = "http://127.0.0.1:8095/api/look",
                 every: float = 12.0, max_terms: int = 40,
                 timeout: float = 45.0, recent=None, reader=None):
        self.api = api
        self.every = max(4.0, every)
        self.max_terms = max(4, max_terms)
        self.timeout = timeout
        self.recent = recent or (lambda: [])
        self._reader = reader          # 测试注入：text -> [(surface, 平假名读音)]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.terms: list[tuple[str, str]] = []
        self.polls = 0
        self.hits = 0
        self.fails = 0
        self.lines_n = 0
        self.hash = ""
        self.at = ""
        self.last_ms = 0
        self.last_err = ""
        self.last_hint = ""

    # ------------------------------------------------------------ 读音
    def _morphemes(self, text: str) -> list[tuple[str, str]]:
        if self._reader is not None:
            return list(self._reader(text))
        try:
            import pyopenjtalk            # 语音助理的 TTS 已经带了，不额外装
        except Exception:                 # noqa: BLE001
            self.last_err = "没有 pyopenjtalk，屏幕提示不可用"
            return []
        try:
            return [(n.get("string", ""), _hira(n.get("read") or ""))
                    for n in pyopenjtalk.run_frontend(text)]
        except Exception as e:            # noqa: BLE001
            self.last_err = f"{type(e).__name__}: {e}"
            return []

    @staticmethod
    def _windows(morphs: list[tuple[str, str]],
                 max_join: int = 4, max_mora: int = 14):
        """把词素拼成 1~4 个词素的窗口，只留读音 4 拍以上的（太短必然撞车）。"""
        out = []
        for i in range(len(morphs)):
            surf, rd = "", ""
            for j in range(i, min(i + max_join, len(morphs))):
                surf += morphs[j][0]
                rd += morphs[j][1]
                if len(rd) > max_mora:
                    break
                if len(rd) >= 4:
                    out.append((surf, rd))
        return out

    # ------------------------------------------------------------ 取屏幕
    def _keep(self, lines: list[str]) -> list[str]:
        recent = [re.sub(r"\s+", "", x) for x in self.recent() if x]
        out = []
        for ln in lines:
            ln = ln.strip()
            if len(ln) < 3:
                continue
            if any(w in ln for w in _SCREEN_UI):
                continue
            if any(w in ln for w in _SCREEN_JUNK):   # 邮箱/网址整行都不值钱
                continue
            if sum(ln.count(c) for c in _SCREEN_CN) >= 2:   # 中文界面/聊天窗口
                continue
            # 与最近字幕有 8 字以上共同子串 → 认为是从我们自己页面抄来的，丢掉
            flat = re.sub(r"\s+", "", ln)
            junk = False
            for r in recent:
                for i in range(0, len(r) - 7):
                    if r[i:i + 8] in flat:
                        junk = True
                        break
                if junk:
                    break
            if not junk:
                out.append(ln)
        return out

    def set_text(self, text: str) -> None:
        """把一屏 OCR 文本压成词表（过滤 UI/自我循环 + 取读音 + 封顶条数）。"""
        lines = [x for x in (text or "").splitlines() if x.strip()]
        self.lines_n = len(lines)
        kept = self._keep(lines)
        seen: dict[str, str] = {}
        for ln in kept:
            for rx in _SCREEN_RES:
                for m in rx.finditer(ln):
                    w = m.group()
                    if w in seen:
                        continue
                    rd = "".join(r for _s, r in self._morphemes(w))
                    if not 4 <= len(rd) <= 14:   # 太短必撞车、太长读不准
                        continue
                    seen[w] = rd
        # 长读音优先：越长越具体，越不容易误命中
        terms = sorted(seen.items(), key=lambda kv: -len(kv[1]))[:self.max_terms]
        with self._lock:
            self.terms = terms

    def fetch(self) -> None:
        t0 = time.time()
        req = urllib.request.Request(
            self.api, data=b"{}", method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            info = json.loads(r.read().decode("utf-8", "replace") or "{}")
        self.polls += 1
        self.last_ms = int((time.time() - t0) * 1000)
        if not info.get("ok"):
            self.fails += 1
            self.last_err = str(info.get("message") or info.get("reason") or "看屏幕失败")[:120]
            return
        self.hash = (info.get("hash") or self.hash)[:16]
        self.at = time.strftime("%H:%M:%S")
        self.set_text(info.get("text") or "")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.fetch()
            except Exception as e:        # noqa: BLE001
                self.fails += 1
                self.last_err = f"{type(e).__name__}: {e}"[:120]
            self._stop.wait(self.every)

    def start(self) -> None:
        self._t = threading.Thread(target=self._loop, name="screen-hints", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------ 命中
    def suggest(self, text: str) -> str:
        """有命中就返回一句提示，否则空串（空串 = 请求体一个 token 都不加）。"""
        with self._lock:
            terms = list(self.terms)
        if not terms or len(text) < 4:
            return ""
        wins = self._windows(self._morphemes(text))
        if not wins:
            return ""
        for word, wrd in terms:
            if word in text:              # 已经写对了，不必提示
                continue
            for surf, rd in wins:
                if surf == word or abs(len(rd) - len(wrd)) > 1:
                    continue
                # 至少共用一个字：同音异字的真实形态是「畔蒜泰助 / 阪井泰介」（共用「泰」）、
                # 「小委員会 / 小委部」（共用「小委」）。一个字都不共用的纯读音巧合几乎
                # 都是假命中（尤其中文界面词被 OpenJTalk 硬读出日语读音时）。
                if not (set(surf) & set(word)):
                    continue
                if _lev(rd, wrd) <= 1:
                    self.hits += 1
                    self.last_hint = (f"【屏幕词】「{word}」——本句的「{surf}」"
                                      f"若指同一事物，请改用屏幕写法。")
                    return self.last_hint
        return ""

    def status(self) -> dict:
        with self._lock:
            n = len(self.terms)
            sample = [w for w, _r in self.terms[:6]]
        return {"on": True, "polls": self.polls, "hits": self.hits, "fails": self.fails,
                "terms": n, "sample": sample,
                "lines": self.lines_n, "hash": self.hash, "at": self.at,
                "last_ms": self.last_ms, "last_err": self.last_err or None,
                "last_hint": self.last_hint or None}


# ------------------------------------------------------------------ 引擎
class Captioner:
    """识别 + 双语字幕。

    在线路径是三段流水：`submit()` 入队 → asr 线程识别 → 立刻落一行只有原文的字幕
    → 翻译线程流式回填同一行的 `trans`（首字 ~96ms 就推上去），并登记补丁给前端。
    看字不用等翻译，更不用等译文整句生成完。
    离线路径 `handle()` 同步跑完，行为等价，供 wav 自测和单测用。
    """

    def __init__(self, asr: Asr, *, bilingual: bool = True,
                 llm_base: str = "http://127.0.0.1:8000/v1", llm_model: str = "qwen",
                 asr_workers: int = 1, tr_workers: int = 2, ctx_lines: int = 2,
                 stream_trans: bool = True, draft_on: bool = True,
                 draft_every: float = 0.8, screen=None):
        self.asr = asr
        self.bilingual = bilingual
        # 屏幕词表（可选）。命中才给翻译递一句提示，见 ScreenHints 的注释。
        self.screen = screen
        self.llm_base = llm_base
        self.llm_model = llm_model
        self.ctx_lines = ctx_lines
        self.stream_trans = stream_trans
        self.lines: collections.deque = collections.deque(maxlen=400)
        self.seq = itertools.count(1)
        self.segments = 0
        self.errors = 0
        self.dropped = 0
        self.started = time.time()
        self.asr_ms = 0
        self.trans_ms = 0
        self.ttft_ms = 0
        self.lag_ms = 0
        self.asr_workers = max(1, asr_workers)
        self.tr_workers = max(1, tr_workers)
        self._q_asr: queue.Queue = queue.Queue(maxsize=24)
        self._q_tr: queue.Queue = queue.Queue(maxsize=64)
        self._updates: collections.deque = collections.deque(maxlen=120)
        self._lock = threading.RLock()
        # 草稿线程和正式识别共用同一个 recognizer。sherpa-onnx 支持多线程各自
        # create_stream，但两路同时 decode 会抢 CPU 核；串行一下，代价只是草稿
        # 让路 200ms，换来正式链路的 lag 稳定。
        self._asr_lock = threading.Lock()
        self._started = False

        # 草稿字幕：说话过程中先出灰字，段结束后被正式行替换。
        # 单槽队列（只留最新），草稿识别慢于说话节奏时自然丢掉过期的，绝不排队堆积。
        self.draft_on = draft_on
        self.draft_every = draft_every
        self.draft_text = ""
        self.draft_epoch = 0
        self.draft_ms = 0
        self.drafts = 0
        self._q_draft: queue.Queue = queue.Queue(maxsize=1)

    @property
    def busy(self) -> bool:
        return not self._q_asr.empty() or not self._q_tr.empty()

    def start_workers(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        for i in range(self.asr_workers):
            threading.Thread(target=self._asr_loop, name=f"asr{i}", daemon=True).start()
        for i in range(self.tr_workers):
            threading.Thread(target=self._tr_loop, name=f"tr{i}", daemon=True).start()
        if self.draft_on:
            threading.Thread(target=self._draft_loop, name="draft", daemon=True).start()

    def submit(self, pcm: np.ndarray, dur: float, end_ts: float | None = None) -> None:
        """采集线程只负责把段丢进来，绝不在这里等识别。"""
        if not self._started:
            self.start_workers()
        # 这一段已定稿：草稿作废，正在跑的那次草稿识别结果回来也会被 epoch 挡掉
        with self._lock:
            self.draft_epoch += 1
            self.draft_text = ""
        item = (pcm, dur, end_ts if end_ts else time.time())
        try:
            self._q_asr.put_nowait(item)
        except queue.Full:
            # 宁可丢最老的一段，也不能让字幕越拖越远（会议里迟到 10 秒的字幕等于没有）
            self.dropped += 1
            try:
                self._q_asr.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q_asr.put_nowait(item)
            except queue.Full:
                pass

    def push_draft(self, pcm: np.ndarray, dur: float) -> None:
        """采集线程把「当前段到目前为止的音频」丢进来做草稿识别。

        单槽：草稿是给人看的临时状态，旧的一律没用，来了新的就顶掉旧的。
        绝不排队——排队会让草稿越拖越落后，最后显示的是几秒前的话。
        """
        if not (self.draft_on and self._started):
            return
        if self._q_draft.full():
            try:
                self._q_draft.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q_draft.put_nowait((pcm, dur, self.draft_epoch))
        except queue.Full:
            pass

    def _draft_loop(self) -> None:
        while True:
            pcm, _dur, ep = self._q_draft.get()
            # 有正式段在排队时草稿毫无意义（马上就会被真实结果替换），直接让路
            if not self._q_asr.empty():
                continue
            t0 = time.time()
            try:
                with self._asr_lock:
                    text, lang = self.recognize(pcm)
            except Exception:  # noqa: BLE001
                continue
            self.draft_ms = int((time.time() - t0) * 1000)
            self.drafts += 1
            with self._lock:
                # 识别期间这一段可能已经定稿了（submit 会 epoch+1），结果作废
                if ep != self.draft_epoch:
                    continue
                self.draft_text = text
            if text:
                print(f"  [草稿] {self.draft_ms}ms {lang} {text}", flush=True)

    def take_updates(self) -> list[dict]:
        """译文补丁：前端取走即清空。行对象里的 trans 也会回填，历史照样完整。"""
        out: list[dict] = []
        with self._lock:
            while self._updates:
                out.append(self._updates.popleft())
        return out

    def _push_update(self, seq: int, trans: str, final: bool,
                     trans_ms: int = 0, ttft_ms: int = 0) -> None:
        """登记一条译文补丁；同一个 seq 只保留最新一条。

        前端是 250ms 轮询，而流式每 ~80ms 就有一个增量。如果不合并，
        一次 poll 会拿到同一行的 3 条中间态，页面白重绘 2 次。
        """
        with self._lock:
            for u in self._updates:
                if u["seq"] == seq:
                    u["trans"] = trans
                    u["final"] = final
                    if final:
                        u["trans_ms"] = trans_ms
                        u["ttft_ms"] = ttft_ms
                    return
            self._updates.append({"seq": seq, "trans": trans, "final": final,
                                  "trans_ms": trans_ms, "ttft_ms": ttft_ms})

    def clear(self) -> None:
        with self._lock:
            self.lines.clear()
            self._updates.clear()
            self.draft_text = ""
            self.draft_epoch += 1

    def _ctx_before(self, seq: int) -> str:
        # ctx_lines=0 必须真的「不带上文」：[-0:] 等于 [0:] 会把全部历史都塞进去，
        # 曾经把 prompt 从 80 token 涨到 138，首字多花 300ms。
        if self.ctx_lines <= 0:
            return ""
        with self._lock:
            prev = [l["text"] for l in self.lines if l["seq"] < seq][-self.ctx_lines:]
        return " / ".join(prev)

    def recognize(self, pcm: np.ndarray) -> tuple[str, str]:
        """返回 (文本, 语种)。语种从 SenseVoice 的 <|ja|> 标签取，取不到就按字形猜。"""
        x = vp.highpass(pcm.astype(np.float64), 150.0, SR, 2)
        s = self.asr.rec.create_stream()
        s.accept_waveform(SR, (x / vp.FULL_SCALE).astype(np.float32))
        self.asr.rec.decode_stream(s)
        raw = s.result.text or ""
        m = LANG_TAG.search(raw)
        lang = m.group(1) if m else ("ja" if KANA.search(raw) else "zh" if CJK.search(raw) else "")
        text = clean_text(raw)
        if not SUBSTANCE.search(text):          # 只剩标点/空白 = 噪声段，不是一句话
            return "", lang
        return text, lang

    def _asr_loop(self) -> None:
        while True:
            pcm, dur, end_ts = self._q_asr.get()
            t0 = time.time()
            try:
                with self._asr_lock:
                    text, lang = self.recognize(pcm)
            except Exception:  # noqa: BLE001
                self.errors += 1
                continue
            self.asr_ms = int((time.time() - t0) * 1000)
            if not text:
                continue
            lag = int((time.time() - end_ts) * 1000)
            self.lag_ms = lag
            want = self.bilingual and lang in ("zh", "ja")
            line = {
                "seq": next(self.seq), "text": text, "lang": lang, "trans": "",
                "pending": want, "dur": round(dur, 2), "asr_ms": self.asr_ms,
                "lag_ms": lag, "at": time.strftime("%H:%M:%S"),
            }
            with self._lock:
                self.segments += 1
                self.lines.append(line)
            print(f"  [{line['at']}] #{line['seq']} lag={lag}ms asr={self.asr_ms}ms "
                  f"q={self._q_tr.qsize()} {line['lang']} {text}", flush=True)
            if want:
                try:
                    self._q_tr.put_nowait(line)
                except queue.Full:
                    line["pending"] = False

    def _tr_loop(self) -> None:
        while True:
            line = self._q_tr.get()
            target = "ja" if line["lang"] == "zh" else "zh"
            # 只清洗送进翻译的文本，line["text"]（页面显示）保持逐字
            ctx = strip_fillers(self._ctx_before(line["seq"]))
            src = strip_fillers(line["text"])
            seq = line["seq"]

            if not self.stream_trans:
                t0 = time.time()
                trans = translate(self.llm_base, self.llm_model, src, target, ctx=ctx)
                dt = int((time.time() - t0) * 1000)
                line["pending"] = False
                if not trans:
                    line["failed"] = True
                    continue
                line["trans"] = trans
                line["trans_ms"] = dt
                self.trans_ms = dt
                self._push_update(seq, trans, True, dt, 0)
                continue

            # 流式：首字 ~96ms 就推上去，整句 300~1300ms 才完
            state = {"last": 0.0, "last_len": 0}
            # 屏幕词表：只有「读音对得上、写法不同」才递一句提示；否则空串，零代价
            sh = self.screen.suggest(src) if self.screen is not None else ""
            if sh:
                line["screen_hint"] = sh

            def on_delta(txt: str, _seq=seq, _line=line, _st=state) -> None:
                now = time.time()
                # 节流到 ~80ms 或 +4 字；推送本身会按 seq 合并，多推不会造成重绘
                if now - _st["last"] < 0.08 and len(txt) - _st["last_len"] < 4:
                    return
                _st["last"] = now
                _st["last_len"] = len(txt)
                _line["trans"] = txt
                self._push_update(_seq, txt, False)

            trans, ttft, total = translate_stream(
                self.llm_base, self.llm_model, src, target, ctx=ctx,
                on_delta=on_delta, screen_hint=sh)
            line["pending"] = False
            line["ttft_ms"] = ttft
            if not trans:
                line["failed"] = True
                continue
            line["trans"] = trans
            line["trans_ms"] = total
            self.trans_ms = total
            self.ttft_ms = ttft
            self._push_update(seq, trans, True, total, ttft)

    def handle(self, pcm: np.ndarray, dur: float) -> dict | None:
        """同步跑完一段：离线 wav 自测用。在线路径走 submit()。"""
        t0 = time.time()
        try:
            text, lang = self.recognize(pcm)
        except Exception:  # noqa: BLE001
            self.errors += 1
            return None
        asr_s = time.time() - t0
        if not text:
            return None

        trans = ""
        if self.bilingual and lang in ("zh", "ja"):
            target = "ja" if lang == "zh" else "zh"
            t1 = time.time()
            trans = translate(self.llm_base, self.llm_model, strip_fillers(text),
                              target,
                              ctx=strip_fillers(self._ctx_before(10 ** 9)))
            self.trans_ms = int((time.time() - t1) * 1000)

        line = {
            "seq": next(self.seq), "text": text, "lang": lang, "trans": trans,
            "pending": False, "dur": round(dur, 2), "asr_ms": int(asr_s * 1000),
            "lag_ms": int(asr_s * 1000), "at": time.strftime("%H:%M:%S"),
        }
        with self._lock:
            self.segments += 1
            self.lines.append(line)
        return line

    def draft_state(self) -> dict:
        """草稿行当前该显示什么。空字符串=没有草稿，前端据此清掉灰色行。"""
        with self._lock:
            return {"text": self.draft_text, "epoch": self.draft_epoch,
                    "ms": self.draft_ms}

    def status(self) -> dict:
        return {"segments": self.segments, "errors": self.errors, "busy": self.busy,
                "bilingual": self.bilingual, "up_s": round(time.time() - self.started, 1),
                "asr_ms": self.asr_ms, "trans_ms": self.trans_ms,
                "ttft_ms": self.ttft_ms, "stream_trans": self.stream_trans,
                "lag_ms": self.lag_ms, "dropped": self.dropped,
                "draft_on": self.draft_on, "drafts": self.drafts,
                "draft_ms": self.draft_ms, "tr_prompt": TR_STYLE,
                "q_asr": self._q_asr.qsize(), "q_tr": self._q_tr.qsize(),
                "screen": self.screen.status() if self.screen is not None else None}


# ------------------------------------------------------------------ 页面
# 注意：必须是 raw 字符串。PAGE 里嵌着 JS，非 raw 时 Python 会把 JS 里的 '\n'
# 变成真换行，把单引号字符串截断 → 整段脚本语法错误 → 页面停在「连接中…」且按钮全失效。
PAGE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>会议字幕</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{--bg:#0b0d11;--panel:#151920;--line:#252b36;--fg:#e8eefc;--dim:#7f8a9e;
        --accent:#5b9cff;--ok:#41c48a;--warn:#e0a34a;--err:#e0645a}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font-family:"Noto Sans CJK JP","Noto Sans CJK SC",system-ui,sans-serif}
  header{position:sticky;top:0;z-index:9;background:rgba(11,13,17,.92);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--line);
    padding:14px 22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.5px}
  .meter{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--dim)}
  .bar{width:140px;height:6px;background:#1d222b;border-radius:99px;overflow:hidden}
  .bar i{display:block;height:100%;width:0;background:var(--ok);transition:width .12s}
  .pill{font-size:12px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);
    background:var(--panel);color:var(--dim)}
  .pill b{color:var(--fg)}
  main{padding:16px 22px 60px;max-width:1100px;margin:0 auto;
    overflow-anchor:none}   /* 自己管滚动锚点，别让浏览器的滚动锚定插手 */
  .line{padding:10px 0;border-bottom:1px dashed #1e242e}
  .line .t{font-size:13px;color:#5f6a7d;font-variant-numeric:tabular-nums}
  .line .o{font-size:26px;line-height:1.45;margin:2px 0 0}
  .line .tr{font-size:19px;line-height:1.4;color:#9fb4d6;margin:4px 0 0}
  .line .tr.pend{color:#4d5766;font-style:italic;font-size:16px}
  .line .tr .cur{color:var(--accent);animation:blink 1s steps(2) infinite}
  @keyframes blink{50%{opacity:0}}
  .ja .o{color:#dbe8ff} .zh .o{color:#e6ffe9}
  .empty{color:#4d5766;font-style:italic;padding:40px 0;text-align:center}
  .err{color:var(--err);font-size:13px}
  button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--line);
    background:var(--panel);color:var(--fg);padding:6px 14px;font-size:13px;transition:.15s}
  button:hover{border-color:#3a4356;background:#1c2029}
  #stop.run{border-color:#6b3438;background:#2a191b;color:#ffb3ad}
  #stop.idle{border-color:#2f6b52;background:#152a22;color:#a8e6c8}
  #mon.run{border-color:#2f6b52;background:#152a22;color:#a8e6c8}
  .spacer{flex:1}
  /* 草稿行：说话过程中先出灰字，段结束后被正式行替换。
     刻意压暗＋斜体，让人一眼看出「这句可能还会改」，不会误当成定稿去引用。 */
  .line.draft{opacity:.62}
  .line.draft .o{font-style:italic;color:#b9c2d4}
  .line.draft .t{color:#5f6a7d}
</style></head><body>
<header>
  <h1>会议字幕</h1>
  <span class="pill">设备 <b>__DEV__</b></span>
  <span class="pill">已识别 <b id="seg">0</b> 段</span>
  <span class="pill" id="lag">延迟 <b>—</b></span>
  <span class="pill" id="bi">双语</span>
  <span class="pill" id="scr" style="display:none">屏幕</span>
  <div class="meter"><div class="bar"><i id="lvl"></i></div><span id="db">-99 dB</span></div>
  <span class="pill" id="live">连接中…</span>
  <span class="err" id="err"></span>
  <span class="spacer"></span>
  <button id="mon">听 HDMI</button>
  <button id="stop" class="run">停止</button>
  <button id="clear">清屏</button>
</header>
<main id="main"><div class="empty">等待声音…</div></main>
<script>
const $ = s => document.querySelector(s);
let last = 0, lines = [], running = true, idx = new Map(), draft = '';

// ---------- 增量渲染 ----------
// 为什么不再用「拼 HTML 字符串 + innerHTML 全量替换」（2026-11-21 改）：
// 旧实现每收一次推送（流式时 ~80ms 一次）就把最后 30 行整个重建，再用像素
// `window.scrollY` 回填滚动位置。长句译文正在逐字变高时，重建后的总高度变了，
// 同一个 scrollY 指向的已经是别的内容 —— 用户正在读的那一行被推出视口，
// 表现就是「下一句一来，上一句整行没了」。句子越长、高度变化越大，所以只有长句明显。
// 改成增量后：每行一个常驻节点，只有变化的文本节点会被写，已有内容的几何位置不动。
// 排列方向（2026-11-21 下午改）：**最新的一条在最上面，老行往下推**，草稿行占最顶。
// 所以视线锚点从「底部」变成「顶部」：贴顶时新行插入即等于整屏下移；翻到下面看旧行时，
// 按锚点元素补偿 scrollY，避免新行把正在读的那段顶跑（和旧 bug 同一类问题，方向相反）。
const dom = new Map();      // seq -> {root, head, o, tr, key}
let draftEl = null;

function makeLine(l){
  const root = document.createElement('div');
  root.className = 'line ' + (l.lang || '');
  const head = document.createElement('div'); head.className = 't';
  const o = document.createElement('div'); o.className = 'o';
  const tr = document.createElement('div'); tr.className = 'tr';
  root.append(head, o, tr);
  return {root: root, head: head, o: o, tr: tr, key: {}};
}

function syncLine(l){
  let d = dom.get(l.seq);
  if(!d){
    d = makeLine(l);
    dom.set(l.seq, d);
    // 最新在最上：新行插到最前面（草稿行随后由 syncDraft 再提到它前面）
    const first = $('#main').firstElementChild;
    if(first) $('#main').insertBefore(d.root, first);
    else $('#main').appendChild(d.root);
  }
  const head = (l.at || '') + ' · ' + (l.dur != null ? l.dur : '') + 's' +
    (l.lag_ms != null ? ' · 延迟 ' + (l.lag_ms/1000).toFixed(1) + 's' : '') +
    (l.trans_ms ? ' · 译 ' + (l.trans_ms/1000).toFixed(1) + 's' : '');
  if(d.key.head !== head){ d.head.textContent = head; d.key.head = head; }

  const o = l.text || '';
  if(d.key.o !== o){ d.o.textContent = o; d.key.o = o; }

  const t = l.trans || '';
  const pend = !!l.pending;
  const k = t + (pend ? '\u0001' : '');
  // 只在内容真的变了才动 DOM：流式时这一行每 80ms 变一次，其余 29 行一个字都不碰
  if(d.key.tr !== k){
    d.tr.textContent = '';
    if(t){
      d.tr.className = 'tr';
      d.tr.appendChild(document.createTextNode(t));
      if(pend){
        const c = document.createElement('span');
        c.className = 'cur'; c.textContent = '▍';
        d.tr.appendChild(c);
      }
    } else if(pend){
      d.tr.className = 'tr pend';
      d.tr.textContent = '译文中…';
    } else {
      d.tr.className = 'tr';
    }
    d.key.tr = k;
  }
  const cls = 'line ' + (l.lang || '');
  if(d.root.className !== cls) d.root.className = cls;
}

function syncDraft(){
  if(!draftEl){
    draftEl = document.createElement('div');
    draftEl.className = 'line draft';
    const t = document.createElement('div'); t.className = 't'; t.textContent = '正在说…';
    const o = document.createElement('div'); o.className = 'o';
    draftEl.append(t, o);
    draftEl.txt = '';
  }
  const o = draftEl.lastElementChild;
  if(draft){
    if(!draftEl.isConnected){
      // 草稿是「最新的一条」，永远排在最上面
      const first = $('#main').firstElementChild;
      if(first) $('#main').insertBefore(draftEl, first);
      else $('#main').appendChild(draftEl);
    }
    if(draftEl.txt !== draft){
      o.textContent = draft;
      const c = document.createElement('span'); c.className = 'cur'; c.textContent = '▍';
      o.appendChild(c);
      draftEl.txt = draft;
    }
  } else if(draftEl.isConnected){
    draftEl.remove();
    draftEl.txt = '';
  }
}

// 期望顺序：草稿（最新）在最上，其余按 seq 从大到小往下排
function orderOk(m, visible){
  let prev = null;
  if(draftEl && draftEl.isConnected){
    if(m.firstElementChild !== draftEl) return false;
    prev = draftEl;
  }
  for(let i = visible.length - 1; i >= 0; i--){
    const r = dom.get(visible[i].seq).root;
    if(prev ? prev.nextElementSibling !== r : m.firstElementChild !== r) return false;
    prev = r;
  }
  return true;
}

function render(){
  const m = $('#main');
  if(!lines.length && !draft){
    if(!m.querySelector('.empty')){
      m.innerHTML = '<div class="empty">等待声音…</div>';
      dom.clear(); draftEl = null;
    }
    return;
  }
  const empty = m.querySelector('.empty');
  if(empty) empty.remove();

  // 视线锚点：贴顶时不动滚动，新行把内容往下推（这就是「字幕往下滚」）；
  // 往下翻看旧行时按锚点补偿位移，否则插入新行会把正在读的段落顶跑。
  const anchor = m.firstElementChild;
  const anchorTop = anchor ? anchor.getBoundingClientRect().top : 0;
  const atTop = window.scrollY <= 24;

  const visible = lines.slice(-30);
  const keep = new Set(visible.map(l => l.seq));
  for(const [seq, d] of dom){
    if(!keep.has(seq)){ d.root.remove(); dom.delete(seq); }   // 只回收窗口外的旧行（现在在底部）
  }
  for(const l of visible) syncLine(l);

  // 真的乱序时才重排。appendChild 对已有节点是「移动」，不重建、不丢内容
  if(!orderOk(m, visible)){
    if(draftEl && draftEl.isConnected) m.appendChild(draftEl);   // 先落到最前，后面依次接上
    for(let i = visible.length - 1; i >= 0; i--) m.appendChild(dom.get(visible[i].seq).root);
  }

  syncDraft();

  if(atTop){
    if(window.scrollY) window.scrollTo(0, 0);
  }else if(anchor && anchor.isConnected){
    // 内容被推下去多少像素（top 变大），就多滚多少，视线保持在原来那句上
    const delta = anchor.getBoundingClientRect().top - anchorTop;
    if(Math.abs(delta) > 0.5) window.scrollBy(0, delta);
  }
}
function pushLines(arr){
  if(!arr || !arr.length) return false;
  let added = false;
  for(const l of arr){
    if(idx.has(l.seq)) continue;      // SSE 重连会把 after 之前的行再发一遍，必须去重
    lines.push(l); idx.set(l.seq, l); added = true;
  }
  if(!added) return false;
  if(lines.length > 400){
    lines = lines.slice(-300);
    idx = new Map(lines.map(l => [l.seq, l]));
  }
  return true;
}
// 译文补丁：同一个 seq 把 trans 填进去，不新增行
function applyUpd(list){
  if(!list || !list.length) return false;
  let hit = false;
  for(const u of list){
    const l = idx.get(u.seq);
    if(l){
      l.trans = u.trans;
      // 流式中间态只更新文字；耗时和状态等 final 再定，否则数字一路往上跳
      if(u.final){
        l.trans_ms = u.trans_ms;
        if(u.ttft_ms) l.ttft_ms = u.ttft_ms;
        l.pending = false;
      }
      hit = true;
    }
  }
  return hit;
}

function paint(r){
  const b = $('#stop');
  if(b){ b.textContent = r ? '停止' : '开始'; b.className = r ? 'run' : 'idle'; }
  const lv = $('#live');
  if(lv) lv.textContent = r ? '运行中' : '已停止';
}

// 「听 HDMI」：把采集卡收到的声音回放到耳机，开会时可当监听口用
let monOn = null;
function paintMon(on, err){
  monOn = on;
  const b = $('#mon');
  if(!b) return;
  b.textContent = on ? '正在听 HDMI' : '听 HDMI';
  b.className = on ? 'run' : '';
  if(err) $('#err').textContent = err;
}
async function mon(){
  const b = $('#mon');
  if(b) b.disabled = true;
  try{
    const r = await fetch('/api/monitor', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({on: !monOn})});
    const j = await r.json();
    paintMon(j.monitor.on, j.monitor.error || '');
  }catch(e){ $('#err').textContent = '回放切换失败：' + e; }
  finally{ if(b) b.disabled = false; }
}
$('#mon').onclick = () => mon();

async function ctl(action){
  try{
    const r = await fetch('/api/control', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({action})});
    const j = await r.json();
    if(action === 'clear'){
      lines = []; idx = new Map(); last = 0; draft = '';
      dom.clear(); draftEl = null;
      $('#main').innerHTML = '<div class="empty">等待声音…</div>';
      return;
    }
    running = !!j.running;
    paint(running);
    $('#err').textContent = (!j.ok && j.error) ? j.error : '';
  }catch(e){ $('#err').textContent = '控制失败：' + e; }
}

// 一份 payload 的处理逻辑，SSE 和轮询共用
function applyPayload(j){
  const st = j.status || {};
  if(st.running !== running){ running = !!st.running; paint(running); }
  if(st.monitor && st.monitor.on !== monOn){ paintMon(st.monitor.on); }
  const got = pushLines(j.lines);
  if(got) last = Math.max(last, j.lines[j.lines.length-1].seq);
  const patched = applyUpd(j.upd);
  // 草稿行：服务端在 submit 时清空，正式行随即出现，所以这里直接跟服务端走
  const dt = (j.draft && j.draft.text) || '';
  const draftHit = dt !== draft;
  if(draftHit) draft = dt;
  $('#seg').textContent = st.segments ?? 0;
  const lg = $('#lag');
  if(lg) lg.innerHTML = '延迟 <b>' + (st.lag_ms ? (st.lag_ms/1000).toFixed(1) + 's' : '—') + '</b>'
                        + ' · 首字 <b>' + (st.ttft_ms ? st.ttft_ms + 'ms' : '—') + '</b>'
                        + ' · 译完 <b>' + (st.trans_ms ? (st.trans_ms/1000).toFixed(1) + 's' : '—') + '</b>';
  const db = running ? (j.level_db ?? -99) : -99;
  $('#db').textContent = running ? db.toFixed(0) + ' dB' : '—';
  const thr = j.thr_db ?? -50;
  const pct = Math.max(0, Math.min(100, (db + 70) / 60 * 100));
  const bar = $('#lvl'); bar.style.width = pct + '%';
  bar.style.background = db > thr ? '#41c48a' : '#3a4356';
  if(st.bilingual === false) $('#bi').textContent = '仅原文';
  // 屏幕词表：看不到就等于没生效，所以把「几词 / 命中几次」直接摆在标题栏
  const sc = $('#scr');
  if(sc){
    const s = st.screen;
    if(!s){ sc.style.display = 'none'; }
    else{
      sc.style.display = '';
      sc.innerHTML = '屏幕 <b>' + (s.terms || 0) + '</b> 词 · 命中 <b>' + (s.hits || 0) + '</b>';
      // PAGE 是 raw 字符串，这里的 \n 会原样进 JS（改回非 raw 就会把脚本打断）
      sc.title = (s.sample && s.sample.length ? '词表：' + s.sample.join('、') + '\n' : '')
               + (s.last_hint ? '最近命中：' + s.last_hint + '\n' : '')
               + '看屏幕一次 ' + (s.last_ms || 0) + 'ms'
               + (s.last_err ? '\n错误：' + s.last_err : '');
    }
  }
  $('#err').textContent = j.error || '';
  if(got || patched || draftHit) render();
}

// SSE 正常时由服务端推，轮询让位；SSE 一旦出错就关掉自己，回落到轮询
let es = null;
function startSSE(){
  if(!window.EventSource) return false;
  try{ es = new EventSource('/api/events?after=' + last); }catch(e){ es = null; return false; }
  es.onmessage = ev => { try{ applyPayload(JSON.parse(ev.data)); }catch(e){} };
  es.onerror = () => { if(es){ es.close(); es = null; } setTimeout(tick, 300); };
  return true;
}

async function tick(){
  if(es) return;                       // SSE 在跑，不再轮询
  try{
    const r = await fetch('/api/poll?after=' + last, {cache:'no-store'});
    applyPayload(await r.json());
  }catch(e){ $('#live').textContent = '断开'; }
  setTimeout(tick, es ? 1000 : 250);
}

$('#stop').onclick = () => ctl(running ? 'stop' : 'start');
$('#clear').onclick = () => ctl('clear');
paint(true);
if(!startSSE()) tick();     // 有 EventSource 就走推送，否则轮询兜底
</script></body></html>"""


# ------------------------------------------------------------------ HTTP
def build_app(cap: Captioner, pipe: Pipeline, device: str, state: dict,
              monitor: MonitorPipe | None = None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="meeting captions")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return PAGE.replace("__DEV__", device)

    @app.get("/api/events")
    async def events(request: Request, after: int = Query(0)):
        """SSE 推送。译文每 ~80ms 有一个增量，轮询要等最多 250ms 才看到。

        服务端内部仍是 50ms 检查一次（cap 是同步对象），但**客户端不再有往返**，
        中间态一到就画出来。前端断了会自动回落到 /api/poll。

        这里不用 cap.take_updates()：那是「取走即清空」，两个页面同时开会
        互相抢补丁，第二个永远停在「译文中…」。改成每个连接自己比对译文快照，
        多端各自独立。
        """
        import asyncio

        async def gen():
            last = after
            seen: dict[int, str] = {}
            last_status = 0.0
            last_draft: tuple[str, int] = ("", -1)
            while True:
                if await request.is_disconnected():
                    break
                lines = [dict(l) for l in cap.lines if l["seq"] > last]
                if lines:
                    last = lines[-1]["seq"]
                    for l in lines:
                        seen[l["seq"]] = l.get("trans") or ""

                upd = []
                for l in cap.lines:
                    cur = l.get("trans") or ""
                    if cur and seen.get(l["seq"]) != cur:
                        seen[l["seq"]] = cur
                        upd.append({"seq": l["seq"], "trans": cur,
                                    "final": not l.get("pending"),
                                    "trans_ms": l.get("trans_ms") or 0,
                                    "ttft_ms": l.get("ttft_ms") or 0})

                # 草稿行：说话过程中每 ~0.8s 更新一次，段结束被正式行替换。
                # 必须和译文补丁分开推，它不是某一行，是「列表末尾正在长的那一行」。
                d = cap.draft_state()
                now = time.time()
                tick_status = (now - last_status) >= 1.0
                draft_changed = (d["text"], d["epoch"]) != last_draft
                if draft_changed:
                    last_draft = (d["text"], d["epoch"])

                if lines or upd or draft_changed or tick_status:
                    if tick_status:
                        last_status = now
                    payload = {
                        "lines": lines,
                        "upd": upd,
                        "draft": d,
                        "status": {**cap.status(), "running": pipe.running,
                                   "monitor": monitor.status() if monitor else None},
                        "level_db": round(state.get("level_db", -99.0), 1),
                        "thr_db": round(state.get("thr_db", -50.0), 1),
                        "error": state.get("error", ""),
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/poll")
    async def poll(after: int = Query(0)):
        # 复制一份：翻译线程会就地回填 line["trans"]，别让序列化读到半截状态
        lines = [dict(l) for l in cap.lines if l["seq"] > after]
        return JSONResponse({
            "lines": lines,
            "upd": cap.take_updates(),
            "draft": cap.draft_state(),
            "status": {**cap.status(), "running": pipe.running,
                       "monitor": monitor.status() if monitor else None},
            "level_db": round(state.get("level_db", -99.0), 1),
            "thr_db": round(state.get("thr_db", -50.0), 1),
            "error": state.get("error", ""),
        })

    @app.get("/api/status")
    async def status():
        return {**cap.status(), "running": pipe.running,
                "level_db": state.get("level_db"), "thr_db": state.get("thr_db"),
                "device": device,
                # 断句参数原样回显：分段不对劲时第一件事是确认线上真的用了这组值，
                # 而不是对着默认值猜（unit 里显式写死的参数会盖掉默认）
                "seg": {"max_s": pipe.args.max_s, "hold": pipe.args.hold,
                        "hold_short": pipe.args.hold_short,
                        "hold_long": pipe.args.hold_long,
                        "mid": pipe.args.seg_mid, "long": pipe.args.seg_long},
                "tr": dict(TR_STATS),          # retry/fixed：脏译文检出与救回次数
                "monitor": monitor.status() if monitor else None}

    @app.post("/api/monitor")
    async def monitor_ctl(request: Request):
        """把 HDMI 声音回放到耳机：{on: true|false}。"""
        import asyncio

        if monitor is None:
            raise HTTPException(503, "回放未启用（启动时没建 MonitorPipe）")
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        want = bool(payload.get("on", not monitor.running))
        loop = asyncio.get_running_loop()
        if want:
            ok = await loop.run_in_executor(None, monitor.start)
            # PipeWire 里启动瞬间还在协商，稍等一下再读状态，否则第一次查总是 false
            if ok:
                await asyncio.sleep(0.35)
        else:
            await loop.run_in_executor(None, monitor.stop)
        return {"ok": True, "monitor": monitor.status()}

    @app.post("/api/control")
    async def control(request: Request):
        """stop / start / clear。stop 会释放声卡，start 重新估本底。"""
        import asyncio

        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        action = str(payload.get("action") or "")
        loop = asyncio.get_running_loop()
        if action == "stop":
            # stop 里有 wait(timeout=2)，start 要等 arecord 站住，都不能卡事件循环
            await loop.run_in_executor(None, pipe.stop)
            return {"ok": True, "running": pipe.running}
        if action == "start":
            ok = await loop.run_in_executor(None, pipe.start)
            return {"ok": ok, "running": pipe.running,
                    "error": state.get("error", "") if not ok else ""}
        if action == "clear":
            cap.clear()
            return {"ok": True, "running": pipe.running}
        raise HTTPException(400, f"未知动作：{action}")

    return app


# ------------------------------------------------------------------ 管线
class Pipeline:
    """采集管线的启停。

    停止必须真正释放声卡：只停线程不 terminate arecord，下次 start 会「设备或资源忙」。
    每次 start 重建 Segmenter —— 本底电平要按当时的房间重新估，不能沿用上一场会议的。
    """

    def __init__(self, cap: Captioner, device: str, args, state: dict):
        self.cap = cap
        self.device = device
        self.args = args
        self.state = state
        self.running = False
        self._src: Capture | None = None
        self._seg: Segmenter | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self._seg = Segmenter(margin_db=self.args.margin, hold_s=self.args.hold,
                                  max_s=self.args.max_s,
                                  min_s=getattr(self.args, "min_s", 0.25),
                                  soft_s=getattr(self.args, "soft_cut", 0.7),
                                  **seg_tuning(self.args))
            # USB 声卡被 terminate 后要过一小会儿才真的从 ALSA 释放，
            # 立刻重开必然 "device busy"，所以重试几次而不是一次判死。
            last_err = "打开声卡失败：未知原因"
            for attempt in range(4):
                src = Capture(self.device, SR, backend=getattr(self.args, "backend", "pw"),
                              target=getattr(self.args, "pw_target", None))
                try:
                    src.start()
                except Exception as e:  # noqa: BLE001
                    last_err = f"打开声卡失败：{e}"
                    time.sleep(0.35)
                    continue
                time.sleep(0.45)                   # arecord 是异步起进程，等它站住
                if src.alive():
                    self._src = src
                    self.state["error"] = ""
                    self.running = True
                    self._thread = threading.Thread(target=self._loop, name="capture-loop",
                                                    daemon=True)
                    self._thread.start()
                    print(f"[采集] 开始 {src.target or self.device}  backend={self.args.backend}",
                          flush=True)
                    return True
                src.stop()
                last_err = f"打开声卡失败：{src.last_error or '设备仍被占用'}"
                time.sleep(0.35 + 0.25 * attempt)
            self.state["error"] = last_err + "（已重试，稍等一下再点开始）"
            return False

    def stop(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.running = False
            if self._src:
                self._src.stop()
            self.state["level_db"] = -99.0
            self.state["thr_db"] = -50.0
            print("[采集] 停止", flush=True)
            return True

    def _loop(self) -> None:
        seg, src = self._seg, self._src
        assert seg and src
        last_draft = 0.0
        while self.running:
            try:
                frame = src.frames.get(timeout=0.5)
            except queue.Empty:
                if src.errors:
                    self.state["error"] = "采集中断，检查设备是否被占用"
                continue
            db = vp.db_of(frame.astype(np.float64))
            self.state["level_db"] = seg.level_db
            self.state["thr_db"] = seg.threshold()
            pcm = seg.push(frame, db)
            if pcm is not None and pcm.size:
                # 只入队，不等识别：识别和翻译各有自己的线程，采集循环永不被阻塞
                self.cap.submit(pcm, pcm.size / SR, end_ts=time.time())
                last_draft = 0.0                 # 新段开始，草稿节流重新计时
                continue
            # 段还没说完：隔 draft_every 秒把累积音频送去识别一次，屏幕上先出灰字。
            # 不加这段，离线 ASR 必须等整段（最长 6 秒）说完才有第一行，观感就是「不流式」。
            if self.cap.draft_on and pcm is None:
                now = time.time()
                if now - last_draft >= self.cap.draft_every:
                    d = seg.draft()
                    if d is not None and d[1] >= 0.8:      # 太短的草稿没信息量
                        last_draft = now
                        self.cap.push_draft(*d)
        src.stop()


def run_wav(cap: Captioner, path: str, args) -> None:
    audio, sr = vp.read_wav(path)
    if sr != SR:                                   # 线性重采样，仅供离线自测
        n = int(round(audio.size * SR / sr))
        audio = np.interp(np.linspace(0, audio.size - 1, n),
                          np.arange(audio.size), audio)
    print(f"[文件] {path}  {audio.size/SR:.2f}s  {sr}Hz → {SR}Hz", flush=True)
    seg = Segmenter(margin_db=args.margin, hold_s=args.hold, max_s=args.max_s,
                    min_s=getattr(args, "min_s", 0.25),
                    soft_s=getattr(args, "soft_cut", 0.7),
                    **seg_tuning(args))
    # 末尾补 1 秒静音，否则最后一句还在缓冲里不会被 flush
    audio = np.concatenate([audio, np.zeros(SR, dtype=audio.dtype)])
    for i in range(0, audio.size - FRAME_N, FRAME_N):
        frame = audio[i:i + FRAME_N]
        db = vp.db_of(frame.astype(np.float64))
        pcm = seg.push(frame, db)
        if pcm is not None and pcm.size:
            line = cap.handle(pcm, pcm.size / SR)
            if line:
                print(f"  [{line['at']}] {line['lang']} {line['text']}"
                      + (f"  → {line['trans']}" if line['trans'] else ""), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="会议字幕（采集卡 → SenseVoice → 中日双语）")
    p.add_argument("--device", default="plughw:CARD=MS2109,DEV=0")
    p.add_argument("--backend", default="pw", choices=["pw", "alsa"],
                   help="pw=PipeWire（不独占，推荐）；alsa=arecord 直连")
    p.add_argument("--pw-target", help="PipeWire 源节点名，默认自动找 MACROSILICON")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--wav", help="不采集，直接跑一个 wav 文件然后退出")
    p.add_argument("--asr-dir", default=str(paths.SENSEVOICE_DIR))
    p.add_argument("--lang", default="auto", help="SenseVoice 语种：auto/zh/ja/en")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--margin", type=float, default=8.0, help="高于本底多少 dB 算说话")
    p.add_argument("--hold", type=float, default=0.45,
                   help="基准静音断句门限（段长介于 --seg-mid 与 --seg-long 之间时用）。"
                        "开会要实时就压小：0.4 秒够分开句子，再小会把词切成两段")
    p.add_argument("--max-s", type=float, default=6.0,
                   help="一段最长多少秒，兜底硬切。4.5 秒仍有 26%% 的段被切断、把"
                        "「大会」切成「大」+「会议」；6.0 秒能把半句并成完整句。"
                        "别超 7 秒：实测 7 秒时同一段音频会出幻觉（「等一长会议」→「德拉兰会议」）")
    p.add_argument("--min-s", type=float, default=0.25,
                   help="短于这个长度的段直接丢（咔哒声、咳嗽）")
    p.add_argument("--soft-cut", type=float, default=0.7,
                   help="硬切前在尾巴里回退找能量低谷的搜索窗（秒）。"
                        "切在音节中间会出半截词，退几百毫秒就能避开")
    p.add_argument("--hold-long", type=float, default=0.90,
                   help="段还短（有声 < --seg-mid）时要等多久静音才断句。"
                        "放大它，短句会跟下一句并成一段，不再被词间小停顿切碎")
    p.add_argument("--hold-short", type=float, default=0.30,
                   help="段已长（有声 ≥ --seg-long）时的断句门限。"
                        "压低它，长句一出现停顿就出字幕，延迟不再往上堆")
    p.add_argument("--seg-mid", type=float, default=1.20,
                   help="多久算「段还短」（秒）。低于它用 --hold-long")
    p.add_argument("--seg-long", type=float, default=2.40,
                   help="多久算「段已长」（秒）。高于它用 --hold-short")
    p.add_argument("--asr-workers", type=int, default=1,
                   help="识别线程数。ASR 只占 0.3 秒/段，1 个够；不够会体现在 q_asr")
    p.add_argument("--tr-workers", type=int, default=2,
                   help="翻译线程数。vLLM 会自己 batch，2 路能把译文追上原文")
    p.add_argument("--ctx-lines", type=int, default=2,
                   help="翻译时带上前几句原文做上文，短句译文才连贯")
    p.add_argument("--no-stream-trans", action="store_true",
                   help="译文退化成整句出（不流式）。流式首字约 96ms，"
                        "整句出要等 300~1300ms；只在排查问题时才关")
    p.add_argument("--llm-think", action="store_true",
                   help="打开模型的思考模式（Qwen3.5 系）。默认关：思考会把 "
                        "max_tokens 吃光导致译文空白，且每句多几百毫秒")
    p.add_argument("--no-bilingual", action="store_true", help="只出原文，不翻译")
    p.add_argument("--no-draft", action="store_true",
                   help="关掉草稿字幕。草稿是说话过程中先出灰字、段结束再定稿，"
                        "用来填掉「等一整段说完」那几秒空白；关掉就退回整段出字")
    p.add_argument("--draft-every", type=float, default=0.8,
                   help="草稿刷新间隔（秒）。压太小会白跑识别抢 CPU，"
                        "放大到 1.5 以上屏幕又开始一顿一顿")
    p.add_argument("--llm-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--llm-model", default="qwen")
    p.add_argument("--llm-key", default="",
                   help="LLM 的 Bearer token。Magnitude 的 /inference/v1 必填，vLLM 留空")
    p.add_argument("--tr-prompt", choices=["full", "short"], default="full",
                   help="翻译 system prompt 档位。full 带反例（7B 时代调出来的，约 250 token），"
                        "short 是 35B 短版（约 70 token）。短版省的是每句的 prefill，"
                        "但丢了示例，质量要另测（tools/tr_quality.py）")
    p.add_argument("--screen-hints", action="store_true",
                   help="用屏幕上的专有名词纠正 ASR 同音错字。只有「读音相似、写法不同」"
                        "才给翻译递一句提示，没有命中就不加任何 token（默认关）")
    p.add_argument("--screen-api", default="http://127.0.0.1:8095/api/look",
                   help="向谁要屏幕 OCR。采集卡是独占的，字幕服务不自己开设备，"
                        "默认借 voice-bench 的 /api/look（它内部按画面指纹缓存）")
    p.add_argument("--screen-every", type=float, default=12.0,
                   help="隔几秒看一次屏幕。OCR 一次约 0.7~3.5 秒，别小于 4 秒")
    p.add_argument("--monitor", action="store_true",
                   help="启动时就把 HDMI 声音回放到耳机（也可在页面上开关）")
    p.add_argument("--monitor-sink", default="H340",
                   help="回放目标节点名关键字；空则用系统默认输出")
    p.add_argument("--monitor-rate", type=int, default=44100,
                   help="回放采样率。耳机节点是 44100，传同值可省一次重采样")
    p.add_argument("--monitor-latency", type=int, default=20,
                   help="回放缓冲毫秒。20ms 实测无欠载；开会听着别超过 60")
    args = p.parse_args(argv)

    global LLM_THINK, _LLM_KEY, TR_STYLE
    LLM_THINK = args.llm_think
    _LLM_KEY = args.llm_key
    TR_STYLE = args.tr_prompt

    print(f"[1/2] 加载 ASR … {args.lang}", flush=True)
    asr = Asr(Path(args.asr_dir), threads=args.threads, language=args.lang)
    screen = None
    if args.screen_hints:
        screen = ScreenHints(api=args.screen_api, every=args.screen_every)
    cap = Captioner(asr, bilingual=not args.no_bilingual,
                    llm_base=args.llm_base, llm_model=args.llm_model,
                    asr_workers=args.asr_workers, tr_workers=args.tr_workers,
                    ctx_lines=args.ctx_lines,
                    stream_trans=not args.no_stream_trans,
                    draft_on=not args.no_draft, draft_every=args.draft_every,
                    screen=screen)
    if screen is not None:
        # 自我循环防护：屏幕词表要能认出「这一行是我们自己的字幕」
        screen.recent = lambda: [x for l in list(cap.lines)[-30:]
                                 for x in (l.get("text", ""), l.get("trans", ""))]
        screen.start()
        print(f"[屏幕] 每 {screen.every:.0f}s 看一次 {screen.api}", flush=True)

    if args.wav:
        print("[2/2] 离线模式", flush=True)
        run_wav(cap, args.wav, args)
        return 0

    state: dict = {"level_db": -99.0, "thr_db": -50.0, "error": ""}
    label = args.device if args.backend == "alsa" else "USB3.0 Capture (PipeWire)"
    pipe = Pipeline(cap, args.device, args, state)
    monitor = MonitorPipe(src=resolve_pw_source(args.pw_target or "MACROSILICON"),
                          sink=resolve_pw_sink(args.monitor_sink),
                          rate=args.monitor_rate,
                          latency_ms=args.monitor_latency)
    app = build_app(cap, pipe, label, state, monitor=monitor)
    if not pipe.start():
        print(f"!! 启动采集失败：{state['error']}", flush=True)
    if args.monitor:
        print(f"[监听] 开机即开：{monitor.src} → {monitor.sink}", flush=True)
        monitor.start()

    print(f"[2/2] 打开 http://<jetson-ip>:{args.port}   双语={'关' if args.no_bilingual else '开'}"
          f"   译文={'流式（首字先出）' if not args.no_stream_trans else '整句'}"
          f"   草稿={'开 %.1fs' % args.draft_every if not args.no_draft else '关'}",
          flush=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
