#!/usr/bin/env python3
"""诊断 SGLang 的流式行为：非流式有没有内容、SSE 读完后连接会不会关。

字幕用的是 `for raw in r:` 迭代，如果服务端发完 [DONE] 不结束 HTTP body，
每个请求都会卡到 timeout，线上表现是「译文一直是空的」。这个脚本把时间点打出来。
"""
from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000/v1/chat/completions"


def body(stream: bool) -> bytes:
    return json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": "把这句话翻译成日语：中方刻意给韩国好处"}],
        "stream": stream, "temperature": 0.1, "max_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()


def nonstream() -> None:
    t0 = time.time()
    req = urllib.request.Request(BASE, data=body(False),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        print(f"[非流式] {int((time.time() - t0) * 1000)}ms  content={msg.get('content')!r}"
              f"  reasoning={msg.get('reasoning_content')!r}  finish={d['choices'][0].get('finish_reason')}")
    except Exception as e:  # noqa: BLE001
        print(f"[非流式] 失败 {type(e).__name__}: {e}")


def stream() -> None:
    t0 = time.time()
    req = urllib.request.Request(BASE, data=body(True),
                                 headers={"Content-Type": "application/json"})
    n, done_at, first = 0, None, None
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                n += 1
                line = raw.decode("utf-8", "replace").strip()
                if first is None:
                    first = int((time.time() - t0) * 1000)
                if line.startswith("data:") and line[5:].strip() == "[DONE]":
                    done_at = int((time.time() - t0) * 1000)
                if n <= 3 or line == "" or "[DONE]" in line:
                    print(f"  +{int((time.time() - t0) * 1000):5d}ms  {line[:110]}")
        print(f"[流式] 读完 {n} 行，首行 {first}ms，[DONE] @{done_at}ms，"
              f"EOF @{int((time.time() - t0) * 1000)}ms → 连接正常关闭")
    except Exception as e:  # noqa: BLE001
        print(f"[流式] 异常 {type(e).__name__}: {e}  已读 {n} 行，[DONE] @{done_at}ms")


if __name__ == "__main__":
    nonstream()
    stream()
