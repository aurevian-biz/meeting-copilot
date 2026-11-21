#!/usr/bin/env python3
"""换模型后的冒烟：确认三件事——能出中文/日文、思考块没把 token 吃光、流式首字正常。

    ./tools/llm_switch.sh status
    /opt/meeting-copilot/data/asr-venv/bin/python tools/smoke_llm.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/v1"
CASES = [("ja", "中方刻意给韩国好处"), ("zh", "台湾問題を取引材料にするのか")]


def main() -> int:
    for target, text in CASES:
        body = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": f"把用户说的这句话翻译成{'日语' if target == 'ja' else '中文'}，只输出译文。"},
                {"role": "user", "content": text},
            ],
            "stream": True, "temperature": 0.1, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            BASE + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        ttft, parts, reasoning = -1, [], []
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    p = line[5:].strip()
                    if p == "[DONE]":
                        break
                    try:
                        d = json.loads(p)
                    except json.JSONDecodeError:
                        continue
                    delta = (d.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if delta.get("content"):
                        if ttft < 0:
                            ttft = int((time.time() - t0) * 1000)
                        parts.append(delta["content"])
        except Exception as e:  # noqa: BLE001
            print(f"[{target}] ERROR {e}")
            return 1
        out = "".join(parts).strip()
        total = int((time.time() - t0) * 1000)
        print(f"[{target}] {ttft}/{total}ms  out={out!r}")
        if "".join(reasoning):
            print(f"         注意：有 reasoning_content {len(''.join(reasoning))} 字"
                  f"（思考没关掉，max_tokens=200 时译文会被吃光）")
        if not out:
            print("         失败：译文为空")
            return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
