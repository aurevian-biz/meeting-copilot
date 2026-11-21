#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 meeting-copilot contributors

"""Confirm the production environment: cuda-python available + voice-bench still on the TensorRT backend."""
import json
import subprocess
import urllib.request

PY = "/opt/meeting-copilot/data/asr-venv/bin/python"


def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (p.stdout + p.stderr).strip()


print("== cuda-python ==")
print(run(f'{PY} -c "import cuda; from cuda.bindings import runtime as rt; '
          'e,p = rt.cudaMalloc(16); print(\'cudaMalloc ->\', e); rt.cudaFree(p); print(\'cudaFree ok\')"'))

print("\n== voice-bench 健康 ==")
try:
    with urllib.request.urlopen("http://127.0.0.1:8095/api/health", timeout=30) as r:
        h = json.loads(r.read().decode())
    v = h.get("vision") or {}
    print("trt        :", json.dumps(v.get("trt"), ensure_ascii=False))
    print("trt_note   :", v.get("trt_note"))
    print("w/threads  :", v.get("w"), "/", v.get("threads"))
    sp = h.get("speaker")
    print("speaker    :", sp if not isinstance(sp, dict) else sp.get("enabled"))
    llm = h.get("llm")
    print("llm        :", llm.get("model") if isinstance(llm, dict) else llm)
except Exception as e:  # noqa: BLE001
    print("health 取失败:", e)

print("\n== 服务状态 ==")
print(run("systemctl is-active voice-bench caption-bench memmy-memory"))
