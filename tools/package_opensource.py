#!/usr/bin/env python3
"""把工作区拷贝成一份可发布的仓库目录（脱敏 + 只挑该公开的文件）。

为什么要有这个脚本，而不是手工拷：
  工作区里混着 5 张调试截图、health.json 之类的运行时产物、__pycache__，以及
  写死了本机绝对路径 / 内网 IP / LLM 的 Bearer token 的 service 文件。手工拷一次
  就会漏一次；而且以后代码还会改，公开仓需要能一条命令重新生成。

用法：
    python tools/package_opensource.py --dest ~/meeting-copilot [--force]

只做两件事：**挑文件** + **替换掉不该出现的字符串**。文档（README/docs）由人写，
不在这里生成，脚本也不会覆盖已存在的 docs。
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent

# 只收这些：运行时真正需要的源码 / 部署 / 工具 / 测试
COPY = [
    "voice_server.py", "caption_server.py", "vision.py", "trt_rec.py",
    "tts_probe.py", "voice_probe.py",
    "test_barge.py", "test_caption_pipe.py", "test_tts_routing.py",
    "test_vision.py", "test_voice_probe.py", "test_wake.py",
]
COPY_DIRS = {"tools": ["*.py", "*.js", "*.sh"], "deploy": ["*"]}

# 有意排除
SKIP_NAMES = {
    "memmy_add_dep.py", "memmy_row_check.py", "memmy_search_check.py",  # 与另一套记忆库项目耦合
    "__pycache__", "health.json", "_page_check.js",
}

# 脱敏规则：顺序敏感，先长后短
SECRETS = [
    # LLM 的 Bearer token（会写进 service 的 --llm-key），一律清空
    (re.compile(r"mag-[A-Za-z0-9]{24,}"), "__LLM_KEY__"),
    # 本机绝对路径 → 标准安装根
    (re.compile(r"/media/[A-Za-z0-9_.-]+/llm/data/voice-app"), "/opt/meeting-copilot"),
    (re.compile(r"/media/[A-Za-z0-9_.-]+/llm/data"), "/opt/meeting-copilot/data"),
    (re.compile(r"/media/[A-Za-z0-9_.-]+/llm/memmy/data"), "/opt/meeting-copilot/data"),
    # 内网地址 → 本机
    (re.compile(r"192\.168\.\d{1,3}\.\d{1,3}"), "127.0.0.1"),
    # systemd 单元里的本机用户名 / uid
    (re.compile(r"^User=\w+$", re.M), "User=__USER__"),
    (re.compile(r"^Group=\w+$", re.M), "Group=__USER__"),
    (re.compile(r"/run/user/\d+"), "/run/user/__UID__"),
    # ssh 别名
    (re.compile(r"\bssh u\d+\b"), "ssh ${JETSON_HOST}"),
]

TEXT_EXT = {".py", ".js", ".sh", ".service", ".md", ".json", ".txt", ".yml"}


def sanitize(text: str) -> str:
    for pat, rep in SECRETS:
        text = pat.sub(rep, text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="目标已存在也继续（不删 docs/）")
    a = ap.parse_args(argv)

    dest: Path = a.dest.expanduser().resolve()
    if dest.exists() and not a.force:
        print(f"目标已存在：{dest}（加 --force 覆盖代码，docs 不动）")
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    files: list[Path] = [SRC / n for n in COPY]
    for d, patterns in COPY_DIRS.items():
        for p in patterns:
            files += sorted((SRC / d).glob(p))

    n = 0
    for f in files:
        if not f.is_file() or f.name in SKIP_NAMES:
            continue
        rel = f.relative_to(SRC)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in TEXT_EXT:
            txt = f.read_text(encoding="utf-8")
            new = sanitize(txt)
            if new != txt:
                print(f"  脱敏 {rel}")
            out.write_text(new, encoding="utf-8")
        else:
            shutil.copy2(f, out)
        n += 1
    print(f"拷贝 {n} 个文件 → {dest}")

    # 兜底自检：公开仓里不该再出现这些东西
    bad = []
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix in TEXT_EXT:
            t = f.read_text(encoding="utf-8", errors="ignore")
            for pat, what in [(r"mag-[A-Za-z0-9]{24,}", "LLM key"),
                              (r"/media/[A-Za-z0-9_.-]+/", "本机外置盘路径"),
                              (r"192\.168\.", "内网地址")]:
                if re.search(pat, t):
                    bad.append(f"{f.relative_to(dest)}: {what}")
    if bad:
        print("!! 仍有残留：")
        for b in bad:
            print("   ", b)
        return 2
    print("脱敏自检通过：无密钥 / 无本机路径 / 无内网 IP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
