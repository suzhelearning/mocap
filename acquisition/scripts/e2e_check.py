#!/usr/bin/env python3
"""e2e_check.py — 端到端验证:真实流 → 按键录制 → HDF5 检查。

流程(通过 pty 模拟按键):r 开始录制 → 等 REC seconds → s 保存 → q 退出,
然后检查 captures/ 下生成的 HDF5 并跑 inspect。

前置:zenohd 已在 7447 监听;有真实动捕流(Motive/Windows publisher)
与 manus 流(Manus 手套发布)发布到 router(合成 demo 数据已移除)。

用法: pixi run python scripts/e2e_check.py [--seconds 5]
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--no-viz", action="store_true", default=True)
    args = ap.parse_args()

    master, slave = pty.openpty()
    env = {**os.environ, "PYTHONPATH": "src"}
    cmd = ["pixi", "run", "record", "--config", args.config]
    if args.no_viz:
        cmd.append("--no-viz")
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env,
        stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    os.close(slave)

    def send(ch: str) -> None:
        os.write(master, ch.encode())
        time.sleep(0.3)

    lines: list[str] = []
    _reader_done = threading.Event()

    def _reader() -> None:
        """后台读 stdout:状态栏用 \\r 不换行,readline 会阻塞,必须独立线程。"""
        for line in proc.stdout:
            lines.append(line.rstrip())
            print("  | " + line.rstrip(), flush=True)
        _reader_done.set()

    threading.Thread(target=_reader, daemon=True).start()

    def pump(seconds: float) -> None:
        time.sleep(seconds)

    try:
        print("[e2e] 等待流进入…")
        pump(3.0)
        print("[e2e] 按 r 开始录制")
        send("r")
        pump(args.seconds)
        print("[e2e] 按 s 保存")
        send("s")
        pump(1.5)
        print("[e2e] 按 q 退出")
        send("q")
        pump(3.0)
        proc.wait(timeout=10)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.kill()

    out = "\n".join(lines)
    if "已连接" not in out:
        print("[FAIL] record 未成功连接 router", file=sys.stderr)
        return 1

# 输出目录来自配置(output_dir/<日期>/*.h5),非写死 captures/
    try:
        from acquisition.config import load_config
        cfg_dir = load_config(ROOT / "config.yaml").output_dir
    except Exception:
        cfg_dir = ROOT / "captures"
    h5s = sorted(cfg_dir.rglob("*.h5"))
    if not h5s:
        print("[FAIL] 未生成 HDF5 文件", file=sys.stderr)
        return 1
    h5 = h5s[-1]
    print(f"[e2e] 生成: {h5}")

    # inspect 输出校验
    result = subprocess.run(
        ["pixi", "run", "inspect", "--", str(h5)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[FAIL] inspect 失败", file=sys.stderr)
        return 1
    if "最大误差" in result.stdout:
        m = re.search(r"最大误差: ([0-9.e+-]+) m", result.stdout)
        if m and float(m.group(1)) > 1e-3:
            print(f"[FAIL] 手腕拼接一致性误差过大: {m.group(1)}", file=sys.stderr)
            return 1
    if "Hz(实测)" not in result.stdout:
        print("[FAIL] 时间戳统计缺失", file=sys.stderr)
        return 1

    print("[e2e] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
