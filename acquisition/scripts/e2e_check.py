#!/usr/bin/env python3
"""真实流 → 录制 → 新 HDF5 → 严格质量门的端到端检查。"""

from __future__ import annotations

import argparse
import os
import pty
import subprocess
import sys
import threading
import time
from pathlib import Path

from acquisition.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def _captures(root: Path) -> set[Path]:
    return {path.resolve() for path in root.rglob("*.h5")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--viz", action="store_true",
                        help="E2E 时也启动 Viser（默认纯采集）")
    parser.add_argument("--min-rate-ratio", type=float, default=0.7)
    parser.add_argument("--max-gap-ms", type=float, default=100.0)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    cfg = load_config(config_path)
    output_dir = cfg.output_dir
    before = _captures(output_dir) if output_dir.exists() else set()
    started_ns = time.time_ns()

    master, slave = pty.openpty()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, "-m", "acquisition.cli", "--config", str(config_path)]
    if not args.viz:
        cmd.append("--no-viz")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.close(slave)
    lines: list[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip()
            lines.append(text)
            print("  | " + text, flush=True)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    def send(key: str) -> None:
        if proc.poll() is not None:
            raise RuntimeError(f"record 提前退出，code={proc.returncode}")
        os.write(master, key.encode())
        time.sleep(0.4)

    try:
        print("[e2e] 等待真实流进入…")
        time.sleep(3.0)
        print("[e2e] 开始录制")
        send(cfg.keymap["start"])
        time.sleep(args.seconds)
        print("[e2e] 保存")
        send(cfg.keymap["save"])
        time.sleep(1.0)
        print("[e2e] 退出")
        send(cfg.keymap["quit"])
        proc.wait(timeout=10)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        reader_thread.join(timeout=2)

    output = "\n".join(lines)
    if proc.returncode != 0 or "[saved]" not in output:
        print("[FAIL] record 未完成保存状态转移", file=sys.stderr)
        return 1

    after = _captures(output_dir)
    new_files = sorted(
        path for path in after - before
        if path.stat().st_mtime_ns >= started_ns
    )
    if len(new_files) != 1:
        print(
            f"[FAIL] 本次应生成且只生成 1 个 HDF5，实际 {len(new_files)}:"
            f" {new_files}",
            file=sys.stderr,
        )
        return 1
    capture = new_files[0]
    print(f"[e2e] 本次生成: {capture}")

    inspect_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "inspect_hdf5.py"),
        "--strict",
        "--min-rate-ratio", str(args.min_rate_ratio),
        "--max-gap-ms", str(args.max_gap_ms),
        str(capture),
    ]
    result = subprocess.run(
        inspect_cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print("[FAIL] 新录制文件未通过严格质量门", file=sys.stderr)
        return 1

    print("[e2e] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
