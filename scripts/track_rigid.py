#!/usr/bin/env python3
"""track_rigid.py — 按名字/ID 提取动捕刚体位姿：实时监控 + JSONL 记录 + 位移测量。

订阅 zenoh 动捕流（``mocap/hands/frame``，120Hz，坐标系
``motive_y_up_right_handed``、单位米），按名字（``mocap/rigid_body_names``）
或 ID 提取刚体（如天机右臂末端刚体 ``right_arm`` id=10），用于：

- 在线监控：实时打印位姿、帧率、跟踪误差与有效率；
- 记录：写入 JSONL（每行一帧，含时间戳）；
- 位移测量：对记录的 JSONL 报告指定窗口内的位移（各轴 mm 与范数 mm）。

位移范数在坐标系旋转下不变，因此“真机 50mm 验收”不需要先标定
Motive↔机器人系：命令位移 50mm ↔ 动捕实测位移 50mm 可直接对比。

用法：
    pixi run track-rigid -- --names right_arm                        # 监控
    pixi run track-rigid -- --names right_arm --output cap.jsonl     # 记录
    pixi run track-rigid measure cap.jsonl --start 0 --end 5         # 位移测量
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

FRAME_KEY = "mocap/hands/frame"
NAMES_KEY = "mocap/rigid_body_names"
DEFAULT_ENDPOINT = "tcp/127.0.0.1:7447"
_NAMES_WAIT_S = 12.0


# ---------------------------------------------------------------------------
# 纯函数（可独立单元测试，不依赖 zenoh）
# ---------------------------------------------------------------------------

def resolve_ids(
    names_payload: dict[str, Any] | None,
    requested_names: list[str],
    requested_ids: list[int],
) -> set[int]:
    """把请求的名字解析为刚体 ID；名字未找到时抛 ValueError。"""
    ids = set(requested_ids)
    if requested_names:
        name_map = {}
        if names_payload is not None:
            raw = names_payload.get("names", {})
            if isinstance(raw, dict):
                for key, value in raw.items():
                    try:
                        name_map[str(value)] = int(key)
                    except (TypeError, ValueError):
                        continue
        missing = [
            name for name in requested_names if name not in name_map
        ]
        if missing:
            raise ValueError(
                f"刚体名字未在动捕中注册：{missing}（当前 {sorted(name_map)}）"
            )
        ids.update(name_map[name] for name in requested_names)
    if not ids:
        raise ValueError("必须提供 --names 或 --ids")
    return ids


def select_rigids(
    frame: dict[str, Any], wanted_ids: set[int]
) -> list[dict[str, Any]]:
    """从一帧中提取目标刚体；字段缺失的刚体跳过。"""
    selected = []
    for body in frame.get("rigid_bodies", []):
        body_id = body.get("id")
        if body_id in wanted_ids:
            position = body.get("position")
            quaternion = body.get("quaternion_xyzw")
            if (
                isinstance(position, (list, tuple)) and len(position) == 3
                and isinstance(quaternion, (list, tuple))
                and len(quaternion) == 4
            ):
                selected.append(body)
    return selected


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON"
                ) from exc
            records.append(record)
    if not records:
        raise ValueError(f"{path} 没有任何记录")
    return records


def _position(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["position"], dtype=np.float64)


def measure_displacement(
    records: list[dict[str, Any]],
    *,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """报告 [start_s, end_s]（相对首条记录，秒）窗口内的位移。

    窗口内取第一个与最后一个有效帧计算位移；返回各轴位移（mm）、
    范数（mm）、窗口有效帧数与跟踪误差统计。
    """
    if end_s <= start_s:
        raise ValueError("end_s 必须大于 start_s")
    t0 = float(records[0]["t_ns"])
    window = [
        record for record in records
        if start_s <= (float(record["t_ns"]) - t0) / 1.0e9 <= end_s
    ]
    valid = [record for record in window if record.get("tracking_valid")]
    if len(valid) < 2:
        raise ValueError(
            f"窗口 [{start_s}, {end_s}]s 内有效帧不足 2（共 {len(window)} 帧）"
        )
    delta_m = _position(valid[-1]) - _position(valid[0])
    mean_errors = [
        float(record["mean_error"]) for record in valid
        if isinstance(record.get("mean_error"), (int, float))
    ]
    return {
        "start_s": start_s,
        "end_s": end_s,
        "window_frames": len(window),
        "valid_frames": len(valid),
        "delta_mm": [float(value) * 1000.0 for value in delta_m],
        "displacement_mm": float(np.linalg.norm(delta_m)) * 1000.0,
        "mean_error_mean_mm": (
            float(np.mean(mean_errors)) * 1000.0 if mean_errors else None
        ),
        "mean_error_max_mm": (
            float(np.max(mean_errors)) * 1000.0 if mean_errors else None
        ),
    }


# ---------------------------------------------------------------------------
# zenoh 记录/监控
# ---------------------------------------------------------------------------

class RigidTracker:
    def __init__(
        self,
        *,
        names: list[str],
        ids: list[int],
        connect_endpoint: str,
        output: Path | None,
        stats_interval_s: float,
        duration_s: float,
    ):
        self._names = names
        self._ids = list(ids)
        self._connect_endpoint = connect_endpoint
        self._output = output
        self._stats_interval_s = stats_interval_s
        self._duration_s = duration_s
        self._wanted_ids: set[int] | None = None
        self._names_payload: dict[str, Any] | None = None
        self._frame_count = 0
        self._received_count = 0
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._stream = None

    def _on_names(self, sample: object) -> None:
        try:
            self._names_payload = json.loads(sample.payload.to_string())
        except (ValueError, AttributeError):
            pass

    def _on_frame(self, sample: object) -> None:
        self._frame_count += 1
        try:
            frame = json.loads(sample.payload.to_string())
        except ValueError:
            return
        for body in select_rigids(frame, self._wanted_ids or set()):
            self._received_count += 1
            record = {
                "t_ns": time.monotonic_ns(),
                "frame_number": frame.get("frame_number"),
                "motive_timestamp": frame.get("motive_timestamp"),
                "id": body["id"],
                "position": body["position"],
                "quaternion_xyzw": body["quaternion_xyzw"],
                "mean_error": body.get("mean_error"),
                "tracking_valid": bool(body.get("tracking_valid")),
            }
            if self._stream is not None:
                self._stream.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                self._stream.flush()
            self._print_pose(record)

    @staticmethod
    def _print_pose(record: dict[str, Any]) -> None:
        position = record["position"]
        mean_error = record.get("mean_error")
        error_mm = (
            f"{float(mean_error) * 1000.0:.3f}mm"
            if isinstance(mean_error, (int, float)) else "n/a"
        )
        print(
            f"[rigid {record['id']}] "
            f"pos=({position[0]:+.4f},{position[1]:+.4f},"
            f"{position[2]:+.4f}) valid={record['tracking_valid']} "
            f"err={error_mm}",
            flush=True,
        )

    def _resolve_wanted_ids(self) -> set[int]:
        deadline = time.monotonic() + _NAMES_WAIT_S
        while time.monotonic() < deadline:
            try:
                return resolve_ids(
                    self._names_payload, self._names, self._ids
                )
            except ValueError:
                time.sleep(0.5)
        return resolve_ids(self._names_payload, self._names, self._ids)

    def _stats(self) -> None:
        elapsed = time.monotonic() - self._started
        if elapsed < self._stats_interval_s:
            return
        self._started = time.monotonic()
        rate = self._frame_count / elapsed if elapsed > 0 else 0.0
        print(
            f"[stats] 帧={self._frame_count} ({rate:.1f}Hz) "
            f"目标刚体样本={self._received_count} "
            f"names={json.dumps(self._names_payload, ensure_ascii=False)}",
            flush=True,
        )
        self._frame_count = 0
        self._received_count = 0

    def run(self) -> int:
        import zenoh

        from natnet_zenoh.zenoh_transport import build_client_config

        if self._output is not None:
            self._output.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._output.open("w", encoding="utf-8")

        config = build_client_config(connect_endpoint=self._connect_endpoint)
        session = zenoh.open(config)
        session.declare_subscriber(FRAME_KEY, self._on_frame)
        session.declare_subscriber(NAMES_KEY, self._on_names)
        try:
            self._wanted_ids = self._resolve_wanted_ids()
            print(
                f"跟踪刚体 id={sorted(self._wanted_ids)} "
                f"names={self._names or '（按 ID）'} "
                f"记录={'开' if self._stream else '关'}（Ctrl-C 停止）",
                flush=True,
            )
            while not self._stop.is_set():
                if (
                    self._duration_s > 0
                    and time.monotonic() - self._started
                    >= self._duration_s
                ):
                    break
                self._stats()
                time.sleep(0.05)
        finally:
            session.close()
            if self._stream is not None:
                self._stream.close()
        return 0


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive finite")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="订阅动捕流并按名字/ID 跟踪刚体（监控/记录/位移测量）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser(
        "record", help="实时监控并可选记录刚体位姿 JSONL"
    )
    record.add_argument("--names", default="", help="逗号分隔的刚体名字")
    record.add_argument("--ids", default="", help="逗号分隔的刚体 ID")
    record.add_argument("--connect-endpoint", default=DEFAULT_ENDPOINT)
    record.add_argument("--output", type=Path, help="JSONL 输出路径")
    record.add_argument("--stats-interval", type=_positive_float, default=2.0)
    record.add_argument("--duration", type=_positive_float, default=0.0,
                        help="自动停止秒数（0=直到 Ctrl-C）")
    record.set_defaults(handler=_run_record)

    measure = subparsers.add_parser(
        "measure", help="测量记录文件指定窗口内的位移"
    )
    measure.add_argument("track", type=Path, help="track_rigid record 的 JSONL")
    measure.add_argument("--start", type=float, required=True)
    measure.add_argument("--end", type=float, required=True)
    measure.set_defaults(handler=_run_measure)
    return parser


def _run_record(args) -> int:
    tracker = RigidTracker(
        names=[name for name in args.names.split(",") if name],
        ids=[int(value) for value in args.ids.split(",") if value.strip()],
        connect_endpoint=args.connect_endpoint,
        output=args.output,
        stats_interval_s=args.stats_interval,
        duration_s=args.duration,
    )
    try:
        return tracker.run()
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


def _run_measure(args) -> int:
    records = read_records(args.track)
    try:
        result = measure_displacement(
            records, start_s=args.start, end_s=args.end
        )
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
