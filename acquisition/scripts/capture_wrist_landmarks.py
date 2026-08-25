#!/usr/bin/env python3
"""从 Motive point_cloud 自动采集左右五指桌面地标并写入 YAML。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import numpy as np
import ruamel.yaml
import zenoh

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "config" / "wrist_landmarks.yaml"
# +Y → -Y 的物理顺序。配置文件内部按每只手 thumb→little 输出。
ORDER_Y_DESC = (
    ("left_little", "left", "little", 20),
    ("left_ring", "left", "ring", 15),
    ("left_middle", "left", "middle", 10),
    ("left_index", "left", "index", 5),
    ("left_thumb", "left", "thumb", 24),
    ("right_thumb", "right", "thumb", 24),
    ("right_index", "right", "index", 5),
    ("right_middle", "right", "middle", 10),
    ("right_ring", "right", "ring", 15),
    ("right_little", "right", "little", 20),
)
FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")


def build_session(endpoint: str) -> zenoh.Session:
    config = zenoh.Config.from_json5(json.dumps({
        "mode": "client",
        "connect": {"endpoints": [endpoint]},
    }))
    return zenoh.open(config)


def capture(
    endpoint: str,
    seconds: float,
    max_marker_z_m: float,
) -> dict[int, np.ndarray]:
    """返回 raw_id → (N,3) 桌面 point_cloud 样本。"""
    rows: dict[int, list[np.ndarray]] = defaultdict(list)
    lock = threading.Lock()

    def on_frame(sample: object) -> None:
        try:
            frame = json.loads(sample.payload.to_string())
        except Exception:
            return
        accepted: list[tuple[int, np.ndarray]] = []
        for marker in frame.get("markers", []):
            position = np.asarray(marker.get("position", []), dtype=float)
            raw_id = marker.get("raw_id")
            if (
                marker.get("id_kind") == "point_cloud"
                and raw_id is not None
                and position.shape == (3,)
                and np.isfinite(position).all()
                and float(position[2]) < max_marker_z_m
                and not bool(marker.get("occluded", False))
            ):
                accepted.append((int(raw_id), position))
        with lock:
            for raw_id, position in accepted:
                rows[raw_id].append(position)

    session = build_session(endpoint)
    subscriber = session.declare_subscriber("mocap/hands/frame", on_frame)
    try:
        time.sleep(seconds)
    finally:
        subscriber.undeclare()
        session.close()
    return {raw_id: np.asarray(values, dtype=float) for raw_id, values in rows.items()}


def summarize(
    samples: dict[int, np.ndarray],
    *,
    max_std_mm: float,
    contact_z_m: float,
) -> list[dict]:
    if len(samples) != 10:
        counts = {raw_id: len(values) for raw_id, values in samples.items()}
        raise ValueError(f"期望 10 个桌面 point_cloud,实际 {len(samples)} 个:{counts}")
    maximum = max(len(values) for values in samples.values())
    points = []
    for raw_id, values in samples.items():
        if len(values) < max(5, int(maximum * 0.8)):
            raise ValueError(f"marker {raw_id} 可见帧不足:{len(values)}/{maximum}")
        mean = values.mean(axis=0)
        std_mm = values.std(axis=0) * 1000.0
        if float(np.linalg.norm(std_mm)) > max_std_mm:
            raise ValueError(
                f"marker {raw_id} 抖动过大:{std_mm.tolist()}mm > {max_std_mm}mm"
            )
        points.append({
            "raw_id": raw_id,
            "mean": mean,
            "std_mm": std_mm,
            "samples": len(values),
            "xyz": np.array([mean[0], mean[1], contact_z_m], dtype=float),
        })
    points.sort(key=lambda item: float(item["mean"][1]), reverse=True)
    if not all(float(points[index]["mean"][1]) > 0 for index in range(5)):
        raise ValueError("+Y 侧不是恰好 5 个点")
    if not all(float(points[index]["mean"][1]) < 0 for index in range(5, 10)):
        raise ValueError("-Y 侧不是恰好 5 个点")
    for point, (full_name, side, finger, node) in zip(points, ORDER_Y_DESC):
        point.update({"full_name": full_name, "side": side, "name": finger, "node": node})
    return points


def build_yaml(points: list[dict], *, seconds: float, contact_z_m: float) -> dict:
    data: dict = {
        "version": 1,
        "coordinate_system": "x_forward_y_left_z_up",
        "unit": "meter",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "capture_seconds": float(seconds),
        "contact_z_m": float(contact_z_m),
        "hands": {"left": [], "right": []},
    }
    for side in ("left", "right"):
        by_name = {item["name"]: item for item in points if item["side"] == side}
        for name in FINGER_ORDER:
            item = by_name[name]
            data["hands"][side].append({
                "name": name,
                "node": int(item["node"]),
                "xyz": [round(float(value), 6) for value in item["xyz"]],
                "marker_raw_id": int(item["raw_id"]),
                "measured_center_xyz": [round(float(value), 6) for value in item["mean"]],
                "std_mm": [round(float(value), 3) for value in item["std_mm"]],
                "samples": int(item["samples"]),
            })
    return data


def atomic_write(path: Path, data: dict) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            yaml.dump(data, stream)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="采集十个桌面 marker 坐标并生成五指地标 YAML")
    parser.add_argument("--router", default="tcp/127.0.0.1:7447")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--max-marker-z", type=float, default=0.02,
                        help="只保留低于该世界 Z 的 point_cloud marker(默认 0.02m)")
    parser.add_argument("--contact-z", type=float, default=0.0,
                        help="fingertip 实际接触面的世界 Z(默认桌面 z=0)")
    parser.add_argument("--max-std-mm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.max_std_mm <= 0:
        parser.error("--seconds/--max-std-mm 必须为正数")

    print(f"[landmarks] 采集 {args.seconds:.1f}s: {args.router}")
    try:
        samples = capture(args.router, args.seconds, args.max_marker_z)
        points = summarize(samples, max_std_mm=args.max_std_mm, contact_z_m=args.contact_z)
    except (OSError, ValueError) as exc:
        print(f"[landmarks] 失败:{exc}")
        return 1

    print("[landmarks] +Y → -Y:")
    for item in points:
        xyz = item["xyz"]
        print(
            f"  {item['full_name']:>13s} raw_id={item['raw_id']} "
            f"xyz=[{xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f}] "
            f"std={np.linalg.norm(item['std_mm']):.3f}mm n={item['samples']}"
        )
    if args.dry_run:
        print("[landmarks] dry-run:未写文件")
        return 0
    data = build_yaml(points, seconds=args.seconds, contact_z_m=args.contact_z)
    backup = atomic_write(args.output, data)
    print(f"[landmarks] 已写入:{args.output}")
    if backup is not None:
        print(f"[landmarks] 旧配置备份:{backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
