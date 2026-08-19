#!/usr/bin/env python3
"""calibrate_tcp.py — Motive 动捕系 ↔ 机器人 TCP 系的固定变换标定（Umeyama）。

输入两段同步采集的位姿序列（各 K 个“保持位姿”窗口）：

- motive：``track_rigid record`` 的 JSONL（动捕实测 right_arm 末端位姿，
  Motive 系 y-up、米制）；
- robot：机器人侧同一末端位姿的 JSONL（同构：每行
  ``{"t_ns": ..., "position": [...], "quaternion_xyzw": [...]}``；
  sim 下取 ``/pico_body_sim/right_arm/solved_pose``（FK，right_chest 系），
  真机下应取反馈关节角的 FK —— 见 docs/mocap_real_acceptance.md）。

求解：两序列各按时间均匀切成 K 个窗口（用户需在每个位姿上保持
>= 一个窗口时长），逐窗口配对求平均位置，用 Umeyama（无缩放）
求 T = (R, t)：``p_robot = R @ p_motive + t``。输出旋转矩阵/平移/
四元数、位置残差（mm）与姿态残差（度）。

位移范数验收不需要本标定；标定用于方向核对与坐标换算。

用法：
    pixi run calibrate-tcp -- --motive right_arm_cap.jsonl \\
        --robot solved_right.jsonl --pair-count 4

依赖：numpy、scipy（无 zenoh 依赖，可离线运行）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.track_rigid import read_records


def rotation_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → 单位四元数 [x, y, z, w]（纯 numpy，避免 scipy 依赖）。"""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = rotation.trace()
    if trace > 0.0:
        root = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * root
        x = (rotation[2, 1] - rotation[1, 2]) / root
        y = (rotation[0, 2] - rotation[2, 0]) / root
        z = (rotation[1, 0] - rotation[0, 1]) / root
    else:
        axis = int(np.argmax(np.diag(rotation)))
        next_axis = (axis + 1) % 3
        next_next = (axis + 2) % 3
        root = np.sqrt(
            1.0 + rotation[axis, axis]
            - rotation[next_axis, next_axis]
            - rotation[next_next, next_next]
        ) * 2.0
        components = [0.0, 0.0, 0.0]
        components[axis] = 0.25 * root
        components[next_axis] = (
            rotation[axis, next_axis] + rotation[next_axis, axis]
        ) / root
        components[next_next] = (
            rotation[axis, next_next] + rotation[next_next, axis]
        ) / root
        w = (
            rotation[next_next, next_axis]
            - rotation[next_axis, next_next]
        ) / root
        x, y, z = components
    quaternion = np.array([x, y, z, w])
    return quaternion / np.linalg.norm(quaternion)


# ---------------------------------------------------------------------------
# 纯函数（可独立单元测试）
# ---------------------------------------------------------------------------

def window_centers(
    records: list[dict[str, Any]], pair_count: int
) -> np.ndarray:
    """把记录序列按时间均匀切成 pair_count 个窗口，返回各窗口平均位置。

    窗口内有效帧数不足 2 时抛 ValueError。
    """
    if pair_count < 3:
        raise ValueError("pair_count 必须 >= 3")
    t0 = float(records[0]["t_ns"])
    t_end = float(records[-1]["t_ns"])
    duration = t_end - t0
    if duration <= 0:
        raise ValueError("记录时间跨度必须为正")
    positions = np.asarray(
        [record["position"] for record in records], dtype=np.float64
    )
    times = np.asarray(
        [(float(record["t_ns"]) - t0) / duration for record in records]
    )
    centers = []
    for index in range(pair_count):
        start, end = index / pair_count, (index + 1) / pair_count
        mask = (times >= start) & (times <= end)
        if mask.sum() < 2:
            raise ValueError(
                f"窗口 {index + 1}/{pair_count} 有效帧不足 2"
            )
        centers.append(positions[mask].mean(axis=0))
    return np.asarray(centers, dtype=np.float64)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """无缩放 Umeyama：dst ≈ R @ src + t，返回 (R(3x3), t(3,))。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src/dst 须为同形的 (N,3) 位置数组")
    if src.shape[0] < 3:
        raise ValueError("至少需要 3 个非共线配对点")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.T @ src_centered
    u, _sigma, vt = np.linalg.svd(covariance)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    translation = dst_mean - rotation @ src_mean
    return rotation, translation


def calibration_residuals(
    src: np.ndarray, dst: np.ndarray, rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, float]:
    mapped = (rotation @ src.T).T + translation
    errors_m = np.linalg.norm(mapped - dst, axis=1)
    return {
        "position_error_mean_mm": float(errors_m.mean()) * 1000.0,
        "position_error_p95_mm": float(
            np.percentile(errors_m, 95)
        ) * 1000.0,
        "position_error_max_mm": float(errors_m.max()) * 1000.0,
    }


def calibration_result(
    motive_records: list[dict[str, Any]],
    robot_records: list[dict[str, Any]],
    pair_count: int,
) -> dict[str, Any]:
    motive_centers = window_centers(motive_records, pair_count)
    robot_centers = window_centers(robot_records, pair_count)
    rotation, translation = umeyama(motive_centers, robot_centers)
    residuals = calibration_residuals(
        motive_centers, robot_centers, rotation, translation
    )
    quaternion = rotation_to_quat_xyzw(rotation)
    return {
        "pair_count": pair_count,
        "rotation": rotation.tolist(),
        "rotation_quat_xyzw": [float(value) for value in quaternion],
        "translation_m": [float(value) for value in translation],
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Motive 动捕系 ↔ 机器人 TCP 系固定变换标定（Umeyama）"
    )
    parser.add_argument("--motive", required=True, type=Path,
                        help="track_rigid record 的 JSONL（动捕实测末端位姿）")
    parser.add_argument("--robot", required=True, type=Path,
                        help="机器人侧末端位姿 JSONL（同构格式）")
    parser.add_argument("--pair-count", type=_positive_int, default=4,
                        help="保持位姿窗口数（默认 4，需 >= 3）")
    args = parser.parse_args(argv)

    try:
        motive = read_records(args.motive)
        robot = read_records(args.robot)
        result = calibration_result(motive, robot, args.pair_count)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
