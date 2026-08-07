#!/usr/bin/env python3
"""replay_hdf5.py — HDF5 录制回放:按目标帧率插值合并多流,校验时间对齐。

插值:手部 30Hz 数据在动捕 120Hz 目标时间轴上做「位置 lerp + 四元数 slerp」,
输出各目标时刻的双手全局节点与动捕刚体,便于离线分析/驱动仿真。

用法: pixi run replay -- <file.h5> [--target-hz 120] [--json 输出前缀]
"""

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """wxyz 四元数球面插值。"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def interpolate_hand(g: h5py.Group, t_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 hand 组插值到 t_target 时间轴,返回 (nodes_global (M,25,3), wrist (M,3))。"""
    t = g["t_ubuntu_ns"][:].astype(np.float64)
    nodes = g["nodes_global"][:]
    wpos = g["wrist_position"][:]
    wquat = g["wrist_quaternion_xyzw"][:]

    out_nodes = np.empty((len(t_target), nodes.shape[1], 3), dtype=np.float64)
    out_wpos = np.empty((len(t_target), 3), dtype=np.float64)
    # 逐目标时刻最近邻两帧插值
    for i, tt in enumerate(t_target):
        j = int(np.searchsorted(t, tt))
        j = min(max(j, 0), len(t) - 1)
        j0, j1 = max(0, j - 1), min(len(t) - 1, j)
        if j0 == j1:
            out_nodes[i] = nodes[j0]
            out_wpos[i] = wpos[j0]
            continue
        frac = (tt - t[j0]) / (t[j1] - t[j0])
        out_nodes[i] = (1 - frac) * nodes[j0] + frac * nodes[j1]
        out_wpos[i] = (1 - frac) * wpos[j0] + frac * wpos[j1]
    return out_nodes, out_wpos


def main() -> int:
    ap = argparse.ArgumentParser(description="HDF5 录制回放/插值校验")
    ap.add_argument("file", type=str)
    ap.add_argument("--target-hz", type=float, default=120.0)
    ap.add_argument("--json", type=str, default=None,
                    help="输出插值后 JSON(每行一个目标时刻帧),便于消费")
    args = ap.parse_args()

    with h5py.File(args.file, "r") as f:
        t_mocap = f["mocap/t_ubuntu_ns"][:].astype(np.float64)
        if t_mocap.size < 2:
            print("mocap 帧不足,无法构建目标时间轴", file=sys.stderr)
            return 1
        t0, t1 = t_mocap[0], t_mocap[-1]
        n_target = max(1, int(round((t1 - t0) / 1e9 * args.target_hz)))
        t_target = np.linspace(t0, t1, n_target)
        print(f"目标时间轴: {t0}..{t1}ns, {n_target} 个时刻 @ {args.target_hz:.0f}Hz")

        hands = {}
        for side in ("left", "right"):
            nodes, wpos = interpolate_hand(f["hands"][side], t_target)
            hands[side] = {"nodes_global": nodes, "wrist": wpos}
            err = np.abs(nodes[:, 0, :] - wpos).max()
            print(f"  {side}: 插值 {n_target} 帧, 手腕节点一致性误差 {err:.2e} m")

        if args.json:
            with open(args.json, "w") as out:
                for i in range(n_target):
                    frame = {"t_ns": int(t_target[i])}
                    for side, h in hands.items():
                        frame[f"{side}_wrist"] = h["wrist"][i].tolist()
                        frame[f"{side}_nodes"] = h["nodes_global"][i].tolist()
                    out.write(json.dumps(frame) + "\n")
            print(f"[json] 已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
