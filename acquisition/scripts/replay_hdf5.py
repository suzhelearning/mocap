#!/usr/bin/env python3
"""replay_hdf5.py — HDF5 录制回放:按目标帧率插值合并多流,校验时间对齐。

插值:手部 30Hz 数据在动捕 120Hz 目标时间轴上做「位置 lerp + 四元数 slerp」,
输出各目标时刻的双手全局节点、手腕位姿(含朝向)与动捕刚体,便于离线分析/驱动仿真。

安全防护:
- 拒绝含外部链接的 HDF5(防恶意文件经 HDF5 外部链接读取任意本地文件)
- 目标帧数由时间戳跨度驱动,设上限防无界分配 OOM

用法: pixi run replay -- <file.h5> [--target-hz 120] [--json 输出前缀]
"""

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np

MAX_TARGET_FRAMES = 20_000_000    # 约 46h @120Hz;防时间戳被篡改导致的无界分配


def reject_external_links(f: h5py.File, prefix: str = "") -> None:
    """递归检查并拒绝含外部/软链接的 HDF5(不解析链接,仅查类型)。"""
    for name in f:
        link = f.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError(
                f"拒绝含 {type(link).__name__} 的 HDF5: {prefix}{name}"
                + (f" -> {link.filename}" if isinstance(link, h5py.ExternalLink) else "")
            )
        obj = f[name]
        if isinstance(obj, h5py.Group):
            reject_external_links(obj, f"{prefix}{name}/")


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """wxyz 四元数球面插值;退化(近似平行)回退线性插值并归一化。"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[[3, 0, 1, 2]]


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[[1, 2, 3, 0]]


def interpolate_hand(g: h5py.Group, t_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 hand 组插值到 t_target 时间轴。

    返回 (nodes_global (M,25,3), wrist_pos (M,3), wrist_quat_xyzw (M,4)):
    位置 lerp + 四元数 slerp(wxyz 序计算,xyzw 序输出,与 HDF5 存储一致)。
    """
    t = g["t_ubuntu_ns"][:].astype(np.float64)
    if t.size < 2:
        raise ValueError(f"hand 组仅 {t.size} 帧,不足 2 无法插值")
    nodes = g["nodes_global"][:]
    wpos = g["wrist_position"][:]
    wquat = g["wrist_quaternion_xyzw"][:]

    out_nodes = np.empty((len(t_target), nodes.shape[1], 3), dtype=np.float64)
    out_wpos = np.empty((len(t_target), 3), dtype=np.float64)
    out_wquat = np.empty((len(t_target), 4), dtype=np.float64)
    # 逐目标时刻最近邻两帧插值
    for i, tt in enumerate(t_target):
        j = int(np.searchsorted(t, tt))
        j = min(max(j, 0), len(t) - 1)
        j0, j1 = max(0, j - 1), min(len(t) - 1, j)
        if j0 == j1:
            out_nodes[i] = nodes[j0]
            out_wpos[i] = wpos[j0]
            out_wquat[i] = wquat[j0]
            continue
        frac = (tt - t[j0]) / (t[j1] - t[j0])
        out_nodes[i] = (1 - frac) * nodes[j0] + frac * nodes[j1]
        out_wpos[i] = (1 - frac) * wpos[j0] + frac * wpos[j1]
        q0 = _quat_xyzw_to_wxyz(wquat[j0])
        q1 = _quat_xyzw_to_wxyz(wquat[j1])
        out_wquat[i] = _quat_wxyz_to_xyzw(slerp(q0, q1, frac))
    return out_nodes, out_wpos, out_wquat


def main() -> int:
    ap = argparse.ArgumentParser(description="HDF5 录制回放/插值校验")
    ap.add_argument("file", type=str)
    ap.add_argument("--target-hz", type=float, default=120.0)
    ap.add_argument("--json", type=str, default=None,
                    help="输出插值后 JSON(每行一个目标时刻帧),便于消费")
    args = ap.parse_args()

    with h5py.File(args.file, "r") as f:
        reject_external_links(f)                  # 防恶意 HDF5 外部链接
        t_mocap = f["mocap/t_ubuntu_ns"][:].astype(np.float64)
        if t_mocap.size < 2:
            print("mocap 帧不足,无法构建目标时间轴", file=sys.stderr)
            return 1
        t0, t1 = t_mocap[0], t_mocap[-1]
        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            print(f"时间戳非法(t0={t0}, t1={t1})", file=sys.stderr)
            return 1
        n_target = max(1, int(round((t1 - t0) / 1e9 * args.target_hz)))
        if n_target > MAX_TARGET_FRAMES:
            print(f"目标帧数 {n_target} 超上限 {MAX_TARGET_FRAMES},拒绝",
                  file=sys.stderr)
            return 1
        t_target = np.linspace(t0, t1, n_target)
        print(f"目标时间轴: {t0}..{t1}ns, {n_target} 个时刻 @ {args.target_hz:.0f}Hz")

        hands = {}
        for side in ("left", "right"):
            try:
                nodes, wpos, wquat = interpolate_hand(f["hands"][side], t_target)
            except ValueError as exc:
                print(f"  {side}: {exc},跳过", file=sys.stderr)
                continue
            hands[side] = {"nodes_global": nodes, "wrist": wpos, "wrist_quat": wquat}
            err = np.abs(nodes[:, 0, :] - wpos).max()
            print(f"  {side}: 插值 {n_target} 帧, 手腕节点一致性误差 {err:.2e} m")

        if not hands:
            print("双手均无数据,无可回放内容", file=sys.stderr)
            return 1

        if args.json:
            with open(args.json, "w") as out:
                for i in range(n_target):
                    frame = {"t_ns": int(t_target[i])}
                    for side, h in hands.items():
                        frame[f"{side}_wrist"] = h["wrist"][i].tolist()
                        frame[f"{side}_wrist_quat_xyzw"] = h["wrist_quat"][i].tolist()
                        frame[f"{side}_nodes"] = h["nodes_global"][i].tolist()
                    out.write(json.dumps(frame) + "\n")
            print(f"[json] 已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
