"""四元数与旋转矩阵纯函数(全部 numpy,输入输出规范明确)。

四元数约定:
- 协议/线格式为 xyzw 序(如 NatNet rigid body 的 quaternion_xyzw)
- 内部计算一律 wxyz 序,转换函数在边界使用
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def quat_xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray:
    """xyzw → wxyz,长度为 4。"""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"四元数长度须为 4,实际 {q.shape}")
    return q[[3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: Sequence[float]) -> np.ndarray:
    """wxyz → xyzw。"""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"四元数长度须为 4,实际 {q.shape}")
    return q[[1, 2, 3, 0]]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """四元数乘法(wxyz 序),结果为单位四元数(若输入为单位)。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    wa, xa, ya, za = a
    wb, xb, yb, zb = b
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """单位四元数(wxyz)旋转向量 v。"""
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    v = np.asarray(v, dtype=float)
    return rotmat_from_wxyz(q) @ v


def rotmat_from_wxyz(q: np.ndarray) -> np.ndarray:
    """单位四元数(wxyz) → 3×3 旋转矩阵。"""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_wxyz(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """身体系 ZYX 欧拉角(度)→ 四元数(wxyz 序)。

    yaw=绕 Z、pitch=绕 Y、roll=绕 X,按 Z→Y→X 复合,
    用于手腕相对背部刚体的固定姿态修正 R_extra。
    """
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """四元数球面插值(wxyz 序),t ∈ [0,1]。

    退化(近似平行)回退线性插值并归一化;输入无需预归一化。
    """
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def compose_axis(permutation: Sequence[int], signs: Sequence[float]) -> np.ndarray:
    """由 permutation + signs 构造轴变换矩阵 A:(A·d)_j = signs[j]·d[permutation[j]]。

    供 config.axis_matrix 使用;这里独立实现便于纯函数测试。
    """
    A = np.zeros((3, 3), dtype=float)
    for j in range(3):
        A[j, permutation[j]] = signs[j]
    return A
