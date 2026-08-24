"""kinematics.py 纯函数测试。"""

from __future__ import annotations

import numpy as np
import pytest

from acquisition.kinematics import (
    compose_axis,
    euler_wxyz,
    quat_mul,
    quat_rotate,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    rotmat_from_wxyz,
)


def test_quat_conversions_roundtrip():
    q = [0.1, 0.2, 0.3, 0.9]
    assert np.allclose(quat_xyzw_to_wxyz(q), [0.9, 0.1, 0.2, 0.3])
    assert np.allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q)


def test_quat_mul_identity():
    ident = np.array([1.0, 0, 0, 0])
    q = np.array([0.9239, 0.3827, 0, 0])   # 绕 x 45°
    assert np.allclose(quat_mul(q, ident), q)
    assert np.allclose(quat_mul(ident, q), q)


def test_quat_rotate_z_90():
    """绕 z 轴 90° 旋转 +x 应得 +y。"""
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # wxyz
    v = np.array([1.0, 0.0, 0.0])
    assert np.allclose(quat_rotate(q, v), [0.0, 1.0, 0.0], atol=1e-9)


def test_rotmat_from_wxyz():
    q = np.array([1.0, 0, 0, 0])
    assert np.allclose(rotmat_from_wxyz(q), np.eye(3))
    # 绕 z 90°
    qz = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    R = rotmat_from_wxyz(qz)
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_euler_wxyz_identity():
    assert np.allclose(euler_wxyz(0, 0, 0), [1, 0, 0, 0])


def test_euler_wxyz_yaw_90():
    """yaw=90° → 绕 z 90°。"""
    q = euler_wxyz(90, 0, 0)
    assert np.allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-9)


def test_compose_axis_default():
    """默认 A·d = (-d_y, d_x, d_z)(骨架 z-up → motive x-forward z-up)。"""
    A = compose_axis([1, 0, 2], [-1, 1, 1])
    assert np.allclose(A @ np.array([1.0, 2.0, 3.0]), [-2.0, 1.0, 3.0])
    assert np.isclose(np.linalg.det(A), 1.0)


@pytest.mark.parametrize("perm,signs,det", [
    ([0, 1, 2], [1, 1, 1], 1.0),     # 恒等
    ([1, 0, 2], [-1, 1, 1], 1.0),    # 默认(绕 Y +90°)
    ([1, 2, 0], [1, 1, 1], 1.0),     # 循环移位
    ([0, 2, 1], [1, 1, 1], -1.0),    # 镜像(应被 config 拒绝)
])
def test_compose_axis_is_orthogonal(perm, signs, det):
    """任意排列+符号组合都是正交阵,det 按组合为 ±1;真旋转需 det=+1。"""
    A = compose_axis(perm, signs)
    assert np.isclose(np.linalg.det(A), det)
    assert np.allclose(A @ A.T, np.eye(3), atol=1e-12)
