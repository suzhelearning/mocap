"""stitching.py 拼接测试 — 数值示例来自计划文档(Plan agent 推导)。"""

from __future__ import annotations

import numpy as np

from acquisition.config import WristOffset
from acquisition.kinematics import compose_axis, euler_wxyz, quat_mul, quat_rotate
from acquisition.stitching import hand_nodes_to_global, stitch_hand, wrist_pose_from_back

AXIS = compose_axis([1, 0, 2], [-1, 1, 1])   # A·d = (-d_y, d_x, d_z)


def _skeleton_frame() -> np.ndarray:
    """25 节点,手掌 root 在 (0.020, 0.010, 0.000),食指尖(节点 15)在 +z 上方 9cm。"""
    nodes = np.zeros((25, 3))
    nodes[0] = [0.020, 0.010, 0.000]
    nodes[15] = [0.030, 0.015, 0.090]
    return nodes


def test_hand_nodes_to_global_no_back_rotation():
    """情形 A:背部刚体无旋转,offset body 模式。"""
    nodes = _skeleton_frame()
    d = nodes[15] - nodes[0]                     # (0.010, 0.005, 0.090)
    assert np.allclose(AXIS @ d, [-0.005, 0.010, 0.090], atol=1e-12)

    p_b = np.array([0.30, 1.20, -0.20])
    q_b = np.array([0.0, 0.0, 0.0, 1.0])          # xyzw 恒等
    offset = WristOffset(mode="body", xyz=(0.15, -0.30, -0.05))

    p_w, q_w = wrist_pose_from_back(p_b, q_b, offset)
    assert np.allclose(p_w, [0.45, 0.90, -0.25], atol=1e-12)
    assert np.allclose(q_w, [1, 0, 0, 0], atol=1e-12)

    g = hand_nodes_to_global(nodes, 0, p_w, q_w, AXIS)
    assert np.allclose(g[0], p_w, atol=1e-12)              # 手腕节点恒等于手腕位姿
    assert np.allclose(g[15], [0.445, 0.91, -0.16], atol=1e-9)


def test_hand_nodes_to_global_with_back_rotation():
    """情形 B:背部刚体绕 motive z 轴(新系竖轴)转 30°。"""
    nodes = _skeleton_frame()
    angle = np.deg2rad(30)
    q_b = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])  # xyzw,绕 z
    p_b = np.array([0.30, 1.20, -0.20])
    offset = WristOffset(mode="body", xyz=(0.15, -0.30, -0.05))

    p_w, q_w = wrist_pose_from_back(p_b, q_b, offset)
    assert np.allclose(p_w, [0.579904, 1.015192, -0.25], atol=1e-5)
    assert np.allclose(p_w, p_b + quat_rotate(q_w, offset.xyz), atol=1e-9)

    g = hand_nodes_to_global(nodes, 0, p_w, q_w, AXIS)
    assert np.allclose(g[0], p_w, atol=1e-12)
    # 物理自洽:指尖相对手腕 = R_w·(A·d)
    d = nodes[15] - nodes[0]
    assert np.allclose(g[15] - g[0], quat_rotate(q_w, AXIS @ d), atol=1e-9)
    # 手指仍相对世界竖直向上(9cm),欧氏距离保持
    assert np.isclose(g[15][2] - g[0][2], 0.09, atol=1e-5)
    assert np.isclose(np.linalg.norm(g[15] - g[0]), np.linalg.norm(AXIS @ d), atol=1e-12)


def test_wrist_pose_world_mode():
    """world 模式:offset 不随背板旋转。"""
    p_b = np.array([0.3, 1.2, -0.2])
    q_b = np.array([0.0, np.sin(np.deg2rad(15)), 0.0, np.cos(np.deg2rad(15))])
    offset = WristOffset(mode="world", xyz=(0.15, -0.30, -0.05))
    p_w, q_w = wrist_pose_from_back(p_b, q_b, offset)
    assert np.allclose(p_w, p_b + offset.xyz, atol=1e-12)
    assert np.allclose(q_w, [1, 0, 0, 0], atol=1e-12)


def test_wrist_pose_with_orientation_correction():
    """yaw 修正:手腕姿态 = 背部姿态 ⊗ R_extra。"""
    p_b = np.zeros(3)
    q_b = np.array([0.0, 0.0, 0.0, 1.0])
    offset = WristOffset(mode="body", xyz=(0.0, 0.0, 0.0), yaw_deg=90)
    _, q_w = wrist_pose_from_back(p_b, q_b, offset)
    assert np.allclose(q_w, euler_wxyz(90, 0, 0), atol=1e-9)


def test_stitch_hand_returns_xyzw():
    """stitch_hand 的存储四元数为 xyzw 序(与协议一致)。"""
    nodes = _skeleton_frame()
    p_b = np.array([0.30, 1.20, -0.20])
    q_b = np.array([0.0, 0.0, 0.0, 1.0])
    offset = WristOffset(mode="body", xyz=(0.15, -0.30, -0.05))
    g, p_w, q_xyzw = stitch_hand(nodes, 0, p_b, q_b, offset, AXIS)
    assert np.allclose(g[0], p_w, atol=1e-12)
    assert np.allclose(q_xyzw, [0, 0, 0, 1], atol=1e-12)
    assert np.allclose(g[15], [0.445, 0.91, -0.16], atol=1e-9)


def test_hand_nodes_to_global_palm_offset_independent():
    """拼接与骨架世界原点无关:整体平移骨架,全局节点只平移手腕对应量。"""
    nodes = _skeleton_frame()
    p_b = np.array([0.30, 1.20, -0.20])
    q_b = np.array([0.0, 0.0, 0.0, 1.0])
    offset = WristOffset(mode="body", xyz=(0.15, -0.30, -0.05))
    p_w, q_w = wrist_pose_from_back(p_b, q_b, offset)

    g1 = hand_nodes_to_global(nodes, 0, p_w, q_w, AXIS)
    shifted = nodes + np.array([1.0, -2.0, 0.5])
    g2 = hand_nodes_to_global(shifted, 0, p_w, q_w, AXIS)
    # 局部偏移不变 → 全局节点不变(手掌位置不同只影响偏移基准,抵消)
    assert np.allclose(g1, g2, atol=1e-12)
