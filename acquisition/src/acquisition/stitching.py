"""手腕位姿计算 + 手部骨架拼接(纯函数,拼接公式见计划文档)。

坐标系:
- Motive 系 M:x-forward z-up 右手系、米制(rigid body 的 position/quaternion_xyzw 所在系,
  2026-08-24 起 Motive 全局坐标系已改为 x 前向、z 向上)
- 骨架系 S:z-up 右手系、米制(manus raw_skeleton 节点,每手套各自建系,
  拼接只用帧内相对量,与 S 世界原点无关)

拼接:g_i = p_w + R_w · (A · (p_i^S − p_0^S))
其中 p_0 为手掌 root 节点(节点 0,edges 中 chainType==13 校验),g_0 ≡ p_w。
"""

from __future__ import annotations

import numpy as np

from .config import WristOffset
from .kinematics import (
    euler_wxyz,
    quat_mul,
    quat_rotate,
    quat_xyzw_to_wxyz,
    quat_wxyz_to_xyzw,
    rotmat_from_wxyz,
)


def wrist_pose_from_back(
    pos_b: np.ndarray,
    quat_b_xyzw: np.ndarray,
    offset: WristOffset,
) -> tuple[np.ndarray, np.ndarray]:
    """背部刚体位姿(Motive 系,四元数 xyzw 序)→ 手腕位姿。

    返回 (position[3], quaternion_wxyz[4]):
    - mode=body:  p_w = p_b + R_b·o,R_w = R_b·R_extra(随躯干转动,物理正确)
    - mode=world: p_w = p_b + o,R_w = R_extra(固定世界方向)
    R_extra 由 yaw/pitch/roll 构造,默认单位阵。
    """
    p_b = np.asarray(pos_b, dtype=float)
    q_b = quat_xyzw_to_wxyz(quat_b_xyzw)
    R_b = rotmat_from_wxyz(q_b)
    o = np.asarray(offset.xyz, dtype=float)

    q_extra = euler_wxyz(offset.yaw_deg, offset.pitch_deg, offset.roll_deg)
    if offset.mode == "body":
        p_w = p_b + R_b @ o
        q_w = quat_mul(q_b, q_extra)
    else:  # world
        p_w = p_b + o
        q_w = q_extra
    return p_w, q_w / np.linalg.norm(q_w)


def extract_rigid_body(
    frame: dict, rigid_id: int,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """从 mocap 帧的 rigid_bodies 中按 ID 取 (position, quaternion_xyzw, tracking_valid)。

    找不到返回 None。
    """
    for rb in frame.get("rigid_bodies", []):
        if rb.get("id") == rigid_id:
            return (
                np.asarray(rb["position"], dtype=float),
                np.asarray(rb["quaternion_xyzw"], dtype=float),
                bool(rb.get("tracking_valid", True)),
            )
    return None


def hand_nodes_to_global(
    nodes_local: np.ndarray,          # (N,3) 骨架系原始节点坐标
    palm_idx: int,                    # 手掌 root 节点索引(默认 0)
    wrist_pos: np.ndarray,            # (3,) Motive 系
    wrist_q_wxyz: np.ndarray,         # (4,) wxyz 序
    axis: np.ndarray,                 # (3,3) 骨架系→Motive 系轴变换 A
) -> np.ndarray:                      # (N,3) Motive 系全局节点
    """以手腕为基准拼接:局部偏移经轴变换后叠加到手腕位姿。"""
    nodes = np.asarray(nodes_local, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"nodes_local 须为 (N,3),实际 {nodes.shape}")
    if not (0 <= palm_idx < nodes.shape[0]):
        raise ValueError(f"palm_idx {palm_idx} 越界(共 {nodes.shape[0]} 节点)")
    d = nodes - nodes[palm_idx]               # (N,3) 骨架系局部偏移
    d_motive = (axis @ d.T).T                 # (N,3) Motive 系偏移
    R_w = rotmat_from_wxyz(wrist_q_wxyz)
    return np.asarray(wrist_pos, dtype=float) + (R_w @ d_motive.T).T


def stitch_hand(
    nodes_local: np.ndarray,
    palm_idx: int,
    pos_b: np.ndarray,
    quat_b_xyzw: np.ndarray,
    offset: WristOffset,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一步完成:手腕位姿计算 + 骨架拼接。

    返回 (nodes_global (N,3), wrist_position (3,), wrist_quaternion_xyzw (4,))。
    """
    p_w, q_w = wrist_pose_from_back(pos_b, quat_b_xyzw, offset)
    nodes_global = hand_nodes_to_global(nodes_local, palm_idx, p_w, q_w, axis)
    return nodes_global, p_w, quat_wxyz_to_xyzw(q_w)
