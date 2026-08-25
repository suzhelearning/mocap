"""手腕位姿计算 + 手部骨架拼接(纯函数,拼接公式见计划文档)。

坐标系:
- 世界系 G:Motive Streaming 已输出 x-forward / y-left / z-up 右手系、米制;
  rigid body 的 position/quaternion_xyzw 已在 G 中,消费端不得重复做 Up Axis 变换。
- 手背刚体局部系 B:由 Motive 刚体模型定义;body 模式 wrist_offset 位于 B 中。
- Manus 骨架局部系 H:z-up 右手系、米制;每手套各自建系,只使用帧内相对量。

拼接:g_i^G = p_w^G + R_{G←W} · (A_{W←H} · (p_i^H − p_0^H))
其中 p_0 为手掌 root 节点(节点 0),A 是局部→局部轴对齐,不是世界坐标变换。
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
    """Streaming 世界系 G 中的手背刚体位姿 → 解剖手腕位姿。

    Motive 已对 position/quaternion 做 Z-up Streaming 变换,这里不再转换世界轴。
    返回 (position[3], quaternion_wxyz[4]):
    - mode=body:  p_w = p_b + R_b·o_B,R_w = R_b·R_extra(局部外参,推荐)
    - mode=world: p_w = p_b + o_G,R_w = R_extra(固定世界方向)
    R_extra 是手背刚体局部系 B → 手腕局部系 W 的标定旋转。
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
    wrist_pos: np.ndarray,            # (3,) Streaming 世界系 G
    wrist_q_wxyz: np.ndarray,         # (4,) R_{G←W},wxyz 序
    axis: np.ndarray,                 # (3,3) Manus H→手腕局部 W 轴变换 A
) -> np.ndarray:                      # (N,3) 世界系 G 全局节点
    """以手腕为基准拼接:局部偏移经轴变换后叠加到手腕位姿。"""
    nodes = np.asarray(nodes_local, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"nodes_local 须为 (N,3),实际 {nodes.shape}")
    if not (0 <= palm_idx < nodes.shape[0]):
        raise ValueError(f"palm_idx {palm_idx} 越界(共 {nodes.shape[0]} 节点)")
    d = nodes - nodes[palm_idx]               # (N,3) Manus H 局部偏移
    d_wrist = (axis @ d.T).T                  # (N,3) 手腕局部系 W 偏移
    R_w = rotmat_from_wxyz(wrist_q_wxyz)      # R_{G←W}
    return np.asarray(wrist_pos, dtype=float) + (R_w @ d_wrist.T).T


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
