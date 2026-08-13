#!/usr/bin/env python3
"""mano_fit.py — 纯 numpy MANO 前向层与关键点直接驱动表面。

MANO 模型输入仍是内部骨骼旋转，但调用方只需提供：
  - ``mano_skeleton(21,3)``：MediaPipe 顺序的世界系关键点；
  - ``beta(10)``：从最多 1000 帧稳健段长估计的一次性手形参数。

表面驱动使用关键点构造确定性关节变换并执行 MANO blend-shape + LBS，
不做逐帧数值拟合。显示的 16 个 MANO 关节直接取自 ``mano_skeleton``。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MANO_DIR = Path(__file__).resolve().parents[2] / "assets" / "mano"

# ── MANO ↔ MediaPipe 关节映射 ─────────────────────────────────────────
# MANO 的 J_regressor / kintree_table 只有 16 个运动学关节，顺序由
# MANO_{LEFT,RIGHT}.pkl 的 kintree_table 固定为:
#   0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
# 采集文件的 mano_skeleton 是 MediaPipe 21 点:
#   0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# 注意: MediaPipe 的 5 个 fingertip 不是 MANO 的 J_regressor 关节，
# 而是由官方 MANO vertex_ids.py 指定的表面顶点补齐。
_MANO_TO_MP = np.asarray(
    (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3),
    dtype=np.int64,
)
MANO_JOINT_NAMES = (
    "wrist",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "thumb_cmc", "thumb_mcp", "thumb_ip",
)


def mano_joints16_from_joints21(joints21: np.ndarray) -> np.ndarray:
    """MediaPipe 顺序的 21 点转为 MANO 原生 16 运动学关节顺序。"""
    joints21 = np.asarray(joints21)
    if joints21.ndim < 2 or joints21.shape[-2:] != (21, 3):
        raise ValueError(
            f"joints21 必须以 (21,3) 结尾，实际为 {joints21.shape}",
        )
    return joints21[..., _MANO_TO_MP, :]

_MP_TIPS_JOINT = (3, 6, 9, 12, 15)   # index/middle/pinky/ring/thumb
# 官方 smplx.vertex_ids['mano']，按 _MP_TIPS_JOINT 的顺序排列。
# 不能用“候选蒙皮顶点最大投影”推断，middle/ring/pinky 会产生相邻
# 顶点的 off-by-one，导致可视化尖端和采集 keypoint 不再是同一点。
_MANO_TIP_VERTICES = (320, 443, 671, 554, 744)
_MP_TIP_SLOT_FROM_CHAIN = {15: 4, 3: 8, 6: 12, 12: 16, 9: 20}
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回把三维向量 a 旋到 b 的最小旋转矩阵。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.eye(3)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    K = np.asarray([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))




def _kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """求列向量旋转 R，使 R @ A[i] 尽量接近 B[i]。"""
    H = np.asarray(A, dtype=np.float64).T @ np.asarray(B, dtype=np.float64)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    return R




def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """批量轴角→旋转矩阵。输入 (...,3),输出 (...,3,3)。"""
    aa = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(theta, 1e-10)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=np.float64)
    K[..., 0, 1] = -z
    K[..., 0, 2] = y
    K[..., 1, 0] = z
    K[..., 1, 2] = -x
    K[..., 2, 0] = -y
    K[..., 2, 1] = x
    I = np.eye(3)
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    K2 = np.einsum("...ij,...jk->...ik", K, K)
    return I + sin_t * K + (1.0 - cos_t) * K2


class ManoLayer:
    """numpy MANO 前向层。batch 支持:theta/root_orient/trans 首维为 batch。"""

    def __init__(self, pkl_path: Path) -> None:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        self.v_template = np.asarray(d["v_template"], dtype=np.float64)       # (778,3)
        self.shapedirs = np.asarray(d["shapedirs"].r, dtype=np.float64)       # (778,3,10)
        self.posedirs = np.asarray(d["posedirs"], dtype=np.float64)           # (778,3,135)
        W = d["weights"]
        W = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
        self.weights = np.asarray(W, dtype=np.float64)                        # (778,16)
        Jr = d["J_regressor"]
        Jr = Jr.toarray() if hasattr(Jr, "toarray") else np.asarray(Jr)
        self.J_regressor = np.asarray(Jr, dtype=np.float64)                   # (16,778)
        kt = np.asarray(d["kintree_table"], dtype=np.int64)
        parents = kt[0].copy()
        parents[parents >= 2**31] = -1          # uint32 存的 -1 溢出还原
        if kt.shape != (2, 16):
            raise ValueError(
                f"MANO kintree_table 必须是 (2,16),实际为 {kt.shape}",
            )
        self.parents = parents.tolist()                                       # parent of joint i
        self.faces = np.asarray(d["f"], dtype=np.int64)                       # (1538,3)
        rest_J = self.J_regressor @ self.v_template
        if self.J_regressor.shape != (16, 778):
            raise ValueError(
                f"MANO J_regressor 必须是 (16,778),实际为 {self.J_regressor.shape}",
            )
        if self.weights.shape != (778, 16):
            raise ValueError(
                f"MANO weights 必须是 (778,16),实际为 {self.weights.shape}",
            )
        # MANO 原生只有 16 个 J_regressor 关节。MediaPipe 的 5 个
        # fingertip 使用官方 MANO 顶点索引，不把相邻表面顶点误当成关节。
        self.tips = _MANO_TIP_VERTICES
        self._rest_joints = rest_J
        self._rest_cache: np.ndarray | None = None

    @property
    def rest_verts(self) -> np.ndarray:
        """β=0、θ=0 的模板顶点(供初始渲染)。"""
        return self.v_template.copy()

    def _shaped(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """v_shaped (B,778,3) 与 J (B,16,3);beta (B,10) 或 (10,)。"""
        single = beta.ndim == 1
        if single:
            beta = beta[None]
        v_shaped = self.v_template[None] + np.einsum(
            "vck,bk->bvc", self.shapedirs, beta)
        J = np.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)
        return (v_shaped, J) if not single else (v_shaped[0], J[0])

    def forward(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        root_orient: np.ndarray,
        trans: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """前向:返回 (verts (...,778,3), joints21 (...,21,3)),MediaPipe 顺序。"""
        single = theta.ndim == 1
        if single:
            theta, root_orient, trans = theta[None], root_orient[None], trans[None]
        B = theta.shape[0]
        beta_b = beta[None] if beta.ndim == 1 else beta
        v_shaped, J = self._shaped(beta_b)

        # 姿态 blend shapes:只对 15 个手关节 (R-I) 展平 135 维
        pose_aa = np.concatenate([root_orient, theta], axis=1).reshape(B, 16, 3)
        R = _rodrigues(pose_aa)                                               # (B,16,3,3)
        I = np.eye(3)
        pose_feat = (R[:, 1:, :, :] - I).reshape(B, -1)                       # (B,135)
        v_posed = v_shaped + np.einsum("vcp,bp->bvc", self.posedirs, pose_feat)

        # 全局关节旋转与平移(kintree 链累积;root 平移 = J_0,与官方实现一致)
        R_glob = np.zeros((B, 16, 3, 3), dtype=np.float64)
        t_glob = np.zeros((B, 16, 3), dtype=np.float64)
        t_glob[:, 0] = J[:, 0]
        R_glob[:, 0] = R[:, 0]
        for j in range(1, 16):
            p = self.parents[j]
            R_glob[:, j] = np.einsum("bij,bjk->bik", R_glob[:, p], R[:, j])
            t_glob[:, j] = (
                np.einsum("bij,bj->bi", R_glob[:, p], J[:, j] - J[:, p])
                + t_glob[:, p]
            )

        # LBS 蒙皮(官方形式):verts[v] = Σ_j W[v,j] (R_j (v_posed[v] - J_j) + t_j)
        rotated = np.einsum("bimn,bvn->bimv", R_glob, v_posed)             # (B,16,3,778)
        rotated = rotated.transpose(0, 1, 3, 2)                            # (B,16,778,3)
        RJ = np.einsum("bimn,bin->bim", R_glob, J)                         # R_j J_j (B,16,3)
        rotated = rotated - RJ[:, :, None, :] + t_glob[:, :, None, :]
        verts = np.einsum("vj,bjvm->bvm", self.weights, rotated)

        # 关节全局位置 = 变换矩阵平移列(官方 A_global)
        J_glob = t_glob.copy()                                              # (B,16,3)

        # 21 点按 MediaPipe 顺序组装(与 H5 mano_skeleton 一致):
        #   mp0=wrist, mp1-4=thumb(MANO 13-15 + tip), mp5-8=index(MANO 1-3 + tip),
        #   mp9-12=middle(MANO 4-6 + tip), mp13-16=ring(MANO 10-12 + tip),
        #   mp17-20=pinky(MANO 7-9 + tip)
        J21 = np.empty((B, 21, 3), dtype=np.float64)
        J21[:, 0] = J_glob[:, 0]
        J21[:, 1:4] = J_glob[:, 13:16]          # thumb
        J21[:, 5:8] = J_glob[:, 1:4]            # index
        J21[:, 9:12] = J_glob[:, 4:7]           # middle
        J21[:, 13:16] = J_glob[:, 10:13]        # ring
        J21[:, 17:20] = J_glob[:, 7:10]         # pinky
        tips_v = verts[:, np.asarray(self.tips), :]    # 顺序= _MP_TIPS_JOINT
        for chain_end, slot in _MP_TIP_SLOT_FROM_CHAIN.items():
            J21[:, slot] = tips_v[:, _MP_TIPS_JOINT.index(chain_end)]

        verts = verts + trans[:, None, :]
        J21 = J21 + trans[:, None, :]
        if single:
            return verts[0], J21[0]
        return verts, J21


def load_mano(side: str) -> ManoLayer:
    name = "RIGHT" if side == "right" else "LEFT"
    return ManoLayer(MANO_DIR / f"MANO_{name}.pkl")


def mesh_from_skeleton(
    layer: ManoLayer,
    skeleton: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """用 MediaPipe 21 点和 beta 直接驱动 MANO 表面。

    返回 ``(vertices(778,3), joints16(16,3))``。``joints16`` 是输入关键点
    按 MANO 原生关节顺序的直接重排，不是模型回归结果。表面通过确定性的
    目标骨架变换与 LBS 生成，不运行 IK/最小二乘等逐帧优化。
    """
    skeleton = np.asarray(skeleton, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if skeleton.shape != (21, 3):
        raise ValueError(
            f"skeleton 必须是 (21,3)，实际为 {skeleton.shape}",
        )
    if beta.shape != (10,):
        raise ValueError(f"beta 必须是 (10,)，实际为 {beta.shape}")
    if not np.isfinite(skeleton).all() or not np.isfinite(beta).all():
        raise ValueError("skeleton 或 beta 含 NaN/Inf")

    shaped_verts, rest_joints = layer._shaped(beta)
    target_joints = mano_joints16_from_joints21(skeleton)
    parents = np.asarray(layer.parents, dtype=np.int64)

    # wrist 朝向由四指 MCP 的掌面整体决定。
    palm = np.asarray((1, 4, 10, 7), dtype=np.int64)
    global_rot = np.tile(np.eye(3), (16, 1, 1))
    global_rot[0] = _kabsch_rotation(
        rest_joints[palm] - rest_joints[0],
        target_joints[palm] - target_joints[0],
    )

    # 每个关节使用下一骨段方向确定弯曲；DIP 使用对应 fingertip 作为虚拟
    # 子节点。绕骨轴的不可观测 twist 采用从父节点平行传递的最小旋转。
    tip_vertex = dict(zip(_MP_TIPS_JOINT, _MANO_TIP_VERTICES))
    for joint in range(1, 16):
        children = np.flatnonzero(parents == joint)
        if children.size:
            child = int(children[0])
            rest_direction = rest_joints[child] - rest_joints[joint]
            target_direction = target_joints[child] - target_joints[joint]
        else:
            rest_direction = (
                shaped_verts[tip_vertex[joint]] - rest_joints[joint]
            )
            target_direction = (
                skeleton[_MP_TIP_SLOT_FROM_CHAIN[joint]]
                - target_joints[joint]
            )
        parent = int(parents[joint])
        transported = global_rot[parent] @ rest_direction
        global_rot[joint] = (
            _rotation_between(transported, target_direction)
            @ global_rot[parent]
        )

    # MANO pose blend shapes 使用父节点局部旋转；目标骨架坐标直接作为每个
    # 关节变换的平移锚点，因此无需另存 translation 或逐帧 scale。
    local_rot = np.empty_like(global_rot)
    local_rot[0] = global_rot[0]
    for joint in range(1, 16):
        parent = int(parents[joint])
        local_rot[joint] = global_rot[parent].T @ global_rot[joint]
    pose_feature = (local_rot[1:] - np.eye(3)).reshape(-1)
    posed_verts = shaped_verts + np.einsum(
        "vcp,p->vc", layer.posedirs, pose_feature,
    )

    transformed = (
        np.einsum("imn,vn->ivm", global_rot, posed_verts)
        - np.einsum(
            "imn,in->im", global_rot, rest_joints,
        )[:, None, :]
        + target_joints[:, None, :]
    )
    vertices = np.einsum("vj,jvm->vm", layer.weights, transformed)
    return vertices.astype(np.float32), target_joints.astype(np.float32)


# 每指 4 段(MP 槽位对):wrist→MCP, MCP→PIP, PIP→DIP, DIP→tip
_SEGMENT_PAIRS = np.asarray((
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
), dtype=np.int64)


def _segment_lengths(joints: np.ndarray) -> np.ndarray:
    """返回 (...,20) 骨段长度。"""
    return np.linalg.norm(
        joints[..., _SEGMENT_PAIRS[:, 1], :]
        - joints[..., _SEGMENT_PAIRS[:, 0], :],
        axis=-1,
    )


def _robust_segment_lengths(obs: np.ndarray) -> np.ndarray:
    """逐帧计算段长，经 MAD 剔除异常值后聚合；不平均旋转中的坐标。"""
    lengths = _segment_lengths(obs)
    if lengths.ndim == 1:
        return lengths
    median = np.median(lengths, axis=0)
    mad = np.median(np.abs(lengths - median), axis=0)
    # 四倍稳健标准差；1 mm 下限避免零 MAD 时轻微量化噪声被误删。
    limit = np.maximum(4.0 * 1.4826 * mad, 1e-3)
    inlier = np.abs(lengths - median) <= limit
    return np.asarray([
        lengths[inlier[:, i], i].mean()
        for i in range(lengths.shape[1])
    ])


def _model_segment_lengths(layer: ManoLayer, beta: np.ndarray) -> np.ndarray:
    _, joints = layer.forward(
        np.zeros(45), beta, np.zeros(3), np.zeros(3),
    )
    return _segment_lengths(joints)


def estimate_beta(layer: ManoLayer, obs: np.ndarray) -> np.ndarray:
    """从一帧或多帧 21 点稳健估计形状参数 beta (10,)。

    多帧时先逐帧计算 20 个姿态无关的骨段长度，再用 MAD 剔除各段异常值；
    不能先平均关节点坐标，否则旋转中的骨段会被系统性缩短。先用 β 基向量
    构造线性近似求初值，再对真实非线性段长做有界稳健最小二乘精修。
    """
    from scipy.optimize import least_squares, lsq_linear

    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim == 2:
        obs = obs[None]
    if obs.ndim != 3 or obs.shape[1:] != (21, 3):
        raise ValueError(f"obs 必须是 (21,3) 或 (N,21,3),实际为 {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError("obs 含 NaN 或 Inf")

    obs_len = _robust_segment_lengths(obs)
    base_len = _model_segment_lengths(layer, np.zeros(10))
    A = np.column_stack([
        _model_segment_lengths(layer, np.eye(10)[k]) - base_len
        for k in range(10)
    ])
    initial = lsq_linear(
        A, obs_len - base_len, bounds=(-3.0, 3.0),
    ).x
    result = least_squares(
        lambda beta: _model_segment_lengths(layer, beta) - obs_len,
        initial,
        bounds=(-3.0, 3.0),
        loss="soft_l1",
        f_scale=1e-3,
        max_nfev=100,
    )
    return np.asarray(result.x, dtype=np.float64)


def beta_segment_rms_mm(
    layer: ManoLayer, beta: np.ndarray, obs: np.ndarray,
) -> float:
    """估计后的模型段长与稳健观测段长 RMS，单位 mm。"""
    observed = _robust_segment_lengths(np.asarray(obs, dtype=np.float64))
    fitted = _model_segment_lengths(layer, np.asarray(beta, dtype=np.float64))
    return float(np.sqrt(np.mean((fitted - observed) ** 2)) * 1e3)
