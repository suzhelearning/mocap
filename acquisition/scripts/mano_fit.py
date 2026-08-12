#!/usr/bin/env python3
"""mano_fit.py — 纯 numpy MANO 前向层 + 21 点拟合(方案 B,可视化辅助)。

MANO 模型(/home/current/syz/mocap/assets/mano/MANO_{LEFT,RIGHT}.pkl):
  输入: theta(45 手姿态轴角) + beta(10 形状) + root_orient(3) + trans(3)
  输出: verts(778,3) 表面 + joints21(21,3)(MediaPipe 顺序,与 mano_skeleton 一致)

拟合策略(可视化用,精度优先于速度):
  - beta: 用前若干帧平均观测,固定 theta=0 优化 (root, trans, beta) 匹配 21 点
  - theta: 逐帧高斯牛顿(数值雅可比批量前向),上一帧热启动
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MANO_DIR = Path(__file__).resolve().parents[2] / "assets" / "mano"

# ── MANO ↔ MediaPipe 关节映射 ─────────────────────────────────────────
# MANO 原生 16 关节顺序(kintree 定义,非拇指在前):
#   0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
# MediaPipe 21 点(与 H5 mano_skeleton 一致):
#   0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# 每指 = 3 关节 + 1 指尖顶点(tip 从该指末端关节的蒙皮区域选取)。
# MANO 关节 j → MediaPipe 槽位:
_MANO_TO_MP = np.asarray(
    (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3),
    dtype=np.int64,
)
_MP_TIPS_JOINT = (3, 6, 9, 12, 15)   # MANO 各指链末端关节(index/middle/pinky/ring/thumb)
# 每个 MANO 链末端关节 → 对应指尖顶点的 MediaPipe 槽位:
#   mp4=thumb tip(链15), mp8=index tip(链3), mp12=middle tip(链6),
#   mp16=ring tip(链12), mp20=pinky tip(链9)
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


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """旋转矩阵转轴角，覆盖零角和接近 π 的数值退化。"""
    R = np.asarray(R, dtype=np.float64)
    cos_angle = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-7:
        return 0.5 * np.asarray([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
    if np.pi - angle < 1e-5:
        diag = np.maximum((np.diag(R) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diag)
        k = int(np.argmax(axis))
        for j in range(3):
            if j != k and axis[k] > 1e-8:
                axis[j] = (R[k, j] + R[j, k]) / (4.0 * axis[k])
        axis /= max(np.linalg.norm(axis), 1e-10)
        return axis * angle
    axis = np.asarray([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def _kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """求列向量旋转 R，使 R @ A[i] 尽量接近 B[i]。"""
    H = np.asarray(A, dtype=np.float64).T @ np.asarray(B, dtype=np.float64)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    return R


def pose_from_skeleton(
    layer: ManoLayer,
    obs: np.ndarray,
    beta: np.ndarray | None = None,
) -> np.ndarray:
    """从 MediaPipe 21 点快速构造 MANO 姿态初值。

    只用骨骼方向估计局部旋转，不做数值优化；返回与 ``fit_frame`` 相同
    的 52 维 ``(root, trans, theta, scale)``，适合实时网格预览。
    """
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape != (21, 3):
        raise ValueError(f"obs 必须是 (21,3),实际为 {obs.shape}")
    beta = np.zeros(10, dtype=np.float64) if beta is None else np.asarray(beta)
    _, rest_J = layer._shaped(beta)
    mp = _MANO_TO_MP
    # 掌平面锚点:四指 MCP(MANO 1/4/10/7 = index/middle/ring/pinky),避开拇指
    palm_mano = np.asarray((1, 4, 10, 7), dtype=np.int64)
    root_R = _kabsch_rotation(
        rest_J[palm_mano] - rest_J[0],
        obs[mp[palm_mano]] - obs[mp[0]],
    )
    global_R = np.tile(np.eye(3), (16, 1, 1))
    global_R[0] = root_R
    theta = np.zeros(45, dtype=np.float64)
    ratios: list[float] = []
    for j in range(1, 16):
        parent = layer.parents[j]
        children = np.flatnonzero(np.asarray(layer.parents) == j)
        if children.size:
            child = int(children[0])
            rest_dir = rest_J[child] - rest_J[j]
            obs_dir = obs[mp[child]] - obs[mp[j]]
            local_rest = global_R[parent].T @ rest_dir
            local_obs = global_R[parent].T @ obs_dir
            local_R = _rotation_between(local_rest, local_obs)
            global_R[j] = global_R[parent] @ local_R
            theta[(j - 1) * 3:j * 3] = _matrix_to_axis_angle(local_R)
        else:
            global_R[j] = global_R[parent]
        p_len = np.linalg.norm(rest_J[j] - rest_J[parent])
        o_len = np.linalg.norm(obs[mp[j]] - obs[mp[parent]])
        if p_len > 1e-6 and o_len > 1e-6:
            ratios.append(float(o_len / p_len))
    x = np.zeros(52, dtype=np.float64)
    x[:3] = _matrix_to_axis_angle(root_R)
    x[3:6] = obs[0] - rest_J[0]
    x[6:51] = theta
    x[51] = float(np.clip(np.median(ratios) if ratios else 1.0, 0.5, 3.0))
    return x


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
        self.parents = parents.tolist()                                       # parent of joint i
        self.faces = np.asarray(d["f"], dtype=np.int64)                       # (1538,3)
        rest_J = self.J_regressor @ self.v_template                           # (16,3)
        # tips 顶点:每指末端关节蒙皮下,沿手指方向(末端关节→指尖)投影最远的顶点
        tips = []
        for j_end in _MP_TIPS_JOINT:
            parent_j = self.parents[j_end]
            direction = rest_J[j_end] - rest_J[parent_j]
            direction = direction / np.linalg.norm(direction)
            candidate = np.where(self.weights[:, j_end] > 0.05)[0]
            if candidate.size < 5:
                candidate = np.argsort(self.weights[:, j_end])[-20:]
            proj = (self.v_template[candidate] - rest_J[j_end]) @ direction
            tips.append(int(candidate[int(np.argmax(proj))]))
        self.tips = tuple(tips)
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


def _procrustes_align(
    layer: ManoLayer, obs: np.ndarray, beta: np.ndarray,
) -> np.ndarray:
    """闭式最优对齐(旋转+平移)作拟合初值:用模板 21 点(θ=0)对齐观测。"""
    _, model21 = layer.forward(np.zeros(45), beta, np.zeros(3), np.zeros(3))
    A = model21 - model21[0]                      # 以 wrist 为中心
    B = obs - obs[0]
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    # 轴角
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    axis = np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1],
    ])
    n = np.linalg.norm(axis)
    if n > 1e-8:
        axis = axis / n * angle
    x = np.zeros(52)
    x[:3] = axis
    x[3:6] = obs[0] - model21[0]                  # wrist 是旋转中心,trans 直接对齐
    # 全局缩放初值:观测/模型 中指长比例(以 wrist 为原点)
    obs_len = np.linalg.norm(obs[12] - obs[0])
    model_len = np.linalg.norm(model21[12] - model21[0])
    x[51] = obs_len / model_len if model_len > 1e-6 else 1.0
    return x


def fit_frame(
    layer: ManoLayer,
    obs: np.ndarray,
    beta: np.ndarray,
    init: np.ndarray | None = None,
    iters: int = 6,
    lam: float = 1e-3,
) -> np.ndarray:
    """逐帧拟合:优化 x = (root_orient(3), trans(3), theta(45), scale(1)) 共 52 维。

    高斯牛顿 + 数值雅可比(批量前向差分);init 为上一帧结果(热启动)。
    scale 以 wrist 为原点的全局缩放,补偿模板手与真实手尺寸差(β=0 时)。
    返回 x (52,)。观测 obs (21,3) 为 MediaPipe 顺序全局坐标。
    """
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape != (21, 3):
        raise ValueError(f"obs 必须是 (21,3),实际为 {obs.shape}")
    if init is None:
        x = pose_from_skeleton(layer, obs, beta)
    else:
        x = np.asarray(init, dtype=np.float64).copy()
        if x.shape != (52,):
            raise ValueError(f"init 必须是 (52,),实际为 {x.shape}")

    # wrist 是采集链路中定义明确的根点，固定平移而不是让模型误差把腕点拉偏。
    _, shaped_J = layer._shaped(np.asarray(beta, dtype=np.float64))
    trans_fixed = obs[0] - shaped_J[0]
    x[3:6] = trans_fixed
    free = np.concatenate((np.arange(3), np.arange(6, 52)))  # root + theta + scale

    def residuals(params: np.ndarray) -> np.ndarray:
        root, theta, scale = params[:3], params[6:51], params[51]
        _, J21 = layer.forward(theta, beta, root, trans_fixed)
        J21 = J21[0] + scale * (J21 - J21[0])     # 以 wrist 为原点缩放
        return (J21 - obs).ravel()

    for _ in range(iters):
        r = residuals(x)
        # 数值雅可比:每行只扰动一个自由参数,避免批量扰动串列。
        eps_rot = 1e-3
        step = np.full(free.size, 1e-5)
        step[:3] = eps_rot
        step[-1] = 1e-4
        x_batch = np.repeat(x[None], free.size + 1, axis=0)
        x_batch[1:, free] += np.diag(step)
        roots = x_batch[:, :3]
        thetas = x_batch[:, 6:51]
        scales = x_batch[:, 51:]
        trans_batch = np.repeat(trans_fixed[None], free.size + 1, axis=0)
        _, J21b = layer.forward(thetas, beta, roots, trans_batch)
        J21b = J21b[:, 0:1] + scales[:, None] * (J21b - J21b[:, 0:1])
        J = ((J21b[1:] - J21b[:1]).reshape(free.size, -1)
             / step[:, None]).T
        # 高斯牛顿: Δ = -(JᵀJ + λI)⁻¹ Jᵀr。
        try:
            A = J.T @ J + lam * np.eye(free.size)
            delta_free = -np.linalg.solve(A, J.T @ r)
        except np.linalg.LinAlgError:
            break
        delta = np.zeros(52)
        delta[free] = delta_free
        # 线搜索:只接受确实降低残差的更新。
        alpha = 1.0
        while alpha > 1e-4:
            xn = x + alpha * delta
            if np.linalg.norm(residuals(xn)) < np.linalg.norm(r):
                break
            alpha *= 0.5
        if alpha <= 1e-4:
            break
        x = xn
        x[3:6] = trans_fixed
    return x


def fit_mesh(
    layer: ManoLayer,
    obs: np.ndarray,
    beta: np.ndarray | None = None,
    init: np.ndarray | None = None,
    iters: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """拟合一帧并返回 ``(vertices, joints21, params)``。

    ``params[51]`` 是以腕点为中心的全局尺寸补偿；顶点和 21 点均已应用
    该补偿，且输出腕点与观测 ``obs[0]`` 重合。
    """
    beta_arr = np.zeros(10, dtype=np.float64) if beta is None else np.asarray(beta)
    params = fit_frame(layer, obs, beta_arr, init=init, iters=iters)
    verts, joints = layer.forward(
        params[6:51], beta_arr, params[:3], params[3:6],
    )
    scale = params[51]
    wrist = joints[0]
    verts = wrist + scale * (verts - wrist)
    joints = wrist + scale * (joints - wrist)
    return verts.astype(np.float32), joints.astype(np.float32), params
