#!/usr/bin/env python3
"""calibrate_wrist_offset.py — 交互式手腕刚体标定。

用多个静态手型求刚体局部系到 Manus 手套系的固定变换:

    p_ref = p_b + R_b · (o + R_gb · v_ref)

其中 o 是刚体原点到手掌 root 的偏移，R_gb 是手套系到刚体系的旋转，
v_ref 是 Manus 中指 DIP 相对手掌 root 的向量。求解采用刚体配准
(Kabsch)，并在写配置前强制检查：

- 至少三种非共线的中指姿态，避免只用“伸直/弯曲”两点造成旋转欠约束；
- 位置残差、方向残差和留一交叉验证必须全部达标；
- 任一质量门失败时保留旧配置，不写 offset/<user>.yaml 或 config.yaml。

交互流程：摆姿势 → Enter → 静止采集 → 切换手型 → 求解 → 质量门 → 写入。

用法:
    pixi run python scripts/calibrate_wrist_offset.py --side left --auto --apply
    pixi run python scripts/calibrate_wrist_offset.py --side right \
        --ref-rigid-id 5 --node 9 --poses 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import ruamel.yaml
import zenoh

# 每姿势采集期间的判定:位置/姿态/骨架漂移上限
STATIC_POS_MM = 2.0
STATIC_ANG_DEG = 1.5
STATIC_REF_MM = 2.0

# 自动模式:运动判定阈值(相邻帧)
MOVE_POS_M = 0.004          # 位移 >4mm/帧 视为移动
MOVE_ANG_DEG = 1.5          # 姿态变化 >1.5°/帧 视为移动

# 标定质量门默认阈值
MAX_RMS_MM = 2.0
MAX_DIRECTION_DEG = 10.0
MAX_LOO_POSITION_MM = 10.0
MIN_VECTOR_SPREAD_DEG = 25.0
MIN_SECOND_SPAN_MM = 3.0

POSE_HINTS = [
    "掌心朝下,中指沿手掌完全伸直",
    "掌心朝上,中指半弯约 45°",
    "掌心朝左,中指深弯至近似垂直掌心",
    "掌心朝右,中指沿手掌完全伸直",
    "掌心朝自己,中指半弯约 45°",
]


def build_session(endpoint: str) -> zenoh.Session:
    config = zenoh.Config.from_json5(json.dumps({
        "mode": "client",
        "connect": {"endpoints": [endpoint]},
    }))
    return zenoh.open(config)


class LiveStreams:
    """缓存最新 mocap 帧与 manus 骨架(zenoh 回调线程写入)。"""

    def __init__(self, endpoint: str, side: str, ref_rigid_id: int):
        self._session = build_session(endpoint)
        self._lock = threading.Lock()
        self._mocap: dict | None = None
        self._nodes: list | None = None
        self._rigid_body_ids: dict[str, int] = {}    # 名字 → id(来自名字表)
        self.ref_rigid_id = ref_rigid_id
        raw_key = f"manus/raw_skeleton/{side}_hand"
        self._subs = [
            self._session.declare_subscriber("mocap/hands/frame", self._on_mocap),
            self._session.declare_subscriber(raw_key, self._on_manus),
            self._session.declare_subscriber(
                "mocap/rigid_body_names", self._on_names),
        ]
        self.side = side
        self.back_id: int | None = None

    def _on_names(self, sample: object) -> None:
        try:
            msg = json.loads(sample.payload.to_string())
        except Exception:
            return
        names = msg.get("names")
        if not isinstance(names, dict):
            return
        mapping: dict[str, int] = {}
        for rid_s, name in names.items():
            try:
                mapping[str(name)] = int(rid_s)
            except (ValueError, TypeError):
                continue
        with self._lock:
            self._rigid_body_ids = mapping

    def rigid_body_id(self, name: str) -> int | None:
        with self._lock:
            return self._rigid_body_ids.get(name)

    def _on_mocap(self, sample: object) -> None:
        try:
            frame = json.loads(sample.payload.to_string())
        except Exception:
            return
        with self._lock:
            self._mocap = frame

    def _on_manus(self, sample: object) -> None:
        try:
            msg = json.loads(sample.payload.to_string())
        except Exception:
            return
        nodes = msg.get("nodes")
        if isinstance(nodes, list) and len(nodes) >= 25:
            with self._lock:
                self._nodes = nodes

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """返回 (back_pos, back_quat_xyzw, nodes, ref_pos) 或 None。"""
        with self._lock:
            if self._mocap is None or self._nodes is None:
                return None
            mocap, nodes = self._mocap, list(self._nodes)
        back = None
        ref = None
        for rb in mocap.get("rigid_bodies", []):
            if rb.get("id") == self.back_id:
                back = rb
            elif rb.get("id") == self.ref_rigid_id:
                ref = rb
        if back is None or ref is None:
            return None
        return (np.asarray(back["position"], dtype=float),
                np.asarray(back["quaternion_xyzw"], dtype=float),
                np.asarray(nodes, dtype=float),
                np.asarray(ref["position"], dtype=float))

    def close(self) -> None:
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._session.close()


def collect_pose(streams: LiveStreams, hold_s: float, ref_node: int) -> list:
    """采集 hold_s 秒的帧序列:(p_b, q_b, nodes, p_ref),参照刚体缺失的帧跳过。"""
    frames = []
    deadline = time.time() + hold_s
    while time.time() < deadline:
        snap = streams.snapshot()
        if snap is not None:
            frames.append(snap)
        time.sleep(0.01)
    return frames


def static_check(frames) -> tuple[bool, float, float, float]:
    """静止判定:位置/姿态/骨架漂移。返回 (ok, pos_mm, ang_deg, ref_mm)。"""
    if len(frames) < 5:
        return False, math.inf, math.inf, math.inf
    poss = np.asarray([f[0] for f in frames])
    quats = np.asarray([f[1] for f in frames])
    nodes = np.asarray([f[2] for f in frames])
    pos_mm = float(np.linalg.norm(poss.std(axis=0))) * 1000.0
    # 姿态漂移:四元数两两夹角(度)
    q0 = quats[0] / np.linalg.norm(quats[0])
    ang_max = 0.0
    for q in quats[1:]:
        q = q / np.linalg.norm(q)
        dot = abs(float(np.dot(q0, q)))
        ang_max = max(ang_max, math.degrees(math.acos(min(1.0, dot))) * 2.0)
    # 骨架漂移(节点 1,掌骨起点)
    ref_mm = float(np.linalg.norm(nodes[:, 1, :].std(axis=0))) * 1000.0
    ok = (pos_mm < STATIC_POS_MM and ang_max < STATIC_ANG_DEG
          and ref_mm < STATIC_REF_MM)
    return ok, pos_mm, ang_max, ref_mm


def pose_average(frames, ref_node: int, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """姿势样本:窗口平均的 (p_b, q_b_xyzw, v_ref, p_ref)。

    v_ref = A·(参照节点 − 手掌 root):先做骨架系→Motive 轴约定变换,
    使标定解出的 R_gb 只含贴装残余旋转(与拼接公式 g = p_w + R_w·A·d 一致,
    避免 A 被 R_gb 吸收后双重旋转)。
    """
    poss = np.asarray([f[0] for f in frames]).mean(axis=0)
    mid = frames[len(frames) // 2][1]                 # 四元数取中间帧
    nodes = np.asarray([f[2] for f in frames]).mean(axis=0)
    v_ref = axis @ (nodes[ref_node] - nodes[0])       # 轴变换后的参照向量
    p_ref = np.asarray([f[3] for f in frames]).mean(axis=0)
    return poss, mid, v_ref, p_ref


def euler_to_rotmat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ZYX 欧拉角(与 config 的 yaw/pitch/roll 定义一致)→ 旋转矩阵。"""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """四元数(xyzw)→ 旋转矩阵。"""
    q = np.asarray(q_xyzw, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def residuals(theta: np.ndarray, samples) -> np.ndarray:
    """位置残差；theta=[o(3), yaw, pitch, roll]。"""
    o = theta[:3]
    rgb = euler_to_rotmat(theta[3], theta[4], theta[5])
    rows = []
    for p_b, q_b, v_ref, p_ref in samples:
        rb = quat_xyzw_to_rotmat(q_b)
        rows.append(p_ref - (p_b + rb @ (o + rgb @ v_ref)))
    return np.concatenate(rows)


def _rotation_to_euler_zyx(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵转 ZYX yaw/pitch/roll（度）。"""
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
    else:
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
        roll = 0.0
    return np.degrees([yaw, pitch, roll])


def _local_targets(samples) -> tuple[np.ndarray, np.ndarray]:
    """返回刚体系中的 Manus 参照向量 X 与 Motive 实测端点 Y。"""
    x = np.asarray([sample[2] for sample in samples], dtype=float)
    y = np.asarray([
        quat_xyzw_to_rotmat(q_b).T @ (p_ref - p_b)
        for p_b, q_b, _v_ref, p_ref in samples
    ], dtype=float)
    return x, y


def pose_diversity(samples) -> dict[str, float]:
    """量化参照向量覆盖范围；刚体配准至少需要三个非共线端点。"""
    x, _ = _local_targets(samples)
    if len(x) < 3:
        return {"spread_deg": 0.0, "second_span_mm": 0.0}
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    dots = np.clip(unit @ unit.T, -1.0, 1.0)
    spread = float(np.degrees(np.arccos(dots)).max())
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    second = float(singular[1] * 1000.0) if singular.size > 1 else 0.0
    return {"spread_deg": spread, "second_span_mm": second}


def _solve_rigid(samples) -> tuple[np.ndarray, float]:
    """Kabsch 刚体配准，返回 [o,yaw,pitch,roll] 与坐标 RMS。"""
    x, y = _local_targets(samples)
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    u, _singular, vt = np.linalg.svd(xc.T @ yc)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    offset = y.mean(axis=0) - rotation @ x.mean(axis=0)
    theta = np.concatenate([offset, _rotation_to_euler_zyx(rotation)])
    rms = float(np.sqrt(np.mean(residuals(theta, samples) ** 2)))
    return theta, rms


def calibration_quality(theta: np.ndarray, samples) -> dict[str, float]:
    """计算正确扣除 wrist offset 后的方向误差与留一预测误差。"""
    offset = theta[:3]
    rgb = euler_to_rotmat(theta[3], theta[4], theta[5])
    angles = []
    for p_b, q_b, v_ref, p_ref in samples:
        observed = quat_xyzw_to_rotmat(q_b).T @ (p_ref - p_b) - offset
        predicted = rgb @ v_ref
        denom = np.linalg.norm(observed) * np.linalg.norm(predicted)
        angle = math.degrees(math.acos(float(np.clip(
            np.dot(observed, predicted) / max(denom, 1e-12), -1.0, 1.0))))
        angles.append(angle)

    loo_pos = []
    loo_dir = []
    loo_valid = True
    if len(samples) >= 4:
        for index, held_out in enumerate(samples):
            train = samples[:index] + samples[index + 1:]
            diversity = pose_diversity(train)
            if diversity["second_span_mm"] < 1e-3:
                loo_valid = False
                break
            candidate, _ = _solve_rigid(train)
            pos_err = residuals(candidate, [held_out]).reshape(3)
            loo_pos.append(float(np.linalg.norm(pos_err) * 1000.0))
            metric = calibration_quality(candidate, [held_out])
            loo_dir.append(metric["direction_rms_deg"])

    return {
        "direction_rms_deg": float(np.sqrt(np.mean(np.square(angles)))),
        "direction_max_deg": float(max(angles)),
        "loo_position_max_mm": float(max(loo_pos)) if loo_pos else 0.0,
        "loo_direction_max_deg": float(max(loo_dir)) if loo_dir else 0.0,
        "loo_valid": float(loo_valid),
    }


def quality_errors(
    samples,
    theta: np.ndarray,
    rms: float,
    *,
    max_rms_mm: float = MAX_RMS_MM,
    max_direction_deg: float = MAX_DIRECTION_DEG,
) -> tuple[list[str], dict[str, float]]:
    """返回阻止写配置的质量错误及完整指标。"""
    diversity = pose_diversity(samples)
    metrics = {**diversity, **calibration_quality(theta, samples),
               "rms_mm": rms * 1000.0}
    errors = []
    if diversity["spread_deg"] < MIN_VECTOR_SPREAD_DEG:
        errors.append(
            f"手型方向覆盖仅 {diversity['spread_deg']:.1f}°"
            f"(<{MIN_VECTOR_SPREAD_DEG:.0f}°)")
    if diversity["second_span_mm"] < MIN_SECOND_SPAN_MM:
        errors.append(
            f"参照端点近共线(second span {diversity['second_span_mm']:.1f}mm"
            f"<{MIN_SECOND_SPAN_MM:.0f}mm)，至少需要直/半弯/深弯三种手型")
    if metrics["rms_mm"] > max_rms_mm:
        errors.append(f"位置 RMS {metrics['rms_mm']:.2f}mm>{max_rms_mm:.2f}mm")
    if metrics["direction_max_deg"] > max_direction_deg:
        errors.append(
            f"方向最大误差 {metrics['direction_max_deg']:.1f}°"
            f">{max_direction_deg:.1f}°")
    if not bool(metrics["loo_valid"]):
        errors.append("留一交叉验证退化，姿势仍不足以约束旋转")
    elif metrics["loo_position_max_mm"] > MAX_LOO_POSITION_MM:
        errors.append(
            f"留一位置误差 {metrics['loo_position_max_mm']:.1f}mm"
            f">{MAX_LOO_POSITION_MM:.1f}mm")
    elif metrics["loo_direction_max_deg"] > max_direction_deg:
        errors.append(
            f"留一方向误差 {metrics['loo_direction_max_deg']:.1f}°"
            f">{max_direction_deg:.1f}°")
    return errors, metrics


def is_moving(prev, curr) -> bool:
    """相邻帧运动判定:位移或姿态角速度超阈值。"""
    if prev is None:
        return False
    d = float(np.linalg.norm(curr[0] - prev[0]))
    q0 = prev[1] / np.linalg.norm(prev[1])
    q1 = curr[1] / np.linalg.norm(curr[1])
    ang = 2.0 * math.degrees(math.acos(min(1.0, abs(float(np.dot(q0, q1))))))
    return d > MOVE_POS_M or ang > MOVE_ANG_DEG


class AutoSegmenter:
    """自动姿势分段状态机:静止确认 → 采集 → 移动切换下一姿势。"""

    def __init__(self, poses: int, hold_s: float, settle_s: float, ref_node: int,
                 axis: np.ndarray):
        self.target = poses
        self.hold_s = hold_s
        self.settle_s = settle_s
        self.ref_node = ref_node
        self.axis = axis
        self.samples = []                     # 完成的姿势样本
        self._state = "IDLE"                  # IDLE / SETTLING / SAMPLING
        self._settle_start: float | None = None
        self._seg_start: float | None = None
        self._seg: list = []

    def feed(self, snap, t: float) -> bool:
        """喂一帧 (p_b, q_b, nodes, p_ref);返回 True 表示有新姿势完成。"""
        done = False
        if self._state == "IDLE":
            if self._settle_start is None:
                self._settle_start = t
            elif t - self._settle_start >= self.settle_s:
                self._state = "SAMPLING"
                self._seg_start = t
                self._seg = [snap]
                print(f"[标定] 姿势 {len(self.samples) + 1}/{self.target} 静止确认,"
                      f"采集 {self.hold_s:.0f}s...", flush=True)
        elif self._state == "SAMPLING":
            self._seg.append(snap)
            if t - self._seg_start >= self.hold_s:
                ok, pos_mm, ang_deg, ref_mm = static_check(self._seg)
                if not ok:
                    print(f"[标定]   静止质量不足(位移 {pos_mm:.1f}mm/姿态 {ang_deg:.1f}°),"
                          "请重新摆好并保持", file=sys.stderr)
                else:
                    p_b, q_b, v_ref, p_ref = pose_average(self._seg, self.ref_node,
                                                          self.axis)
                    self.samples.append((p_b, q_b, v_ref, p_ref))
                    print(f"[标定]   姿势 {len(self.samples)}/{self.target} 完成"
                          f"(位移 {pos_mm:.1f}mm,姿态 {ang_deg:.1f}°)。"
                          "请切换下一个姿势", flush=True)
                    done = len(self.samples) >= self.target
                self._state = "IDLE"
                self._settle_start = None
                self._seg = []
        return done

    def abort_if_moving(self, moving: bool) -> None:
        """检测到移动:取消当前静置/采集进度。"""
        if not moving:
            return
        if self._state == "SETTLING":
            self._state = "IDLE"
            self._settle_start = None
        elif self._state == "SAMPLING":
            print("[标定]   检测到移动,当前姿势作废,请重新摆好并保持",
                  file=sys.stderr)
            self._state = "IDLE"
            self._settle_start = None
            self._seg = []


def run_auto(streams: LiveStreams, args, back_id: int, axis: np.ndarray) -> list:
    """分段标定:每姿势按 Enter 确认开始,段内自动静止检测与重采。"""
    print("[标定] 分段模式:每个姿势摆好后按 Enter 开始采集;"
          f"采集中手动了会自动重采。共 {args.poses} 个姿势")
    print("[标定] 关键:相邻姿势间切换「中指伸直/弯曲」手型(采集时保持不动),"
          "配合手腕朝向变化,才能分离位置与旋转")
    samples = []
    for i in range(args.poses):
        hint = POSE_HINTS[i] if i < len(POSE_HINTS) else "任意与前几个明显不同的朝向"
        input(f"[标定] 姿势 {i + 1}/{args.poses}:{hint}。摆好后按 Enter 开始")
        while True:
            print(f"[标定]   采集 {args.hold:.0f}s,请保持静止...", flush=True)
            frames = collect_pose(streams, args.hold, args.node)
            ok, pos_mm, ang_deg, ref_mm = static_check(frames)
            if not ok:
                print(f"[标定]   检测到手部移动(位移 {pos_mm:.1f}mm / "
                      f"姿态 {ang_deg:.1f}°),自动重采...", file=sys.stderr)
                continue
            p_b, q_b, v_ref, p_ref = pose_average(frames, args.node, axis)
            print(f"[标定]   姿势 {i + 1}/{args.poses} 完成"
                  f"(位移 {pos_mm:.1f}mm,姿态 {ang_deg:.1f}°)。请切换下一个姿势",
                  flush=True)
            samples.append((p_b, q_b, v_ref, p_ref))
            break
    return samples


def solve(samples) -> tuple[np.ndarray, float]:
    """刚体配准求解 [o(3), yaw, pitch, roll]；质量门由 quality_errors 执行。"""
    return _solve_rigid(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description="交互式 wrist offset 标定(方案 C)")
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--config", default="config.yaml", help="运行时配置(读 back 刚体 id)")
    ap.add_argument("--router", default="tcp/127.0.0.1:7447")
    ap.add_argument("--ref-rigid-id", type=int, default=None,
                    help="参照点刚体 id(与 --ref-name 二选一;默认优先 --ref-name)")
    ap.add_argument("--ref-name", default="left_dip",
                    help="参照点刚体名字(如 left_dip;从 mocap/rigid_body_names 解析 id)")
    ap.add_argument("--back-name", default=None,
                    help="手背刚体名字(如 left_wrist;缺省用 config 的 back_rigid_id)")
    ap.add_argument("--node", type=int, default=9,
                    help="参照骨架节点索引(默认 9 = 中指 DIP)")
    ap.add_argument("--poses", type=int, default=5, help="标定姿势数(推荐 4~6)")
    ap.add_argument("--hold", type=float, default=3.0, help="每姿势采集秒数")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="自动模式:静止确认时长(秒)")
    ap.add_argument("--auto", action="store_true",
                    help="自动模式:检测静止段自动采集,无需按 Enter")
    ap.add_argument("--user", default="default",
                    help="操作者名字:结果写入 offset/<user>.yaml(按人区分标定)")
    ap.add_argument("--apply", action="store_true",
                    help="同时写回 config.yaml 的 hands.<side>.wrist_offset"
                         "(默认只写 offset/<user>.yaml)")
    ap.add_argument("--max-rms-mm", type=float, default=MAX_RMS_MM,
                    help=f"写配置允许的位置 RMS 上限(默认 {MAX_RMS_MM:g}mm)")
    ap.add_argument("--max-direction-deg", type=float, default=MAX_DIRECTION_DEG,
                    help=f"写配置允许的方向误差上限(默认 {MAX_DIRECTION_DEG:g}°)")
    args = ap.parse_args()

    global args_node
    args_node = args.node

    # 读 config:back 刚体 id 与轴变换 A(骨架系→Motive 系)
    try:
        import yaml as _pyyaml
        cfg_raw = _pyyaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        back_id_cfg = cfg_raw["hands"][args.side]["back_rigid_id"]
        ax = cfg_raw.get("axis_transform", {})
        perm = ax.get("permutation", [0, 2, 1])
        signs = ax.get("signs", [1, 1, -1])
        A = np.zeros((3, 3), dtype=float)
        for j in range(3):
            A[j, perm[j]] = signs[j]
    except Exception as exc:
        print(f"[标定] 读取 {args.config} 失败: {exc}", file=sys.stderr)
        return 2

    print(f"[标定] 侧={args.side}  back刚体=config#{back_id_cfg}"
          f"  参照刚体=名字[{args.ref_name}] 参照节点索引={args.node}"
          f"  姿势数={args.poses}")
    print("[标定] 连接数据流,请确认手套与 Motive 均在发布数据...")

    streams = LiveStreams(args.router, args.side, 0)   # ref id 稍后按名字解析
    try:
        # 等待数据 + 刚体名字表就绪
        deadline = time.time() + 15
        ref_id: int | None = None
        back_id: int | None = None
        while time.time() < deadline:
            with streams._lock:
                names_ready = bool(streams._rigid_body_ids)
            if args.ref_name:
                ref_id = streams.rigid_body_id(args.ref_name)
            elif args.ref_rigid_id is not None:
                ref_id = args.ref_rigid_id
            if args.back_name:
                back_id = streams.rigid_body_id(args.back_name)
            else:
                back_id = back_id_cfg
            if names_ready and ref_id is not None and back_id is not None:
                break
            time.sleep(0.2)
        if ref_id is None or back_id is None:
            print(f"[标定] 无法解析刚体: 参照[{args.ref_name}]→{ref_id}, "
                  f"back[{args.back_name or 'config#' + str(back_id_cfg)}]→{back_id}",
                  file=sys.stderr)
            print("[标定] 提示:请确认 Windows publisher 已更新并发布"
                  " mocap/rigid_body_names(刚体名字表)", file=sys.stderr)
            return 1
        streams.ref_rigid_id = ref_id
        streams.back_id = back_id
        print(f"[标定] 刚体解析: back={args.back_name or back_id_cfg}→id {back_id}, "
              f"参照={args.ref_name}→id {ref_id}")

        if streams.snapshot() is None:
            print("[标定] 未收到帧数据(检查 router/手套/动捕),退出",
                  file=sys.stderr)
            return 1

        if args.auto:
            samples = run_auto(streams, args, back_id, A)
        else:
            samples = []
            for i in range(args.poses):
                hint = POSE_HINTS[i] if i < len(POSE_HINTS) else "任意与前几个明显不同的朝向"
                input(f"[标定] 姿势 {i + 1}/{args.poses}:{hint}。摆好后按 Enter 开始采集")
                while True:
                    print(f"[标定]   采集 {args.hold:.0f}s,请保持静止...", flush=True)
                    frames = collect_pose(streams, args.hold, args.node)
                    ok, pos_mm, ang_deg, ref_mm = static_check(frames)
                    if not ok:
                        print(f"[标定]   检测到手部移动(位移 {pos_mm:.1f}mm / 姿态 {ang_deg:.1f}°"
                              f" / 骨架 {ref_mm:.1f}mm),请重新摆好姿势按 Enter 重采",
                              file=sys.stderr)
                        input("   重新摆好后按 Enter")
                        continue
                    p_b, q_b, v_ref, p_ref = pose_average(frames, args.node, A)
                    print(f"[标定]   姿势 {i + 1} 采集完成"
                          f"(位移 {pos_mm:.1f}mm,姿态 {ang_deg:.1f}°)")
                    samples.append((p_b, q_b, v_ref, p_ref))
                    break

        if len(samples) < 3:
            print("[标定] 有效姿势不足 3 个，无法约束刚体旋转", file=sys.stderr)
            return 1
        diversity = pose_diversity(samples)
        if (diversity["spread_deg"] < MIN_VECTOR_SPREAD_DEG
                or diversity["second_span_mm"] < MIN_SECOND_SPAN_MM):
            print("[标定] 质量门拒绝:姿势多样性不足，旧配置保持不变", file=sys.stderr)
            print(f"[标定] 方向覆盖 {diversity['spread_deg']:.1f}°，"
                  f"second span {diversity['second_span_mm']:.1f}mm", file=sys.stderr)
            print("[标定] 请使用中指直伸、半弯、深弯三种非共线手型重新采集",
                  file=sys.stderr)
            return 1
        theta, rms = solve(samples)
        o = theta[:3]
        result = {
            "mode": "body",
            "xyz": [round(float(v), 4) for v in o],
            "yaw_deg": round(float(theta[3]), 2),
            "pitch_deg": round(float(theta[4]), 2),
            "roll_deg": round(float(theta[5]), 2),
        }
        errors, metrics = quality_errors(
            samples, theta, rms,
            max_rms_mm=args.max_rms_mm,
            max_direction_deg=args.max_direction_deg,
        )
        print("\n[标定] ===== 求解完成 =====")
        print(f"[标定] 位置 RMS: {metrics['rms_mm']:.2f} mm"
              f"(门限 {args.max_rms_mm:.2f}mm)")
        print(f"[标定] 方向 RMS/最大: {metrics['direction_rms_deg']:.1f}°/"
              f"{metrics['direction_max_deg']:.1f}°")
        print(f"[标定] 姿势覆盖: {metrics['spread_deg']:.1f}°，"
              f"second span {metrics['second_span_mm']:.1f}mm")
        print(f"[标定] 留一最大误差: {metrics['loo_position_max_mm']:.1f}mm / "
              f"{metrics['loo_direction_max_deg']:.1f}°")
        print(f"[标定] wrist_offset.xyz = {result['xyz']}")
        print(f"[标定] yaw={result['yaw_deg']}°  pitch={result['pitch_deg']}°"
              f"  roll={result['roll_deg']}°")
        if errors:
            print("[标定] ===== 质量门拒绝，未写入任何配置 =====", file=sys.stderr)
            for error in errors:
                print(f"[标定]   - {error}", file=sys.stderr)
            return 1

        # 写入按用户区分的 offset 文件:acquisition/offset/<user>.yaml
        offset_dir = Path(args.config).resolve().parent / "offset"
        offset_dir.mkdir(parents=True, exist_ok=True)
        offset_path = offset_dir / f"{args.user}.yaml"
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        if offset_path.is_file():
            offset_cfg = yaml.load(offset_path.read_text(encoding="utf-8")) or {}
        else:
            offset_cfg = {}
        side_cfg = offset_cfg.setdefault(args.side, {})
        side_cfg.clear()
        side_cfg.update(result)
        yaml.dump(offset_cfg, offset_path)
        print(f"[标定] 已写入 {offset_path} (用户 {args.user!r}, {args.side})")

        if args.apply:
            # 兼容旧行为:同时写回 config.yaml
            cfg = yaml.load(Path(args.config).read_text(encoding="utf-8"))
            off = cfg["hands"][args.side]["wrist_offset"]
            off["mode"] = "body"
            off["xyz"] = list(result["xyz"])
            off["yaw_deg"] = result["yaw_deg"]
            off["pitch_deg"] = result["pitch_deg"]
            off["roll_deg"] = result["roll_deg"]
            yaml.dump(cfg, Path(args.config))
            print(f"[标定] 已同时写回 {args.config}")
        return 0
    finally:
        streams.close()


if __name__ == "__main__":
    sys.exit(main())
