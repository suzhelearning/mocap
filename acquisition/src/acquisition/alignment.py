"""中央 60 Hz 物理时间轴：把 Motive、左 Manus、右 Manus 插值为同帧数据。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import enum
import math
import threading
from typing import Generic, TypeVar

import numpy as np

from .config import Config
from .kinematics import quat_slerp, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, rotmat_from_wxyz
from .manus_schema import NODE_COUNT, manus_to_mediapipe
from .object_offset import ObjectOffset, transform_object_poses
from .stitching import hand_nodes_to_global, wrist_pose_from_back


class InvalidReason(enum.IntFlag):
    """统一帧无效原因；多个原因可按位组合。"""

    NONE = 0
    MOCAP_MISSING = 1 << 0
    MOCAP_GAP = 1 << 1
    OBJECT_MISSING = 1 << 2
    TRACKING_INVALID = 1 << 3
    LEFT_MISSING = 1 << 4
    LEFT_GAP = 1 << 5
    RIGHT_MISSING = 1 << 6
    RIGHT_GAP = 1 << 7


@dataclass(frozen=True)
class InterpolationInfo:
    source_seq_before: int
    source_seq_after: int
    source_gap_ns: int
    interpolation_alpha: float


@dataclass(frozen=True)
class AlignedObject:
    rigid_position: np.ndarray
    rigid_quaternion_xyzw: np.ndarray
    object_position: np.ndarray
    object_quaternion_xyzw: np.ndarray
    tracking_valid: bool
    mean_error: float
    valid: bool


@dataclass(frozen=True)
class AlignedHand:
    nodes_local: np.ndarray
    node_quaternions_wxyz: np.ndarray
    nodes_world: np.ndarray
    mano_skeleton: np.ndarray
    wrist_position: np.ndarray
    wrist_quaternion_xyzw: np.ndarray
    valid: bool
    interpolation: InterpolationInfo


@dataclass(frozen=True)
class AlignedFrame:
    frame_index: int
    t_phys_ns: int
    t_emit_ns: int
    frame_valid: bool
    reason_flags: int
    mocap_frame: dict | None
    mocap_interpolation: InterpolationInfo
    objects: dict[str, AlignedObject]
    hands: dict[str, AlignedHand]
    interaction: dict[str, dict[str, np.ndarray]]


_T = TypeVar("_T")


class _TimeRing(Generic[_T]):
    """严格单调的小型时间环；只保留对齐窗口所需历史。"""

    def __init__(self, history_ns: int) -> None:
        self._history_ns = int(history_ns)
        self._items: deque[tuple[int, int, _T]] = deque()
        self.nonmonotonic = 0

    def append(self, t_ns: int, sequence: int, value: _T) -> bool:
        if self._items and t_ns <= self._items[-1][0]:
            self.nonmonotonic += 1
            return False
        self._items.append((int(t_ns), int(sequence), value))
        cutoff = t_ns - self._history_ns
        while len(self._items) > 2 and self._items[1][0] < cutoff:
            self._items.popleft()
        return True

    def first_time(self) -> int | None:
        return self._items[0][0] if self._items else None

    def bracket(
        self, t_ns: int,
    ) -> tuple[tuple[int, int, _T], tuple[int, int, _T], float] | None:
        if not self._items or t_ns < self._items[0][0] or t_ns > self._items[-1][0]:
            return None
        items = self._items
        lo, hi = 0, len(items) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if items[mid][0] < t_ns:
                lo = mid + 1
            else:
                hi = mid
        after = items[lo]
        if after[0] == t_ns or lo == 0:
            return after, after, 0.0
        before = items[lo - 1]
        alpha = (t_ns - before[0]) / (after[0] - before[0])
        return before, after, float(alpha)


_NAN3 = np.full(3, np.nan, dtype=np.float64)
_NAN4 = np.full(4, np.nan, dtype=np.float64)
_NAN_NODES = np.full((NODE_COUNT, 3), np.nan, dtype=np.float64)
_NAN_QUATS = np.full((NODE_COUNT, 4), np.nan, dtype=np.float64)
_NAN_MANO = np.full((21, 3), np.nan, dtype=np.float64)


def _interp_info(bracket: tuple[tuple[int, int, object], tuple[int, int, object], float] | None) -> InterpolationInfo:
    if bracket is None:
        return InterpolationInfo(-1, -1, -1, math.nan)
    before, after, alpha = bracket
    return InterpolationInfo(before[1], after[1], after[0] - before[0], alpha)


def _invalid_hand(info: InterpolationInfo) -> AlignedHand:
    return AlignedHand(
        _NAN_NODES.copy(), _NAN_QUATS.copy(), _NAN_NODES.copy(), _NAN_MANO.copy(),
        _NAN3.copy(), _NAN4.copy(), False, info,
    )


def _slerp_rows_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray([
        quat_slerp(a, b, alpha) for a, b in zip(q0, q1)
    ], dtype=np.float64)


def _interpolate_rigid_body(
    frame_a: dict, frame_b: dict, rigid_id: int, alpha: float,
) -> dict | None:
    by_id_a = {int(rb["id"]): rb for rb in frame_a.get("rigid_bodies", [])}
    by_id_b = {int(rb["id"]): rb for rb in frame_b.get("rigid_bodies", [])}
    a = by_id_a.get(rigid_id)
    b = by_id_b.get(rigid_id)
    if a is None or b is None:
        return None
    pa = np.asarray(a["position"], dtype=np.float64)
    pb = np.asarray(b["position"], dtype=np.float64)
    qa = quat_xyzw_to_wxyz(a["quaternion_xyzw"])
    qb = quat_xyzw_to_wxyz(b["quaternion_xyzw"])
    return {
        "id": rigid_id,
        "position": (pa + alpha * (pb - pa)).tolist(),
        "quaternion_xyzw": quat_wxyz_to_xyzw(quat_slerp(qa, qb, alpha)).tolist(),
        "mean_error": float(a.get("mean_error", math.nan)) + alpha * (
            float(b.get("mean_error", math.nan)) - float(a.get("mean_error", math.nan))
        ),
        "tracking_valid": bool(a.get("tracking_valid", True)) and bool(
            b.get("tracking_valid", True)
        ),
    }


def _frame_with_interpolated_bodies(
    frame_a: dict, frame_b: dict, alpha: float,
) -> dict:
    ids_a = {int(rb["id"]) for rb in frame_a.get("rigid_bodies", [])}
    ids_b = {int(rb["id"]) for rb in frame_b.get("rigid_bodies", [])}
    bodies = [
        body for rigid_id in sorted(ids_a & ids_b)
        if (body := _interpolate_rigid_body(frame_a, frame_b, rigid_id, alpha)) is not None
    ]
    out = dict(frame_b if alpha >= 0.5 else frame_a)
    out["rigid_bodies"] = bodies
    return out


class AlignmentEngine:
    """线程安全中央对齐器；输入回调只入环，主循环按固定节拍取帧。"""

    def __init__(
        self,
        config: Config,
        *,
        object_offsets: dict[str, ObjectOffset] | None = None,
    ) -> None:
        self._cfg = config
        self._period_num = 1_000_000_000
        self._period_den = int(round(config.alignment_hz))
        self._latency_ns = int(round(config.alignment_latency_ms * 1e6))
        self._mocap_gap_ns = int(round(config.mocap_max_gap_ms * 1e6))
        self._manus_gap_ns = int(round(config.manus_max_gap_ms * 1e6))
        history_ns = max(2_000_000_000, 4 * self._latency_ns)
        self._mocap: _TimeRing[dict] = _TimeRing(history_ns)
        self._hands = {
            "left": _TimeRing[dict](history_ns),
            "right": _TimeRing[dict](history_ns),
        }
        self._object_offsets = object_offsets or {}
        self._next_tick: int | None = None
        self._frame_index = 0
        self._lock = threading.RLock()

    @property
    def output_hz(self) -> float:
        return float(self._period_den)

    @property
    def latency_ns(self) -> int:
        return self._latency_ns

    def push_mocap(self, frame: dict) -> bool:
        with self._lock:
            return self._mocap.append(
                int(frame["t_phys_ns"]),
                int(frame.get("frame_number", -1)),
                frame,
            )

    def push_manus(self, side: str, msg: dict) -> bool:
        if side not in self._hands:
            raise ValueError(f"side 须为 left/right，实际 {side!r}")
        with self._lock:
            return self._hands[side].append(
                int(msg["t_phys_ns"]), int(msg.get("seq", -1)), msg,
            )

    def reset_timeline(self, start_ns: int | None = None) -> None:
        """开始新 take：输出序号归零，可选从指定物理时刻的首个 60Hz tick 开始。"""
        with self._lock:
            self._next_tick = (
                None if start_ns is None
                else (
                    int(start_ns) * self._period_den
                    + self._period_num - 1
                ) // self._period_num
            )
            self._frame_index = 0

    def _tick_time(self, index: int) -> int:
        return (
            index * self._period_num + self._period_den // 2
        ) // self._period_den

    def _initialize_tick(self) -> bool:
        first = self._mocap.first_time()
        if first is None:
            return False
        self._next_tick = (
            first * self._period_den + self._period_num - 1
        ) // self._period_num
        return True

    def emit_ready(self, now_ns: int) -> list[AlignedFrame]:
        """生成截至 ``now-latency`` 的全部 60 Hz 帧；无效 tick 也保留。"""
        with self._lock:
            if self._next_tick is None and not self._initialize_tick():
                return []
            assert self._next_tick is not None
            limit_ns = int(now_ns) - self._latency_ns
            frames: list[AlignedFrame] = []
            while self._tick_time(self._next_tick) <= limit_ns:
                t_ns = self._tick_time(self._next_tick)
                frames.append(self._align_one(t_ns, int(now_ns)))
                self._next_tick += 1
                self._frame_index += 1
            return frames

    def _align_one(self, t_ns: int, emit_ns: int) -> AlignedFrame:
        reasons = InvalidReason.NONE
        mocap_bracket = self._mocap.bracket(t_ns)
        mocap_info = _interp_info(mocap_bracket)
        mocap_frame: dict | None = None
        if mocap_bracket is None:
            reasons |= InvalidReason.MOCAP_MISSING
        elif mocap_info.source_gap_ns > self._mocap_gap_ns:
            reasons |= InvalidReason.MOCAP_GAP
        else:
            before, after, alpha = mocap_bracket
            mocap_frame = _frame_with_interpolated_bodies(
                before[2], after[2], alpha,
            )
            mocap_frame["t_phys_ns"] = t_ns

        objects: dict[str, AlignedObject] = {}
        for name, rigid_id in self._cfg.objects.items():
            rb = None if mocap_frame is None else next(
                (
                    item for item in mocap_frame["rigid_bodies"]
                    if item["id"] == rigid_id
                ),
                None,
            )
            if rb is None:
                reasons |= InvalidReason.OBJECT_MISSING
                objects[name] = AlignedObject(
                    _NAN3.copy(), _NAN4.copy(), _NAN3.copy(), _NAN4.copy(),
                    False, math.nan, False,
                )
                continue
            rigid_pos = np.asarray(rb["position"], dtype=np.float64)
            rigid_quat = np.asarray(
                rb["quaternion_xyzw"], dtype=np.float64,
            )
            object_pos = rigid_pos.copy()
            object_quat = rigid_quat.copy()
            offset = self._object_offsets.get(name)
            if offset is not None:
                pos_batch, quat_batch = transform_object_poses(
                    rigid_pos[None, :], rigid_quat[None, :], offset,
                )
                object_pos = pos_batch[0]
                object_quat = quat_batch[0]
            tracking_valid = bool(rb["tracking_valid"])
            if not tracking_valid:
                reasons |= InvalidReason.TRACKING_INVALID
            objects[name] = AlignedObject(
                rigid_pos,
                rigid_quat,
                object_pos,
                object_quat,
                tracking_valid,
                float(rb.get("mean_error", math.nan)),
                tracking_valid,
            )

        hands: dict[str, AlignedHand] = {}
        for side in ("left", "right"):
            missing_flag = (
                InvalidReason.LEFT_MISSING
                if side == "left" else InvalidReason.RIGHT_MISSING
            )
            gap_flag = (
                InvalidReason.LEFT_GAP
                if side == "left" else InvalidReason.RIGHT_GAP
            )
            bracket = self._hands[side].bracket(t_ns)
            info = _interp_info(bracket)
            if bracket is None:
                reasons |= missing_flag
                hands[side] = _invalid_hand(info)
                continue
            if info.source_gap_ns > self._manus_gap_ns:
                reasons |= gap_flag
                hands[side] = _invalid_hand(info)
                continue
            hand_cfg = self._cfg.hands[side]
            rigid_id = hand_cfg.wrist_rigid_id or hand_cfg.back_rigid_id
            rb = None if mocap_frame is None or rigid_id is None else next(
                (
                    item for item in mocap_frame["rigid_bodies"]
                    if item["id"] == rigid_id
                ),
                None,
            )
            if rb is None or not bool(rb["tracking_valid"]):
                reasons |= InvalidReason.TRACKING_INVALID
                hands[side] = _invalid_hand(info)
                continue
            before, after, alpha = bracket
            nodes_a = np.asarray(before[2]["nodes"], dtype=np.float64)
            nodes_b = np.asarray(after[2]["nodes"], dtype=np.float64)
            quats_a = np.asarray(
                before[2]["node_quaternions_wxyz"], dtype=np.float64,
            )
            quats_b = np.asarray(
                after[2]["node_quaternions_wxyz"], dtype=np.float64,
            )
            nodes_local = nodes_a + alpha * (nodes_b - nodes_a)
            node_quats = _slerp_rows_wxyz(quats_a, quats_b, alpha)
            if hand_cfg.wrist_rigid_id is not None:
                wrist_pos = np.asarray(rb["position"], dtype=np.float64)
                wrist_q_wxyz = quat_xyzw_to_wxyz(
                    rb["quaternion_xyzw"],
                )
            else:
                wrist_pos, wrist_q_wxyz = wrist_pose_from_back(
                    np.asarray(rb["position"], dtype=np.float64),
                    np.asarray(rb["quaternion_xyzw"], dtype=np.float64),
                    hand_cfg.wrist_offset,
                )
            nodes_world = hand_nodes_to_global(
                nodes_local,
                0,
                wrist_pos,
                wrist_q_wxyz,
                self._cfg.axis_matrix(),
            )
            hands[side] = AlignedHand(
                nodes_local,
                node_quats,
                nodes_world,
                np.asarray(
                    manus_to_mediapipe(nodes_world), dtype=np.float64,
                ),
                wrist_pos,
                quat_wxyz_to_xyzw(wrist_q_wxyz),
                True,
                info,
            )

        interaction: dict[str, dict[str, np.ndarray]] = {}
        for name, obj in objects.items():
            if obj.valid:
                rotation = rotmat_from_wxyz(
                    quat_xyzw_to_wxyz(obj.object_quaternion_xyzw),
                )
                interaction[name] = {
                    side: (
                        rotation.T
                        @ (hand.mano_skeleton - obj.object_position).T
                    ).T
                    if hand.valid else _NAN_MANO.copy()
                    for side, hand in hands.items()
                }
            else:
                interaction[name] = {
                    side: _NAN_MANO.copy() for side in hands
                }

        return AlignedFrame(
            frame_index=self._frame_index,
            t_phys_ns=t_ns,
            t_emit_ns=emit_ns,
            frame_valid=reasons == InvalidReason.NONE,
            reason_flags=int(reasons),
            mocap_frame=mocap_frame,
            mocap_interpolation=mocap_info,
            objects=objects,
            hands=hands,
            interaction=interaction,
        )
