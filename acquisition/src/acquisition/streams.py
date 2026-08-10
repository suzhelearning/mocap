"""StreamHub:单 Zenoh session 多流订阅(mocap 帧 + manus 双手骨架/拓扑)。

回调在 zenoh 库线程上执行,只做「校验 → 打 t_ubuntu_ns → 广播给订阅者」,
绝不阻塞。订阅者回调须自行入队/拷贝(录制器、可视化各自处理)。
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import zenoh

RATE_WINDOW = 2.0            # 帧率统计滑动窗口(秒),窗口满结算重置

MOCAP_HISTORY_SEC = 1.0      # mocap 帧历史保留时长(插值对齐用,120Hz 下 ~120 帧)

from natnet_zenoh.schema import FRAME_KEY, decode_frame

from .kinematics import quat_slerp

from .manus_schema import (
    MANUS_EDGE_KEYS,
    MANUS_RAW_KEYS,
    MEDIAPIPE_KEYS,
    ManusError,
    decode_mano,
    decode_manus,
    parse_edges,
    side_from_key,
)

FrameCallback = Callable[[dict], None]   # 已打 t_ubuntu_ns 的帧


def _stamp(frame: dict, t_ns: int) -> dict:
    frame["t_ubuntu_ns"] = t_ns
    return frame


class StreamHub:
    """订阅 mocap/hands/frame + manus/raw_skeleton/* + manus/skeleton_edges/*。

    每类流维护:
    - latest:最近一帧(带 t_ubuntu_ns),供状态栏/初始化
    - 订阅者回调列表(多订阅者时按注册顺序调用)
    - 帧计数与最近接收时间(状态栏帧率)
    """

    def __init__(self, router_endpoint: str) -> None:
        self._router_endpoint = router_endpoint
        self._session: zenoh.Session | None = None
        self._subscribers: list[object] = []
        self._lock = threading.Lock()

        self._latest_mocap: dict | None = None
        self._mocap_history: deque[tuple[int, dict]] = deque()   # (t_ubuntu_ns, frame)
        self._latest_manus: dict[str, dict] = {"left": None, "right": None}
        self._latest_mano: dict[str, dict] = {"left": None, "right": None}
        self._latest_edges: dict[str, list[tuple[int, int, int]]] = {}

        self._mocap_cbs: list[FrameCallback] = []
        self._manus_cbs: dict[str, list[FrameCallback]] = {"left": [], "right": []}
        self._mano_cbs: dict[str, list[FrameCallback]] = {"left": [], "right": []}

        # 帧率统计:滑动窗口计数(窗口满 RATE_WINDOW 秒结算重置)
        self._rate_counts = {"mocap": 0, "left": 0, "right": 0}
        now = time.time()
        self._rate_start = {"mocap": now, "left": now, "right": now}
        self._last_rate: dict[str, float] = {"mocap": 0.0, "left": 0.0, "right": 0.0}

    # -- 订阅注册 ---------------------------------------------------------

    def on_mocap(self, cb: FrameCallback) -> None:
        """订阅 mocap 帧(帧已校验并带 t_ubuntu_ns)。"""
        self._mocap_cbs.append(cb)

    def on_manus(self, side: str, cb: FrameCallback) -> None:
        """订阅 manus 骨架帧(side ∈ left/right)。"""
        if side not in self._manus_cbs:
            raise ValueError(f"side 须为 left/right,实际 {side!r}")
        self._manus_cbs[side].append(cb)

    def on_mano(self, side: str, cb: FrameCallback) -> None:
        """订阅 MANO/MediaPipe 21 点帧(side ∈ left/right)。"""
        if side not in self._mano_cbs:
            raise ValueError(f"side 须为 left/right,实际 {side!r}")
        self._mano_cbs[side].append(cb)

    # -- 最新帧查询 -------------------------------------------------------

    def latest_mocap(self) -> dict | None:
        return self._latest_mocap

    def mocap_at(self, t_ns: int) -> dict | None:
        """按目标时刻对 mocap 刚体流插值(位置 lerp + 四元数 slerp)。

        用于与手套帧对齐:手套帧 t_ubuntu_ns 时刻的刚体位姿 = 包围该时刻的
        两帧 mocap 插值,消除双流帧周期错位(0~8.3ms)。

        刚体按 id 配对:两帧都有 → 插值;仅一帧有 → 取该帧;
        tracking_valid 取较近一帧。t 超出历史范围 → 取最近端帧。
        无历史返回 None。
        """
        hist = self._mocap_history
        if not hist:
            return self._latest_mocap
        t0, f0 = hist[0]
        t1, f1 = hist[-1]
        if t_ns <= t0:
            return f0
        if t_ns >= t1:
            return f1
        # 找包围 t_ns 的两帧(二分)
        lo, hi = 0, len(hist) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if hist[mid][0] <= t_ns:
                lo = mid
            else:
                hi = mid
        ta, fa = hist[lo]
        tb, fb = hist[hi]
        frac = (t_ns - ta) / (tb - ta) if tb > ta else 0.0

        rbs_a = {rb["id"]: rb for rb in fa.get("rigid_bodies", [])}
        rbs_b = {rb["id"]: rb for rb in fb.get("rigid_bodies", [])}
        out: list[dict] = []
        for rid in sorted(set(rbs_a) | set(rbs_b)):
            a, b = rbs_a.get(rid), rbs_b.get(rid)
            if a is None:
                out.append(dict(b))
            elif b is None:
                out.append(dict(a))
            else:
                pa = np.asarray(a["position"], dtype=float)
                pb = np.asarray(b["position"], dtype=float)
                qa = np.asarray(a["quaternion_xyzw"], dtype=float)[[3, 0, 1, 2]]
                qb = np.asarray(b["quaternion_xyzw"], dtype=float)[[3, 0, 1, 2]]
                qm = quat_slerp(qa, qb, frac)[[1, 2, 3, 0]]     # wxyz→xyzw
                out.append({
                    "id": rid,
                    "position": ((1 - frac) * pa + frac * pb).tolist(),
                    "quaternion_xyzw": qm.tolist(),
                    "mean_error": (a["mean_error"] + b["mean_error"]) / 2.0,
                    "tracking_valid": (b["tracking_valid"] if frac >= 0.5
                                       else a["tracking_valid"]),
                })
        frame = dict(fb)
        frame["rigid_bodies"] = out
        return frame

    def latest_manus(self, side: str) -> dict | None:
        return self._latest_manus[side]

    def latest_mano(self, side: str) -> dict | None:
        return self._latest_mano[side]

    def latest_edges(self, side: str) -> list[tuple[int, int, int]] | None:
        return self._latest_edges.get(side)

    def rates_hz(self) -> dict[str, float]:
        """每路流最近帧率估计(滑动窗口,窗口满 RATE_WINDOW 秒结算重置)。

        结算后窗口从 0 重新计时,若不足 0.5s 直接返回 0.0 会导致状态栏
        周期性闪现 0(每 2s 一次,持续约 0.5s)——用上次速率填充该死区。
        """
        rates = {}
        now = time.time()
        with self._lock:
            for name in self._rate_counts:
                window = now - self._rate_start[name]
                if window >= RATE_WINDOW:
                    rate = self._rate_counts[name] / window
                    self._rate_counts[name] = 0
                    self._rate_start[name] = now
                elif window > 0.5:          # 窗口太短时避免抖动
                    rate = self._rate_counts[name] / window
                else:
                    rate = self._last_rate[name]   # 死区:沿用上次速率
                self._last_rate[name] = rate
                rates[name] = rate
        return rates

    # -- 生命周期 ---------------------------------------------------------

    def start(self) -> None:
        """打开 session(connect router)并声明全部订阅。

        用 client 模式而非 peer 模式:本机 zenohd(v005d518-modified)对
        「关闭 scouting 的 peer → router」连接不路由数据(client 模式正常)。
        """
        with self._lock:
            if self._session is not None:
                raise RuntimeError("StreamHub 已启动")
            config = zenoh.Config.from_json5(json.dumps({
                "mode": "client",
                "connect": {"endpoints": [self._router_endpoint]},
            }))
            self._session = zenoh.open(config)
            self._subscribers.append(
                self._session.declare_subscriber(FRAME_KEY, self._on_mocap)
            )
            for key in MANUS_RAW_KEYS:
                self._subscribers.append(
                    self._session.declare_subscriber(key, self._on_manus)
                )
            for key in MEDIAPIPE_KEYS:
                self._subscribers.append(
                    self._session.declare_subscriber(key, self._on_mano)
                )
            for key in MANUS_EDGE_KEYS:
                self._subscribers.append(
                    self._session.declare_subscriber(key, self._on_edges)
                )

    def stop(self) -> None:
        """Undeclare 全部订阅并关闭 session,可重复调用。"""
        with self._lock:
            subscribers, self._subscribers = self._subscribers, []
            session, self._session = self._session, None
        for sub in subscribers:
            try:
                sub.undeclare()
            except BaseException:
                # 清理路径不可中断:Ctrl-C 若在 undeclare 期间到达,
                # KeyboardInterrupt(BaseException)也要吞掉,保证 stop 完成
                pass
        if session is not None:
            try:
                session.close()
            except BaseException:
                pass

    # -- zenoh 回调(库线程,禁止阻塞) ---------------------------------------

    def _on_mocap(self, sample: object) -> None:
        try:
            frame = decode_frame(sample.payload.to_string())
        except Exception:
            return
        t_ns = time.time_ns()
        stamped = _stamp(frame, t_ns)
        self._latest_mocap = stamped
        # 维护插值历史(按时间排序;清理超过保留时长的旧帧)
        cutoff = t_ns - int(MOCAP_HISTORY_SEC * 1e9)
        hist = self._mocap_history
        hist.append((t_ns, stamped))
        while hist and hist[0][0] < cutoff:
            hist.popleft()
        with self._lock:
            cbs = list(self._mocap_cbs)
            self._rate_counts["mocap"] += 1
        for cb in cbs:
            cb(self._latest_mocap)

    def _on_manus(self, sample: object) -> None:
        side = side_from_key(str(sample.key_expr))
        # rawviz 异常状态可能输出 Unknown 侧:不注册对应槽位,直接忽略
        if side not in self._latest_manus:
            return
        try:
            msg = decode_manus(sample.payload.to_bytes())
        except ManusError:
            return
        t_ns = time.time_ns()
        msg["side"] = side
        stamped = _stamp(msg, t_ns)
        self._latest_manus[side] = stamped
        with self._lock:
            cbs = list(self._manus_cbs[side])
            self._rate_counts[side] += 1
        for cb in cbs:
            cb(stamped)

    def _on_mano(self, sample: object) -> None:
        side = side_from_key(str(sample.key_expr))
        if side not in self._latest_mano:
            return
        try:
            msg = decode_mano(sample.payload.to_bytes())
        except ManusError:
            return
        t_ns = time.time_ns()
        msg["side"] = side
        stamped = _stamp(msg, t_ns)
        self._latest_mano[side] = stamped
        with self._lock:
            cbs = list(self._mano_cbs[side])
        for cb in cbs:
            cb(stamped)

    def _on_edges(self, sample: object) -> None:
        side = side_from_key(str(sample.key_expr))
        if side not in self._latest_manus:
            return
        try:
            edges = parse_edges(sample.payload.to_bytes())
        except ManusError:
            return
        self._latest_edges[side] = edges
