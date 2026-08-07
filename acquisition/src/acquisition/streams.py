"""StreamHub:单 Zenoh session 多流订阅(mocap 帧 + manus 双手骨架/拓扑)。

回调在 zenoh 库线程上执行,只做「校验 → 打 t_ubuntu_ns → 广播给订阅者」,
绝不阻塞。订阅者回调须自行入队/拷贝(录制器、可视化各自处理)。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

import zenoh

from natnet_zenoh.schema import FRAME_KEY, decode_frame

from .manus_schema import (
    MANUS_EDGE_KEYS,
    MANUS_RAW_KEYS,
    ManusError,
    decode_manus,
    parse_edges,
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
        self._latest_manus: dict[str, dict] = {"left": None, "right": None}
        self._latest_edges: dict[str, list[tuple[int, int, int]]] = {}

        self._mocap_cbs: list[FrameCallback] = []
        self._manus_cbs: dict[str, list[FrameCallback]] = {"left": [], "right": []}

        self._counters = {"mocap": 0, "left": 0, "right": 0}
        now = time.time()
        self._last_time = {"mocap": now, "left": now, "right": now}

    # -- 订阅注册 ---------------------------------------------------------

    def on_mocap(self, cb: FrameCallback) -> None:
        """订阅 mocap 帧(帧已校验并带 t_ubuntu_ns)。"""
        self._mocap_cbs.append(cb)

    def on_manus(self, side: str, cb: FrameCallback) -> None:
        """订阅 manus 骨架帧(side ∈ left/right)。"""
        if side not in self._manus_cbs:
            raise ValueError(f"side 须为 left/right,实际 {side!r}")
        self._manus_cbs[side].append(cb)

    # -- 最新帧查询 -------------------------------------------------------

    def latest_mocap(self) -> dict | None:
        return self._latest_mocap

    def latest_manus(self, side: str) -> dict | None:
        return self._latest_manus[side]

    def latest_edges(self, side: str) -> list[tuple[int, int, int]] | None:
        return self._latest_edges.get(side)

    def rates_hz(self) -> dict[str, float]:
        """每路流最近帧率估计。"""
        rates = {}
        now = time.time()
        for name, count in self._counters.items():
            window = now - self._last_time[name]
            rates[name] = count / window if window > 1.0 else 0.0
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
            except Exception:
                pass
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    # -- zenoh 回调(库线程,禁止阻塞) ---------------------------------------

    def _on_mocap(self, sample: object) -> None:
        try:
            frame = decode_frame(sample.payload.to_string())
        except Exception:
            return
        t_ns = time.time_ns()
        self._latest_mocap = _stamp(frame, t_ns)
        with self._lock:
            cbs = list(self._mocap_cbs)
        self._counters["mocap"] += 1
        self._last_time["mocap"] = time.time()
        for cb in cbs:
            cb(self._latest_mocap)

    def _on_manus(self, sample: object) -> None:
        side = str(sample.key_expr).rsplit("/", 1)[-1]
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
        self._counters[side] += 1
        self._last_time[side] = time.time()
        for cb in cbs:
            cb(stamped)

    def _on_edges(self, sample: object) -> None:
        side = str(sample.key_expr).rsplit("/", 1)[-1]
        try:
            edges = parse_edges(sample.payload.to_bytes())
        except ManusError:
            return
        self._latest_edges[side] = edges
