"""Zenoh 多流订阅；保留源时间并映射到本机 CLOCK_MONOTONIC 时间域。

Zenoh 回调只做解码、时钟映射、缓存和入队。``t_phys_ns`` 是中央对齐器
唯一使用的时间字段；``t_ubuntu_ns`` 仅保留为到达 wall-clock 诊断字段。
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable

import zenoh

RATE_WINDOW = 2.0            # 帧率统计滑动窗口(秒),窗口满结算重置

DISPATCH_QUEUE_CAPACITY = 2048

from natnet_zenoh.schema import FRAME_KEY, decode_frame
from .clock_sync import ClockAligner


RIGID_BODY_NAMES_KEY = "mocap/rigid_body_names"   # Windows publisher 发布的刚体名→ID

from .manus_schema import (
    MANUS_EDGE_KEYS,
    MANUS_RAW_KEYS,
    ManusError,
    decode_manus,
    parse_edges,
    side_from_key,
)

FrameCallback = Callable[[dict], None]   # 已打 t_ubuntu_ns 的帧


def _stamp(frame: dict, *, wall_ns: int, monotonic_ns: int, phys_ns: int) -> dict:
    frame["t_ubuntu_ns"] = wall_ns
    frame["arrival_monotonic_ns"] = monotonic_ns
    frame["t_phys_ns"] = phys_ns
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
        self._lock = threading.RLock()

        self._latest_mocap: dict | None = None
        self._mocap_clock = ClockAligner()
        self._manus_clocks = {
            "left": ClockAligner(),
            "right": ClockAligner(),
        }
        self._rigid_body_names: dict[int, str] = {}
        self._rigid_body_ids: dict[str, int] = {}
        self._latest_manus: dict[str, dict | None] = {"left": None, "right": None}
        self._latest_edges: dict[str, list[tuple[int, int, int]]] = {}

        self._mocap_cbs: list[FrameCallback] = []
        self._manus_cbs: dict[str, list[FrameCallback]] = {"left": [], "right": []}

        self._dispatch_q: queue.Queue[tuple[str, list[FrameCallback], dict]] = (
            queue.Queue(maxsize=DISPATCH_QUEUE_CAPACITY))
        self._dispatch_stop = threading.Event()
        self._dispatch_thread: threading.Thread | None = None
        self._health = {
            name: {
                "received": 0,
                "decode_errors": 0,
                "sequence_gaps": 0,
                "out_of_order": 0,
                "callback_queue_dropped": 0,
                "callback_errors": 0,
                "last_received_ns": 0,
            }
            for name in ("mocap", "left", "right")
        }
        self._last_sequence: dict[str, int | None] = {
            "mocap": None, "left": None, "right": None}

        self._rate_counts = {"mocap": 0, "left": 0, "right": 0}
        now = time.time()
        self._rate_start = {"mocap": now, "left": now, "right": now}
        self._last_rate = {"mocap": 0.0, "left": 0.0, "right": 0.0}

    # -- 订阅注册 ---------------------------------------------------------

    def on_mocap(self, cb: FrameCallback) -> None:
        """订阅 mocap 帧；回调由内部 worker 串行执行，不占 Zenoh 线程。"""
        with self._lock:
            self._mocap_cbs.append(cb)

    def on_manus(self, side: str, cb: FrameCallback) -> None:
        if side not in self._manus_cbs:
            raise ValueError(f"side 须为 left/right,实际 {side!r}")
        with self._lock:
            self._manus_cbs[side].append(cb)


    # -- 最新帧查询 -------------------------------------------------------

    def latest_mocap(self) -> dict | None:
        with self._lock:
            return self._latest_mocap


    def latest_manus(self, side: str) -> dict | None:
        with self._lock:
            return self._latest_manus[side]


    def latest_edges(self, side: str) -> list[tuple[int, int, int]] | None:
        with self._lock:
            edges = self._latest_edges.get(side)
            return list(edges) if edges is not None else None

    def rigid_body_names(self) -> dict[int, str]:
        with self._lock:
            return dict(self._rigid_body_names)

    def health_snapshot(self) -> dict[str, object]:
        """线程安全健康快照，供状态栏和 HDF5 元数据落盘。"""
        with self._lock:
            streams = {name: dict(values) for name, values in self._health.items()}
        return {
            "streams": streams,
            "dispatch_queue_depth": self._dispatch_q.qsize(),
            "clock_alignment": {
                "mocap": self._mocap_clock.quality(),
                "left": self._manus_clocks["left"].quality(),
                "right": self._manus_clocks["right"].quality(),
            },
        }

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

    def _observe_sequence(self, stream: str, sequence: int | None, t_ns: int) -> None:
        health = self._health[stream]
        health["received"] += 1
        health["last_received_ns"] = t_ns
        if sequence is None or sequence < 0:
            return
        previous = self._last_sequence[stream]
        if previous is not None:
            if sequence <= previous:
                health["out_of_order"] += 1
            elif sequence > previous + 1:
                health["sequence_gaps"] += sequence - previous - 1
        self._last_sequence[stream] = sequence

    def _dispatch_loop(self) -> None:
        while not self._dispatch_stop.is_set() or not self._dispatch_q.empty():
            try:
                stream, callbacks, frame = self._dispatch_q.get(timeout=0.05)
            except queue.Empty:
                continue
            for callback in callbacks:
                try:
                    callback(frame)
                except Exception:
                    with self._lock:
                        self._health[stream]["callback_errors"] += 1

    def _enqueue(self, stream: str, callbacks: list[FrameCallback], frame: dict) -> None:
        if not callbacks:
            return
        if self._dispatch_thread is None:
            # 直接调用 _on_* 的单元测试/离线用法保持同步语义。
            for callback in callbacks:
                try:
                    callback(frame)
                except Exception:
                    with self._lock:
                        self._health[stream]["callback_errors"] += 1
            return
        item = (stream, callbacks, frame)
        try:
            self._dispatch_q.put_nowait(item)
            return
        except queue.Full:
            pass
        # 实时系统保留最新帧：淘汰队首并准确记到被淘汰的流。
        try:
            dropped_stream, _callbacks, _frame = self._dispatch_q.get_nowait()
            with self._lock:
                self._health[dropped_stream]["callback_queue_dropped"] += 1
        except queue.Empty:
            pass
        try:
            self._dispatch_q.put_nowait(item)
        except queue.Full:
            # 多生产者竞争同一空位；丢当前帧而不是让 Zenoh 回调抛异常。
            with self._lock:
                self._health[stream]["callback_queue_dropped"] += 1

    # -- 生命周期 ---------------------------------------------------------

    def start(self) -> None:
        """启动回调 worker，连接 router 并声明全部订阅。"""
        with self._lock:
            if self._session is not None:
                raise RuntimeError("StreamHub 已启动")
        self._dispatch_stop.clear()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="stream-dispatch", daemon=True)
        self._dispatch_thread.start()
        session = None
        subscribers: list[object] = []
        try:
            config = zenoh.Config.from_json5(json.dumps({
                "mode": "client",
                "connect": {"endpoints": [self._router_endpoint]},
            }))
            session = zenoh.open(config)
            subscribers = [
                session.declare_subscriber(FRAME_KEY, self._on_mocap),
                session.declare_subscriber(
                    RIGID_BODY_NAMES_KEY, self._on_rigid_body_names),
            ]
            subscribers.extend(
                session.declare_subscriber(key, self._on_manus)
                for key in MANUS_RAW_KEYS)
            subscribers.extend(
                session.declare_subscriber(key, self._on_edges)
                for key in MANUS_EDGE_KEYS)
            with self._lock:
                self._session = session
                self._subscribers = subscribers
        except BaseException:
            for subscriber in subscribers:
                try:
                    subscriber.undeclare()
                except BaseException:
                    pass
            if session is not None:
                try:
                    session.close()
                except BaseException:
                    pass
            self._dispatch_stop.set()
            self._dispatch_thread.join(timeout=2)
            self._dispatch_thread = None
            raise

    def stop(self) -> None:
        """停止输入后排空回调队列并关闭 session；可重复调用。"""
        with self._lock:
            subscribers, self._subscribers = self._subscribers, []
            session, self._session = self._session, None
        for subscriber in subscribers:
            try:
                subscriber.undeclare()
            except BaseException:
                pass
        if session is not None:
            try:
                session.close()
            except BaseException:
                pass
        self._dispatch_stop.set()
        thread, self._dispatch_thread = self._dispatch_thread, None
        if thread is not None:
            thread.join(timeout=5)

    # -- zenoh 回调(库线程,禁止阻塞) ---------------------------------------

    def _on_rigid_body_names(self, sample: object) -> None:
        """缓存 Windows publisher 发布的刚体名字→ID 映射。"""
        try:
            msg = json.loads(sample.payload.to_string())
            names = msg.get("names")
            if not isinstance(names, dict):
                return
            mapping = {
                int(rid): str(name)
                for rid, name in names.items()
            }
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return
        with self._lock:
            self._rigid_body_names = mapping
            self._rigid_body_ids = {name: rid for rid, name in mapping.items()}

    def rigid_body_id(self, name: str) -> int | None:
        with self._lock:
            return self._rigid_body_ids.get(name)

    def rigid_body_name(self, rid: int) -> str | None:
        with self._lock:
            return self._rigid_body_names.get(rid)

    def _on_mocap(self, sample: object) -> None:
        try:
            frame = decode_frame(sample.payload.to_string())
        except Exception:
            with self._lock:
                self._health["mocap"]["decode_errors"] += 1
            return
        arrival_wall_ns = time.time_ns()
        arrival_monotonic_ns = time.monotonic_ns()
        aligned_ns = self._mocap_clock.observe(
            int(frame.get("publisher_received_time_ns", 0)),
            arrival_monotonic_ns,
        )
        stamped = _stamp(
            frame,
            wall_ns=arrival_wall_ns,
            monotonic_ns=arrival_monotonic_ns,
            phys_ns=aligned_ns,
        )
        with self._lock:
            self._latest_mocap = stamped
            self._observe_sequence(
                "mocap", frame.get("frame_number"), arrival_monotonic_ns)
            self._rate_counts["mocap"] += 1
            callbacks = list(self._mocap_cbs)
        self._enqueue("mocap", callbacks, stamped)

    def _on_manus(self, sample: object) -> None:
        side = side_from_key(str(sample.key_expr))
        if side not in self._latest_manus:
            return
        try:
            msg = decode_manus(sample.payload.to_bytes())
        except ManusError:
            with self._lock:
                self._health[side]["decode_errors"] += 1
            return
        arrival_wall_ns = time.time_ns()
        arrival_monotonic_ns = time.monotonic_ns()
        source_ns = int(msg.get("source_monotonic_ns", 0))
        aligned_ns = self._manus_clocks[side].observe(
            source_ns, arrival_monotonic_ns,
        )
        msg["side"] = side
        stamped = _stamp(
            msg,
            wall_ns=arrival_wall_ns,
            monotonic_ns=arrival_monotonic_ns,
            phys_ns=aligned_ns,
        )
        with self._lock:
            self._latest_manus[side] = stamped
            self._observe_sequence(side, msg.get("seq"), arrival_monotonic_ns)
            self._rate_counts[side] += 1
            callbacks = list(self._manus_cbs[side])
        self._enqueue(side, callbacks, stamped)


    def _on_edges(self, sample: object) -> None:
        side = side_from_key(str(sample.key_expr))
        if side not in self._latest_manus:
            return
        try:
            edges = parse_edges(sample.payload.to_bytes())
        except ManusError:
            return
        with self._lock:
            self._latest_edges[side] = edges
