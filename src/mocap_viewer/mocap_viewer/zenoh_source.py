"""Zenoh subscription that feeds validated frames into a LatestFrameQueue."""

from __future__ import annotations

import json
import threading

import zenoh

from natnet_zenoh.frame_queue import LatestFrameQueue
from natnet_zenoh.schema import encode_frame
from natnet_zenoh.subscriber import FrameHandler, FrameStats

# 转发标记：本机作为数据 hub 时，把收到的帧重新发布到同一 key，让
# connect 模式消费者能直连本机订阅。Zenoh 1.9 peer 只在直连对之间交换
# 订阅路由，多跳转发不可靠，因此 hub 必须自己重新发布。同 session 内
# 发布会触发自己的订阅回调（自回环），用该标记跳过已转发的帧。
RELAY_MARKER = "relayed_by_mocap_viewer"


class ZenohSource:
    """Owns the Zenoh session and routes validated frames into a queue.

    The Zenoh library invokes the subscriber callback on its own thread; the
    callback must never block. ``FrameHandler`` validates and updates stats,
    then we enqueue via ``LatestFrameQueue.put_latest`` which drops the oldest
    frame instead of blocking when full.
    """

    def __init__(
        self,
        key: str,
        *,
        listen_endpoint: str | None = None,
        connect_endpoint: str | None = None,
        queue_capacity: int = 8,
        relay: bool = False,
        zenoh_module: object = zenoh,
        stats: FrameStats | None = None,
        handler_factory=FrameHandler,
    ) -> None:
        if (listen_endpoint is None) == (connect_endpoint is None):
            raise ValueError("exactly one of listen_endpoint or connect_endpoint is required")
        self._zenoh = zenoh_module
        self._listen_endpoint = listen_endpoint
        self._connect_endpoint = connect_endpoint
        self._key = key
        self._relay = relay
        self.queue = LatestFrameQueue(queue_capacity)
        self.stats = stats if stats is not None else FrameStats()
        self._handler = handler_factory(self.stats)
        self._session = None
        self._subscriber = None
        self._publisher = None
        self._lock = threading.Lock()

    def _build_config(self):
        if self._connect_endpoint is not None:
            config: dict[str, object] = {"connect": {"endpoints": [self._connect_endpoint]}}
        else:
            config = {
                "listen": {"endpoints": [self._listen_endpoint]},
                # 显式 TCP 直连,不参与组播/发现:避免误加入本地其它 zenoh 网络
                # (如常驻 router),收到/扩散非本链路数据(安全暴露面 + 测试污染)
                "scouting": {
                    "multicast": {"enabled": False},
                    "gossip": {"enabled": False},
                },
            }
        return zenoh.Config.from_json5(json.dumps(config))

    def _on_sample(self, sample: object) -> None:
        """Zenoh callback: validate, update stats, then enqueue without blocking."""
        frame = self._handler.handle_json(sample.payload.to_string())
        if frame is None:
            return
        if self._relay:
            if frame.get(RELAY_MARKER):
                # 自己转发出去的回环副本:不入队、不转发,否则每帧重复入队
                # (统计失真)且经 router 重复投递给其它订阅者(重复录制)。
                return
            # 转发给 connect 模式的消费者;带标记的帧是自己发布的,跳过防回环
            frame[RELAY_MARKER] = True
            self._publisher.put(encode_frame(frame))
        self.queue.put_latest(frame)

    def start(self) -> None:
        """Open the Zenoh session and declare the subscriber. Idempotence guarded."""
        with self._lock:
            if self._session is not None:
                raise RuntimeError("ZenohSource is already started")
            self._session = self._zenoh.open(self._build_config())
            self._subscriber = self._session.declare_subscriber(self._key, self._on_sample)
            if self._relay:
                self._publisher = self._session.declare_publisher(
                    self._key, encoding=zenoh.Encoding.APPLICATION_JSON
                )

    def stop(self) -> None:
        """Undeclare the subscriber and close the session. Safe to call twice."""
        with self._lock:
            subscriber, self._subscriber = self._subscriber, None
            publisher, self._publisher = self._publisher, None
            session, self._session = self._session, None
        if subscriber is not None:
            subscriber.undeclare()
        if publisher is not None:
            publisher.undeclare()
        if session is not None:
            session.close()
