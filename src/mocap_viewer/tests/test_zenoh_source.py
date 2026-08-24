"""Tests for ZenohSource using a fake Zenoh module (no real network)."""

from __future__ import annotations

from natnet_zenoh.schema import FRAME_KEY, decode_frame, encode_frame

from mocap_viewer.zenoh_source import RELAY_MARKER, ZenohSource


def valid_frame(number: int = 42) -> dict:
    return {
        "schema_version": 1,
        "frame_number": number,
        "motive_timestamp": 1.25,
        "publisher_received_time_ns": 123456789,
        "coordinate_system": "motive_x_forward_z_up_right_handed",
        "unit": "meter",
        "publisher_dropped_frames": 0,
        "markers": [],
        "rigid_bodies": [],
    }


class FakePayload:
    def __init__(self, data: str) -> None:
        self._data = data

    def to_string(self) -> str:
        return self._data


class FakeSample:
    def __init__(self, data: str) -> None:
        self.payload = FakePayload(data)


class FakeSubscriber:
    def __init__(self, key: str) -> None:
        self.key = key
        self.undeclared = False

    def undeclare(self) -> None:
        self.undeclared = True


class FakePublisher:
    def __init__(self) -> None:
        self.put_payloads: list[str] = []
        self.undeclared = False

    def put(self, payload: str) -> None:
        self.put_payloads.append(payload)

    def undeclare(self) -> None:
        self.undeclared = True


class FakeSession:
    def __init__(self) -> None:
        self.closed = False
        self.key = None
        self.handler = None
        self.subscriber = FakeSubscriber("<unset>")
        self.publisher = None

    def declare_subscriber(self, key: str, handler) -> FakeSubscriber:
        self.key = key
        self.handler = handler
        self.subscriber = FakeSubscriber(key)
        return self.subscriber

    def declare_publisher(self, key: str, encoding=None) -> FakePublisher:
        self.publisher = FakePublisher()
        return self.publisher

    def close(self) -> None:
        self.closed = True


class FakeZenoh:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.opened_configs: list = []

    def open(self, config) -> FakeSession:
        self.opened_configs.append(config)
        return self.session


def make_source(session: FakeSession, queue_capacity: int = 8, relay: bool = False) -> ZenohSource:
    return ZenohSource(
        FRAME_KEY,
        listen_endpoint="tcp/0.0.0.0:7447",
        queue_capacity=queue_capacity,
        relay=relay,
        zenoh_module=FakeZenoh(session),
    )


def feed(source: ZenohSource, frame: dict) -> None:
    source._on_sample(FakeSample(encode_frame(frame)))


def test_source_routes_valid_frame_into_queue() -> None:
    session = FakeSession()
    source = make_source(session)
    source.start()
    feed(source, valid_frame(7))
    frame = source.queue.get(timeout=1.0)
    assert frame["frame_number"] == 7
    assert source.stats.received_frames == 1
    source.stop()


def test_source_rejects_invalid_payload() -> None:
    session = FakeSession()
    source = make_source(session)
    source.start()
    source._on_sample(FakeSample("not json"))
    assert source.queue.empty()
    assert source.stats.invalid_messages == 1
    source.stop()


def test_source_declares_expected_key() -> None:
    session = FakeSession()
    source = make_source(session)
    source.start()
    assert session.key == FRAME_KEY
    source.stop()


def test_source_queue_overflow_counts_drops() -> None:
    session = FakeSession()
    source = make_source(session, queue_capacity=1)
    source.start()
    feed(source, valid_frame(1))
    feed(source, valid_frame(2))
    assert source.queue.dropped_frames == 1
    kept = source.queue.get(timeout=1.0)
    assert kept["frame_number"] == 2
    source.stop()


def test_source_stop_undeclares_and_closes_session() -> None:
    session = FakeSession()
    source = make_source(session, relay=True)
    source.start()
    source.stop()
    assert session.subscriber.undeclared
    assert session.publisher.undeclared
    assert session.closed
    source.stop()  # 幂等


def test_source_start_twice_raises_runtime_error() -> None:
    session = FakeSession()
    source = make_source(session)
    source.start()
    try:
        source.start()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    source.stop()


def test_source_connect_mode_passes_connect_config() -> None:
    session = FakeSession()
    zenoh_mod = FakeZenoh(session)
    source = ZenohSource(
        FRAME_KEY,
        connect_endpoint="tcp/127.0.0.1:7447",
        zenoh_module=zenoh_mod,
    )
    source.start()
    config_str = str(zenoh_mod.opened_configs[0])
    assert "connect" in config_str
    assert "tcp/127.0.0.1:7447" in config_str
    source.stop()


def test_source_requires_exactly_one_endpoint() -> None:
    try:
        ZenohSource(FRAME_KEY)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        ZenohSource(FRAME_KEY, listen_endpoint="a", connect_endpoint="b")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_source_relay_forwards_unmarked_frame() -> None:
    session = FakeSession()
    source = make_source(session, relay=True)
    source.start()
    feed(source, valid_frame(3))
    assert len(session.publisher.put_payloads) == 1
    decoded = decode_frame(session.publisher.put_payloads[0])
    assert decoded.get(RELAY_MARKER) is True
    assert decoded["frame_number"] == 3
    source.stop()


def test_source_relay_skips_marked_frame() -> None:
    """自己转发回来的回环副本:不转发、也不入队(否则每帧重复显示/统计失真)。"""
    session = FakeSession()
    source = make_source(session, relay=True)
    source.start()
    frame = valid_frame(4)
    frame[RELAY_MARKER] = True  # 模拟自己转发回来的帧
    feed(source, frame)
    assert session.publisher.put_payloads == []  # 防回环
    assert source.queue.empty()                   # 回环副本不入队
    assert source.stats.received_frames == 1      # 统计仍计一次(原始帧)
    source.stop()
