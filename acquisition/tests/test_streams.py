"""StreamHub 集成测试:真实 zenohd + 真实订阅(需要 zenohd 可用)。"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zenoh

from acquisition.manus_schema import MANUS_EDGE_KEYS, MANUS_RAW_KEYS
from acquisition.streams import StreamHub
from natnet_zenoh.schema import FRAME_KEY, encode_frame
from natnet_zenoh.zenoh_transport import build_peer_config

# 默认 zenohd:位于合并结构的 mocap/manus/.pixi(基于本文件位置推导)
_DEFAULT_ZENOHD = str(
    Path(__file__).resolve().parent.parent.parent
    / "manus" / ".pixi" / "envs" / "default" / "bin" / "zenohd"
)
ZENOHD = os.environ.get("ZENOHD_BIN", _DEFAULT_ZENOHD)
ROUTER = "tcp/127.0.0.1:7447"


def _port_open(port: int = 7447) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


@pytest.fixture(scope="module")
def router():
    """优先复用已在 7447 监听的 zenohd;否则自起一个(仅当二进制存在)。"""
    proc = None
    if not _port_open():
        if not os.path.exists(ZENOHD):
            pytest.skip(f"zenohd 不存在: {ZENOHD}")
        proc = subprocess.Popen(
            [ZENOHD], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.time() + 10
        while time.time() < deadline and not _port_open():
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if not _port_open():
            proc.terminate()
            pytest.skip("zenohd 端口未就绪")
    yield proc
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=5)


def _publisher():
    """测试发布端:client 模式连 router(与 StreamHub 一致)。"""
    return zenoh.open(zenoh.Config.from_json5(json.dumps({
        "mode": "client",
        "connect": {"endpoints": [ROUTER]},
    })))


def _mocap_frame(number: int) -> dict:
    return {
        "schema_version": 1,
        "frame_number": number,
        "motive_timestamp": float(number),
        "publisher_received_time_ns": 0,
        "coordinate_system": "motive_y_up_right_handed",
        "unit": "meter",
        "publisher_dropped_frames": 0,
        "markers": [],
        "rigid_bodies": [
            {"id": 5, "position": [0.3, 1.2, -0.2],
             "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
             "mean_error": 0.0004, "tracking_valid": True},
        ],
    }
def _manus_frame(sequence: int, side: str) -> dict:
    nodes = [[float(index), 0.0, 0.0] for index in range(25)]
    return {
        "glove_id": side,
        "side": side,
        "seq": sequence,
        "source_monotonic_ns": time.monotonic_ns(),
        "sdk_publish_time": sequence,
        "nodes": nodes,
        "node_quaternions_wxyz": [[1.0, 0.0, 0.0, 0.0] for _ in range(25)],
    }




def test_hub_receives_mocap_and_manus(router):
    """用独特帧号/seq 匹配自己发布的数据(router 上可能有真实流干扰)。"""
    hub = StreamHub(ROUTER)
    got = {"mocap": [], "left": [], "right": [], "edges": []}
    hub.on_mocap(lambda f: got["mocap"].append(f))
    hub.on_manus("left", lambda f: got["left"].append(f))
    hub.on_manus("right", lambda f: got["right"].append(f))
    hub.start()
    try:
        with _publisher() as session:
            for i in range(3):
                session.put(FRAME_KEY, encode_frame(_mocap_frame(9_000_000 + i)))
            session.put(
                MANUS_RAW_KEYS[0],
                json.dumps(_manus_frame(90001, "left")),
            )
            session.put(
                MANUS_RAW_KEYS[1],
                json.dumps(_manus_frame(90002, "right")),
            )
            session.put(MANUS_EDGE_KEYS[0], json.dumps(
                {"glove_id": "a", "edges": [[1, 0, 13]]}))

        deadline = time.time() + 5
        while time.time() < deadline:
            frames = {f["frame_number"] for f in got["mocap"]}
            seqs_l = {f["seq"] for f in got["left"]}
            seqs_r = {f["seq"] for f in got["right"]}
            if (all(9_000_000 + i in frames for i in range(3))
                    and 90001 in seqs_l and 90002 in seqs_r and got["edges"]):
                break
            time.sleep(0.05)

        assert all(9_000_000 + i in {f["frame_number"] for f in got["mocap"]}
                   for i in range(3))
        mine = next(f for f in got["mocap"] if f["frame_number"] == 9_000_000)
        assert mine["t_ubuntu_ns"] > 0                    # 已打时间戳
        assert mine["t_phys_ns"] > 0
        assert any(f["side"] == "left" and f["seq"] == 90001 for f in got["left"])
        left = next(f for f in got["left"] if f["seq"] == 90001)
        assert left["nodes"][1] == [1.0, 0.0, 0.0]
        assert any(f["seq"] == 90002 for f in got["right"])
        # router 上可能有真实/合成流覆盖 latest_edges:只要求拓扑已就绪(有手掌边)
        edges = hub.latest_edges("left")
        assert edges is not None and any(chain == 13 for _, _, chain in edges)
        # latest 可能已被真实/合成流覆盖,只要求我们发布的帧曾到达
        assert any(f["seq"] == 90001 for f in got["left"])
    finally:
        hub.stop()


class _FakePayload:
    def __init__(self, data: bytes | str):
        self._data = data

    def to_string(self) -> str:
        return self._data if isinstance(self._data, str) else self._data.decode()

    def to_bytes(self) -> bytes:
        return self._data.encode() if isinstance(self._data, str) else self._data


class _FakeSample:
    def __init__(self, key: str, payload: bytes | str):
        self.key_expr = key
        self.payload = _FakePayload(payload)


def test_hub_rejects_bad_frames():
    """非法帧不触发回调、不更新 latest(纯单元测试,无网络依赖)。"""
    hub = StreamHub("tcp/127.0.0.1:7447")     # 不 start,直接测回调
    got = []
    hub.on_mocap(lambda f: got.append(f))
    hub.on_manus("left", lambda f: got.append(f))

    hub._on_mocap(_FakeSample(FRAME_KEY, "not a frame"))
    hub._on_manus(_FakeSample(MANUS_RAW_KEYS[0], "not json"))
    assert got == []
    assert hub.latest_mocap() is None
    assert hub.latest_manus("left") is None

    # 好帧正常进入
    hub._on_mocap(_FakeSample(FRAME_KEY, encode_frame(_mocap_frame(7))))
    assert len(got) == 1
    assert hub.latest_mocap()["frame_number"] == 7
    assert hub.latest_mocap()["t_ubuntu_ns"] > 0


def test_health_snapshot_counts_gaps_order_and_decode_errors():
    hub = StreamHub("tcp/127.0.0.1:7447")
    for frame_number in (10, 13, 12):
        hub._on_mocap(_FakeSample(
            FRAME_KEY, encode_frame(_mocap_frame(frame_number))))
    hub._on_mocap(_FakeSample(FRAME_KEY, "not-json"))

    health = hub.health_snapshot()["streams"]["mocap"]
    assert health["received"] == 3
    assert health["sequence_gaps"] == 2
    assert health["out_of_order"] == 1
    assert health["decode_errors"] == 1


def test_callbacks_are_serialized_off_input_threads():
    hub = StreamHub("tcp/127.0.0.1:7447")
    active = 0
    max_active = 0
    received = 0
    lock = threading.Lock()

    def slow_callback(_frame):
        nonlocal active, max_active, received
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.002)
        with lock:
            active -= 1
            received += 1

    hub.on_mocap(slow_callback)
    hub.on_manus("left", slow_callback)
    hub._dispatch_stop.clear()
    hub._dispatch_thread = threading.Thread(
        target=hub._dispatch_loop, daemon=True)
    hub._dispatch_thread.start()
    for sequence in range(10):
        hub._on_mocap(_FakeSample(
            FRAME_KEY, encode_frame(_mocap_frame(sequence))))
        hub._on_manus(_FakeSample(
            MANUS_RAW_KEYS[0],
            json.dumps(_manus_frame(sequence, "left")),
        ))
    hub.stop()

    assert received == 20
    assert max_active == 1


def test_hub_ignores_unknown_side():
    """rawviz 异常输出的 Unknown 侧:忽略而非 KeyError(纯单元测试)。"""
    hub = StreamHub("tcp/127.0.0.1:7447")
    unknown = _manus_frame(1, "unknown")
    hub._on_manus(_FakeSample(
        "manus/raw_skeleton/unknown", json.dumps(unknown),
    ))
    hub._on_edges(_FakeSample("manus/skeleton_edges/unknown", json.dumps(
        {"glove_id": "x", "edges": [[1, 0, 13]]})))
    # 未崩溃即通过;left/right 槽位未被污染
    assert hub.latest_manus("left") is None
    assert hub.latest_manus("right") is None
    assert hub.latest_edges("left") is None


def test_hub_rejects_huge_edges():
    """不受信注入的超大 edges 列表被 parse_edges 上限拒绝,不触发回调。"""
    hub = StreamHub("tcp/127.0.0.1:7447")
    hub._on_edges(_FakeSample(MANUS_EDGE_KEYS[0], json.dumps(
        {"glove_id": "x", "edges": [[i, 0, 13] for i in range(1, 70)]})))
    assert hub.latest_edges("left") is None


def test_hub_rejects_nan_manus():
    """含 NaN 的 manus 帧被拒绝(有限性校验)。"""
    hub = StreamHub("tcp/127.0.0.1:7447")
    message = _manus_frame(1, "left")
    message["nodes"][0][0] = float("nan")
    hub._on_manus(_FakeSample(
        MANUS_RAW_KEYS[0], json.dumps(message),
    ))
    assert hub.latest_manus("left") is None



def test_hub_rigid_body_names_mapping():
    """刚体名字表订阅:按名字查 id、按 id 查名字。"""
    hub = StreamHub("tcp/127.0.0.1:7447")
    hub._on_rigid_body_names(_FakeSample(
        "mocap/rigid_body_names",
        json.dumps({"names": {"2": "left_wrist", "4": "left_dip", "1": "right_wrist"}})))
    assert hub.rigid_body_id("left_dip") == 4
    assert hub.rigid_body_id("right_wrist") == 1
    assert hub.rigid_body_name(2) == "left_wrist"
    assert hub.rigid_body_id("no_such") is None
    # 覆盖更新
    hub._on_rigid_body_names(_FakeSample(
        "mocap/rigid_body_names", json.dumps({"names": {"5": "left_dip"}})))
    assert hub.rigid_body_id("left_dip") == 5
    assert hub.rigid_body_id("left_wrist") is None
