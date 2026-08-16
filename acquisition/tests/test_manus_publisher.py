"""Manus Zenoh 发布器协议与拓扑周期重发测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "manus" / "zenoh_pub.py"
_SPEC = importlib.util.spec_from_file_location("manus_zenoh_pub", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

EDGE_PERIOD = _MOD.EDGE_REPUBLISH_PERIOD
ZenohPublisher = _MOD.ZenohPublisher


class _Session:
    def __init__(self):
        self.rows: list[tuple[str, dict]] = []

    def put(self, key: str, payload: str) -> None:
        self.rows.append((key, json.loads(payload)))


def _pose(seq: int) -> str:
    row = "0 0 0 1 0 0 0"
    return f"POSE glove {seq} {1_000_000_000 + seq} {seq} " + " ".join(
        [row] * 25
    )


def test_edges_republish_uses_last_edge_sequence_not_last_frame():
    pub = ZenohPublisher()
    session = _Session()
    pub.handle_line(session, "HAND glove right 25")
    pub.handle_line(session, "EDGE glove 2 1 6")

    pub.handle_line(session, _pose(10))
    pub.handle_line(session, _pose(11))
    pub.handle_line(session, _pose(10 + EDGE_PERIOD))

    edges = [row for row in session.rows if row[0].startswith("manus/skeleton_edges/")]
    assert len(edges) == 2
    assert edges[0][1]["edges"] == [[1, 0, 6]]


def test_sequence_wrap_immediately_republishes_edges():
    pub = ZenohPublisher()
    session = _Session()
    pub.handle_line(session, "HAND glove left 25")
    pub.handle_line(session, "EDGE glove 2 1 5")
    pub.handle_line(session, _pose(100))
    pub.handle_line(session, _pose(1))

    edges = [row for row in session.rows if row[0].startswith("manus/skeleton_edges/")]
    assert len(edges) == 2


def test_each_position_frame_publishes_only_one_skeleton_payload():
    pub = ZenohPublisher()
    session = _Session()
    pub.handle_line(session, "HAND glove left 25")
    pub.handle_line(session, _pose(7))

    skeleton_keys = [
        key for key, _payload in session.rows
        if "skeleton" in key and "edges" not in key
    ]
    assert skeleton_keys == ["manus/raw_skeleton/left_hand"]
    payload = session.rows[-2][1]
    assert payload["source_monotonic_ns"] == 1_000_000_007
    assert len(payload["node_quaternions_wxyz"]) == 25
