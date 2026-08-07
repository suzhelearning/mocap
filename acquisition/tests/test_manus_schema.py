"""manus_schema.py 解析测试。"""

from __future__ import annotations

import json
import struct

import pytest

from acquisition.manus_schema import (
    ManusError,
    decode_manus,
    palm_node_index,
    parse_edges,
)


def _raw_msg(nodes=None):
    if nodes is None:
        nodes = [[float(i), 0.0, 0.0] for i in range(25)]
    return {"glove_id": "aabbccdd", "side": "left", "seq": 42, "nodes": nodes}


def test_decode_manus_json():
    payload = json.dumps(_raw_msg())
    msg = decode_manus(payload)
    assert msg["side"] == "left"
    assert msg["seq"] == 42
    assert len(msg["nodes"]) == 25
    assert all(len(n) == 3 for n in msg["nodes"])


def test_decode_manus_binary():
    vals = [float(i) for i in range(75)]
    payload = struct.pack(f"<{75}f", *vals)
    msg = decode_manus(payload)
    assert msg["nodes"][1][0] == 3.0
    assert len(msg["nodes"]) == 25


def test_decode_manus_binary_wrong_length():
    with pytest.raises(ManusError, match="二进制节点须为"):
        decode_manus(struct.pack("<3f", 1.0, 2.0, 3.0))


def test_decode_manus_bad_nodes():
    msg = _raw_msg(nodes=[[0.0, 0.0, 0.0]])   # 少于 25 节点
    with pytest.raises(ManusError, match="nodes 须为 25×3"):
        decode_manus(json.dumps(msg))


def test_decode_manus_bytes_input():
    msg = decode_manus(json.dumps(_raw_msg()).encode())
    assert msg["seq"] == 42


def test_parse_edges():
    payload = json.dumps({"glove_id": "aabbccdd",
                          "edges": [[1, 0, 13], [2, 1, 6], [5, 0, 5]]})
    edges = parse_edges(payload)
    assert edges == [(1, 0, 13), (2, 1, 6), (5, 0, 5)]


def test_parse_edges_bad():
    with pytest.raises(ManusError, match="JSON 解析失败"):
        parse_edges("not json")


def test_palm_node_index():
    edges = [(1, 0, 6), (5, 0, 13), (2, 0, 5)]
    assert palm_node_index(edges) == 5
    assert palm_node_index([(1, 0, 6)]) == 0      # 无手掌边 → 默认 0
