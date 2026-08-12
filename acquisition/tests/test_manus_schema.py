"""manus_schema.py 解析测试。"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from acquisition.manus_schema import (
    MEDIAPIPE_FROM_MANUS,
    ManusError,
    decode_manus,
    manus_to_mediapipe,
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
    assert msg["seq"] == -1              # 二进制模式无 seq 来源,给默认值


def test_decode_manus_binary_wrong_length():
    with pytest.raises(ManusError, match="二进制节点须为"):
        decode_manus(struct.pack("<3f", 1.0, 2.0, 3.0))


def test_decode_manus_rejects_non_finite_json():
    """NaN/Inf 不得进入拼接与 HDF5(与 mocap schema 的有限性校验对齐)。"""
    msg = _raw_msg()
    msg["nodes"][0][0] = float("nan")
    with pytest.raises(ManusError, match="非有限"):
        decode_manus(json.dumps(msg))


def test_decode_manus_rejects_non_finite_binary():
    vals = [float(i) for i in range(75)]
    vals[0] = float("inf")
    with pytest.raises(ManusError, match="非有限"):
        decode_manus(struct.pack(f"<{75}f", *vals))


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


def test_parse_edges_too_many():
    """边数量超上限拒绝(防不受信注入超大列表的渲染/内存放大)。"""
    edges = [[i % 24, 0, 13] for i in range(1, 70)]     # 69 条 > 64,索引均在界内
    with pytest.raises(ManusError, match="超上限"):
        parse_edges(json.dumps({"edges": edges}))


def test_parse_edges_index_out_of_range():
    with pytest.raises(ManusError, match="越界"):
        parse_edges(json.dumps({"edges": [[0, 25, 13]]}))
    with pytest.raises(ManusError, match="越界"):
        parse_edges(json.dumps({"edges": [[-1, 0, 13]]}))


def test_palm_node_index():
    """锚点约定固定为节点 0(与 viz.py 一致;chain 语义不可靠,见模块注释)。"""
    edges = [(1, 0, 6), (5, 0, 13), (2, 0, 5)]
    assert palm_node_index(edges) == 0
    assert palm_node_index([(1, 0, 6)]) == 0
    assert palm_node_index([]) == 0


# -- MediaPipe 21 点转换(与 wuji-hand-teleop 一致) -------------------------

def test_manus_to_mediapipe_mapping():
    """25 节点 → 21 点:仅筛选索引,坐标不变(与 raw_skeleton 对齐)。"""
    nodes = [[float(i), float(i), float(i)] for i in range(25)]
    kp = manus_to_mediapipe(nodes)
    assert len(kp) == 21
    assert kp[0] == [0.0, 0.0, 0.0]         # mp0 wrist ← idx0(手掌 root)
    assert kp[1] == [21.0, 21.0, 21.0]      # mp1 拇指 ← idx21
    assert kp[4] == [24.0, 24.0, 24.0]      # mp4 拇指尖 ← idx24
    assert kp[5] == [2.0, 2.0, 2.0]         # mp5 食指 ← idx2(跳过掌骨起点 1)
    assert kp[8] == [5.0, 5.0, 5.0]         # mp8 食指尖 ← idx5
    assert kp[9] == [7.0, 7.0, 7.0]         # mp9 中指 ← idx7
    assert kp[13] == [12.0, 12.0, 12.0]     # mp13 无名指 ← idx12
    assert kp[17] == [17.0, 17.0, 17.0]     # mp17 小指 ← idx17
    assert kp[20] == [20.0, 20.0, 20.0]     # mp20 小指尖 ← idx20
    # 关键:同一索引的点与 raw 完全一致(不 y 取反)
    raw = np.asarray(nodes)
    for mp_i, raw_i in enumerate(MEDIAPIPE_FROM_MANUS):
        assert kp[mp_i] == raw[raw_i].tolist()


def test_manus_to_mediapipe_matches_wuji_implementation():
    """与 wuji-hand-teleop 的 _MEDIAPIPE_TO_MANUS 一致(wuji 为 1-based,减 1 换算)。"""
    # 来源: /home/current/syz/wuji-hand-teleop/src/controller/controller/wujihand_node.py
    wuji_mapping_1based = (
        1, 22, 23, 24, 25,
        3, 4, 5, 6,
        8, 9, 10, 11,
        13, 14, 15, 16,
        18, 19, 20, 21,
    )
    assert MEDIAPIPE_FROM_MANUS == tuple(v - 1 for v in wuji_mapping_1based)


def test_manus_to_mediapipe_short_input_padded():
    """节点不足 25 时缺失位置补零,不越界。"""
    nodes = [[0.0, 1.0, 0.0] for _ in range(10)]
    kp = manus_to_mediapipe(nodes)
    assert len(kp) == 21
    assert kp[0] == [0.0, 1.0, 0.0]         # idx0 在范围内,坐标不变
    assert kp[4] == [0.0, 0.0, 0.0]         # idx24 缺失 → 补零


