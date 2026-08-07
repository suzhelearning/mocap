"""Manus 消息解析/校验 + 骨骼拓扑。

线格式事实来源:manus/zenoh_pub.py
- manus/raw_skeleton/<side>   JSON {glove_id, side, seq, nodes: 25×[x,y,z]}
                             或 --binary: 75 个 little-endian float32(25 节点 × xyz)
- manus/skeleton_edges/<side> JSON {glove_id, edges: [child, parent, chainType]}(0-based)

节点语义(manus/viz.py):节点 0 为手掌 root(chainType 13);手指链
5=拇指 6=食指 7=中指 8=无名指 9=小指。消息无时间戳 → 采集端打点。
"""

from __future__ import annotations

import json
import struct

NODE_COUNT = 25
CHAIN_PALM = 13

MANUS_RAW_KEYS = ("manus/raw_skeleton/left", "manus/raw_skeleton/right")
MANUS_EDGE_KEYS = ("manus/skeleton_edges/left", "manus/skeleton_edges/right")
MANUS_KEYS = MANUS_RAW_KEYS + MANUS_EDGE_KEYS


class ManusError(ValueError):
    """manus 消息格式不合法。"""


def _side_from_key(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def decode_manus(payload: bytes | str) -> dict:
    """解码一条 raw_skeleton 消息(JSON 或 --binary 二进制),校验后返回 dict。

    返回 {glove_id, side, seq, nodes: list[list[float]]}。失败抛 ManusError。
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError:
        text = None

    if text is not None and text.lstrip().startswith("{"):
        try:
            msg = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManusError(f"raw_skeleton JSON 解析失败: {exc}") from exc
        nodes = msg.get("nodes")
        if not (isinstance(nodes, list) and len(nodes) == NODE_COUNT
                and all(isinstance(n, list) and len(n) == 3 for n in nodes)):
            raise ManusError(f"nodes 须为 {NODE_COUNT}×3 数组")
        if not all(isinstance(v, (int, float)) for n in nodes for v in n):
            raise ManusError("nodes 含非数值")
        msg["nodes"] = [[float(v) for v in n] for n in nodes]
        return msg

    # 二进制:75 个 float32 little-endian
    data = bytes(payload)
    if len(data) != NODE_COUNT * 3 * 4:
        raise ManusError(f"二进制节点须为 {NODE_COUNT * 3 * 4} 字节,实际 {len(data)}")
    vals = struct.unpack(f"<{NODE_COUNT * 3}f", data)
    nodes = [list(vals[i:i + 3]) for i in range(0, len(vals), 3)]
    return {"glove_id": None, "side": None, "seq": None, "nodes": nodes}


def parse_edges(payload: bytes | str) -> list[tuple[int, int, int]]:
    """解析 skeleton_edges 消息,返回 [(child, parent, chainType)] 列表。"""
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        msg = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManusError(f"edges JSON 解析失败: {exc}") from exc
    edges = msg.get("edges")
    if not isinstance(edges, list):
        raise ManusError("edges 须为列表")
    result = []
    for e in edges:
        if not (isinstance(e, list) and len(e) == 3
                and all(isinstance(v, int) for v in e)):
            raise ManusError(f"边 {e} 须为 [child, parent, chainType]")
        result.append(tuple(e))
    return result


def palm_node_index(edges: list[tuple[int, int, int]]) -> int:
    """手掌 root 节点索引:chainType==13 的边 child;无则 0(与 viz.py 约定一致)。"""
    for child, _parent, chain in edges:
        if chain == CHAIN_PALM:
            return child
    return 0
