"""Manus 消息解析/校验 + 骨骼拓扑。

线格式事实来源:manus/zenoh_pub.py
- manus/raw_skeleton/<hand>   JSON {glove_id, side, seq, nodes: 25×[x,y,z]}
                             或 --binary: 75 个 little-endian float32(25 节点 × xyz)
- manus/skeleton_edges/<hand> JSON {glove_id, edges: [child, parent, chainType]}(0-based)
<hand> ∈ {left_hand, right_hand}:Manus 侧命名,与 Motive 的
left_wrist/right_wrist(手腕刚体)区分——本模块内部统一归一化为 left/right。

节点语义(manus/viz.py):节点 0 为手掌 root(chainType 13);手指链
5=拇指 6=食指 7=中指 8=无名指 9=小指。消息无时间戳 → 采集端打点。
"""

from __future__ import annotations

import json
import math
import struct

NODE_COUNT = 25
CHAIN_PALM = 13
MAX_EDGES = 64                # 骨骼边数量上限(正常 ~30 条;防不受信注入超大列表)

# Manus 手骨架 topic(左手/右手独立流,类似 ROS 双 topic)
MANUS_RAW_KEYS = ("manus/raw_skeleton/left_hand", "manus/raw_skeleton/right_hand")
MANUS_EDGE_KEYS = ("manus/skeleton_edges/left_hand", "manus/skeleton_edges/right_hand")
MANUS_KEYS = MANUS_RAW_KEYS + MANUS_EDGE_KEYS

# MANO/MediaPipe 21 点发布 topic(由 zenoh_pub 对同帧额外发布,原始 25 点保留)。
# 顺序与 MANO FK 重排输出一致(manopth reorder),即 MediaPipe 约定。
MEDIAPIPE_KEYS = ("manus/mano_skeleton/left_hand", "manus/mano_skeleton/right_hand")

# MediaPipe 21 点 ← Manus 25 节点(数组索引 0-based;与 wuji-hand-teleop 的
# _MEDIAPIPE_TO_MANUS 一致——wuji 的 node_id 为 1-based,此处已换算为数组索引)。
# 真机布局:0=手掌 root;1-5/6-10/11-15/16-20 各指(掌骨起点,掌骨,MCP,PIP,TIP);
# 21-24 拇指(4 节点)。MediaPipe: 0=WRIST, 1-4=THUMB, 5-8=INDEX, 9-12=MIDDLE,
# 13-16=RING, 17-20=PINKY
MEDIAPIPE_FROM_MANUS = (
    0,                  # mp0  wrist  ← 索引 0(手掌 root)
    21, 22, 23, 24,     # mp1-4 拇指
    2, 3, 4, 5,         # mp5-8 食指(跳过掌骨起点 1)
    7, 8, 9, 10,        # mp9-12 中指(跳过 6)
    12, 13, 14, 15,     # mp13-16 无名指(跳过 11)
    17, 18, 19, 20,     # mp17-20 小指(跳过 16)
)


def manus_to_mediapipe(nodes) -> list:
    """Manus 25 节点 → 21 点:仅按索引筛选,坐标不变(与 raw_skeleton 同一坐标系)。

    与 manus/zenoh_pub.py 的同名函数保持一致(发布端/消费端共用同一映射);
    索引顺序同 wuji-hand-teleop 的 _MEDIAPIPE_TO_MANUS(即 MediaPipe/MANO FK
    重排顺序),但不做 y 取反——保证 21 点与 25 点位置对齐。索引越界置 [0,0,0]。
    """
    out = []
    for idx in MEDIAPIPE_FROM_MANUS:
        p = nodes[idx] if idx < len(nodes) else [0.0, 0.0, 0.0]
        out.append([p[0], p[1], p[2]])
    return out


def decode_mano(payload: bytes | str) -> dict:
    """解码一条 manus/mano_skeleton 消息(21×3 keypoints,MANO/MediaPipe 顺序)。

    返回 {glove_id, side, seq, keypoints: list[list[float]]}。失败抛 ManusError。
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        msg = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManusError(f"mano JSON 解析失败: {exc}") from exc
    kp = msg.get("keypoints")
    if not (isinstance(kp, list) and len(kp) == 21
            and all(isinstance(p, list) and len(p) == 3 for p in kp)):
        raise ManusError("keypoints 须为 21×3 数组")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for p in kp for v in p):
        raise ManusError("keypoints 含非数值或非有限值")
    msg["keypoints"] = [[float(v) for v in p] for p in kp]
    return msg

# topic 后缀 → 内部 side(Hand/Left_Hand 统一为 left/right)
_SIDE_ALIASES = {"left_hand": "left", "right_hand": "right",
                 "left_skeleton": "left", "right_skeleton": "right"}


def side_from_key(key: str) -> str:
    """从 topic key 提取并归一化 side:left_hand → left,right_hand → right。"""
    suffix = key.rsplit("/", 1)[-1]
    return _SIDE_ALIASES.get(suffix, suffix)


class ManusError(ValueError):
    """manus 消息格式不合法。"""


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
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for n in nodes for v in n):
            raise ManusError("nodes 含非数值或非有限值")
        msg["nodes"] = [[float(v) for v in n] for n in nodes]
        return msg

    # 二进制:75 个 float32 little-endian
    data = bytes(payload)
    if len(data) != NODE_COUNT * 3 * 4:
        raise ManusError(f"二进制节点须为 {NODE_COUNT * 3 * 4} 字节,实际 {len(data)}")
    vals = struct.unpack(f"<{NODE_COUNT * 3}f", data)
    if not all(math.isfinite(v) for v in vals):
        raise ManusError("二进制节点含非有限值")
    nodes = [list(vals[i:i + 3]) for i in range(0, len(vals), 3)]
    return {"glove_id": None, "side": None, "seq": -1, "nodes": nodes}


def parse_edges(payload: bytes | str) -> list[tuple[int, int, int]]:
    """解析 skeleton_edges 消息,返回 [(child, parent, chainType)] 列表。

    数量与节点范围都做上限校验:不受信发布者可注入超大列表(内存/渲染放大)
    或越界节点索引(拼接时越界)。
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        msg = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManusError(f"edges JSON 解析失败: {exc}") from exc
    edges = msg.get("edges")
    if not isinstance(edges, list):
        raise ManusError("edges 须为列表")
    if len(edges) > MAX_EDGES:
        raise ManusError(f"edges 数量超上限({len(edges)} > {MAX_EDGES})")
    result = []
    for e in edges:
        if not (isinstance(e, list) and len(e) == 3
                and all(isinstance(v, int) and not isinstance(v, bool) for v in e)):
            raise ManusError(f"边 {e} 须为 [child, parent, chainType]")
        child, parent, _chain = e
        if not (0 <= child < NODE_COUNT and 0 <= parent < NODE_COUNT):
            raise ManusError(f"边 {e} 节点索引越界(节点数 {NODE_COUNT})")
        result.append(tuple(e))
    return result


def palm_node_index(edges: list[tuple[int, int, int]]) -> int:
    """手掌 root 节点索引。

    约定固定返回 0(与 viz.py 一致):真机 rawviz 对 parentId==0 的边不输出,
    合成 demo 的 chain==13 边 child 为 21..24,按 chain 语义取锚点不可靠
    (曾导致锚点偏到手掌侧节点);节点 0 是双方一致约定的手掌 root。
    """
    return 0
