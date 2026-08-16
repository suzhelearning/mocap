"""Manus 消息解析/校验 + 骨骼拓扑。

线格式事实来源:manus/zenoh_pub.py
- JSON: {glove_id, side, seq, source_monotonic_ns, sdk_publish_time,
         nodes: 25×[x,y,z], node_quaternions_wxyz: 25×[w,x,y,z]}
- binary: ``MNS1`` 头 + seq/source/sdk 时间 + 75 位置 float32 + 100 旋转 float32
- manus/skeleton_edges/<hand>:JSON {glove_id, edges:[child,parent,chainType]}

<hand> ∈ {left_hand, right_hand}；模块内部归一化为 left/right。
节点 0 为手掌 root。位置与旋转均为 Manus 局部骨架系；统一时间轴阶段
先在此局部系插值，再与同一物理时刻的 Motive 手腕位姿拼接。
"""
from __future__ import annotations

import json
import math
import struct

NODE_COUNT = 25
CHAIN_PALM = 13
MAX_EDGES = 64                # 骨骼边数量上限(正常 ~30 条;防不受信注入超大列表)
BINARY_MAGIC = b"MNS1"
BINARY_FORMAT = "<4sQQQ175f"
BINARY_SIZE = struct.calcsize(BINARY_FORMAT)

# Manus 手骨架 topic(左手/右手独立流,类似 ROS 双 topic)
MANUS_RAW_KEYS = ("manus/raw_skeleton/left_hand", "manus/raw_skeleton/right_hand")
MANUS_EDGE_KEYS = ("manus/skeleton_edges/left_hand", "manus/skeleton_edges/right_hand")
MANUS_KEYS = MANUS_RAW_KEYS + MANUS_EDGE_KEYS


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
    """把已拼接的 Manus 25 节点筛选为本地可视化所需 21 点。

    只做索引重排，不再发布/订阅第二份 Zenoh 数据。顺序同
    wuji-hand-teleop 的 _MEDIAPIPE_TO_MANUS；缺失节点补零。
    """
    out = []
    for idx in MEDIAPIPE_FROM_MANUS:
        p = nodes[idx] if idx < len(nodes) else [0.0, 0.0, 0.0]
        out.append([p[0], p[1], p[2]])
    return out



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
    """解码并严格校验一条 raw_skeleton 位姿消息。"""
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
        quaternions = msg.get("node_quaternions_wxyz")
        if not (isinstance(nodes, list) and len(nodes) == NODE_COUNT
                and all(isinstance(n, list) and len(n) == 3 for n in nodes)):
            raise ManusError(f"nodes 须为 {NODE_COUNT}×3 数组")
        if not (isinstance(quaternions, list) and len(quaternions) == NODE_COUNT
                and all(isinstance(q, list) and len(q) == 4 for q in quaternions)):
            raise ManusError(
                f"node_quaternions_wxyz 须为 {NODE_COUNT}×4 数组"
            )
        values = [v for row in nodes for v in row] + [
            v for row in quaternions for v in row
        ]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in values):
            raise ManusError("节点位姿含非数值或非有限值")
        for field in ("seq", "source_monotonic_ns", "sdk_publish_time"):
            value = msg.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ManusError(f"{field} 须为非负整数")
        quat_array = [
            [float(v) for v in quaternion] for quaternion in quaternions
        ]
        if any(sum(v * v for v in quaternion) <= 1e-12
               for quaternion in quat_array):
            raise ManusError("node_quaternions_wxyz 含零范数四元数")
        msg["nodes"] = [[float(v) for v in node] for node in nodes]
        msg["node_quaternions_wxyz"] = quat_array
        return msg

    data = bytes(payload)
    if len(data) != BINARY_SIZE:
        raise ManusError(f"二进制位姿须为 {BINARY_SIZE} 字节，实际 {len(data)}")
    unpacked = struct.unpack(BINARY_FORMAT, data)
    if unpacked[0] != BINARY_MAGIC:
        raise ManusError("二进制位姿 magic 不是 MNS1")
    seq, source_monotonic_ns, sdk_publish_time = unpacked[1:4]
    values = unpacked[4:]
    nodes_flat = values[:NODE_COUNT * 3]
    quats_flat = values[NODE_COUNT * 3:]
    if not all(math.isfinite(v) for v in values):
        raise ManusError("二进制节点位姿含非有限值")
    nodes = [list(nodes_flat[i:i + 3]) for i in range(0, len(nodes_flat), 3)]
    quaternions = [
        list(quats_flat[i:i + 4]) for i in range(0, len(quats_flat), 4)
    ]
    if any(sum(v * v for v in quaternion) <= 1e-12 for quaternion in quaternions):
        raise ManusError("二进制节点旋转含零范数四元数")
    return {
        "glove_id": None,
        "side": None,
        "seq": int(seq),
        "source_monotonic_ns": int(source_monotonic_ns),
        "sdk_publish_time": int(sdk_publish_time),
        "nodes": nodes,
        "node_quaternions_wxyz": quaternions,
    }


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
