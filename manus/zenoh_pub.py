#!/usr/bin/env python3
"""zenoh_pub.py — Manus raw 骨架 → Zenoh 发布器

从 stdin 读取 rawviz 协议（HAND/EDGE/POS，左右手独立流），发布到 Zenoh：

  manus/raw_skeleton/<side>        每手套独立流:每帧 25 节点位置（JSON 或 --binary）
  manus/skeleton_edges/<side>      骨骼连接（child,parent,chainType），一次性

左右手完全解耦（类似 ROS 双 topic）:每只手独立 seq、独立发布,
一只遮挡/无数据不影响另一只。

用法:
    ./rawviz.out | python zenoh_pub.py                  # JSON（默认，跨语言易解析）
    ./rawviz.out | python zenoh_pub.py --binary         # float32 二进制（300B/帧）
    ./rawviz.out | python zenoh_pub.py --router tcp/localhost:7447
                                                        # 通过 zenohd 路由器（跨网络）

订阅验证:  python zenoh_sub.py
"""

import argparse
import json
import struct
import sys

import zenoh


MAX_NODE_COUNT = 64              # rawviz 上报节点数上限(正常 25),防畸形设备数据
EDGE_REPUBLISH_PERIOD = 300      # 拓扑重发周期(seq 差值),让后启动的订阅者也能收到

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


def manus_to_mediapipe(nodes):
    """Manus 25 节点 → 21 点:仅按索引筛选,坐标不变(与 raw_skeleton 同一坐标系,
    保证 21 点与 25 点位置对齐)。

    索引顺序与 wuji-hand-teleop 的 _MEDIAPIPE_TO_MANUS 一致(MediaPipe/MANO FK
    重排顺序);wuji 的 y 取反是为其 MediaPipe 消费端约定,此处不做——
    消费端需要时自行变换。索引越界/缺失节点置 [0,0,0]。
    """
    out = []
    for idx in MEDIAPIPE_FROM_MANUS:
        p = nodes[idx] if idx < len(nodes) else [0.0, 0.0, 0.0]
        out.append([p[0], p[1], p[2]])
    return out


class ZenohPublisher:
    def __init__(self, binary=False):
        self.binary = binary
        self.hands = {}                # gloveId -> {side, node_count, edges, last_seq}
        self.edges_sent = set()

    def put(self, session, key, obj):
        if self.binary and isinstance(obj, dict) and "nodes" in obj:
            vals = [v for p in obj["nodes"] for v in p]
            if len(vals) == 75:        # 25 节点契约;其他节点数回退 JSON
                payload = struct.pack("<75f", *vals)
            else:
                payload = json.dumps(obj)
        else:
            payload = json.dumps(obj)
        session.put(key, payload)

    def _send_edges(self, session, gid, h):
        """发布一次骨骼拓扑(首见/回绕/周期性)。"""
        self.put(session, f"manus/skeleton_edges/{h['side']}", {
            "glove_id": gid,
            "edges": h["edges"],
        })
        print(f"[pub] 边信息已发布: manus/skeleton_edges/{h['side']} "
              f"({len(h['edges'])} 条)", file=sys.stderr)

    def handle_line(self, session, line):
        parts = line.split()
        if not parts:
            return
        tag = parts[0]

        if tag == "HAND" and len(parts) >= 4:
            try:
                n = int(parts[3])
            except ValueError:
                print(f"[pub] 坏 HAND 行(非数字节点数),跳过: {line.strip()}",
                      file=sys.stderr)
                return
            if not (1 <= n <= MAX_NODE_COUNT):
                print(f"[pub] HAND 节点数非法({n}),跳过: {line.strip()}", file=sys.stderr)
                return
            gid, side = parts[1], parts[2]
            # topic 命名:left_hand/right_hand(Manus 侧),与 Motive 的
            # left_wrist/right_wrist(手腕刚体)区分
            side = side.lower()
            if side == "left":
                side = "left_hand"
            elif side == "right":
                side = "right_hand"
            self.hands[gid] = {"side": side, "node_count": n,
                               "edges": [], "last_seq": 0}

        elif tag == "EDGE" and len(parts) == 5:
            h = self.hands.get(parts[1])
            if h:
                try:
                    child, parent, chain = (int(parts[2]) - 1,
                                            int(parts[3]) - 1, int(parts[4]))
                except ValueError:
                    print(f"[pub] 坏 EDGE 行,跳过: {line.strip()}", file=sys.stderr)
                    return
                if 0 <= child < h["node_count"] and 0 <= parent < h["node_count"]:
                    h["edges"].append([child, parent, chain])

        elif tag == "POS" and len(parts) >= 5:
            # POS <gloveId> <seq> <x0 y0 z0 ...>:每手套独立流,收到即发布
            gid = parts[1]
            h = self.hands.get(gid)
            if h:
                try:
                    seq = int(parts[2])
                    vals = [float(v) for v in parts[3:3 + h["node_count"] * 3]]
                except ValueError:
                    print(f"[pub] 坏 POS 行(非数字),跳过: {line.strip()}", file=sys.stderr)
                    return
                if len(vals) == h["node_count"] * 3:
                    nodes = [vals[i:i + 3] for i in range(0, len(vals), 3)]
                    self.put(session, f"manus/raw_skeleton/{h['side']}", {
                        "glove_id": gid,
                        "side": h["side"],
                        "seq": seq,
                        "nodes": nodes,
                    })
                    # MANO 兼容 21 点(新格式):保留原始 25 点之外另行发布。
                    # key: manus/mano_skeleton/{left,right}_hand
                    # 顺序 = MediaPipe 约定 = MANO FK 重排输出(manopth
                    # reorder [0,13..16,1..3,17,4..6,18,10..12,19,7..9,20]):
                    # 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
                    self.put(session, f"manus/mano_skeleton/{h['side']}", {
                        "glove_id": gid,
                        "side": h["side"],
                        "seq": seq,
                        "keypoints": manus_to_mediapipe(nodes),
                    })
                    # 拓扑重发:首见立即发;seq 回绕/rawviz 重启(seq 后退)立即发;
                    # 否则每 ~EDGE_REPUBLISH_PERIOD 帧重发一次
                    if gid not in self.edges_sent:
                        self._send_edges(session, gid, h)
                        self.edges_sent.add(gid)
                    elif seq < h["last_seq"] or seq - h["last_seq"] >= EDGE_REPUBLISH_PERIOD:
                        self._send_edges(session, gid, h)
                    h["last_seq"] = seq


def main():
    ap = argparse.ArgumentParser(description="Manus raw 骨架 → Zenoh")
    ap.add_argument("--binary", action="store_true",
                    help="节点用 float32 二进制发布（默认 JSON）")
    ap.add_argument("--router", type=str, default=None,
                    help="zenohd 路由器地址，如 tcp/localhost:7447（默认 peer 模式）")
    ap.add_argument("--config", type=str, default=None,
                    help="zenoh 配置文件路径（可选）")
    args = ap.parse_args()

    cfg = zenoh.Config()
    if args.config:
        cfg = zenoh.Config.from_file(args.config)
    if args.router:
        cfg.insert_json5("connect/endpoints", json.dumps([args.router]))

    session = zenoh.open(cfg)
    pub = ZenohPublisher(binary=args.binary)
    print(f"[pub] Zenoh 会话已打开（{'二进制' if args.binary else 'JSON'} 模式）"
          f"，等待 rawviz 数据...", file=sys.stderr)
    try:
        for line in sys.stdin:
            pub.handle_line(session, line.strip())
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        print("\n[pub] 已关闭", file=sys.stderr)


if __name__ == "__main__":
    main()
