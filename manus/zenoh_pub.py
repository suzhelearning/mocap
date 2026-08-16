#!/usr/bin/env python3
"""zenoh_pub.py — Manus raw 骨架 → Zenoh 发布器。

从 stdin 读取 rawviz 的 HAND/EDGE/POSE 协议。每个 POSE 同时携带：
- 25 个局部节点位置；
- 25 个局部节点四元数（wxyz）；
- rawviz SDK 回调入口的 Linux monotonic 纳秒；
- Manus SDK 原始 publishTime。

发布：
  manus/raw_skeleton/<side>        每手套独立位姿流（JSON 或 --binary）
  manus/skeleton_edges/<side>      骨骼连接（child,parent,chainType）

用法:
    ./rawviz.out | python zenoh_pub.py
    ./rawviz.out | python zenoh_pub.py --binary
    ./rawviz.out | python zenoh_pub.py --router tcp/localhost:7447
"""


import argparse
import json
import struct
import sys

import zenoh


MAX_NODE_COUNT = 64              # rawviz 上报节点数上限(正常 25),防畸形设备数据
EDGE_REPUBLISH_PERIOD = 300      # 拓扑重发周期(seq 差值),让后启动的订阅者也能收到
BINARY_MAGIC = b"MNS1"



class ZenohPublisher:
    def __init__(self, binary=False):
        self.binary = binary
        # gloveId -> {side,node_count,edges,last_seq,last_edges_seq}
        self.hands = {}

    def put(self, session, key, obj):
        if self.binary and isinstance(obj, dict) and "nodes" in obj:
            nodes = [v for p in obj["nodes"] for v in p]
            rotations = [v for q in obj["node_quaternions_wxyz"] for v in q]
            if len(nodes) == 75 and len(rotations) == 100:
                payload = struct.pack(
                    "<4sQQQ175f",
                    BINARY_MAGIC,
                    int(obj["seq"]),
                    int(obj["source_monotonic_ns"]),
                    int(obj["sdk_publish_time"]),
                    *(nodes + rotations),
                )
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
            self.hands[gid] = {
                "side": side,
                "node_count": n,
                "edges": [],
                "last_seq": None,
                "last_edges_seq": None,
            }

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

        elif tag == "POSE" and len(parts) >= 5:
            # POSE <gid> <seq> <source_monotonic_ns> <sdk_publish_time>
            #      25 × <x y z qw qx qy qz>
            gid = parts[1]
            h = self.hands.get(gid)
            if h:
                try:
                    seq = int(parts[2])
                    source_monotonic_ns = int(parts[3])
                    sdk_publish_time = int(parts[4])
                    values = [
                        float(v) for v in parts[5:5 + h["node_count"] * 7]
                    ]
                except ValueError:
                    print(f"[pub] 坏 POSE 行(非数字),跳过: {line.strip()}",
                          file=sys.stderr)
                    return
                if len(values) == h["node_count"] * 7:
                    rows = [
                        values[i:i + 7] for i in range(0, len(values), 7)
                    ]
                    self.put(session, f"manus/raw_skeleton/{h['side']}", {
                        "glove_id": gid,
                        "side": h["side"],
                        "seq": seq,
                        "source_monotonic_ns": source_monotonic_ns,
                        "sdk_publish_time": sdk_publish_time,
                        "nodes": [row[:3] for row in rows],
                        "node_quaternions_wxyz": [row[3:] for row in rows],
                    })
                    last_seq = h["last_seq"]
                    last_edges_seq = h["last_edges_seq"]
                    wrapped = last_seq is not None and seq < last_seq
                    due = (last_edges_seq is None
                           or seq - last_edges_seq >= EDGE_REPUBLISH_PERIOD)
                    if wrapped or due:
                        self._send_edges(session, gid, h)
                        h["last_edges_seq"] = seq
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
