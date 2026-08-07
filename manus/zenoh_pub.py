#!/usr/bin/env python3
"""zenoh_pub.py — Manus raw 骨架 → Zenoh 发布器

从 stdin 读取 rawviz 协议（HAND/EDGE/FRAME/POS），发布到 Zenoh：

  manus/raw_skeleton/<side>        每帧 25 节点位置（JSON 或 --binary 紧凑二进制）
  manus/skeleton_edges/<side>      骨骼连接（child,parent,chainType），一次性

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


class ZenohPublisher:
    def __init__(self, binary=False):
        self.binary = binary
        self.hands = {}
        self.seq = 0
        self.edges_sent = set()

    def put(self, session, key, obj):
        if self.binary and isinstance(obj, dict) and "nodes" in obj:
            # 紧凑格式: 75 个 float32（25 节点 × xyz）
            payload = struct.pack("<75f", *[v for p in obj["nodes"] for v in p])
        else:
            payload = json.dumps(obj)
        session.put(key, payload)

    def handle_line(self, session, line):
        parts = line.split()
        if not parts:
            return
        tag = parts[0]

        if tag == "HAND" and len(parts) >= 4:
            gid, side, n = parts[1], parts[2], int(parts[3])
            self.hands[gid] = {"side": side.lower(), "node_count": n,
                               "edges": [], "pos": None}

        elif tag == "EDGE" and len(parts) == 5:
            h = self.hands.get(parts[1])
            if h:
                h["edges"].append([int(parts[2]) - 1, int(parts[3]) - 1,
                                   int(parts[4])])

        elif tag == "POS" and len(parts) >= 4:
            h = self.hands.get(parts[1])
            if h:
                vals = [float(v) for v in parts[2:2 + h["node_count"] * 3]]
                if len(vals) == h["node_count"] * 3:
                    h["pos"] = [vals[i:i + 3]
                                for i in range(0, len(vals), 3)]

        elif tag == "FRAME":
            self.seq += 1
            for gid, h in self.hands.items():
                if h["pos"] is None:
                    continue
                self.put(session, f"manus/raw_skeleton/{h['side']}", {
                    "glove_id": gid,
                    "side": h["side"],
                    "seq": self.seq,
                    "nodes": h["pos"],
                })
                # 拓扑每 ~10s 重发一次,让后启动的订阅者(采集程序)也能收到;
                # 第一次发布时打印日志
                if gid not in self.edges_sent:
                    self.put(session, f"manus/skeleton_edges/{h['side']}", {
                        "glove_id": gid,
                        "edges": h["edges"],
                    })
                    self.edges_sent.add(gid)
                    print(f"[pub] 边信息已发布: manus/skeleton_edges/{h['side']} "
                          f"({len(h['edges'])} 条)", file=sys.stderr)
                elif self.seq % 300 == 0:
                    self.put(session, f"manus/skeleton_edges/{h['side']}", {
                        "glove_id": gid,
                        "edges": h["edges"],
                    })


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
        cfg.insert_json5("connect/endpoints", f'["{args.router}"]')

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
