#!/usr/bin/env python3
"""zenoh_sub.py — 订阅验证 Manus 骨架数据（Zenoh）

用法:
    python zenoh_sub.py                    # 订阅 manus/** 打印每帧概要
    python zenoh_sub.py --verbose          # 打印节点坐标详情
    python zenoh_sub.py --raw              # 打印原始 payload（JSON 或二进制字节）
    python zenoh_sub.py --router tcp/localhost:7447   # 通过 zenohd 路由器
"""

import argparse
import json
import struct
import sys
import time

import zenoh


def main():
    ap = argparse.ArgumentParser(description="订阅 Manus 骨架 (Zenoh)")
    ap.add_argument("--key", type=str, default="manus/**",
                    help="订阅的 key 表达式（默认 manus/**）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--router", type=str, default=None)
    ap.add_argument("--duration", type=float, default=None,
                    help="订阅时长（秒），默认持续到 Ctrl+C")
    args = ap.parse_args()

    cfg = zenoh.Config()
    if args.router:
        cfg.insert_json5("connect/endpoints", json.dumps([args.router]))

    session = zenoh.open(cfg)
    counts = {}
    t0 = time.time()

    def on_sample(sample):
        key = sample.key_expr
        payload = sample.payload.to_bytes()
        counts[key] = counts.get(key, 0) + 1

        if args.raw:
            print(f"{key}: {payload}")
            return

        try:
            data = json.loads(payload)
            if "nodes" in data:
                nodes = data["nodes"]
                n = len(nodes)
                p0 = nodes[0] if n else []
                print(f"[帧 {data.get('seq', '?')}] {key}: {n} 节点, "
                      f"根节点({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})")
            elif "edges" in data:
                print(f"[{key}]: {len(data['edges'])} 条骨骼边 "
                      f"(child,parent,chainType)")
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 二进制 payload: 75 float32
            try:
                vals = struct.unpack("<75f", payload)
                print(f"[{key}]: 二进制 {len(payload)} 字节, "
                      f"首节点({vals[0]:+.3f}, {vals[1]:+.3f}, {vals[2]:+.3f})")
            except struct.error:
                print(f"[{key}]: {len(payload)} 字节（无法解析）")

        if args.verbose and "nodes" in locals():
            if isinstance(locals().get("nodes"), list):
                for i, p in enumerate(nodes):
                    print(f"   节点{i+1:2d}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    sub = session.declare_subscriber(args.key, on_sample)
    print(f"[sub] 订阅 {args.key}，Ctrl+C 退出", file=sys.stderr)

    try:
        end = time.time() + args.duration if args.duration else None
        while True:
            if end and time.time() >= end:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - t0
        print(f"\n[sub] 汇总（{elapsed:.1f}s）:")
        for key, c in sorted(counts.items()):
            print(f"  {key}: {c} 条, {c/elapsed:.1f} 条/秒")
        sub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
