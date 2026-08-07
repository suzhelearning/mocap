#!/usr/bin/env python3
"""demo_manus_pub.py — 合成 Manus 骨架发布器(无手套时本地验证)

消息格式与 manus/zenoh_pub.py 完全同构:
  manus/raw_skeleton/{left,right}   JSON {glove_id, side, seq, nodes: 25×[x,y,z]}
  manus/skeleton_edges/{left,right} JSON {glove_id, edges: [child,parent,chain]}

节点布局(0=手掌 root):
  1..4 拇指(chain 5)  5..8 食指(6)  9..12 中指(7)  13..16 无名指(8)
  17..20 小指(9)  21..24 手掌(13)
手指沿本地 +z 伸出;指尖以正弦摆动模拟弯曲;手掌 root 小幅平移。

用法: pixi run demo-manus   (连 router,发布 tcp/127.0.0.1:7447)
"""

from __future__ import annotations

import json
import math
import time

import zenoh

FPS = 30.0
ROUTER = "tcp/127.0.0.1:7447"

# 每根手指的 4 个节点索引(手掌 root 之后)
FINGER_NODES = {
    5: list(range(1, 5)),      # 拇指
    6: list(range(5, 9)),      # 食指
    7: list(range(9, 13)),     # 中指
    8: list(range(13, 17)),    # 无名指
    9: list(range(17, 21)),    # 小指
}
PALM_NODES = list(range(21, 25))

CHAIN_PER_NODE: dict[int, int] = {}
for chain, nodes in FINGER_NODES.items():
    for n in nodes:
        CHAIN_PER_NODE[n] = chain
for n in PALM_NODES:
    CHAIN_PER_NODE[n] = 13
CHAIN_PER_NODE[0] = 13

# edges:每根手指 3 条(0→f0, f0→f1, f1→f2, f2→f3 → 4 条),手掌节点连 root
EDGES: list[list[int]] = []
for chain, nodes in FINGER_NODES.items():
    prev = 0
    for n in nodes:
        EDGES.append([n, prev, chain])
        prev = n
for n in PALM_NODES:
    EDGES.append([n, 0, 13])


def make_hand_frame(seq: int, t: float, side: str) -> dict:
    nodes = [[0.0, 0.0, 0.0] for _ in range(25)]
    # 手掌 root:小幅平移(左右手反向)
    sign = -1.0 if side == "left" else 1.0
    palm = [0.04 * sign * math.sin(t * 0.7), 0.01 * math.sin(t * 1.1), 0.0]
    nodes[0] = palm
    for n in PALM_NODES:
        k = n - PALM_NODES[0]
        nodes[n] = [palm[0] + 0.015 * k * sign, palm[1] + 0.012 * math.sin(k), 0.005 * k]
    # 手指:沿本地 +z 伸出,指尖弯曲摆动
    for chain, idxs in FINGER_NODES.items():
        amp = 0.04 if chain == 5 else 0.06        # 拇指幅度小
        phase = t * (2.2 if chain % 2 else 1.8) + chain * 0.6
        for k, n in enumerate(idxs):
            z = 0.015 + 0.022 * k
            bend = amp * math.sin(phase) * (k / 3.0) ** 2
            spread = 0.008 * (chain - 5) * sign
            nodes[n] = [palm[0] + spread, palm[1] + bend, palm[2] + z]
    return {"glove_id": f"demo{side[0]}", "side": side, "seq": seq, "nodes": nodes}


def main() -> int:
    config = zenoh.Config.from_json5(
        json.dumps({"mode": "client",
                    "connect": {"endpoints": [ROUTER]}})
    )
    seq = 0
    t0 = time.time()
    with zenoh.open(config) as session:
        print(f"[demo-manus] 发布合成骨架 @ {FPS:.0f}Hz → {ROUTER}(Ctrl-C 停止)", flush=True)
        while True:
            t = time.time() - t0
            seq += 1
            if seq % 150 == 1:                     # 每 ~5s 重发拓扑,让后订阅者能收到
                for side in ("left", "right"):
                    session.put(f"manus/skeleton_edges/{side}", json.dumps(
                        {"glove_id": f"demo{side[0]}", "edges": EDGES}))
            for side in ("left", "right"):
                session.put(f"manus/raw_skeleton/{side}", json.dumps(
                    make_hand_frame(seq, t, side)))
            time.sleep(1.0 / FPS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[demo-manus] 已停止")
