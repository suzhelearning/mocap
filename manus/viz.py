#!/usr/bin/env python3
"""viz.py — Manus Metaglove raw 骨架 25 关键点 3D 可视化（Viser / 浏览器）

从 stdin 读取 rawviz 输出（HAND/EDGE/POS 独立流协议），推送到 Viser 服务器，
浏览器打开 http://<host>:<port> 实时查看。

用法:
    ./rawviz.out | python viz.py                  # 实时可视化
    python viz.py --replay skel_sample.txt        # 回放数据文件（按原节奏）
    python viz.py --replay skel_sample.txt --fast # 快速回放

参数:
    --port PORT     Viser 端口（默认 8080）
    --host HOST     监听地址（默认 127.0.0.1；局域网访问需显式指定本机 IP 并自行防护）
    --replay FILE   回放模式：从文件读数据
    --fast          回放时不按帧间隔 sleep，直接推完
"""

import argparse
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 手指链配色（按 SDK ChainType）
#   5=拇指 6=食指 7=中指 8=无名指 9=小指 13=手掌
# ---------------------------------------------------------------------------
CHAIN_COLORS = {
    5: (244, 162, 97),    # 拇指 橙
    6: (42, 157, 143),    # 食指 青
    7: (233, 196, 106),   # 中指 黄
    8: (231, 111, 81),    # 无名指 橙红
    9: (69, 123, 157),    # 小指 蓝
    13: (210, 210, 210),  # 手掌 亮灰
}
DEFAULT_COLOR = (160, 160, 160)


class Hand:
    def __init__(self, glove_id, side, node_count):
        self.glove_id = glove_id
        self.side = side
        self.node_count = node_count
        self.pos = np.zeros((node_count, 3))
        self.valid = False
        self.edges = []                # [(child_idx, parent_idx, chain)]
        self.node_color = np.full((node_count, 3), DEFAULT_COLOR, dtype=np.uint8)
        self.node_color[0] = CHAIN_COLORS[13]   # 根节点=手掌
        self.pc = None                 # viser 点云对象
        self.ls = None                 # viser 线段对象

    def add_edge(self, child, parent, chain):
        self.edges.append((child - 1, parent - 1, chain))
        self.node_color[child - 1] = CHAIN_COLORS.get(chain, DEFAULT_COLOR)

    # -- viser 场景对象 ---------------------------------------------------
    def attach_scene(self, server):
        base = f"/hand/{self.glove_id}"
        self.pc = server.scene.add_point_cloud(
            f"{base}/points",
            points=self.pos,
            colors=self.node_color,
            point_size=0.006,
        )
        if self.edges:
            seg = np.array([[self.pos[c], self.pos[p]] for c, p, _ in self.edges])
            # viser 要求 colors 为 (N, 2, 3)：每段两个端点各一色（同色）
            per_seg = np.array([CHAIN_COLORS.get(ch, DEFAULT_COLOR)
                                for _, _, ch in self.edges], dtype=np.uint8)
            colors = np.repeat(per_seg[:, None, :], 2, axis=1)
            self.ls = server.scene.add_line_segments(
                f"{base}/bones",
                points=seg,            # 形状 (N, 2, 3)
                colors=colors,
                line_width=2.0,
            )

    def update_scene(self):
        if not self.valid or self.pc is None:
            return
        self.pc.points = self.pos
        if self.ls is not None and self.edges:
            seg = np.array([[self.pos[c], self.pos[p]] for c, p, _ in self.edges])
            self.ls.points = seg


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def handle_line(line, hands):
    """返回 True 表示有新骨架数据（POSE 行）。

    POSE:gid/seq/source_monotonic/sdk_publish_time 后接每节点 xyz+四元数。
    """
    parts = line.split()
    if not parts:
        return False
    tag = parts[0]
    try:
        if tag == "HAND" and len(parts) >= 4:
            node_count = int(parts[3])
            if node_count <= 0:
                return False
            h = Hand(parts[1], parts[2], node_count)
            hands[h.glove_id] = h
        elif tag == "EDGE" and len(parts) == 5:
            h = hands.get(parts[1])
            if h:
                h.add_edge(int(parts[2]), int(parts[3]), int(parts[4]))
        elif tag == "POSE" and len(parts) >= 5:
            h = hands.get(parts[1])
            if h:
                vals = np.asarray(
                    parts[5:5 + h.node_count * 7], dtype=float,
                )
                if vals.size == h.node_count * 7:
                    h.pos = vals.reshape(h.node_count, 7)[:, :3]
                    h.valid = True
                    return True
    except ValueError:
        # 单行畸形(非数字 token)不影响链路:跳过,继续下一行
        return False
    # 其他行（SDK 初始化日志等）忽略
    return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Manus raw 骨架 Viser 可视化")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--replay", type=str, default=None)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    import socket
    with socket.socket() as s:
        busy = s.connect_ex((args.host, args.port)) == 0
    if busy:
        for p in range(args.port + 1, args.port + 50):
            with socket.socket() as s:
                if s.connect_ex((args.host, p)) != 0:
                    print(f"[警告] 端口 {args.port} 被占用，改用 {p}")
                    args.port = p
                    break

    import viser
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/grid", width=1.0, cell_size=0.05)
    status = server.gui.add_text("/status", "waiting for data...")

    hands = {}
    frame_no = 0
    t0 = time.time()

    def ensure_attached():
        for h in hands.values():
            if h.pc is None:
                h.attach_scene(server)

    last_report = time.time()

    def refresh():
        nonlocal frame_no, last_report
        ensure_attached()
        for h in hands.values():
            h.update_scene()
        hands_txt = "  ".join(f"{h.side} {h.glove_id}" for h in hands.values())
        fps = frame_no / max(time.time() - t0, 1e-6)
        status.text = f"帧 {frame_no}  @ {fps:.0f} fps  |  {hands_txt or '无手套'}"
        if time.time() - last_report > 2.0:
            print(f"[viz] 已推送 {frame_no} 帧 @ {fps:.0f} fps, {len(hands)} 只手",
                  file=sys.stderr)
            last_report = time.time()

    if args.replay:
        print(f"回放 {args.replay} → 浏览器打开上方 viser 输出的地址")
        with open(args.replay) as f:
            last = time.time()
            for line in f:
                if handle_line(line.strip(), hands):
                    frame_no += 1
                    refresh()
                    if not args.fast:
                        time.sleep(max(0, 0.008 - (time.time() - last)))
                        last = time.time()
        print(f"回放结束，共 {frame_no} 帧，服务器保持运行（Ctrl+C 退出）")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    # 实时模式（左右手独立流:任何一手 POS 即刷新）
    print("可视化已启动：浏览器打开上方 viser 输出的地址（本机或局域网）")
    try:
        for line in sys.stdin:
            if handle_line(line.strip(), hands):
                frame_no += 1
                refresh()
    except KeyboardInterrupt:
        pass
    print("\n[stdin 关闭，结束]")


if __name__ == "__main__":
    main()
