"""采集程序主入口:装配 StreamHub + 拼接 + 可视化 + 按键录制。

用法:
    pixi run record                # 带 web 可视化
    pixi run record-noviz          # 不带可视化(纯采集)

按键(默认):r 开始录制 / Space 暂停恢复 / s 保存 / d 丢弃 / q 退出
浏览器:http://<host>:8081(可视化模式)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from .config import ConfigError, load_config
from .keyboard import raw_keyboard
from .live_view import StitchedScene
from .manus_schema import palm_node_index
from .recorder import TakeWriter
from .state_machine import TakeController
from .stitching import extract_rigid_body, stitch_hand
from .streams import StreamHub

FLUSH_INTERVAL = 0.5          # 录制缓冲落盘周期(秒)
STATUS_INTERVAL = 0.2         # 状态栏刷新周期(秒)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="人类操作演示数据采集")
    ap.add_argument("--config", type=str, default="config.yaml",
                    help="配置文件路径(默认 config.yaml)")
    ap.add_argument("--no-viz", action="store_true",
                    help="不启动 web 可视化(纯采集)")
    ap.add_argument("--host", type=str, default="0.0.0.0",
                    help="可视化监听地址(默认 0.0.0.0)")
    ap.add_argument("--no-markers", action="store_true",
                    help="可视化不渲染原始 markers")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[错误] 配置加载失败: {exc}", file=sys.stderr)
        return 2

    hub = StreamHub(cfg.router_endpoint)
    latest_hands: dict[str, dict | None] = {"left": None, "right": None}
    status_lock = threading.Lock()
    last_flush = time.time()
    last_status = time.time()

    viz: StitchedScene | None = None
    if not args.no_viz:
        viz = StitchedScene(cfg, host=args.host)
        viz._show_markers = not args.no_markers

    def writer_factory(take_id: int, path):
        return TakeWriter(path, cfg)

    ctrl = TakeController(cfg, writer_factory, take_dir=cfg.output_dir)

    # -- zenoh 回调(库线程,快速) -------------------------------------------
    def on_mocap(frame: dict) -> None:
        writer = ctrl.writer
        if writer is not None:
            writer.append_mocap(frame)

    def on_manus(side: str, msg: dict) -> None:
        latest = hub.latest_mocap()
        back = extract_rigid_body(latest, cfg.back_rigid_id) if latest else None
        if back is None or not back[2]:
            latest_hands[side] = None
            return
        hcfg = cfg.hands[side]
        edges = hub.latest_edges(side) or []
        palm = palm_node_index(edges)
        nodes = np.asarray(msg["nodes"], dtype=float)
        g, p_w, q_w = stitch_hand(
            nodes, palm, back[0], back[1], hcfg.wrist_offset, cfg.axis_matrix()
        )
        latest_hands[side] = {
            "nodes_global": g, "wrist_pos": p_w, "wrist_quat_xyzw": q_w,
        }
        writer = ctrl.writer
        if writer is not None:
            writer.set_edges(side, edges)
            writer.append_manus(side, msg, p_w, q_w, g)

    hub.on_mocap(on_mocap)
    hub.on_manus("left", lambda m: on_manus("left", m))
    hub.on_manus("right", lambda m: on_manus("right", m))

    # -- 主循环 tick(键盘轮询空隙执行) --------------------------------------
    def tick() -> None:
        nonlocal last_flush, last_status
        now = time.time()
        writer = ctrl.writer
        if writer is not None and now - last_flush >= FLUSH_INTERVAL:
            writer.flush()
            last_flush = now
        if viz is not None:
            edges = {s: hub.latest_edges(s) or [] for s in ("left", "right")}
            viz.update(hub.latest_mocap(), latest_hands, edges,
                       status_text=_status_text())
        if now - last_status >= STATUS_INTERVAL:
            sys.stdout.write(f"\r\x1b[K{_status_text()}")
            sys.stdout.flush()
            last_status = now

    def _status_text() -> str:
        rates = hub.rates_hz()
        with status_lock:
            line = (
                f"{ctrl.status_line()}  |  "
                f"动捕 {rates['mocap']:5.1f}Hz  左手 {rates['left']:5.1f}Hz  "
                f"右手 {rates['right']:5.1f}Hz  "
                f"[r]录 [空格]停 [s]存 [d]丢 [q]退"
            )
        return line

    def on_key(ch: str) -> None:
        changed = ctrl.handle(ch)
        if changed:
            sys.stdout.write(f"\n{ctrl.last_result or _status_text()}\n")
            sys.stdout.flush()
        if ctrl.quit_requested:
            stop.set()

    # -- 启动 -------------------------------------------------------------
    try:
        hub.start()
    except Exception as exc:
        print(f"[错误] Zenoh 连接失败({cfg.router_endpoint}): {exc}", file=sys.stderr)
        print("请先启动 zenohd(pixi run start-router 或 systemctl --user start zenohd)")
        return 1

    if viz is not None:
        print(f"[viz] 浏览器打开 http://{args.host or '127.0.0.1'}:{cfg.viz_port}"
              f"(局域网用本机 IP)", file=sys.stderr)
    print(f"[采集] 已连接 {cfg.router_endpoint}。"
          f"按 {cfg.keymap['start']!r} 开始录制…", file=sys.stderr)

    stop = threading.Event()

    try:
        raw_keyboard(on_key, stop)
    except KeyboardInterrupt:
        pass
    finally:
        if ctrl.writer is not None:      # 退出前丢弃未保存的 take
            ctrl.writer.discard()
        hub.stop()
        if viz is not None:
            viz.stop()
        sys.stdout.write("\n[退出]\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
