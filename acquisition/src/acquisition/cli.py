"""采集程序主入口:装配 StreamHub + 拼接 + 可视化 + 按键/按钮录制。

用法:
    pixi run record                # 带 web 可视化
    pixi run record-noviz          # 不带可视化(纯采集)

按键(默认):r 开始录制 / s 保存 / d 丢弃 / q 退出
浏览器:http://<host>:8081(可视化模式)——「录制控制」面板同功能按钮,
另有「采集频率」下拉可运行时改落盘采样率(默认 100Hz,输入流 120Hz 下采样)。
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .config import ConfigError, load_config
from .keyboard import raw_keyboard
from .kinematics import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from .live_view import StitchedScene
from .manus_schema import manus_to_mediapipe, palm_node_index
from .rate import RateGate
from .recorder import TakeWriter
from .state_machine import TakeController
from .stitching import extract_rigid_body, hand_nodes_to_global, stitch_hand
from .streams import StreamHub

FLUSH_INTERVAL = 0.5          # 录制缓冲落盘周期(秒)
STATUS_INTERVAL = 0.2         # 状态栏刷新周期(秒)


def _cleanup_orphan_tmp(output_dir: Path) -> None:
    """清理上次异常退出残留的临时录制文件(单实例采集,进程内串行)。

    覆盖:旧格式的 .take_*_tmp.h5 与私有目录 .tmp_<pid>/(含日期子目录内,
    递归查找)。
    """
    try:
        if not output_dir.is_dir():
            return
        n = 0
        for p in output_dir.rglob(".take_*_tmp.h5"):
            try:
                p.unlink(missing_ok=True)
                n += 1
            except OSError:
                pass
        for d in output_dir.rglob(".tmp_*/"):
            try:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
            except OSError:
                pass
        if n:
            print(f"[采集] 已清理 {n} 个异常退出残留的临时录制文件", file=sys.stderr)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="人类操作演示数据采集")
    ap.add_argument("--config", type=str, default="config.yaml",
                    help="配置文件路径(默认 config.yaml)")
    ap.add_argument("--no-viz", action="store_true",
                    help="不启动 web 可视化(纯采集)")
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="可视化监听地址(默认 127.0.0.1;局域网访问须显式指定"
                         "本机 IP 并自行防护——Viser 无认证,任何能连到该端口的"
                         "客户端都能查看数据并操控录制)")
    ap.add_argument("--no-markers", action="store_true",
                    help="可视化不渲染原始 markers")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[错误] 配置加载失败: {exc}", file=sys.stderr)
        return 2

    # 清理上次异常退出(崩溃/强杀)残留的临时文件与私有目录
    _cleanup_orphan_tmp(cfg.output_dir)

    hub = StreamHub(cfg.router_endpoint)
    latest_hands: dict[str, dict | None] = {"left": None, "right": None}
    latest_mano: dict[str, dict | None] = {"left": None, "right": None}
    status_lock = threading.Lock()
    stitched_lock = threading.Lock()
    last_flush = time.time()
    last_status = time.time()
    no_data_hint_shown = False

    # web 按钮命令队列(viser 回调线程 → 主循环 drain)+ 落盘频率门控
    command_q: queue.Queue[str] = queue.Queue()
    rate_gate = RateGate(cfg.sample_hz)
    current_object: dict[str, str] = {
        "name": "cylinder" if "cylinder" in cfg.objects else next(iter(cfg.objects), ""),
    }

    def on_object(name: str) -> None:
        current_object["name"] = name
        sys.stdout.write(f"\n[物体] 当前操作物体: {name}\n")
        sys.stdout.flush()

    viz: StitchedScene | None = None
    if not args.no_viz:
        viz = StitchedScene(cfg, host=args.host,
                            on_command=command_q.put,
                            on_freq=rate_gate.set_hz,
                            on_object=on_object)
        viz._show_markers = not args.no_markers

    def writer_factory(take_id: int, path):
        return TakeWriter(path, cfg, rate_gate=rate_gate)

    ctrl = TakeController(cfg, writer_factory, take_dir=cfg.output_dir)

    # -- zenoh 回调(库线程,快速) -------------------------------------------
    def on_mocap(frame: dict) -> None:
        writer = ctrl.writer
        if writer is not None:
            writer.append_mocap(frame)

    def on_manus(side: str, msg: dict) -> None:
        # 时间对齐:以本手套帧到达时刻为基准,对 mocap 刚体流插值,
        # 消除双流帧周期错位(0~8.3ms → <1ms 级)
        latest = hub.mocap_at(msg["t_ubuntu_ns"])
        hcfg = cfg.hands[side]
        edges = hub.latest_edges(side) or []
        palm = palm_node_index(edges)
        nodes = np.asarray(msg["nodes"], dtype=float)

        if hcfg.wrist_rigid_id is not None:
            # Motive 直接追踪手腕刚体:位姿直接用,跳过 back/offset
            rb = extract_rigid_body(latest, hcfg.wrist_rigid_id) if latest else None
            if rb is None or not rb[2]:
                with stitched_lock:
                    latest_hands[side] = None
                    latest_mano[side] = None
                return
            p_w = np.asarray(rb[0], dtype=float)
            q_w = quat_xyzw_to_wxyz(rb[1])
            g = hand_nodes_to_global(nodes, palm, p_w, q_w, cfg.axis_matrix())
            q_w_xyzw = quat_wxyz_to_xyzw(q_w)
        else:
            # 背部/手背刚体 + offset 推算手腕:必须用每只手自己的
            # back_rigid_id(cfg.back_rigid_id 只是全局 back 渲染用),
            # 否则左右手会绑定到同一刚体,一侧无数据
            back = (extract_rigid_body(latest, hcfg.back_rigid_id)
                    if latest and hcfg.back_rigid_id is not None else None)
            if back is None or not back[2]:
                with stitched_lock:
                    latest_hands[side] = None
                    latest_mano[side] = None
                return
            g, p_w, q_w_xyzw = stitch_hand(
                nodes, palm, back[0], back[1], hcfg.wrist_offset, cfg.axis_matrix()
            )

        with stitched_lock:
            latest_hands[side] = {
                "nodes_global": g, "wrist_pos": p_w,
                "wrist_quat_xyzw": q_w_xyzw,
            }
            latest_mano[side] = {
                "keypoints_global": np.asarray(
                    manus_to_mediapipe(g), dtype=float),
            }
        writer = ctrl.writer
        if writer is not None:
            writer.set_edges(side, edges)
            writer.append_manus(
                side, msg, p_w, q_w_xyzw,
                np.asarray(manus_to_mediapipe(g), dtype=float),
            )


    hub.on_mocap(on_mocap)
    hub.on_manus("left", lambda m: on_manus("left", m))
    hub.on_manus("right", lambda m: on_manus("right", m))

    def _handle_cmd(ch: str) -> None:
        """处理一个命令字符(键盘或 web 按钮,均为主线程调用,无竞态)。"""
        if ch == cfg.keymap["save"] and ctrl.writer is not None:
            health = hub.health_snapshot()
            health["rate_gate"] = rate_gate.stats()
            ctrl.writer.set_stream_health(health)
            ctrl.writer.set_rigid_body_names(hub.rigid_body_names())
        changed = ctrl.handle(ch)
        if changed:
            sys.stdout.write(f"\n{ctrl.last_result or _status_text()}\n")
            sys.stdout.flush()
        if ctrl.quit_requested:
            stop.set()

    # -- 主循环 tick(键盘轮询空隙执行) --------------------------------------
    def tick() -> None:
        nonlocal last_flush, last_status, no_data_hint_shown
        # 没有任何流到达时,一次性醒目提示(状态栏持续显示等待状态)
        if not no_data_hint_shown and not any(hub.rates_hz().values()):
            sys.stderr.write(
                "[采集] ⏳ 等待设备启动:未收到动捕/手部数据流\n")
            sys.stderr.flush()
            no_data_hint_shown = True
        now = time.time()
        writer = ctrl.writer
        if writer is not None and now - last_flush >= FLUSH_INTERVAL:
            writer.flush()
            last_flush = now
        if viz is not None:
            edges = {s: hub.latest_edges(s) or [] for s in ("left", "right")}
            calib_ids = {
                rid
                for name in ("left_wrist", "left_dip", "right_wrist", "right_dip")
                if (rid := hub.rigid_body_id(name)) is not None
            }
            with stitched_lock:
                hands_snapshot = dict(latest_hands)
                mano_snapshot = dict(latest_mano)
            viz.update(hub.latest_mocap(), hands_snapshot, edges,
                       latest_mano=mano_snapshot,
                       calib_rigid_ids=calib_ids,
                       status=_status_fields(), state=ctrl.state)
        if now - last_status >= STATUS_INTERVAL:
            sys.stdout.write(f"\r\x1b[K{_status_text()}")
            sys.stdout.flush()
            last_status = now

    def _status_fields() -> dict[str, str]:
        """状态栏拆分为独立字段(web 端各用独立 item,避免整块重排跳变)。"""
        rates = hub.rates_hz()
        with status_lock:
            if not any(rates.values()):
                ctrl_part = "⏳ 等待设备启动(未收到动捕/手部数据)"
            else:
                ctrl_part = ctrl.status_line()
            health = hub.health_snapshot()["streams"]
            losses = sum(
                stream["sequence_gaps"]
                + stream["callback_queue_dropped"]
                + stream["decode_errors"]
                for stream in health.values()
            )
            return {
                "ctrl": ctrl_part,
                "rate_mocap": f"动捕 {rates['mocap']:5.1f}Hz",
                "rate_left": f"左手 {rates['left']:5.1f}Hz",
                "rate_right": f"右手 {rates['right']:5.1f}Hz",
                "sample": f"采集@{rate_gate.hz:.0f}Hz",
                "health": f"异常/缺帧 {losses}",
                "keys": "[r]录 [s]存 [d]丢 [q]退",
            }

    def _status_text() -> str:
        """终端单行状态栏(与 web 字段同源)。"""
        f = _status_fields()
        return (f"{f['ctrl']}  |  {f['rate_mocap']}  {f['rate_left']}  {f['rate_right']}"
                f"  |  {f['health']}  {f['sample']}  {f['keys']}")

    def on_key(ch: str) -> None:
        _handle_cmd(ch)

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
        raw_keyboard(on_key, stop, on_idle=tick)
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
