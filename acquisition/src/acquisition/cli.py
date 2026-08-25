"""采集程序主入口:装配 StreamHub + 拼接 + 可视化 + 按键/按钮录制。

用法(--user 必填且 offset/<user>.yaml 必须含左右标定):
    pixi run record -- --object hammer --user shd
    pixi run record-noviz -- --object hammer tianji_wrist --user shd

按键(默认):r 开始录制 / s 保存 / d 丢弃 / q 退出
浏览器:http://<host>:8081(可视化模式)——「录制控制」面板同功能按钮。
所有数据由中央对齐器输出到一个固定 60Hz 公共物理时间轴。
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path


from .alignment import AlignedFrame, AlignmentEngine
from .config import ConfigError, load_config
from .keyboard import raw_keyboard
from .live_view import StitchedScene
from .object_offset import (
    ObjectOffsetError,
    load_object_offsets,
    object_offsets_sha256,
    require_object_offsets,
)
from .recorder import TakeWriter
from .state_machine import TakeController
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
    ap.add_argument("--user", required=True,
                    help="操作者名字;必须存在 offset/<user>.yaml 的左右手标定")
    ap.add_argument("--no-viz", action="store_true",
                    help="不启动 web 可视化(纯采集)")
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="可视化监听地址(默认 127.0.0.1;局域网访问须显式指定"
                         "本机 IP 并自行防护——Viser 无认证,任何能连到该端口的"
                         "客户端都能查看数据并操控录制)")
    ap.add_argument("--no-markers", action="store_true",
                    help="可视化不渲染原始 markers")
    ap.add_argument("--object", action="extend", nargs="+", metavar="NAME",
                    required=True,
                    help="采集物体(必须显式指定,如 --object hammer cube;"
                         "可多个 --object 重复,如 --object hammer --object cube)")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(
            args.config,
            user=args.user,
            require_user_calibration=True,
        )
    except ConfigError as exc:
        print(f"[错误] 配置加载失败: {exc}", file=sys.stderr)
        return 2

    # --object:过滤为物体子集(录制 HDF5/可视化/外参校验全部跟随)
    if args.object:
        wanted = dict.fromkeys(args.object)   # 保序去重
        unknown = [n for n in wanted if n not in cfg.objects]
        if unknown:
            print(f"[错误] 未知物体: {', '.join(unknown)}。"
                  f"可选: {', '.join(cfg.objects) or '(配置中无物体)'}",
                  file=sys.stderr)
            return 2
        cfg = replace(cfg, objects={n: cfg.objects[n] for n in wanted})
        print(f"[采集] 本次只采集物体: {', '.join(cfg.objects)}",
              file=sys.stderr)

    # 清理上次异常退出(崩溃/强杀)残留的临时文件与私有目录
    _cleanup_orphan_tmp(cfg.output_dir)

    hub = StreamHub(cfg.router_endpoint)
    try:
        configured_offsets = load_object_offsets()
        active_offsets = require_object_offsets(
            cfg.objects.keys(),
            configured_offsets,
        )
        offset_sha256 = object_offsets_sha256()
    except ObjectOffsetError as exc:
        print(f"[错误] 物体外参加载失败: {exc}", file=sys.stderr)
        return 2
    alignment = AlignmentEngine(cfg, object_offsets=active_offsets)
    latest_aligned: AlignedFrame | None = None
    aligned_lock = threading.Lock()
    status_lock = threading.Lock()
    last_flush = time.monotonic()
    last_status = time.monotonic()
    no_data_hint_shown = False

    command_q: queue.Queue[str] = queue.Queue()
    viz: StitchedScene | None = None
    if not args.no_viz:
        viz = StitchedScene(
            cfg,
            host=args.host,
            on_command=command_q.put,
        )
        viz._show_markers = not args.no_markers

    def writer_factory(take_id: int, path: Path) -> TakeWriter:
        return TakeWriter(
            path,
            cfg,
            object_offset_sha256=offset_sha256,
            object_pose_frames={
                name: ("obj" if name in active_offsets else "motive_rigid")
                for name in cfg.objects
            },
        )

    ctrl = TakeController(cfg, writer_factory, take_dir=cfg.output_dir)

    # 原始回调只更新对齐缓冲；主循环生成的统一帧才进入 HDF5。
    def on_mocap(frame: dict) -> None:
        alignment.push_mocap(frame)

    def on_manus(side: str, msg: dict) -> None:
        alignment.push_manus(side, msg)

    hub.on_mocap(on_mocap)
    hub.on_manus("left", lambda msg: on_manus("left", msg))
    hub.on_manus("right", lambda msg: on_manus("right", msg))

    def _handle_cmd(ch: str) -> None:
        """处理键盘或 web 命令；只在主线程改变状态机。"""
        if ch == cfg.keymap["start"] and ctrl.writer is None:
            alignment.reset_timeline(time.monotonic_ns())
        changed = ctrl.handle(ch)
        if changed:
            sys.stdout.write(f"\n{ctrl.last_result or _status_text()}\n")
            sys.stdout.flush()
        if ctrl.quit_requested:
            stop.set()

    def tick() -> None:
        nonlocal last_flush, last_status, no_data_hint_shown, latest_aligned
        while True:
            try:
                _handle_cmd(command_q.get_nowait())
            except queue.Empty:
                break
        rates = hub.rates_hz()
        if not no_data_hint_shown and not any(rates.values()):
            sys.stderr.write("[采集] 等待设备启动：未收到动捕/手部数据流\n")
            sys.stderr.flush()
            no_data_hint_shown = True

        now_monotonic_ns = time.monotonic_ns()
        for frame in alignment.emit_ready(now_monotonic_ns):
            with aligned_lock:
                latest_aligned = frame
            writer = ctrl.writer
            if writer is not None:
                writer.append_aligned_frame(frame)

        now = time.monotonic()
        writer = ctrl.writer
        if writer is not None and now - last_flush >= FLUSH_INTERVAL:
            writer.flush()
            last_flush = now

        if viz is not None:
            with aligned_lock:
                frame = latest_aligned
            viz.update_aligned(
                frame,
                edges={
                    side: hub.latest_edges(side) or []
                    for side in ("left", "right")
                },
                calib_rigid_ids={
                    rigid_id
                    for name in (
                        "left_back", "left_dip", "right_back", "right_dip",
                    )
                    if (rigid_id := hub.rigid_body_id(name)) is not None
                },
                status=_status_fields(),
                state=ctrl.state,
            )
        if now - last_status >= STATUS_INTERVAL:
            sys.stdout.write(f"\r\x1b[K{_status_text()}")
            sys.stdout.flush()
            last_status = now

    def _status_fields() -> dict[str, str]:
        rates = hub.rates_hz()
        with status_lock:
            ctrl_part = (
                "等待设备启动(未收到动捕/手部数据)"
                if not any(rates.values())
                else ctrl.status_line()
            )
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
                "sample": "统一时间轴@60Hz",
                "health": f"异常/缺帧 {losses}",
                "keys": "[r]录 [s]存 [d]丢 [q]退",
            }

    def _status_text() -> str:
        fields = _status_fields()
        return (
            f"{fields['ctrl']}  |  {fields['rate_mocap']} "
            f"{fields['rate_left']}  {fields['rate_right']}  |  "
            f"{fields['health']}  {fields['sample']}  {fields['keys']}"
        )

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
