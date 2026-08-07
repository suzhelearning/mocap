"""Viser live scene driven by a LatestFrameQueue of validated NatNet frames."""

from __future__ import annotations

import queue as _queue
import threading
import time
from collections.abc import Mapping, Sequence

import numpy as np
import viser

# ---------- 纯函数（可单测） ----------


def xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray:
    """NatNet xyzw 四元数 → Viser wxyz 顺序。"""
    if len(q) != 4:
        raise ValueError(f"quaternion must have length 4, got {len(q)}")
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


MARKER_COLORS: dict[str, tuple[int, int, int]] = {
    "active": (255, 90, 90),  # 红
    "asset_member": (90, 160, 255),  # 蓝
    "point_cloud": (120, 220, 140),  # 绿
    "unknown": (220, 220, 220),  # 浅灰
}
OCCLUDED_COLOR = (90, 90, 90)  # 深灰暗点
RIGID_VALID_COLOR = (80, 220, 120)  # 绿
RIGID_INVALID_COLOR = (255, 80, 80)  # 红


def marker_color(marker: Mapping[str, object]) -> tuple[int, int, int]:
    """occluded 优先显示暗灰；否则按 id_kind 分色，未知 kind 回退浅灰。"""
    if marker.get("occluded"):
        return OCCLUDED_COLOR
    return MARKER_COLORS.get(str(marker.get("id_kind")), MARKER_COLORS["unknown"])


# ---------- 场景 ----------


class LiveScene:
    """一帧一帧消费 queue 中最新帧，覆盖更新 Viser 场景。"""

    def __init__(self, source, host: str, port: int, fps: float, *, start_thread: bool = True) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._source = source  # 鸭子类型：.queue / .stats / .stop()
        self._fps = float(fps)
        self._stop_event = threading.Event()
        self._last_frame_number: int | None = None
        self._last_status_at = 0.0
        self._status_interval = 0.5  # snapshot() 会重置帧率窗口，必须限频
        self._rigid_handles: dict[int, object] = {}  # id -> FrameHandle
        self._known_rigid_ids: set[int] = set()

        self.server = viser.ViserServer(host=host, port=port, label="NatNet Markers (Zenoh)")
        self.server.scene.add_grid(
            "/ground", width=1.0, height=1.0, cell_size=0.05, section_size=0.25, plane="xy"
        )
        self.server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.005)
        self._markers = self.server.scene.add_point_cloud(
            "/markers",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.01,
            point_shape="diamond",
        )
        self._rigid_centers = self.server.scene.add_point_cloud(
            "/rigid_bodies/centers",
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.015,
            point_shape="circle",
        )
        self._build_gui()
        self._configure_camera()
        if start_thread:
            self._thread = threading.Thread(
                target=self._render_loop, name="mocap-viewer-render", daemon=True
            )
            self._thread.start()
        else:
            self._thread = None

    def _build_gui(self) -> None:
        self.server.gui.add_markdown("# NatNet Live")
        self._status = self.server.gui.add_markdown("Waiting for Zenoh frames…")
        self._paused = self.server.gui.add_checkbox("Pause", initial_value=False)
        self._marker_size = self.server.gui.add_slider(
            "Marker size", min=0.001, max=0.05, step=0.001, initial_value=0.01
        )
        self._show_markers = self.server.gui.add_checkbox("Markers", initial_value=True)
        self._show_rigid = self.server.gui.add_checkbox("Rigid bodies", initial_value=True)
        self._show_ground = self.server.gui.add_checkbox("Ground grid", initial_value=True)

    def _configure_camera(self) -> None:
        look_at = np.asarray([0.0, 0.0, 0.0])
        position = np.asarray([0.6, -0.6, 0.35])  # y-up 世界，俯视手部区域
        up = np.asarray([0.0, 1.0, 0.0])
        self.server.initial_camera.look_at = look_at
        self.server.initial_camera.position = position
        self.server.initial_camera.up = up

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.look_at = look_at
            client.camera.position = position
            client.camera.up_direction = up

    # ---------- 渲染 ----------

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick()

    def _tick(self) -> None:  # 单步可测
        delay = 1.0 / max(self._fps, 0.001)
        if self._paused.value:  # 暂停：不消费，队列保持最新帧
            self._stop_event.wait(delay)
            return
        try:
            frame = self._source.queue.get(timeout=delay)
        except _queue.Empty:
            return
        while True:  # 排空到最新帧，渲染最新
            try:
                frame = self._source.queue.get(timeout=0.0)
            except _queue.Empty:
                break
        self._render_frame(frame)
        now = time.monotonic()
        if now - self._last_status_at >= self._status_interval:
            self._update_status()
            self._last_status_at = now

    def _render_frame(self, frame: dict) -> None:
        self._last_frame_number = int(frame["frame_number"])
        markers = frame["markers"]
        points = np.asarray([m["position"] for m in markers], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray([marker_color(m) for m in markers], dtype=np.uint8).reshape(-1, 3)
        self._markers = self.server.scene.add_point_cloud(
            "/markers",
            points=points,
            colors=colors,
            point_size=float(self._marker_size.value),
            point_shape="diamond",
        )
        self._markers.visible = bool(self._show_markers.value)
        self._render_rigid_bodies(frame["rigid_bodies"])

    def _render_rigid_bodies(self, rigid_bodies: list) -> None:
        centers = []
        center_colors = []
        current_ids = set()
        for rb in rigid_bodies:
            rb_id = int(rb["id"])
            current_ids.add(rb_id)
            centers.append(rb["position"])
            valid = bool(rb["tracking_valid"])
            center_colors.append(RIGID_VALID_COLOR if valid else RIGID_INVALID_COLOR)
            wxyz = xyzw_to_wxyz(rb["quaternion_xyzw"])
            handle = self.server.scene.add_frame(  # 同 path 重加即更新
                f"/rigid_bodies/{rb_id}",
                position=np.asarray(rb["position"]),
                wxyz=wxyz,
                axes_length=0.05,
                axes_radius=0.002,
            )
            handle.visible = valid and bool(self._show_rigid.value)
            self._rigid_handles[rb_id] = handle
        self._known_rigid_ids |= current_ids
        for rb_id in self._known_rigid_ids - current_ids:  # 消失的刚体：重加并隐藏
            handle = self.server.scene.add_frame(f"/rigid_bodies/{rb_id}")
            handle.visible = False
            self._rigid_handles[rb_id] = handle
        self._rigid_centers = self.server.scene.add_point_cloud(
            "/rigid_bodies/centers",
            points=np.asarray(centers, dtype=np.float32).reshape(-1, 3),
            colors=np.asarray(center_colors, dtype=np.uint8).reshape(-1, 3),
            point_size=0.015,
            point_shape="circle",
        )
        self._rigid_centers.visible = bool(self._show_rigid.value)

    def _update_status(self) -> None:
        s = self._source.stats.snapshot()
        self._status.content = (
            f"**Frame:** {self._last_frame_number or '-'}  ·  "
            f"**Rate:** {s['frame_rate_hz']:.1f} Hz  ·  "
            f"**Markers:** {s['last_marker_count']}\n\n"
            f"missing={s['missing_frames']}  out_of_order={s['out_of_order_frames']}  "
            f"invalid={s['invalid_messages']}  restarts={s['source_restarts']}\n"
            f"publisher_dropped={s['publisher_dropped_frames']}  "
            f"queue_dropped={self._source.queue.dropped_frames}"
        )

    # ---------- 生命周期（照 moshpp ManoViewer） ----------

    def close(self) -> None:
        self._stop_event.set()
        self._source.stop()
        self.server.stop()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def sleep_forever(self) -> None:
        self.server.sleep_forever()
