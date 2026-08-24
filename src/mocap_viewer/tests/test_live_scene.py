"""Tests for live_scene pure functions and the Viser LiveScene (port=0, no browser)."""

from __future__ import annotations

import numpy as np
import pytest

from natnet_zenoh.frame_queue import LatestFrameQueue
from natnet_zenoh.subscriber import FrameStats

from mocap_viewer.live_scene import (
    MARKER_COLORS,
    OCCLUDED_COLOR,
    LiveScene,
    marker_color,
    xyzw_to_wxyz,
)


def make_marker(
    id_kind: str = "point_cloud",
    position: list[float] | None = None,
    occluded: bool = False,
) -> dict:
    return {
        "raw_id": 18,
        "model_id": 0,
        "member_id": 18,
        "id_kind": id_kind,
        "position": position if position is not None else [0.1, 0.2, 0.3],
        "size": 0.009,
        "residual_m_per_ray": 0.0001,
        "occluded": occluded,
        "point_cloud_solved": True,
        "model_filled": False,
        "has_model": False,
        "unlabeled": True,
        "active": False,
        "established": True,
        "measurement": False,
    }


def make_rigid(rb_id: int = 1, valid: bool = True) -> dict:
    return {
        "id": rb_id,
        "position": [0.2, 0.4, 0.6],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "mean_error": 0.0004,
        "tracking_valid": valid,
    }


def make_frame(number: int = 1, markers=None, rigid_bodies=None) -> dict:
    return {
        "schema_version": 1,
        "frame_number": number,
        "motive_timestamp": 1.25,
        "publisher_received_time_ns": 123456789,
        "coordinate_system": "motive_x_forward_z_up_right_handed",
        "unit": "meter",
        "publisher_dropped_frames": 0,
        "markers": markers if markers is not None else [],
        "rigid_bodies": rigid_bodies if rigid_bodies is not None else [],
    }


# ---------- 纯函数 ----------


def test_xyzw_to_wxyz_identity() -> None:
    np.testing.assert_array_equal(xyzw_to_wxyz([0, 0, 0, 1]), [1, 0, 0, 0])


def test_xyzw_to_wxyz_permutes() -> None:
    np.testing.assert_array_equal(xyzw_to_wxyz([1, 2, 3, 4]), [4, 1, 2, 3])


def test_xyzw_to_wxyz_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        xyzw_to_wxyz([1, 2, 3])


def test_marker_color_by_kind_and_occlusion() -> None:
    assert marker_color(make_marker("point_cloud")) == MARKER_COLORS["point_cloud"]
    assert marker_color(make_marker("asset_member")) == MARKER_COLORS["asset_member"]
    assert marker_color(make_marker("active")) == MARKER_COLORS["active"]
    assert marker_color(make_marker("unknown")) == MARKER_COLORS["unknown"]
    assert marker_color(make_marker("mystery_kind")) == MARKER_COLORS["unknown"]
    assert marker_color(make_marker("active", occluded=True)) == OCCLUDED_COLOR


# ---------- 场景 ----------


class FakeSource:
    def __init__(self) -> None:
        self.queue = LatestFrameQueue(8)
        self.stats = FrameStats()
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture()
def scene() -> LiveScene:
    scene = LiveScene(FakeSource(), "127.0.0.1", 0, 30.0, start_thread=False)
    yield scene
    scene.close()


def test_scene_initial_state(scene: LiveScene) -> None:
    assert scene._markers is not None
    assert scene._rigid_centers is not None
    assert scene._status.content.startswith("Waiting")
    # GUI 控件句柄（照 moshpp 测试的 hasattr 风格）
    assert scene._paused is not None
    assert scene._marker_size is not None
    assert scene._show_markers is not None
    assert scene._show_rigid is not None
    assert scene._show_ground is not None


def test_render_frame_updates_markers_and_status(scene: LiveScene) -> None:
    frame = make_frame(
        5,
        markers=[make_marker(), make_marker(occluded=True)],
        rigid_bodies=[make_rigid(1, valid=True), make_rigid(2, valid=False)],
    )
    scene._render_frame(frame)
    assert scene._last_frame_number == 5
    scene._update_status()
    assert "Frame:" in scene._status.content
    assert "Rate:" in scene._status.content
    assert scene._rigid_handles[1].visible
    assert not scene._rigid_handles[2].visible


def test_render_frame_hides_vanished_rigid_body(scene: LiveScene) -> None:
    scene._render_frame(make_frame(1, rigid_bodies=[make_rigid(1, valid=True)]))
    assert scene._rigid_handles[1].visible
    scene._render_frame(make_frame(2, rigid_bodies=[]))
    assert not scene._rigid_handles[1].visible


def test_tick_skips_when_paused(scene: LiveScene) -> None:
    scene._paused.value = True
    frame = make_frame(9)
    scene._source.queue.put_latest(frame)
    scene._tick()
    assert not scene._source.queue.empty()  # 暂停不消费


def test_tick_drains_to_newest(scene: LiveScene) -> None:
    for n in (1, 2, 3):
        scene._source.queue.put_latest(make_frame(n))
    scene._tick()
    assert scene._last_frame_number == 3
    assert scene._source.queue.empty()


def test_close_stops_source_and_thread() -> None:
    threaded = LiveScene(FakeSource(), "127.0.0.1", 0, 30.0)  # 带真实渲染线程
    try:
        assert threaded._stop_event.is_set() is False
        assert threaded._thread.is_alive()
        threaded.close()
        assert threaded._stop_event.is_set()
        assert threaded._source.stopped
        assert not threaded._thread.is_alive()
    finally:
        threaded.close()
