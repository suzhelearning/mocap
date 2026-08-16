"""中央 60 Hz 对齐器的可观察时间、插值、有效性与交互坐标契约。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acquisition.alignment import AlignmentEngine, InvalidReason
from acquisition.config import load_config
from acquisition.object_offset import ObjectOffset


_CONFIG = """
router: {endpoint: tcp/127.0.0.1:7447}
rigid_bodies:
  back: 5
  objects: {cup: 11}
hands:
  left: {back_rigid_id: 5}
  right: {back_rigid_id: 5}
alignment:
  output_hz: 60
  latency_ms: 50
  mocap_max_gap_ms: 25
  manus_max_gap_ms: 35
"""


def _config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG, encoding="utf-8")
    return load_config(path)


def _mocap(t_ns: int, sequence: int, x: float) -> dict:
    return {
        "t_phys_ns": t_ns,
        "frame_number": sequence,
        "rigid_bodies": [
            {
                "id": 5,
                "position": [x, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "mean_error": 0.0001,
                "tracking_valid": True,
            },
            {
                "id": 11,
                "position": [1.0 + x, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "mean_error": 0.0002,
                "tracking_valid": True,
            },
        ],
        "markers": [],
    }


def _manus(t_ns: int, sequence: int, tip_x: float) -> dict:
    nodes = np.zeros((25, 3), dtype=float)
    nodes[5, 0] = tip_x
    rotations = np.zeros((25, 4), dtype=float)
    rotations[:, 0] = 1.0
    return {
        "t_phys_ns": t_ns,
        "seq": sequence,
        "nodes": nodes.tolist(),
        "node_quaternions_wxyz": rotations.tolist(),
    }


def _feed_regular(engine: AlignmentEngine, t0: int, count: int = 25) -> int:
    for index in range(count):
        t_ns = t0 + round(index * 1e9 / 120)
        engine.push_mocap(_mocap(t_ns, index, index / 120.0))
        for side in ("left", "right"):
            engine.push_manus(side, _manus(t_ns, index, index / 120.0))
    return t0 + round((count - 1) * 1e9 / 120)


def test_emits_one_common_fixed_60hz_timeline(tmp_path):
    cfg = _config(tmp_path)
    engine = AlignmentEngine(cfg)
    t0 = 1_000_000_000
    last = _feed_regular(engine, t0)

    frames = engine.emit_ready(last + engine.latency_ns)

    assert len(frames) == 13
    assert [frame.frame_index for frame in frames] == list(range(13))
    times = np.asarray([frame.t_phys_ns for frame in frames])
    assert np.all(np.abs(np.diff(times) - 1e9 / 60) <= 1)
    assert all(frame.frame_valid for frame in frames)
    for frame in frames:
        assert frame.hands["left"].valid
        assert frame.hands["right"].valid
        assert frame.objects["cup"].valid
        assert frame.hands["left"].mano_skeleton.shape == (21, 3)
        assert frame.interaction["cup"]["left"].shape == (21, 3)


def test_interpolates_all_sources_at_same_physical_time(tmp_path):
    cfg = _config(tmp_path)
    engine = AlignmentEngine(cfg)
    t0 = 1_000_000_000
    last = _feed_regular(engine, t0, count=5)

    frame = engine.emit_ready(last + engine.latency_ns)[1]
    elapsed = (frame.t_phys_ns - t0) / 1e9

    np.testing.assert_allclose(
        frame.objects["cup"].rigid_position[0], 1.0 + elapsed, atol=1e-7
    )
    np.testing.assert_allclose(
        frame.hands["left"].wrist_position[0], elapsed, atol=1e-7
    )
    assert frame.mocap_interpolation.interpolation_alpha == 0.0
    assert frame.hands["left"].interpolation.interpolation_alpha == 0.0


def test_large_manus_gap_keeps_tick_and_marks_only_that_hand_invalid(tmp_path):
    cfg = _config(tmp_path)
    engine = AlignmentEngine(cfg)
    t0 = 1_000_000_000
    for index in range(13):
        t_ns = t0 + round(index * 1e9 / 120)
        engine.push_mocap(_mocap(t_ns, index, 0.0))
        engine.push_manus("right", _manus(t_ns, index, 0.0))
    engine.push_manus("left", _manus(t0, 0, 0.0))
    engine.push_manus("left", _manus(t0 + 100_000_000, 1, 0.1))

    frames = engine.emit_ready(t0 + 100_000_000 + engine.latency_ns)
    middle = next(frame for frame in frames if frame.t_phys_ns == t0 + 50_000_000)

    assert middle.frame_valid is False
    assert middle.hands["left"].valid is False
    assert middle.hands["right"].valid is True
    assert middle.reason_flags & int(InvalidReason.LEFT_GAP)


def test_object_offset_and_hand_object_coordinates_are_same_frame(tmp_path):
    cfg = _config(tmp_path)
    offset = ObjectOffset(
        translation_m=np.asarray([0.25, 0.0, 0.0]),
        rotation_matrix=np.eye(3),
        quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    engine = AlignmentEngine(cfg, object_offsets={"cup": offset})
    t0 = 1_000_000_000
    last = _feed_regular(engine, t0, count=3)

    frame = engine.emit_ready(last + engine.latency_ns)[0]

    np.testing.assert_allclose(frame.objects["cup"].object_position, [1.25, 0, 0])
    expected = frame.hands["left"].mano_skeleton - np.asarray([1.25, 0, 0])
    np.testing.assert_allclose(frame.interaction["cup"]["left"], expected)


def test_new_take_resets_frame_index_and_does_not_replay_idle_history(tmp_path):
    cfg = _config(tmp_path)
    engine = AlignmentEngine(cfg)
    t0 = 1_000_000_000
    last = _feed_regular(engine, t0)
    assert engine.emit_ready(t0 + 100_000_000)[-1].frame_index > 0

    start_ns = t0 + 150_000_001
    engine.reset_timeline(start_ns)
    frames = engine.emit_ready(last + engine.latency_ns)

    assert frames[0].frame_index == 0
    assert frames[0].t_phys_ns == t0 + 166_666_667
    assert all(frame.t_phys_ns >= start_ns for frame in frames)
