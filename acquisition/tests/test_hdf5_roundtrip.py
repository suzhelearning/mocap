"""统一时间轴 TakeWriter 的 HDF5 写入、检查、保存与丢弃契约。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from acquisition.alignment import AlignmentEngine
from acquisition.config import Config, load_config
from acquisition.recorder import EV_SAVE, EV_START, TakeWriter
from acquisition.viser_core import extract_hdf5

_INSPECT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_hdf5.py"
_INSPECT_SPEC = importlib.util.spec_from_file_location("inspect_hdf5", _INSPECT_PATH)
assert _INSPECT_SPEC is not None and _INSPECT_SPEC.loader is not None
_INSPECT = importlib.util.module_from_spec(_INSPECT_SPEC)
sys.modules[_INSPECT_SPEC.name] = _INSPECT
_INSPECT_SPEC.loader.exec_module(_INSPECT)
_REPLAY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "replay_hdf5.py"
_REPLAY_SPEC = importlib.util.spec_from_file_location("replay_hdf5", _REPLAY_PATH)
assert _REPLAY_SPEC is not None and _REPLAY_SPEC.loader is not None
_REPLAY = importlib.util.module_from_spec(_REPLAY_SPEC)
sys.modules[_REPLAY_SPEC.name] = _REPLAY
_REPLAY_SPEC.loader.exec_module(_REPLAY)
replay_compact = _REPLAY.replay_compact


inspect_file = _INSPECT.inspect_file

_TEST_CONFIG = """
router: {endpoint: tcp/127.0.0.1:7447}
rigid_bodies:
  back: 5
  objects: {cup: 11}
hands:
  left: {back_rigid_id: 5}
  right: {back_rigid_id: 5}
alignment: {output_hz: 60, latency_ms: 50, mocap_max_gap_ms: 25, manus_max_gap_ms: 35}
recording: {output_dir: captures, chunk_frames: 16}
"""


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text(_TEST_CONFIG, encoding="utf-8")
    loaded = load_config(path)
    return Config(**{**loaded.__dict__, "output_dir": tmp_path / "captures"})


def _mocap(t_ns: int, sequence: int) -> dict:
    return {
        "schema_version": 1,
        "frame_number": sequence,
        "motive_timestamp": sequence / 120,
        "publisher_received_time_ns": t_ns + 3_000_000,
        "coordinate_system": "motive_y_up_right_handed",
        "unit": "meter",
        "publisher_dropped_frames": 0,
        "t_phys_ns": t_ns,
        "arrival_monotonic_ns": t_ns + 4_000_000,
        "t_ubuntu_ns": 1_800_000_000_000_000_000 + t_ns,
        "markers": [{"position": [0.1, 0.2, 0.3], "id_kind": "point_cloud"}],
        "rigid_bodies": [
            {"id": 5, "position": [sequence / 120, 0, 0],
             "quaternion_xyzw": [0, 0, 0, 1], "mean_error": 0.0001,
             "tracking_valid": True},
            {"id": 11, "position": [1, 0, 0],
             "quaternion_xyzw": [0, 0, 0, 1], "mean_error": 0.0002,
             "tracking_valid": True},
        ],
    }


def _manus(t_ns: int, sequence: int) -> dict:
    nodes = np.zeros((25, 3), dtype=float)
    nodes[5] = [0.1, 0.0, 0.0]
    rotations = np.zeros((25, 4), dtype=float)
    rotations[:, 0] = 1.0
    return {
        "seq": sequence,
        "source_monotonic_ns": t_ns,
        "sdk_publish_time": sequence,
        "t_phys_ns": t_ns,
        "arrival_monotonic_ns": t_ns + 1_000_000,
        "t_ubuntu_ns": 1_800_000_000_000_000_000 + t_ns,
        "nodes": nodes.tolist(),
        "node_quaternions_wxyz": rotations.tolist(),
    }


def _write_take(writer: TakeWriter, cfg: Config, count: int = 9) -> int:
    writer.begin(1, start_wall_ns=1_800_000_000_000_000_000)
    writer.append_event(EV_START)
    engine = AlignmentEngine(cfg)
    t0 = 1_000_000_000
    last = t0
    for index in range(count):
        last = t0 + round(index * 1e9 / 120)
        mocap = _mocap(last, index)
        engine.push_mocap(mocap)
        for side in ("left", "right"):
            manus = _manus(last, index)
            engine.push_manus(side, manus)
    frames = engine.emit_ready(last + engine.latency_ns)
    for frame in frames:
        writer.append_aligned_frame(frame)
    writer.append_event(EV_SAVE)
    return len(frames)


def test_save_roundtrip_has_exact_compact_schema(cfg: Config):
    writer = TakeWriter(cfg.output_dir / "take_001.h5", cfg)
    n = _write_take(writer, cfg)
    writer.finalize_save()

    assert not writer.tmp_path.exists()
    with h5py.File(writer.path, "r") as f:
        assert f.attrs["h5_version"] == "4.0"
        assert f.attrs["schema_layout"] == "compact-aligned-60hz-v1"
        assert f.attrs["time_domain"] == "linux-clock-monotonic"
        assert set(f.keys()) == {"time_ns", "valid", "hands", "objects", "events"}
        assert f["time_ns"].shape == (n,)
        assert f["valid"].shape == (n,)
        assert np.all(np.abs(np.diff(f["time_ns"][:]) - 1e9 / 60) <= 1)
        assert set(f["objects/cup"].keys()) == {
            "object_position", "object_quaternion_xyzw", "valid",
        }
        assert f["objects/cup/object_position"].shape == (n, 3)
        assert f["objects/cup/object_quaternion_xyzw"].shape == (n, 4)
        for side in ("left", "right"):
            group = f[f"hands/{side}"]
            assert set(group.keys()) == {
                "keypoints_world",
                "wrist_position",
                "wrist_quaternion_xyzw",
                "valid",
            }
            assert group["keypoints_world"].shape == (n, 21, 3)
            assert group["wrist_position"].shape == (n, 3)
            assert group["wrist_quaternion_xyzw"].shape == (n, 4)
            np.testing.assert_allclose(
                group["keypoints_world"][:, 0],
                group["wrist_position"][:],
            )
        assert set(f["events"].keys()) == {"frame_index", "type"}
        np.testing.assert_array_equal(f["events/frame_index"][:], [0, n])
        np.testing.assert_array_equal(
            f["events/type"][:], [EV_START, EV_SAVE],
        )
        viewer_data = extract_hdf5(f)
        np.testing.assert_array_equal(viewer_data["t_mocap"], f["time_ns"][:])
        assert viewer_data["hands"]["left"]["valid"].all()


def test_flush_exposes_complete_equal_length_batch(cfg: Config):
    writer = TakeWriter(cfg.output_dir / "batched.h5", cfg)
    n = _write_take(writer, cfg)
    writer.flush()
    with h5py.File(writer.tmp_path, "r") as f:
        assert len(f["time_ns"]) == n
        assert len(f["hands/left/keypoints_world"]) == n
        assert len(f["objects/cup/object_position"]) == n
    writer.discard()


def test_discard_removes_temp_and_target(cfg: Config):
    writer = TakeWriter(cfg.output_dir / "discard.h5", cfg)
    _write_take(writer, cfg)
    temp = writer.tmp_path
    writer.discard()
    assert not temp.exists()
    assert not writer.path.exists()


def test_finalize_never_overwrites_existing_capture(cfg: Config):
    target = cfg.output_dir / "existing.h5"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    writer = TakeWriter(target, cfg)
    _write_take(writer, cfg)
    with pytest.raises(FileExistsError):
        writer.finalize_save()
    assert target.read_bytes() == b"existing"
    assert writer.tmp_path.exists()
    writer.discard()


def test_inspect_v4_passes_and_detects_nonmonotonic_timeline(cfg: Config):
    writer = TakeWriter(cfg.output_dir / "inspect.h5", cfg)
    _write_take(writer, cfg)
    writer.finalize_save()
    assert inspect_file(writer.path, strict=True, min_rate_ratio=0.0) == []

    with h5py.File(writer.path, "r+") as f:
        f["time_ns"][1] = f["time_ns"][0]
    errors = inspect_file(writer.path)
    assert any("非严格单调" in error for error in errors)


def test_replay_v4_exports_exact_common_rows_without_resampling(
    cfg: Config, tmp_path: Path,
) -> None:
    writer = TakeWriter(cfg.output_dir / "replay.h5", cfg)
    n = _write_take(writer, cfg)
    writer.finalize_save()
    output = tmp_path / "replay.jsonl"

    with h5py.File(writer.path, "r") as h5:
        assert replay_compact(h5, str(output)) == 0
        timeline = h5["time_ns"][:].tolist()
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert len(rows) == n
    assert [row["frame_index"] for row in rows] == list(range(n))
    assert [row["time_ns"] for row in rows] == timeline


def test_inspect_detects_contract_extra_dataset(cfg: Config):
    writer = TakeWriter(cfg.output_dir / "extra.h5", cfg)
    _write_take(writer, cfg)
    writer.finalize_save()
    with h5py.File(writer.path, "r+") as h5:
        h5.create_dataset("raw", data=np.zeros(1))

    errors = inspect_file(writer.path)

    assert any("契约外字段" in error for error in errors)

def test_inspect_v4_requires_wrist_position(cfg: Config):
    writer = TakeWriter(cfg.output_dir / "missing_wrist.h5", cfg)
    _write_take(writer, cfg)
    writer.finalize_save()
    with h5py.File(writer.path, "r+") as h5:
        del h5["hands/left/wrist_position"]
    errors = inspect_file(writer.path)
    assert any("wrist_position" in error for error in errors)
