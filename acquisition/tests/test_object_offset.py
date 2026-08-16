"""Motive object offset 配置、位姿组合与非破坏 HDF5 预处理测试。"""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from acquisition import cli as acquisition_cli
from acquisition.object_offset import (
    DEFAULT_OBJECT_OFFSETS_PATH,
    ObjectOffsetError,
    load_object_offsets,
    offset_from_motive_visuals,
    preprocess_hdf5,
    require_object_offsets,
    transform_object_poses,
)
from acquisition.viser_core import SceneNodes, apply_frame


_ADD_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "add_object_offset.py"
_ADD_SPEC = importlib.util.spec_from_file_location("add_object_offset", _ADD_SCRIPT)
assert _ADD_SPEC is not None and _ADD_SPEC.loader is not None
_ADD = importlib.util.module_from_spec(_ADD_SPEC)
_ADD_SPEC.loader.exec_module(_ADD)


RIGID_FROM_OBJ = np.asarray([
    [0.0, -0.9961946980917455, -0.0871557427476582],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0871557427476582, -0.9961946980917455],
])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(path: Path, matrix: np.ndarray = RIGID_FROM_OBJ) -> None:
    rows = "\n".join(
        "        - [" + ", ".join(f"{value:.16g}" for value in row) + "]"
        for row in matrix
    )
    path.write_text(
        "version: 1\n"
        "objects:\n"
        "  cylinder:\n"
        "    motive_rigid_from_obj:\n"
        "      translation_m: [-0.010, 0.0, 0.0]\n"
        "      rotation_matrix:\n"
        f"{rows}\n",
        encoding="utf-8",
    )


def _write_source_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    half_angle = np.pi / 4.0
    quaternions = np.asarray([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, np.sin(half_angle), np.cos(half_angle)],
    ], dtype=np.float32)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_name"] = "mocap-acquisition"
        objects = h5.create_group("objects")
        group = objects.create_group("cylinder")
        group.create_dataset("t_ubuntu_ns", data=np.asarray([10, 20], dtype=np.int64))
        group.create_dataset("object_position", data=positions)
        group.create_dataset("object_quaternion_xyzw", data=quaternions)
        group.create_dataset("tracking_valid", data=np.ones(2, dtype=np.uint8))
        h5.create_group("sentinel").create_dataset("unchanged", data=np.arange(4))
    return positions, quaternions


def test_default_cylinder_offset_matches_authoritative_matrix() -> None:
    offset = load_object_offsets(DEFAULT_OBJECT_OFFSETS_PATH)["cylinder"]

    np.testing.assert_allclose(offset.translation_m, [-0.010, 0.0, 0.0])
    np.testing.assert_allclose(offset.rotation_matrix, RIGID_FROM_OBJ, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(offset.quaternion_xyzw).as_matrix(),
        RIGID_FROM_OBJ,
        atol=1e-12,
    )


def test_motive_visuals_gl_go_converts_mm_and_xyz_euler_order() -> None:
    offset = offset_from_motive_visuals(
        [12.0, -34.0, 56.0],
        [10.0, 20.0, 30.0],
    )

    np.testing.assert_allclose(offset.translation_m, [0.012, -0.034, 0.056])
    np.testing.assert_allclose(
        offset.rotation_matrix,
        Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix(),
        atol=1e-12,
    )


def test_required_offsets_rejects_every_missing_capture_object() -> None:
    configured = load_object_offsets(DEFAULT_OBJECT_OFFSETS_PATH)

    with pytest.raises(ObjectOffsetError) as caught:
        require_object_offsets(["cylinder", "cup", "bottle"], configured)

    message = str(caught.value)
    assert "bottle" in message and "cup" in message
    assert "Geometry Location (GL)" in message
    assert "Geometry Orientation (GO)" in message
    assert "add-object-offset" in message


def test_add_object_offset_script_writes_loadable_config_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    config = tmp_path / "object_offsets.yaml"
    _write_config(config)
    args = [
        "cup",
        "--gl-mm", "12", "-34", "56",
        "--go-deg", "10", "20", "30",
        "--config", str(config),
    ]

    assert _ADD.main(args) == 0
    offsets = load_object_offsets(config)
    assert set(offsets) == {"cylinder", "cup"}
    np.testing.assert_allclose(offsets["cup"].translation_m, [0.012, -0.034, 0.056])
    np.testing.assert_allclose(
        offsets["cup"].rotation_matrix,
        Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix(),
        atol=1e-12,
    )
    config_hash = _sha256(config)
    assert _ADD.main(args) == 1
    assert _sha256(config) == config_hash


def test_cli_refuses_configured_object_without_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "router:\n"
        "  endpoint: tcp/127.0.0.1:7447\n"
        "rigid_bodies:\n"
        "  back: 5\n"
        "  objects: {cup: 11}\n"
        "hands:\n"
        "  left: {back_rigid_id: 5}\n"
        "  right: {back_rigid_id: 5}\n"
        "alignment: {output_hz: 60, latency_ms: 50}\n"
        f"recording: {{output_dir: {tmp_path / 'captures'}}}\n",
        encoding="utf-8",
    )
    cylinder = load_object_offsets(DEFAULT_OBJECT_OFFSETS_PATH)["cylinder"]
    monkeypatch.setattr(
        acquisition_cli,
        "load_object_offsets",
        lambda: {"cylinder": cylinder},
    )

    assert acquisition_cli.main([
        "--config", str(config), "--no-viz",
    ]) == 2
    error = capsys.readouterr().err
    assert "cup" in error
    assert "add-object-offset" in error


def test_offset_config_rejects_non_rigid_rotation(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    invalid = RIGID_FROM_OBJ.copy()
    invalid[0, 0] = 0.2
    _write_config(config, invalid)

    with pytest.raises(ObjectOffsetError, match="不是正交矩阵"):
        load_object_offsets(config)


def test_transform_composes_offset_in_motive_rigid_coordinates() -> None:
    offset = load_object_offsets(DEFAULT_OBJECT_OFFSETS_PATH)["cylinder"]
    half_angle = np.pi / 4.0
    positions, quaternions = transform_object_poses(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        np.asarray([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sin(half_angle), np.cos(half_angle)],
        ]),
        offset,
    )

    np.testing.assert_allclose(positions[0], [-0.010, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(positions[1], [1.0, 1.99, 3.0], atol=1e-12)
    world_from_rigid = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    np.testing.assert_allclose(
        Rotation.from_quat(quaternions[0]).as_matrix(), RIGID_FROM_OBJ, atol=1e-12,
    )
    np.testing.assert_allclose(
        Rotation.from_quat(quaternions[1]).as_matrix(),
        world_from_rigid @ RIGID_FROM_OBJ,
        atol=1e-12,
    )


def test_preprocess_creates_derived_h5_and_keeps_source_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "take.h5"
    output = tmp_path / "take_obj.h5"
    config = tmp_path / "offsets.yaml"
    raw_positions, _raw_quaternions = _write_source_h5(source)
    _write_config(config)
    source_hash = _sha256(source)

    result = preprocess_hdf5(
        source, output, config_path=config, chunk_frames=1,
    )

    assert result.output_path == output
    assert result.processed_objects == ("cylinder",)
    assert _sha256(source) == source_hash == result.source_sha256
    with h5py.File(source, "r") as raw_h5:
        np.testing.assert_array_equal(
            raw_h5["objects/cylinder/object_position"][:], raw_positions,
        )
        assert "object_pose_frame" not in raw_h5["objects/cylinder"].attrs
    with h5py.File(output, "r") as derived_h5:
        group = derived_h5["objects/cylinder"]
        np.testing.assert_allclose(
            group["object_position"][:],
            [[-0.010, 0.0, 0.0], [1.0, 1.99, 3.0]],
            atol=1e-7,
        )
        assert group.attrs["object_pose_frame"] == "obj"
        assert group.attrs["source_object_pose_frame"] == "motive_rigid"
        assert derived_h5.attrs["object_pose_frame"] == "obj"
        assert derived_h5.attrs["source_hdf5_sha256"] == source_hash
        np.testing.assert_array_equal(derived_h5["sentinel/unchanged"][:], np.arange(4))


def test_preprocess_refuses_overwrite_and_double_processing(tmp_path: Path) -> None:
    source = tmp_path / "take.h5"
    output = tmp_path / "take_obj.h5"
    second = tmp_path / "take_obj_again.h5"
    config = tmp_path / "offsets.yaml"
    _write_source_h5(source)
    _write_config(config)
    preprocess_hdf5(source, output, config_path=config)

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        preprocess_hdf5(source, output, config_path=config)
    with pytest.raises(ObjectOffsetError, match="拒绝重复处理"):
        preprocess_hdf5(output, second, config_path=config)
    assert not second.exists()


def test_preprocess_rejects_v3_to_preserve_interaction_coordinates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "aligned_v3.h5"
    output = tmp_path / "aligned_v3_obj.h5"
    config = tmp_path / "offsets.yaml"
    _write_source_h5(source)
    _write_config(config)
    with h5py.File(source, "r+") as h5:
        h5.attrs["h5_version"] = "3.0"

    with pytest.raises(ObjectOffsetError, match="object 坐标"):
        preprocess_hdf5(source, output, config_path=config)
    assert not output.exists()


def test_viewer_applies_hdf5_object_pose_without_offset() -> None:
    raw_position = np.asarray([0.4, 0.5, 0.6])
    raw_quaternion_xyzw = np.asarray([0.1, 0.2, 0.3, 0.9])
    frame = SimpleNamespace(position=None, wxyz=None, visible=False)
    label = SimpleNamespace(position=None, visible=False)
    mesh = SimpleNamespace(visible=False)
    marker = SimpleNamespace(points=None, colors=None)
    nodes = SceneNodes(
        obj_frames={"cylinder": frame},
        obj_labels={"cylinder": label},
        obj_meshes={"cylinder": mesh},
        marker_pc=marker,
    )
    data = {
        "t_mocap": np.asarray([100], dtype=np.int64),
        "rb_frames": [],
        "obj_frames": {
            "cylinder": [(100, raw_position, raw_quaternion_xyzw, True)],
        },
        "mk_frames": [(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))],
        "hands": {},
    }

    apply_frame(nodes, data, 100.0)

    np.testing.assert_array_equal(frame.position, raw_position)
    np.testing.assert_array_equal(frame.wxyz, raw_quaternion_xyzw[[3, 0, 1, 2]])
    assert frame.visible and label.visible and mesh.visible
