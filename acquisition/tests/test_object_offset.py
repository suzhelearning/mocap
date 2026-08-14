"""Motive object offset 配置、位姿组合与非破坏 HDF5 预处理测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from acquisition.object_offset import (
    DEFAULT_OBJECT_OFFSETS_PATH,
    ObjectOffsetError,
    load_object_offsets,
    preprocess_hdf5,
    transform_object_poses,
)
from acquisition.viser_core import SceneNodes, apply_frame


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
