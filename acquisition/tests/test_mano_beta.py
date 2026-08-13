"""离线 MANO beta 写回与最小 HDF5 契约测试。"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from acquisition.viser_core import extract_hdf5, reject_external_links


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mano_beta.py"
_SPEC = importlib.util.spec_from_file_location("mano_beta_test_module", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


def _make_h5(path: Path, side: str, n: int = 10) -> np.ndarray:
    """构造真实 MANO 关键点生成的最小 H5。"""
    from mano_fit import load_mano

    layer = load_mano(side)
    rng = np.random.default_rng(3)
    beta = rng.uniform(-1.5, 1.5, 10)
    _, joints = layer.forward(
        np.zeros(45), beta, np.zeros(3), np.zeros(3),
    )
    skeleton = np.repeat(joints[None], n, axis=0)
    with h5py.File(path, "w") as f:
        group = f.create_group("hands").create_group(side)
        group.create_dataset("t_ubuntu_ns", data=np.arange(n, dtype=np.int64))
        group.create_dataset("wrist_position", data=skeleton[:, 0].astype(np.float32))
        group.create_dataset("mano_skeleton", data=skeleton.astype(np.float32))
    return beta


def test_estimate_and_write_beta_only(tmp_path):
    path = tmp_path / "take.h5"
    beta_true = _make_h5(path, "left", n=8)
    estimate, error = _MOD.estimate_side(path, "left", samples=6)

    assert error is None
    assert estimate is not None
    assert estimate.skeleton_source == "mano_skeleton"
    assert estimate.samples_used == 6
    assert np.abs(estimate.beta - beta_true).max() < 0.3

    # 模拟旧文件；新写回必须删除所有逐帧派生缓存。
    with h5py.File(path, "r+") as f:
        group = f["hands/left"]
        group.create_dataset("mano_pose", data=np.zeros((8, 16, 3)))
        group.create_dataset("mano_translation", data=np.zeros((8, 3)))
        group.create_dataset("mano_scale", data=np.ones(8))
        group.create_dataset("mano_joints16", data=np.zeros((8, 16, 3)))
        group.create_dataset("mano_fit_valid", data=np.ones(8, dtype=np.uint8))

    _MOD.write_beta(path, "left", estimate, force=False)
    assert _MOD.has_mano_beta(path, "left")
    with h5py.File(path, "r") as f:
        reject_external_links(f)
        group = f["hands/left"]
        assert group["mano_beta"].shape == (10,)
        assert group["mano_beta"].attrs["samples_used"] == 6
        assert not any(name in group for name in _MOD._LEGACY_DERIVED_FIELDS)


def test_write_rejects_current_beta_unless_force(tmp_path):
    path = tmp_path / "take.h5"
    _make_h5(path, "left", n=4)
    estimate, error = _MOD.estimate_side(path, "left", samples=4)
    assert error is None and estimate is not None
    _MOD.write_beta(path, "left", estimate, force=False)
    with pytest.raises(FileExistsError):
        _MOD.write_beta(path, "left", estimate, force=False)

    replacement = replace(estimate, beta=np.zeros(10, dtype=np.float32))
    _MOD.write_beta(path, "left", replacement, force=True)
    with h5py.File(path, "r") as f:
        assert np.allclose(f["hands/left/mano_beta"][:], 0.0)


def test_extract_hdf5_loads_skeleton_and_beta_only(tmp_path):
    path = tmp_path / "minimal.h5"
    timestamps = np.arange(3, dtype=np.int64)
    with h5py.File(path, "w") as f:
        hands = f.create_group("hands")
        for side in ("left", "right"):
            group = hands.create_group(side)
            group.create_dataset("t_ubuntu_ns", data=timestamps)
            group.create_dataset("mano_skeleton", data=np.zeros((3, 21, 3)))
            group.create_dataset("mano_beta", data=np.zeros(10))

        data = extract_hdf5(f)
    assert data["hands"]["left"]["mano_beta"].shape == (10,)
    assert data["hands"]["left"]["nodes"].shape == (3, 21, 3)
    assert "mano" not in data["hands"]["left"]


def test_extract_hdf5_rejects_invalid_beta_shape(tmp_path):
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as f:
        hands = f.create_group("hands")
        for side in ("left", "right"):
            group = hands.create_group(side)
            group.create_dataset("t_ubuntu_ns", data=np.arange(2))
            group.create_dataset("mano_skeleton", data=np.zeros((2, 21, 3)))
            group.create_dataset("mano_beta", data=np.zeros(10))
        del hands["left/mano_beta"]
        hands["left"].create_dataset("mano_beta", data=np.zeros(3))
        with pytest.raises(ValueError, match="mano_beta"):
            extract_hdf5(f)


def test_estimate_side_rejects_missing_skeleton(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_group("hands").create_group("right")
    estimate, error = _MOD.estimate_side(path, "right", samples=4)
    assert estimate is None
    assert "缺少" in error


def test_estimate_side_fallback_nodes_global(tmp_path):
    """旧 schema 的 25 点只用于 beta 估计时转换为 21 点。"""
    from mano_fit import load_mano
    from acquisition.manus_schema import MEDIAPIPE_FROM_MANUS

    path = tmp_path / "old.h5"
    layer = load_mano("right")
    _, joints = layer.forward(np.zeros(45), np.zeros(10), np.zeros(3), np.zeros(3))
    nodes25 = np.zeros((5, 25, 3), dtype=np.float32)
    nodes25[:, np.asarray(MEDIAPIPE_FROM_MANUS, dtype=np.int64), :] = joints
    with h5py.File(path, "w") as f:
        group = f.create_group("hands").create_group("right")
        group.create_dataset("nodes_global", data=nodes25)

    estimate, error = _MOD.estimate_side(path, "right", samples=4)
    assert error is None
    assert estimate is not None
    assert "25→21" in estimate.skeleton_source
    assert estimate.beta.shape == (10,)


def test_collect_h5_recursive_dedup(tmp_path):
    directory = tmp_path / "20260812"
    directory.mkdir()
    first = directory / "take1.h5"
    second = directory / "take2.h5"
    for path in (first, second):
        with h5py.File(path, "w") as f:
            f.create_group("hands")
    assert _MOD.collect_h5([str(tmp_path), str(first)]) == [first, second]
