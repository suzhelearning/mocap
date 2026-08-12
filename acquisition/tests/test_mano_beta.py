"""mano_beta 离线写入脚本:beta 估计写回 H5、重复拒绝、--force 覆盖。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from acquisition.viser_core import reject_external_links


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mano_beta.py"
_SPEC = importlib.util.spec_from_file_location("mano_beta_test_module", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


def _make_h5(path: Path, side: str, n: int = 10) -> None:
    """构造最小 hands/<side>/mano_skeleton H5(真实 MANO 网格生成,带手形)。"""
    from mano_fit import load_mano

    layer = load_mano(side)
    rng = np.random.default_rng(3)
    beta = rng.uniform(-1.5, 1.5, 10)
    _, joints = layer.forward(
        np.zeros(45), beta, np.zeros(3), np.zeros(3),
    )
    skel = np.repeat(joints[None], n, axis=0)
    with h5py.File(path, "w") as f:
        g = f.create_group("hands").create_group(side)
        g.create_dataset("mano_skeleton", data=skel.astype(np.float32))
    return beta


def test_write_and_read_beta(tmp_path):
    path = tmp_path / "take.h5"
    beta_true = _make_h5(path, "left")
    estimate, error = _MOD.estimate_side(path, "left", samples=8)
    assert error is None
    assert estimate.skeleton_source == "mano_skeleton"
    assert estimate.samples_used == 8
    assert np.abs(estimate.beta - beta_true).max() < 0.3

    _MOD.write_beta(path, "left", estimate, force=False)
    with h5py.File(path, "r") as f:
        reject_external_links(f)
        d = f["hands/left/mano_beta"]
        assert d.shape == (10,)
        assert d.attrs["skeleton"] == "mano_skeleton"
        assert d.attrs["samples_used"] == 8
        assert d.attrs["segment_rms_mm"] >= 0
        assert np.allclose(np.asarray(d[:]), estimate.beta, atol=1e-5)


def test_write_rejects_existing_unless_force(tmp_path):
    from dataclasses import replace

    path = tmp_path / "take.h5"
    _make_h5(path, "left")
    estimate, _ = _MOD.estimate_side(path, "left", samples=4)
    _MOD.write_beta(path, "left", estimate, force=False)
    with pytest.raises(FileExistsError):
        _MOD.write_beta(path, "left", estimate, force=False)
    # --force 覆盖
    estimate2 = replace(estimate, beta=np.zeros(10, dtype=np.float32))
    _MOD.write_beta(path, "left", estimate2, force=True)
    with h5py.File(path, "r") as f:
        assert np.allclose(np.asarray(f["hands/left/mano_beta"][:]), 0.0)


def test_estimate_side_rejects_missing_skeleton(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_group("hands").create_group("right")
    beta, err = _MOD.estimate_side(path, "right", samples=4)
    assert beta is None
    assert "缺少" in err


def test_estimate_side_fallback_nodes_global(tmp_path):
    """旧 schema:只有 nodes_global(25 点)时按 25→21 重排后同样估计。"""
    from mano_fit import load_mano
    from acquisition.manus_schema import MEDIAPIPE_FROM_MANUS

    path = tmp_path / "old.h5"
    layer = load_mano("right")
    _, joints = layer.forward(np.zeros(45), np.zeros(10), np.zeros(3), np.zeros(3))
    nodes25 = np.zeros((5, 25, 3), dtype=np.float32)
    nodes25[:, np.asarray(MEDIAPIPE_FROM_MANUS, dtype=np.int64), :] = joints
    with h5py.File(path, "w") as f:
        g = f.create_group("hands").create_group("right")
        g.create_dataset("nodes_global", data=nodes25)

    estimate, error = _MOD.estimate_side(path, "right", samples=4)
    assert error is None
    assert estimate is not None
    assert "25→21" in estimate.skeleton_source
    assert estimate.beta.shape == (10,)


def test_collect_h5_recursive_dedup(tmp_path):
    (tmp_path / "20260812").mkdir()
    a = tmp_path / "20260812" / "take1.h5"
    b = tmp_path / "20260812" / "take2.h5"
    for p in (a, b):
        with h5py.File(p, "w") as f:
            f.create_group("hands")
    files = _MOD.collect_h5([str(tmp_path), str(a)])
    assert files == [a, b]          # 递归扫描 + 去重保序
