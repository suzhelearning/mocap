"""MANO 前向、beta 估计与关键点直接驱动表面契约测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mano_fit.py"
_SPEC = importlib.util.spec_from_file_location("mano_fit_test_module", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MANO = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MANO
_SPEC.loader.exec_module(_MANO)


@pytest.mark.parametrize("side", ["left", "right"])
def test_rest_pose_reproduces_template(side):
    layer = _MANO.load_mano(side)
    verts, joints = layer.forward(
        np.zeros(45), np.zeros(10), np.zeros(3), np.zeros(3),
    )

    assert verts.shape == (778, 3)
    assert joints.shape == (21, 3)
    assert np.allclose(verts, layer.v_template, atol=1e-12)

@pytest.mark.parametrize("side", ["left", "right"])
def test_mano_joint_reading_matches_regressor_and_official_tips(side):
    """16 个 MANO 原生关节和 5 个官方 fingertip 必须映射到 MP21。"""
    layer = _MANO.load_mano(side)
    verts, joints = layer.forward(
        np.zeros(45), np.zeros(10), np.zeros(3), np.zeros(3),
    )
    native = layer.J_regressor @ layer.v_template
    for native_index, mp_index in enumerate(_MANO._MANO_TO_MP):
        assert np.allclose(joints[mp_index], native[native_index], atol=1e-12)
    assert layer.tips == _MANO._MANO_TIP_VERTICES
    for mp_index, vertex_index in (
        (4, 744), (8, 320), (12, 443), (16, 554), (20, 671),
    ):
        assert np.allclose(joints[mp_index], verts[vertex_index], atol=1e-12)


def test_batch_forward_is_finite():
    layer = _MANO.load_mano("right")
    theta = np.zeros((3, 45))
    roots = np.zeros((3, 3))
    trans = np.zeros((3, 3))
    verts, joints = layer.forward(theta, np.zeros(10), roots, trans)

    assert verts.shape == (3, 778, 3)
    assert joints.shape == (3, 21, 3)
    assert np.isfinite(verts).all()
    assert np.isfinite(joints).all()


@pytest.mark.parametrize("side", ["left", "right"])
def test_mesh_from_skeleton_uses_direct_observed_joints(side):
    layer = _MANO.load_mano(side)
    rng = np.random.default_rng(41)
    beta = rng.uniform(-1.0, 1.0, 10)
    theta = rng.normal(0.0, 0.2, 45)
    expected_verts, skeleton = layer.forward(
        theta,
        beta,
        np.asarray([0.10, -0.20, 0.05]),
        np.asarray([0.30, 1.10, -0.20]),
    )

    verts, joints16 = _MANO.mesh_from_skeleton(layer, skeleton, beta)

    assert verts.shape == (778, 3)
    assert joints16.shape == (16, 3)
    assert np.array_equal(
        joints16, _MANO.mano_joints16_from_joints21(skeleton).astype(np.float32),
    )
    assert np.sqrt(np.mean((verts - expected_verts) ** 2)) < 2e-3


def test_mano_joints16_use_native_kinematic_order():
    layer = _MANO.load_mano("right")
    _, joints21 = layer.forward(
        np.zeros(45), np.zeros(10), np.zeros(3), np.zeros(3),
    )
    joints16 = _MANO.mano_joints16_from_joints21(joints21)
    assert joints16.shape == (16, 3)
    assert len(_MANO.MANO_JOINT_NAMES) == 16
    assert _MANO.MANO_JOINT_NAMES[:4] == (
        "wrist", "index_mcp", "index_pip", "index_dip",
    )
    assert np.allclose(joints16, joints21[_MANO._MANO_TO_MP])


def test_mesh_from_skeleton_rejects_invalid_input():
    layer = _MANO.load_mano("left")
    with pytest.raises(ValueError, match="skeleton"):
        _MANO.mesh_from_skeleton(layer, np.zeros((16, 3)), np.zeros(10))
    with pytest.raises(ValueError, match="beta"):
        _MANO.mesh_from_skeleton(layer, np.zeros((21, 3)), np.zeros(3))
    skeleton = np.zeros((21, 3))
    skeleton[4, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _MANO.mesh_from_skeleton(layer, skeleton, np.zeros(10))


def test_joints21_are_mediapipe_order():
    """J21 输出顺序必须是 MediaPipe:中指(DIP->TIP 段)最长,拇指最短。"""
    layer = _MANO.load_mano("right")
    _, joints = layer.forward(
        np.zeros(45), np.zeros(10), np.zeros(3), np.zeros(3),
    )
    lengths = {name: np.linalg.norm(joints[i] - joints[0])
               for name, i in (("thumb", 4), ("index", 8),
                               ("middle", 12), ("ring", 16), ("pinky", 20))}
    assert lengths["middle"] > lengths["index"] > lengths["ring"] \
        > lengths["pinky"] > lengths["thumb"]


def test_estimate_beta_recovers_synthetic_shape():
    layer = _MANO.load_mano("right")
    rng = np.random.default_rng(11)
    beta_true = rng.uniform(-2.0, 2.0, 10)
    _, joints = layer.forward(
        np.zeros(45), beta_true, np.zeros(3), np.zeros(3),
    )
    beta_est = _MANO.estimate_beta(layer, joints)
    assert beta_est.shape == (10,)
    assert np.abs(beta_est - beta_true).max() < 0.3


def test_estimate_beta_uses_per_frame_lengths_and_rejects_outlier():
    """旋转中的坐标不能先平均；单帧关节异常也不应污染恒定手形。"""
    layer = _MANO.load_mano("right")
    rng = np.random.default_rng(23)
    beta_true = rng.uniform(-1.5, 1.5, 10)
    n = 12
    _, joints = layer.forward(
        np.zeros((n, 45)),
        beta_true,
        rng.normal(0.0, 0.8, (n, 3)),
        rng.normal(0.0, 0.3, (n, 3)),
    )
    joints[-1, 7] += np.array([0.2, -0.1, 0.15])  # 模拟单帧异常关节点

    beta_est = _MANO.estimate_beta(layer, joints)
    assert np.abs(beta_est - beta_true).max() < 0.3
