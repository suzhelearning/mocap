"""MANO 前向、快速姿态初始化和拟合契约测试。"""

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


def test_batch_forward_and_skeleton_pose_are_finite():
    layer = _MANO.load_mano("right")
    theta = np.zeros((3, 45))
    roots = np.zeros((3, 3))
    trans = np.zeros((3, 3))
    verts, joints = layer.forward(theta, np.zeros(10), roots, trans)

    assert verts.shape == (3, 778, 3)
    assert joints.shape == (3, 21, 3)
    params = _MANO.pose_from_skeleton(layer, joints[0])
    assert params.shape == (52,)
    assert np.isfinite(params).all()
    assert params[51] > 0.0


def test_fit_mesh_recovers_synthetic_pose():
    layer = _MANO.load_mano("right")
    root = np.asarray([0.10, -0.20, 0.05])
    theta = np.zeros(45)
    theta[:6] = [0.20, 0.10, -0.10, -0.15, 0.05, 0.08]
    trans = np.asarray([0.30, 1.10, -0.20])
    _, joints = layer.forward(theta, np.zeros(10), root, trans)
    scale = 1.30
    observed = joints[0] + scale * (joints - joints[0])

    verts, fitted, params = _MANO.fit_mesh(
        layer, observed, iters=6,
    )

    assert verts.shape == (778, 3)
    assert fitted.shape == (21, 3)
    assert params.shape == (52,)
    assert np.linalg.norm(fitted - observed) / np.sqrt(21) < 1e-3
    assert np.allclose(fitted[0], observed[0], atol=1e-6)


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
