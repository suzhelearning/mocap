"""手腕标定刚体配准与质量门测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_wrist_offset.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_wrist_offset", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_CAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CAL)

calibration_quality = _CAL.calibration_quality
euler_to_rotmat = _CAL.euler_to_rotmat
pose_diversity = _CAL.pose_diversity
quality_errors = _CAL.quality_errors
solve = _CAL.solve


def _rotmat_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    """测试用旋转矩阵转 xyzw。"""
    trace = np.trace(rotation)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        return np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            scale / 4,
        ])
    raise AssertionError("测试旋转须保持正 trace")


def _samples(vectors: list[list[float]], *, outlier: int | None = None):
    offset = np.array([0.025, -0.012, -0.090])
    glove_rotation = euler_to_rotmat(25.0, -18.0, 12.0)
    samples = []
    for index, vector in enumerate(vectors):
        back_rotation = euler_to_rotmat(index * 12.0, index * -6.0, index * 4.0)
        back_position = np.array([0.3 + index * 0.02, 1.1, -0.2])
        ref = back_position + back_rotation @ (offset + glove_rotation @ vector)
        if outlier == index:
            ref = ref + np.array([0.025, 0.0, 0.0])
        samples.append((back_position, _rotmat_to_xyzw(back_rotation),
                        np.asarray(vector), ref))
    return samples


DIVERSE_VECTORS = [
    [0.00, 0.160, 0.000],
    [0.00, 0.145, 0.055],
    [0.00, 0.105, 0.105],
    [0.01, 0.155, 0.020],
    [-0.01, 0.125, 0.085],
]


def test_kabsch_recovers_known_wrist_transform():
    samples = _samples(DIVERSE_VECTORS)
    theta, rms = solve(samples)
    errors, metrics = quality_errors(samples, theta, rms)

    assert rms < 1e-10
    assert np.allclose(theta[:3], [0.025, -0.012, -0.090], atol=1e-9)
    assert errors == []
    assert metrics["direction_max_deg"] < 1e-5
    assert metrics["loo_position_max_mm"] < 1e-5


def test_two_repeated_hand_shapes_are_rejected_as_collinear():
    vectors = [
        [0.0, 0.160, 0.0],
        [0.0, 0.100, 0.100],
        [0.0, 0.160, 0.0],
        [0.0, 0.100, 0.100],
        [0.0, 0.160, 0.0],
    ]
    samples = _samples(vectors)
    theta, rms = solve(samples)
    errors, metrics = quality_errors(samples, theta, rms)

    assert metrics["second_span_mm"] < 3.0
    assert any("近共线" in error for error in errors)


def test_direction_metric_subtracts_wrist_offset():
    samples = _samples(DIVERSE_VECTORS)
    theta, _ = solve(samples)
    metrics = calibration_quality(theta, samples)
    # 不能直接拿 p_ref-p_back 当手指方向；正确指标必须先扣除 offset。
    assert metrics["direction_rms_deg"] < 1e-5


def test_leave_one_out_rejects_single_bad_pose():
    samples = _samples(DIVERSE_VECTORS, outlier=4)
    theta, rms = solve(samples)
    errors, metrics = quality_errors(samples, theta, rms)

    assert metrics["loo_position_max_mm"] > 10.0
    assert errors


def test_pose_diversity_reports_real_direction_coverage():
    diversity = pose_diversity(_samples(DIVERSE_VECTORS))
    assert diversity["spread_deg"] > 25.0
    assert diversity["second_span_mm"] > 3.0
