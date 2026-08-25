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

_CAPTURE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_wrist_landmarks.py"
_CAPTURE_SPEC = importlib.util.spec_from_file_location("capture_wrist_landmarks", _CAPTURE_SCRIPT)
assert _CAPTURE_SPEC is not None and _CAPTURE_SPEC.loader is not None
_CAPTURE = importlib.util.module_from_spec(_CAPTURE_SPEC)
_CAPTURE_SPEC.loader.exec_module(_CAPTURE)

calibration_quality = _CAL.calibration_quality
euler_to_rotmat = _CAL.euler_to_rotmat
fingertip_landmark_samples = _CAL.fingertip_landmark_samples
load_landmarks = _CAL.load_landmarks
pose_diversity = _CAL.pose_diversity
quality_errors = _CAL.quality_errors
solve = _CAL.solve
build_landmark_yaml = _CAPTURE.build_yaml
summarize_landmark_samples = _CAPTURE.summarize


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


def test_repository_fingertip_landmarks_are_complete_ordered_and_noncollinear():
    path = Path(__file__).resolve().parents[1] / "config" / "wrist_landmarks.yaml"
    left = load_landmarks(path, "left")
    right = load_landmarks(path, "right")
    assert [node for _name, node, _xyz in left] == [24, 5, 10, 15, 20]
    assert [node for _name, node, _xyz in right] == [24, 5, 10, 15, 20]
    left_xyz = np.asarray([xyz for _name, _node, xyz in left])
    right_xyz = np.asarray([xyz for _name, _node, xyz in right])
    assert np.all(left_xyz[:, 1] > 0) and np.all(np.diff(left_xyz[:, 1]) > 0)
    assert np.all(right_xyz[:, 1] < 0) and np.all(np.diff(right_xyz[:, 1]) < 0)
    assert np.allclose(left_xyz[:, 2], 0.0) and np.allclose(right_xyz[:, 2], 0.0)
    assert np.linalg.matrix_rank(left_xyz[:, :2] - left_xyz[:, :2].mean(0)) == 2
    assert np.linalg.matrix_rank(right_xyz[:, :2] - right_xyz[:, :2].mean(0)) == 2


def test_capture_landmarks_sorts_y_and_maps_fingertip_nodes():
    # 构造 +Y→-Y 十点;raw_id 故意乱序，排序必须由世界 y 决定。
    samples = {}
    for index, y in enumerate(np.linspace(0.10, -0.10, 10)):
        center = np.array([0.1 + abs(y), y, 0.009])
        values = np.repeat(center[None, :], 20, axis=0)
        samples[100 + (9 - index)] = values
    points = summarize_landmark_samples(samples, max_std_mm=1.0, contact_z_m=0.0)
    assert [item["full_name"] for item in points] == [
        "left_little", "left_ring", "left_middle", "left_index", "left_thumb",
        "right_thumb", "right_index", "right_middle", "right_ring", "right_little",
    ]
    assert [item["node"] for item in points] == [20, 15, 10, 5, 24, 24, 5, 10, 15, 20]
    assert all(np.isclose(item["xyz"][2], 0.0) for item in points)
    data = build_landmark_yaml(points, seconds=3.0, contact_z_m=0.0)
    assert [item["node"] for item in data["hands"]["left"]] == [24, 5, 10, 15, 20]
    assert [item["node"] for item in data["hands"]["right"]] == [24, 5, 10, 15, 20]


def test_fingertip_landmarks_recover_known_back_to_wrist_transform():
    offset = np.array([0.025, -0.012, -0.090])
    wrist_rotation = euler_to_rotmat(25.0, -18.0, 12.0)
    back_rotation = euler_to_rotmat(15.0, -8.0, 5.0)
    back_position = np.array([0.3, 0.1, 0.2])
    nodes = np.zeros((25, 3), dtype=float)
    node_ids = (24, 5, 10, 15, 20)
    vectors = np.asarray([
        [0.030, -0.020, 0.080],
        [0.015, 0.025, 0.155],
        [0.000, 0.045, 0.175],
        [-0.015, 0.030, 0.160],
        [-0.030, 0.005, 0.135],
    ])
    landmarks = []
    for name, node, vector in zip(("thumb", "index", "middle", "ring", "pinky"),
                                  node_ids, vectors):
        nodes[node] = vector
        world = back_position + back_rotation @ (offset + wrist_rotation @ vector)
        landmarks.append((name, node, world))
    frame = (back_position, _rotmat_to_xyzw(back_rotation), nodes, np.zeros(3))
    samples = fingertip_landmark_samples([frame] * 10, landmarks, np.eye(3))
    theta, rms = solve(samples)
    assert rms < 1e-10
    assert np.allclose(theta[:3], offset, atol=1e-9)
    assert np.allclose(euler_to_rotmat(*theta[3:]), wrist_rotation, atol=1e-9)
