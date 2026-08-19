from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.calibrate_tcp import (
    calibration_result,
    calibration_residuals,
    umeyama,
    window_centers,
)


def _records_for_centers(
    centers: np.ndarray, frames_per_window: int = 20
) -> list[dict]:
    records = []
    for window_index, center in enumerate(centers):
        for frame in range(frames_per_window):
            records.append({
                "t_ns": (
                    (window_index * frames_per_window + frame)
                    * 100_000_000
                ),
                "position": [
                    center[0] + frame * 0.0001,
                    center[1] - frame * 0.0001,
                    center[2],
                ],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "tracking_valid": True,
            })
    return records


class UmeyamaTest(unittest.TestCase):
    def test_recovers_known_transform(self) -> None:
        # 已知 R（绕 z 旋转 30°）与 t。
        angle = np.deg2rad(30.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.array([0.1, -0.2, 0.3])
        # 非共线、非共面的 8 个点
        src = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.3, 0.7, 0.9],
        ])
        dst = (rotation @ src.T).T + translation
        recovered_r, recovered_t = umeyama(src, dst)
        np.testing.assert_allclose(recovered_r, rotation, atol=1e-9)
        np.testing.assert_allclose(recovered_t, translation, atol=1e-9)

    def test_rejects_insufficient_points(self) -> None:
        with self.assertRaises(ValueError):
            umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


class WindowCentersTest(unittest.TestCase):
    def test_segments_into_pair_count_windows(self) -> None:
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        records = _records_for_centers(centers, frames_per_window=20)
        result = window_centers(records, 4)
        self.assertEqual(result.shape, (4, 3))
        # 窗口均值含帧内漂移（每帧 ±0.0001m × 平均 9.5 帧），容差 2mm。
        np.testing.assert_allclose(result[0], centers[0], atol=2e-3)
        np.testing.assert_allclose(result[3], centers[3], atol=2e-3)


class CalibrationResultTest(unittest.TestCase):
    def test_full_pipeline_recovers_transform(self) -> None:
        rotation = np.eye(3)
        translation = np.array([0.5, 0.0, -0.2])
        centers = np.array([
            [0.0, 0.0, 0.0], [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0], [0.0, 0.0, 0.4],
        ])
        motive = _records_for_centers(centers)
        robot = _records_for_centers(
            (rotation @ centers.T).T + translation
        )
        result = calibration_result(motive, robot, pair_count=4)
        np.testing.assert_allclose(
            np.asarray(result["rotation"]), rotation, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(result["translation_m"]), translation, atol=1e-6
        )
        self.assertLess(result["residuals"]["position_error_max_mm"], 0.1)

    def test_residuals_report_numeric_error(self) -> None:
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        dst = src + np.array([0.01, 0.0, 0.0])
        rotation, translation = umeyama(src, dst)
        residuals = calibration_residuals(
            src, dst, rotation, translation
        )
        self.assertAlmostEqual(
            residuals["position_error_mean_mm"], 0.0, places=6
        )

    def test_roundtrip_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            motive_path = Path(tmp) / "motive.jsonl"
            robot_path = Path(tmp) / "robot.jsonl"
            centers = np.array([
                [0.0, 0.0, 0.0], [0.4, 0.0, 0.0],
                [0.0, 0.4, 0.0], [0.0, 0.0, 0.4],
            ])
            with motive_path.open("w") as stream:
                for record in _records_for_centers(centers):
                    stream.write(json.dumps(record) + "\n")
            with robot_path.open("w") as stream:
                for record in _records_for_centers(centers + 0.1):
                    stream.write(json.dumps(record) + "\n")
            from scripts.calibrate_tcp import main
            from scripts.track_rigid import read_records

            motive = read_records(motive_path)
            robot = read_records(robot_path)
            result = calibration_result(motive, robot, pair_count=4)
            np.testing.assert_allclose(
                np.asarray(result["translation_m"]), [0.1, 0.1, 0.1],
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
