from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.track_rigid import (
    measure_displacement,
    read_records,
    resolve_ids,
    select_rigids,
)

_NAMES = {"names": {"10": "right_arm", "3": "cylinder"}}

_FRAME = {
    "frame_number": 7,
    "motive_timestamp": 123.0,
    "rigid_bodies": [
        {
            "id": 10,
            "position": [0.1, 0.2, 0.3],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "mean_error": 0.0001,
            "tracking_valid": True,
        },
        {
            "id": 3,
            "position": [0.5, 0.5, 0.5],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "mean_error": 0.0002,
            "tracking_valid": True,
        },
        {"id": 99, "position": [0.0, 0.0, 0.0]},
    ],
}


class ResolveIdsTest(unittest.TestCase):
    def test_resolves_names_from_payload(self) -> None:
        self.assertEqual(resolve_ids(_NAMES, ["right_arm"], []), {10})

    def test_merges_names_and_ids(self) -> None:
        self.assertEqual(resolve_ids(_NAMES, ["right_arm"], [3]), {3, 10})

    def test_missing_name_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "right_hand"):
            resolve_ids(_NAMES, ["right_hand"], [])

    def test_no_names_no_ids_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "names"):
            resolve_ids(_NAMES, [], [])

    def test_none_payload_with_ids_ok(self) -> None:
        self.assertEqual(resolve_ids(None, [], [10]), {10})


class SelectRigidsTest(unittest.TestCase):
    def test_selects_only_wanted_ids(self) -> None:
        selected = select_rigids(_FRAME, {10})
        self.assertEqual([body["id"] for body in selected], [10])
        self.assertEqual(selected[0]["position"], [0.1, 0.2, 0.3])

    def test_skips_malformed_bodies(self) -> None:
        selected = select_rigids(_FRAME, {99})
        self.assertEqual(selected, [])


class MeasureDisplacementTest(unittest.TestCase):
    def _records(self) -> list[dict]:
        records = []
        for index in range(10):
            records.append({
                "t_ns": index * 100_000_000,  # 0.1s 间隔
                "id": 10,
                "position": [0.0 + index * 0.01, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "mean_error": 0.0001,
                "tracking_valid": True,
            })
        return records

    def test_displacement_norm_and_axes(self) -> None:
        records = self._records()
        result = measure_displacement(records, start_s=0.0, end_s=0.9)
        self.assertAlmostEqual(result["displacement_mm"], 90.0, places=3)
        self.assertAlmostEqual(result["delta_mm"][0], 90.0, places=3)
        self.assertEqual(result["valid_frames"], 10)
        self.assertAlmostEqual(result["mean_error_mean_mm"], 0.1)

    def test_window_filters_records(self) -> None:
        records = self._records()
        result = measure_displacement(records, start_s=0.3, end_s=0.5)
        # 窗口内首帧 index=3 (0.03m)，末帧 index=5 (0.05m)
        self.assertAlmostEqual(result["displacement_mm"], 20.0, places=3)
        self.assertEqual(result["window_frames"], 3)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "end_s"):
            measure_displacement(self._records(), start_s=1.0, end_s=0.5)

    def test_no_valid_frames_raises(self) -> None:
        records = self._records()
        for record in records:
            record["tracking_valid"] = False
        with self.assertRaisesRegex(ValueError, "有效帧不足"):
            measure_displacement(records, start_s=0.0, end_s=0.9)


class ReadRecordsTest(unittest.TestCase):
    def test_roundtrip_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cap.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for index in range(3):
                    stream.write(
                        json.dumps({"t_ns": index, "position": [0, 0, 0]})
                        + "\n"
                    )
            records = read_records(path)
            self.assertEqual(len(records), 3)

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "没有任何记录"):
                read_records(path)


if __name__ == "__main__":
    unittest.main()
