from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_viewer.adapters.loader import load_adapter
from data_viewer.contracts import ViewerConfig


class H5AdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "take001.h5"
        timestamps = np.asarray([1_000_000_000, 1_010_000_000, 1_020_000_000], dtype=np.int64)
        with h5py.File(self.path, "w") as f:
            hands = f.create_group("hands")
            for side in ("left", "right"):
                group = hands.create_group(side)
                group.create_dataset("t_ubuntu_ns", data=timestamps)
                group.create_dataset(
                    "mano_skeleton",
                    data=np.zeros((3, 21, 3), dtype=np.float32),
                )
                group.create_dataset(
                    "mano_beta", data=np.zeros(10, dtype=np.float32),
                )
            objects = f.create_group("objects")
            cylinder = objects.create_group("cylinder")
            cylinder.create_dataset("t_ubuntu_ns", data=timestamps)
            cylinder.create_dataset(
                "position", data=np.zeros((3, 3), dtype=np.float64)
            )
            cylinder.create_dataset(
                "quaternion_xyzw",
                data=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
            )
            cylinder.create_dataset(
                "tracking_valid", data=np.asarray([True, True, True])
            )

        adapter_path = Path(__file__).resolve().parents[2] / "scripts" / "h5_viewer_adapter.py"
        self.adapter = load_adapter(
            self.root,
            ViewerConfig(adapter_path=adapter_path),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_indexes_metadata_and_cached_recording(self):
        index = self.adapter.index()
        self.assertEqual(index.sample_count if hasattr(index, "sample_count") else len(index.samples), 1)
        sample = index.samples[0]
        self.assertEqual(sample.label, "take001")
        self.assertIn("0.0s", sample.summary)

        detail = self.adapter.metadata(sample.id)
        self.assertEqual(detail.metadata["scene"]["objects"], ["cylinder"])
        self.assertEqual(detail.metadata["scene"]["mocap_frames"], 3)
        self.assertTrue(detail.metadata["mano"]["available"])
        self.assertEqual(detail.metadata["mano"]["sides"]["left"]["shape"], [10])

        with tempfile.TemporaryDirectory() as cache:
            cache_dir = Path(cache)
            playback = self.adapter.build_viser(sample.id, cache_dir)
            self.assertTrue(playback.recording_path.is_file())
            self.assertGreater(playback.recording_path.stat().st_size, 0)
            mtime_ns = playback.recording_path.stat().st_mtime_ns
            cached = self.adapter.build_viser(sample.id, cache_dir)
            self.assertEqual(cached.recording_path, playback.recording_path)
            self.assertEqual(cached.recording_path.stat().st_mtime_ns, mtime_ns)

    def test_mano_mode_loads_beta_and_caches_separate_recording(self):
        sample = self.adapter.index().samples[0]
        with tempfile.TemporaryDirectory() as cache:
            playback = self.adapter.build_viser_with_mode(
                sample.id, Path(cache), "mano",
            )
            self.assertTrue(playback.recording_path.name.endswith(".mano.viser"))
            self.assertTrue(playback.recording_path.is_file())
            self.assertGreater(playback.recording_path.stat().st_size, 0)

    def test_mano_mode_rejects_invalid_beta_shape(self):
        with h5py.File(self.path, "r+") as f:
            del f["hands/left/mano_beta"]
            f["hands/left"].create_dataset(
                "mano_beta", data=np.zeros(3, dtype=np.float32),
            )
        sample = self.adapter.index().samples[0]
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaisesRegex(ValueError, "mano_beta"):
                self.adapter.build_viser_with_mode(
                    sample.id, Path(cache), "mano",
                )

if __name__ == "__main__":
    unittest.main()
