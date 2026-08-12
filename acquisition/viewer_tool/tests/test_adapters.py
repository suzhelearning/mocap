import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from data_viewer.adapters.default import DefaultAdapter
from data_viewer.adapters.loader import load_adapter
from data_viewer.contracts import ViewerConfig


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_adapter_indexes_viser_recordings_and_metadata(self):
        sample_dir = self.root / "runs" / "sample_a"
        sample_dir.mkdir(parents=True)
        (sample_dir / "playback.viser").write_bytes(b"viser-recording")
        (sample_dir / "metadata.json").write_text(
            json.dumps({"score": 0.91, "task": "lift"}), encoding="utf-8"
        )

        adapter = DefaultAdapter(self.root, ViewerConfig())
        index = adapter.index()
        detail = adapter.metadata(index.samples[0].id)
        playback = adapter.build_viser(index.samples[0].id, self.root / ".cache")

        self.assertEqual(index.title, self.root.name)
        self.assertEqual(len(index.samples), 1)
        self.assertEqual(index.samples[0].label, "runs/sample_a/playback.viser")
        self.assertEqual(detail.metadata["metadata.json"]["score"], 0.91)
        self.assertEqual(playback.recording_path, sample_dir / "playback.viser")

    def test_default_adapter_reuses_index_for_sample_lookup(self):
        sample_dir = self.root / "runs" / "sample_a"
        sample_dir.mkdir(parents=True)
        playback_path = sample_dir / "playback.viser"
        playback_path.write_bytes(b"viser-recording")

        adapter = DefaultAdapter(self.root, ViewerConfig())
        index = adapter.index()
        playback_path.unlink()

        detail = adapter.metadata(index.samples[0].id)

        self.assertEqual(detail.sample.id, "runs/sample_a/playback.viser")

    def test_load_adapter_uses_project_viewer_adapter_when_present(self):
        (self.root / "viewer_adapter.py").write_text(
            textwrap.dedent(
                """
                from data_viewer.contracts import ProjectIndex, ViewerAdapter

                class CustomAdapter(ViewerAdapter):
                    def index(self):
                        return ProjectIndex(title="custom", root=".", samples=[])

                    def metadata(self, sample_id):
                        return {}

                    def build_viser(self, sample_id, cache_dir):
                        raise KeyError(sample_id)

                def create_adapter(root, config):
                    return CustomAdapter()
                """
            ),
            encoding="utf-8",
        )

        adapter = load_adapter(self.root, ViewerConfig())

        self.assertEqual(adapter.index().title, "custom")

    def test_load_adapter_uses_explicit_adapter_path(self):
        adapter_path = self.root / "custom_adapter.py"
        adapter_path.write_text(
            textwrap.dedent(
                """
                from data_viewer.contracts import ProjectIndex, ViewerAdapter

                class CustomAdapter(ViewerAdapter):
                    def index(self):
                        return ProjectIndex(title="explicit", root=".", samples=[])

                    def metadata(self, sample_id):
                        return {}

                    def build_viser(self, sample_id, cache_dir):
                        raise KeyError(sample_id)

                def create_adapter(root, config):
                    return CustomAdapter()
                """
            ),
            encoding="utf-8",
        )

        adapter = load_adapter(self.root, ViewerConfig(adapter_path=adapter_path))

        self.assertEqual(adapter.index().title, "explicit")

    def test_load_adapter_auto_selects_hdf5_adapter(self):
        (self.root / "take001.h5").write_bytes(b"not parsed until indexing")

        adapter = load_adapter(self.root, ViewerConfig())

        self.assertEqual(type(adapter).__name__, "H5ViewerAdapter")

if __name__ == "__main__":
    unittest.main()
