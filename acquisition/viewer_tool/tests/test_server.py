import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from data_viewer.adapters.loader import load_adapter
from data_viewer.contracts import ViewerConfig, ViserPlayback
from data_viewer.server import ApiState, default_data_root, file_response_info, make_api_response


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        sample_dir = self.root / "sample"
        sample_dir.mkdir()
        (sample_dir / "motion.viser").write_bytes(b"0123456789")
        (sample_dir / "metadata.json").write_text(
            json.dumps({"frames": 10}), encoding="utf-8"
        )
        adapter = load_adapter(self.root, ViewerConfig())
        self.state = ApiState(root=self.root, adapter=adapter)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_data_root_uses_yyyymmdd_folder(self):
        root = default_data_root(date(2026, 8, 12))

        self.assertEqual(root.name, "20260812")
    def test_api_load_sample_and_viser_playback(self):
        load_payload = make_api_response(self.state, "POST", "/api/load", {})
        self.assertEqual(load_payload["kind"], "project_root")
        self.assertEqual(load_payload["index"]["sample_count"], 1)

        sample_id = load_payload["index"]["samples"][0]["id"]
        detail_payload = make_api_response(
            self.state, "GET", f"/api/sample?id={sample_id}", None
        )
        self.assertEqual(detail_payload["metadata"]["metadata.json"]["frames"], 10)

        viser_payload = make_api_response(
            self.state, "GET", f"/api/viser?id={sample_id}", None
        )
        self.assertEqual(viser_payload["client"], "/viser-client/")
        self.assertEqual(viser_payload["path"], "/file?path=sample/motion.viser")

    def test_file_response_info_supports_byte_ranges(self):
        video = self.root / "sample" / "motion.viser"

        info = file_response_info(video, "bytes=2-5")

        self.assertEqual(info.status, 206)
        self.assertEqual(info.start, 2)
        self.assertEqual(info.end, 5)
        self.assertEqual(info.length, 4)
        self.assertEqual(info.headers["Content-Range"], "bytes 2-5/10")

    def test_file_response_info_rejects_invalid_byte_ranges(self):
        video = self.root / "sample" / "motion.viser"

        invalid = file_response_info(video, "bytes=abc-def")
        unsatisfied = file_response_info(video, "bytes=20-30")

        self.assertEqual(invalid.status, 416)
        self.assertEqual(invalid.length, 0)
        self.assertEqual(invalid.headers["Content-Range"], "bytes */10")
        self.assertEqual(unsatisfied.status, 416)
        self.assertEqual(unsatisfied.length, 0)
        self.assertEqual(unsatisfied.headers["Content-Range"], "bytes */10")

    def test_viser_payload_encodes_file_paths_with_reserved_characters(self):
        sample_dir = self.root / "reserved names"
        sample_dir.mkdir()
        recording = sample_dir / "motion & pose.viser"
        recording.write_bytes(b"recording")

        class ReservedPathAdapter:
            def index(self):
                raise AssertionError("not needed")

            def metadata(self, sample_id):
                raise AssertionError("not needed")

            def build_viser(self, sample_id, cache_dir):
                return ViserPlayback(recording_path=recording, label="reserved")

        state = ApiState(root=self.root, adapter=ReservedPathAdapter())

        payload = make_api_response(state, "GET", "/api/viser?id=sample", None)

        self.assertEqual(payload["path"], "/file?path=reserved%20names/motion%20%26%20pose.viser")


if __name__ == "__main__":
    unittest.main()
