import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendStaticTest(unittest.TestCase):
    def test_frontend_is_generic_viser_metadata_viewer(self):
        app = (ROOT / "data_viewer" / "static" / "app.js").read_text(encoding="utf-8")
        html = (ROOT / "data_viewer" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("/api/viser?id=", app)
        self.assertIn("playbackPath=", app)
        self.assertIn("metadata-panel", html)
        self.assertNotIn("SPIDER", app + html)
        self.assertNotIn("camera_lift", app + html)
        self.assertNotIn("embodiment", app + html)
        self.assertNotIn("raw/processed", app + html)

    def test_frontend_has_no_external_font_or_cdn_links(self):
        html = (ROOT / "data_viewer" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("fonts.googleapis.com", html)
        self.assertNotIn("https://", html)

    def test_frontend_guards_against_stale_sample_responses(self):
        app = (ROOT / "data_viewer" / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("selectionToken", app)
        self.assertIn("if (token !== state.selectionToken) return", app)

    def test_frontend_filters_samples_by_object(self):
        app = (ROOT / "data_viewer" / "static" / "app.js").read_text(encoding="utf-8")
        html = (ROOT / "data_viewer" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="object-filter"', html)
        self.assertIn("populateObjectFilter", app)
        self.assertIn("els.objectFilter.value", app)
        self.assertIn("sample.facets?.dataset", app)
        self.assertIn("sample.facets?.object", app)
        self.assertIn("dataset-group", app)
        self.assertNotIn("object-group", app)


if __name__ == "__main__":
    unittest.main()
