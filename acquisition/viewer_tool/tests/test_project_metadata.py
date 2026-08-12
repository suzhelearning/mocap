import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectMetadataTest(unittest.TestCase):
    def test_cli_script_points_to_data_viewer_server(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        scripts = pyproject["project"]["scripts"]
        dependencies = pyproject["project"]["dependencies"]
        package_data = pyproject["tool"]["setuptools"]["package-data"]

        self.assertEqual(scripts["data-viewer"], "data_viewer.server:main")
        self.assertIn("viser", dependencies)
        self.assertIn("static/*", package_data["data_viewer"])
        self.assertNotIn("spider_viewer.server:main", scripts.values())


if __name__ == "__main__":
    unittest.main()
