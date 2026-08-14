from __future__ import annotations

from pathlib import Path

import numpy as np

from acquisition.viser_core import find_object_mesh, load_object_mesh


def test_cylinder_asset_is_meter_scale_and_matches_object_name() -> None:
    loaded = load_object_mesh("cylinder")
    assert loaded is not None
    path, vertices, faces = loaded

    assert path.name == "cylinder_m.obj"
    assert vertices.shape == (810, 3)
    assert faces.shape == (1616, 3)
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [0.018, 0.150, 0.018])
    np.testing.assert_allclose(
        (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0,
        [5e-6, 0.0, 5e-6],
        atol=1e-9,
    )


def test_obj_loader_triangulates_faces_and_resolves_negative_indices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "part_m.obj"
    path.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "f -4 -3 -2 -1\n",
        encoding="utf-8",
    )

    loaded = load_object_mesh("part", tmp_path)
    assert loaded is not None
    _, vertices, faces = loaded
    assert vertices.shape == (4, 3)
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])


def test_plain_obj_vertices_are_converted_from_millimeters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "part.obj"
    path.write_text(
        "v 0 0 0\nv 1000 0 0\nv 0 1000 0\nf 1 2 3\n",
        encoding="utf-8",
    )

    loaded = load_object_mesh("part", tmp_path)
    assert loaded is not None
    _, vertices, _faces = loaded
    np.testing.assert_allclose(vertices[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(vertices[2], [0.0, 1.0, 0.0])


def test_object_name_cannot_escape_mesh_directory(tmp_path: Path) -> None:
    (tmp_path / "safe_m.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    assert find_object_mesh("../safe", tmp_path) is None
