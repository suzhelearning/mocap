"""config.py 加载与校验测试。"""

from __future__ import annotations

import numpy as np
import pytest

from acquisition.config import ConfigError, load_config

SAMPLE = """
router:
  endpoint: "tcp/127.0.0.1:7447"
rigid_bodies:
  back: 5
  objects:
    cup: 11
hands:
  left:
    back_rigid_id: 5
    wrist_offset: { mode: body, xyz: [0.15, -0.30, -0.05] }
  right:
    back_rigid_id: 5
    wrist_offset: { mode: body, xyz: [-0.15, -0.30, -0.05] }
axis_transform:
  permutation: [0, 2, 1]
  signs: [1, 1, -1]
recording:
  output_dir: "captures"
  store_markers: true
keys:
  start: r
  pause: " "
  save: s
  discard: d
  quit: q
viz:
  port: 8081
"""


def _load(text: str):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return load_config(path)
    finally:
        import os

        os.unlink(path)


def test_load_sample():
    cfg = _load(SAMPLE)
    assert cfg.router_endpoint == "tcp/127.0.0.1:7447"
    assert cfg.back_rigid_id == 5
    assert cfg.objects == {"cup": 11}
    assert set(cfg.hands) == {"left", "right"}
    assert cfg.hands["left"].wrist_offset.xyz == (0.15, -0.30, -0.05)
    assert cfg.keymap["save"] == "s"
    assert cfg.viz_port == 8081


def test_axis_matrix_default():
    cfg = _load(SAMPLE)
    A = cfg.axis_matrix()
    # A·d = (d_x, d_z, -d_y)
    assert np.allclose(A @ np.array([1.0, 2.0, 3.0]), [1.0, 3.0, -2.0])
    assert np.isclose(np.linalg.det(A), 1.0)


def test_missing_hand():
    """缺 left 时必须报错。"""
    text = """
router:
  endpoint: "tcp/127.0.0.1:7447"
rigid_bodies:
  back: 5
  objects: {}
hands:
  right:
    back_rigid_id: 5
"""
    with pytest.raises(ConfigError, match="hands 缺少 left"):
        _load(text)


def test_mirror_axis_rejected():
    """perm [0,2,1] + signs [1,1,1] 合成 det=-1(镜像),必须拒绝。"""
    bad = SAMPLE.replace("signs: [1, 1, -1]", "signs: [1, 1, 1]")
    with pytest.raises(ConfigError, match="不是真旋转"):
        _load(bad)


def test_object_id_collides_with_back():
    bad = SAMPLE.replace("cup: 11", "cup: 5")
    with pytest.raises(ConfigError, match="与背部刚体重复"):
        _load(bad)


def test_duplicate_keys_rejected():
    bad = SAMPLE.replace("save: s", "save: r")
    with pytest.raises(ConfigError, match="重复键位"):
        _load(bad)


def test_defaults_applied():
    """缺省字段用默认值。"""
    minimal = """
router:
  endpoint: "tcp/127.0.0.1:7447"
rigid_bodies:
  back: 5
  objects: {}
hands:
  left:
    back_rigid_id: 5
  right:
    back_rigid_id: 5
"""
    cfg = _load(minimal)
    assert cfg.hands["left"].wrist_offset.mode == "body"
    assert cfg.axis_permutation == (0, 2, 1)
    assert cfg.keymap == {"start": "r", "pause": " ", "save": "s",
                          "discard": "d", "quit": "q"}
