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
    assert cfg.keymap == {"start": "r", "save": "s",
                          "discard": "d", "quit": "q"}


def test_no_back_with_all_wrist_rigid_ids():
    """Motive 直接追踪双手腕(无背部刚体):back 可省略,wrist_rigid_id 全配。"""
    cfg_text = """
router:
  endpoint: "tcp/127.0.0.1:7447"
rigid_bodies:
  objects:
    cylinder: 3
hands:
  left:
    wrist_rigid_id: 2
  right:
    wrist_rigid_id: 1
"""
    cfg = _load(cfg_text)
    assert cfg.back_rigid_id is None
    assert cfg.objects == {"cylinder": 3}
    assert cfg.hands["left"].back_rigid_id is None
    assert cfg.hands["left"].wrist_rigid_id == 2
    assert cfg.hands["right"].wrist_rigid_id == 1


def test_no_back_requires_wrist_rigid_id():
    """无背部刚体时,任一 hand 缺 wrist_rigid_id 必须报错。"""
    bad = """
router:
  endpoint: "tcp/127.0.0.1:7447"
rigid_bodies:
  objects: {}
hands:
  left:
    wrist_rigid_id: 2
  right: {}
"""
    with pytest.raises(ConfigError, match="wrist_rigid_id"):
        _load(bad)


def test_back_rigid_id_may_reference_any_rigid():
    """hands.back_rigid_id 允许引用任意 Motive 刚体 ID(如手套背面 marker 刚体),
    不要求出现在 rigid_bodies.back/objects 中。"""
    cfg_text = """
router:
  endpoint: "tcp/127.0.0.1:7447"
rigid_bodies:
  back: 1
  objects:
    cylinder: 3
hands:
  left:
    back_rigid_id: 2              # 未在 back/objects 声明,但合法(手部基准刚体)
    wrist_offset: { mode: body, xyz: [0.0, 0.0, 0.0] }
  right:
    back_rigid_id: 1
    wrist_offset: { mode: body, xyz: [0.0, 0.0, 0.0] }
"""
    cfg = _load(cfg_text)
    assert cfg.hands["left"].back_rigid_id == 2
    assert cfg.hands["right"].back_rigid_id == 1
    assert cfg.back_rigid_id == 1
    assert cfg.objects == {"cylinder": 3}


def test_hand_needs_pose_source():
    """每只手必须指定 back_rigid_id 或 wrist_rigid_id 至少一个。"""
    bad = """
router:
  endpoint: "tcp/127.0.0.1:7447"
rigid_bodies:
  back: 5
  objects: {}
hands:
  left:
    wrist_rigid_id: 2
  right: {}
"""
    with pytest.raises(ConfigError, match="back_rigid_id 或 wrist_rigid_id"):
        _load(bad)


def test_user_offset_merge(tmp_path):
    """offset/<user>.yaml 按用户覆盖 hands.<side>.wrist_offset(与 manus --user 一致)。"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("""
router:
  endpoint: "tcp/127.0.0.1:7447"
user: syz
rigid_bodies:
  back: 2
  objects: { cylinder: 3 }
hands:
  left:  { back_rigid_id: 2, wrist_offset: { mode: body, xyz: [0, 0, 0] } }
  right: { back_rigid_id: 1, wrist_offset: { mode: body, xyz: [0, 0, 0] } }
axis_transform: { permutation: [0, 2, 1], signs: [1, 1, -1] }
recording: { output_dir: "captures", store_markers: false, sample_hz: 100 }
""")
    (tmp_path / "offset").mkdir()
    (tmp_path / "offset" / "syz.yaml").write_text("""
left:
  mode: body
  xyz: [0.021, -0.0125, 0.0148]
  yaw_deg: 8.2
  pitch_deg: -5.1
  roll_deg: 3.0
""")
    cfg = load_config(cfg_path)
    assert cfg.user == "syz"
    lo = cfg.hands["left"].wrist_offset
    assert lo.xyz == (0.021, -0.0125, 0.0148)
    assert lo.yaw_deg == 8.2 and lo.pitch_deg == -5.1 and lo.roll_deg == 3.0
    # 未标定的 right 保持配置默认
    assert cfg.hands["right"].wrist_offset.xyz == (0.0, 0.0, 0.0)


def test_user_offset_ignored_when_user_missing(tmp_path):
    """无对应 offset 文件(或 user 不同)时不合并。"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("""
router:
  endpoint: "tcp/127.0.0.1:7447"
user: shd
rigid_bodies:
  back: 2
  objects: {}
hands:
  left:  { back_rigid_id: 2 }
  right: { back_rigid_id: 1 }
axis_transform: { permutation: [0, 2, 1], signs: [1, 1, -1] }
recording: { output_dir: "captures", store_markers: false, sample_hz: 100 }
""")
    (tmp_path / "offset").mkdir()
    (tmp_path / "offset" / "syz.yaml").write_text("left:\n  xyz: [9, 9, 9]\n")
    cfg = load_config(cfg_path)
    assert cfg.user == "shd"
    assert cfg.hands["left"].wrist_offset.xyz == (0.0, 0.0, 0.0)
