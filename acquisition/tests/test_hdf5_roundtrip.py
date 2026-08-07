"""TakeWriter HDF5 写入 → 读回校验;保存改名 / 丢弃删除。"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from acquisition.config import Config, load_config
from acquisition.recorder import EV_START, TakeWriter
from acquisition.stitching import extract_rigid_body, stitch_hand
from acquisition.kinematics import compose_axis

TEST_CONFIG = """
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
"""


@pytest.fixture
def cfg(tmp_path) -> Config:
    import os

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(TEST_CONFIG)
    cfg = load_config(cfg_path)
    cfg = Config(
        **{**cfg.__dict__, "output_dir": tmp_path / "captures"},
    )
    return cfg


def _mocap_frame(number: int, t: float) -> dict:
    """back 刚体(id 5)+ 物体刚体(id 11)+ 2 个 markers。"""
    return {
        "schema_version": 1,
        "frame_number": number,
        "motive_timestamp": t,
        "publisher_received_time_ns": 0,
        "coordinate_system": "motive_y_up_right_handed",
        "unit": "meter",
        "publisher_dropped_frames": 0,
        "markers": [
            {"raw_id": 101, "model_id": 0, "member_id": 101, "id_kind": "point_cloud",
             "position": [0.1, 0.2, 0.3], "size": 0.009, "residual_m_per_ray": 0.0002,
             "occluded": False, "point_cloud_solved": True, "model_filled": False,
             "has_model": False, "unlabeled": True, "active": False,
             "established": True, "measurement": False},
            {"raw_id": 102, "model_id": 0, "member_id": 102, "id_kind": "point_cloud",
             "position": [0.2, 0.1, 0.4], "size": 0.009, "residual_m_per_ray": 0.0002,
             "occluded": True, "point_cloud_solved": True, "model_filled": False,
             "has_model": False, "unlabeled": True, "active": False,
             "established": True, "measurement": False},
        ],
        "rigid_bodies": [
            {"id": 5, "position": [0.3, 1.2, -0.2], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
             "mean_error": 0.0004, "tracking_valid": True},
            {"id": 11, "position": [0.9, 0.7, 0.1], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
             "mean_error": 0.0006, "tracking_valid": number % 2 == 0},
        ],
    }


def _manus_msg(side: str, seq: int) -> dict:
    nodes = np.zeros((25, 3))
    nodes[0] = [0.020, 0.010, 0.000]
    nodes[15] = [0.030, 0.015, 0.090]     # 食指尖 +z 9cm
    return {"glove_id": "aa", "side": side, "seq": seq,
            "nodes": nodes.tolist(), "t_ubuntu_ns": 1_000_000_000 + seq}


def _stitched(msg: dict, cfg: Config):
    frame = _mocap_frame(0, 0.0)
    back = extract_rigid_body(frame, cfg.back_rigid_id)
    assert back is not None
    h = cfg.hands["left" if msg["side"] == "left" else "right"]
    axis = compose_axis(cfg.axis_permutation, cfg.axis_signs)
    nodes = np.asarray(msg["nodes"], dtype=float)
    g, p_w, q_w = stitch_hand(nodes, 0, back[0], back[1], h.wrist_offset, axis)
    return g, p_w, q_w


def _write_take(writer: TakeWriter, cfg: Config, n_mocap: int = 5, n_hand: int = 3) -> None:
    writer.begin(1, start_wall_ns=1_000_000_000)
    writer.append_event(EV_START, "start")
    for i in range(n_mocap):
        fr = _mocap_frame(i, float(i) / 60.0)
        fr["t_ubuntu_ns"] = 2_000_000_000 + i
        writer.append_mocap(fr)
    for side in ("left", "right"):
        for k in range(n_hand):
            msg = _manus_msg(side, 100 + k)
            g, p_w, q_w = _stitched(msg, cfg)
            writer.append_manus(side, msg, p_w, q_w, g)
    writer.flush()          # 模拟主循环周期 flush
    # 追加一批再 flush(验证分批写入)
    for i in range(3):
        fr = _mocap_frame(100 + i, float(100 + i) / 60.0)
        fr["t_ubuntu_ns"] = 3_000_000_000 + i
        writer.append_mocap(fr)
    writer.flush()


def test_save_roundtrip(cfg):
    writer = TakeWriter(cfg.output_dir / "take_001.h5", cfg)
    _write_take(writer, cfg)
    writer.finalize_save()

    assert writer.tmp_path.exists() is False       # 临时文件已改名
    assert writer.path.exists()

    with h5py.File(writer.path, "r") as f:
        assert f.attrs["take_id"] == 1
        assert f.attrs["h5_version"] == "1.0"
        assert "back: 5" in f.attrs["config_yaml"]
        assert f.attrs["end_wall_ns"] > f.attrs["start_wall_ns"]

        mocap = f["mocap"]
        assert mocap["t_ubuntu_ns"].shape == (8,)        # 5 + 3
        assert mocap["frame_number"][-1] == 102
        # vlen 刚体
        ids = mocap["rigid_bodies"]["ids"]
        assert len(ids) == 8
        assert all(len(row) == 2 for row in ids)         # back + cup
        assert np.allclose(mocap["rigid_bodies"]["positions"][0][:3], [0.3, 1.2, -0.2])
        # vlen markers:positions 为 3k 铺平,其余为 k
        mk = mocap["markers"]
        assert all(len(row) == 6 for row in mk["positions"])
        assert all(len(row) == 2 for row in mk["raw_ids"])
        assert mk["occluded"][0][1] == 1

        # 手部:拼接后的手腕节点 == 手腕位姿
        for side in ("left", "right"):
            g = f["hands"][side]
            assert g["t_ubuntu_ns"].shape == (3,)
            assert g["nodes_raw"].shape == (3, 25, 3)
            assert g["nodes_global"].shape == (3, 25, 3)
            # 手腕节点 g_0 恒等于 wrist_position
            assert np.allclose(g["nodes_global"][:, 0, :], g["wrist_position"][:], atol=1e-6)
            # 指尖相对手腕 +9cm(轴变换后 y 方向)
            d = g["nodes_global"][:, 15, :] - g["wrist_position"][:]
            assert np.allclose(d[:, 1], 0.09, atol=1e-5)
            # 原始骨架留底
            assert np.allclose(g["nodes_raw"][0, 15], [0.030, 0.015, 0.090])

        # 物体子表(两批 flush 共 8 帧都有 cup 刚体)
        cup = f["objects"]["cup"]
        assert cup["position"].shape == (8, 3)
        assert set(cup["tracking_valid"][:].tolist()) == {0, 1}

        # 事件
        ev = f["events"]
        assert ev["type"].shape[0] == 1
        assert ev["note"][0] == b"start"     # h5py vlen str 读回为 bytes


def test_discard_removes_file(cfg):
    writer = TakeWriter(cfg.output_dir / "take_002.h5", cfg)
    _write_take(writer, cfg)
    writer.discard()
    assert not writer.path.exists()
    assert not writer.tmp_path.exists()


def test_flush_is_batched(cfg):
    """flush 之间的数据在 finalize 前已落盘(临时文件可读)。"""
    writer = TakeWriter(cfg.output_dir / "take_003.h5", cfg)
    writer.begin(3, start_wall_ns=1_000_000_000)
    fr = _mocap_frame(0, 0.0)
    fr["t_ubuntu_ns"] = 1
    writer.append_mocap(fr)
    writer.flush()
    with h5py.File(writer.tmp_path, "r") as f:
        assert f["mocap"]["t_ubuntu_ns"].shape == (1,)
    writer.discard()
