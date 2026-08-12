"""状态机转移矩阵测试:注入 FakeWriter 验证全部转移与 writer 生命周期。"""

from __future__ import annotations

from pathlib import Path

from acquisition.config import Config, load_config
from acquisition.recorder import EV_DISCARD, EV_SAVE, EV_START
from acquisition.state_machine import State, TakeController

TEST_CONFIG = """
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


class FakeWriter:
    """记录生命周期调用,不做真实 HDF5 IO。"""

    def __init__(self):
        self.events = []
        self.begin_args = None
        self.saved = False
        self.discarded = False
        self.counts_ = {"mocap": 3, "left": 1, "right": 1}
        self.path = Path("take_fake.h5")

    def begin(self, take_id, start_wall_ns):
        self.begin_args = (take_id, start_wall_ns)
        self.events.append("begin")

    def append_event(self, ev_type, note=""):
        self.events.append(("event", ev_type, note))

    def finalize_save(self):
        self.events.append("save")
        self.saved = True

    def discard(self):
        self.events.append("discard")
        self.discarded = True

    def counts(self):
        return self.counts_


def _make_ctrl(tmp_path) -> tuple[TakeController, list[FakeWriter]]:
    cfg_path = Path(tmp_path) / "cfg.yaml"
    cfg_path.write_text(TEST_CONFIG)
    cfg = load_config(cfg_path)
    writers = []

    def factory(take_id: int, path: Path):
        w = FakeWriter()
        w.path = path
        writers.append(w)
        return w

    return TakeController(cfg, factory, take_dir=tmp_path / "captures"), writers


def test_full_cycle_save(tmp_path):
    ctrl, writers = _make_ctrl(tmp_path)
    assert ctrl.state is State.IDLE
    assert ctrl.handle("r") is True
    assert ctrl.state is State.RECORDING
    assert ctrl.writer is writers[0]
    assert writers[0].begin_args[0] == 1
    assert ("event", EV_START, "start") in writers[0].events

    assert ctrl.handle("s") is True
    assert ctrl.state is State.IDLE
    assert writers[0].saved
    assert not writers[0].discarded
    assert ctrl.writer is None
    assert "saved" in ctrl.last_result


def test_discard(tmp_path):
    ctrl, writers = _make_ctrl(tmp_path)
    ctrl.handle("r")
    assert ctrl.handle("d") is True
    assert ctrl.state is State.IDLE
    assert writers[0].discarded
    assert not writers[0].saved
    assert "discarded" in ctrl.last_result


def test_quit_recording_discards(tmp_path):
    ctrl, writers = _make_ctrl(tmp_path)
    ctrl.handle("r")
    assert ctrl.handle("q") is True
    assert ctrl.state is State.IDLE
    assert writers[0].discarded
    assert ctrl.quit_requested
    assert ("event", EV_DISCARD, "discard") in writers[0].events


def test_quit_idle(tmp_path):
    ctrl, _ = _make_ctrl(tmp_path)
    assert ctrl.handle("q") is True
    assert ctrl.quit_requested
    assert ctrl.state is State.IDLE


def test_ignored_keys(tmp_path):
    ctrl, _ = _make_ctrl(tmp_path)
    assert ctrl.handle("s") is False      # IDLE 下 save 无效果
    assert ctrl.handle("d") is False
    assert ctrl.handle("x") is False      # 未知键
    assert ctrl.handle("r") is True
    assert ctrl.handle("r") is False      # RECORDING 下 start 无效果


def test_multiple_takes_sequential(tmp_path):
    ctrl, writers = _make_ctrl(tmp_path)
    ctrl.handle("r")
    ctrl.handle("s")
    ctrl.handle("r")
    assert ctrl.take_id == 2
    assert writers[0].saved
    assert writers[1].begin_args[0] == 2
    assert writers[0].path != writers[1].path
    assert "_take001.h5" in writers[0].path.name
    assert "_take002.h5" in writers[1].path.name


def test_status_line(tmp_path):
    ctrl, _ = _make_ctrl(tmp_path)
    assert "空闲" in ctrl.status_line()
    ctrl.handle("r")
    assert "录制中" in ctrl.status_line()
    ctrl.handle("s")
    assert "空闲" in ctrl.status_line()
