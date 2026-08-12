"""HDF5 录制器(TakeWriter):一个 take 一个实例。

设计:
- append_* 只入内存缓冲；flush() 在录制主循环批量写入，避免阻塞 Zenoh 回调。
- 临时文件位于 output_dir 下的私有 0700 目录；保存采用同文件系统 hard-link，
  目标已存在时失败且不覆盖既有采集。
- HDF5 schema 2.0 使用 offsets + flat 数组保存可变长度刚体/marker 帧，
  避免 vlen 数组的跨语言读取和追加成本；inspect/replay 兼容旧 v1 文件。

时间基准:
- t_ubuntu_ns：采集端收到消息的 wall-clock；
- t_aligned_ubuntu_ns：用 publisher_received_time_ns 在线估计跨机时钟后的 mocap 轴。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import h5py
import numpy as np

from .config import Config
from .rate import RateGate

H5_VERSION = "2.0"

# events type 枚举(与 state_machine 对应)
EV_START = 0
EV_PAUSE = 1
EV_RESUME = 2
EV_SAVE = 3
EV_DISCARD = 4
EV_QUIT = 5

# id_kind → 数字(markers 存储)
_KIND_INDEX = {"active": 0, "asset_member": 1, "point_cloud": 2, "unknown": 3}


class TakeWriter:
    """流式 HDF5 录制器;begin → append* → finalize_save | discard。"""

    def __init__(self, path: Path, config: Config,
                 rate_gate: RateGate | None = None) -> None:
        self._path = path
        self._tmp_path: Path | None = None      # 创建于私有 0700 目录,随机名
        self._tmp_dir: Path | None = None
        self._cfg = config
        self._rate_gate = rate_gate
        self._f: h5py.File | None = None
        self._lock = threading.Lock()

        self._mocap: list[dict] = []
        self._manus: dict[str, list[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]] = {
            "left": [], "right": [],
        }
        self._events: list[tuple[int, str, int]] = []   # (t_ns, type, note)
        self._counts = {"mocap": 0, "left": 0, "right": 0}
        self._start_ns = 0
        self._ended = False
        self._stream_health_json = "{}"


    def _configured_rigid_body_names(self) -> dict[str, str]:
        names: dict[str, str] = {}
        if self._cfg.back_rigid_id is not None:
            names[str(self._cfg.back_rigid_id)] = "back"
        for name, rigid_id in self._cfg.objects.items():
            names.setdefault(str(rigid_id), name)
        for side, hand in self._cfg.hands.items():
            rigid_id = (
                hand.wrist_rigid_id
                if hand.wrist_rigid_id is not None
                else hand.back_rigid_id
            )
            if rigid_id is not None:
                names.setdefault(
                    str(rigid_id),
                    f"{side}_wrist" if hand.wrist_rigid_id is not None
                    else f"{side}_back",
                )
        return names
    # -- 生命周期 ---------------------------------------------------------

    def begin(self, take_id: int, start_wall_ns: int) -> None:
        """创建临时文件并写静态 attrs。

        临时文件放在 output_dir 下的私有 0700 目录(本进程独占),文件名随机
        (mkstemp):防符号链接/预测性临时文件攻击——共享写权限目录下攻击者
        无法预建符号链接,也无法写入 0700 私有目录。
        """
        if self._f is not None:
            raise RuntimeError("TakeWriter 已 begin")
        self._start_ns = start_wall_ns
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = self._path.parent / f".tmp_{os.getpid()}"
        self._tmp_dir.mkdir(mode=0o700, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.stem}_", suffix=".tmp.h5", dir=self._tmp_dir)
        os.close(fd)
        self._tmp_path = Path(tmp_name)
        self._f = h5py.File(self._tmp_path, "w")
        f = self._f
        f.attrs["h5_version"] = H5_VERSION
        f.attrs["schema_name"] = "mocap-acquisition"
        f.attrs["schema_layout"] = "offsets-flat-v2"
        f.attrs["take_id"] = take_id
        f.attrs["start_wall_ns"] = start_wall_ns
        # config_yaml 保持兼容名称，但语义改为“实际运行配置”。
        f.attrs["config_yaml"] = self._cfg.config_text
        f.attrs["effective_config_yaml"] = self._cfg.config_text
        f.attrs["base_config_yaml"] = self._cfg.base_config_text
        if self._cfg.calibration_text:
            f.attrs["calibration_yaml"] = self._cfg.calibration_text
        f.attrs["keymap"] = json.dumps(self._cfg.keymap, ensure_ascii=False)
        f.attrs["natnet_schema_version"] = 1
        f.attrs["axis_permutation"] = list(self._cfg.axis_permutation)
        f.attrs["axis_signs"] = list(self._cfg.axis_signs)
        f.attrs["rigid_body_names_json"] = json.dumps(
            self._configured_rigid_body_names(),
            ensure_ascii=False, sort_keys=True)

        # 采集范围:仅双手 MANO + 指定物体刚体(命名 object),不保存 mocap 全量流。
        # mocap 帧仍进入内存缓冲用于提取 objects 刚体,落盘不写 mocap/ 组。

        hands = f.create_group("hands")
        for side in ("left", "right"):
            g = hands.create_group(side)
            g.attrs["node_count"] = 21
            g.attrs["source"] = "manus"
            g.create_dataset("t_ubuntu_ns", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
            g.create_dataset("seq", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
            g.create_dataset("mano_skeleton", (0, 21, 3), maxshape=(None, 21, 3), dtype=np.float32, chunks=(1024, 21, 3))
            g.create_dataset("wrist_position", (0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(4096, 3))
            g.create_dataset("wrist_quaternion_xyzw", (0, 4), maxshape=(None, 4), dtype=np.float32, chunks=(4096, 4))

        objs = f.create_group("objects")
        for name in self._cfg.objects:
            g = objs.create_group(name)
            g.create_dataset("t_ubuntu_ns", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
            g.create_dataset("object_position", (0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(4096, 3))
            g.create_dataset("object_quaternion_xyzw", (0, 4), maxshape=(None, 4), dtype=np.float32, chunks=(4096, 4))
            g.create_dataset("tracking_valid", (0,), maxshape=(None,), dtype=np.uint8, chunks=(4096,))

        ev = f.create_group("events")
        ev.create_dataset("t_ubuntu_ns", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
        ev.create_dataset("type", (0,), maxshape=(None,), dtype=np.uint8, chunks=(4096,))
        ev.create_dataset("note", (0,), maxshape=(None,), dtype=h5py.vlen_dtype(str), chunks=(4096,))

    # -- 数据追加(轻量,只入缓冲) -------------------------------------------

    def append_mocap(self, frame: dict) -> None:
        """frame 须带 t_ubuntu_ns(StreamHub 已打点);按目标频率门控落盘。"""
        if self._rate_gate is not None and not self._rate_gate.should_write(
            int(frame["t_ubuntu_ns"]), stream="mocap"):
            return
        with self._lock:
            if self._f is not None and not self._ended:
                self._mocap.append(frame)
                self._counts["mocap"] += 1

    def append_manus(
        self,
        side: str,
        msg: dict,
        wrist_pos: np.ndarray,
        wrist_quat_xyzw: np.ndarray,
        mano_skeleton: np.ndarray,
    ) -> None:
        # 每路流独立 RateGate 窗口:三路流不共享节拍,各自达到目标频率
        if self._rate_gate is not None and not self._rate_gate.should_write(
            int(msg["t_ubuntu_ns"]), stream=side):
            return
        with self._lock:
            if self._f is not None and not self._ended:
                self._manus[side].append(
                    (msg, wrist_pos, wrist_quat_xyzw, mano_skeleton))
                self._counts[side] += 1

    def append_event(self, event_type: int, note: str = "") -> None:
        with self._lock:
            if self._f is not None:
                self._events.append((time.time_ns(), event_type, note))

    def set_edges(self, side: str, edges: list[tuple[int, int, int]]) -> None:
        """记录一次骨骼拓扑(每侧一次,供离线重建骨架);空拓扑不写。"""
        if not edges:
            return
        with self._lock:
            if self._f is not None and "edges_json" not in self._f["hands"][side].attrs:
                self._f["hands"][side].attrs["edges_json"] = json.dumps(
                    [list(e) for e in edges], ensure_ascii=False
                )

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    # -- 落盘 -------------------------------------------------------------

    def flush(self) -> None:
        """把内存缓冲批量写入 HDF5。可在主循环周期调用;finalize 前必须调用。"""
        with self._lock:
            if self._f is None:
                return
            f = self._f
            mocap_buf, self._mocap = self._mocap, []
            manus_buf = {s: b for s, b in self._manus.items()}
            self._manus = {"left": [], "right": []}
            events, self._events = self._events, []

        if mocap_buf:
            self._flush_objects(f, mocap_buf)
        for side, buf in manus_buf.items():
            if buf:
                self._flush_manus(f, side, buf)
        if events:
            self._flush_events(f, events)
        f.flush()

    def set_stream_health(self, health: dict[str, object]) -> None:
        """保存本 take 截止当前的输入健康计数，finalize 时写入 attrs。"""
        encoded = json.dumps(health, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._stream_health_json = encoded

    def set_rigid_body_names(self, names: dict[int, str]) -> None:
        """在保存前刷新运行时收到的 Motive 刚体名称表。"""
        normalized = {
            str(int(rigid_id)): str(name)
            for rigid_id, name in names.items()
        }
        with self._lock:
            if self._f is not None and not self._ended:
                self._f.attrs["rigid_body_names_json"] = json.dumps(
                    normalized, ensure_ascii=False, sort_keys=True)

    def finalize_save(self) -> None:
        """flush + 写收尾 attrs + 原子改名为正式文件名。

        顺序:先在锁内置 _ended=True 封口(新 append 不再入缓冲),再 flush
        写完全部缓冲,最后取 f 写 attrs 并置 None。若先 flush 再封口,
        flush 换空缓冲之后、置 None 之前入队的帧会静默丢失。
        """
        with self._lock:
            if self._f is None or self._ended:
                return
            self._ended = True
        self.flush()
        with self._lock:
            if self._f is None:
                return
            f, self._f = self._f, None
            f.attrs["end_wall_ns"] = time.time_ns()
            f.attrs["stream_health_json"] = self._stream_health_json
        f.close()
        # tmp 与正式文件位于同一文件系统；hard-link 创建是原子的，且目标
        # 已存在时抛 FileExistsError，绝不覆盖既有采集数据。
        os.link(self._tmp_path, self._path)
        self._tmp_path.unlink()
        self._cleanup_tmp_dir()

    def discard(self) -> None:
        """关闭并删除临时文件(不留数据)。"""
        with self._lock:
            f, self._f = self._f, None
            self._ended = True
        if f is not None:
            f.close()
        if self._tmp_path is not None:
            self._tmp_path.unlink(missing_ok=True)
        self._cleanup_tmp_dir()

    def _cleanup_tmp_dir(self) -> None:
        """删除私有临时目录;非空(不应发生)则忽略。"""
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    @property
    def tmp_path(self) -> Path:
        return self._tmp_path

    @property
    def path(self) -> Path:
        return self._path

    # -- 内部写入 ---------------------------------------------------------

    @staticmethod
    def _append_ragged(
        group: h5py.Group,
        rows: dict[str, list[np.ndarray]],
    ) -> None:
        """追加一批 offsets+flat 可变长度帧。"""
        frame_offsets = group["frame_offsets"]
        frame_count = len(next(iter(rows.values())))
        old_frame_count = frame_offsets.shape[0] - 1
        flat_start = int(frame_offsets[-1])
        lengths = np.asarray(
            [len(row) for row in next(iter(rows.values()))],
            dtype=np.int64,
        )
        flat_end = flat_start + int(lengths.sum())
        frame_offsets.resize((old_frame_count + frame_count + 1,))
        frame_offsets[old_frame_count + 1:] = (
            flat_start + np.concatenate(([0], np.cumsum(lengths)))
        )[1:]

        for name, values in rows.items():
            dataset = group[name]
            dataset.resize((flat_end, *dataset.shape[1:]))
            if flat_end > flat_start:
                dataset[flat_start:flat_end] = np.concatenate(values, axis=0)

    def _flush_objects(self, f: h5py.File, buf: list[dict]) -> None:
        """从 mocap 帧缓冲提取命名物体刚体写入 objects/(mocap 组不落盘)。"""
        objs = f["objects"]
        for name, oid in self._cfg.objects.items():
            g = objs[name]
            rows = [(i, fr) for i, fr in enumerate(buf) if any(r["id"] == oid for r in fr.get("rigid_bodies", []))]
            if not rows:
                continue
            grow = lambda ds, extra: ds.resize(ds.shape[0] + extra, axis=0)  # noqa: E731
            grow(g["t_ubuntu_ns"], len(rows))
            grow(g["object_position"], len(rows))
            grow(g["object_quaternion_xyzw"], len(rows))
            grow(g["tracking_valid"], len(rows))
            base_obj = g["t_ubuntu_ns"].shape[0] - len(rows)
            g["t_ubuntu_ns"][base_obj:] = [fr["t_ubuntu_ns"] for _, fr in rows]
            for k, (_, fr) in enumerate(rows):
                rb_ = next(r for r in fr["rigid_bodies"] if r["id"] == oid)
                g["object_position"][base_obj + k] = rb_["position"]
                g["object_quaternion_xyzw"][base_obj + k] = rb_["quaternion_xyzw"]
                g["tracking_valid"][base_obj + k] = int(rb_["tracking_valid"])

    def _flush_manus(self, f: h5py.File, side: str, buf: list) -> None:
        g = f["hands"][side]
        n = len(buf)
        for ds in (g["t_ubuntu_ns"], g["seq"], g["wrist_position"], g["wrist_quaternion_xyzw"]):
            ds.resize(ds.shape[0] + n, axis=0)
        g["mano_skeleton"].resize(g["mano_skeleton"].shape[0] + n, axis=0)
        base = g["t_ubuntu_ns"].shape[0] - n
        for k, (msg, wrist_pos, wrist_quat, mano_skeleton) in enumerate(buf):
            seq = msg.get("seq")
            if seq is None:          # 二进制模式等无 seq 来源时用 -1
                seq = -1
            g["t_ubuntu_ns"][base + k] = msg["t_ubuntu_ns"]
            g["seq"][base + k] = seq
            g["mano_skeleton"][base + k] = mano_skeleton
            g["wrist_position"][base + k] = wrist_pos
            g["wrist_quaternion_xyzw"][base + k] = wrist_quat

    def _flush_events(self, f: h5py.File, buf: list) -> None:
        ev = f["events"]
        n = len(buf)
        ev["t_ubuntu_ns"].resize(ev["t_ubuntu_ns"].shape[0] + n, axis=0)
        ev["type"].resize(ev["type"].shape[0] + n, axis=0)
        ev["note"].resize(ev["note"].shape[0] + n, axis=0)
        base = ev["t_ubuntu_ns"].shape[0] - n
        ev["t_ubuntu_ns"][base:] = [e[0] for e in buf]
        ev["type"][base:] = [e[1] for e in buf]
        ev["note"][base:] = [e[2] for e in buf]
