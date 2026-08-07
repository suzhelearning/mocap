"""HDF5 录制器(TakeWriter):一个 take 一个实例。

设计:
- append_* 只入内存缓冲(在 zenoh 回调/主线程调用,开销极小)
- flush() 把缓冲批量写入 HDF5(在录制主循环周期性调用,避开回调线程)
- 写入临时文件 .take_{id}_tmp.h5;保存 = flush + os.replace 原子改名;
  丢弃 = close + unlink(不留任何数据)

时间基准:全部用 t_ubuntu_ns(接收端墙钟)。mocap 帧自带 t_ubuntu_ns
(StreamHub 打点);manus 帧同样。

HDF5 schema 见计划文档;markers 每帧数量可变 → h5py.vlen_dtype 一维铺平。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import h5py
import numpy as np

from .config import Config

H5_VERSION = "1.0"

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

    def __init__(self, path: Path, config: Config) -> None:
        self._path = path
        self._tmp_path = path.with_name(f".{path.stem}_tmp.h5")
        self._cfg = config
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

    # -- 生命周期 ---------------------------------------------------------

    def begin(self, take_id: int, start_wall_ns: int) -> None:
        """创建临时文件并写静态 attrs。"""
        if self._f is not None:
            raise RuntimeError("TakeWriter 已 begin")
        self._start_ns = start_wall_ns
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self._tmp_path, "w")
        f = self._f
        f.attrs["h5_version"] = H5_VERSION
        f.attrs["take_id"] = take_id
        f.attrs["start_wall_ns"] = start_wall_ns
        f.attrs["config_yaml"] = self._cfg.config_text
        f.attrs["keymap"] = json.dumps(self._cfg.keymap, ensure_ascii=False)
        f.attrs["natnet_schema_version"] = 1
        f.attrs["axis_permutation"] = list(self._cfg.axis_permutation)
        f.attrs["axis_signs"] = list(self._cfg.axis_signs)

        mocap = f.create_group("mocap")
        mocap.attrs["coordinate_system"] = "motive_y_up_right_handed"
        mocap.attrs["unit"] = "meter"
        for name, dt in (
            ("frame_number", np.int64),
            ("motive_timestamp", np.float64),
            ("publisher_received_time_ns", np.int64),
            ("t_ubuntu_ns", np.int64),
            ("publisher_dropped_frames", np.int32),
        ):
            mocap.create_dataset(name, (0,), maxshape=(None,), dtype=dt, chunks=(4096,))
        rb = mocap.create_group("rigid_bodies")
        for name, dt in (
            ("ids", np.int32),
            ("positions", np.float32),
            ("quaternions_xyzw", np.float32),
            ("tracking_valid", np.uint8),
            ("mean_error", np.float32),
        ):
            rb.create_dataset(
                name, (0,), maxshape=(None,),
                dtype=h5py.vlen_dtype(dt), chunks=(4096,),
            )
        if self._cfg.store_markers:
            mk = mocap.create_group("markers")
            for name, dt in (
                ("positions", np.float32),
                ("raw_ids", np.int32),
                ("occluded", np.uint8),
                ("id_kinds", np.uint8),
            ):
                mk.create_dataset(
                    name, (0,), maxshape=(None,),
                    dtype=h5py.vlen_dtype(dt), chunks=(4096,),
                )

        hands = f.create_group("hands")
        for side in ("left", "right"):
            g = hands.create_group(side)
            g.attrs["node_count"] = 25
            g.attrs["source"] = "manus"
            g.create_dataset("t_ubuntu_ns", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
            g.create_dataset("seq", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
            g.create_dataset("nodes_raw", (0, 25, 3), maxshape=(None, 25, 3), dtype=np.float32, chunks=(1024, 25, 3))
            g.create_dataset("wrist_position", (0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(4096, 3))
            g.create_dataset("wrist_quaternion_xyzw", (0, 4), maxshape=(None, 4), dtype=np.float32, chunks=(4096, 4))
            g.create_dataset("nodes_global", (0, 25, 3), maxshape=(None, 25, 3), dtype=np.float32, chunks=(1024, 25, 3))

        objs = f.create_group("objects")
        for name in self._cfg.objects:
            g = objs.create_group(name)
            g.create_dataset("t_ubuntu_ns", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
            g.create_dataset("position", (0, 3), maxshape=(None, 3), dtype=np.float32, chunks=(4096, 3))
            g.create_dataset("quaternion_xyzw", (0, 4), maxshape=(None, 4), dtype=np.float32, chunks=(4096, 4))
            g.create_dataset("tracking_valid", (0,), maxshape=(None,), dtype=np.uint8, chunks=(4096,))

        ev = f.create_group("events")
        ev.create_dataset("t_ubuntu_ns", (0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
        ev.create_dataset("type", (0,), maxshape=(None,), dtype=np.uint8, chunks=(4096,))
        ev.create_dataset("note", (0,), maxshape=(None,), dtype=h5py.vlen_dtype(str), chunks=(4096,))

    # -- 数据追加(轻量,只入缓冲) -------------------------------------------

    def append_mocap(self, frame: dict) -> None:
        """frame 须带 t_ubuntu_ns(StreamHub 已打点)。"""
        with self._lock:
            if self._f is not None:
                self._mocap.append(frame)
                self._counts["mocap"] += 1

    def append_manus(
        self,
        side: str,
        msg: dict,
        wrist_pos: np.ndarray,
        wrist_quat_xyzw: np.ndarray,
        nodes_global: np.ndarray,
    ) -> None:
        with self._lock:
            if self._f is not None:
                self._manus[side].append((msg, wrist_pos, wrist_quat_xyzw, nodes_global))
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
            self._flush_mocap(f, mocap_buf)
        for side, buf in manus_buf.items():
            if buf:
                self._flush_manus(f, side, buf)
        if events:
            self._flush_events(f, events)
        f.flush()

    def finalize_save(self) -> None:
        """flush + 写收尾 attrs + 原子改名为正式文件名。

        注意:必须先 flush(此时 _ended 仍为 False)再标记 ended,
        否则 flush 会因 ended 跳过导致数据丢失。
        """
        with self._lock:
            if self._f is None or self._ended:
                return
        self.flush()
        with self._lock:
            if self._f is None:
                return
            f = self._f
            self._ended = True
            self._f = None
            f.attrs["end_wall_ns"] = time.time_ns()
        f.close()
        os.replace(self._tmp_path, self._path)

    def discard(self) -> None:
        """关闭并删除临时文件(不留数据)。"""
        with self._lock:
            f, self._f = self._f, None
            self._ended = True
        if f is not None:
            f.close()
        self._tmp_path.unlink(missing_ok=True)

    @property
    def tmp_path(self) -> Path:
        return self._tmp_path

    @property
    def path(self) -> Path:
        return self._path

    # -- 内部写入 ---------------------------------------------------------

    def _flush_mocap(self, f: h5py.File, buf: list[dict]) -> None:
        mocap = f["mocap"]
        n = len(buf)
        def grow(ds, extra):
            ds.resize(ds.shape[0] + extra, axis=0)
        grow(mocap["frame_number"], n)
        grow(mocap["motive_timestamp"], n)
        grow(mocap["publisher_received_time_ns"], n)
        grow(mocap["t_ubuntu_ns"], n)
        grow(mocap["publisher_dropped_frames"], n)
        base = mocap["t_ubuntu_ns"].shape[0] - n
        mocap["frame_number"][base:] = [fr["frame_number"] for fr in buf]
        mocap["motive_timestamp"][base:] = [fr["motive_timestamp"] for fr in buf]
        mocap["publisher_received_time_ns"][base:] = [fr["publisher_received_time_ns"] for fr in buf]
        mocap["t_ubuntu_ns"][base:] = [fr["t_ubuntu_ns"] for fr in buf]
        mocap["publisher_dropped_frames"][base:] = [fr["publisher_dropped_frames"] for fr in buf]

        rb = mocap["rigid_bodies"]
        grow(rb["ids"], n)
        grow(rb["positions"], n)
        grow(rb["quaternions_xyzw"], n)
        grow(rb["tracking_valid"], n)
        grow(rb["mean_error"], n)
        ids, pos, quat, valid, err = [], [], [], [], []
        for fr in buf:
            rbs = fr.get("rigid_bodies", [])
            ids.append(np.asarray([r["id"] for r in rbs], dtype=np.int32))
            pos.append(np.asarray([r["position"] for r in rbs], dtype=np.float32).ravel())
            quat.append(np.asarray([r["quaternion_xyzw"] for r in rbs], dtype=np.float32).ravel())
            valid.append(np.asarray([int(r["tracking_valid"]) for r in rbs], dtype=np.uint8))
            err.append(np.asarray([r["mean_error"] for r in rbs], dtype=np.float32))
        rb["ids"][base:] = ids
        rb["positions"][base:] = pos
        rb["quaternions_xyzw"][base:] = quat
        rb["tracking_valid"][base:] = valid
        rb["mean_error"][base:] = err

        if self._cfg.store_markers and "markers" in mocap:
            mk = mocap["markers"]
            grow(mk["positions"], n)
            grow(mk["raw_ids"], n)
            grow(mk["occluded"], n)
            grow(mk["id_kinds"], n)
            mk_pos, mk_ids, mk_occ, mk_kind = [], [], [], []
            for fr in buf:
                markers = fr.get("markers", [])
                mk_pos.append(np.asarray([m["position"] for m in markers], dtype=np.float32).ravel())
                mk_ids.append(np.asarray([m["raw_id"] for m in markers], dtype=np.int32))
                mk_occ.append(np.asarray([int(m["occluded"]) for m in markers], dtype=np.uint8))
                mk_kind.append(np.asarray([_KIND_INDEX[m["id_kind"]] for m in markers], dtype=np.uint8))
            mk["positions"][base:] = mk_pos
            mk["raw_ids"][base:] = mk_ids
            mk["occluded"][base:] = mk_occ
            mk["id_kinds"][base:] = mk_kind

        # objects 便捷子表
        objs = f["objects"]
        for name, oid in self._cfg.objects.items():
            g = objs[name]
            rows = [(i, fr) for i, fr in enumerate(buf) if any(r["id"] == oid for r in fr.get("rigid_bodies", []))]
            if not rows:
                continue
            grow(g["t_ubuntu_ns"], len(rows))
            grow(g["position"], len(rows))
            grow(g["quaternion_xyzw"], len(rows))
            grow(g["tracking_valid"], len(rows))
            base_obj = g["t_ubuntu_ns"].shape[0] - len(rows)
            g["t_ubuntu_ns"][base_obj:] = [fr["t_ubuntu_ns"] for _, fr in rows]
            for k, (_, fr) in enumerate(rows):
                rb_ = next(r for r in fr["rigid_bodies"] if r["id"] == oid)
                g["position"][base_obj + k] = rb_["position"]
                g["quaternion_xyzw"][base_obj + k] = rb_["quaternion_xyzw"]
                g["tracking_valid"][base_obj + k] = int(rb_["tracking_valid"])

    def _flush_manus(self, f: h5py.File, side: str, buf: list) -> None:
        g = f["hands"][side]
        n = len(buf)
        for ds in (g["t_ubuntu_ns"], g["seq"], g["wrist_position"], g["wrist_quaternion_xyzw"]):
            ds.resize(ds.shape[0] + n, axis=0)
        g["nodes_raw"].resize(g["nodes_raw"].shape[0] + n, axis=0)
        g["nodes_global"].resize(g["nodes_global"].shape[0] + n, axis=0)
        base = g["t_ubuntu_ns"].shape[0] - n
        for k, (msg, wrist_pos, wrist_quat, nodes_global) in enumerate(buf):
            g["t_ubuntu_ns"][base + k] = msg["t_ubuntu_ns"]
            g["seq"][base + k] = msg.get("seq", -1)
            g["nodes_raw"][base + k] = msg["nodes"]
            g["wrist_position"][base + k] = wrist_pos
            g["wrist_quaternion_xyzw"][base + k] = wrist_quat
            g["nodes_global"][base + k] = nodes_global

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
