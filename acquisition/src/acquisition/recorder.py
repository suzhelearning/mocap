"""固定 60 Hz 紧凑 HDF5 录制器。"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import h5py
import numpy as np

from .alignment import AlignedFrame
from .config import Config

H5_VERSION = "4.0"
SCHEMA_LAYOUT = "compact-aligned-60hz-v1"

EV_START = 0
EV_SAVE = 1


class TakeWriter:
    """``begin → append_* → finalize_save|discard`` 的线程安全 writer。"""

    def __init__(
        self,
        path: Path,
        config: Config,
        *,
        object_offset_sha256: str = "",
        object_pose_frames: dict[str, str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._tmp_path: Path | None = None
        self._tmp_dir: Path | None = None
        self._cfg = config
        self._object_offset_sha256 = object_offset_sha256
        self._f: h5py.File | None = None
        self._object_pose_frames = {
            name: (object_pose_frames or {}).get(name, "motive_rigid")
            for name in config.objects
        }
        self._lock = threading.Lock()
        self._aligned: list[AlignedFrame] = []
        self._events: list[tuple[int, int]] = []
        self._counts = {"aligned": 0}
        self._ended = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def tmp_path(self) -> Path:
        if self._tmp_path is None:
            raise RuntimeError("TakeWriter 尚未 begin")
        return self._tmp_path


    @staticmethod
    def _create_resizable(
        group: h5py.Group,
        name: str,
        tail: tuple[int, ...],
        dtype: object,
        chunk_frames: int,
    ) -> h5py.Dataset:
        chunk = min(max(1, chunk_frames), 4096)
        return group.create_dataset(
            name,
            shape=(0, *tail),
            maxshape=(None, *tail),
            dtype=dtype,
            chunks=(chunk, *tail),
        )

    def begin(self, take_id: int, start_wall_ns: int) -> None:
        if self._f is not None:
            raise RuntimeError("TakeWriter 已 begin")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = self._path.parent / f".tmp_{os.getpid()}"
        self._tmp_dir.mkdir(mode=0o700, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.stem}_",
            suffix=".tmp.h5",
            dir=self._tmp_dir,
        )
        os.close(fd)
        self._tmp_path = Path(tmp_name)
        self._f = h5py.File(self._tmp_path, "w")
        f = self._f
        f.attrs["h5_version"] = H5_VERSION
        f.attrs["schema_name"] = "mocap-acquisition"
        f.attrs["schema_layout"] = SCHEMA_LAYOUT
        f.attrs["time_domain"] = "linux-clock-monotonic"
        f.attrs["output_hz"] = self._cfg.alignment_hz
        f.attrs["take_id"] = take_id
        f.attrs["start_wall_ns"] = start_wall_ns
        f.attrs["effective_config_yaml"] = self._cfg.config_text
        pose_frames = set(self._object_pose_frames.values())
        f.attrs["object_pose_frame"] = (
            "none" if not pose_frames
            else next(iter(pose_frames)) if len(pose_frames) == 1
            else "mixed"
        )
        if self._object_offset_sha256:
            f.attrs["object_offset_config_sha256"] = self._object_offset_sha256

        chunk = self._cfg.chunk_frames
        self._create_resizable(f, "time_ns", (), np.int64, chunk)
        self._create_resizable(f, "valid", (), np.uint8, chunk)

        objects = f.create_group("objects")
        for name in self._cfg.objects:
            group = objects.create_group(name)
            self._create_resizable(
                group, "object_position", (3,), np.float32, chunk,
            )
            self._create_resizable(
                group, "object_quaternion_xyzw", (4,), np.float32, chunk,
            )
            self._create_resizable(group, "valid", (), np.uint8, chunk)
            group.attrs["object_pose_frame"] = self._object_pose_frames[name]

        hands = f.create_group("hands")
        for side in ("left", "right"):
            group = hands.create_group(side)
            group.attrs["source"] = "manus"
            group.attrs["keypoint_count"] = 21
            self._create_resizable(
                group, "keypoints_world", (21, 3), np.float32, chunk,
            )
            self._create_resizable(
                group, "wrist_position", (3,), np.float32, chunk,
            )
            self._create_resizable(
                group, "wrist_quaternion_xyzw", (4,), np.float32, chunk,
            )
            self._create_resizable(group, "valid", (), np.uint8, chunk)

        events = f.create_group("events")
        self._create_resizable(events, "frame_index", (), np.int64, chunk)
        self._create_resizable(events, "type", (), np.uint8, chunk)

    def append_aligned_frame(self, frame: AlignedFrame) -> None:
        with self._lock:
            if self._f is not None and not self._ended:
                self._aligned.append(frame)
                self._counts["aligned"] += 1

    def append_event(self, event_type: int) -> None:
        """记录帧边界事件；索引表示事件之后首个 aligned frame。"""
        with self._lock:
            if self._f is not None and not self._ended:
                self._events.append((self._counts["aligned"], event_type))

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    @staticmethod
    def _append(dataset: h5py.Dataset, values: object) -> None:
        array = np.asarray(values, dtype=dataset.dtype)
        count = len(array)
        if count == 0:
            return
        base = len(dataset)
        dataset.resize(base + count, axis=0)
        dataset[base:] = array

    def flush(self) -> None:
        with self._lock:
            if self._f is None:
                return
            f = self._f
            aligned, self._aligned = self._aligned, []
            events, self._events = self._events, []
        if aligned:
            self._flush_aligned(f, aligned)
        if events:
            self._flush_events(f, events)
        f.flush()

    def _flush_aligned(self, f: h5py.File, frames: list[AlignedFrame]) -> None:
        self._append(f["time_ns"], [frame.t_phys_ns for frame in frames])
        self._append(f["valid"], [frame.frame_valid for frame in frames])

        for name in self._cfg.objects:
            group = f["objects"][name]
            values = [frame.objects[name] for frame in frames]
            for field in (
                "object_position", "object_quaternion_xyzw", "valid",
            ):
                self._append(
                    group[field],
                    [getattr(value, field) for value in values],
                )

        for side in ("left", "right"):
            group = f["hands"][side]
            values = [frame.hands[side] for frame in frames]
            self._append(
                group["keypoints_world"],
                [value.mano_skeleton for value in values],
            )
            for field in (
                "wrist_position", "wrist_quaternion_xyzw", "valid",
            ):
                self._append(
                    group[field],
                    [getattr(value, field) for value in values],
                )

    def _flush_events(
        self, f: h5py.File, events: list[tuple[int, int]],
    ) -> None:
        group = f["events"]
        self._append(group["frame_index"], [event[0] for event in events])
        self._append(group["type"], [event[1] for event in events])



    def finalize_save(self) -> None:
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
        f.close()
        assert self._tmp_path is not None
        os.link(self._tmp_path, self._path)
        self._tmp_path.unlink()
        self._cleanup_tmp_dir()

    def discard(self) -> None:
        with self._lock:
            f, self._f = self._f, None
            self._ended = True
        if f is not None:
            f.close()
        if self._tmp_path is not None:
            self._tmp_path.unlink(missing_ok=True)
        self._cleanup_tmp_dir()

    def _cleanup_tmp_dir(self) -> None:
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
