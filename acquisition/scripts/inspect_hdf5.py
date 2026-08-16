#!/usr/bin/env python3
"""HDF5 采集质量检查；结构或质量门失败时返回非零退出码。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml
EV_START = 0
EV_SAVE = 1



@dataclass(frozen=True)
class Timing:
    count: int
    duration_s: float
    rate_hz: float
    monotonic: bool
    max_gap_ms: float


def reject_external_links(f: h5py.File, prefix: str = "") -> None:
    """拒绝外部/软链接，避免检查文件时跟随到本机其它路径。"""
    for name in f:
        link = f.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError(
                f"拒绝含 {type(link).__name__} 的 HDF5: {prefix}{name}")
        obj = f[name]
        if isinstance(obj, h5py.Group):
            reject_external_links(obj, f"{prefix}{name}/")


def _timing(dataset: h5py.Dataset) -> Timing:
    t = np.asarray(dataset[:], dtype=np.int64)
    if t.size < 2:
        return Timing(int(t.size), 0.0, 0.0, True, 0.0)
    dt = np.diff(t)
    duration = float((t[-1] - t[0]) / 1e9)
    rate = float((t.size - 1) / duration) if duration > 0 else 0.0
    return Timing(
        count=int(t.size),
        duration_s=duration,
        rate_hz=rate,
        monotonic=bool(np.all(dt > 0)),
        max_gap_ms=float(dt.max() / 1e6),
    )


def _require(container, path: str, errors: list[str]):
    try:
        return container[path]
    except KeyError:
        errors.append(f"缺少 {path}")
        return None


def _target_rate(f: h5py.File) -> float | None:
    text = f.attrs.get("effective_config_yaml", f.attrs.get("config_yaml"))
    if text is None:
        return None
    try:
        raw = yaml.safe_load(str(text))
        return float(raw["recording"]["sample_hz"])
    except (TypeError, ValueError, KeyError, yaml.YAMLError):
        return None


def _check_timing(
    label: str,
    dataset: h5py.Dataset,
    *,
    errors: list[str],
    warnings: list[str],
    max_gap_ms: float,
    min_rate_hz: float | None,
) -> Timing:
    timing = _timing(dataset)
    print(
        f"  {label}: {timing.count} 样本, {timing.duration_s:.2f}s, "
        f"{timing.rate_hz:.1f}Hz, 单调={timing.monotonic}, "
        f"最大间隙={timing.max_gap_ms:.1f}ms")
    if timing.count < 2:
        errors.append(f"{label} 样本不足 2")
    if not timing.monotonic:
        errors.append(f"{label} 时间戳非严格单调")
    if timing.max_gap_ms > max_gap_ms:
        errors.append(
            f"{label} 最大间隙 {timing.max_gap_ms:.1f}ms>{max_gap_ms:.1f}ms")
    if min_rate_hz is not None and timing.rate_hz < min_rate_hz:
        errors.append(
            f"{label} 频率 {timing.rate_hz:.1f}Hz<{min_rate_hz:.1f}Hz")
    return timing


def _flat_frame_count(group: h5py.Group) -> int:
    """兼容 v1 vlen 与 v2 offsets+flat 两种 ragged 表示。"""
    if "frame_offsets" in group:
        return max(0, len(group["frame_offsets"]) - 1)
    return len(group["ids"]) if "ids" in group else 0


def _check_ragged(
    group: h5py.Group,
    label: str,
    *,
    frame_count: int,
    fields: tuple[str, ...],
    errors: list[str],
) -> None:
    """校验 v2 offsets 与每个 flat 字段的长度、单调性一致。"""
    offsets = group.get("frame_offsets")
    if offsets is None:
        return
    values = np.asarray(offsets[:], dtype=np.int64)
    if len(values) != frame_count + 1:
        errors.append(
            f"{label}/frame_offsets 长度 {len(values)} != {frame_count + 1}")
        return
    if len(values) == 0 or values[0] != 0 or np.any(np.diff(values) < 0):
        errors.append(f"{label}/frame_offsets 非法")
        return
    flat_count = int(values[-1])
    for field in fields:
        dataset = _require(group, field, errors)
        if dataset is not None and len(dataset) != flat_count:
            errors.append(
                f"{label}/{field} flat 长度 {len(dataset)} != {flat_count}")

def _check_quaternions(
    dataset: h5py.Dataset,
    valid: np.ndarray,
    label: str,
    errors: list[str],
) -> None:
    values = np.asarray(dataset[:], dtype=np.float64)[valid].reshape(-1, 4)
    if len(values) == 0:
        return
    if not np.isfinite(values).all():
        errors.append(f"{label} 有效帧含 NaN/Inf")
        return
    norm_error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1.0)))
    if norm_error > 1e-3:
        errors.append(f"{label} 四元数范数最大误差 {norm_error:.2e}>1e-3")


def _rotation_matrices_xyzw(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ), axis=1).reshape(-1, 3, 3)


_OBSOLETE_MANO_FIELDS = (
    "mano_joints16",
    "mano_pose",
    "mano_translation",
    "mano_scale",
    "mano_fit_valid",
)


def _check_mano_beta(
    group: h5py.Group,
    label: str,
    errors: list[str],
) -> None:
    """校验最小 MANO 契约：原始关键点 + 一次性 beta。"""
    if "mano_beta" in group:
        beta = np.asarray(group["mano_beta"][:])
        if beta.shape != (10,):
            errors.append(f"{label}/mano_beta 形状 {beta.shape} != (10,)")
        elif not np.isfinite(beta).all():
            errors.append(f"{label}/mano_beta 含 NaN/Inf")
    obsolete = [name for name in _OBSOLETE_MANO_FIELDS if name in group]
    if obsolete:
        errors.append(
            f"{label} 含已废弃的逐帧 MANO 派生字段 {obsolete}；"
            "运行 mano-beta --force 清理",
        )


def _check_compact_dataset(
    dataset: h5py.Dataset,
    label: str,
    *,
    count: int,
    tail: tuple[int, ...],
    dtype: np.dtype,
    errors: list[str],
) -> None:
    expected_shape = (count, *tail)
    if dataset.shape != expected_shape:
        errors.append(f"{label} 形状 {dataset.shape} != {expected_shape}")
    if dataset.dtype != dtype:
        errors.append(f"{label} dtype {dataset.dtype} != {dtype}")


def _check_exact_children(
    group: h5py.Group | h5py.File,
    label: str,
    required: set[str],
    errors: list[str],
    *,
    optional: set[str] = frozenset(),
) -> None:
    actual = set(group.keys())
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing:
        errors.append(f"{label} 缺少字段 {missing}")
    if extra:
        errors.append(f"{label} 含契约外字段 {extra}")


def _inspect_compact_v4(
    f: h5py.File,
    *,
    strict: bool,
    min_rate_ratio: float,
    max_gap_ms: float,
    errors: list[str],
    warnings: list[str],
) -> None:
    """校验 compact-aligned-60hz-v1 的最小消费者契约。"""
    if str(f.attrs.get("schema_layout", "")) != "compact-aligned-60hz-v1":
        errors.append("v4 schema_layout 必须为 compact-aligned-60hz-v1")
    if str(f.attrs.get("time_domain", "")) != "linux-clock-monotonic":
        errors.append("v4 time_domain 必须为 linux-clock-monotonic")
    if float(f.attrs.get("output_hz", 0.0)) != 60.0:
        errors.append("v4 output_hz 必须为 60")
    _check_exact_children(
        f,
        "/",
        {"time_ns", "valid", "hands", "objects", "events"},
        errors,
    )
    time_ds = _require(f, "time_ns", errors)
    valid_ds = _require(f, "valid", errors)
    if time_ds is None or valid_ds is None:
        return
    count = len(time_ds)
    _check_compact_dataset(
        time_ds, "time_ns", count=count, tail=(), dtype=np.dtype("int64"),
        errors=errors,
    )
    _check_compact_dataset(
        valid_ds, "valid", count=count, tail=(), dtype=np.dtype("uint8"),
        errors=errors,
    )
    target_hz = 60.0
    timing = _check_timing(
        "time_ns",
        time_ds,
        errors=errors,
        warnings=warnings,
        max_gap_ms=max_gap_ms,
        min_rate_hz=target_hz * min_rate_ratio if strict else None,
    )
    if timing.count >= 2:
        dt = np.diff(np.asarray(time_ds[:], dtype=np.int64))
        ideal = 1e9 / target_hz
        if np.max(np.abs(dt - ideal)) > 1.0:
            errors.append("time_ns 不是严格固定 60 Hz 栅格")

    valid_values = np.asarray(valid_ds[:], dtype=np.uint8)
    if not np.isin(valid_values, (0, 1)).all():
        errors.append("valid 只能包含 0/1")
    frame_valid = valid_values.astype(bool)
    if strict and count and np.mean(frame_valid) < min_rate_ratio:
        errors.append(
            f"valid 有效率 {np.mean(frame_valid):.1%}<{min_rate_ratio:.0%}",
        )

    hands = _require(f, "hands", errors)
    if hands is not None:
        _check_exact_children(hands, "hands", {"left", "right"}, errors)
        for side in ("left", "right"):
            group = _require(hands, side, errors)
            if group is None:
                continue
            label = f"hands/{side}"
            required = {
                "keypoints_world",
                "wrist_position",
                "wrist_quaternion_xyzw",
                "valid",
            }
            _check_exact_children(
                group, label, required, errors, optional={"mano_beta"},
            )
            fields = (
                ("keypoints_world", (21, 3), np.dtype("float32")),
                ("wrist_position", (3,), np.dtype("float32")),
                ("wrist_quaternion_xyzw", (4,), np.dtype("float32")),
                ("valid", (), np.dtype("uint8")),
            )
            for name, tail, dtype in fields:
                dataset = _require(group, name, errors)
                if dataset is not None:
                    _check_compact_dataset(
                        dataset,
                        f"{label}/{name}",
                        count=count,
                        tail=tail,
                        dtype=dtype,
                        errors=errors,
                    )
            if (
                not required.issubset(group.keys())
                or any(
                    group[name].shape != (count, *tail)
                    for name, tail, _ in fields
                )
            ):
                continue
            hand_valid_values = np.asarray(group["valid"][:], dtype=np.uint8)
            if not np.isin(hand_valid_values, (0, 1)).all():
                errors.append(f"{label}/valid 只能包含 0/1")
            hand_valid = hand_valid_values.astype(bool)
            keypoints = np.asarray(group["keypoints_world"][:])
            wrist = np.asarray(group["wrist_position"][:])
            if np.any(hand_valid):
                if not np.isfinite(keypoints[hand_valid]).all():
                    errors.append(f"{label}/keypoints_world 有效帧含 NaN/Inf")
                root_error = float(np.max(
                    np.abs(keypoints[hand_valid, 0] - wrist[hand_valid]),
                ))
                if root_error > 1e-5:
                    errors.append(
                        f"{label} root/wrist 误差 {root_error:.2e}m",
                    )
            _check_quaternions(
                group["wrist_quaternion_xyzw"],
                hand_valid,
                f"{label}/wrist_quaternion_xyzw",
                errors,
            )
            _check_mano_beta(group, label, errors)
            if strict and count and np.mean(hand_valid) < min_rate_ratio:
                errors.append(
                    f"{label} 有效率 "
                    f"{np.mean(hand_valid):.1%}<{min_rate_ratio:.0%}",
                )

    objects = _require(f, "objects", errors)
    if objects is not None:
        for name, group in objects.items():
            label = f"objects/{name}"
            required = {
                "object_position", "object_quaternion_xyzw", "valid",
            }
            _check_exact_children(group, label, required, errors)
            fields = (
                ("object_position", (3,), np.dtype("float32")),
                ("object_quaternion_xyzw", (4,), np.dtype("float32")),
                ("valid", (), np.dtype("uint8")),
            )
            for field, tail, dtype in fields:
                dataset = _require(group, field, errors)
                if dataset is not None:
                    _check_compact_dataset(
                        dataset,
                        f"{label}/{field}",
                        count=count,
                        tail=tail,
                        dtype=dtype,
                        errors=errors,
                    )
            if (
                not required.issubset(group.keys())
                or any(
                    group[field].shape != (count, *tail)
                    for field, tail, _ in fields
                )
            ):
                continue
            object_valid_values = np.asarray(group["valid"][:], dtype=np.uint8)
            if not np.isin(object_valid_values, (0, 1)).all():
                errors.append(f"{label}/valid 只能包含 0/1")
            object_valid = object_valid_values.astype(bool)
            if np.any(object_valid) and not np.isfinite(
                np.asarray(group["object_position"][:])[object_valid],
            ).all():
                errors.append(f"{label}/object_position 有效帧含 NaN/Inf")
            _check_quaternions(
                group["object_quaternion_xyzw"],
                object_valid,
                f"{label}/object_quaternion_xyzw",
                errors,
            )
            if strict and count and np.mean(object_valid) < min_rate_ratio:
                errors.append(
                    f"{label} 有效率 "
                    f"{np.mean(object_valid):.1%}<{min_rate_ratio:.0%}",
                )

    events = _require(f, "events", errors)
    if events is not None:
        _check_exact_children(
            events, "events", {"frame_index", "type"}, errors,
        )
        frame_index = _require(events, "frame_index", errors)
        event_type = _require(events, "type", errors)
        if frame_index is not None and event_type is not None:
            event_count = len(frame_index)
            _check_compact_dataset(
                frame_index,
                "events/frame_index",
                count=event_count,
                tail=(),
                dtype=np.dtype("int64"),
                errors=errors,
            )
            _check_compact_dataset(
                event_type,
                "events/type",
                count=event_count,
                tail=(),
                dtype=np.dtype("uint8"),
                errors=errors,
            )
            indices = np.asarray(frame_index[:], dtype=np.int64)
            types = np.asarray(event_type[:], dtype=np.uint8)
            if np.any(np.diff(indices) < 0):
                errors.append("events/frame_index 非单调")
            if np.any(indices < 0) or np.any(indices > count):
                errors.append(f"events/frame_index 必须在 [0,{count}]")
            if not np.isin(types, (EV_START, EV_SAVE)).all():
                errors.append("events/type 只能包含 start(0)/save(1)")
            if EV_START not in types:
                errors.append("events 缺少 start")
            if EV_SAVE not in types:
                errors.append("events 缺少 save")


def _inspect_aligned_v3(
    f: h5py.File,
    *,
    strict: bool,
    min_rate_ratio: float,
    max_gap_ms: float,
    errors: list[str],
    warnings: list[str],
) -> None:
    """校验 aligned-60hz-v1：所有派生数组必须与唯一 timeline 等长。"""
    for attr in (
        "schema_layout", "time_domain", "output_hz", "effective_config_yaml",
        "base_config_yaml", "rigid_body_names_json",
    ):
        if attr not in f.attrs:
            errors.append(f"v3 缺少根属性 {attr}")
    if str(f.attrs.get("schema_layout", "")) != "aligned-60hz-v1":
        errors.append("v3 schema_layout 必须为 aligned-60hz-v1")
    if str(f.attrs.get("time_domain", "")) != "linux-clock-monotonic":
        errors.append("v3 time_domain 必须为 linux-clock-monotonic")
    if float(f.attrs.get("output_hz", 0.0)) != 60.0:
        errors.append("v3 output_hz 必须固定为 60")
    timeline = _require(f, "timeline", errors)
    if timeline is None:
        return
    t_ds = _require(timeline, "t_phys_ns", errors)
    if t_ds is None:
        return
    target_hz = float(f.attrs.get("output_hz", 60.0))
    timing = _check_timing(
        "timeline", t_ds, errors=errors, warnings=warnings,
        max_gap_ms=max_gap_ms,
        min_rate_hz=target_hz * min_rate_ratio if strict else None,
    )
    n = timing.count
    required_timeline = (
        "frame_index", "t_emit_ns", "emission_latency_ns",
        "frame_valid", "reason_flags",
    )
    for name in required_timeline:
        dataset = _require(timeline, name, errors)
        if dataset is not None and len(dataset) != n:
            errors.append(f"timeline/{name} 长度 {len(dataset)} != {n}")
    if "frame_valid" in timeline and n:
        frame_valid_ratio = float(np.mean(timeline["frame_valid"][:]))
        print(f"  frame valid={frame_valid_ratio:.1%}")
        if strict and frame_valid_ratio < min_rate_ratio:
            errors.append(
                f"timeline/frame_valid 有效率 "
                f"{frame_valid_ratio:.1%}<{min_rate_ratio:.0%}"
            )
    if "frame_index" in timeline:
        indices = np.asarray(timeline["frame_index"][:], dtype=np.int64)
        if n and not np.array_equal(indices, np.arange(n)):
            errors.append("timeline/frame_index 必须从 0 连续递增")
    if n >= 2:
        dt = np.diff(np.asarray(t_ds[:], dtype=np.int64))
        ideal = 1e9 / target_hz
        if np.max(np.abs(dt - ideal)) > 1.0:
            errors.append("timeline/t_phys_ns 不是严格固定 60 Hz 栅格")
    if "t_emit_ns" in timeline:
        latency = (
            np.asarray(timeline["t_emit_ns"][:], dtype=np.int64)
            - np.asarray(t_ds[:], dtype=np.int64)
        )
        if np.any(latency < 0):
            errors.append("timeline 存在早于目标物理时刻的 emission")
        if len(latency):
            print(
                "  emission latency: "
                f"p50={np.percentile(latency, 50) / 1e6:.1f}ms, "
                f"p95={np.percentile(latency, 95) / 1e6:.1f}ms"
            )

    def check_group_lengths(
        group: h5py.Group,
        label: str,
        static_fields: frozenset[str] = frozenset(),
    ) -> None:
        for name, value in group.items():
            if (
                isinstance(value, h5py.Dataset)
                and name not in static_fields
                and len(value) != n
            ):
                errors.append(f"{label}/{name} 长度 {len(value)} != {n}")

    quality = _require(f, "quality/mocap", errors)
    if quality is not None:
        check_group_lengths(quality, "quality/mocap")
    for side in ("left", "right"):
        group = _require(f, f"hands/{side}", errors)
        if group is None:
            continue
        check_group_lengths(
            group, f"hands/{side}", frozenset({"mano_beta"}),
        )
        for field, shape in (
            ("nodes_local", (25, 3)),
            ("node_quaternions_wxyz", (25, 4)),
            ("nodes_world", (25, 3)),
            ("mano_skeleton", (21, 3)),
            ("wrist_position", (3,)),
            ("wrist_quaternion_xyzw", (4,)),
        ):
            dataset = _require(group, field, errors)
            if dataset is not None and dataset.shape[1:] != shape:
                errors.append(
                    f"hands/{side}/{field} 尾形状 {dataset.shape[1:]} != {shape}"
                )
        if "valid" in group and "mano_skeleton" in group:
            valid = np.asarray(group["valid"][:], dtype=bool)
            nodes = np.asarray(group["mano_skeleton"][:])
            if "wrist_quaternion_xyzw" in group:
                _check_quaternions(
                    group["wrist_quaternion_xyzw"], valid,
                    f"hands/{side}/wrist_quaternion_xyzw", errors,
                )
            if "node_quaternions_wxyz" in group:
                _check_quaternions(
                    group["node_quaternions_wxyz"], valid,
                    f"hands/{side}/node_quaternions_wxyz", errors,
                )
            if np.any(valid) and not np.isfinite(nodes[valid]).all():
                errors.append(f"hands/{side} 有效帧含 NaN/Inf")
            if np.any(valid) and "wrist_position" in group:
                wrist = np.asarray(group["wrist_position"][:])
                root_error = float(np.max(np.abs(nodes[valid, 0] - wrist[valid])))
                print(f"  hands/{side} root/wrist 最大误差: {root_error:.2e}m")
                if root_error > 1e-5:
                    errors.append(
                        f"hands/{side} root/wrist 误差 {root_error:.2e}m"
                    )
            if n:
                valid_ratio = float(np.mean(valid))
                print(f"  hands/{side} valid={valid_ratio:.1%}")
                if strict and valid_ratio < min_rate_ratio:
                    errors.append(
                        f"hands/{side} 有效率 "
                        f"{valid_ratio:.1%}<{min_rate_ratio:.0%}"
                    )
        if strict and not group.attrs.get("edges_json"):
            errors.append(f"hands/{side} 缺少 edges_json 拓扑")
        _check_mano_beta(group, f"hands/{side}", errors)

    objects = _require(f, "objects", errors)
    interaction = _require(f, "interaction", errors)
    if objects is not None:
        for name, group in objects.items():
            check_group_lengths(group, f"objects/{name}")
            for field, shape in (
                ("rigid_position", (3,)),
                ("rigid_quaternion_xyzw", (4,)),
                ("object_position", (3,)),
                ("object_quaternion_xyzw", (4,)),
            ):
                dataset = _require(group, field, errors)
                if dataset is not None and dataset.shape[1:] != shape:
                    errors.append(
                        f"objects/{name}/{field} 尾形状 "
                        f"{dataset.shape[1:]} != {shape}"
                    )
            object_valid = (
                np.asarray(group["valid"][:], dtype=bool)
                if "valid" in group else np.zeros(n, dtype=bool)
            )
            if len(object_valid):
                valid_ratio = float(np.mean(object_valid))
                print(f"  object/{name} valid={valid_ratio:.1%}")
                if strict and valid_ratio < min_rate_ratio:
                    errors.append(
                        f"object/{name} 有效率 "
                        f"{valid_ratio:.1%}<{min_rate_ratio:.0%}"
                    )
            for field in (
                "rigid_quaternion_xyzw", "object_quaternion_xyzw",
            ):
                if field in group:
                    _check_quaternions(
                        group[field], object_valid,
                        f"objects/{name}/{field}", errors,
                    )
            if np.any(object_valid):
                for field in ("rigid_position", "object_position"):
                    if field in group and not np.isfinite(
                        np.asarray(group[field][:])[object_valid]
                    ).all():
                        errors.append(
                            f"objects/{name}/{field} 有效帧含 NaN/Inf"
                        )
            if interaction is None or name not in interaction:
                errors.append(f"缺少 interaction/{name}")
                continue
            interaction_group = interaction[name]
            check_group_lengths(interaction_group, f"interaction/{name}")
            for side in ("left", "right"):
                hand_group = f[f"hands/{side}"]
                hand_valid = np.asarray(
                    hand_group["valid"][:], dtype=bool,
                )
                expected_valid = object_valid & hand_valid
                valid_dataset = _require(
                    interaction_group, f"{side}_valid", errors,
                )
                points_dataset = _require(
                    interaction_group, f"{side}_nodes_object", errors,
                )
                if points_dataset is not None and points_dataset.shape[1:] != (21, 3):
                    errors.append(
                        f"interaction/{name}/{side}_nodes_object 尾形状 "
                        f"{points_dataset.shape[1:]} != (21, 3)"
                    )
                if valid_dataset is not None and not np.array_equal(
                    np.asarray(valid_dataset[:], dtype=bool), expected_valid,
                ):
                    errors.append(
                        f"interaction/{name}/{side}_valid "
                        "不等于 hand_valid & object_valid"
                    )
                if points_dataset is None or not np.any(expected_valid):
                    continue
                rotations = _rotation_matrices_xyzw(
                    np.asarray(
                        group["object_quaternion_xyzw"][:],
                        dtype=np.float64,
                    )[expected_valid],
                )
                delta = (
                    np.asarray(
                        hand_group["mano_skeleton"][:],
                        dtype=np.float64,
                    )[expected_valid]
                    - np.asarray(
                        group["object_position"][:],
                        dtype=np.float64,
                    )[expected_valid, None, :]
                )
                expected = np.einsum(
                    "nji,nkj->nki", rotations, delta,
                )
                actual = np.asarray(
                    points_dataset[:], dtype=np.float64,
                )[expected_valid]
                error = float(np.max(np.abs(actual - expected)))
                if error > 1e-5:
                    errors.append(
                        f"interaction/{name}/{side}_nodes_object "
                        f"坐标误差 {error:.2e}m"
                    )

    raw = _require(f, "raw", errors)
    quality_root = _require(f, "quality", errors)
    if quality_root is not None:
        if "stream_gap_counts_json" not in quality_root.attrs:
            errors.append("quality 缺少 stream_gap_counts_json")
        clock_root = quality_root.get("clock_alignment")
        if strict and not isinstance(clock_root, h5py.Group):
            errors.append("quality 缺少 clock_alignment")
        if isinstance(clock_root, h5py.Group):
            for stream in ("mocap", "left", "right"):
                clock = clock_root.get(stream)
                if strict and not isinstance(clock, h5py.Group):
                    errors.append(f"clock_alignment 缺少 {stream}")
                    continue
                if not isinstance(clock, h5py.Group):
                    continue
                for field in (
                    "valid", "sample_count", "offset_ms", "drift_ppm",
                    "jitter_p95_ms", "resets",
                ):
                    if field not in clock:
                        errors.append(
                            f"clock_alignment/{stream} 缺少 {field}"
                        )
                if strict and "valid" in clock and not bool(clock["valid"][()]):
                    errors.append(f"clock_alignment/{stream} 尚未收敛")

    if raw is not None:
        for label in ("mocap", "hands/left", "hands/right"):
            group = _require(raw, label, errors)
            if group is not None and "t_phys_ns" in group:
                raw_timing = _timing(group["t_phys_ns"])
                print(
                    f"  raw/{label}: {raw_timing.count} 样本, "
                    f"{raw_timing.rate_hz:.1f}Hz"
                )
                if not raw_timing.monotonic:
                    errors.append(f"raw/{label} 时间戳非严格单调")

    events = _require(f, "events", errors)
    if events is not None and "type" in events:
        event_types = set(int(value) for value in events["type"][:])
        if 0 not in event_types:
            errors.append("events 缺少 start")
        if strict and 3 not in event_types:
            errors.append("events 缺少 save")


def inspect_file(
    path: str | Path,
    *,
    strict: bool = False,
    min_rate_ratio: float = 0.7,
    max_gap_ms: float = 100.0,
) -> list[str]:
    """检查单个文件并返回错误列表；warnings 只展示、不阻止普通 inspect。"""
    errors: list[str] = []
    warnings: list[str] = []
    path = Path(path)
    print(f"== {path} ==")
    try:
        f = h5py.File(path, "r")
    except (OSError, ValueError) as exc:
        return [f"无法打开 HDF5: {exc}"]
    with f:
        try:
            reject_external_links(f)
        except ValueError as exc:
            return [str(exc)]

        version = str(f.attrs.get("h5_version", "?"))
        required_attrs = (
            "h5_version",
            "take_id",
            "start_wall_ns",
            "end_wall_ns",
            "effective_config_yaml",
        ) if version == "4.0" else (
            "h5_version",
            "take_id",
            "start_wall_ns",
            "end_wall_ns",
            "config_yaml",
        )
        for attr in required_attrs:
            if attr not in f.attrs:
                errors.append(f"缺少根属性 {attr}")
        print(f"  schema: {version}, take_id={f.attrs.get('take_id')}")
        if version not in {"1.0", "2.0", "3.0", "4.0"}:
            errors.append(f"不支持的 h5_version={version}")
        if version == "4.0":
            _inspect_compact_v4(
                f,
                strict=strict,
                min_rate_ratio=min_rate_ratio,
                max_gap_ms=max_gap_ms,
                errors=errors,
                warnings=warnings,
            )
            for warning in warnings:
                print(f"[WARN] {warning}")
            for error in errors:
                print(f"[FAIL] {error}", file=sys.stderr)
            return errors
        if version == "3.0":
            _inspect_aligned_v3(
                f,
                strict=strict,
                min_rate_ratio=min_rate_ratio,
                max_gap_ms=max_gap_ms,
                errors=errors,
                warnings=warnings,
            )
            if "stream_health_json" in f.attrs:
                try:
                    json.loads(str(f.attrs["stream_health_json"]))
                except (TypeError, json.JSONDecodeError):
                    errors.append("stream_health_json 不是合法 JSON")
            for warning in warnings:
                print(f"[WARN] {warning}")
            for error in errors:
                print(f"[FAIL] {error}", file=sys.stderr)
            return errors
        if version == "2.0":
            for attr in (
                "effective_config_yaml", "base_config_yaml",
                "rigid_body_names_json",
            ):
                if attr not in f.attrs:
                    errors.append(f"v2 缺少根属性 {attr}")

        target = _target_rate(f)
        min_rate = target * min_rate_ratio if strict and target else None
        if target is None:
            warnings.append("无法从有效配置解析 recording.sample_hz")

        # 新 schema(双手 MANO + 物体)无 mocap/ 组;旧文件保留完整检查
        mocap = f.get("mocap")
        if mocap is not None:
            t_ds = _require(mocap, "t_ubuntu_ns", errors)
            if t_ds is not None:
                _check_timing("mocap", t_ds, errors=errors, warnings=warnings,
                              max_gap_ms=max_gap_ms, min_rate_hz=min_rate)
            aligned_ds = mocap.get("t_aligned_ubuntu_ns")
            if aligned_ds is not None:
                _check_timing(
                    "mocap/aligned", aligned_ds,
                    errors=errors, warnings=warnings,
                    max_gap_ms=max_gap_ms, min_rate_hz=min_rate)
                if t_ds is not None and len(aligned_ds) != len(t_ds):
                    errors.append(
                        "mocap/t_aligned_ubuntu_ns 长度与 t_ubuntu_ns 不一致")
            rb = _require(mocap, "rigid_bodies", errors)
            if rb is not None:
                count = _flat_frame_count(rb)
                print(f"  rigid_bodies 帧: {count}")
                if t_ds is not None and count != len(t_ds):
                    errors.append(
                        f"rigid_bodies 帧数 {count} != mocap 时间戳 {len(t_ds)}")
                if version == "2.0":
                    if "frame_offsets" not in rb:
                        errors.append("v2 缺少 mocap/rigid_bodies/frame_offsets")
                    _check_ragged(
                        rb,
                        "mocap/rigid_bodies",
                        frame_count=count,
                        fields=("ids", "positions", "quaternions_xyzw",
                                "tracking_valid", "mean_error"),
                        errors=errors,
                    )
            if "markers" in mocap:
                markers = mocap["markers"]
                if "frame_offsets" in markers:
                    marker_count = max(0, len(markers["frame_offsets"]) - 1)
                elif "positions" in markers:
                    marker_count = len(markers["positions"])
                else:
                    marker_count = 0
                    errors.append("markers 缺少 positions/frame_offsets")
                if t_ds is not None and marker_count != len(t_ds):
                    errors.append(
                        f"markers 帧数 {marker_count} != mocap 时间戳 {len(t_ds)}")
                if version == "2.0":
                    if "frame_offsets" not in markers:
                        errors.append("v2 缺少 mocap/markers/frame_offsets")
                    _check_ragged(
                        markers,
                        "mocap/markers",
                        frame_count=marker_count,
                        fields=("positions", "raw_ids", "occluded", "id_kinds"),
                        errors=errors,
                    )

        for side in ("left", "right"):
            group = _require(f, f"hands/{side}", errors)
            if group is None:
                continue
            t_ds = _require(group, "t_ubuntu_ns", errors)
            if t_ds is not None:
                _check_timing(
                    f"hands/{side}", t_ds, errors=errors, warnings=warnings,
                    max_gap_ms=max_gap_ms, min_rate_hz=min_rate)
            for name in ("seq", "wrist_position", "wrist_quaternion_xyzw"):
                ds = _require(group, name, errors)
                if ds is not None and t_ds is not None and len(ds) != len(t_ds):
                    errors.append(
                        f"hands/{side}/{name} 长度 {len(ds)} != {len(t_ds)}")
            # 节点数据:新 schema 必存 mano_skeleton(21 点),旧文件为 nodes_global(25 点)
            node_ds = group.get("mano_skeleton", group.get("nodes_global"))
            if node_ds is None:
                errors.append(f"hands/{side} 缺少 mano_skeleton/nodes_global")
            elif t_ds is not None and len(node_ds) != len(t_ds):
                errors.append(
                    f"hands/{side}/{node_ds.name.rsplit('/', 1)[-1]} "
                    f"长度 {len(node_ds)} != {len(t_ds)}")
            if node_ds is not None and not np.isfinite(
                    np.asarray(node_ds[:])).all():
                errors.append(f"hands/{side} 节点含 NaN/Inf")
            # 手腕节点(21 点布局的 mp0 / 25 点布局的节点 0)恒等于 wrist_position
            if node_ds is not None and "wrist_position" in group:
                nodes = np.asarray(node_ds[:], dtype=float)
                wrist = np.asarray(group["wrist_position"][:], dtype=float)
                if len(nodes):
                    root_error = float(np.max(np.abs(nodes[:, 0, :] - wrist)))
                    print(f"  hands/{side} root/wrist 最大误差: {root_error:.2e}m")
                    if root_error > 1e-5:
                        errors.append(
                            f"hands/{side} root/wrist 误差 {root_error:.2e}m")
            _check_mano_beta(
                group,
                f"hands/{side}",
                errors,
            )
            edges = group.attrs.get("edges_json")
            if strict and not edges:
                errors.append(f"hands/{side} 缺少 edges_json 拓扑")

        events = _require(f, "events", errors)
        if events is not None and "type" in events:
            event_types = set(int(v) for v in events["type"][:])
            if 0 not in event_types:
                errors.append("events 缺少 start")
            if strict and 3 not in event_types:
                errors.append("events 缺少 save")

        if "objects" in f:
            for name, group in f["objects"].items():
                t_ds = group.get("t_ubuntu_ns")
                if t_ds is not None:
                    _check_timing(
                        f"objects/{name}", t_ds,
                        errors=errors, warnings=warnings,
                        max_gap_ms=max_gap_ms, min_rate_hz=None)
                if len(group.get("tracking_valid", [])):
                    ratio = float(np.mean(group["tracking_valid"][:]))
                    print(f"  object/{name} tracking_valid={ratio:.1%}")
                    if strict and ratio < 0.7:
                        errors.append(
                            f"object/{name} 有效跟踪率 {ratio:.1%}<70%")

        if "stream_health_json" in f.attrs:
            try:
                health = json.loads(str(f.attrs["stream_health_json"]))
                print(f"  stream health: {json.dumps(health, ensure_ascii=False)}")
            except (TypeError, json.JSONDecodeError):
                errors.append("stream_health_json 不是合法 JSON")

    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[FAIL] {error}", file=sys.stderr)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 HDF5 结构与采集质量")
    parser.add_argument("files", nargs="+", help="一个或多个 .h5 文件")
    parser.add_argument("--strict", action="store_true",
                        help="启用频率、拓扑、物体跟踪率等 E2E 质量门")
    parser.add_argument("--min-rate-ratio", type=float, default=0.7,
                        help="严格模式最低频率/目标频率比例(默认 0.7)")
    parser.add_argument("--max-gap-ms", type=float, default=100.0,
                        help="允许的最大帧间隙(默认 100ms)")
    args = parser.parse_args(argv)
    all_errors = []
    for path in args.files:
        all_errors.extend(inspect_file(
            path,
            strict=args.strict,
            min_rate_ratio=args.min_rate_ratio,
            max_gap_ms=args.max_gap_ms,
        ))
    if all_errors:
        print(f"[FAIL] 共 {len(all_errors)} 项质量错误", file=sys.stderr)
        return 1
    print("[PASS] HDF5 质量检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
