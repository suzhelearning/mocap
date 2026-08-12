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

        for attr in ("h5_version", "take_id", "start_wall_ns", "end_wall_ns",
                     "config_yaml"):
            if attr not in f.attrs:
                errors.append(f"缺少根属性 {attr}")
        version = str(f.attrs.get("h5_version", "?"))
        print(f"  schema: {version}, take_id={f.attrs.get('take_id')}")
        if version not in {"1.0", "2.0"}:
            errors.append(f"不支持的 h5_version={version}")
        if version == "2.0":
            for attr in ("effective_config_yaml", "base_config_yaml",
                         "rigid_body_names_json"):
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
