#!/usr/bin/env python3
"""把 Motive Visuals 的 GL XYZ / GO Pitch-Yaw-Roll 写入物体外参配置。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from acquisition.object_offset import (
    CONFIG_VERSION,
    DEFAULT_OBJECT_OFFSETS_PATH,
    ObjectOffset,
    ObjectOffsetError,
    load_object_offsets,
    offset_from_motive_visuals,
)


def _flow(values: Sequence[float]) -> CommentedSeq:
    sequence = CommentedSeq(float(value) for value in values)
    sequence.fa.set_flow_style()
    return sequence


def _entry(offset: ObjectOffset) -> CommentedMap:
    matrix = CommentedSeq(_flow(row) for row in offset.rotation_matrix)
    transform = CommentedMap()
    transform["translation_m"] = _flow(offset.translation_m)
    transform["rotation_matrix"] = matrix
    return CommentedMap({"motive_rigid_from_obj": transform})


def write_offset(
    path: str | Path,
    object_name: str,
    gl_xyz_mm: Sequence[float],
    go_pyr_deg: Sequence[float],
    *,
    force: bool = False,
) -> ObjectOffset:
    """原子写入一个物体外参；已存在时须显式 ``force``。"""
    if not object_name or "/" in object_name or object_name in {".", ".."}:
        raise ObjectOffsetError(f"非法物体名: {object_name!r}")
    offset = offset_from_motive_visuals(gl_xyz_mm, go_pyr_deg)
    config_path = Path(path).expanduser().resolve(strict=False)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                root = yaml.load(stream)
        except YAMLError as exc:
            raise ObjectOffsetError(
                f"物体外参 YAML 解析失败: {exc}"
            ) from exc
        except OSError as exc:
            raise ObjectOffsetError(
                f"无法读取物体外参配置 {config_path}: {exc}"
            ) from exc
    else:
        root = CommentedMap({"version": CONFIG_VERSION, "objects": CommentedMap()})

    if not isinstance(root, dict) or root.get("version") != CONFIG_VERSION:
        raise ObjectOffsetError(
            f"物体外参配置 version 必须为 {CONFIG_VERSION}"
        )
    objects = root.get("objects")
    if not isinstance(objects, dict):
        raise ObjectOffsetError("物体外参 objects 必须是映射")
    if object_name in objects and not force:
        raise ObjectOffsetError(
            f"objects/{object_name} 已存在；确认覆盖时添加 --force"
        )
    objects[object_name] = _entry(offset)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.dump(root, stream)
            stream.flush()
            os.fsync(stream.fileno())
        load_object_offsets(temporary_path)
        if config_path.exists():
            os.chmod(temporary_path, config_path.stat().st_mode & 0o777)
        os.replace(temporary_path, config_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return offset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把 Motive Rigid Body > Visuals 中调好的 Geometry Location/"
            "Geometry Orientation 写为 T_motive_rigid_from_obj"
        ),
    )
    parser.add_argument(
        "object_name",
        help="物体名；必须与 acquisition/config.yaml 和 OBJ 文件名一致",
    )
    parser.add_argument(
        "--gl-mm",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Motive Geometry Location XYZ，单位 mm",
    )
    parser.add_argument(
        "--go-deg",
        nargs=3,
        type=float,
        required=True,
        metavar=("PITCH", "YAW", "ROLL"),
        help="Motive Geometry Orientation Pitch/Yaw/Roll，单位 deg、XYZ 顺序",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_OBJECT_OFFSETS_PATH,
        help=f"外参配置路径（默认 {DEFAULT_OBJECT_OFFSETS_PATH}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的同名物体外参",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        offset = write_offset(
            args.config,
            args.object_name,
            args.gl_mm,
            args.go_deg,
            force=args.force,
        )
    except (ObjectOffsetError, OSError) as exc:
        print(f"[add-object-offset] 失败: {exc}", file=sys.stderr)
        return 1

    print(f"[add-object-offset] 已写入: {args.config}")
    print(f"[add-object-offset] 物体: {args.object_name}")
    print(f"[add-object-offset] GL XYZ [mm]: {args.gl_mm}")
    print(f"[add-object-offset] GO Pitch/Yaw/Roll [deg]: {args.go_deg}")
    print(f"[add-object-offset] translation_m: {offset.translation_m.tolist()}")
    print("[add-object-offset] rotation_matrix:")
    for row in offset.rotation_matrix:
        print(f"  {row.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
