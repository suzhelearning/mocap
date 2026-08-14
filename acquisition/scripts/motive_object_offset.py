#!/usr/bin/env python3
"""把 Motive 刚体位姿非破坏地预处理为真实 OBJ 坐标系位姿。

输入 HDF5 永远只读；输出默认是同目录 ``<stem>_obj.h5``。输出文件的
``objects/<name>/object_position`` 与 ``object_quaternion_xyzw`` 已处于 OBJ frame，
viewer 应直接读取，不再应用任何 offset。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from acquisition.object_offset import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_OBJECT_OFFSETS_PATH,
    ObjectOffsetError,
    default_output_path,
    preprocess_hdf5,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motive object rigid frame → 真实 OBJ frame 的非破坏 HDF5 预处理",
    )
    parser.add_argument("input", type=Path, help="原始 HDF5 文件（只读）")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="派生 HDF5；默认 <input_stem>_obj.h5，已存在时拒绝覆盖",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_OBJECT_OFFSETS_PATH,
        help=f"物体外参 YAML（默认 {DEFAULT_OBJECT_OFFSETS_PATH}）",
    )
    parser.add_argument(
        "--object", dest="objects", action="append",
        help="只处理指定物体，可重复；默认处理 HDF5 中全部物体",
    )
    parser.add_argument(
        "--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES,
        help=f"批处理帧数（默认 {DEFAULT_CHUNK_FRAMES}）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or default_output_path(args.input)
    try:
        result = preprocess_hdf5(
            args.input,
            output,
            config_path=args.config,
            object_names=args.objects,
            chunk_frames=args.chunk_frames,
        )
    except (ObjectOffsetError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"[object-offset] 失败: {exc}", file=sys.stderr)
        return 1

    print(f"[object-offset] 原始文件: {args.input}")
    print(f"[object-offset] 派生文件: {result.output_path}")
    print(f"[object-offset] 已处理物体: {', '.join(result.processed_objects)}")
    print(f"[object-offset] 原始 SHA-256: {result.source_sha256}")
    print(f"[object-offset] 配置 SHA-256: {result.config_sha256}")
    print("[object-offset] viewer 直接读取派生 HDF5 位姿，不再应用 offset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
