"""Motive 刚体位姿到真实 OBJ 坐标系位姿的非破坏预处理。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


DEFAULT_OBJECT_OFFSETS_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "object_offsets.yaml"
)
CONFIG_VERSION = 1
PROCESSING_NAME = "motive-object-offset"
PROCESSING_VERSION = 1
DEFAULT_CHUNK_FRAMES = 65_536


class ObjectOffsetError(ValueError):
    """物体外参配置或待处理 HDF5 不符合契约。"""


@dataclass(frozen=True)
class ObjectOffset:
    """固定外参 ``T_motive_rigid_from_obj``。"""

    translation_m: np.ndarray
    rotation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray


def offset_from_motive_visuals(
    geometry_location_xyz_mm: Sequence[float],
    geometry_orientation_pyr_deg: Sequence[float],
) -> ObjectOffset:
    """把 Motive Visuals 的 GL XYZ 与 GO Pitch/Yaw/Roll 转为物体外参。

    Motive 使用右手系和 XYZ 旋转顺序：Pitch 绕 X、Yaw 绕 Y、Roll 绕 Z。
    Geometry Location 的界面单位为毫米，配置与采集位姿统一使用米。
    """
    translation_mm = _readonly_f64(
        geometry_location_xyz_mm,
        (3,),
        "Geometry Location XYZ",
    )
    pyr_deg = _readonly_f64(
        geometry_orientation_pyr_deg,
        (3,),
        "Geometry Orientation Pitch/Yaw/Roll",
    )
    translation_m = _readonly_f64(
        translation_mm / 1000.0,
        (3,),
        "T_motive_rigid_from_obj translation_m",
    )
    rotation_matrix = _readonly_f64(
        Rotation.from_euler("xyz", pyr_deg, degrees=True).as_matrix(),
        (3, 3),
        "T_motive_rigid_from_obj rotation_matrix",
    )
    quaternion_xyzw = _readonly_f64(
        Rotation.from_matrix(rotation_matrix).as_quat(),
        (4,),
        "T_motive_rigid_from_obj quaternion_xyzw",
    )
    return ObjectOffset(translation_m, rotation_matrix, quaternion_xyzw)


@dataclass(frozen=True)
class PreprocessResult:
    """一次非破坏预处理的结果摘要。"""

    output_path: Path
    processed_objects: tuple[str, ...]
    source_sha256: str
    config_sha256: str


def _readonly_f64(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ObjectOffsetError(f"{label} 必须为 {shape}，实际 {array.shape}")
    if not np.isfinite(array).all():
        raise ObjectOffsetError(f"{label} 含 NaN/Inf")
    array.setflags(write=False)
    return array


def load_object_offsets(
    path: str | Path = DEFAULT_OBJECT_OFFSETS_PATH,
) -> dict[str, ObjectOffset]:
    """加载并严格校验每个物体的 ``T_motive_rigid_from_obj``。"""
    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as exc:
        raise ObjectOffsetError(f"无法读取物体外参配置 {config_path}: {exc}") from exc
    try:
        root = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ObjectOffsetError(f"物体外参 YAML 解析失败: {exc}") from exc
    if not isinstance(root, dict):
        raise ObjectOffsetError("物体外参配置根节点必须是映射")
    unknown_root = set(root) - {"version", "objects"}
    if unknown_root:
        raise ObjectOffsetError(f"物体外参配置含未知字段: {sorted(unknown_root)}")
    if root.get("version") != CONFIG_VERSION:
        raise ObjectOffsetError(
            f"物体外参 version 必须为 {CONFIG_VERSION}，实际 {root.get('version')!r}"
        )
    objects = root.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ObjectOffsetError("物体外参 objects 必须是非空映射")

    result: dict[str, ObjectOffset] = {}
    for name, entry in objects.items():
        if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
            raise ObjectOffsetError(f"非法物体名: {name!r}")
        if not isinstance(entry, dict) or set(entry) != {"motive_rigid_from_obj"}:
            raise ObjectOffsetError(
                f"objects/{name} 必须且只能包含 motive_rigid_from_obj"
            )
        transform = entry["motive_rigid_from_obj"]
        required = {"translation_m", "rotation_matrix"}
        if not isinstance(transform, dict) or set(transform) != required:
            raise ObjectOffsetError(
                f"objects/{name}/motive_rigid_from_obj 必须且只能包含 "
                "translation_m、rotation_matrix"
            )
        translation = _readonly_f64(
            transform["translation_m"], (3,),
            f"objects/{name}/motive_rigid_from_obj/translation_m",
        )
        matrix = _readonly_f64(
            transform["rotation_matrix"], (3, 3),
            f"objects/{name}/motive_rigid_from_obj/rotation_matrix",
        )
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6, rtol=0.0):
            raise ObjectOffsetError(f"objects/{name} rotation_matrix 不是正交矩阵")
        determinant = float(np.linalg.det(matrix))
        if not np.isclose(determinant, 1.0, atol=1e-6, rtol=0.0):
            raise ObjectOffsetError(
                f"objects/{name} rotation_matrix det 必须为 +1，实际 {determinant:.9g}"
            )
        quaternion = Rotation.from_matrix(matrix).as_quat()
        quaternion = _readonly_f64(
            quaternion, (4,), f"objects/{name}/quaternion_xyzw",
        )
        result[name] = ObjectOffset(translation, matrix, quaternion)
    return result


def require_object_offsets(
    object_names: Sequence[str],
    configured_offsets: dict[str, ObjectOffset],
) -> dict[str, ObjectOffset]:
    """返回当前采集物体的外参；任一缺失都拒绝启动采集。"""
    names = tuple(dict.fromkeys(object_names))
    missing = sorted(set(names) - set(configured_offsets))
    if missing:
        commands = "\n".join(
            "  pixi run add-object-offset -- "
            f"{name} --gl-mm X Y Z --go-deg PITCH YAW ROLL"
            for name in missing
        )
        raise ObjectOffsetError(
            f"当前物体缺少 OBJ offset: {missing}。\n"
            "请先在 Motive 的 Rigid Body > Visuals 中把几何体对齐，"
            "记录 Geometry Location (GL) XYZ [mm] 和 Geometry Orientation "
            "(GO) Pitch/Yaw/Roll [deg]，然后在 acquisition 目录运行:\n"
            f"{commands}"
        )
    return {name: configured_offsets[name] for name in names}


def object_offsets_sha256(path: str | Path = DEFAULT_OBJECT_OFFSETS_PATH) -> str:
    """返回外参配置字节的 SHA-256。"""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ObjectOffsetError(f"无法读取物体外参配置 {path}: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256，不把整份 HDF5 载入内存。"""
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise ObjectOffsetError(f"无法计算文件校验和 {path}: {exc}") from exc


def transform_object_poses(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    offset: ObjectOffset,
) -> tuple[np.ndarray, np.ndarray]:
    """批量计算 ``T_world_from_obj = T_world_from_rigid @ T_rigid_from_obj``。"""
    positions_f64 = np.asarray(positions, dtype=np.float64)
    quaternions_f64 = np.asarray(quaternions_xyzw, dtype=np.float64)
    if positions_f64.ndim != 2 or positions_f64.shape[1:] != (3,):
        raise ObjectOffsetError(f"positions 必须是 (N,3)，实际 {positions_f64.shape}")
    if quaternions_f64.shape != (len(positions_f64), 4):
        raise ObjectOffsetError(
            f"quaternions_xyzw 必须是 ({len(positions_f64)},4)，"
            f"实际 {quaternions_f64.shape}"
        )
    if not np.isfinite(positions_f64).all() or not np.isfinite(quaternions_f64).all():
        raise ObjectOffsetError("物体位姿含 NaN/Inf")
    if not len(positions_f64):
        return positions_f64.copy(), quaternions_f64.copy()

    norms = np.linalg.norm(quaternions_f64, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        index = int(np.flatnonzero(norms <= np.finfo(np.float64).eps)[0])
        raise ObjectOffsetError(f"第 {index} 帧四元数范数为零")
    q_world_from_rigid = quaternions_f64 / norms[:, None]

    vector = q_world_from_rigid[:, :3]
    scalar = q_world_from_rigid[:, 3:4]
    translation = np.broadcast_to(offset.translation_m, vector.shape)
    cross_once = np.cross(vector, translation)
    rotated_translation = translation + 2.0 * (
        scalar * cross_once + np.cross(vector, cross_once)
    )
    output_positions = positions_f64 + rotated_translation

    offset_vector = offset.quaternion_xyzw[:3]
    offset_scalar = float(offset.quaternion_xyzw[3])
    output_vector = (
        scalar * offset_vector
        + offset_scalar * vector
        + np.cross(vector, offset_vector)
    )
    output_scalar = (
        scalar[:, 0] * offset_scalar
        - np.einsum("ij,j->i", vector, offset_vector)
    )
    output_quaternions = np.concatenate(
        (output_vector, output_scalar[:, None]), axis=1,
    )
    output_quaternions /= np.linalg.norm(output_quaternions, axis=1)[:, None]
    return output_positions, output_quaternions


def _reject_external_links(group: h5py.Group, prefix: str = "") -> None:
    """拒绝外部/软链接，避免预处理结果携带文件系统引用。"""
    for name in group:
        link = group.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ObjectOffsetError(
                f"拒绝含 {type(link).__name__} 的 HDF5: {prefix}{name}"
            )
        value = group[name]
        if isinstance(value, h5py.Group):
            _reject_external_links(value, f"{prefix}{name}/")


def _validate_object_group(name: str, group: h5py.Group) -> None:
    if group.attrs.get("object_pose_frame") == "obj":
        raise ObjectOffsetError(f"objects/{name} 已是 OBJ 位姿，拒绝重复处理")
    required = {
        "object_position": (3,),
        "object_quaternion_xyzw": (4,),
    }
    lengths: set[int] = set()
    for dataset_name, tail_shape in required.items():
        if dataset_name not in group:
            raise ObjectOffsetError(f"objects/{name} 缺少 {dataset_name}")
        dataset = group[dataset_name]
        if not isinstance(dataset, h5py.Dataset):
            raise ObjectOffsetError(f"objects/{name}/{dataset_name} 不是 dataset")
        if dataset.ndim != 2 or dataset.shape[1:] != tail_shape:
            raise ObjectOffsetError(
                f"objects/{name}/{dataset_name} 形状必须是 (N,{tail_shape[0]})，"
                f"实际 {dataset.shape}"
            )
        if not np.issubdtype(dataset.dtype, np.floating):
            raise ObjectOffsetError(
                f"objects/{name}/{dataset_name} 必须是浮点 dataset，"
                f"实际 {dataset.dtype}"
            )
        lengths.add(dataset.shape[0])
    if len(lengths) != 1:
        raise ObjectOffsetError(f"objects/{name} position/quaternion 帧数不一致")


def default_output_path(source: str | Path) -> Path:
    """返回默认派生文件名 ``<source_stem>_obj.h5``。"""
    source_path = Path(source)
    return source_path.with_name(f"{source_path.stem}_obj{source_path.suffix}")


def preprocess_hdf5(
    source: str | Path,
    output: str | Path | None = None,
    *,
    config_path: str | Path = DEFAULT_OBJECT_OFFSETS_PATH,
    object_names: Sequence[str] | None = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> PreprocessResult:
    """复制原始 HDF5，并在派生副本中把指定物体标准位姿字段转换到 OBJ frame。"""
    source_path = Path(source).resolve(strict=True)
    output_path = Path(output) if output is not None else default_output_path(source_path)
    output_path = output_path.resolve(strict=False)
    if source_path == output_path:
        raise ObjectOffsetError("输出路径不能与原始 HDF5 相同")
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，拒绝覆盖: {output_path}")
    if chunk_frames < 1:
        raise ObjectOffsetError(f"chunk_frames 必须 >=1，实际 {chunk_frames}")

    offsets = load_object_offsets(config_path)
    config_hash = object_offsets_sha256(config_path)
    with h5py.File(source_path, "r") as source_h5:
        _reject_external_links(source_h5)
        if str(source_h5.attrs.get("h5_version", "")) in {"3.0", "4.0"}:
            raise ObjectOffsetError(
                "v3/v4 已包含同 tick 的 object 坐标；"
                "请在实时采集配置中应用外参，拒绝离线改写"
            )
        objects = source_h5.get("objects")
        if not isinstance(objects, h5py.Group) or len(objects) == 0:
            raise ObjectOffsetError("HDF5 缺少非空 objects 组")
        available = tuple(sorted(objects.keys()))
        if object_names is None:
            selected = available
        else:
            selected = tuple(dict.fromkeys(object_names))
            if not selected:
                raise ObjectOffsetError("--object 至少指定一个物体")
        missing_h5 = sorted(set(selected) - set(available))
        if missing_h5:
            raise ObjectOffsetError(f"HDF5 不含物体: {missing_h5}")
        missing_config = sorted(set(selected) - set(offsets))
        if missing_config:
            raise ObjectOffsetError(f"外参配置缺少物体: {missing_config}")
        for name in selected:
            _validate_object_group(name, objects[name])

    source_hash = file_sha256(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(source_path, temporary_path)
        if file_sha256(temporary_path) != source_hash:
            raise ObjectOffsetError("复制期间原始 HDF5 发生变化，拒绝生成派生文件")
        with h5py.File(temporary_path, "r+") as derived_h5:
            for name in selected:
                group = derived_h5["objects"][name]
                positions = group["object_position"]
                quaternions = group["object_quaternion_xyzw"]
                offset = offsets[name]
                for start in range(0, positions.shape[0], chunk_frames):
                    stop = min(start + chunk_frames, positions.shape[0])
                    output_positions, output_quaternions = transform_object_poses(
                        positions[start:stop], quaternions[start:stop], offset,
                    )
                    positions[start:stop] = output_positions
                    quaternions[start:stop] = output_quaternions
                group.attrs["object_pose_frame"] = "obj"
                group.attrs["source_object_pose_frame"] = "motive_rigid"
                group.attrs["motive_rigid_from_obj_translation_m"] = offset.translation_m
                group.attrs["motive_rigid_from_obj_rotation_matrix"] = offset.rotation_matrix
                group.attrs["object_offset_config_sha256"] = config_hash
            derived_h5.attrs["object_pose_processing"] = (
                f"{PROCESSING_NAME}/v{PROCESSING_VERSION}"
            )
            derived_h5.attrs["object_pose_frame"] = (
                "obj" if set(selected) == set(available) else "mixed"
            )
            derived_h5.attrs["processed_objects_json"] = json.dumps(
                selected, ensure_ascii=False,
            )
            derived_h5.attrs["source_hdf5_filename"] = source_path.name
            derived_h5.attrs["source_hdf5_sha256"] = source_hash
            derived_h5.attrs["object_offset_config_sha256"] = config_hash
            derived_h5.flush()
        os.link(temporary_path, output_path)
        temporary_path.unlink()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return PreprocessResult(output_path, selected, source_hash, config_hash)
