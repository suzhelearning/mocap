"""配置加载与校验:yaml → 冻结 dataclass。

校验规则:
- hands 必须含 left/right,且各自的 back_rigid_id 在 rigid_bodies 中存在
- 物体 ID 不得与 back 重复
- axis_transform 合成的 3×3 矩阵必须 det == +1(真旋转,保证右手系)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULT_KEYMAP = {
    "start": "r",
    "save": "s",
    "discard": "d",
    "quit": "q",
}


class ConfigError(ValueError):
    """配置不合法。"""


@dataclass(frozen=True)
class WristOffset:
    mode: str = "body"                     # "body" 随背部刚体旋转 | "world" 固定世界方向
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)   # 背部刚体身体系 / 世界系下的偏移
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass(frozen=True)
class HandConfig:
    side: str
    back_rigid_id: int | None = None      # 背部刚体基准(可省略,改用 wrist_rigid_id)
    wrist_offset: WristOffset = field(default_factory=WristOffset)
    wrist_rigid_id: int | None = None     # Motive 直接追踪手腕刚体时跳过 back/offset


@dataclass(frozen=True)
class Config:
    router_endpoint: str
    back_rigid_id: int | None              # None=全部手用 wrist_rigid_id 直接追踪
    objects: dict[str, int]
    hands: dict[str, HandConfig]
    axis_permutation: tuple[int, int, int]
    axis_signs: tuple[int, int, int]
    output_dir: Path
    store_markers: bool
    sample_hz: float               # 采集落盘目标频率(输入 120Hz 时默认降到 100Hz)
    keymap: dict[str, str]
    viz_port: int
    chunk_frames: int
    config_path: Path
    config_text: str                      # 原样保存进 HDF5 attrs,便于溯源

    def axis_matrix(self) -> np.ndarray:
        """骨架系 → Motive 系的 3×3 轴变换矩阵。(A·d)_j = signs[j]·d[permutation[j]]"""
        A = np.zeros((3, 3), dtype=float)
        for j in range(3):
            A[j, self.axis_permutation[j]] = self.axis_signs[j]
        return A


def _require(mapping: dict, key: str, path: str) -> object:
    if key not in mapping:
        raise ConfigError(f"配置缺少 {path}.{key}")
    return mapping[key]


def _parse_offset(data: object, path: str) -> WristOffset:
    if data is None:
        return WristOffset()
    if not isinstance(data, dict):
        raise ConfigError(f"{path} 必须是映射")
    mode = data.get("mode", "body")
    if mode not in ("body", "world"):
        raise ConfigError(f"{path}.mode 必须是 body 或 world,实际 {mode!r}")
    xyz = data.get("xyz", (0.0, 0.0, 0.0))
    if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3
            and all(isinstance(v, (int, float)) for v in xyz)):
        raise ConfigError(f"{path}.xyz 必须是 3 个数字")
    return WristOffset(
        mode=mode,
        xyz=tuple(float(v) for v in xyz),
        yaw_deg=float(data.get("yaw_deg", 0.0)),
        pitch_deg=float(data.get("pitch_deg", 0.0)),
        roll_deg=float(data.get("roll_deg", 0.0)),
    )


def load_config(path: str | Path) -> Config:
    """加载 yaml 并校验,失败抛 ConfigError。"""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 {path} 不是合法 YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("配置根必须是映射")

    router = _require(raw, "router", "config")
    if not isinstance(router, dict) or "endpoint" not in router:
        raise ConfigError("config.router.endpoint 必须指定,如 tcp/127.0.0.1:7447")

    rb = _require(raw, "rigid_bodies", "config")
    if not isinstance(rb, dict):
        raise ConfigError("config.rigid_bodies 必须是映射")
    # back 可选:Motive 直接追踪手腕刚体(hands 全部配 wrist_rigid_id)时可省略
    back_id = rb.get("back")
    if back_id is not None and not isinstance(back_id, int):
        raise ConfigError("rigid_bodies.back 必须是整数 ID")
    objects_raw = rb.get("objects", {})
    if not isinstance(objects_raw, dict):
        raise ConfigError("rigid_bodies.objects 必须是映射")
    objects: dict[str, int] = {}
    for name, oid in objects_raw.items():
        if not isinstance(oid, int):
            raise ConfigError(f"rigid_bodies.objects.{name} 必须是整数 ID")
        if oid == back_id:
            raise ConfigError(f"物体 {name} 的 ID {oid} 与背部刚体重复")
        objects[name] = oid

    hands_raw = _require(raw, "hands", "config")
    if not isinstance(hands_raw, dict):
        raise ConfigError("config.hands 必须是映射")
    hands: dict[str, HandConfig] = {}
    for side in ("left", "right"):
        if side not in hands_raw:
            raise ConfigError(f"config.hands 缺少 {side}")
        h = hands_raw[side]
        if not isinstance(h, dict):
            raise ConfigError(f"config.hands.{side} 必须是映射")
        wrist_rid = h.get("wrist_rigid_id")
        if wrist_rid is not None and not isinstance(wrist_rid, int):
            raise ConfigError(f"hands.{side}.wrist_rigid_id 必须是整数 ID")
        hid = h.get("back_rigid_id")
        if hid is not None and not isinstance(hid, int):
            raise ConfigError(f"hands.{side}.back_rigid_id 必须是整数 ID")
        # 每只手必须能确定位姿来源:back_rigid_id 或 wrist_rigid_id 至少一个。
        # back_rigid_id 允许引用任意 Motive 刚体 ID(如手套背面 marker 刚体),
        # 不要求出现在 rigid_bodies.back/objects 中;back 用于全局渲染与默认。
        if hid is None and wrist_rid is None:
            raise ConfigError(
                f"hands.{side} 必须指定 back_rigid_id 或 wrist_rigid_id(至少一个)"
            )
        hands[side] = HandConfig(
            side=side,
            back_rigid_id=hid,
            wrist_offset=_parse_offset(h.get("wrist_offset"), f"hands.{side}.wrist_offset"),
            wrist_rigid_id=wrist_rid,
        )
    # 无背部刚体时,所有手必须直接追踪手腕(否则拼接无基准)
    if back_id is None and any(h.back_rigid_id is not None and h.wrist_rigid_id is None
                               for h in hands.values()):
        raise ConfigError(
            "rigid_bodies.back 未配置,所有 hands 都必须使用 wrist_rigid_id"
        )

    axis = raw.get("axis_transform", {"permutation": [0, 2, 1], "signs": [1, 1, -1]})
    if not isinstance(axis, dict):
        raise ConfigError("axis_transform 必须是映射")
    perm = axis.get("permutation", [0, 2, 1])
    signs = axis.get("signs", [1, 1, -1])
    if not (isinstance(perm, list) and isinstance(signs, list)
            and sorted(perm) == [0, 1, 2] and len(signs) == 3
            and all(s in (-1, 1) for s in signs)):
        raise ConfigError("axis_transform.permutation 须为 0..2 的排列,signs 须为 ±1")
    # det == +1 校验(真旋转)
    A = np.zeros((3, 3))
    for j in range(3):
        A[j, perm[j]] = signs[j]
    if not np.isclose(np.linalg.det(A), 1.0):
        raise ConfigError(
            f"axis_transform 不是真旋转(det={np.linalg.det(A):.2f}):"
            "permutation/signs 组合非法,请参考文档调整"
        )

    rec = raw.get("recording", {})
    if not isinstance(rec, dict):
        raise ConfigError("recording 必须是映射")
    output_dir = Path(rec.get("output_dir", "captures"))
    store_markers = bool(rec.get("store_markers", True))
    sample_hz = float(rec.get("sample_hz", 100.0))
    if not (sample_hz > 0 and sample_hz <= 10000):
        raise ConfigError(f"recording.sample_hz 必须是正数,实际 {sample_hz}")
    chunk_frames = int(rec.get("chunk_frames", 4096))

    keys_raw = raw.get("keys", {})
    keymap = {k: str(v) for k, v in DEFAULT_KEYMAP.items()}
    if isinstance(keys_raw, dict):
        for k, v in keys_raw.items():
            if k in keymap:
                keymap[k] = str(v)
    if len(set(keymap.values())) != len(keymap):
        raise ConfigError("keys 存在重复键位")

    viz = raw.get("viz", {})
    viz_port = int(viz.get("port", 8081)) if isinstance(viz, dict) else 8081

    return Config(
        router_endpoint=str(router["endpoint"]),
        back_rigid_id=back_id,
        objects=objects,
        hands=hands,
        axis_permutation=tuple(perm),
        axis_signs=tuple(signs),
        output_dir=output_dir,
        store_markers=store_markers,
        sample_hz=sample_hz,
        keymap=keymap,
        viz_port=viz_port,
        chunk_frames=chunk_frames,
        config_path=path.resolve(),
        config_text=path.read_text(encoding="utf-8"),
    )
