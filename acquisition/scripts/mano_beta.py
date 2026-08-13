#!/usr/bin/env python3
"""mano_beta.py — 从手部关键点估计一次性 MANO beta 并写入 HDF5。

每只手从最多 1000 个有效 ``mano_skeleton`` 帧稳健估计 ``mano_beta(10)``。
逐帧表面直接由 ``mano_skeleton + mano_beta`` 驱动；不拟合、也不保存 pose、
translation、scale、joints16 或 valid 等可派生数组。

用法:
  pixi run mano-beta -- [<file-or-dir>] [--samples 1000] [--force]
  pixi run mano-beta --                         # 当前日期目录
  pixi run mano-beta -- /home/current/data/20260812
  bash ../add_mano_beta.sh [-f] [日期目录]

安全防护：写回前拒绝含 HDF5 外部/软链接的文件。
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from acquisition.viser_core import reject_external_links

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mano_fit import beta_segment_rms_mm, estimate_beta, load_mano  # noqa: E402


@dataclass(frozen=True)
class SideEstimate:
    beta: np.ndarray
    skeleton_source: str
    samples_used: int
    segment_rms_mm: float


# 旧版逐帧拟合缓存；新契约可由 mano_skeleton + mano_beta 确定性派生。
_LEGACY_DERIVED_FIELDS = (
    "mano_pose",
    "mano_translation",
    "mano_scale",
    "mano_joints16",
    "mano_fit_valid",
)


def collect_h5(paths: list[str]) -> list[Path]:
    """展开参数为 H5 文件列表；目录递归扫描，结果去重且稳定排序。"""
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.h5")))
        else:
            files.append(path)
    seen: set[Path] = set()
    result: list[Path] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _load_skeleton(
    path: Path, side: str,
) -> tuple[np.ndarray | None, str]:
    """读取 MediaPipe 21 点；旧 25 点文件仅用于 beta 估计时自动转换。"""
    from acquisition.manus_schema import MEDIAPIPE_FROM_MANUS

    with h5py.File(path, "r") as f:
        reject_external_links(f)
        hands = f.get("hands")
        group = None if hands is None else hands.get(side)
        if group is None:
            return None, f"缺少 hands/{side}"
        if "mano_skeleton" in group:
            skeleton = np.asarray(group["mano_skeleton"][:], dtype=np.float64)
            source = "mano_skeleton"
        elif "nodes_global" in group:
            nodes = np.asarray(group["nodes_global"][:], dtype=np.float64)
            if nodes.ndim != 3 or nodes.shape[1:] != (25, 3):
                return None, f"hands/{side}/nodes_global 形状 {nodes.shape} 不支持"
            skeleton = nodes[:, np.asarray(MEDIAPIPE_FROM_MANUS), :]
            source = "nodes_global(25→21)"
        else:
            return None, f"hands/{side} 缺少 mano_skeleton 或 nodes_global"
    if skeleton.ndim != 3 or skeleton.shape[1:] != (21, 3):
        return None, f"hands/{side} 骨架形状 {skeleton.shape} 不是 (N,21,3)"
    return skeleton, source


def _estimate_skeleton(
    layer, skeleton: np.ndarray, source: str, samples: int,
) -> tuple[SideEstimate | None, str | None]:
    if samples < 1:
        return None, f"samples 必须 >=1，实际为 {samples}"
    finite = np.isfinite(skeleton).all(axis=(1, 2))
    if not finite.any():
        return None, "骨架全为无效帧"
    valid = skeleton[finite]
    count = min(samples, len(valid))
    indices = np.linspace(0, len(valid) - 1, count, dtype=np.int64)
    frames = valid[indices]
    try:
        beta = estimate_beta(layer, frames)
        rms_mm = beta_segment_rms_mm(layer, beta, frames)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return SideEstimate(
        beta=np.asarray(beta, dtype=np.float32),
        skeleton_source=source,
        samples_used=count,
        segment_rms_mm=rms_mm,
    ), None


def estimate_side(
    path: Path, side: str, samples: int, layer=None,
) -> tuple[SideEstimate | None, str | None]:
    """从最多 ``samples`` 个有效关键点帧估计单侧 beta。"""
    if layer is None:
        layer = load_mano(side)
    skeleton, source = _load_skeleton(path, side)
    if skeleton is None:
        return None, source
    return _estimate_skeleton(layer, skeleton, source, samples)


def _group_has_current_beta(group: h5py.Group) -> bool:
    if "mano_beta" not in group:
        return False
    beta = np.asarray(group["mano_beta"][:])
    return (
        beta.shape == (10,)
        and np.isfinite(beta).all()
        and not any(name in group for name in _LEGACY_DERIVED_FIELDS)
    )


def has_mano_beta(path: Path, side: str) -> bool:
    with h5py.File(path, "r") as f:
        reject_external_links(f)
        hands = f.get("hands")
        group = None if hands is None else hands.get(side)
        return group is not None and _group_has_current_beta(group)


def write_beta(
    path: Path, side: str, estimate: SideEstimate, force: bool,
) -> None:
    """写入 beta，并移除旧版逐帧拟合缓存。"""
    beta_value = np.asarray(estimate.beta, dtype=np.float32)
    if beta_value.shape != (10,) or not np.isfinite(beta_value).all():
        raise ValueError(f"MANO beta 必须是有限 (10,)，实际为 {beta_value.shape}")

    with h5py.File(path, "r+") as f:
        reject_external_links(f)
        group = f["hands"][side]
        if _group_has_current_beta(group) and not force:
            raise FileExistsError(
                f"{path.name} hands/{side}/mano_beta 已存在(--force 覆盖)",
            )
        for name in _LEGACY_DERIVED_FIELDS:
            if name in group:
                del group[name]
        if "mano_beta" in group:
            del group["mano_beta"]
        beta = group.create_dataset("mano_beta", data=beta_value, dtype=np.float32)
        beta.attrs["source"] = "robust per-frame segment lengths"
        beta.attrs["skeleton"] = estimate.skeleton_source
        beta.attrs["samples_used"] = estimate.samples_used
        beta.attrs["segment_rms_mm"] = estimate.segment_rms_mm
        beta.attrs["units"] = "MANO shape parameters (beta)"


def _default_today_dir() -> Path:
    from datetime import date

    return Path("/home/current/data") / date.today().strftime("%Y%m%d")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从关键点估计 MANO beta 并写入 HDF5",
    )
    parser.add_argument(
        "paths", nargs="*", type=str,
        help="H5 文件或文件夹(递归)；缺省为当前日期目录",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="beta 估计均匀抽样的有效帧数上限(默认 1000)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="覆盖已有 beta；旧版逐帧拟合字段会被移除",
    )
    args = parser.parse_args()

    targets = args.paths or [str(_default_today_dir())]
    files = collect_h5(targets)
    if not files:
        print("[mano-beta] 未找到 .h5 文件", file=sys.stderr)
        return 2

    layers = {side: load_mano(side) for side in ("left", "right")}
    written = skipped = failed = 0
    for path in files:
        for side, layer in layers.items():
            if not args.force and has_mano_beta(path, side):
                skipped += 1
                continue
            print(
                f"[mano-beta] 开始 {path.name} {side} "
                f"(beta≤{args.samples}帧，无逐帧拟合)",
                flush=True,
            )
            estimate, error = estimate_side(
                path, side, args.samples, layer=layer,
            )
            if estimate is None:
                assert error is not None
                if error.startswith(("缺少", "hands/", "骨架全为")):
                    print(f"[mano-beta] 跳过 {path.name} {side}: {error}")
                    skipped += 1
                else:
                    print(f"[mano-beta] 失败 {path.name} {side}: {error}")
                    failed += 1
                continue
            try:
                write_beta(path, side, estimate, args.force)
                print(
                    f"[mano-beta] {path.name} {side}: "
                    f"beta={estimate.beta.round(2).tolist()} "
                    f"[{estimate.skeleton_source}; {estimate.samples_used}帧; "
                    f"段长RMS={estimate.segment_rms_mm:.2f}mm]",
                )
                written += 1
            except FileExistsError:
                skipped += 1
            except Exception as exc:
                print(f"[mano-beta] 写入失败 {path.name} {side}: {exc}")
                failed += 1
    print(
        f"[mano-beta] 完成:写入 {written},跳过 {skipped},失败 {failed}"
        f"(共 {len(files)} 个文件)",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
