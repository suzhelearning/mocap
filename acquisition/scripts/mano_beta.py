#!/usr/bin/env python3
"""mano_beta.py — 离线估计 MANO 形状参数 β 并写入 HDF5 录制文件。

从录制骨架逐帧计算姿态无关的段长，做 MAD 异常值剔除和非线性有界拟合，
为每只手估计 β(10,)，写入 hands/<side>/mano_beta；回放/可视化直接使用。

用法:
  pixi run mano-beta -- [<file-or-dir>] [--samples 1000] [--force]
  pixi run mano-beta --                         # 当前日期目录
  pixi run mano-beta -- /home/current/data/20260812
  bash ../add_mano_beta.sh [-f] [日期目录]

批量模式幂等：已有 mano_beta 默认跳过；--force 用新算法全部重算。

安全防护:写回前先做外部链接检查(与 inspect/replay 一致),拒绝含
外部/软链接的 HDF5。
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



def collect_h5(paths: list[str]) -> list[Path]:
    """展开参数为 H5 文件列表;目录递归扫描(数据根/日期目录均可)。"""
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.h5")))
        else:
            files.append(path)
    # 去重保序
    seen: set[Path] = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def estimate_side(
    path: Path, side: str, samples: int,
    layer=None,
) -> tuple[SideEstimate | None, str | None]:
    """估计单侧 β；成功返回结果，失败返回错误信息。

    骨架来源优先 mano_skeleton(21 点)；旧录制只有 nodes_global(25 点)，
    按 MEDIAPIPE_FROM_MANUS 重排为 21 点后同样可估计。
    """
    from acquisition.manus_schema import MEDIAPIPE_FROM_MANUS

    if samples < 1:
        return None, f"samples 必须 >=1，实际为 {samples}"
    if layer is None:
        layer = load_mano(side)
    with h5py.File(path, "r") as f:
        reject_external_links(f)
        g = f.get("hands", {}).get(side)
        if g is None:
            return None, f"缺少 hands/{side}"
        if "mano_skeleton" in g:
            skel = g["mano_skeleton"][:]
            source = "mano_skeleton"
        elif "nodes_global" in g:
            nodes = np.asarray(g["nodes_global"][:], dtype=np.float64)
            if nodes.ndim == 3 and nodes.shape[1] == 25:
                skel = nodes[:, np.asarray(MEDIAPIPE_FROM_MANUS, dtype=np.int64), :]
                source = "nodes_global(25→21)"
            else:
                return None, f"hands/{side}/nodes_global 形状 {nodes.shape} 不支持"
        else:
            return None, f"hands/{side} 缺少 mano_skeleton 或 nodes_global"
    if skel.shape[1:] != (21, 3):
        return None, f"hands/{side} 骨架形状 {skel.shape} 不是 (N,21,3)"
    finite = np.isfinite(skel).all(axis=(1, 2))
    if not finite.any():
        return None, f"hands/{side} 骨架全为无效帧"

    valid = np.asarray(skel[finite], dtype=np.float64)
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


def has_beta(path: Path, side: str) -> bool:
    with h5py.File(path, "r") as f:
        reject_external_links(f)
        g = f.get("hands", {}).get(side)
        return g is not None and "mano_beta" in g


def write_beta(
    path: Path, side: str, estimate: SideEstimate, force: bool,
) -> None:
    with h5py.File(path, "r+") as f:
        reject_external_links(f)
        g = f["hands"][side]
        if "mano_beta" in g:
            if not force:
                raise FileExistsError(
                    f"{path.name} hands/{side}/mano_beta 已存在(--force 覆盖)")
            del g["mano_beta"]
        d = g.create_dataset(
            "mano_beta", (10,), dtype=np.float32, data=estimate.beta)
        d.attrs["source"] = (
            "mano_beta.py robust per-frame segment lengths "
            "+ nonlinear least squares"
        )
        d.attrs["skeleton"] = estimate.skeleton_source
        d.attrs["samples_used"] = estimate.samples_used
        d.attrs["segment_rms_mm"] = estimate.segment_rms_mm
        d.attrs["units"] = "MANO shape parameters (beta)"



def _default_today_dir() -> Path:
    """默认只处理当前日期目录(/home/current/data/YYYYMMDD)。"""
    from datetime import date

    root = Path("/home/current/data")
    return root / date.today().strftime("%Y%m%d")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="离线估计并写入 MANO beta 到 HDF5")
    ap.add_argument("paths", nargs="*", type=str,
                    help="H5 文件或文件夹(递归)。缺省 = 数据根目录下当前日期目录")
    ap.add_argument("--samples", type=int, default=1000,
                    help="均匀抽样的有效帧数上限(默认 1000)")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的 mano_beta(默认跳过,幂等)")
    args = ap.parse_args()

    targets = args.paths or [str(_default_today_dir())]
    files = collect_h5(targets)
    if not files:
        print("[mano-beta] 未找到 .h5 文件", file=sys.stderr)
        return 2

    layers = {side: load_mano(side) for side in ("left", "right")}
    n_written = 0
    n_skipped = 0
    n_failed = 0
    for path in files:
        for side, layer in layers.items():
            if not args.force and has_beta(path, side):
                n_skipped += 1
                continue
            estimate, error = estimate_side(
                path, side, args.samples, layer=layer,
            )
            if estimate is None:
                # 无骨架/形状不符/全无效帧 = 该文件无此数据，归为跳过；
                # 其余(估计异常等)才算失败。
                assert error is not None
                if error.startswith(("缺少", "hands/", "全为无效")):
                    n_skipped += 1
                else:
                    print(f"[mano-beta] 失败 {path.name} {side}: {error}")
                    n_failed += 1
                continue
            try:
                write_beta(path, side, estimate, args.force)
                print(
                    f"[mano-beta] {path.name} {side}: "
                    f"beta={estimate.beta.round(2).tolist()} "
                    f"[{estimate.skeleton_source}; {estimate.samples_used} 帧; "
                    f"段长RMS={estimate.segment_rms_mm:.2f}mm]"
                )
                n_written += 1
            except FileExistsError:
                n_skipped += 1
            except Exception as exc:
                print(f"[mano-beta] 写入失败 {path.name} {side}: {exc}")
                n_failed += 1
    print(f"[mano-beta] 完成:新增 {n_written},跳过 {n_skipped},失败 {n_failed}"
          f"(共 {len(files)} 个文件)")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
