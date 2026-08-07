#!/usr/bin/env python3
"""inspect_hdf5.py — HDF5 录制文件检查:结构、形状、时间戳连续性、帧率。

用法: pixi run inspect -- <file.h5>
"""

from __future__ import annotations

import sys

import h5py
import numpy as np


def _timing(ds: h5py.Dataset, label: str) -> None:
    t = ds[:].astype(np.float64)
    if t.size < 2:
        print(f"  {label}: {t.size} 样本(不足 2,无法统计)")
        return
    dt = np.diff(t)
    dur = (t[-1] - t[0]) / 1e9
    mono = bool(np.all(dt > 0))
    gaps = np.sum(dt > np.median(dt) * 5) if dt.size > 2 else 0
    print(f"  {label}: {t.size} 样本, {dur:.2f}s, "
          f"{t.size / dur:.1f}Hz(实测), 单调={mono}, 大间隙={gaps}")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: pixi run inspect -- <file.h5>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        print(f"== {path} ==")
        for key in ("take_id", "start_wall_ns", "end_wall_ns"):
            print(f"  attr {key}: {f.attrs.get(key)}")
        print(f"  config: {len(f.attrs['config_yaml'])} 字节 yaml, "
              f"keymap={f.attrs['keymap']}")
        print("  -- mocap --")
        _timing(f["mocap/t_ubuntu_ns"], "t_ubuntu_ns")
        n_rb = len(f["mocap/rigid_bodies/ids"])
        n_mk = len(f["mocap/markers/positions"]) if "markers" in f["mocap"] else 0
        print(f"  rigid_bodies 帧: {n_rb}, markers 帧: {n_mk}")
        rb_ids = set()
        for row in f["mocap/rigid_bodies/ids"]:
            rb_ids.update(int(i) for i in row)
        print(f"  出现过的刚体 ID: {sorted(rb_ids)}")
        for side in ("left", "right"):
            g = f["hands"][side]
            print(f"  -- hands/{side} --")
            _timing(g["t_ubuntu_ns"], "t_ubuntu_ns")
            # 拼接自洽:手腕节点 g_0 == wrist_position
            g0 = g["nodes_global"][:, 0, :]
            err = np.abs(g0 - g["wrist_position"][:]).max()
            print(f"  nodes_global[0] vs wrist_position 最大误差: {err:.2e} m")
            print(f"  edges attr: {g.attrs.get('edges_json', '(无)')}")
        for name in f["objects"]:
            g = f["objects"][name]
            print(f"  -- objects/{name} --")
            _timing(g["t_ubuntu_ns"], name)
        ev = f["events"]
        print(f"  -- events: {len(ev['type'])} 条 --")
        for i in range(len(ev["type"])):
            print(f"    t={ev['t_ubuntu_ns'][i]} type={int(ev['type'][i])} "
                  f"note={ev['note'][i]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
