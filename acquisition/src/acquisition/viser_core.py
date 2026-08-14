"""HDF5 动捕数据 → viser 场景的共享核心(viz_hdf5 与 data-viewer adapter 共用)。

职责:配色拓扑常量、HDF5 提取、场景节点构建、单帧应用。
纯场景操作,不持有 ViserServer / 播放状态——实时服务(viz_hdf5)和离线录制
(data-viewer h5 adapter)都只依赖本模块,保证两处视觉与数据语义一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import viser

from .kinematics import quat_xyzw_to_wxyz

# ---- 配色与拓扑(与 live_view.py 保持一致) ----
CHAIN_COLORS = {
    5: (244, 162, 97),    # 拇指 橙
    6: (42, 157, 143),    # 食指 青
    7: (233, 196, 106),   # 中指 黄
    8: (231, 111, 81),    # 无名指 橙红
    9: (69, 123, 157),    # 小指 蓝
    13: (210, 210, 210),  # 手掌 亮灰
}
DEFAULT_COLOR = (160, 160, 160)

MARKER_COLORS = {
    "active": (45, 212, 191),
    "asset_member": (74, 222, 128),
    "point_cloud": (251, 146, 60),
    "unknown": (148, 163, 184),
}
OCCLUDED_COLOR = (239, 68, 68)

# MANO 25 节点:0=手腕,1-4=拇指,5-8=食指,9-12=中指,13-16=无名指,17-20=小指,
# 21-24=手掌扩展点(不连线)
MANO_PALETTE = np.asarray(
    [(210, 210, 210)]                       # 0 手腕
    + [(244, 162, 97)] * 4                  # 拇指
    + [(42, 157, 143)] * 4                  # 食指
    + [(233, 196, 106)] * 4                 # 中指
    + [(231, 111, 81)] * 4                  # 无名指
    + [(69, 123, 157)] * 4                  # 小指
    + [(160, 160, 160)] * 4,                # 手掌扩展
    dtype=np.uint8,
)
MANO_PALM_EDGES = ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17))
MANO_FINGER_EDGES = (
    (1, 2), (2, 3), (3, 4),        # 拇指
    (5, 6), (6, 7), (7, 8),        # 食指
    (9, 10), (10, 11), (11, 12),   # 中指
    (13, 14), (14, 15), (15, 16),  # 无名指
    (17, 18), (18, 19), (19, 20),  # 小指
)

# 刚体 ID → 标签(与 config.yaml 对应;未配置的 ID 显示原始编号)
RIGID_LABELS = {1: "左腕(back)", 2: "右腕(back)", 3: "cylinder"}

HAND_EDGES = [(c, p) for c, p in MANO_PALM_EDGES] + list(MANO_FINGER_EDGES)

# 存储为数字索引(recorder._KIND_INDEX)
KIND_INDEX = {"active": 0, "asset_member": 1, "point_cloud": 2, "unknown": 3}
_KIND_COLORS = {v: MARKER_COLORS[k] for k, v in KIND_INDEX.items()}

OBJECT_MESH_DIR = Path(__file__).resolve().parents[3] / "assets" / "objects"
OBJECT_MESH_COLOR = (245, 158, 11)


def find_object_mesh(name: str, mesh_dir: Path = OBJECT_MESH_DIR) -> Path | None:
    """按 HDF5 物体名匹配 OBJ：优先米制 ``<name>_m.obj``，兼容毫米制 ``<name>.obj``。"""
    if not mesh_dir.is_dir():
        return None
    wanted = (f"{name}_m".casefold(), name.casefold())
    by_stem = {
        path.stem.casefold(): path
        for path in mesh_dir.glob("*.obj")
        if path.is_file()
    }
    return next((by_stem[stem] for stem in wanted if stem in by_stem), None)


@lru_cache(maxsize=32)
def _load_obj_mesh_cached(
    path_str: str, mtime_ns: int, scale_to_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 OBJ 顶点和面；mtime 作为缓存键，资源替换后自动重载。"""
    del mtime_ns
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with Path(path_str).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"{path_str}:{line_no} 顶点字段不足")
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
                continue
            if fields[0] != "f":
                continue
            if len(fields) < 4:
                raise ValueError(f"{path_str}:{line_no} 面少于 3 个顶点")
            indices: list[int] = []
            for field in fields[1:]:
                raw = field.split("/", 1)[0]
                if not raw:
                    raise ValueError(f"{path_str}:{line_no} 面顶点索引为空")
                index = int(raw)
                index = len(vertices) + index if index < 0 else index - 1
                if not 0 <= index < len(vertices):
                    raise ValueError(f"{path_str}:{line_no} 面顶点索引越界")
                indices.append(index)
            faces.extend(
                (indices[0], indices[i], indices[i + 1])
                for i in range(1, len(indices) - 1)
            )
    verts = np.asarray(vertices, dtype=np.float32) * np.float32(scale_to_m)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1:] != (3,) or not len(verts):
        raise ValueError(f"{path_str} 不含有效 OBJ 顶点")
    if tris.ndim != 2 or tris.shape[1:] != (3,) or not len(tris):
        raise ValueError(f"{path_str} 不含有效 OBJ 面")
    if not np.isfinite(verts).all():
        raise ValueError(f"{path_str} 顶点含 NaN/Inf")
    verts.setflags(write=False)
    tris.setflags(write=False)
    return verts, tris


def load_object_mesh(
    name: str, mesh_dir: Path = OBJECT_MESH_DIR,
) -> tuple[Path, np.ndarray, np.ndarray] | None:
    """加载同名 OBJ 并统一为米，返回 ``(path, vertices, triangle_faces)``。

    ``<name>_m.obj`` 的顶点已经是米；SolidWorks/Blender 导出的普通
    ``<name>.obj`` 按毫米解释并乘 ``0.001``。
    """
    path = find_object_mesh(name, mesh_dir)
    if path is None:
        return None
    scale_to_m = 1.0 if path.stem.casefold().endswith("_m") else 1e-3
    vertices, faces = _load_obj_mesh_cached(
        str(path), path.stat().st_mtime_ns, scale_to_m,
    )
    return path, vertices, faces


def object_mesh_assets_mtime_ns(mesh_dir: Path = OBJECT_MESH_DIR) -> int:
    """返回 OBJ 资源的最新修改时间，供离线 viewer 缓存失效判断。"""
    if not mesh_dir.is_dir():
        return 0
    return max(
        (path.stat().st_mtime_ns for path in mesh_dir.glob("*.obj") if path.is_file()),
        default=0,
    )


def reject_external_links(f: h5py.File, prefix: str = "") -> None:
    """递归检查并拒绝含外部/软链接的 HDF5(不解析链接,仅查类型)。"""
    for name in f:
        link = f.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError(
                f"拒绝含 {type(link).__name__} 的 HDF5: {prefix}{name}"
                + (f" -> {link.filename}" if isinstance(link, h5py.ExternalLink) else "")
            )
        obj = f[name]
        if isinstance(obj, h5py.Group):
            reject_external_links(obj, f"{prefix}{name}/")


def nearest_idx(t: np.ndarray, t_ns: float) -> int:
    """在单调时间戳数组上取 t_ns 的最近邻下标(夹取到边界)。"""
    i = int(np.searchsorted(t, t_ns, side="right")) - 1
    return min(max(i, 0), t.size - 1)

def marker_colors(kinds, occluded) -> np.ndarray:
    colors = np.zeros((len(kinds), 3), dtype=np.uint8)
    for i, (kind, occ) in enumerate(zip(kinds, occluded)):
        if bool(occ):
            colors[i] = OCCLUDED_COLOR
            continue
        colors[i] = _KIND_COLORS.get(int(kind), MARKER_COLORS["unknown"])
    return colors


def extract_hdf5(f: h5py.File) -> dict:
    """从 HDF5 提取回放数据(全部入内存;22s 录段仅数 MB)。"""
    mocap = f.get("mocap")
    if mocap is not None:
        t_mocap = (
            mocap["t_aligned_ubuntu_ns"][:]
            if "t_aligned_ubuntu_ns" in mocap
            else mocap["t_ubuntu_ns"][:]
        ).astype(np.int64)
    elif "objects" in f and f["objects"]:
        # 新 schema(双手 MANO + 物体):以物体刚体轴为参考时间轴
        t_mocap = next(iter(f["objects"].values()))["t_ubuntu_ns"][:].astype(np.int64)
    else:
        t_mocap = f["hands"]["left"]["t_ubuntu_ns"][:].astype(np.int64)
    n = len(t_mocap)

    def ragged_row(group: h5py.Group, field: str, index: int) -> np.ndarray:
        dataset = group[field]
        if "frame_offsets" in group:
            start = int(group["frame_offsets"][index])
            end = int(group["frame_offsets"][index + 1])
            return np.asarray(dataset[start:end])
        return np.asarray(dataset[index])

    # 刚体:兼容 v1 vlen 行与 v2 offsets+flat;新 schema 无 mocap 时为空。
    rb_frames: list[dict[int, tuple[np.ndarray, np.ndarray, bool]]] = []
    rb_ids: set[int] = set()
    if mocap is not None:
        rb = mocap["rigid_bodies"]
        for i in range(n):
            ids = ragged_row(rb, "ids", i).astype(np.int64)
            pos = ragged_row(rb, "positions", i).astype(np.float64).reshape(-1, 3)
            quat = ragged_row(
                rb, "quaternions_xyzw", i).astype(np.float64).reshape(-1, 4)
            valid = ragged_row(rb, "tracking_valid", i)
            frame = {}
            for j, rid in enumerate(ids):
                rid = int(rid)
                frame[rid] = (pos[j], quat[j], bool(valid[j]))
                rb_ids.add(rid)
            rb_frames.append(frame)

    # markers:兼容 v1 vlen 行与 v2 offsets+flat;新 schema 无 mocap 时为空。
    mk_frames: list[tuple[np.ndarray, np.ndarray]] = []
    if mocap is not None and "markers" in mocap:
        mk = mocap["markers"]
        for i in range(n):
            pts = ragged_row(mk, "positions", i).astype(np.float32).reshape(-1, 3)
            colors = marker_colors(
                ragged_row(mk, "id_kinds", i),
                ragged_row(mk, "occluded", i),
            )
            mk_frames.append((pts, colors))
    else:
        mk_frames = [
            (np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8))
            for _ in range(n)
        ]

    # 双手骨架按自身时间戳对齐。MANO 表面只需要原始 21 点和一次性 beta；
    # 16 关键点由 mano_skeleton 直接重排，不加载任何逐帧拟合缓存。
    hands: dict[str, dict] = {}
    for side in ("left", "right"):
        g = f["hands"][side]
        node_key = "mano_skeleton" if "mano_skeleton" in g else "nodes_global"
        t_hand = g["t_ubuntu_ns"][:].astype(np.int64)
        nodes = np.asarray(g[node_key][:])
        if nodes.ndim != 3 or nodes.shape[2] != 3:
            raise ValueError(
                f"hands/{side}/{node_key} 必须是 (N,K,3)，实际为 {nodes.shape}",
            )
        beta = None
        if "mano_beta" in g:
            beta = np.asarray(g["mano_beta"][:], dtype=np.float64)
            if beta.shape != (10,) or not np.isfinite(beta).all():
                raise ValueError(f"hands/{side}/mano_beta 必须是有限 (10,)")
        hands[side] = {
            "t": t_hand,
            "nodes": nodes,
            "mano_beta": beta,
        }

    # 命名物体刚体(新 schema 核心数据):名字 -> 帧列表
    obj_frames: dict[str, list[tuple[int, np.ndarray, np.ndarray, bool]]] = {}
    if "objects" in f:
        for name, g in f["objects"].items():
            pos_key = "object_position" if "object_position" in g else "position"
            quat_key = ("object_quaternion_xyzw"
                        if "object_quaternion_xyzw" in g else "quaternion_xyzw")
            obj_frames[name] = [
                (int(t), np.asarray(p, dtype=np.float64),
                 np.asarray(q, dtype=np.float64), bool(v))
                for t, p, q, v in zip(
                    g["t_ubuntu_ns"][:], g[pos_key][:],
                    g[quat_key][:], g["tracking_valid"][:])
            ]

    return {
        "t_mocap": t_mocap,
        "rb_frames": rb_frames,
        "rb_ids": rb_ids,
        "mk_frames": mk_frames,
        "hands": hands,
        "obj_frames": obj_frames,
    }


@dataclass
class SceneNodes:
    """场景节点句柄集合;build_scene_nodes 创建,apply_frame 逐帧更新。"""

    rigid_frames: dict[int, viser.FrameHandle] = field(default_factory=dict)
    rigid_labels: dict[int, viser.LabelHandle] = field(default_factory=dict)
    obj_frames: dict[str, viser.FrameHandle] = field(default_factory=dict)
    obj_labels: dict[str, viser.LabelHandle] = field(default_factory=dict)
    obj_meshes: dict[str, viser.MeshHandle] = field(default_factory=dict)
    marker_pc: viser.PointCloudHandle | None = None
    hand_pc: dict[str, viser.PointCloudHandle] = field(default_factory=dict)
    hand_ls: dict[str, viser.LineSegmentsHandle] = field(default_factory=dict)
    handles: list[Any] = field(default_factory=list)   # 全部节点,便于整体移除


def build_scene_nodes(scene: viser.ViserScene, data: dict) -> SceneNodes:
    """按数据内容创建全部场景节点(文件切换时整体重建)。"""
    nodes = SceneNodes()

    grid = scene.add_grid(
        "/grid", width=8.0, height=8.0, cell_size=0.1, plane="xz",
        plane_color=(235, 240, 250), plane_opacity=0.15,
    )
    nodes.handles.append(grid)
    # 桌面:半透明平面(TableSpec 默认值,相对 /world 的 y=0 平面,实际 y=1)
    x0, x1 = -0.72, 0.72
    z0, z1 = -0.45, 0.45
    verts = np.asarray([
        [x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    table = scene.add_mesh_simple(
        "/world/table", vertices=verts, faces=faces,
        color=(100, 149, 237), opacity=0.55, side="double",
    )
    nodes.handles.append(table)

    # 刚体:每个出现过的 ID 一个坐标轴 + 标签
    for rid in sorted(data["rb_ids"]):
        label = RIGID_LABELS.get(rid, f"rigid:{rid}")
        fh = scene.add_frame(f"/world/rigid/{rid}", axes_length=0.15, axes_radius=0.008)
        lb = scene.add_label(f"/world/rigid/{rid}/label", label, position=(0, 0.05, 0))
        nodes.rigid_frames[rid] = fh
        nodes.rigid_labels[rid] = lb
        nodes.handles.extend((fh, lb))

    # 命名物体：坐标轴和网格直接使用 HDF5 位姿；坐标预处理由独立
    # object-offset 程序在派生文件中完成，viewer 不应用任何外参。
    for name in data["obj_frames"]:
        fh = scene.add_frame(
            f"/world/object/{name}", axes_length=0.02, axes_radius=0.0008)
        lb = scene.add_label(
            f"/world/object/{name}/label", name, position=(0, 0.025, 0))
        loaded = load_object_mesh(name)
        if loaded is not None:
            _path, vertices, faces = loaded
            mesh = scene.add_mesh_simple(
                f"/world/object/{name}/mesh",
                vertices=vertices,
                faces=faces,
                color=OBJECT_MESH_COLOR,
                opacity=1.0,
                side="double",
                material="standard",
            )
            nodes.obj_meshes[name] = mesh
            nodes.handles.append(mesh)
        nodes.obj_frames[name] = fh
        nodes.obj_labels[name] = lb
        nodes.handles.extend((fh, lb))

    # markers 点云
    marker_pc = scene.add_point_cloud(
        "/world/markers", points=np.zeros((0, 3), np.float32),
        colors=np.zeros((0, 3), np.uint8), point_size=0.01,
        point_shape="circle", precision="float32", point_shading="gradient",
    )
    nodes.marker_pc = marker_pc
    nodes.handles.append(marker_pc)

    # 双手骨架
    for side in ("left", "right"):
        n_nodes = data["hands"][side]["nodes"].shape[1]
        pc = scene.add_point_cloud(
            f"/world/hand/{side}/points",
            points=np.zeros((n_nodes, 3), np.float32),
            colors=MANO_PALETTE[:n_nodes],
            point_size=0.006, point_shape="circle", precision="float32",
            point_shading="gradient",
        )
        seg0 = np.zeros((len(HAND_EDGES), 2, 3), np.float32)
        per = np.asarray([CHAIN_COLORS.get(6, DEFAULT_COLOR)] * len(HAND_EDGES), np.uint8)
        ls = scene.add_line_segments(
            f"/world/hand/{side}/bones", points=seg0,
            colors=np.repeat(per[:, None, :], 2, axis=1), line_width=2.0,
        )
        nodes.hand_pc[side] = pc
        nodes.hand_ls[side] = ls
        nodes.handles.extend((pc, ls))

    return nodes


def apply_frame(nodes: SceneNodes, data: dict, t_ns: float) -> dict:
    """把数据推进到 t_ns 对应帧并更新节点;返回统计信息。"""
    i = nearest_idx(data["t_mocap"], t_ns)
    t_cur = data["t_mocap"][i]

    # 刚体(新 schema 无 mocap 时跳过)
    if data["rb_frames"]:
        frame = data["rb_frames"][i]
        for rid, fh in nodes.rigid_frames.items():
            if rid in frame:
                pos, quat, _valid = frame[rid]
                fh.position = pos
                fh.wxyz = quat_xyzw_to_wxyz(quat)
                fh.visible = True
                nodes.rigid_labels[rid].position = (pos[0], pos[1] + 0.05, pos[2])
                nodes.rigid_labels[rid].visible = True
            else:
                fh.visible = False
                nodes.rigid_labels[rid].visible = False

    # 命名物体刚体:按各自时间戳最近邻
    for name, frames in data["obj_frames"].items():
        if not frames:
            continue
        fh = nodes.obj_frames[name]
        lb = nodes.obj_labels[name]
        t_arr = np.asarray([fr[0] for fr in frames], dtype=np.int64)
        k = int(np.searchsorted(t_arr, t_ns))
        k = min(max(k, 0), len(frames) - 1)
        while k > 0 and t_arr[k] > t_ns:
            k -= 1
        t_k, pos, quat, valid = frames[k]
        mesh = nodes.obj_meshes.get(name)
        if valid:
            fh.position = pos
            fh.wxyz = quat_xyzw_to_wxyz(quat)
            fh.visible = True
            lb.position = (0.0, 0.025, 0.0)
            lb.visible = True
            if mesh is not None:
                mesh.visible = True
        else:
            fh.visible = False
            lb.visible = False
            if mesh is not None:
                mesh.visible = False

    # markers
    pts, colors = data["mk_frames"][i]
    nodes.marker_pc.points = pts
    nodes.marker_pc.colors = colors

    # 双手
    for side, h in data["hands"].items():
        j = nearest_idx(h["t"], t_ns)
        frame_nodes = h["nodes"][j]
        nodes.hand_pc[side].points = frame_nodes
        seg = np.asarray(
            [[frame_nodes[c], frame_nodes[p]] for c, p in HAND_EDGES], np.float32
        )
        nodes.hand_ls[side].points = seg

    return {"t": t_cur, "i": i, "n_mk": pts.shape[0]}


def probe_h5(path: Path) -> dict | None:
    """预扫描单个 H5 文件信息(仅读头部,安全防护同 inspect);异常返回 None。"""
    try:
        with h5py.File(path, "r") as f:
            reject_external_links(f)
            if "mocap" in f:
                g = f["mocap"]
                t = (g["t_aligned_ubuntu_ns"][:]
                     if "t_aligned_ubuntu_ns" in g else g["t_ubuntu_ns"][:])
            elif "objects" in f and f["objects"]:
                t = next(iter(f["objects"].values()))["t_ubuntu_ns"][:]
            else:
                t = f["hands"]["left"]["t_ubuntu_ns"][:]
            dur = (t[-1] - t[0]) / 1e9
            return {"name": path.name, "path": str(path),
                    "dur": dur, "n_frames": len(t)}
    except Exception:
        return None
