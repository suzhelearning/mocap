"""HDF5 动捕录制 → data-viewer 的适配器。

把日期文件夹下的 .h5 录制索引为样本;build_viser 时把 HDF5 内容离线圈成
viser recording(.viser,缓存到 cache_dir),由 viser client 本地回放。
场景构建/配色/数据提取与 viz_hdf5.py 共用 acquisition.viser_core,保证
实时服务与离线回放视觉一致。

用法:
    data-viewer --root /home/current/data/20260812 \
        --adapter acquisition/scripts/h5_viewer_adapter.py --port 8082
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _path in (_SRC, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from acquisition.viser_core import (  # noqa: E402
    apply_frame,
    build_scene_nodes,
    extract_hdf5,
    nearest_idx,
    probe_h5,
    reject_external_links,
)
from data_viewer.contracts import (  # noqa: E402
    ProjectIndex,
    SampleDetail,
    SampleRecord,
    ViewerAdapter,
    ViewerConfig,
    ViserPlayback,
)
from viser import ViserServer  # noqa: E402

# 录制时间轴步长(帧):原始 ~100Hz 全量写入文件过大,隔帧采样 ~50Hz 已足够流畅
RECORD_STEP = 2


class H5ViewerAdapter(ViewerAdapter):
    """data-viewer 的 HDF5 录制文件夹适配器。"""

    def __init__(self, root: str | Path, config: ViewerConfig | None = None):
        self.root = Path(root).expanduser().resolve()
        self.config = config or ViewerConfig()
        self._index: ProjectIndex | None = None
        self._samples_by_id: dict[str, SampleRecord] = {}

    # ---- contracts ----

    def index(self) -> ProjectIndex:
        if self._index is not None:
            return self._index
        samples: list[SampleRecord] = []
        for path in sorted(self.root.rglob("*.h5")):
            if not path.is_file() or self._is_hidden(path):
                continue
            info = probe_h5(path) or {}
            parent = path.parent.relative_to(self.root).as_posix()
            group = parent if parent != "." else "recordings"
            samples.append(SampleRecord(
                id=path.name,
                label=path.stem,
                path=path,
                group_key=group,
                group_label=group,
                variant_label=path.stem,
                summary=(f"{info.get('dur', 0.0):.1f}s · "
                         f"{info.get('n_frames', 0)} 帧"),
                facets={"dataset": group, "dur": info.get("dur"),
                        "n_frames": info.get("n_frames")},
            ))
        warnings = [] if samples else ["未找到 .h5 录制文件。"]
        self._index = ProjectIndex(
            title=self.config.title or f"HDF5 · {self.root.name}",
            root=self.root,
            samples=samples,
            warnings=warnings,
        )
        self._samples_by_id = {s.id: s for s in samples}
        return self._index

    def metadata(self, sample_id: str) -> SampleDetail:
        sample = self._sample_by_id(sample_id)
        meta: dict = {}
        info = probe_h5(sample.path)
        if info:
            meta["record"] = info
        try:
            with h5py.File(sample.path, "r") as f:
                reject_external_links(f)
                data = extract_hdf5(f)
                mano_sides = {}
                hands = f.get("hands")
                for side in ("left", "right"):
                    group = hands.get(side) if hands is not None else None
                    if group is None or "mano_beta" not in group:
                        mano_sides[side] = {"available": False}
                        continue
                    beta = np.asarray(group["mano_beta"][:])
                    mano_sides[side] = {
                        "available": True,
                        "shape": list(beta.shape),
                        "beta": beta.reshape(-1).astype(float).tolist(),
                    }
            meta["scene"] = {
                "rigid_body_ids": sorted(data["rb_ids"]),
                "objects": sorted(data["obj_frames"].keys()),
                "hands": {
                    side: {"frames": len(h["t"]),
                           "nodes": h["nodes"].shape[1],
                           "points_per_node": h["nodes"].shape[2]}
                    for side, h in data["hands"].items()
                },
                "markers_first_frame": data["mk_frames"][0][0].shape[0],
                "duration_s": round((data["t_mocap"][-1] - data["t_mocap"][0]) / 1e9, 3),
                "mocap_frames": len(data["t_mocap"]),
            }
            meta["mano"] = {
                "available": any(item["available"] for item in mano_sides.values()),
                "sides": mano_sides,
            }
        except Exception as exc:
            meta["warning"] = f"{type(exc).__name__}: {exc}"
        return SampleDetail(sample=sample, metadata=meta)

    def build_viser(self, sample_id: str, cache_dir: Path) -> ViserPlayback:
        sample = self._sample_by_id(sample_id)
        if not sample.path.is_file():
            raise FileNotFoundError(sample.path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        recording = cache_dir / (sample.path.stem + ".viser")
        if not (recording.is_file()
                and recording.stat().st_mtime >= sample.path.stat().st_mtime):
            recording.write_bytes(_record_h5(sample.path))
        return ViserPlayback(recording_path=recording, label=sample.label,
                             warnings=[] if recording.is_file() else ["录制失败"])
    def build_viser_with_mode(
        self, sample_id: str, cache_dir: Path, mode: str,
    ) -> ViserPlayback:
        """按前端表示模式生成 recording;MANO 模式包含 beta 驱动网格。"""
        if mode != "mano":
            return self.build_viser(sample_id, cache_dir)
        sample = self._sample_by_id(sample_id)
        if not sample.path.is_file():
            raise FileNotFoundError(sample.path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        recording = cache_dir / (sample.path.stem + ".mano.viser")
        if not (recording.is_file()
                and recording.stat().st_mtime >= sample.path.stat().st_mtime):
            recording.write_bytes(_record_h5(sample.path, mano=True))
        return ViserPlayback(
            recording_path=recording,
            label=f"{sample.label} · MANO",
            warnings=[] if recording.is_file() else ["MANO 录制失败"],
        )

    # ---- helpers ----

    def _sample_by_id(self, sample_id: str) -> SampleRecord:
        self.index()
        try:
            return self._samples_by_id[sample_id]
        except KeyError:
            raise KeyError(sample_id) from None

    def _is_hidden(self, path: Path) -> bool:
        rel_parts = path.relative_to(self.root).parts
        hidden = set(self.config.hidden_names)
        return any(part in hidden or part.endswith(".egg-info") for part in rel_parts)

def _load_mano_backend():
    """延迟加载 MANO，保证仅使用关键点模式时不读取模型资产。"""
    from mano_fit import fit_mesh, load_mano

    return fit_mesh, {side: load_mano(side) for side in ("left", "right")}


def _load_mano_betas(path: Path) -> dict[str, np.ndarray]:
    """读取 hands/<side>/mano_beta；缺失时使用 MANO 中性形状。"""
    betas: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        reject_external_links(f)
        hands = f.get("hands")
        for side in ("left", "right"):
            group = hands.get(side) if hands is not None else None
            if group is None or "mano_beta" not in group:
                betas[side] = np.zeros(10, dtype=np.float64)
                continue
            beta = np.asarray(group["mano_beta"][:], dtype=np.float64)
            if beta.shape != (10,):
                raise ValueError(
                    f"hands/{side}/mano_beta 必须是 (10,),实际为 {beta.shape}",
                )
            if not np.isfinite(beta).all():
                raise ValueError(f"hands/{side}/mano_beta 含非有限值")
            betas[side] = beta
    return betas


def _record_h5(path: Path, *, mano: bool = False) -> bytes:
    """把单个 H5 文件离线圈成 .viser recording 字节。"""
    with h5py.File(path, "r") as f:
        reject_external_links(f)
        data = extract_hdf5(f)

    fit_mesh = None
    mano_layers: dict[str, object] = {}
    mano_betas: dict[str, np.ndarray] = {}
    if mano:
        fit_mesh, mano_layers = _load_mano_backend()
        mano_betas = _load_mano_betas(path)

    server = ViserServer(host="127.0.0.1", port=0)   # 随机空闲端口,无需客户端
    try:
        server.scene.set_up_direction((0.0, 1.0, 0.0))
        server.initial_camera.position = (0.038, 4.176, -5.413)
        server.initial_camera.look_at = (0.0, 1.0, 0.0)
        server.initial_camera.up = (0.0, -1.0, 0.0)
        server.initial_camera.fov = 50.0
        # 与 viz-h5 相同:数据/桌面整体抬高 1m,grid 仍在 y=0
        server.scene.add_frame("/world", position=(0.0, 1.0, 0.0), show_axes=False)

        serializer = server.get_scene_serializer()
        nodes = build_scene_nodes(server.scene, data)
        mano_meshes: dict[str, object] = {}
        if mano:
            for side in ("left", "right"):
                hand = data["hands"][side]
                if hand["nodes"].shape[1] != 21:
                    continue
                index = nearest_idx(hand["t"], int(data["t_mocap"][0]))
                verts, _joints, _params = fit_mesh(
                    mano_layers[side],
                    hand["nodes"][index],
                    beta=mano_betas[side],
                    iters=0,
                )
                mano_meshes[side] = server.scene.add_mesh_simple(
                    f"/world/mano/{side}",
                    vertices=verts,
                    faces=mano_layers[side].faces,
                    color=(224, 154, 154) if side == "left" else (154, 178, 224),
                    opacity=0.62,
                    side="double",
                    material="standard",
                )

        t_mocap = data["t_mocap"]
        prev_ns: int | None = None
        for i in range(0, len(t_mocap), RECORD_STEP):
            t_ns = int(t_mocap[i])
            apply_frame(nodes, data, float(t_ns))
            if mano:
                for side, mesh in mano_meshes.items():
                    hand = data["hands"][side]
                    index = nearest_idx(hand["t"], t_ns)
                    verts, _joints, _params = fit_mesh(
                        mano_layers[side],
                        hand["nodes"][index],
                        beta=mano_betas[side],
                        iters=0,
                    )
                    mesh.vertices = verts
            if prev_ns is not None:
                serializer.insert_sleep((t_ns - prev_ns) / 1e9)
            prev_ns = t_ns
        return serializer.serialize()
    finally:
        server.stop()


def create_adapter(root: Path, config: ViewerConfig) -> ViewerAdapter:
    return H5ViewerAdapter(root, config)
