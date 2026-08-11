#!/usr/bin/env python3
"""viz_hdf5.py — 录制文件夹离线 3D 回放(自定义网页布局)。

启动时指定某一天的录制文件夹,网页布局为:
  - 左侧栏:文件夹内所有 .h5 录制文件列表(点击切换)
  - 右侧:viser 3D 窗口(动捕刚体、markers、双手骨架、物体)
  - 窗口下方:播放/暂停按键、倍速、进度轨迹 bar

多流(动捕 ~97Hz / 手部 ~80Hz)按各自 t_ubuntu_ns 最近邻对齐到同一播放时间轴。
配色与拓扑约定与 acquisition/live_view.py 保持一致。

端口:控制页 --port(默认 8082),viser 场景自动 +1(8083,iframe 内嵌)。

安全防护:拒绝含外部链接的 HDF5(与 inspect/replay 一致)。

用法:
  pixi run viz-h5 -- /home/current/data/20260811 [--port 8082] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np
import viser
from viser import ViserServer

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

_IFRAME_PORT = 0   # 占位:main 里按控制端口 +1 设置


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


def _nearest_idx(t: np.ndarray, t_ns: float) -> int:
    """在单调时间戳数组上取 t_ns 的最近邻下标(夹取到边界)。"""
    i = int(np.searchsorted(t, t_ns, side="right")) - 1
    return min(max(i, 0), t.size - 1)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[[3, 0, 1, 2]]


def _marker_colors(kinds, occluded) -> np.ndarray:
    colors = np.zeros((len(kinds), 3), dtype=np.uint8)
    for i, (kind, occ) in enumerate(zip(kinds, occluded)):
        if bool(occ):
            colors[i] = OCCLUDED_COLOR
            continue
        colors[i] = _KIND_COLORS.get(int(kind), MARKER_COLORS["unknown"])
    return colors


@dataclass
class _Playback:
    t_ns: float = 0.0
    playing: bool = True
    speed: float = 1.0


class VizScene:
    """viser 3D 回放场景;动画由服务端线程驱动,支持运行时切换文件。"""

    def __init__(self, port: int) -> None:
        self.playback = _Playback()
        self.t0 = 0
        self.t1 = 0
        self.dur_s = 0.0
        self.path: Path | None = None
        self.error: str | None = None

        self.server = ViserServer(host="127.0.0.1", port=port)
        self._scene_ready = False
        self._anim_thread = threading.Thread(
            target=self._animate, name="viz-h5-anim", daemon=True,
        )
        self._anim_thread.start()

    # ---- 数据加载(全部入内存;22s 录段仅数 MB) ----
    def load_file(self, path: Path) -> str | None:
        """加载/切换 H5 文件;返回 None 成功,否则错误信息。"""
        try:
            with h5py.File(path, "r") as f:
                reject_external_links(f)
                data = self._extract(f)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return self.error

        self._data = data
        self.path = path
        self.error = None
        self.t0 = int(data["t_mocap"][0])
        self.t1 = int(data["t_mocap"][-1])
        self.dur_s = (self.t1 - self.t0) / 1e9
        self.playback.t_ns = float(self.t0)
        self._rebuild_scene()
        return None

    def _extract(self, f: h5py.File) -> dict:
        mocap = f["mocap"]
        t_mocap = mocap["t_ubuntu_ns"][:].astype(np.int64)
        n = len(t_mocap)

        # 刚体:vlen 行(一维铺平)→ 每帧 {id: (pos, quat_xyzw, valid)}
        rb = mocap["rigid_bodies"]
        rb_frames: list[dict[int, tuple[np.ndarray, np.ndarray, bool]]] = []
        rb_ids: set[int] = set()
        for i in range(n):
            ids = np.asarray(rb["ids"][i])
            pos = np.asarray(rb["positions"][i], dtype=np.float64).reshape(-1, 3)
            quat = np.asarray(rb["quaternions_xyzw"][i], dtype=np.float64).reshape(-1, 4)
            valid = np.asarray(rb["tracking_valid"][i])
            frame = {}
            for j, rid in enumerate(ids):
                rid = int(rid)
                frame[rid] = (pos[j], quat[j], bool(valid[j]))
                rb_ids.add(rid)
            rb_frames.append(frame)

        # markers:vlen 行 → 每帧 (points, colors)
        mk = mocap["markers"]
        mk_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n):
            pts = np.asarray(mk["positions"][i], dtype=np.float32).reshape(-1, 3)
            colors = _marker_colors(mk["id_kinds"][i], mk["occluded"][i])
            mk_frames.append((pts, colors))

        # 双手骨架(按自身时间戳对齐)
        hands: dict[str, dict] = {}
        for side in ("left", "right"):
            g = f["hands"][side]
            hands[side] = {
                "t": g["t_ubuntu_ns"][:].astype(np.int64),
                "nodes": g["nodes_global"][:],
            }

        return {
            "t_mocap": t_mocap,
            "rb_frames": rb_frames,
            "rb_ids": rb_ids,
            "mk_frames": mk_frames,
            "hands": hands,
        }

    # ---- 场景节点(文件切换时整体重建) ----
    def _rebuild_scene(self) -> None:
        if self._scene_ready:
            for h in self._scene_handles:
                h.remove()
        sc = self.server.scene
        data = self._data
        handles: list = []

        grid = sc.add_grid(
            "/grid", width=2.0, height=2.0, cell_size=0.1, plane="xz",
        )
        handles.append(grid)
        # 桌面区域参考框(TableSpec 默认值,y=0 平面)
        x0, x1 = -0.72, 0.72
        z0, z1 = -0.45, 0.45
        seg = np.asarray([
            [[x0, 0, z0], [x1, 0, z0]], [[x1, 0, z0], [x1, 0, z1]],
            [[x1, 0, z1], [x0, 0, z1]], [[x0, 0, z1], [x0, 0, z0]],
        ], dtype=np.float32)
        table = sc.add_line_segments(
            "/table", points=seg,
            colors=np.full((4, 2, 3), (140, 140, 140), np.uint8), line_width=2.0,
        )
        handles.append(table)

        # 刚体:每个出现过的 ID 一个坐标轴 + 标签
        rigid_frames: dict[int, viser.FrameHandle] = {}
        rigid_labels: dict[int, viser.LabelHandle] = {}
        for rid in sorted(data["rb_ids"]):
            label = RIGID_LABELS.get(rid, f"rigid:{rid}")
            fh = sc.add_frame(f"/rigid/{rid}", axes_length=0.15, axes_radius=0.008)
            lb = sc.add_label(f"/rigid/{rid}/label", label, position=(0, 0.05, 0))
            rigid_frames[rid] = fh
            rigid_labels[rid] = lb
            handles.extend((fh, lb))

        # markers 点云
        marker_pc = sc.add_point_cloud(
            "/markers", points=np.zeros((0, 3), np.float32),
            colors=np.zeros((0, 3), np.uint8), point_size=0.01,
            point_shape="circle", precision="float32", point_shading="gradient",
        )
        handles.append(marker_pc)

        # 双手骨架
        hand_pc: dict[str, viser.PointCloudHandle] = {}
        hand_ls: dict[str, viser.LineSegmentsHandle] = {}
        for side in ("left", "right"):
            pc = sc.add_point_cloud(
                f"/hand/{side}/points",
                points=np.zeros((25, 3), np.float32), colors=MANO_PALETTE,
                point_size=0.006, point_shape="circle", precision="float32",
                point_shading="gradient",
            )
            seg0 = np.zeros((len(HAND_EDGES), 2, 3), np.float32)
            per = np.asarray([CHAIN_COLORS.get(6, DEFAULT_COLOR)] * len(HAND_EDGES), np.uint8)
            ls = sc.add_line_segments(
                f"/hand/{side}/bones", points=seg0,
                colors=np.repeat(per[:, None, :], 2, axis=1), line_width=2.0,
            )
            hand_pc[side] = pc
            hand_ls[side] = ls
            handles.extend((pc, ls))

        self._rigid_frames = rigid_frames
        self._rigid_labels = rigid_labels
        self._marker_pc = marker_pc
        self._hand_pc = hand_pc
        self._hand_ls = hand_ls
        self._scene_handles = handles
        self._scene_ready = True
        self._apply_frame()

    # ---- 每帧更新 ----
    def _apply_frame(self) -> None:
        pb = self.playback
        sc = self.server.scene
        data = self._data

        i = _nearest_idx(data["t_mocap"], pb.t_ns)
        t_cur = data["t_mocap"][i]

        # 刚体
        frame = data["rb_frames"][i]
        for rid, fh in self._rigid_frames.items():
            if rid in frame:
                pos, quat, _valid = frame[rid]
                fh.position = pos
                fh.wxyz = _quat_xyzw_to_wxyz(quat)
                fh.visible = True
                self._rigid_labels[rid].position = (pos[0], pos[1] + 0.05, pos[2])
                self._rigid_labels[rid].visible = True
            else:
                fh.visible = False
                self._rigid_labels[rid].visible = False

        # markers
        pts, colors = data["mk_frames"][i]
        self._marker_pc.points = pts
        self._marker_pc.colors = colors

        # 双手
        for side, h in data["hands"].items():
            j = _nearest_idx(h["t"], pb.t_ns)
            nodes = h["nodes"][j]
            self._hand_pc[side].points = nodes
            seg = np.asarray(
                [[nodes[c], nodes[p]] for c, p in HAND_EDGES], np.float32
            )
            self._hand_ls[side].points = seg

        n_mk = pts.shape[0]
        sc_elapsed = (t_cur - self.t0) / 1e9
        return {"t": sc_elapsed, "dur": self.dur_s, "i": i, "n_mk": n_mk}

    def _animate(self) -> None:
        """服务端动画线程:推进播放时间轴并更新场景(约 60fps)。"""
        last = time.perf_counter()
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            pb = self.playback
            if pb.playing and self._scene_ready:
                pb.t_ns += pb.speed * dt * 1e9
                if pb.t_ns > self.t1:
                    pb.t_ns = float(self.t0)      # 循环回放
            time.sleep(0.016)


# ---- 网页(左侧文件列表 + 右侧 viser iframe + 底部播放条) ----
_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>动捕录制回放</title>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif; }
  body { display: flex; flex-direction: column; }
  #main { display: flex; flex: 1; min-height: 0; }
  #sidebar { width: 280px; min-width: 280px; border-right: 1px solid #e2e2e2;
             overflow-y: auto; padding: 10px; background: #fafafa; box-sizing: border-box; }
  #sidebar h3 { margin: 4px 2px 10px; font-size: 13px; color: #444; font-weight: 600; }
  .file { padding: 7px 10px; margin-bottom: 3px; border-radius: 6px; cursor: pointer;
          font-size: 13px; color: #222; word-break: break-all; border: 1px solid transparent; }
  .file:hover { background: #eef2ff; }
  .file.active { background: #dbeafe; border-color: #93c5fd; font-weight: 500; }
  .file .meta { font-size: 11px; color: #888; margin-top: 2px; }
  #viewer { flex: 1; min-width: 0; position: relative; }
  iframe { width: 100%; height: 100%; border: 0; display: block; }
  #controls { display: flex; align-items: center; gap: 12px; padding: 8px 14px;
              border-top: 1px solid #e2e2e2; background: #f8f8f8; user-select: none; }
  #playbtn { width: 76px; padding: 7px 0; font-size: 14px; border-radius: 6px;
             border: 1px solid #c4c4c4; background: #fff; cursor: pointer; }
  #playbtn:hover { background: #eef2ff; }
  #seekbar { flex: 1; }
  #timeinfo { min-width: 220px; text-align: right; font-family: ui-monospace, monospace;
              font-size: 12px; color: #333; }
  #speedwrap { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #444; }
  #speed { width: 90px; }
  #status { font-size: 12px; color: #b91c1c; }
</style>
</head>
<body>
<div id="main">
  <div id="sidebar"><h3>录制文件</h3><div id="filelist"></div></div>
  <div id="viewer"><iframe id="viserframe" src=""></iframe></div>
</div>
<div id="controls">
  <button id="playbtn">▶ 播放</button>
  <div id="speedwrap"><span>倍速</span><input id="speed" type="range" min="10" max="500" value="100" step="10"><span id="speedval">1.0x</span></div>
  <input id="seekbar" type="range" min="0" max="1000" value="0" step="1">
  <span id="timeinfo">-- / --</span>
  <span id="status"></span>
</div>
<script>
const $ = (s) => document.querySelector(s);
let dragging = false;
let curFile = null;

async function api(path, body) {
  const opt = body !== undefined
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  return r.json();
}

async function refreshFiles() {
  const d = await api('/api/files');
  const list = $('#filelist');
  list.innerHTML = '';
  for (const f of d.files) {
    const div = document.createElement('div');
    div.className = 'file' + (f.name === d.current ? ' active' : '');
    div.innerHTML = `<div>${f.name}</div><div class="meta">${f.dur.toFixed(1)}s · ${f.n_frames} 帧</div>`;
    div.onclick = () => loadFile(f.name);
    list.appendChild(div);
  }
  curFile = d.current;
  if (d.current) $('#timeinfo').textContent = '0.00s / ' + d.dur.toFixed(2) + 's';
}

async function loadFile(name) {
  const d = await api('/api/load', { file: name });
  if (d.error) { $('#status').textContent = '加载失败: ' + d.error; return; }
  $('#status').textContent = '';
  await refreshFiles();
}

async function togglePlay() {
  const d = await api('/api/play');
  $('#playbtn').textContent = d.playing ? '⏸ 暂停' : '▶ 播放';
}

async function pollState() {
  if (dragging) return;
  const d = await api('/api/state');
  if (d.dur > 0) {
    $('#seekbar').value = Math.round(d.t / d.dur * 1000);
    $('#timeinfo').textContent = d.t.toFixed(2) + 's / ' + d.dur.toFixed(2) + 's';
    $('#playbtn').textContent = d.playing ? '⏸ 暂停' : '▶ 播放';
  }
}

$('#playbtn').onclick = togglePlay;
$('#speed').oninput = async (e) => {
  const s = e.target.value / 100;
  $('#speedval').textContent = s.toFixed(1) + 'x';
  await api('/api/speed', { s });
};
$('#seekbar').addEventListener('input', () => { dragging = true; });
$('#seekbar').addEventListener('change', async (e) => {
  dragging = false;
  await api('/api/seek', { frac: e.target.value / 1000 });
});
$('#viserframe').src = 'http://127.0.0.1:__IFRAME_PORT__/';

refreshFiles();
setInterval(pollState, 200);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    scene: VizScene = None          # 由 serve() 注入
    files: list[dict] = []          # [{name, path, dur, n_frames}]

    def log_message(self, fmt, *args):  # 安静日志
        pass

    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        scene = self.scene
        if self.path in ("/", "/index.html"):
            html = _PAGE_HTML.replace("__IFRAME_PORT__", str(_IFRAME_PORT))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/files":
            self._send_json({"files": self.files,
                             "current": scene.path.name if scene.path else None,
                             "dur": scene.dur_s})
        elif self.path == "/api/state":
            t = 0.0
            if scene._scene_ready:
                i = _nearest_idx(scene._data["t_mocap"], scene.playback.t_ns)
                t = (scene._data["t_mocap"][i] - scene.t0) / 1e9
            self._send_json({"t": t, "dur": scene.dur_s,
                             "playing": scene.playback.playing,
                             "speed": scene.playback.speed})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        scene = self.scene
        body = self._read_body()
        if self.path == "/api/load":
            name = body.get("file", "")
            for f in self.files:
                if f["name"] == name:
                    err = scene.load_file(Path(f["path"]))
                    if err:
                        self._send_json({"error": err})
                    else:
                        self._send_json({"ok": True, "name": name})
                    return
            self._send_json({"error": f"文件不存在: {name}"}, 404)
        elif self.path == "/api/play":
            scene.playback.playing = not scene.playback.playing
            self._send_json({"playing": scene.playback.playing})
        elif self.path == "/api/seek":
            frac = float(body.get("frac", 0.0))
            scene.playback.t_ns = scene.t0 + frac * (scene.t1 - scene.t0)
            self._send_json({"ok": True})
        elif self.path == "/api/speed":
            scene.playback.speed = max(0.1, min(5.0, float(body.get("s", 1.0))))
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)


def main() -> int:
    ap = argparse.ArgumentParser(description="录制文件夹离线 3D 回放")
    ap.add_argument("dir", type=str, help="日期文件夹,如 /home/current/data/20260811")
    ap.add_argument("--port", type=int, default=8082, help="控制页端口(viser 自动 +1)")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.is_dir():
        print(f"[viz-h5] 目录不存在: {data_dir}", file=sys.stderr)
        return 2
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        print(f"[viz-h5] 目录中没有 .h5 文件: {data_dir}", file=sys.stderr)
        return 2

    global _IFRAME_PORT
    _IFRAME_PORT = args.port + 1

    # 预扫描文件信息(仅读头部,安全防护同 inspect)
    files: list[dict] = []
    for p in h5_files:
        try:
            with h5py.File(p, "r") as f:
                reject_external_links(f)
                t = f["mocap/t_ubuntu_ns"][:]
                dur = (t[-1] - t[0]) / 1e9
                files.append({"name": p.name, "path": str(p),
                              "dur": dur, "n_frames": len(t)})
        except Exception as exc:
            print(f"[viz-h5] 跳过 {p.name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    scene = VizScene(_IFRAME_PORT)
    scene.playback.speed = args.speed
    err = scene.load_file(Path(files[0]["path"]))
    if err:
        print(f"[viz-h5] 加载 {files[0]['name']} 失败: {err}", file=sys.stderr)
        return 1

    _Handler.scene = scene
    _Handler.files = files
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"[viz-h5] 打开 http://127.0.0.1:{args.port} "
          f"(左栏 {len(files)} 个文件,viser iframe :{_IFRAME_PORT})")
    print(f"[viz-h5] 当前: {scene.path.name}, {scene.dur_s:.2f}s, "
          f"刚体 ID {sorted(scene._data['rb_ids'])}, Ctrl-C 退出")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        scene.server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
