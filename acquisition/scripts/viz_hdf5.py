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
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import h5py
import viser
from viser import ViserServer

from acquisition.viser_core import (
    apply_frame,
    build_scene_nodes,
    extract_hdf5,
    nearest_idx,
    probe_h5,
    reject_external_links,
)

_IFRAME_PORT = 0   # 占位:main 里按控制端口 +1 设置


@dataclass
class _Playback:
    t_ns: float = 0.0
    playing: bool = True
    speed: float = 1.0

class VizScene:
    """viser 3D 回放场景;动画由服务端线程驱动,支持运行时切换文件。"""

    def __init__(self, port: int, mano_enabled: bool = False) -> None:
        self.playback = _Playback()
        self.t0 = 0
        self.t1 = 0
        self.dur_s = 0.0
        self.path: Path | None = None
        self.error: str | None = None
        self.mano_enabled = bool(mano_enabled)
        self.mano_error: str | None = None
        self._mano_layers: dict[str, object] = {}
        self._mano_betas: dict[str, np.ndarray] = {}   # side -> beta (10,) 或 None
        self._mano_fit_mesh = None
        self._mano_handles: dict[str, object] = {}
        self._mano_last_idx: dict[str, int] = {}
        self._mano_last_params: dict[str, np.ndarray] = {}
        self._mano_generation = 0
        self._mano_lock = threading.Condition()
        self._mano_pending: dict[str, tuple[int, int, np.ndarray, np.ndarray | None]] = {}
        self._mano_results: dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}

        self.server = ViserServer(host="127.0.0.1", port=port)
        self.server.gui.configure_theme(dark_mode=True)   # 黑色主题
        # Motive 数据是 y-up；通过 viser 官方 up-direction 机制保持 y-up。
        # 这样根节点负责统一变换，数据、相机和灯光处在同一个坐标约定中；
        # 不再手动把 y-up 数据旋到 z-up，也不覆盖 viser 的根节点姿态。
        self.server.scene.set_up_direction((0.0, 1.0, 0.0))
        self.server.initial_camera.position = (0.038, 4.176, -5.413)
        self.server.initial_camera.look_at = (0.0, 1.0, 0.0)
        self.server.initial_camera.up = (0.0, -1.0, 0.0)
        self.server.initial_camera.fov = 50.0
        # 每次连接（含断线重连）强制回到同一 y-up 俯视姿态。
        self.server.on_client_connect(self._reset_camera)
        # 数据/桌面整体抬高 1m；原始 H5 坐标不修改，地面 grid 仍在 y=0。
        self.world_frame = self.server.scene.add_frame(
            "/world", position=(0.0, 1.0, 0.0), show_axes=False,
        )
        self._scene_ready = False
        self._anim_thread = threading.Thread(
            target=self._animate, name="viz-h5-anim", daemon=True,
        )
        self._mano_thread = threading.Thread(
            target=self._mano_worker, name="viz-h5-mano", daemon=True,
        )
        self._anim_thread.start()
        self._mano_thread.start()

    def _reset_camera(self, client: viser.ClientHandle) -> None:
        """连接回调:强制相机回到默认视角（y-up 世界,画面上下翻转）。"""
        client.camera.position = (0.038, 4.176, -5.413)
        client.camera.look_at = (0.0, 1.0, 0.0)
        client.camera.up_direction = (0.0, -1.0, 0.0)

    def _camera_state(self) -> dict | None:
        """取最近更新的客户端相机状态(viser 世界坐标),无客户端返回 None。"""
        best: tuple[viser.CameraHandle, float] | None = None
        for c in self.server.get_clients().values():
            cam = c.camera
            try:
                ts = cam.update_timestamp
            except AssertionError:
                continue  # 尚未收到该客户端相机消息
            if best is None or ts > best[1]:
                best = (cam, ts)
        if best is None:
            return None
        cam = best[0]
        return {
            "position": [round(float(v), 3) for v in cam.position],
            "wxyz": [round(float(v), 3) for v in cam.wxyz],
            "look_at": [round(float(v), 3) for v in cam.look_at],
            "up_direction": [round(float(v), 3) for v in cam.up_direction],
        }

    def _ensure_mano_backend(self) -> None:
        """按需加载 MANO 后端；普通骨架回放不依赖它。"""
        if self._mano_fit_mesh is not None or self.mano_error is not None:
            return
        try:
            from mano_fit import fit_mesh, load_mano

            self._mano_fit_mesh = fit_mesh
            self._mano_layers = {
                side: load_mano(side) for side in ("left", "right")
            }
        except Exception as exc:
            self.mano_error = f"{type(exc).__name__}: {exc}"

    def _clear_mano_work(self) -> None:
        with self._mano_lock:
            self._mano_generation += 1
            self._mano_pending.clear()
            self._mano_results.clear()
            self._mano_lock.notify_all()

    def _mano_worker(self) -> None:
        """后台拟合网格，避免数值优化阻塞骨架回放线程。"""
        while True:
            with self._mano_lock:
                while not self._mano_pending:
                    self._mano_lock.wait()
                side = next(iter(self._mano_pending))
                request = self._mano_pending.pop(side)
            generation, index, obs, init = request
            layer = self._mano_layers.get(side)
            if layer is None or self._mano_fit_mesh is None:
                continue
            try:
                verts, _joints, params = self._mano_fit_mesh(
                    layer, obs, beta=self._mano_betas.get(side), init=init, iters=3,
                )
            except Exception as exc:
                self.mano_error = f"MANO {side}: {type(exc).__name__}: {exc}"
                continue
            with self._mano_lock:
                if generation == self._mano_generation:
                    self._mano_results[side] = (
                        generation, index, verts, params,
                    )

    def _request_mano(self, side: str, index: int) -> None:
        if not self.mano_enabled or side not in self._mano_handles:
            return
        hand = self._data["hands"][side]
        generation = self._mano_generation
        with self._mano_lock:
            result = self._mano_results.get(side)
            pending = self._mano_pending.get(side)
            if result is not None and result[0] == generation and result[1] == index:
                return
            if pending is not None and pending[0] == generation and pending[1] == index:
                return
            init = self._mano_last_params.get(side)
            self._mano_pending[side] = (
                generation, index, np.asarray(hand["nodes"][index], dtype=np.float64).copy(),
                None if init is None else init.copy(),
            )
            self._mano_lock.notify()

    def _update_mano_mesh(self, t_ns: float) -> None:
        if not self._mano_handles:
            return
        with self._mano_lock:
            results = list(self._mano_results.items())
            self._mano_results.clear()
        for side, (_generation, index, verts, params) in results:
            if side not in self._mano_handles:
                continue
            self._mano_handles[side].vertices = verts
            self._mano_last_idx[side] = index
            self._mano_last_params[side] = params
        for side, handle in self._mano_handles.items():
            handle.visible = self.mano_enabled
            if not self.mano_enabled:
                continue
            index = nearest_idx(self._data["hands"][side]["t"], t_ns)
            if self._mano_last_idx.get(side) != index:
                self._request_mano(side, index)

    def set_mano_enabled(self, enabled: bool) -> None:
        self.mano_enabled = bool(enabled)
        if self._scene_ready:
            self._update_mano_mesh(self.playback.t_ns)

    def mano_state(self) -> dict:
        return {
            "available": bool(self._mano_handles),
            "enabled": bool(self.mano_enabled and self._mano_handles),
            "error": self.mano_error,
            "beta_from_h5": {side: bool(b is not None)
                             for side, b in self._mano_betas.items()},
        }

    # ---- 数据加载(全部入内存;22s 录段仅数 MB) ----
    def load_file(self, path: Path) -> str | None:
        """加载/切换 H5 文件;返回 None 成功,否则错误信息。"""
        try:
            with h5py.File(path, "r") as f:
                reject_external_links(f)
                data = extract_hdf5(f)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return self.error

        self._clear_mano_work()
        self._data = data
        self.path = path
        self.error = None
        self.mano_error = None
        self.t0 = int(data["t_mocap"][0])
        self.t1 = int(data["t_mocap"][-1])
        self.dur_s = (self.t1 - self.t0) / 1e9
        self.playback.t_ns = float(self.t0)
        self._rebuild_scene()
        return None

    
    # ---- 场景节点(文件切换时整体重建) ----
    def _rebuild_scene(self) -> None:
        if self._scene_ready:
            for h in self._scene_handles:
                h.remove()
        self._mano_handles.clear()
        self._mano_last_idx.clear()
        self._mano_last_params.clear()
        self._mano_betas.clear()
        self._ensure_mano_backend()
        # 读取 H5 中离线写入的 beta(mano_beta.py);缺失则用 β=0
        try:
            with h5py.File(self.path, "r") as f:
                reject_external_links(f)
                for side in ("left", "right"):
                    g = f.get("hands", {}).get(side)
                    if g is not None and "mano_beta" in g:
                        self._mano_betas[side] = np.asarray(
                            g["mano_beta"][:], dtype=np.float64)
        except Exception as exc:
            self.mano_error = f"读取 mano_beta 失败: {exc}"
        self._scene_nodes = build_scene_nodes(self.server.scene, self._data)
        self._scene_handles = self._scene_nodes.handles
        if self._mano_fit_mesh is not None:
            for side in ("left", "right"):
                hand = self._data["hands"][side]
                if hand["nodes"].shape[1] != 21:
                    continue
                try:
                    beta = self._mano_betas.get(side, np.zeros(10, dtype=np.float64))
                    verts, _joints, params = self._mano_fit_mesh(
                        self._mano_layers[side], hand["nodes"][0],
                        beta=beta, iters=0,
                    )
                    mesh = self.server.scene.add_mesh_simple(
                        f"/world/mano/{side}",
                        vertices=verts,
                        faces=self._mano_layers[side].faces,
                        color=(224, 154, 154) if side == "left" else (154, 178, 224),
                        opacity=0.62,
                        side="double",
                        material="standard",
                        visible=self.mano_enabled,
                    )
                    self._mano_handles[side] = mesh
                    self._mano_last_idx[side] = -1
                    self._mano_last_params[side] = params
                    self._scene_handles.append(mesh)
                except Exception as exc:
                    self.mano_error = (
                        f"MANO {side}: {type(exc).__name__}: {exc}"
                    )
        self._scene_ready = True
        self._apply_frame()

    # ---- 每帧更新 ----
    def _apply_frame(self) -> None:
        pb = self.playback
        data = self._data

        stats = apply_frame(self._scene_nodes, data, pb.t_ns)
        self._update_mano_mesh(pb.t_ns)
        i = stats["i"]
        sc_elapsed = (stats["t"] - self.t0) / 1e9
        return {"t": sc_elapsed, "dur": self.dur_s, "i": i, "n_mk": stats["n_mk"]}


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
                self._apply_frame()
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
  #camerapanel { position: absolute; top: 12px; right: 12px; width: 240px; z-index: 10;
                 background: rgba(255,255,255,0.94); border: 1px solid #d4d4d4;
                 border-radius: 8px; padding: 8px 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                 font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 11px;
                 color: #333; pointer-events: none; }
  #camerapanel .cp-title { font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
                           font-size: 12px; font-weight: 600; color: #111; margin-bottom: 5px; }
  #camerapanel .cp-row { display: flex; justify-content: space-between; gap: 8px; line-height: 1.55; }
  #camerapanel .cp-row span:last-child { white-space: nowrap; }
  #camerapanel .cp-k { color: #666; flex-shrink: 0; }
  #controls { display: flex; align-items: center; gap: 12px; padding: 8px 14px;
              border-top: 1px solid #e2e2e2; background: #f8f8f8; user-select: none; }
  #playbtn { width: 76px; padding: 7px 0; font-size: 14px; border-radius: 6px;
             border: 1px solid #c4c4c4; background: #fff; cursor: pointer; }
  #playbtn:hover { background: #eef2ff; }
  #resetbtn { padding: 7px 12px; font-size: 13px; border-radius: 6px;
              border: 1px solid #c4c4c4; background: #fff; cursor: pointer; }
  #resetbtn:hover { background: #eef2ff; }
  #seekbar { flex: 1; }
  #timeinfo { min-width: 220px; text-align: right; font-family: ui-monospace, monospace;
              font-size: 12px; color: #333; }
  #speedwrap { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #444; }
  #manowrap { display: flex; align-items: center; gap: 5px; font-size: 12px; color: #444; }
  #speed { width: 90px; }
  #status { font-size: 12px; color: #b91c1c; }
</style>
</head>
<body>
<div id="main">
  <div id="sidebar"><h3>录制文件</h3><div id="filelist"></div></div>
  <div id="viewer">
    <iframe id="viserframe" src=""></iframe>
    <div id="camerapanel">
      <div class="cp-title">相机视角</div>
      <div id="cp-body">连接中…</div>
    </div>
  </div>
</div>
<div id="controls">
  <button id="playbtn">▶ 播放</button>
  <div id="speedwrap"><span>倍速</span><input id="speed" type="range" min="10" max="500" value="100" step="10"><span id="speedval">1.0x</span></div>
  <label id="manowrap"><input id="manocheck" type="checkbox"> MANO网格</label>
  <input id="seekbar" type="range" min="0" max="1000" value="0" step="1">
  <span id="timeinfo">-- / --</span>
  <button id="resetbtn">重置视角</button>
  <span id="status"></span>
</div>
<script>
const $ = (s) => document.querySelector(s);
let dragging = false;
let curFile = null;

async function api(path, body) {
  const opt = { method: body !== undefined ? 'POST' : 'GET' };
  if (body !== undefined) {
    opt.headers = { 'Content-Type': 'application/json' };
    opt.body = JSON.stringify(body);
  }
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
  const d = await api('/api/play', {});
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
  const mano = $('#manocheck');
  mano.disabled = !d.mano.available;
  mano.checked = d.mano.enabled;
  if (d.mano.error && !d.mano.available) {
    $('#status').textContent = 'MANO不可用: ' + d.mano.error;
  }
}

async function toggleMano() {
  const d = await api('/api/mano', { enabled: $('#manocheck').checked });
  if (d.error) {
    $('#status').textContent = d.error;
    $('#manocheck').checked = false;
  }
}

const camFmt = (v) => '(' + v.map(x => x.toFixed(3)).join(', ') + ')';

$('#manocheck').onchange = toggleMano;
async function pollCamera() {
  const d = await api('/api/camera');
  const body = $('#cp-body');
  if (!d.connected) {
    body.innerHTML = '<span style="color:#b91c1c">未连接客户端</span>';
    return;
  }
  body.innerHTML =
    '<div class="cp-row"><span class="cp-k">位置</span><span>' + camFmt(d.position) + '</span></div>' +
    '<div class="cp-row"><span class="cp-k">注视点</span><span>' + camFmt(d.look_at) + '</span></div>' +
    '<div class="cp-row"><span class="cp-k">朝向wxyz</span><span>' + camFmt(d.wxyz) + '</span></div>';
}

$('#playbtn').onclick = togglePlay;
$('#resetbtn').onclick = async () => { await api('/api/reset-camera', {}); };
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
setInterval(pollCamera, 250);
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
                i = nearest_idx(scene._data["t_mocap"], scene.playback.t_ns)
                t = (scene._data["t_mocap"][i] - scene.t0) / 1e9
            self._send_json({"t": t, "dur": scene.dur_s,
                             "playing": scene.playback.playing,
                             "speed": scene.playback.speed,
                             "mano": scene.mano_state()})
        elif self.path == "/api/camera":
            state = scene._camera_state()
            if state is None:
                self._send_json({"connected": False})
            else:
                self._send_json({"connected": True, **state})
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
        elif self.path == "/api/mano":
            if not scene.mano_state()["available"]:
                self._send_json({"error": scene.mano_error or "当前文件没有21点手骨架"})
            else:
                scene.set_mano_enabled(bool(body.get("enabled", False)))
                self._send_json(scene.mano_state())
        elif self.path == "/api/reset-camera":
            for client in scene.server.get_clients().values():
                scene._reset_camera(client)
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)


def _require_port_free(port: int, what: str) -> None:
    """固定端口:被占则报错退出,不静默偏移(viser 内部会自动 +1,会造成 iframe 错位)。"""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            print(f"[viz-h5] 端口 {port}({what})已被占用,请先释放: "
                  f"ss -ltnp | grep {port}", file=sys.stderr)
            raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="录制文件夹离线 3D 回放")
    ap.add_argument("dir", type=str, help="日期文件夹,如 /home/current/data/20260811")
    ap.add_argument("--port", type=int, default=8082, help="控制页端口(viser 自动 +1)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--mano", action="store_true", help="启动时显示MANO网格")
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
    # 固定端口:冲突直接退出(viser 静默 +1 会使控制页 iframe 指向错误端口)
    _require_port_free(args.port, "控制页")
    _require_port_free(_IFRAME_PORT, "viser 场景")

    # 预扫描文件信息(仅读头部,安全防护同 inspect)
    files: list[dict] = []
    for p in h5_files:
        info = probe_h5(p)
        if info is None:
            print(f"[viz-h5] 跳过 {p.name}(无法读取或含外部链接)", file=sys.stderr)
            continue
        files.append(info)
    scene = VizScene(_IFRAME_PORT, mano_enabled=args.mano)
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
