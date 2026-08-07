"""拼接可视化(StitchedScene):Viser web 展示全局双手 + 物体动作。

渲染内容:
- 背部刚体:绿色 Frame(tracking_valid=false 时红色)
- 物体刚体:每个物体一个紫色 Frame + 名字标签
- 左右手腕:红/蓝小 Frame(offset 推算结果)
- 双手骨架:点云(25 节点,chain 配色)+ 骨骼线段 —— 全局拼接坐标
- 原始 markers:灰色小点云(默认开,可用 --no-markers 关闭)
- 状态栏:顶部文字

手骨架配色常量复制自 manus/viz.py(manus 不是包,不可 import),来源已注明。
Frame 的姿态要求 wxyz 序,与协议 xyzw 互转。
"""

from __future__ import annotations

import numpy as np
import viser

from .config import Config
from .kinematics import quat_xyzw_to_wxyz

# 手指链配色(复制自 manus/viz.py)5=拇指 6=食指 7=中指 8=无名指 9=小指 13=手掌
CHAIN_COLORS = {
    5: (244, 162, 97),    # 拇指 橙
    6: (42, 157, 143),    # 食指 青
    7: (233, 196, 106),   # 中指 黄
    8: (231, 111, 81),    # 无名指 橙红
    9: (69, 123, 157),    # 小指 蓝
    13: (210, 210, 210),  # 手掌 亮灰
}
DEFAULT_COLOR = (160, 160, 160)

WRIST_COLORS = {"left": (214, 39, 40), "right": (31, 119, 180)}
BACK_COLOR = (44, 160, 44)
OBJECT_COLOR = (148, 103, 189)
MARKER_COLOR = (170, 170, 170)


class _HandMesh:
    """一只手的点云 + 骨骼线段,坐标随拼接结果更新。"""

    def __init__(self, server: viser.ViserServer, path: str, node_count: int = 25):
        self._server = server
        self._path = path
        self.node_count = node_count
        self._edges: list[tuple[int, int, int]] = []
        self.node_color = np.full((node_count, 3), DEFAULT_COLOR, dtype=np.uint8)
        self.pc = server.scene.add_point_cloud(
            f"{path}/points",
            points=np.zeros((node_count, 3)),
            colors=self.node_color,
            point_size=0.006,
        )
        self.ls = server.scene.add_line_segments(
            f"{path}/bones", points=np.zeros((0, 2, 3)),
            colors=np.zeros((0, 2, 3), dtype=np.uint8), line_width=2.0,
        )
        self._lines_ready = False

    def set_edges(self, edges: list[tuple[int, int, int]]) -> None:
        """首次设置拓扑并上色;拓扑在会话内不变。"""
        if self._lines_ready or not edges:
            return
        self._edges = list(edges)
        for child, _parent, chain in edges:
            if 0 <= child < self.node_count:
                self.node_color[child] = CHAIN_COLORS.get(chain, DEFAULT_COLOR)
        self.node_color[0] = CHAIN_COLORS[13]      # 手掌 root
        self.pc.colors = self.node_color
        self._lines_ready = True

    def update(self, nodes_global: np.ndarray) -> None:
        """nodes_global: (N,3) 全局拼接坐标。"""
        nodes = np.asarray(nodes_global, dtype=float)
        if nodes.shape != (self.node_count, 3):
            return
        self.pc.points = nodes
        if self._lines_ready:
            seg = np.array([[nodes[c], nodes[p]] for c, p, _ in self._edges])
            per_seg = np.array([CHAIN_COLORS.get(ch, DEFAULT_COLOR)
                                for _, _, ch in self._edges], dtype=np.uint8)
            colors = np.repeat(per_seg[:, None, :], 2, axis=1)
            self.ls.points = seg
            self.ls.colors = colors


class StitchedScene:
    """拼接可视化场景;update() 应在主循环以 ~30fps 调用。"""

    def __init__(self, config: Config, host: str = "0.0.0.0", port: int | None = None) -> None:
        self._cfg = config
        self._port = port or config.viz_port
        self.server = viser.ViserServer(host=host, port=self._port)
        self.server.scene.add_grid("/grid", width=2.0, cell_size=0.1)
        self.status = self.server.gui.add_text("/status", "waiting for data...")

        self._back = self.server.scene.add_frame("/rigid/back", wxyz=(1, 0, 0, 0),
                                                 position=(0, 0, 0), axes_length=0.15)
        self._objects = {
            name: self.server.scene.add_frame(f"/rigid/object/{name}",
                                              wxyz=(1, 0, 0, 0), position=(0, 0, 0),
                                              axes_length=0.08)
            for name in config.objects
        }
        self._object_labels = {
            name: self.server.scene.add_label(f"/label/object/{name}", name, position=(0, 0, 0))
            for name in config.objects
        }
        self._wrists = {
            side: self.server.scene.add_frame(f"/wrist/{side}", wxyz=(1, 0, 0, 0),
                                              position=(0, 0, 0), axes_length=0.07)
            for side in ("left", "right")
        }
        self._hands = {
            side: _HandMesh(self.server, f"/hand/{side}")
            for side in ("left", "right")
        }
        self._marker_pc = self.server.scene.add_point_cloud(
            "/markers", points=np.zeros((0, 3)), colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.004,
        )
        self._show_markers = True

    # -- 更新 -------------------------------------------------------------

    def update(
        self,
        mocap_frame: dict | None,
        hands: dict[str, dict],
        edges: dict[str, list[tuple[int, int, int]]],
        status_text: str = "",
    ) -> None:
        """刷新全部场景元素。

        mocap_frame: 最新 mocap 帧或 None
        hands: {side: {"nodes_global": (25,3), "wrist_pos": (3,),
                       "wrist_quat_xyzw": (4,)}} 或 None
        edges: {side: [(child, parent, chain)]} 或 None
        """
        # 背部刚体
        back_pos = np.zeros(3)
        back_wxyz = (1.0, 0, 0, 0)
        if mocap_frame is not None:
            for rb in mocap_frame.get("rigid_bodies", []):
                if rb.get("id") == self._cfg.back_rigid_id:
                    back_pos = rb["position"]
                    back_wxyz = quat_xyzw_to_wxyz(rb["quaternion_xyzw"])
                    valid = bool(rb.get("tracking_valid", True))
                    self._back.color = BACK_COLOR if valid else (255, 60, 60)
                    break
        self._back.position = back_pos
        self._back.wxyz = back_wxyz

        # 物体刚体
        if mocap_frame is not None:
            for name, oid in self._cfg.objects.items():
                obj = None
                for rb in mocap_frame.get("rigid_bodies", []):
                    if rb.get("id") == oid:
                        obj = rb
                        break
                frame = self._objects[name]
                if obj is not None:
                    frame.position = obj["position"]
                    frame.wxyz = quat_xyzw_to_wxyz(obj["quaternion_xyzw"])
                    frame.visible = True
                    self._object_labels[name].position = (
                        np.asarray(obj["position"], dtype=float) + np.array([0.0, 0.06, 0.0])
                    )
                else:
                    frame.visible = False
                    self._object_labels[name].visible = False

        # 手腕 + 双手
        for side in ("left", "right"):
            h = hands.get(side)
            if h is None:
                self._wrists[side].visible = False
                continue
            self._wrists[side].visible = True
            self._wrists[side].position = h["wrist_pos"]
            self._wrists[side].wxyz = quat_xyzw_to_wxyz(h["wrist_quat_xyzw"])
            mesh = self._hands[side]
            mesh.set_edges(edges.get(side) or [])
            mesh.update(h["nodes_global"])

        # 原始 markers
        if self._show_markers and mocap_frame is not None:
            markers = mocap_frame.get("markers", [])
            pts = np.asarray([m["position"] for m in markers], dtype=float)
            self._marker_pc.points = pts
            self._marker_pc.colors = np.full(pts.shape, MARKER_COLOR, dtype=np.uint8)
        else:
            self._marker_pc.points = np.zeros((0, 3))

        # 状态栏
        if status_text:
            self.status.text = status_text

    def stop(self) -> None:
        try:
            self.server._close()
        except Exception:
            pass
