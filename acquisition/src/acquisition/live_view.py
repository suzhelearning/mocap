"""拼接可视化(StitchedScene):Viser web 展示全局双手 + 物体动作。

UI 风格参考 NatNetViewerSource/src/natnet_zenoh/viewer.py:
- dark 主题(collapsible 布局、品牌色、无 logo/分享按钮)
- Markdown 状态栏
- 可折叠 folder:「录制控制」「视图」「桌面」
- Marker 按跟踪状态配色(id_kind/occluded 红)
- y-up 场景 + 桌面道具(操作物体场景参考)
- 保留采集特有元素:双手骨架(chain 配色)、手腕/物体刚体、录制按钮/对齐状态

手骨架配色常量复制自 manus/viz.py(manus 不是包,不可 import),来源已注明。
Frame 的姿态要求 wxyz 序,与协议 xyzw 互转。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import viser
import socket

from .alignment import AlignedFrame
from .config import Config
from .kinematics import quat_xyzw_to_wxyz
from .viser_core import load_object_mesh
from .state_machine import State


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

# Marker 按跟踪状态配色(参考 NatNetViewerSource/viewer.py)
MARKER_COLORS = {
    "active": (45, 212, 191),
    "asset_member": (74, 222, 128),
    "point_cloud": (251, 146, 60),
    "unknown": (148, 163, 184),
}
OCCLUDED_COLOR = (239, 68, 68)

WRIST_COLORS = {"left": (214, 39, 40), "right": (31, 119, 180)}
BACK_COLOR = (44, 160, 44)
OBJECT_COLOR = (148, 103, 189)

# MANO/MediaPipe 21 点配色(按手指,与 CHAIN_COLORS 一致):
# 0=wrist 灰, 1-4=拇指 橙, 5-8=食指 青, 9-12=中指 黄,
# 13-16=无名指 橙红, 17-20=小指 蓝
MANO_PALETTE = np.asarray([
    (210, 210, 210),
    (244, 162, 97), (244, 162, 97), (244, 162, 97), (244, 162, 97),
    (42, 157, 143), (42, 157, 143), (42, 157, 143), (42, 157, 143),
    (233, 196, 106), (233, 196, 106), (233, 196, 106), (233, 196, 106),
    (231, 111, 81), (231, 111, 81), (231, 111, 81), (231, 111, 81),
    (69, 123, 157), (69, 123, 157), (69, 123, 157), (69, 123, 157),
], dtype=np.uint8)

# MANO 手掌连线:wrist(0) → 五根手指根部(食指/中指/无名指/小指 MCP,拇指 CMC)
MANO_PALM_EDGES = ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17))
# 手指链:四指 MCP→PIP→DIP→TIP;拇指 CMC→MCP→IP→TIP
MANO_FINGER_EDGES = (
    (1, 2), (2, 3), (3, 4),        # 拇指
    (5, 6), (6, 7), (7, 8),        # 食指
    (9, 10), (10, 11), (11, 12),   # 中指
    (13, 14), (14, 15), (15, 16),  # 无名指
    (17, 18), (18, 19), (19, 20),  # 小指
)
MANO_PALM_LINE_COLOR = (170, 170, 170)


@dataclass(frozen=True)
class TableSpec:
    """桌面区域几何(Motive 世界系,米制):y=0 平面上的矩形框,无高度信息。

    默认值与回放 viewer 一致(viser_core.TABLE_*):Motive 原点位于桌子近边,
    中心沿 +z 偏移半个深度(1.440 × 0.900,中心 z=+0.45)。
    """

    width: float = 1.440
    depth: float = 0.900
    center_x: float = 0.0
    center_z: float = 0.45

    def frame_segments(self) -> np.ndarray:
        """桌面区域矩形框四条边(y=0 平面)。"""
        x0 = self.center_x - self.width / 2.0
        x1 = self.center_x + self.width / 2.0
        z0 = self.center_z - self.depth / 2.0
        z1 = self.center_z + self.depth / 2.0
        return np.asarray(
            [
                [[x0, 0.0, z0], [x1, 0.0, z0]],
                [[x1, 0.0, z0], [x1, 0.0, z1]],
                [[x1, 0.0, z1], [x0, 0.0, z1]],
                [[x0, 0.0, z1], [x0, 0.0, z0]],
            ],
            dtype=np.float32,
        )


def marker_color(marker: dict) -> tuple[int, int, int]:
    """按跟踪状态返回稳定显示色。"""
    if bool(marker.get("occluded")) or bool(marker.get("model_filled")):
        return OCCLUDED_COLOR
    return MARKER_COLORS.get(str(marker.get("id_kind")), MARKER_COLORS["unknown"])


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
            point_shape="circle",
            precision="float32",
            point_shading="gradient",
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

    def __init__(
        self,
        config: Config,
        host: str = "127.0.0.1",
        port: int | None = None,
        on_command: Callable[[str], None] | None = None,
    ) -> None:
        """on_command 由 viser 回调线程执行，只允许线程安全入队。"""
        self._cfg = config
        self._port = port or config.viz_port
        # 固定端口:冲突直接报错(viser 内部端口被占会静默 +1,采集页地址会漂移)
        with socket.socket() as _probe:
            try:
                _probe.bind((host, self._port))
            except OSError:
                raise RuntimeError(
                    f"viz 端口 {self._port} 已被占用,请先释放: ss -ltnp | grep {self._port}"
                )
        self._on_command = on_command or (lambda ch: None)
        self._last_state: State | None = None
        self.server = viser.ViserServer(host=host, port=self._port)

        # -- 主题与场景(参考 NatNetViewerSource/viewer.py) -------------------
        self.server.gui.configure_theme(
            control_layout="collapsible",
            control_width="medium",
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            brand_color=(16, 185, 129),
        )
        self.server.scene.set_up_direction((0.0, 1.0, 0.0))   # Motive y-up
        # 世界坐标轴(原点)会落在桌面中心遮挡手部:隐藏,坐标轴改由
        # _render_table 绘制在桌面区域左下角边缘
        self.server.scene.world_axes.visible = False
        self.server.initial_camera.position = (2.2, 1.7, 2.2)
        self.server.initial_camera.look_at = (0.0, 1.0, 0.0)
        self.server.initial_camera.up_direction = (0.0, 1.0, 0.0)
        self.grid = self.server.scene.add_grid(
            "/ground", width=6.0, height=6.0, plane="xz",
            cell_size=0.1, section_size=1.0,
            cell_color=(71, 85, 105), section_color=(148, 163, 184),
            plane_opacity=0.04,
        )
        self.status = self.server.gui.add_text(
            "状态", initial_value="○ 空闲", disabled=True, order=1)
        self.status_rate_mocap = self.server.gui.add_text(
            "动捕", initial_value="动捕 0.0Hz", disabled=True, order=2)
        self.status_rate_left = self.server.gui.add_text(
            "左手", initial_value="左手 0.0Hz", disabled=True, order=3)
        self.status_rate_right = self.server.gui.add_text(
            "右手", initial_value="右手 0.0Hz", disabled=True, order=4)
        self.status_health = self.server.gui.add_text(
            "流健康", initial_value="异常/缺帧 0", disabled=True, order=5)
        self._build_controls()

        self._back = self.server.scene.add_frame("/rigid/back", wxyz=(1, 0, 0, 0),
                                                 position=(0, 0, 0),
                                                 axes_length=0.03, axes_radius=0.002)
        self._objects = {
            name: self.server.scene.add_frame(f"/rigid/object/{name}",
                                              wxyz=(1, 0, 0, 0), position=(0, 0, 0),
                                              axes_length=0.025, axes_radius=0.002)
            for name in config.objects
        }
        self._object_meshes: dict[str, viser.MeshHandle] = {}
        for name in config.objects:
            loaded = load_object_mesh(name)
            if loaded is None:
                continue
            _path, vertices, faces = loaded
            self._object_meshes[name] = self.server.scene.add_mesh_simple(
                f"/rigid/object/{name}/mesh",
                vertices=vertices,
                faces=faces,
                color=OBJECT_COLOR,
                opacity=0.82,
                side="double",
                material="standard",
            )
        self._object_labels = {
            name: self.server.scene.add_label(f"/label/object/{name}", name, position=(0, 0, 0),
                                              anchor="bottom-center", font_screen_scale=0.8)
            for name in config.objects
        }
        self._wrists = {
            side: self.server.scene.add_frame(f"/wrist/{side}", wxyz=(1, 0, 0, 0),
                                              position=(0, 0, 0),
                                              axes_length=0.025, axes_radius=0.002)
            for side in ("left", "right")
        }
        self._hands = {
            side: _HandMesh(self.server, f"/hand/{side}")
            for side in ("left", "right")
        }
        # 默认只显示 MANO 21 点;25 点原始骨架默认隐藏(由开关开启)
        self._show_hands = False
        for mesh in self._hands.values():
            mesh.pc.visible = False
            mesh.ls.visible = False
        # MANO/MediaPipe 21 关键点(每侧一点云,按手指分色,默认显示)
        self._mano_pc = {
            side: self.server.scene.add_point_cloud(
                f"/mano/{side}", points=np.zeros((0, 3)),
                colors=np.zeros((0, 3), dtype=np.uint8),
                point_size=0.005, point_shape="circle", precision="float32",
                point_shading="gradient", visible=False,
            )
            for side in ("left", "right")
        }
        # MANO 手掌连线(腕→各指根),与 21 点同显隐
        self._mano_lines = {
            side: self.server.scene.add_line_segments(
                f"/mano/{side}/palm", points=np.zeros((0, 2, 3)),
                colors=np.zeros((0, 2, 3), dtype=np.uint8),
                line_width=3.0, visible=False,
            )
            for side in ("left", "right")
        }
        self._show_mano = True
        self._marker_pc = self.server.scene.add_point_cloud(
            "/markers", points=np.zeros((0, 3)), colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.012, point_shape="circle", precision="float32",
            point_shading="gradient",
        )
        # 刚体组成 markers(id_kind=asset_member,即 left_wrist/left_dip 等):
        # 标定完成后默认隐藏,由「left_rigid」开关控制
        self._rigid_marker_pc = self.server.scene.add_point_cloud(
            "/rigid_markers", points=np.zeros((0, 3)),
            colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.012, point_shape="circle", precision="float32",
            point_shading="gradient", visible=False,
        )
        self._show_markers = True
        self._show_rigid_markers = False
        self._table_handles: list[object] = []
        self._render_table()

    # -- GUI ---------------------------------------------------------------

    def _build_controls(self) -> None:
        """三个可折叠 folder:录制控制 / 视图 / 桌面。

        回调均只把录制命令转发给主循环。
        """
        keymap = self._cfg.keymap
        # 布局:录制控制(0) → 状态 items(1-6) → 视图(10) → 桌面(11)
        with self.server.gui.add_folder("录制控制", order=0):
            self._btn_start = self.server.gui.add_button(
                "开始录制", color=(44, 160, 44), hint=f"键盘键 {keymap['start']!r}")
            self._btn_save = self.server.gui.add_button(
                "保存", color=(31, 119, 180), hint=f"键盘键 {keymap['save']!r}")
            self._btn_discard = self.server.gui.add_button(
                "丢弃", color=(214, 39, 40), hint=f"键盘键 {keymap['discard']!r}")

        with self.server.gui.add_folder("视图", order=10):
            self._marker_size = self.server.gui.add_slider(
                "标记点大小", min=0.002, max=0.05, step=0.001, initial_value=0.012)
            self._show_markers_cb = self.server.gui.add_checkbox(
                "原始标记点", initial_value=True)
            self._btn_rigid = self.server.gui.add_button(
                "left_rigid", color=(200, 60, 60),
                hint="点击切换标定刚体(left_wrist/left_dip)markers 显示")
            self._show_mano_cb = self.server.gui.add_checkbox(
                "MANO 21 点", initial_value=True)
            self._show_hands_cb = self.server.gui.add_checkbox(
                "25 点骨架", initial_value=False)
            self._show_grid_cb = self.server.gui.add_checkbox("地面网格", initial_value=True)
            self._show_rigid_cb = self.server.gui.add_checkbox("刚体坐标轴", initial_value=True)
            self._reset_view = self.server.gui.add_button("重置视角")

        with self.server.gui.add_folder("桌面", order=11):
            self._show_table = self.server.gui.add_checkbox("显示桌面区域", initial_value=True)
            self._table_width_mm = self.server.gui.add_number(
                "宽 (mm)", initial_value=1440, min=1, step=1)
            self._table_depth_mm = self.server.gui.add_number(
                "深 (mm)", initial_value=900, min=1, step=1)
            self._table_center_x_mm = self.server.gui.add_number(
                "中心 X (mm)", initial_value=0, step=1)
            self._table_center_z_mm = self.server.gui.add_number(
                "中心 Z (mm)", initial_value=450, step=1)

        # -- 回调 ------------------------------------------------------------
        @self._btn_start.on_click
        async def _on_start(_e):
            self._on_command(keymap["start"])

        @self._btn_save.on_click
        async def _on_save(_e):
            self._on_command(keymap["save"])

        @self._btn_discard.on_click
        async def _on_discard(_e):
            self._on_command(keymap["discard"])

        self._marker_size.on_update(lambda _e: self._apply_marker_style())
        self._show_markers_cb.on_update(lambda _e: setattr(self, "_show_markers",
                                                           self._show_markers_cb.value))

        @self._btn_rigid.on_click
        async def _toggle_rigid(_e):
            # 点击切换刚体 markers 显示(状态由场景反馈:开=出现刚体 markers)
            self._show_rigid_markers = not self._show_rigid_markers
        self._show_mano_cb.on_update(lambda _e: setattr(self, "_show_mano",
                                                        self._show_mano_cb.value))
        self._show_hands_cb.on_update(lambda _e: self._apply_hand_visibility())
        self._show_grid_cb.on_update(lambda _e: setattr(self.grid, "visible",
                                                        self._show_grid_cb.value))
        self._show_rigid_cb.on_update(self._update_rigid_visibility)
        self._reset_view.on_click(self._reset_cameras)
        self._show_table.on_update(self._update_table_visibility)
        for control in (self._table_width_mm, self._table_depth_mm,
                        self._table_center_x_mm, self._table_center_z_mm):
            control.on_update(lambda _e: self._render_table())

        self.update_state(State.IDLE)     # 初始按钮态

    def update_state(self, state: State | None) -> None:
        """按状态机状态控制按钮可用性;仅状态变化时推送(避免高频重复消息)。"""
        if state is None or state is self._last_state:
            return
        self._last_state = state
        idle = state is State.IDLE
        recording = state is State.RECORDING
        self._btn_start.disabled = not idle
        self._btn_save.disabled = not recording
        self._btn_discard.disabled = not recording

    # -- 视图/桌面控制 -------------------------------------------------------

    def _apply_marker_style(self) -> None:
        self._marker_pc.point_size = self._marker_size.value

    def _apply_hand_visibility(self) -> None:
        """25 点骨架开关:立即生效(数据到达时 update 也会应用)。"""
        self._show_hands = self._show_hands_cb.value
        for mesh in self._hands.values():
            mesh.pc.visible = self._show_hands
            mesh.ls.visible = self._show_hands

    def _update_rigid_visibility(self, _event: object = None) -> None:
        visible = self._show_rigid_cb.value
        self._back.visible = visible
        for frame in self._objects.values():
            frame.visible = visible
        for frame in self._wrists.values():
            frame.visible = visible

    def _reset_cameras(self, _event: object = None) -> None:
        for client in self.server.get_clients().values():
            client.camera.position = (2.2, 1.7, 2.2)
            client.camera.look_at = (0.0, 1.0, 0.0)
            client.camera.up_direction = (0.0, 1.0, 0.0)

    def _table_spec(self) -> TableSpec:
        return TableSpec(
            width=float(self._table_width_mm.value) / 1000.0,
            depth=float(self._table_depth_mm.value) / 1000.0,
            center_x=float(self._table_center_x_mm.value) / 1000.0,
            center_z=float(self._table_center_z_mm.value) / 1000.0,
        )

    def _render_table(self) -> None:
        """重建桌面区域道具(矩形线框 + 尺寸标签 + 边缘坐标轴;无高度/实体)。"""
        for handle in self._table_handles:
            handle.remove()
        spec = self._table_spec()
        visible = self._show_table.value
        self._table_handles = [
            self.server.scene.add_line_segments(
                "/table/frame",
                points=spec.frame_segments(),
                colors=(16, 185, 129),
                line_width=3.0,
                visible=visible,
            ),
            self.server.scene.add_label(
                "/table/size",
                text=f"{int(self._table_width_mm.value)} × {int(self._table_depth_mm.value)} mm",
                position=(spec.center_x, 0.0, spec.center_z - spec.depth / 2.0 - 0.02),
                anchor="bottom-center",
                font_screen_scale=0.8,
                visible=visible,
            ),
            # 方向参考坐标轴:位于桌面区域左下角外侧(不遮挡手部)
            self.server.scene.add_frame(
                "/table/axes", wxyz=(1, 0, 0, 0),
                position=(spec.center_x - spec.width / 2.0 - 0.05, 0.0,
                          spec.center_z - spec.depth / 2.0 - 0.05),
                axes_length=0.08, axes_radius=0.003,
                visible=visible,
            ),
        ]

    def _update_table_visibility(self, _event: object = None) -> None:
        for handle in self._table_handles:
            handle.visible = self._show_table.value

    # -- 更新 -------------------------------------------------------------

    def update_aligned(
        self,
        frame: AlignedFrame | None,
        *,
        edges: dict[str, list[tuple[int, int, int]]],
        calib_rigid_ids: set[int] | None = None,
        status: dict[str, str] | None = None,
        state: State | None = None,
    ) -> None:
        """只消费一个统一帧；物体、左右手绝不从不同时间索引拼接。"""
        if frame is None:
            self.update(
                None, {}, edges, latest_mano={},
                calib_rigid_ids=calib_rigid_ids, status=status, state=state,
            )
            return
        mocap = None
        if frame.mocap_frame is not None:
            mocap = dict(frame.mocap_frame)
            bodies = [dict(body) for body in mocap.get("rigid_bodies", [])]
            by_id = {int(body["id"]): body for body in bodies}
            for name, rigid_id in self._cfg.objects.items():
                obj = frame.objects[name]
                body = by_id.get(rigid_id)
                if body is not None:
                    body["position"] = obj.object_position
                    body["quaternion_xyzw"] = obj.object_quaternion_xyzw
                    body["tracking_valid"] = obj.valid
            mocap["rigid_bodies"] = bodies
        hands = {
            side: {
                "nodes_global": hand.nodes_world,
                "wrist_pos": hand.wrist_position,
                "wrist_quat_xyzw": hand.wrist_quaternion_xyzw,
            }
            for side, hand in frame.hands.items()
            if hand.valid
        }
        mano = {
            side: {"keypoints_global": hand.mano_skeleton}
            for side, hand in frame.hands.items()
            if hand.valid
        }
        self.update(
            mocap,
            hands,
            edges,
            latest_mano=mano,
            calib_rigid_ids=calib_rigid_ids,
            status=status,
            state=state,
        )

    def update(
        self,
        mocap_frame: dict | None,
        hands: dict[str, dict],
        edges: dict[str, list[tuple[int, int, int]]],
        latest_mano: dict[str, dict | None] | None = None,
        calib_rigid_ids: set[int] | None = None,
        status: dict[str, str] | None = None,
        state: State | None = None,
    ) -> None:
        """刷新全部场景元素。

        mocap_frame: 最新 mocap 帧或 None
        hands: {side: {"nodes_global": (25,3), "wrist_pos": (3,),
                       "wrist_quat_xyzw": (4,)}} 或 None
        edges: {side: [(child, parent, chain)]} 或 None
        """
        rigid_visible = self._show_rigid_cb.value

        # 背部刚体(无 back 配置时隐藏,如 Motive 直接追踪手腕的模式)
        if self._cfg.back_rigid_id is None:
            self._back.visible = False
        elif rigid_visible:
            back_pos = np.zeros(3)
            back_wxyz = (1.0, 0, 0, 0)
            if mocap_frame is not None:
                for rb in mocap_frame.get("rigid_bodies", []):
                    if rb.get("id") == self._cfg.back_rigid_id:
                        back_pos = np.asarray(rb["position"], dtype=float)
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
                    position = np.asarray(obj["position"], dtype=float)
                    frame.position = position
                    frame.wxyz = quat_xyzw_to_wxyz(obj["quaternion_xyzw"])
                    frame.visible = rigid_visible
                    self._object_labels[name].visible = rigid_visible
                    self._object_labels[name].position = (
                        position + np.array([0.0, 0.06, 0.0])
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
            self._wrists[side].visible = rigid_visible
            self._wrists[side].position = h["wrist_pos"]
            self._wrists[side].wxyz = quat_xyzw_to_wxyz(h["wrist_quat_xyzw"])
            mesh = self._hands[side]
            mesh.set_edges(edges.get(side) or [])
            mesh.update(h["nodes_global"])
            mesh.pc.visible = self._show_hands
            mesh.ls.visible = self._show_hands

            # MANO/MediaPipe 21 关键点(分色点云 + 手掌连线)
            mano = (latest_mano or {}).get(side)
            pc = self._mano_pc[side]
            ln = self._mano_lines[side]
            if self._show_mano and mano is not None and mano.get("keypoints_global") is not None:
                kp = np.asarray(mano["keypoints_global"], dtype=float)
                pc.points = kp
                pc.colors = MANO_PALETTE
                pc.visible = True
                # 连线:手掌线(灰)+ 手指链(按手指色,取起点颜色)
                seg = np.asarray(
                    [[kp[a], kp[b]] for a, b in MANO_PALM_EDGES + MANO_FINGER_EDGES],
                    dtype=np.float32)
                palm_colors = np.full((len(MANO_PALM_EDGES), 2, 3),
                                      MANO_PALM_LINE_COLOR, dtype=np.uint8)
                finger_colors = np.asarray(
                    [[MANO_PALETTE[a], MANO_PALETTE[a]]
                     for a, _b in MANO_FINGER_EDGES], dtype=np.uint8)
                ln.points = seg
                ln.colors = np.concatenate([palm_colors, finger_colors], axis=0)
                ln.visible = True
            else:
                pc.points = np.zeros((0, 3))
                pc.visible = False
                ln.points = np.zeros((0, 2, 3))
                ln.visible = False

        # markers 分流:
        #   普通 markers + 其他刚体(cylinder 等)→ 原始标记点(默认显示)
        #   标定刚体(left_wrist/left_dip/right_wrist/right_dip)→ left_rigid button 控制
        if mocap_frame is not None:
            markers = mocap_frame.get("markers", [])
            calib = calib_rigid_ids or set()
            pts = np.asarray([m["position"] for m in markers], dtype=float)
            colors = np.asarray([marker_color(m) for m in markers], dtype=np.uint8)
            calib_mask = np.asarray([
                m.get("id_kind") == "asset_member"
                and m.get("model_id") in calib
                for m in markers])
            if self._show_markers and markers:
                self._marker_pc.points = pts[~calib_mask]
                self._marker_pc.colors = colors[~calib_mask]
            else:
                self._marker_pc.points = np.zeros((0, 3))
            if self._show_rigid_markers and markers:
                self._rigid_marker_pc.points = pts[calib_mask]
                self._rigid_marker_pc.colors = colors[calib_mask]
                self._rigid_marker_pc.visible = True
            else:
                self._rigid_marker_pc.points = np.zeros((0, 3))
                self._rigid_marker_pc.visible = False
        else:
            self._marker_pc.points = np.zeros((0, 3))
            self._rigid_marker_pc.points = np.zeros((0, 3))

        # 录制按钮状态
        self.update_state(state)

        # 状态栏(独立 item,各自更新,避免整块重排跳变)
        if status:
            self.status.value = status.get("ctrl", "")
            self.status_rate_mocap.value = status.get("rate_mocap", "")
            self.status_rate_left.value = status.get("rate_left", "")
            self.status_rate_right.value = status.get("rate_right", "")
            self.status_health.value = status.get("health", "")

    def stop(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass
