# acquisition — 人类操作演示数据采集

采集"人戴 Manus 手套操作物体"的演示数据:动捕背部刚体推算手腕位姿、手部原始骨架
以手腕为基准拼接成全局手、web 可视化、按键控制录制/保存/丢弃(HDF5)。

## 架构

```
Windows (Motive 3.4 + natnet-zenoh publisher)     Ubuntu 本机
  Motive 刚体:背部 + 各物体                          zenohd Router (常驻 tcp/0.0.0.0:7447)
    │                                               │
    └── connect 169.254.1.0:7447 ──────────────────┤
    (参数不变,对面从 viewer 换为 router)             │
                                                    ├── mocap viewer (--connect-endpoint)
  Manus 手套 → rawviz.out → zenoh_pub.py ──────────┤  (加 --router tcp/localhost:7447)
    (manus/raw_skeleton/* 120Hz)                     │
                                                    └── ★ acquisition (client 模式)
                                                        ├── 源时钟映射 + 50ms 缓冲
                                                        ├── 固定 60Hz 公共物理时间轴
                                                        ├── 腕/手/OBJ/交互同 tick 生成
                                                        ├── Viser web 可视化 (:8081)
                                                        └── 按键录制 → HDF5 v4
```

坐标系:Motive 内部为 x 前向/y 向上/z 向右;Streaming Z-up 输出
x 前向/y 向左/z 向上右手系、米制。刚体 position/quaternion 已在该世界系,消费端不重复转换。
Manus 骨架为 z-up 手套局部系;只用帧内相对量,经局部轴对齐 A 和手腕刚体 pose 放到世界。

## 目录结构

```
acquisition/
├── pixi.toml              # Python 3.12 环境(必须,与 natnet-zenoh 兼容)
├── config.yaml            # 运行时配置(刚体映射/offset/轴变换/键位)
├── config/zenohd_acl.yaml # zenohd Router 访问控制(默认拒绝,仅放行 mocap/** manus/**)
├── scripts/
│   ├── start_zenohd.sh            # 手动启动 Zenoh Router(带 ACL)
│   ├── install_zenohd_service.sh   # 安装常驻 systemd 服务(带 ACL)
│   ├── inspect_hdf5.py             # HDF5 检查:结构/时间戳/帧率/拼接一致性
│   ├── replay_hdf5.py              # 回放:v3 公共 60Hz 时间轴直接读取
│   └── e2e_check.py                # 端到端验证:按键录制 → HDF5 → inspect 校验
├── src/acquisition/       # 包:alignment/clock_sync/config/kinematics/stitching/
│                          #     manus_schema/streams/recorder/state_machine/live_view/cli
└── tests/                 # 单测 + zenoh 集成 + HDF5 往返/质量门 + pty 键盘
```

## 快速开始(本机验证,需真实设备流)

```bash
# 1. Router(两种方式任选)
pixi run start-router                    # 前台运行(带 ACL)
bash scripts/install_zenohd_service.sh --start   # 或装常驻 systemd 服务

# 2. 采集(带可视化,浏览器 http://127.0.0.1:8081)
pixi run record
# 按键: r 开始录制 / s 保存 / d 丢弃 / q 退出
# 无数据流入时状态栏提示「⏳ 等待设备启动」,设备就绪后自动恢复

# 3. 检查与回放(输出:/home/current/data/<日期>/<时间>.h5)
pixi run inspect -- /home/current/data/20260810/*.h5
pixi run replay -- /home/current/data/20260810/*.h5 --json out.jsonl
```

## 启动方式(三条独立路线)

```bash
# 路线 1/3:Manus 手套发布(真实 120Hz,需插入 dongle)
bash ../manus_pub.sh                        # 无前缀校准文件
bash ../manus_pub.sh --user syz             # 用 calibration/syzLeft/RightMetaglovePro.mcal
                                            # (无该用户的校准文件时报错退出)

# 路线 2/3:Windows rigid 发布(Motive 动捕,ssh 会话挂载,断开自动清理)
bash ../windows_pub.sh

# 路线 3/3:数据 web 订阅与收集(本程序,web :8081 + 录制按钮)
bash ../record.sh --object cylinder     # 只采集一种物体
bash ../record.sh --object hammer cube  # 采集多种物体(--object 必须显式指定)
```

三条路线独立启动/停止、任意顺序(router 常驻即可)。无数据流入时采集程序
提示「⏳ 等待设备启动」,设备就绪后自动恢复。

## Web 录制控制

浏览器打开 :8081 后，左侧「录制控制」面板提供开始录制、保存和丢弃按钮，与键盘
`r` / `s` / `d` 走同一个状态机。状态栏显示：

- 固定输出 `60 Hz`；
- 当前 Motive、左手和右手输入速率；
- 对齐器累计输出帧数与无效帧数。

输出频率不是运行时控件。所有传感器数据先进入缓冲，再在一个固定 60 Hz tick 上统一插值；
不能配齐的 tick 保留并标无效，不通过各流独立门控来丢帧。

> 端到端一键验证(需真实动捕 + manus 流):
> `pixi run python scripts/e2e_check.py --seconds 5`。
>
> 本目录位于合并后的 `mocap/acquisition/`,natnet(../natnet)与 manus(../manus)
> 为同级子目录;zenohd 二进制在 `../manus/.pixi`(启动脚本按相对位置自动定位)。

## 配置(config.yaml)

| 项 | 说明 |
|---|---|
| `router.endpoint` | zenohd 地址,本机 `tcp/127.0.0.1:7447` |
| `rigid_bodies.back` | Motive 手背/腕刚体 ID;Streaming 已把位姿转成 z-up 世界系 |
| `rigid_bodies.objects` | 物体刚体 ID → 名字(每个物体建一个刚体) |
| `hands.<side>.wrist_offset` | 手背刚体局部系 B 到解剖手腕局部系 W 的固定外参:`mode: body` 推荐且不受 Streaming 世界轴变化影响;`world` 才是固定世界方向 |
| `hands.<side>.wrist_rigid_id` | 可选:若 Motive 直接追踪解剖手腕刚体,用其位姿跳过 offset |
| `axis_transform` | Manus 骨架局部系 H → 手腕局部系 W 的轴对齐(permutation/signs,det=+1);默认 A·d=(d_x,d_z,−d_y),不是世界坐标变换 |
| `recording.output_dir` | 保存目录(临时文件 `.take_*_tmp.h5` 保存时原子改名) |
| `alignment.output_hz` | 唯一公共输出频率；当前必须为 `60` |
| `alignment.latency_ms` | 等待未来包围样本的固定缓冲；默认 `50 ms` |
| `alignment.mocap_max_gap_ms` | Motive 包围样本允许的最大间隔 |
| `alignment.manus_max_gap_ms` | Manus 包围样本允许的最大间隔 |
| `keys` | 键位:start/pause/save/discard/quit |

**五指桌面地标标定（无需新增 Motive 刚体）**:
1. 十个反光点摆好且静止后执行
   `pixi run capture-wrist-landmarks`，自动按 +Y→-Y 排序、3 秒平均并写 `config/wrist_landmarks.yaml`;旧配置保存为 `.yaml.bak`。
2. 在球心投影处画十字并移走反光球;配置使用桌面接触面 z=0,不能直接触碰球顶。
3. 五指 fingertip 同时压住对应点:thumb=24,index=5,middle=10,ring=15,little=20。
4. 左手:
   `bash scripts/calibrate_wrist_offset.sh left --user shd --back-name left_back --landmarks config/wrist_landmarks.yaml --hold 3 --max-rms-mm 5 --max-direction-deg 15`
5. 右手同理,将 `left/left_back` 改成 `right/right_back`。
6. 成功后写 `offset/shd.yaml` 并同步 `config.yaml`;质量门失败不会覆盖旧标定。
7. 重启 `record.sh`,检查 Manus root 与 wrist_position 重合、五指方向正确。

完整原理、十点坐标、实测指标和故障排查见
[`docs/FINGERTIP_LANDMARK_WRIST_CALIBRATION.md`](docs/FINGERTIP_LANDMARK_WRIST_CALIBRATION.md)。

## HDF5 文件结构（schema 4.0）

```text
/ attrs: h5_version=4.0, schema_layout=compact-aligned-60hz-v1,
         time_domain=linux-clock-monotonic, output_hz=60,
         start/end_wall_ns, effective_config_yaml
/time_ns                         (N,)       int64
/valid                           (N,)       uint8
/hands/{left,right}/
    keypoints_world              (N,21,3)   float32
    wrist_position               (N,3)      float32
    wrist_quaternion_xyzw        (N,4)      float32
    valid                        (N,)       uint8
    mano_beta                    (10,)      float32，可选
/objects/{name}/
    object_position              (N,3)      float32
    object_quaternion_xyzw       (N,4)      float32
    valid                        (N,)       uint8
/events/
    frame_index                  (E,)       int64
    type                         (E,)       uint8
```

`N = len(time_ns)`，所有逐帧 dataset 第一维严格等于 `N`。位置使用 lerp，四元数
使用最短弧 SLERP；仅使用包围目标时刻的样本，禁止外推。某一源缺失或间隙超限时仍
写入该 tick，并用全局和实体 `valid` 标记。v4 不再保存 `raw/`、`quality/`、
`interaction/`、markers 或插值诊断字段。

完整字段、dtype、事件边界和时间语义见
[`docs/HDF5_SCHEMA.md`](docs/HDF5_SCHEMA.md)。

**MANO beta 写回**：录制完成后运行：

```bash
bash ../add_mano_beta.sh -f                         # 当前日期
bash ../add_mano_beta.sh -f /home/current/data/20260812
```

每只手从最多 1000 个有效 `keypoints_world` 帧的稳健骨段长度估计一次
`mano_beta(10)`。H5 不保存 `mano_pose`、`mano_translation`、`mano_scale`、
`mano_joints16` 或 `mano_fit_valid` 等逐帧派生数组。

可视化的 MANO 16 关节直接从原始 21 点按以下索引取得：
`[0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3]`，顺序为
`wrist,index×3,middle×3,pinky×3,ring×3,thumb×3`。`viz-h5 --mano`
和根目录 `viewer.sh` 使用这 16 个原始点作为骨骼锚点，以 `beta` 生成手形，
再执行确定性的 MANO blend-shape + LBS 表面蒙皮；每帧不运行数值拟合。

物体网格从根目录 `assets/objects/<object>_m.obj`（米制）或
`assets/objects/<object>.obj`（毫米制，加载时自动乘 `0.001`）加载；
`<object>` 必须与 HDF5 的 `objects/<object>` 同名。当前 `cylinder_m.obj`
加载后的包围盒为 `18×150×18 mm`，原点接近几何中心，轴向为局部 `+Y`。
Motive 刚体 frame 到真实 OBJ frame 的固定外参统一保存在根目录
`config/object_offsets.yaml`。`rigid_bodies.objects` 中的每个物体都必须有同名
offset；任一缺失时采集程序直接报错，不再退化为刚体原点。

先在 Motive `Rigid Body > Visuals` 中把附加几何体对齐，记录 Geometry Location
（GL）XYZ 和 Geometry Orientation（GO）Pitch/Yaw/Roll，然后写入配置：

```bash
cd acquisition
pixi run add-object-offset -- cylinder \
  --gl-mm X Y Z \
  --go-deg PITCH YAW ROLL
```

GL 输入单位为毫米；GO 输入单位为度，遵循 Motive 的右手 XYZ 顺序：
Pitch 绕 X、Yaw 绕 Y、Roll 绕 Z。物体名必须同时匹配
`acquisition/config.yaml`、`config/object_offsets.yaml` 和 OBJ 文件名。已有同名
offset 默认拒绝覆盖，确认替换时添加 `--force`。

实时采集在公共 60 Hz tick 上按
`T_world_from_obj = T_world_from_rigid @ T_rigid_from_obj` 应用外参，并同时保存
`rigid_*` 原始刚体位姿和 `object_*` 真实 OBJ 位姿；Viewer 只读取
`object_position/object_quaternion_xyzw`，不重复应用 offset。

`pixi run object-offset -- <legacy.h5>` 仅用于旧 v1/v2 原始文件，生成独立派生文件；
v3/v4 已包含同 tick 的物体坐标，因此拒绝离线重复处理。

## 已验证

- **自动测试**：`pixi run test` 覆盖 60 Hz 节拍、同步插值、间隙失效、时钟映射、
  Manus 位姿协议、HDF5 v4 紧凑字段契约、StreamHub、状态机和原子保存。
- **端到端**：检查器验证固定节拍、统一长度、有效率、四元数、
  `keypoints_world[:,0] ≡ wrist_position` 和分离的物体位置/四元数。
- **保存/丢弃**：S 原子留存、D 删除无残留；目标文件已存在时保存失败且不覆盖旧文件。
- **可视化**：Viser 8081 正常启动；非终端 stdin 降级为阻塞读不崩溃。

## CI 与发布质量门

仓库级 [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 在 push/PR 上执行
acquisition、natnet-zenoh 和 mocap-viewer 的锁定环境测试。采集文件交付前必须运行：

```bash
pixi run inspect -- --strict /path/to/take.h5
```

退出码非 0 的文件不得进入数据集；格式契约和兼容策略见
[`docs/HDF5_SCHEMA.md`](docs/HDF5_SCHEMA.md)。

## 已知局限
- 未配置 `wrist_rigid_id` 时，腕部朝向由背部刚体加标定旋转推算；它不是手腕真实独立转动。

- 没有共享硬件时钟。采集端仍在线估计源时钟到 Linux monotonic 的
  offset/drift，但 v4 H5 不保存原始时间、序号和拟合诊断；这些信息不能从主文件恢复，
  因此不能宣称硬件级同步精度。
- 本机 zenohd(custom build)对「关闭 scouting 的 peer → router」连接不路由数据；
  本程序与测试统一用 **client 模式**连 router；Windows publisher 若遇此问题需改
  client 模式（见 `../natnet` 项目）。

## 测试

```bash
pixi run test          # acquisition 全部单元/集成测试
```
