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
                                                    └── ★ acquisition (本程序, client 模式)
                                                        ├── 手腕位姿: 背部刚体 + offset
                                                        ├── 拼接: g = p_w + R_w·(A·(p_i − p_0))
                                                        ├── Viser web 可视化 (:8081)
                                                        └── 按键录制 → HDF5
```

坐标系:动捕为 Motive Y-up 右手系、米制;Manus 骨架为 z-up、米制(手套各自建系,
拼接只用帧内相对量)。拼接公式与轴变换见 `config.yaml` 的 `axis_transform` 注释。

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
│   ├── replay_hdf5.py              # 回放:120Hz 目标轴插值(位置 lerp + 四元数 slerp)
│   └── e2e_check.py                # 端到端验证:按键录制 → HDF5 → inspect 校验
├── src/acquisition/       # 包:config/kinematics/stitching/manus_schema/
│                          #     streams/recorder/keyboard/state_machine/live_view/cli
└── tests/                 # 69 项:单测 + zenoh 集成 + HDF5 往返 + pty 键盘
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

# 路线 3/3:数据 web 订阅与收集(本程序,web :8081 + 录制按钮 + 采集频率)
bash ../record.sh
```

三条路线独立启动/停止、任意顺序(router 常驻即可)。无数据流入时采集程序
提示「⏳ 等待设备启动」,设备就绪后自动恢复。

## Web 录制控制与采集频率

浏览器打开 :8081 后,左侧「录制控制」面板:

| 控件 | 作用 |
|---|---|
| 开始录制 / 保存 / 丢弃 | 与键盘 r / s / d 等价(状态机同一路径) |
| 采集频率 (Hz) 下拉(30/60/100/120,默认 100) | 运行时改落盘采样率:输入流 120Hz 时按时间门控下采样,立即生效 |

键盘按键保留不变;状态栏显示当前采集频率(`采集@100Hz`)。落盘频率写入端
门控(`rate.py` RateGate,按流分槽:动捕/左右手各自达标),可视化仍按全帧率
刷新;流速率低于目标时不丢帧。

> 端到端一键验证(需真实动捕 + manus 流):
> `pixi run python scripts/e2e_check.py --seconds 5`。
>
> 本目录位于合并后的 `mocap/acquisition/`,natnet(../natnet)与 manus(../manus)
> 为同级子目录;zenohd 二进制在 `../manus/.pixi`(启动脚本按相对位置自动定位)。

## 配置(config.yaml)

| 项 | 说明 |
|---|---|
| `router.endpoint` | zenohd 地址,本机 `tcp/127.0.0.1:7447` |
| `rigid_bodies.back` | Motive 背部 markers 刚体 ID |
| `rigid_bodies.objects` | 物体刚体 ID → 名字(每个物体建一个刚体) |
| `hands.<side>.wrist_offset` | 背部刚体身体系到手腕的偏移(需真机标定):`mode: body` 随躯干转动 / `world` 固定方向;可选 `yaw/pitch/roll_deg` 姿态修正 |
| `hands.<side>.wrist_rigid_id` | 可选:若 Motive 直接追踪手腕刚体,用其位姿跳过 offset |
| `axis_transform` | 骨架 z-up → Motive y-up 轴变换(permutation/signs,须为真旋转 det=+1);默认 A·d=(d_x, d_z, −d_y) |
| `recording.output_dir` | 保存目录(临时文件 `.take_*_tmp.h5` 保存时原子改名) |
| `keys` | 键位:start/pause/save/discard/quit |

**真机标定流程**:
1. Motive 建"背部刚体"+ 各物体刚体;Windows publisher 参数不变
2. 静止站立,记录手腕实际位置,反推 `wrist_offset.xyz`
3. 开 live 视图看手指方向:上/下颠倒改 `signs`,左/右颠倒改 `permutation`

## HDF5 文件结构

```
/                 attrs: take_id, start/end_wall_ns, config_yaml(全文), keymap, 轴变换
/mocap            frame_number, motive_timestamp, publisher_received_time_ns,
                  t_ubuntu_ns, publisher_dropped_frames,
                  rigid_bodies/{ids,positions,quaternions_xyzw,tracking_valid,mean_error}(vlen)
                  markers/{positions,raw_ids,occluded,id_kinds}(vlen, 可关)
/hands/{left,right}  t_ubuntu_ns, seq, nodes_raw(25,3), wrist_position,
                  wrist_quaternion_xyzw, nodes_global(25,3)
/objects/{name}   t_ubuntu_ns, position, quaternion_xyzw, tracking_valid
/events           t_ubuntu_ns, type(0=start..5=quit), note
```

时间基准:各流独立 `t_ubuntu_ns`(接收端墙钟)。手部 120Hz 与动捕 120Hz 帧率不同,
回放时插值对齐(`replay_hdf5.py`)。markers 每帧数量可变,用 h5py vlen 存储。

## 已验证(2026-08-07,真实环境)

- **单元/集成测试**:47 项全绿(config 校验、拼接数值示例、消息解析、StreamHub 经
  zenohd 收发、HDF5 写读往返、状态机全转移、pty 键盘)
- **端到端**:真实动捕流(120.2Hz,刚体 1/2/3)+ 手部流(29.7Hz)→ 按键录制 →
  HDF5 时间戳单调、无大间隙、`nodes_global[0] ≡ wrist_position`(误差 <1e-6)、
  objects 子表与 events 完整、edges 拓扑落盘
- **保存/丢弃**:S 原子改名留存,D 删除无残留;录制中 Q 先丢弃再退出
- **可视化**:Viser 8081 正常启动;非终端 stdin 降级为阻塞读不崩溃

## 已知局限

- 手腕朝向继承背部刚体(无法测真实腕转)→ 升级路径:建手腕刚体走 `wrist_rigid_id`
- 跨机时钟未同步:Motive 原始时间戳留底,采集端统一打 `t_ubuntu_ns`,段级线性校正留作后续
- 本机 zenohd(custom build)对「关闭 scouting 的 peer → router」连接不路由数据;
  本程序与测试统一用 **client 模式**连 router;Windows publisher 若遇此问题需改
  client 模式(见 `../natnet` 项目)

## 测试

```bash
pixi run test          # 47 项:config/运动学/拼接/解析/流订阅/录制往返/状态机/键盘
```
