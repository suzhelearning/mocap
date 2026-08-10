# 消费指南：订阅 Motive 动捕数据（纯 Zenoh）

本文档说明如何消费 /home/current/syz/mocap 项目接收的 Motive 动捕数据流。整条链路**纯 Zenoh**，不依赖 ROS 2 或其他中间件：数据以 JSON 帧的形式发布在 Zenoh key 上，任何消费者连上 Zenoh 网络即可订阅。

## 数据流架构

```
Windows PC (192.168.110.6)                        Ubuntu (本机)
Motive 3.4 Tracker
   │ NatNet Unicast (loopback)
   ▼
natnet-zenoh publisher ── Zenoh TCP 7447 ──→  Ubuntu 数据服务（viewer 或任意监听者）
                                                │ key: mocap/hands/frame
                                                ▼
                                        消费者（subscribe / 自己的代码）
```

- **发布端**：Windows 上的 natnet-zenoh publisher（Motive 同机，loopback 收数据）
- **手部数据**：`manus/raw_skeleton/{left,right}_hand`（原始 25 节点）+ `manus/mano_skeleton/{left,right}_hand`（MANO/MediaPipe 21 点，同帧同步发布）
- **数据通道**：有线直连 link-local（`169.254.1.0` ↔ `169.254.213.247`），TCP 7447，实测 ~0.3ms、120Hz、0 丢帧
- **Zenoh 网络**：显式 TCP peer 连接；**必须先有人监听 7447（数据服务端），消费者再 connect 加入**
- **服务端转发（relay）**：Zenoh 1.9 peer 只在直连对之间交换订阅路由，多跳转发不可靠。因此数据服务端（`view`）收到帧后会**重新发布到同一 key**（带 `relayed_by_mocap_viewer` 标记防回环），connect 模式的消费者相当于直连发布者，数据稳定可达。不需要转发时可加 `--no-relay`。

## 快速开始

```bash
# 1. 数据服务端（监听 7447）——viewer 或 natnet-zenoh-subscriber 二选一
pixi run view              # web 可视化 http://127.0.0.1:8080（监听 tcp/0.0.0.0:7447）

# 2. 任意消费者（connect 模式，不占端口）
pixi run subscribe                          # 打印帧率/marker 数统计
pixi run subscribe -- --output cap.jsonl    # 同时记录 JSONL（每行一帧）
pixi run subscribe -- --connect-endpoint tcp/169.254.1.0:7447   # 指定数据服务端
```

> 端口说明：`view` 监听 0.0.0.0:7447；`subscribe` 用 connect 模式连接已有服务端，**不**占用端口，可与 view 并存。

> 无数据时的行为：数据服务端/采集程序在未收到任何流时会持续提示「⏳ 等待设备启动」，
> 设备(Motive/Windows publisher、Manus 手套)就绪后自动恢复。

## 用代码消费

```python
import zenoh
from natnet_zenoh.schema import FRAME_KEY, decode_frame
from natnet_zenoh.zenoh_transport import build_peer_config

def on_frame(sample) -> None:
    frame = decode_frame(sample.payload.to_string())   # 校验并解析 JSON 帧
    print(frame["frame_number"], len(frame["markers"]))

config = build_peer_config(connect_endpoint="tcp/127.0.0.1:7447")
with zenoh.open(config) as session:
    session.declare_subscriber(FRAME_KEY, on_frame)
    input("按回车退出\n")
```

不依赖本项目也可以裸用 zenoh（任意语言）：连接 `tcp/<服务端>:7447`，订阅 `mocap/hands/frame`，载荷是 application/json。

## 消息格式

一条消息 = 一个完整 NatNet 帧（JSON）：

```json
{
  "schema_version": 1,
  "frame_number": 12345,
  "motive_timestamp": 105.341,
  "publisher_received_time_ns": 1784000000000000000,
  "coordinate_system": "motive_y_up_right_handed",
  "unit": "meter",
  "publisher_dropped_frames": 0,
  "markers": [
    {
      "raw_id": 18, "model_id": 0, "member_id": 18,
      "id_kind": "point_cloud",
      "position": [0.12, 0.94, -0.31],
      "size": 0.0095, "residual_m_per_ray": 0.0002,
      "occluded": false, "point_cloud_solved": true,
      "model_filled": false, "has_model": false, "unlabeled": true,
      "active": false, "established": true, "measurement": false
    }
  ],
  "rigid_bodies": [
    { "id": 1, "position": [0.1, 0.9, -0.3],
      "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
      "mean_error": 0.0004, "tracking_valid": true }
  ]
}
```

### 字段语义

- **坐标系/单位**：Motive 原始右手 **Y-up** 全局坐标，位置为**米**；消费端如需其他习惯（如 Z-up）必须自行变换
- **`id_kind`**：`asset_member`（model_id/member_id 可作稳定身份）、`point_cloud`（raw_id 是当前跟踪段 ID，**遮挡后可能改变**，不能当永久标签）、`active`、`unknown`
- **`occluded` / `model_filled`**：位置可能来自模型填充而非当前帧直接观测
- **`publisher_dropped_frames`**：发布端累计丢帧计数（单调递增，看增量）
- **`frame_number`**：可检测网络/处理丢帧（不连续 = 丢帧）
- **`quaternion_xyzw`**：四元数 [x,y,z,w] 序

## 统计字段含义（subscribe/viewer 状态栏）

| 字段 | 含义 |
|---|---|
| `rate_hz` | 最近采样窗口平均帧率（Motive 满速 120Hz） |
| `missing` | 帧号不连续累计缺口（含发布端重启间隙，属正常） |
| `publisher_dropped` | 发布端丢帧计数（Windows 侧编码/网络压力） |
| `invalid` | 消息格式非法次数 |
| `markers` | 最近一帧 marker 数 |

## 常见问题

**消费者连不上**：确认数据服务端（view/publisher 监听侧）先启动；`--connect-endpoint` 用服务端实际地址。

**看不了数据**：数据服务端必须有人在监听 7447（viewer 或 natnet-zenoh-subscriber）；Windows publisher 在 Motive 开启时自动推流。

**raw_id 变了**：普通被动 marker 遮挡重现后 Motive 可能分配新 ID，属预期行为。

**两机互通**：网络拓扑与故障排查见 [net.md](net.md)。

## Router 模式(推荐,2026-08-07 起)

本机已部署常驻 zenohd Router(`tcp/0.0.0.0:7447`,见 acquisition 项目)。
Router 模式下 viewer 不再 listen,改用 connect:

```bash
pixi run view -- --connect-endpoint tcp/127.0.0.1:7447 --no-relay
# 或直接 pixi run view-connect(若已添加)
```

- 顺序约束解除:发布者/消费者可任意先后启动
- relay 转发不再需要(多跳由 router 解决),加 `--no-relay` 关闭
- Windows publisher 参数不变(`--zenoh-endpoint tcp/169.254.1.0:7447`,对面换为 router)
- ⚠️ 若 Windows 侧使用 build_peer_config(关闭 scouting 的 peer 模式),连 router 可能
  不路由数据——遇此情况将 natnet 的连接改为 client 模式(见 acquisition/README.md 已知局限)
