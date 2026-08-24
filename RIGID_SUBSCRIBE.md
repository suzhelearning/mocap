# 刚体位姿订阅接口（Zenoh）

本机（Ubuntu 动捕数据服务）以 Zenoh Router 对外提供 Motive 刚体（rigid body）位姿数据流。
任何消费端连接 `tcp/169.254.1.0:7447` 即可订阅，无需鉴权、无需本仓库代码。

## 连接参数

| 项 | 值 |
|---|---|
| Router 地址 | `tcp/169.254.1.0:7447`（**仅此有线地址**，不使用 WiFi） |
| 连接模式 | `client`（连接到 Router） |
| 网络前提 | 消费端须以有线接入同一 169.254 网段（与 Ubuntu 接同一交换机，或直连；APIPA 自动配地址，无需 DHCP） |
| 权限 | 无（ACL 已放行 `mocap/**`、`manus/**` 的订阅，消费端零配置） |
| 频率 | 120Hz（Motive 满速） |

## 主题（key）

| key | 内容 | 频率 |
|---|---|---|
| `mocap/hands/frame` | 完整动捕帧 JSON，含 `rigid_bodies` 数组 | 120Hz |
| `mocap/rigid_body_names` | `{"names": {"<id>": "<刚体名>"}}`，按名字找刚体 | 5s 周期重发 |

## 帧消息格式（`mocap/hands/frame`）

载荷为 JSON，`rigid_bodies` 数组即刚体位姿：

```json
{
  "schema_version": 1,
  "frame_number": 14151299,
  "motive_timestamp": 105.341,
  "coordinate_system": "motive_x_forward_z_up_right_handed",
  "unit": "meter",
  "publisher_dropped_frames": 0,
  "rigid_bodies": [
    {
      "id": 10,
      "position": [-0.3062, 0.1054, 0.1923],
      "quaternion_xyzw": [-0.0019, 0.0016, -0.0002, 0.9999],
      "mean_error": 0.0002,
      "tracking_valid": true
    }
  ]
}
```

### `rigid_bodies` 字段语义

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | int | 刚体 ID，与 `mocap/rigid_body_names` 的 key 对应 |
| `position` | [float×3] | 位置，单位**米** |
| `quaternion_xyzw` | [float×4] | 姿态四元数，**顺序 [x, y, z, w]** |
| `mean_error` | float | 该刚体当前平均残差（越小越可信） |
| `tracking_valid` | bool | 当前帧跟踪是否有效，false 时位姿不可用 |

### 坐标系

Motive 全局 **x 前向、z 向上**右手系（`motive_x_forward_z_up_right_handed`），消费端如需其他习惯必须自行变换。

### 丢帧检测

- `frame_number` 不连续 = 网络/处理丢帧
- `publisher_dropped_frames`：发布端累计丢帧计数（单调递增，看增量）

## 最小订阅示例（Python）

```python
import json
import zenoh

ROUTER = "tcp/169.254.1.0:7447"
FRAME_KEY = "mocap/hands/frame"
NAMES_KEY = "mocap/rigid_body_names"

conf = zenoh.Config.from_json5(
    json.dumps({"mode": "client", "connect": {"endpoints": [ROUTER]}})
)

def on_frame(sample: zenoh.Sample) -> None:
    frame = json.loads(sample.payload.to_string())
    for rb in frame.get("rigid_bodies", []):
        if rb["tracking_valid"]:
            print(rb["id"], rb["position"], rb["quaternion_xyzw"])

def on_names(sample: zenoh.Sample) -> None:
    print(json.loads(sample.payload.to_string())["names"])

with zenoh.open(conf) as session:
    session.declare_subscriber(FRAME_KEY, on_frame)
    session.declare_subscriber(NAMES_KEY, on_names)
    input("订阅中，回车退出\n")
```

其他语言（C/C++/Rust/Go 等）：zenoh 官方 SDK 同样 connect `tcp/169.254.1.0:7447`，订阅同一 key，载荷为 JSON 字符串。

## 注意事项

- 数据**无加密**：ACL 只限制 key 空间、不校验身份，同网段主机理论上可订阅。
- 刚体遮挡时 `tracking_valid=false`，其位姿不应使用。
- 消费端数量不限，均可同时订阅同一 key。
