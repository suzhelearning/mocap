# HDF5 录制格式

## 当前写入格式

新录制只生成：

- `h5_version=4.0`
- `schema_layout=compact-aligned-60hz-v1`
- 单一 60 Hz Linux monotonic 时间轴
- 面向回放、Viewer 和训练的最小对齐数据

v4 不保存 Motive/Manus 原始包、markers、插值来源、时钟拟合、质量统计或手物体派生坐标。需要这些诊断信息时，应在采集进程日志或上游发布端单独记录，不能从 v4 H5 恢复。

## 时间语义

`time_ns` 是采集机 Linux `CLOCK_MONOTONIC` 域内的目标物理时刻：

- shape `(N,)`，dtype `int64`
- 固定 60 Hz
- 严格递增
- 相邻差值只能是 `16,666,666` 或 `16,666,667 ns`

所有逐帧 dataset 的第一维必须等于 `N = len(time_ns)`。源数据只有在包围目标时刻时才插值；位置使用 lerp，四元数使用最短弧 SLERP，禁止外推。无法配齐的目标 tick 仍保留，以 `valid=0` 标记。

`start_wall_ns` 和 `end_wall_ns` 是 wall-clock，仅用于文件命名和审计，不参与物理对齐。

## 根属性

| 属性 | 含义 |
|---|---|
| `h5_version` | `4.0` |
| `schema_name` | `mocap-acquisition` |
| `schema_layout` | `compact-aligned-60hz-v1` |
| `time_domain` | `linux-clock-monotonic` |
| `output_hz` | `60` |
| `take_id` | 进程内 take 序号 |
| `start_wall_ns` / `end_wall_ns` | 录制开始/保存 wall-clock |
| `effective_config_yaml` | 本次运行的有效配置快照 |
| `object_pose_frame` | `obj`、`motive_rigid`、`mixed` 或 `none` |
| `object_offset_config_sha256` | 可选；物体外参配置 SHA-256 |

## 完整 dataset 树

```text
/
├── time_ns                              (N,)       int64
├── valid                                (N,)       uint8
├── hands/
│   ├── left/
│   │   ├── keypoints_world              (N,21,3)   float32
│   │   ├── wrist_position               (N,3)      float32
│   │   ├── wrist_quaternion_xyzw        (N,4)      float32
│   │   ├── valid                        (N,)       uint8
│   │   └── mano_beta                    (10,)      float32  [可选]
│   └── right/
│       ├── keypoints_world              (N,21,3)   float32
│       ├── wrist_position               (N,3)      float32
│       ├── wrist_quaternion_xyzw        (N,4)      float32
│       ├── valid                        (N,)       uint8
│       └── mano_beta                    (10,)      float32  [可选]
├── objects/
│   └── <name>/
│       ├── object_position              (N,3)      float32
│       ├── object_quaternion_xyzw       (N,4)      float32
│       └── valid                        (N,)       uint8
└── events/
    ├── frame_index                      (E,)       int64
    └── type                             (E,)       uint8
```

除可选 `mano_beta` 外，v4 检查器拒绝上述分组中的额外 dataset。

## 全局有效性

`/valid[i]` 表示第 `i` 个 60 Hz tick 是否可直接作为完整训练帧。它综合 Motive、左右手和配置物体的可用性。

每个实体仍保留自己的 `valid`，允许消费者在整帧无效时使用其中有效的部分：

- `hands/<side>/valid[i]`
- `objects/<name>/valid[i]`

有效帧的对应位置和四元数必须为有限值；无效帧允许使用 NaN 占位。

## 手部数据

`keypoints_world` 是世界坐标系中的 MediaPipe/MANO 21 点顺序：

```text
0 wrist
1..4 thumb
5..8 index
9..12 middle
13..16 ring
17..20 pinky
```

契约要求：

```text
keypoints_world[:, 0] == wrist_position
```

`wrist_position` 虽可由第 0 点得到，但按消费接口要求独立保存。`wrist_quaternion_xyzw` 是 Motive 世界系中的腕部朝向，不能由关键点稳定、无歧义地恢复。

可选 `mano_beta(10,)` 是整条 take 共用的手形参数，不带时间维。可在录制后运行：

```bash
pixi run mano-beta -- /path/to/take.h5
```

## 物体数据

每个配置物体分别保存位置和四元数：

- `object_position`: 世界系 OBJ frame 原点
- `object_quaternion_xyzw`: 世界系 OBJ frame 朝向
- `valid`: Motive 追踪和对齐均有效

实时采集使用：

```text
T_world_from_obj = T_world_from_motive_rigid @ T_motive_rigid_from_obj
```

v4 不保存原始 rigid frame 位姿。实际应用的外参由有效配置和 `object_offset_config_sha256` 标识。

## 事件

`events/frame_index` 表示事件之后第一个 aligned frame 的索引，是帧边界而不是 wall-clock：

- `start` 通常为 `0`
- `save` 通常为 `N`
- 合法范围为 `[0, N]`
- 必须单调不减

事件类型：

| 值 | 类型 |
|---:|---|
| 0 | start |
| 1 | save |

保存成功的文件必须含 start 和 save。discard/quit 中止的 take 不会留下目标 H5。

## 不再落盘的派生数据

以下数据不属于 v4 主契约：

- `timeline/frame_index`：逐帧行号可直接作为索引
- `t_emit_ns`、`emission_latency_ns`
- `reason_flags`
- Manus 25 点 local/world 节点和逐节点四元数
- 插值两端序号、gap、alpha
- Motive 原始 rigid pose、mean error、markers
- `interaction/<object>` 手物体坐标
- `quality/`、`raw/`

手物体坐标应由主数据确定性计算：

```text
p_object = R_world_from_object.T @ (p_world - object_position)
interaction_valid = hand_valid & object_valid
```

## 检查、回放与 Viewer

```bash
pixi run inspect -- --strict /path/to/take.h5
pixi run replay -- /path/to/take.h5
pixi run replay -- /path/to/take.h5 --json /tmp/take.jsonl
pixi run viz-h5 -- /path/to/take.h5
```

v4 已是统一 60 Hz，`replay` 拒绝改成其他目标频率。检查器验证精确字段集合、dtype、shape、公共长度、固定节拍、有效值、四元数单位范数、腕部根节点一致性和事件边界。

读取工具继续支持历史 v1/v2/v3 文件；录制器不会再生成旧格式。
