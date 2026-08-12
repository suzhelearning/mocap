# HDF5 采集格式 2.0

## 目标

`TakeWriter` 从 2.0 起写入可跨语言读取的 `offsets + flat` 可变长度数组。旧文件的
`h5_version=1.0`（h5py vlen 数组）仍可由 `inspect_hdf5.py`、`replay_hdf5.py` 和
`viz_hdf5.py` 读取，但新录制不再生成 v1。

## 根属性

| 属性 | 含义 |
|---|---|
| `h5_version` | 格式版本，当前为字符串 `2.0` |
| `schema_name` | `mocap-acquisition` |
| `schema_layout` | `offsets-flat-v2` |
| `take_id` | 采集段序号 |
| `start_wall_ns` / `end_wall_ns` | 采集端 wall-clock 纳秒时间 |
| `config_yaml` / `effective_config_yaml` | 实际运行配置；保留旧名 `config_yaml` 兼容消费者 |
| `base_config_yaml` | 用户配置原文，未合并 `offset/<user>.yaml` |
| `calibration_yaml` | 若使用标定文件，保存其原文 |
| `rigid_body_names_json` | Motive 刚体 ID 到名称的 JSON 映射 |
| `stream_health_json` | 保存时的接收、缺帧、回调队列和时钟质量快照 |

所有 JSON 属性均为 UTF-8 文本。消费者不得把文件中的路径、链接或 JSON 字段当作
可执行内容。

## 时间基准

- `mocap/t_ubuntu_ns`：mocap 帧到达采集端的时间。
- `mocap/t_aligned_ubuntu_ns`：用 publisher 的 `publisher_received_time_ns` 和到达样本
  在线拟合跨机时钟后得到的时间；仅用于与手部流对齐。
- `hands/<side>/t_ubuntu_ns`：手部帧到达采集端的时间。
- `objects/<name>/t_ubuntu_ns`：该物体刚体随 mocap 帧落盘的到达时间。
- 所有同一组内时间数组必须严格递增，且与该组的帧行一一对应。

`publisher_received_time_ns` 和原始 `motive_timestamp` 始终保留，便于离线复核时钟拟合。

## 可变长度帧布局

`mocap/rigid_bodies` 和可选的 `mocap/markers` 都包含：

- `frame_offsets`: `int64`，长度为帧数 `+ 1`，首值为 `0`，非递减；第 `i` 帧使用
  `[frame_offsets[i], frame_offsets[i+1])`。
- 其余字段是按帧拼接的 flat 数组，长度必须等于 `frame_offsets[-1]`。

刚体字段：

| 字段 | shape | dtype |
|---|---:|---|
| `ids` | `(N,)` | `int32` |
| `positions` | `(N,3)` | `float32` |
| `quaternions_xyzw` | `(N,4)` | `float32` |
| `tracking_valid` | `(N,)` | `uint8` |
| `mean_error` | `(N,)` | `float32` |

marker 字段：

| 字段 | shape | dtype |
|---|---:|---|
| `positions` | `(N,3)` | `float32` |
| `raw_ids` | `(N,)` | `int32` |
| `occluded` | `(N,)` | `uint8` |
| `id_kinds` | `(N,)` | `uint8`，`active=0`、`asset_member=1`、`point_cloud=2`、`unknown=3` |

这种布局允许一帧没有刚体或 marker，也不会产生空 vlen 对象；追加时只扩展 flat 数组
和 offsets。

## 固定形状数据

- `hands/<side>`：`t_ubuntu_ns`、`seq`、`wrist_position(3)`、
  `wrist_quaternion_xyzw(4)`、`nodes_raw(25,3)`、`nodes_global(25,3)`；
  可选 `mano_skeleton(21,3)`（MediaPipe 21 点全局手骨架）与
  `mano_beta(10,)`（MANO 形状参数，由 `pixi run mano-beta` 离线估计写入，
  单位注释见 `mano_beta.py`）。
- `objects/<name>`：`t_ubuntu_ns`、`position(3)`、`quaternion_xyzw(4)`、`tracking_valid`。
- `events`：`t_ubuntu_ns`、`type`（`0=start`、`1=pause`、`2=resume`、
  `3=save`、`4=discard`、`5=quit`）、`note`。
- `hands/<side>` 的 `edges_json` 保存该侧首次收到的骨骼拓扑。

## 检查与回放

```bash
pixi run inspect -- --strict /path/to/take.h5
pixi run replay -- /path/to/take.h5 --target-hz 120 --json replay.jsonl
pixi run viz-h5 -- /path/to/take.h5
```

严格检查会验证 offsets/flat 长度、时间单调性、最大帧间隙、频率、拓扑、手腕根节点
一致性和物体跟踪率。发布前应保存 `inspect --strict` 的退出码为 0 作为质量门。
