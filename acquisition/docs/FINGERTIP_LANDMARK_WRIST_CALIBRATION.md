# 五指桌面地标 wrist offset 标定

本文档说明如何在**不新增 Motive 刚体**的情况下，使用桌面上左右各五个已知点，同时标定：

- `left_back/right_back` 刚体原点到解剖手腕的局部平移；
- 手背刚体局部系到 Manus 手腕局部系的固定旋转；
- 输出 `hands.<side>.wrist_offset` 的 `xyz + yaw/pitch/roll`。

## 1. 坐标系与标定模型

Motive 内部坐标系：

```text
X = forward
Y = up
Z = right
```

Motive Streaming 选择 `Up Axis = Z` 后输出：

```text
X = forward
Y = left
Z = up
```

位置和四元数均由 Motive 完成转换，消费端不重复转换。

Manus raw skeleton 是独立的手套局部系。固定局部轴对齐：

```yaml
axis_transform:
  permutation: [0, 2, 1]
  signs: [1, 1, -1]
```

即：

\[
A d=(d_x,d_z,-d_y)
\]

该矩阵是 `Manus 局部系 → 手腕局部系`，不是 Streaming 世界轴转换。

对于 fingertip \(j\)：

\[
p_j^G = p_B^G + R_B^G\left(o_B + R_{B\leftarrow W}A(p_j^H-p_0^H)\right)
\]

已知：

- \(p_B^G,R_B^G\)：`left_back/right_back` 的 Streaming 世界 pose；
- \(p_j^H\)：Manus fingertip 局部坐标；
- \(p_j^G\)：桌面标记点的已知世界坐标。

待求：

- \(o_B\)：手背刚体原点到解剖手腕的局部平移；
- \(R_{B\leftarrow W}\)：手腕局部系相对手背刚体局部系的旋转。

五个非共线 fingertip 对应点通过 Kabsch 一次求解完整 6DoF。

## 2. Motive 刚体与 Manus 节点

当前 Motive 名字表：

```text
id=1 left_back
id=2 right_back
id=3 tianji_wrist
id=4 hammer
```

标定仅使用 `left_back/right_back`；`hammer/tianji_wrist` 不参与。

Raw Manus fingertip 节点：

```text
thumb  = 24
index  = 5
middle = 10
ring   = 15
little = 20
```

## 3. 十个桌面 marker 的布局与排序

十点按世界 Y 从大到小排序：

```text
left_little
left_ring
left_middle
left_index
left_thumb
right_thumb
right_index
right_middle
right_ring
right_little
```

约定：

```text
+Y 侧 = 左手
-Y 侧 = 右手
桌面接触面 = z=0
```

坐标保存在：

```text
acquisition/config/wrist_landmarks.yaml
```

每点同时保存：

- fingertip 名字和 Manus node；
- 标定使用的桌面接触坐标 `xyz`；
- Motive `point_cloud` 的 `raw_id`；
- marker 球心三秒均值；
- 三轴标准差和样本数。

## 4. 重新读取十点坐标

十个反光 marker 摆好并保持静止后：

```bash
cd ~/syz/mocap/acquisition
pixi run capture-wrist-landmarks
```

脚本执行：

1. 订阅 `mocap/hands/frame`；
2. 连续采集 3 秒；
3. 筛选 `id_kind=point_cloud`、非遮挡、`z<0.02m`；
4. 要求 `+Y/-Y` 恰好各五个点；
5. 按 Y 自动匹配左右手和五指；
6. 检查每点可见率与抖动；
7. 原子写入 `config/wrist_landmarks.yaml`；
8. 旧配置备份为 `wrist_landmarks.yaml.bak`。

只查看、不写文件：

```bash
pixi run capture-wrist-landmarks -- --dry-run
```

自定义采集时间：

```bash
pixi run capture-wrist-landmarks -- --seconds 5
```

若桌面 marker 高度不在默认范围：

```bash
pixi run capture-wrist-landmarks -- --max-marker-z 0.03
```

### marker 球心与接触面的区别

Motive 读取的是反光球**球心**，当前球心约高于桌面 8–10mm。标定使用 fingertip 与桌面的接触点，因此默认输出：

```text
xyz.z = 0
```

推荐：

1. 用脚本记录球心 x/y；
2. 在球心投影处画十字；
3. 移走反光球；
4. fingertip 触碰桌面十字。

若直接触碰球顶，必须传入真实接触高度：

```bash
pixi run capture-wrist-landmarks -- --contact-z <meters>
```

否则 wrist z 会产生固定系统误差。

## 5. 启动数据流

三个终端分别执行：

```bash
cd ~/syz/mocap/acquisition
pixi run start-router
```

```bash
cd ~/syz/mocap
bash windows_pub.sh
```

```bash
cd ~/syz/mocap
bash manus_pub.sh --user shd
```

标定时不需要运行 `record.sh`。

可用以下命令确认流：

```bash
bash zenoh_topics.sh --seconds 5
```

应至少包含：

```text
mocap/hands/frame
mocap/rigid_body_names
manus/raw_skeleton/left_hand
manus/raw_skeleton/right_hand
```

## 6. 执行标定

### 左手

```bash
cd ~/syz/mocap
bash acquisition/scripts/calibrate_wrist_offset.sh left \
  --user shd \
  --back-name left_back \
  --landmarks config/wrist_landmarks.yaml \
  --hold 3 \
  --max-rms-mm 5 \
  --max-direction-deg 15
```

### 右手

```bash
bash acquisition/scripts/calibrate_wrist_offset.sh right \
  --user shd \
  --back-name right_back \
  --landmarks config/wrist_landmarks.yaml \
  --hold 3 \
  --max-rms-mm 5 \
  --max-direction-deg 15
```

操作要求：

1. 五根 fingertip 同时压住对应十字；
2. 使用指尖末端，不要用指腹随意覆盖；
3. 手背刚体不能移动或松动；
4. 稳定后按 Enter；
5. 保持 3 秒，移动会自动要求重采。

## 7. 质量门

五指标定默认质量门：

```text
位置 RMS            <= 5mm（命令指定）
方向最大误差        <= 15°（命令指定）
五指方向覆盖        >= 25°
second span         >= 3mm
LOO 最大位置误差    <= 15mm（地标模式专用）
静止位置漂移        < 2mm
静止姿态漂移        < 1.5°
骨架漂移            < 2mm
```

单 DIP、多姿势模式仍保持 LOO 10mm 门限。

脚本还输出每根手指的拟合残差：

```text
thumb/index/middle/ring/little: ... mm
```

某一根明显偏大时，优先检查该手指是否压错点、侧滑或 fingertip node 与接触点不一致。

质量门失败时不会覆盖旧配置。

## 8. 2026-08-25 实测结果

### 右手

```text
位置 RMS:       2.68mm
方向 RMS/最大:  1.3° / 1.6°
方向覆盖:       35.6°
second span:    53.4mm
LOO:            11.9mm / 2.8°
xyz:            [-0.0854, 0.0132, -0.0499]
yaw/pitch/roll: [-6.93°, 0.24°, 76.33°]
```

五指残差：

```text
thumb  4.78mm
index  4.09mm
middle 5.98mm
ring   4.50mm
little 3.44mm
```

### 左手

```text
位置 RMS:       3.48mm
方向 RMS/最大:  2.0° / 4.2°
方向覆盖:       42.6°
second span:    56.5mm
LOO:            14.8mm / 5.6°
xyz:            [-0.0769, 0.0067, -0.0620]
yaw/pitch/roll: [0.18°, -5.16°, 91.36°]
```

五指残差：

```text
thumb  11.02mm
index   4.30mm
middle  4.31mm
ring    4.30mm
little  2.22mm
```

左手 thumb 残差明显高于其他手指。当前解满足整体质量门，可用于采集；若需要更高精度，建议只重做左手并重点控制 thumb 接触位置。

## 9. 输出与生效

成功后写入：

```text
acquisition/offset/shd.yaml
```

wrapper 自带 `--apply`，同时更新：

```text
acquisition/config.yaml
```

运行中的采集进程不会热加载 offset。标定后必须重启：

```bash
cd ~/syz/mocap
bash record.sh --object hammer tianji_wrist
```

## 10. 标定后验证

实时 viewer 检查：

- Manus root/node 0 与 `wrist_position` 重合；
- 手背转动时手腕中心稳定跟随；
- 五指触碰桌面十字时误差符合预期；
- 左右手不镜像、不翻转；
- 物体与手统一为 X前/Y左/Z上。

可录制短 take 后运行：

```bash
cd ~/syz/mocap/acquisition
pixi run inspect -- --strict /path/to/take.h5
```

注意：`inspect` 检查数据结构、时间轴和有效性，不替代物理接触误差检查。
