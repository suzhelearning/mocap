# 动捕刚体 → 真实手腕位姿转换说明

本文档说明如何把 Motive 动捕刚体（手背 marker 刚体，如 `left_wrist`）的位姿，
转换为真实手腕（腕关节中心）位姿。适用于消费端复现转换逻辑，或理解采集链路输出。

## 输入 / 输出

| | 内容 | 单位/约定 |
|---|---|---|
| 输入 | 刚体 `position` (3,) | 米，Motive 世界系（x 前向、z 向上右手系） |
| 输入 | 刚体 `quaternion_xyzw` (4,) | 四元数，**xyzw 序** |
| 输入 | 标定 `wrist_offset` | `xyz`（米）+ `yaw/pitch/roll_deg` + `mode` |
| 输出 | 手腕 `position` (3,) | 米，Motive 世界系 |
| 输出 | 手腕 `quaternion_xyzw` (4,) | 四元数，xyzw 序（由 wxyz 转回） |

## 转换公式

`mode: body`（当前配置，随刚体转动，物理正确）：

$$
p_w = p_b + R_b \cdot o
\qquad
q_w = q_b \otimes q_{\text{extra}}
\qquad
R_w = R_b \cdot R_{\text{extra}}
$$

`mode: world`（固定世界方向，一般不用）：

$$
p_w = p_b + o
\qquad
q_w = q_{\text{extra}}
$$

符号：

| 符号 | 含义 |
|---|---|
| $p_b$, $q_b$ | 刚体 position / quaternion（wxyz 序内部计算） |
| $R_b$ | 刚体旋转矩阵：`rotmat_from_wxyz(quat_xyzw_to_wxyz(q_b))` |
| $o$ | `wrist_offset.xyz`：手背刚体**本地坐标系** → 手腕的固定平移（标定得到） |
| $q_{\text{extra}}$ | 姿态修正四元数：`euler_wxyz(yaw_deg, pitch_deg, roll_deg)` |
| $R_{\text{extra}}$ | 由 $q_{\text{extra}}$ 转出的 3×3 矩阵（默认单位阵，未标定欧拉角时） |

## 使用的旋转矩阵（回答"用哪个旋转矩阵"）

1. **$R_b$**：由刚体 `quaternion_xyzw` 转换出的旋转矩阵——刚体在 Motive 世界系的姿态。
   位置偏移用它旋转到世界系：`p_w = p_b + R_b @ o`。
2. **$R_{\text{extra}}$**：标定姿态修正矩阵，由 `yaw_deg/pitch_deg/roll_deg` 构造。
   最终手腕姿态：$R_w = R_b \cdot R_{\text{extra}}$。

欧拉角约定（`kinematics.py::euler_wxyz`）：身体系 **ZYX** 欧拉角，
yaw=绕 Z、pitch=绕 Y、roll=绕 X，按 Z→Y→X 复合，单位度。

## 当前标定值（`acquisition/config.yaml`）

✅ **左右手已复核（2026-08-19）**：当前 Motive 名字与物理位置**一致**——
`left_wrist`（id=1）在物理左手、`right_wrist`（id=2）在物理右手
（空间位置验证：id=2 与 `right_arm`（id=10）同侧）。
早期 Motive 里名字曾起反（历史勘误注释已过时），现按名字直接消费即可。

```yaml
hands:
  left:                       # 物理左手（Motive 名 right_wrist, id=1）
    back_rigid_id: 1
    wrist_offset:
      mode: body
      xyz:        [0.0206, -0.025, -0.0967]     # ⚠ 注释标注未标定（占位值）
      yaw_deg:    -135.19
      pitch_deg:  -70.4
      roll_deg:   149.43
  right:                      # 物理右手（Motive 名 left_wrist, id=2）
    back_rigid_id: 2
    wrist_offset:
      mode: body
      xyz:        [0.0378, 0.0235, -0.0923]     # 已标定
      yaw_deg:    -118.13
      pitch_deg:  -42.78
      roll_deg:   97.47
```

若 Motive 直接追踪手腕刚体（配置 `wrist_rigid_id` 而非 `back_rigid_id`），
跳过 offset，直接取刚体位姿作为手腕位姿，不做上述转换。

## 代码位置

| 内容 | 位置 |
|---|---|
| 转换主体 `wrist_pose_from_back` | `acquisition/src/acquisition/stitching.py:27` |
| 实时调用点（每帧执行） | `acquisition/src/acquisition/alignment.py:391` |
| 四元数→旋转矩阵 `rotmat_from_wxyz` | `acquisition/src/acquisition/kinematics.py:52` |
| 欧拉角→四元数 `euler_wxyz` | `acquisition/src/acquisition/kinematics.py:64` |
| 四元数乘法 `quat_mul`（wxyz 序） | `acquisition/src/acquisition/kinematics.py:25` |
| 标定值配置 | `acquisition/config.yaml:22-43` |
| 标定工具（交互式求解写回配置） | `acquisition/scripts/calibrate_wrist_offset.py` |

## 最小复现（numpy，不依赖项目代码）

```python
import numpy as np

def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float)[[3, 0, 1, 2]]

def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float)[[1, 2, 3, 0]]

def rotmat_from_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = a
    wb, xb, yb, zb = b
    return np.array([
        wa*wb - xa*xb - ya*yb - za*zb,
        wa*xb + xa*wb + ya*zb - za*yb,
        wa*yb - xa*zb + ya*wb + za*xb,
        wa*zb + xa*yb - ya*xb + za*wb,
    ])

def euler_wxyz(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """身体系 ZYX 欧拉角 → 四元数(wxyz 序)。"""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])

def back_to_wrist(pos_b, quat_b_xyzw, offset_xyz, yaw_deg, pitch_deg, roll_deg):
    """mode=body: 刚体位姿 → 手腕位姿。返回 (position, quaternion_xyzw)。"""
    q_b = quat_xyzw_to_wxyz(quat_b_xyzw)
    R_b = rotmat_from_wxyz(q_b)
    p_w = np.asarray(pos_b, dtype=float) + R_b @ np.asarray(offset_xyz, dtype=float)
    q_w = quat_mul(q_b, euler_wxyz(yaw_deg, pitch_deg, roll_deg))
    q_w /= np.linalg.norm(q_w)
    return p_w, quat_wxyz_to_xyzw(q_w)
```

## 注意事项

- 四元数协议为 **xyzw 序**（NatNet 线格式）；内部计算统一 wxyz 序，仅在边界转换。
- 坐标系为 Motive **x 前向、z 向上右手系**（2026-08-24 起）、单位米；消费端如需其他习惯自行变换。
- 刚体 `tracking_valid=false` 时位姿不可信，不应做转换使用。
