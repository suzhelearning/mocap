"""MetaglovePro 校准文件(.mcal)解析:格式、版本与关键校准参数提取。

.mcal 文件结构(实测):
  ~<hex("MetaglovePro Glove Profile")>~  +  版本 4 字节  +  JSON 对象
  JSON 内含:
    - calibrationProfile: gzip(base64) 压缩的轮廓数据(600KB 级;旧版可能缺失)
    - fingers(index/middle/ring/pinky/thumb): cmcPosition / mcpPosition /
      fingerLength / proportions / sensorRotationOffset / sensorYOffset / sensorZOffset
    - measurements: handLength / handWidth / wristWidth / thumbMeasurements
    - wristPosition / wristRotationOffset / side

用途:不同用户(calibration/<user>Left/RightMetaglovePro.mcal)的手型校准参数
不同,rawviz --user <名> 选择加载;本模块提供可测试的解析与差异比较。
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

TAG = "MetaglovePro Glove Profile"
CALIBRATION_DIR = Path(__file__).resolve().parents[3] / "manus" / "calibration"

FINGER_KEYS = ("index", "middle", "ring", "pinky", "thumb")
MEASUREMENT_KEYS = ("handLength", "handWidth", "wristWidth")


class McaError(ValueError):
    """.mcal 文件格式不合法。"""


def parse_mcal(data: bytes) -> dict:
    """解析一份 .mcal 字节流,返回结构化校准参数。

    返回字段:
      tag / version(hex) / side
      fingers: {指: {finger_length, cmc_position, mcp_position,
                     sensor_rotation_offset, sensor_y_offset, sensor_z_offset}}
      measurements: {hand_length, hand_width, wrist_width, thumb_*}
      wrist_position / wrist_rotation_offset
      profile_bytes / profile_md5(calibrationProfile gzip 解压后,缺失为 None)
    """
    if not data.startswith(b"~"):
        raise McaError("缺少 ~ 头部标记")
    try:
        tag_end = data.index(b"~", 1)
        tag = bytes.fromhex(data[1:tag_end].decode()).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise McaError(f"头部标记解析失败: {exc}") from exc
    if tag != TAG:
        raise McaError(f"非 MetaglovePro 校准文件: tag={tag!r}")

    rest = data[tag_end + 1:]
    if len(rest) < 4:
        raise McaError("版本字段缺失")
    version = rest[:4].hex()
    body = rest[4:]

    # 定位完整 JSON(花括号配对,兼容字符串内括号)
    depth = 0
    in_str = False
    esc = False
    json_end = None
    for i, ch in enumerate(body):
        c = chr(ch)
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break
    if json_end is None:
        raise McaError("JSON 未闭合")
    try:
        obj = json.loads(body[:json_end])
    except json.JSONDecodeError as exc:
        raise McaError(f"JSON 解析失败: {exc}") from exc

    fingers: dict[str, dict] = {}
    for key in FINGER_KEYS:
        f = obj.get(key)
        if not isinstance(f, dict):
            continue
        rot = f.get("sensorRotationOffset")
        # JSON 内为 dict {w,x,y,z},统一规范为 [w,x,y,z] 便于数值比较
        if isinstance(rot, dict):
            rot = [rot.get("w"), rot.get("x"), rot.get("y"), rot.get("z")]
        fingers[key] = {
            "finger_length": f.get("fingerLength"),
            "cmc_position": f.get("cmcPosition"),
            "mcp_position": f.get("mcpPosition"),
            "sensor_rotation_offset": rot,
            "sensor_y_offset": f.get("sensorYOffset"),
            "sensor_z_offset": f.get("sensorZOffset"),
        }

    measurements = obj.get("measurements", {})
    profile = obj.get("calibrationProfile")
    profile_bytes = None
    profile_md5 = None
    if isinstance(profile, str):
        try:
            profile_bytes = gzip.decompress(base64.b64decode(profile))
            profile_md5 = hashlib.md5(profile_bytes).hexdigest()
        except (ValueError, OSError) as exc:
            raise McaError(f"calibrationProfile 解压失败: {exc}") from exc

    return {
        "tag": tag,
        "version": version,
        "side": obj.get("side"),
        "fingers": fingers,
        "measurements": {
            "hand_length": measurements.get("handLength"),
            "hand_width": measurements.get("handWidth"),
            "wrist_width": measurements.get("wristWidth"),
            "thumb": measurements.get("thumbMeasurements"),
        },
        "wrist_position": obj.get("wristPosition"),
        "wrist_rotation_offset": obj.get("wristRotationOffset"),
        "profile_bytes": profile_bytes,
        "profile_md5": profile_md5,
    }


def load_user_calibration(user: str, side: str) -> dict:
    """按 --user 命名规则加载校准:calibration/<user>Left/RightMetaglovePro.mcal。"""
    side_name = "Left" if side == "left" else "Right"
    path = CALIBRATION_DIR / f"{user}{side_name}MetaglovePro.mcal"
    if not path.is_file():
        raise McaError(f"用户 {user!r} 的校准文件不存在: {path}")
    return parse_mcal(path.read_bytes())
