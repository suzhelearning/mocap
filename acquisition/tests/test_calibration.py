"""校准文件(.mcal)测试:格式解析 + 不同用户校准输出差异。

核心问题:同一组输入(同一只手/同一测量协议),yq 与 shd 的校准是否不同?
离线层验证:两人生成校准文件中的手型参数(手长/手宽/腕宽/指长/传感器偏移/
轮廓数据)应有显著差异——这些参数直接决定 SDK 输出的骨架形状。
"""

from __future__ import annotations

import pytest

from acquisition.mcal import (
    CALIBRATION_DIR,
    FINGER_KEYS,
    McaError,
    load_user_calibration,
    parse_mcal,
)

USERS = ("yq", "shd", "syz")


@pytest.fixture(scope="module")
def profiles() -> dict[str, dict[str, dict]]:
    """{user: {side: 解析结果}};校准文件缺失时跳过测试。"""
    result: dict[str, dict[str, dict]] = {}
    for user in USERS:
        sides = {}
        for side in ("left", "right"):
            path = CALIBRATION_DIR / f"{user}{'Left' if side == 'left' else 'Right'}MetaglovePro.mcal"
            if not path.is_file():
                pytest.skip(f"校准文件不存在: {path}")
            sides[side] = parse_mcal(path.read_bytes())
        result[user] = sides
    return result


# -- 格式与结构 ------------------------------------------------------------

def test_mcal_format_wellformed(profiles):
    """所有校准文件结构完整:版本、side、手指参数、measurements(旧版可缺)。"""
    for user, sides in profiles.items():
        for side, m in sides.items():
            assert m["tag"] == "MetaglovePro Glove Profile"
            assert m["version"]  # 非空版本
            assert m["side"] == side
            assert len(m["fingers"]) == 5        # 五指齐全
            for f in FINGER_KEYS:
                d = m["fingers"][f]
                assert d["finger_length"] is not None and d["finger_length"] > 0
                rot = d["sensor_rotation_offset"]
                assert rot is not None and len(rot) == 4
                assert all(isinstance(v, (int, float)) for v in rot)
            # 新版(03010100)含 measurements;旧版(03000000)可缺
            if m["version"] == "03010100":
                assert m["measurements"]["hand_length"] is not None
                assert m["measurements"]["hand_width"] is not None


def test_same_user_left_right_differ(profiles):
    """同一用户左右手校准必须不同(左右手尺寸不同)。"""
    for user in USERS:
        l, r = profiles[user]["left"], profiles[user]["right"]
        if l["profile_md5"] is not None:
            assert l["profile_md5"] != r["profile_md5"]
        # 至少一个手型参数不同
        assert any(
            l["fingers"][f]["finger_length"] != r["fingers"][f]["finger_length"]
            for f in FINGER_KEYS
        )


# -- yq vs shd:同一组输入,输出是否不同 ------------------------------------

def test_yq_shd_measurements_differ(profiles):
    """手型尺寸:至少一只手上手宽差 > 5mm 且手长差 > 3mm(实测:左手显著,右手接近)。"""
    per_side = {}
    for side in ("left", "right"):
        yq = profiles["yq"][side]["measurements"]
        shd = profiles["shd"][side]["measurements"]
        per_side[side] = {
            "hand_length_mm": abs(yq["hand_length"] - shd["hand_length"]) * 1000,
            "hand_width_mm": abs(yq["hand_width"] - shd["hand_width"]) * 1000,
            "wrist_width_mm": abs(yq["wrist_width"] - shd["wrist_width"]) * 1000,
        }
    # 左手:yq 手宽 90.8mm vs shd 76.2mm(差 14.6mm)、手长差 7.3mm
    assert per_side["left"]["hand_width_mm"] > 5.0, per_side
    assert per_side["left"]["hand_length_mm"] > 3.0, per_side
    # 右手:两人生理尺寸更接近,但仍要求至少一项 > 1mm 且方向信息保留
    assert max(per_side["right"].values()) > 1.0, per_side


def test_yq_shd_finger_lengths_differ(profiles):
    """至少一只手上存在指长差 > 2mm(实测:左手 index 差 8mm、pinky 差 12mm)。"""
    per_side = {}
    for side in ("left", "right"):
        yq = profiles["yq"][side]["fingers"]
        shd = profiles["shd"][side]["fingers"]
        per_side[side] = {
            f: abs(yq[f]["finger_length"] - shd[f]["finger_length"]) * 1000
            for f in FINGER_KEYS
        }
    assert max(per_side["left"].values()) > 2.0, per_side
    assert max(per_side["right"].values()) > 1.0, per_side


def test_yq_shd_sensor_offsets_differ(profiles):
    """传感器安装偏移/朝向存在差异(L1 范数,至少一手显著)。"""
    per_side = {}
    for side in ("left", "right"):
        yq = profiles["yq"][side]["fingers"]
        shd = profiles["shd"][side]["fingers"]
        per_side[side] = {
            f: sum(abs(a - b) for a, b in
                   zip(yq[f]["sensor_rotation_offset"], shd[f]["sensor_rotation_offset"]))
            for f in FINGER_KEYS
        }
    # 左手 index 的 z 分量 0.243 vs 0.406 → L1 差约 0.17
    assert max(per_side["left"].values()) > 0.05, per_side


def test_yq_shd_profile_blobs_differ(profiles):
    """calibrationProfile(压缩轮廓数据)内容不同。"""
    for side in ("left", "right"):
        yq = profiles["yq"][side]
        shd = profiles["shd"][side]
        assert yq["profile_bytes"] is not None and shd["profile_bytes"] is not None
        assert yq["profile_md5"] != shd["profile_md5"]
        assert yq["profile_bytes"] != shd["profile_bytes"]


# -- 差异报告(供人工查看"有什么不同",pytest -s) ----------------------------

def test_yq_shd_difference_report(profiles, capsys):
    """打印 yq vs shd 逐项差异表(仅信息输出,不参与断言)。"""
    print("\n" + "=" * 78)
    print("yq vs shd 校准差异(左手示例;单位:米)")
    print("=" * 78)
    yq = profiles["yq"]["left"]
    shd = profiles["shd"]["left"]
    for key, label in (("hand_length", "手长"), ("hand_width", "手宽"),
                       ("wrist_width", "腕宽")):
        a, b = yq["measurements"][key], shd["measurements"][key]
        print(f"  {label:6s}: yq={a:.4f}  shd={b:.4f}  差={abs(a-b)*1000:.1f}mm")
    print("  各指 fingerLength:")
    for f in FINGER_KEYS:
        a, b = yq["fingers"][f]["finger_length"], shd["fingers"][f]["finger_length"]
        print(f"    {f:7s}: yq={a:.4f}  shd={b:.4f}  差={abs(a-b)*1000:.1f}mm")
    print("  传感器旋转偏移(L1 范数差,index 示例 z 分量差异最大):")
    for f in FINGER_KEYS:
        a = yq["fingers"][f]["sensor_rotation_offset"]
        b = shd["fingers"][f]["sensor_rotation_offset"]
        diff = sum(abs(x - y) for x, y in zip(a, b))
        print(f"    {f:7s}: L1差={diff:.4f}")
    print(f"  calibrationProfile: yq={len(yq['profile_bytes'])}B  "
          f"shd={len(shd['profile_bytes'])}B  md5 不同: "
          f"{yq['profile_md5'] != shd['profile_md5']}")


# -- --user 选择规则 ---------------------------------------------------------

def test_load_user_calibration_matches_rawviz_rule(profiles):
    """--user 加载规则与 rawviz 一致:calibration/<user>Left/RightMetaglovePro.mcal。"""
    m = load_user_calibration("yq", "left")
    assert m["side"] == "left"
    assert m["measurements"]["hand_length"] == profiles["yq"]["left"]["measurements"]["hand_length"]


def test_load_user_calibration_missing_raises():
    with pytest.raises(McaError, match="校准文件不存在"):
        load_user_calibration("no_such_user_xyz", "left")
