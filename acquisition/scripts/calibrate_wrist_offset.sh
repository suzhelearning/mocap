#!/usr/bin/env bash
# calibrate_wrist_offset.sh — 一键 wrist offset 标定(分段模式)
#
# 流程:启动后按提示依次摆姿势,每个姿势摆好后按 Enter 开始采集
#       (段内自动静止检测,手动了自动重采)→ 全部完成 → 求解 →
#       写入 acquisition/offset/<user>.yaml。
#
# 用法:
#   bash calibrate_wrist_offset.sh left                     # 标定左手(参照 left_dip)
#   bash calibrate_wrist_offset.sh right --user syz         # 标定右手(用户 syz)
#   bash calibrate_wrist_offset.sh --side left              # 等价写法
#   bash calibrate_wrist_offset.sh left --ref-name left_dip --back-name left_wrist
#
# 可选参数(透传标定脚本):
#   --user NAME        操作者(与 manus_pub.sh --user 一致,如 syz/yq/shd)
#                       结果写入 offset/<NAME>.yaml(默认 default)
#   --ref-name NAME    参照刚体名字(默认 left_dip;从 mocap/rigid_body_names 解析)
#   --back-name NAME   手背刚体名字(默认用 config 的 back_rigid_id)
#   --ref-rigid-id ID  参照刚体 id(与 --ref-name 二选一,优先 --ref-name)
#   --poses N          姿势数(默认 5)
#   --hold S           每姿势采集秒数(默认 3)
#   --node N           参照骨架节点索引(默认 9 = 中指 DIP)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIDE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --side)
            SIDE="${2:-}"
            shift 2
            ;;
        left|right)
            SIDE="$1"
            shift
            ;;
        *)
            EXTRA+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$SIDE" ]]; then
    echo "用法: bash calibrate_wrist_offset.sh <left|right> [--poses N] [--hold S] [--ref-rigid-id ID] [--node N]" >&2
    exit 1
fi

cd "$SCRIPT_DIR/.."          # acquisition/ 根:pixi manifest 与 config.yaml 所在
echo "[标定] 一键启动: side=$SIDE (Enter 分段,结果写 offset/<user>.yaml)"
exec pixi run python scripts/calibrate_wrist_offset.py \
    --side "$SIDE" --auto --apply "${EXTRA[@]}"
