#!/usr/bin/env bash
# add_mano_beta.sh - 从手部关键点估计并写入每只手的 MANO beta。
#
# 默认只处理【当前日期】目录(/home/current/data/YYYYMMDD)；显式传入文件夹
# 则递归处理指定目录。beta 默认均匀抽取最多 1000 帧；不做逐帧姿态拟合。
#
# 用法:
#   bash add_mano_beta.sh                    # 只处理今天
#   bash add_mano_beta.sh /path/to/date-dir  # 处理指定日期/数据目录
#   bash add_mano_beta.sh -f                 # 强制重算今天
#   SAMPLES=2000 bash add_mano_beta.sh -f
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACQ_DIR="$SCRIPT_DIR/acquisition"
DATA_ROOT="${DATA_ROOT:-/home/current/data}"
SAMPLES="${SAMPLES:-1000}"

FORCE=()
TARGET=""
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=(--force) ;;
        *) TARGET="$arg" ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    TARGET="$DATA_ROOT/$(date +%Y%m%d)"
fi

if [[ ! -d "$TARGET" ]]; then
    echo "[add_mano_beta] 目录不存在: $TARGET" >&2
    exit 1
fi
if ! command -v pixi >/dev/null 2>&1; then
    echo "[add_mano_beta] 未找到 pixi,请先在 mocap 环境安装" >&2
    exit 1
fi
if [[ ! "$SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[add_mano_beta] SAMPLES 必须是正整数: $SAMPLES" >&2
    exit 1
fi


echo "[add_mano_beta] 扫描 $TARGET 下的 .h5 录制"
echo "[add_mano_beta] beta≤$SAMPLES 帧,表面由关键点直接驱动(无逐帧拟合)"
cd "$ACQ_DIR"
exec pixi run mano-beta -- "$TARGET" --samples "$SAMPLES" "${FORCE[@]}"
