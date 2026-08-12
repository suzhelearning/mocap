#!/usr/bin/env bash
# add_mano_beta.sh - 为采集录制补写 MANO 形状参数 beta(离线估计)。
#
# 默认只处理【当前日期】目录(/home/current/data/YYYYMMDD)——每天采集完
# 直接跑一次即可；显式传入文件夹则处理指定目录(递归)。
# 默认均匀抽取整条录制中的 1000 帧，逐帧稳健估计；已有 beta 自动跳过，
# --force 全部高精度重算。可用 SAMPLES 环境变量调整帧数。
#
# 用法:
#   bash add_mano_beta.sh                    # 只处理今天
#   bash add_mano_beta.sh /path/to/date-dir  # 处理指定日期/数据目录(递归)
#   bash add_mano_beta.sh -f                 # 强制高精度重算今天
#   SAMPLES=2000 bash add_mano_beta.sh -f    # 自定义均匀抽样帧数
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
echo "[add_mano_beta] 稳健逐帧段长估计:最多 $SAMPLES 帧(已写 beta 自动跳过)"
cd "$ACQ_DIR"
exec pixi run mano-beta -- "$TARGET" --samples "$SAMPLES" "${FORCE[@]}"
