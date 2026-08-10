#!/usr/bin/env bash
# manus_pub.sh — 路线 1/3:Manus 手套发布
#
#   真实手套: rawviz.out | zenoh_pub.py(120Hz,左右手独立流)
#
# 用法:
#   bash manus_pub.sh                     # 使用无前缀校准文件(不存在则跳过校准)
#   bash manus_pub.sh --user syz          # 使用 calibration/syzLeft/RightMetaglovePro.mcal
#
# 前台运行,Ctrl-C 停止。dongle 独占:其他 manus 进程(如 wuji 的
# manus_data_publisher)须先停,否则 rawviz 报 license 错误。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANUS_DIR="$SCRIPT_DIR/manus"

USER_NAME=""
if [[ "${1:-}" == "--user" ]]; then
    USER_NAME="${2:-}"
    if [[ -z "$USER_NAME" ]]; then
        echo "[错误] --user 后需跟用户名,如: bash manus_pub.sh --user syz" >&2
        exit 1
    fi
elif [[ -n "${1:-}" ]]; then
    echo "[错误] 未知参数: $1(仅支持 --user <用户名>)" >&2
    exit 1
fi

if ! ss -ltn 2>/dev/null | grep -q ':7447'; then
    echo "[错误] zenohd Router 未在 7447 监听。请先启动: bash acquisition/scripts/start_zenohd.sh (或 systemctl --user start zenohd)" >&2
    exit 1
fi

echo "[manus] 真实手套发布(rawviz.out | zenoh_pub.py,120Hz)→ tcp/127.0.0.1:7447"
if [[ -n "$USER_NAME" ]]; then
    echo "[manus] 校准用户: $USER_NAME (calibration/${USER_NAME}Left/RightMetaglovePro.mcal)"
fi
# 退出兜底:杀管道子进程(rawviz/zenoh_pub 可能成孤儿,占着 dongle)。
# 管道须后台运行 + wait:bash 在 wait 中收到信号会立即执行 trap
# (直接前台管道时 trap 被推迟到子进程退出,而子进程收不到信号 → 死锁)。
# 用 pkill -P $$ 精确杀子进程,不用 kill 0(进程组语义在后台 & 下不可靠)
trap 'pkill -TERM -P $$ 2>/dev/null || true' EXIT INT TERM
cd "$MANUS_DIR"
if [[ -n "$USER_NAME" ]]; then
    ./rawviz.out --user "$USER_NAME" | .pixi/envs/default/bin/python zenoh_pub.py --router tcp/localhost:7447 &
else
    ./rawviz.out | .pixi/envs/default/bin/python zenoh_pub.py --router tcp/localhost:7447 &
fi
wait
