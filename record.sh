#!/usr/bin/env bash
# record.sh — 路线 3/3:数据 web 订阅与收集(acquisition 采集程序)
#
#   web 可视化 :8081 + 「录制控制」按钮；落盘固定为统一 60Hz 物理时间轴。
#   键盘 r/s/d/q 同样可用。退出(q / Ctrl-C)时录制中先丢弃未保存 take。
#
# 用法(--user 必填):
#   bash record.sh --object hammer --user shd
#   bash record.sh --object hammer tianji_wrist --user syz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACQ_DIR="$SCRIPT_DIR/acquisition"

USER_NAME=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --user)
            if (( i + 1 >= ${#ARGS[@]} )) || [[ "${ARGS[$((i + 1))]}" == --* ]]; then
                echo "[错误] --user 后必须提供用户名,例如:--user shd" >&2
                exit 2
            fi
            USER_NAME="${ARGS[$((i + 1))]}"
            ;;
        --user=*)
            USER_NAME="${ARGS[$i]#--user=}"
            ;;
    esac
done
if [[ -z "$USER_NAME" ]]; then
    echo "[错误] 录制必须指定用户: bash record.sh --object hammer --user <name>" >&2
    echo "[错误] 且 acquisition/offset/<name>.yaml 必须包含 left/right 标定" >&2
    exit 2
fi

if ! ss -ltn 2>/dev/null | grep -q ':7447'; then
    echo "[错误] zenohd Router 未在 7447 监听。请先启动: bash acquisition/scripts/start_zenohd.sh (或 systemctl --user start zenohd)" >&2
    exit 1
fi

# 启动前按端口清理上一实例及任何 8081 监听进程，避免 Viser
# 报 "Address already in use"。VISE_PORT 可覆盖默认端口。
VISE_PORT="${VISE_PORT:-8081}"
bash "$SCRIPT_DIR/clean_record.sh" "$VISE_PORT"

echo "[record] 浏览器打开 http://127.0.0.1:8081 (q 或 Ctrl-C 退出)"
echo "[record] 录制控制见 web 左侧面板;键盘 r/s/d 同样可用"
echo "[record] 操作者: $USER_NAME"
cd "$ACQ_DIR"
exec pixi run record -- "$@"
