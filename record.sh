#!/usr/bin/env bash
# record.sh — 路线 3/3:数据 web 订阅与收集(acquisition 采集程序)
#
#   web 可视化 :8081 + 「录制控制」按钮；落盘固定为统一 60Hz 物理时间轴。
#   键盘 r/s/d/q 同样可用。退出(q / Ctrl-C)时录制中先丢弃未保存 take。
#
# 用法:
#   bash record.sh --object cylinder         # 只采集一种物体
#   bash record.sh --object hammer cube      # 一次采集多种(必须显式指定)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACQ_DIR="$SCRIPT_DIR/acquisition"

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
cd "$ACQ_DIR"
exec pixi run record -- "$@"
