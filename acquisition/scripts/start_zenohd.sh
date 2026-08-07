#!/usr/bin/env bash
# start_zenohd.sh — 手动启动 Zenoh Router(zenohd)
# 常驻建议: pixi run start-router 或 install_zenohd_service.sh 装 systemd 服务
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ → acquisition/ → mocap/ → manus/
ZENOHD="${ZENOHD:-$SCRIPT_DIR/../../manus/.pixi/envs/default/bin/zenohd}"

if ss -ltn 2>/dev/null | grep -q ':7447'; then
    echo "[zenohd] 7447 已被监听(可能已有 router 在跑):" >&2
    ss -ltnp 2>/dev/null | grep ':7447' || true
    exit 0
fi

echo "[zenohd] 启动 $ZENOHD (监听 tcp/0.0.0.0:7447)" >&2
exec "$ZENOHD"
