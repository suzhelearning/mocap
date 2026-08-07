#!/usr/bin/env bash
# install_zenohd_service.sh — 生成并启用 systemd --user 常驻服务(zenohd)
# 用法: bash scripts/install_zenohd_service.sh [--start]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ → acquisition/ → mocap/ → manus/
ZENOHD="${ZENOHD:-$SCRIPT_DIR/../../manus/.pixi/envs/default/bin/zenohd}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/zenohd.service"

mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=Zenoh Router (zenohd) — acquisition 数据总线
After=network-online.target

[Service]
ExecStart=$ZENOHD
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
EOF

echo "[install] 已写入 $UNIT"
systemctl --user daemon-reload
if [[ "${1:-}" == "--start" ]]; then
    systemctl --user enable --now zenohd
    echo "[install] zenohd 已启用并启动"
    systemctl --user status zenohd --no-pager | head -5
fi
