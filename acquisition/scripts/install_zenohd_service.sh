#!/usr/bin/env bash
# install_zenohd_service.sh — 生成并启用 systemd --user 常驻服务(zenohd,带 ACL)
# 用法: bash scripts/install_zenohd_service.sh [--start]
#
# ACL 配置: acquisition/config/zenohd_acl.yaml(默认拒绝,仅放行 mocap/** 与 manus/**)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ → acquisition/ → mocap/ → manus/
ZENOHD="${ZENOHD:-$SCRIPT_DIR/../../manus/.pixi/envs/default/bin/zenohd}"
ACL_CONFIG="${ACL_CONFIG:-$SCRIPT_DIR/../config/zenohd_acl.yaml}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/zenohd.service"

if ! [ -f "$ACL_CONFIG" ]; then
    echo "[install] 错误: ACL 配置不存在: $ACL_CONFIG" >&2
    exit 1
fi
if ! [ -x "$ZENOHD" ]; then
    echo "[install] 错误: zenohd 不存在: $ZENOHD" >&2
    exit 1
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=Zenoh Router (zenohd) — acquisition 数据总线
After=network-online.target

[Service]
ExecStart=$ZENOHD -c $ACL_CONFIG
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
EOF

echo "[install] 已写入 $UNIT"
systemctl --user daemon-reload
if [[ "${1:-}" == "--start" ]]; then
    systemctl --user enable --now zenohd || true
    # 老实例(可能无 ACL)直接重启,确保新配置生效
    systemctl --user restart zenohd
    echo "[install] zenohd 已启用并重启(带 ACL 配置)"
    systemctl --user status zenohd --no-pager | head -5
fi
