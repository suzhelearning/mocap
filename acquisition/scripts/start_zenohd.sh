#!/usr/bin/env bash
# start_zenohd.sh — 手动启动 Zenoh Router(zenohd)
# 常驻建议: pixi run start-router 或 install_zenohd_service.sh 装 systemd 服务
#
# 安全:带 ACL 配置启动(默认拒绝,仅放行 mocap/** 与 manus/** key 空间);
# 有线 link-local 可用时只监听 169.254.1.0 + 127.0.0.1(WiFi 网段不可达),
# 否则回退默认 0.0.0.0(ACL 仍然生效)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ → acquisition/ → mocap/ → manus/
ZENOHD="${ZENOHD:-$SCRIPT_DIR/../../manus/.pixi/envs/default/bin/zenohd}"
ACL_CONFIG="${ACL_CONFIG:-$SCRIPT_DIR/../config/zenohd_acl.yaml}"

if ! [ -f "$ACL_CONFIG" ]; then
    echo "[zenohd] 错误: ACL 配置不存在: $ACL_CONFIG" >&2
    exit 1
fi

if ss -ltn 2>/dev/null | grep -q ':7447'; then
    echo "[zenohd] 7447 已被监听(可能已有 router 在跑):" >&2
    ss -ltnp 2>/dev/null | grep ':7447' || true
    echo "[zenohd] 提示: 现有实例若未带 ACL 配置,请重启以生效:" >&2
    echo "[zenohd]   systemctl --user restart zenohd (或 kill 后重跑本脚本)" >&2
    exit 0
fi

if ! [ -x "$ZENOHD" ]; then
    echo "[zenohd] 错误: zenohd 不存在: $ZENOHD" >&2
    exit 1
fi

echo "[zenohd] 启动 $ZENOHD (ACL: $ACL_CONFIG)" >&2
if ip -4 addr show 2>/dev/null | grep -q '169\.254\.1\.0'; then
    echo "[zenohd] 监听 tcp/169.254.1.0:7447 + tcp/127.0.0.1:7447 (仅有线直连+本机)" >&2
    exec "$ZENOHD" -c "$ACL_CONFIG" -l tcp/169.254.1.0:7447 -l tcp/127.0.0.1:7447
fi
echo "[zenohd] 监听默认 0.0.0.0:7447 (ACL 生效)" >&2
exec "$ZENOHD" -c "$ACL_CONFIG"
