#!/usr/bin/env bash
# zenoh.sh — 便捷启动 Zenoh Router(转发到 acquisition 的 start_zenohd.sh)
#
# 用法:
#   bash zenoh.sh            # 手动启动 zenohd(带 ACL)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/acquisition/scripts/start_zenohd.sh"
