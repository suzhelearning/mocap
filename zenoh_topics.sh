#!/usr/bin/env bash
# zenoh_topics.sh — 列出当前 Zenoh 网络有流量的 topic(ros2 topic list 风格)
#
# 用法:
#   bash zenoh_topics.sh            # 观察 8 秒后列出
#   bash zenoh_topics.sh --watch    # 持续刷新
#   bash zenoh_topics.sh --seconds 15
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/acquisition"
exec pixi run python scripts/zenoh_topics.py "$@"
