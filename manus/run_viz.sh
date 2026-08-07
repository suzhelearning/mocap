#!/usr/bin/env bash
# 一键启动 Manus raw 骨架 25 关键点可视化
# 用法: ./run_viz.sh [viz.py 参数，如 --port 8090]
set -euo pipefail
cd "$(dirname "$0")"
exec ./rawviz.out | PYTHONUNBUFFERED=1 "$PWD/.pixi/envs/default/bin/python" viz.py "$@"
