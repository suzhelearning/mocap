#!/usr/bin/env bash
# 启动本地动捕 viewer。
#
# 无参数: 使用 $HOME/data/$(date +%Y%m%d)
# 指定目录: ./viewer.sh /home/current/data/20260811
# 环境变量:
#   MOCAP_DATA_DIR  日期文件夹的父目录,默认 $HOME/data
#   VIEWER_PORT     HTTP 端口,默认 8082
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_DIR="$SCRIPT_DIR/acquisition/viewer_tool"
DATA_BASE="${MOCAP_DATA_DIR:-$HOME/data}"
PORT="${VIEWER_PORT:-8082}"

usage() {
    cat >&2 <<'EOF'
用法:
  ./viewer.sh                         # 打开今天的 ~/data/YYYYMMDD
  ./viewer.sh /path/to/YYYYMMDD       # 打开指定录制文件夹

环境变量:
  MOCAP_DATA_DIR=/path/to/data         # 覆盖日期文件夹父目录
  VIEWER_PORT=8082                     # 覆盖 HTTP 端口
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 ]]; then
    usage
    exit 2
fi

if [[ $# -eq 1 ]]; then
    ROOT_DIR="$1"
else
    ROOT_DIR="$DATA_BASE/$(date +%Y%m%d)"
fi

ROOT_DIR="$(realpath -m "$ROOT_DIR")"
if [[ ! -d "$ROOT_DIR" ]]; then
    echo "[viewer] 数据目录不存在: $ROOT_DIR" >&2
    echo "[viewer] 可传入其他目录: ./viewer.sh /path/to/data" >&2
    exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
    echo "[viewer] 未找到 pixi,请先安装 pixi" >&2
    exit 1
fi

if [[ ! -d "$VIEWER_DIR" ]]; then
    echo "[viewer] viewer_tool 不存在: $VIEWER_DIR" >&2
    exit 1
fi

echo "[viewer] root: $ROOT_DIR"
echo "[viewer] port: $PORT"
echo "[viewer] adapter: auto"
cd "$VIEWER_DIR"
exec pixi run serve -- --root "$ROOT_DIR" --port "$PORT"
