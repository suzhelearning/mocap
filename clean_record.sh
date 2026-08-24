#!/usr/bin/env bash
# clean_record.sh — 清理采集可视化端口（默认 8081）
#
# 用法:
#   bash clean_record.sh          # 清理 8081
#   bash clean_record.sh 8082     # 清理指定端口
#
# 警告:按端口清理任何监听进程，不区分进程类型。
set -euo pipefail

PORT="${1:-8081}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "[clean-record] 非法端口: $PORT" >&2
    exit 2
fi

listener_pids() {
    lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u || true
}

PIDS="$(listener_pids)"
if [[ -z "$PIDS" ]]; then
    if ss -ltn 2>/dev/null | grep -qE "[.:]${PORT}[[:space:]]"; then
        # 少数情况下 lsof 看不到 PID，交给 fuser 从内核 socket 表处理。
        echo "[clean-record] 端口 $PORT 正在监听但 lsof 未返回 PID，使用 fuser 清理"
        fuser -k "${PORT}/tcp" 2>/dev/null || true
        sleep 0.5
    else
        echo "[clean-record] 端口 $PORT 已空闲"
        exit 0
    fi
else
    echo "[clean-record] 端口 $PORT 的监听进程:"
    for pid in $PIDS; do
        ps -p "$pid" -o pid=,user=,command= 2>/dev/null || true
    done

    for pid in $PIDS; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    # 最多等待 3 秒优雅退出。
    for _ in $(seq 1 30); do
        [[ -z "$(listener_pids)" ]] && break
        sleep 0.1
    done

    PIDS="$(listener_pids)"
    if [[ -n "$PIDS" ]]; then
        echo "[clean-record] 进程未退出，强制终止: $PIDS"
        for pid in $PIDS; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 0.5
    fi
fi

if ss -ltn 2>/dev/null | grep -qE "[.:]${PORT}[[:space:]]"; then
    echo "[clean-record] 端口 $PORT 仍被占用" >&2
    ss -ltnp 2>/dev/null | grep -E "[.:]${PORT}[[:space:]]" >&2 || true
    exit 1
fi

echo "[clean-record] 端口 $PORT 已释放"
