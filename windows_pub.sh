#!/usr/bin/env bash
# windows_pub.sh — 路线 2/3:Windows rigid 发布(Motive 动捕)
#
#   ssh 挂载 natnet_zenoh.publisher(120Hz,client 模式连本机 zenohd Router)。
#   Windows 无 tmux:ssh 会话保持期间 publisher 存活,断开时 Windows
#   OpenSSH job object 自动清理远程进程;退出时另主动 Stop-Process 兜底。
#
# 用法:
#   bash windows_pub.sh          # 前台阻塞, Ctrl-C 停止
#
# 启动时强制清理所有旧 publisher 实例(单实例保证,防双倍数据残留)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- Windows 侧参数(与 net.md / pub_test.bat 一致) --------------------------
# 路径用 %USERPROFILE% 动态拼接(cmd 环境变量,即 ~\Desktop\syz\NatNetSDK)
WIN_SSH_ARGS=(-i "$HOME/.ssh/id_ed25519_windows" -o BatchMode=yes -o ConnectTimeout=6 \
              -l "current robotics" 192.168.110.6)
WIN_PUB_CMD='cd /d "%USERPROFILE%\Desktop\syz\NatNetSDK" && .pixi\envs\windows\python.exe -u -m natnet_zenoh.publisher --motive-server-ip 127.0.0.1 --natnet-client-ip 127.0.0.1 --zenoh-endpoint tcp/169.254.1.0:7447'

# -- PowerShell 辅助:heredoc 构造代码 → UTF-16LE base64 → ssh 执行 -------------
# 为什么:ssh → cmd 会破坏 -Command 内联引号;EncodedCommand 完全绕开解析问题。
win_powershell() {
    local enc
    enc="$(printf '%s' "$1" | iconv -t UTF-16LE | base64 -w0)"
    ssh "${WIN_SSH_ARGS[@]}" "powershell -NoProfile -EncodedCommand $enc"
}

WIN_CHECK_CODE=$(cat <<'PS'
$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*natnet_zenoh.publisher*' } | Select-Object -First 1
if ($p) { [Console]::Write($p.ProcessId) }
PS
)
WIN_STOP_CODE=$(cat <<'PS'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*natnet_zenoh.publisher*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
PS
)

cleanup() {
    echo ""
    echo "[windows] 清理中..."
    # ssh 被 Ctrl-C 杀时连接未正常关闭,Windows 侧 job object 清理可能不执行,
    # 主动 Stop-Process 兜底
    win_powershell "$WIN_STOP_CODE" >/dev/null 2>&1 \
        && echo "[windows] Windows rigid 发布已停止" || true
}
trap cleanup EXIT
trap 'exit 130' INT

if ! ss -ltn 2>/dev/null | grep -q ':7447'; then
    echo "[错误] zenohd Router 未在 7447 监听。请先启动: bash acquisition/scripts/start_zenohd.sh (或 systemctl --user start zenohd)" >&2
    exit 1
fi

echo "[windows] 清理已有 rigid 发布实例(保证单实例)..."
win_powershell "$WIN_STOP_CODE" >/dev/null 2>&1 || true
sleep 1

echo "[windows] SSH 挂载 rigid 发布(会话保持,Ctrl-C 断开自动清理)..."
# publisher 在 Motive streaming 未开/刚关时会 5 秒超时退出 → 循环重连,
# Motive 开启后自动恢复;Ctrl-C 中断 ssh 后退出循环
# (ssh 非 0 退出会触发 set -e,须 || true 吞掉)
while true; do
    ssh "${WIN_SSH_ARGS[@]}" "$WIN_PUB_CMD" || true
    echo "[windows] publisher 已退出(exit $?),5 秒后重连...(Motive streaming 是否开启?)"
    sleep 5
done
echo "[windows] ssh 已断开"
