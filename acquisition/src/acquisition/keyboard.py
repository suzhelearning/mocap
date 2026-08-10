"""termios 原始模式键盘监听。

用法(主循环内,主线程):
    stop = threading.Event()
    raw_keyboard(on_key, stop)
在 stop 前不返回;每个按键回调一次 on_key(chr)。退出时恢复终端。
无按键时以 0.05s 间隔轮询(select 超时),供主循环做周期 flush/刷新。
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty
from collections.abc import Callable

KeyCallback = Callable[[str], None]

POLL_INTERVAL = 0.05


def raw_keyboard(
    on_key: KeyCallback,
    stop_event: threading.Event,
    fd: int | None = None,
    ready_event: threading.Event | None = None,
    on_idle: Callable[[], None] | None = None,
) -> None:
    """阻塞式读取键盘,直到 stop_event 被置位。进入前保存、退出时恢复终端。

    fd 默认 sys.stdin;测试可注入 pty 对端。
    ready_event:setraw 完成后置位——规范模式排队数据在切换 raw 时可能被
    丢弃,调用方(如测试)应先等 ready 再写入。
    on_idle:每次 select 轮询超时(无按键空隙)调用一次——主循环 tick
    (周期 flush / 状态栏 / web 按钮命令消费)经此接线;tty 与非 tty 都生效。
    """
    fd = fd if fd is not None else sys.stdin.fileno()

    # 非终端(后台/管道/EOF)时无法设置 raw 模式:退化为 select 轮询,EOF 即退出
    if not os.isatty(fd):
        while not stop_event.is_set():
            readable, _, _ = select.select([fd], [], [], POLL_INTERVAL)
            if not readable:
                if on_idle is not None:
                    on_idle()
                continue
            try:
                data = os.read(fd, 1024)
            except OSError:
                break
            if not data:
                break
            for b in data:
                on_key(chr(b))
        return

    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if ready_event is not None:
            ready_event.set()
        while not stop_event.is_set():
            readable, _, _ = select.select([fd], [], [], POLL_INTERVAL)
            if not readable:
                if on_idle is not None:
                    on_idle()
                continue
            try:
                data = os.read(fd, 1024)   # 一次读走所有可用字节,避免逐字节竞态
            except OSError:
                break
            if not data:
                break
            for b in data:
                on_key(chr(b))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
