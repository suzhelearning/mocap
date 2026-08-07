"""keyboard.py 测试:通过 pty 对模拟按键输入。"""

from __future__ import annotations

import os
import pty
import threading
import time

from acquisition.keyboard import raw_keyboard


def test_raw_keyboard_reads_keys():
    master, slave = pty.openpty()
    got = []
    stop = threading.Event()
    ready = threading.Event()

    def on_key(ch):
        got.append(ch)

    t = threading.Thread(target=raw_keyboard,
                         args=(on_key, stop, slave, ready), daemon=True)
    t.start()
    try:
        # 等 setraw 完成再写入:规范模式排队数据在切 raw 时可能被丢弃
        assert ready.wait(2.0), "raw 模式未就绪"
        os.write(master, b"r")
        os.write(master, b"s")
        os.write(master, b"\x20")       # 空格
        deadline = time.time() + 3
        while len(got) < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert got == ["r", "s", " "]
    finally:
        stop.set()
        t.join(timeout=3)
        os.close(master)
        os.close(slave)


def test_raw_keyboard_stop_event():
    master, slave = pty.openpty()
    got = []
    stop = threading.Event()
    t = threading.Thread(
        target=raw_keyboard, args=(lambda ch: got.append(ch), stop, slave), daemon=True
    )
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join(timeout=3)
    assert not t.is_alive()
    os.close(master)
    os.close(slave)


def test_raw_keyboard_non_tty():
    """非终端 fd(管道)降级为阻塞读,不抛 termios 错误。"""
    r, w = os.pipe()
    got = []
    stop = threading.Event()
    t = threading.Thread(
        target=raw_keyboard, args=(lambda ch: got.append(ch), stop, r), daemon=True
    )
    t.start()
    os.write(w, b"rs")
    deadline = time.time() + 3
    while len(got) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert got == ["r", "s"]
    stop.set()
    t.join(timeout=3)
    os.close(r)
    os.close(w)
