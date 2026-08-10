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


def test_raw_keyboard_on_idle_called():
    """select 超时(无按键)时 on_idle 被周期调用(主循环 tick 接线点)。"""
    master, slave = pty.openpty()
    idles = []
    stop = threading.Event()
    ready = threading.Event()

    t = threading.Thread(
        target=raw_keyboard, args=(lambda ch: None, stop, slave, ready),
        kwargs={"on_idle": lambda: idles.append(1)}, daemon=True)
    t.start()
    try:
        assert ready.wait(2.0), "raw 模式未就绪"
        deadline = time.time() + 3
        while len(idles) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert len(idles) >= 2        # 50ms 轮询周期,3 秒内应多次触发
        # 按键仍被读取
        os.write(master, b"x")
    finally:
        stop.set()
        t.join(timeout=3)
        os.close(master)
        os.close(slave)


def test_raw_keyboard_on_idle_non_tty():
    """非终端 fd 同样在轮询空隙调用 on_idle(后台运行主循环可用)。"""
    r, w = os.pipe()
    idles = []
    stop = threading.Event()
    t = threading.Thread(
        target=raw_keyboard, args=(lambda ch: None, stop, r),
        kwargs={"on_idle": lambda: idles.append(1)}, daemon=True)
    t.start()
    try:
        deadline = time.time() + 3
        while len(idles) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert len(idles) >= 2
        os.write(w, b"q")             # 管道数据仍被读取
    finally:
        stop.set()
        t.join(timeout=3)
        os.close(r)
        os.close(w)
