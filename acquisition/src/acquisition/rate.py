"""RateGate — 采集落盘频率门控(线程安全,按流分槽)。

输入流(动捕/手套均为 120Hz)在写入端按目标频率下采样:
should_write(stream) 以接收端墙钟判断该流距上次写入是否 >= 1/hz 秒。

规则:
- 目标频率可运行时修改(set_hz),即时生效(web 下拉控件)
- 流速率低于目标时不丢帧(全写,实际频率是多少算多少)
- 被门控丢弃的帧不推迟 last 时间戳,避免实际频率低于目标
- 按流分槽:每路流(动捕/左手/右手)独立维护写窗口,互不挤占,
  保证每路流各自的落盘频率都达到目标值(而非三路共享一个节拍)
"""

from __future__ import annotations

import threading
import time


class RateGate:
    def __init__(self, hz: float = 100.0) -> None:
        self._hz = max(1.0, float(hz))
        self._last_ns: dict[str, int] = {}
        self._lock = threading.Lock()

    def set_hz(self, hz: float) -> None:
        """运行时修改目标频率。"""
        with self._lock:
            self._hz = max(1.0, float(hz))

    @property
    def hz(self) -> float:
        with self._lock:
            return self._hz

    def should_write(self, now_ns: int | None = None, *, stream: str = "default") -> bool:
        """该流该帧应写入返回 True;last 推进"写满窗口"而非帧时间戳。

        帧时间戳语义会导致 120Hz 流 + 100Hz 目标实际降为 60Hz(每次写后
        下一帧距该帧仅 8.3ms);窗口推进语义保证平均写入率 == 目标频率。
        每路流独立窗口(stream 关键字参数,recorder 按流传入),互不影响。
        now_ns 默认取当前墙钟,测试可注入固定时间戳(位置参数,兼容旧调用)。
        """
        now = time.time_ns() if now_ns is None else now_ns
        with self._lock:
            period = 1e9 / self._hz
            last = self._last_ns.get(stream, 0)
            if now - last >= period:
                skip = max(1, int((now - last) // period))
                self._last_ns[stream] = last + int(period) * skip
                return True
            return False

    def reset(self) -> None:
        """清空全部流窗口(新 take 开始第一帧立即写入)。"""
        with self._lock:
            self._last_ns.clear()
