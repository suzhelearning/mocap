"""按流独立的相位累加采样率门控，适配非整数频率与到达抖动。"""

from __future__ import annotations

import threading


class RateGate:
    """将每路输入下采样到目标频率，不让早到帧造成长期欠采样。

    每路累计相邻输入帧贡献的目标周期份额；累计达到 1 才保留一帧。
    这与“距上次保留帧至少一个周期”不同：102Hz 输入降至 100Hz 时，
    早到的 9.8ms 会累计余量，而不会隔帧降成约 51Hz。
    """

    def __init__(self, hz: float) -> None:
        self._hz = max(1.0, float(hz))
        self._lock = threading.Lock()
        self._last_ns: dict[str, int] = {}
        self._phase: dict[str, float] = {}
        self._stats: dict[str, dict[str, int]] = {}

    @property
    def hz(self) -> float:
        with self._lock:
            return self._hz

    def set_hz(self, hz: float) -> None:
        with self._lock:
            self._hz = max(1.0, float(hz))
            # 改频后下一帧作为新相位起点，立即采用新频率。
            self._last_ns.clear()
            self._phase.clear()

    def reset(self) -> None:
        """开始新 take 时重置相位与本 take 统计。"""
        with self._lock:
            self._last_ns.clear()
            self._phase.clear()
            self._stats.clear()

    def should_write(self, t_ns: int, stream: str = "default") -> bool:
        with self._lock:
            stats = self._stats.setdefault(stream, {
                "input": 0,
                "kept": 0,
                "rate_limited": 0,
                "nonmonotonic": 0,
            })
            stats["input"] += 1
            previous = self._last_ns.get(stream)
            if previous is None:
                self._last_ns[stream] = t_ns
                self._phase[stream] = 0.0
                stats["kept"] += 1
                return True
            if t_ns <= previous:
                stats["nonmonotonic"] += 1
                return False
            elapsed = t_ns - previous
            self._last_ns[stream] = t_ns
            phase = min(
                2.0,
                self._phase.get(stream, 0.0) + elapsed * self._hz / 1e9,
            )
            if phase < 1.0:
                self._phase[stream] = phase
                stats["rate_limited"] += 1
                return False
            # 容量 2 的 token bucket 保留轻微早/晚到的相位余量，同时把
            # 长间断后的追赶突发限制为至多一帧。
            self._phase[stream] = phase - 1.0
            stats["kept"] += 1
            return True
    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "target_hz": self._hz,
                "streams": {
                    stream: dict(values)
                    for stream, values in self._stats.items()
                },
            }
