"""把发布端 wall-clock 映射到采集端时钟，估计偏移、漂移与网络抖动。"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class ClockAligner:
    """从 ``(publisher_time_ns, arrival_time_ns)`` 在线拟合时钟映射。

    到达时间包含非负网络延迟。拟合先估计趋势，再只用残差最低四分位
    拟合下包络，因此得到“时钟偏移 + 最小链路延迟”，避免普通均值把
    网络抖动误当成时钟变化。未收集足够样本时退化为已见最小偏移。
    """

    def __init__(self, *, max_samples: int = 600, refit_every: int = 30) -> None:
        self._samples: deque[tuple[int, int]] = deque(maxlen=max_samples)
        self._refit_every = refit_every
        self._lock = threading.Lock()
        self._source_ref_ns = 0
        self._offset_ref_ns = 0.0
        self._drift = 0.0
        self._jitter_p95_ms = 0.0
        self._since_fit = 0
        self._last_source_ns: int | None = None
        self._last_arrival_ns: int | None = None
        self._resets = 0
        self._last_aligned_ns: int | None = None

    def observe(self, source_ns: int, arrival_ns: int) -> int:
        """加入一个观测并返回映射到采集端时钟的源时间。"""
        if source_ns <= 0 or arrival_ns <= 0:
            return arrival_ns
        with self._lock:
            if self._clock_discontinuity(source_ns, arrival_ns):
                self._reset_samples()
                self._resets += 1
            self._last_source_ns = source_ns
            self._last_arrival_ns = arrival_ns
            self._samples.append((source_ns, arrival_ns))
            self._since_fit += 1

            if len(self._samples) == 1:
                self._source_ref_ns = source_ns
                self._offset_ref_ns = float(arrival_ns - source_ns)
            elif len(self._samples) < 30:
                self._offset_ref_ns = float(min(
                    arrival - source for source, arrival in self._samples))
            elif self._since_fit >= self._refit_every:
                self._fit()

            delta = source_ns - self._source_ref_ns
            aligned = int(round(
                source_ns + self._offset_ref_ns + self._drift * delta))
            aligned = min(aligned, arrival_ns)
            if self._last_aligned_ns is not None:
                aligned = max(aligned, self._last_aligned_ns + 1)
            self._last_aligned_ns = aligned
            return aligned
    def quality(self) -> dict[str, object]:
        with self._lock:
            latest_source = self._samples[-1][0] if self._samples else 0
            offset_ns = self._offset_ref_ns + self._drift * (
                latest_source - self._source_ref_ns)
            return {
                "valid": len(self._samples) >= 30,
                "sample_count": len(self._samples),
                "offset_ms": offset_ns / 1e6,
                "drift_ppm": self._drift * 1e6,
                "jitter_p95_ms": self._jitter_p95_ms,
                "resets": self._resets,
            }

    def _clock_discontinuity(self, source_ns: int, arrival_ns: int) -> bool:
        if self._last_source_ns is None or self._last_arrival_ns is None:
            return False
        source_delta = source_ns - self._last_source_ns
        arrival_delta = arrival_ns - self._last_arrival_ns
        return source_delta <= 0 or arrival_delta <= 0 or abs(
            source_delta - arrival_delta) > 5_000_000_000

    def _reset_samples(self) -> None:
        self._samples.clear()
        self._source_ref_ns = 0
        self._offset_ref_ns = 0.0
        self._drift = 0.0
        self._jitter_p95_ms = 0.0
        self._since_fit = 0
        self._last_aligned_ns = None

    def _fit(self) -> None:
        samples = np.asarray(self._samples, dtype=np.int64)
        source_ref = int(samples[0, 0])
        x_s = (samples[:, 0] - source_ref).astype(np.float64) / 1e9
        offset_ms = (samples[:, 1] - samples[:, 0]).astype(np.float64) / 1e6
        if x_s[-1] - x_s[0] < 1.0:
            self._source_ref_ns = source_ref
            self._offset_ref_ns = float(offset_ms.min() * 1e6)
            self._since_fit = 0
            return

        # 时间分箱后各取最早到达样本，直接拟合网络延迟的下包络。
        # 这不会因某类延迟在时间轴上的分布不均而误估时钟漂移。
        bin_count = max(5, min(20, len(samples) // 20))
        lower_x: list[float] = []
        lower_offset: list[float] = []
        for indices in np.array_split(np.arange(len(samples)), bin_count):
            local = int(indices[np.argmin(offset_ms[indices])])
            lower_x.append(float(x_s[local]))
            lower_offset.append(float(offset_ms[local]))
        slope_ms_s, intercept_ms = np.polyfit(lower_x, lower_offset, 1)
        # 超过 500ppm 更可能是网络路径变化或时钟跳变，禁止污染映射。
        drift = float(np.clip(slope_ms_s / 1000.0, -500e-6, 500e-6))
        predicted_ms = intercept_ms + drift * 1000.0 * x_s
        delay_residual_ms = offset_ms - predicted_ms

        self._source_ref_ns = source_ref
        self._offset_ref_ns = float(intercept_ms * 1e6)
        self._drift = drift
        self._jitter_p95_ms = float(np.quantile(
            np.maximum(delay_residual_ms, 0.0), 0.95))
        self._since_fit = 0
