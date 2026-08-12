"""跨机 wall-clock 偏移、漂移与抖动下包络估计测试。"""

from __future__ import annotations

import pytest

from acquisition.clock_sync import ClockAligner


def test_aligner_recovers_offset_and_drift_under_network_jitter():
    aligner = ClockAligner(refit_every=30)
    source_start = 1_800_000_000_000_000_000
    offset_ns = -3_200_000_000
    drift = 35e-6
    delays_ns = (200_000, 350_000, 500_000, 2_500_000, 900_000)
    aligned = 0
    expected = 0
    for index in range(600):
        source = source_start + index * 8_333_333
        expected = int(source + offset_ns + drift * (source - source_start))
        arrival = expected + delays_ns[index % len(delays_ns)]
        aligned = aligner.observe(source, arrival)

    quality = aligner.quality()
    assert quality["valid"] is True
    assert quality["sample_count"] == 600
    assert quality["drift_ppm"] == pytest.approx(35.0, abs=3.0)
    assert abs(aligned - (expected + min(delays_ns))) < 300_000
    assert 0.0 <= quality["jitter_p95_ms"] < 3.0


def test_aligner_resets_on_source_clock_reversal():
    aligner = ClockAligner()
    aligner.observe(10_000, 20_000)
    aligner.observe(20_000, 30_000)
    aligned = aligner.observe(5_000, 40_000)

    quality = aligner.quality()
    assert aligned == 40_000
    assert quality["resets"] == 1
    assert quality["sample_count"] == 1


def test_invalid_source_timestamp_falls_back_to_arrival():
    aligner = ClockAligner()
    assert aligner.observe(0, 123_456) == 123_456
    assert aligner.quality()["sample_count"] == 0
