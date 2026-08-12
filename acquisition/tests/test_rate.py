"""RateGate 频率门控测试:下采样率、动态改频、低频流不丢帧。"""

from __future__ import annotations

import pytest

from acquisition.rate import RateGate


def _stream(times: list[int], gate: RateGate) -> list[int]:
    """按给定时间戳序列过门控,返回被写入的时间戳。"""
    return [t for t in times if gate.should_write(t)]


def _ticks(hz: float, seconds: float) -> list[int]:
    """hz 固定频率流,持续 seconds 秒,返回墙钟(ns)时间戳序列。"""
    interval = int(1e9 / hz)
    return [i * interval for i in range(int(hz * seconds))]


def test_120hz_stream_downsampled_to_100hz():
    """120Hz 输入流 + 100Hz 目标 → 平均写入率 ≈100Hz(10s 内约 1000 帧)。"""
    gate = RateGate(100.0)
    written = _stream(_ticks(120.0, 10.0), gate)
    # 100Hz 目标下 10 秒应写约 1000 帧,容差 ±5%(时间对齐抖动)
    assert len(written) == pytest.approx(1000, abs=50)


def test_100hz_stream_passes_unchanged():
    """100Hz 输入流 + 100Hz 目标 → 首帧起全部写入。"""
    gate = RateGate(100.0)
    ticks = _ticks(100.0, 5.0)
    written = _stream(ticks, gate)
    assert len(written) == len(ticks)


def test_change_freq_takes_effect_immediately():
    """运行中 set_hz 即时生效:60Hz 目标 → 之后按 ~60Hz 门控。"""
    gate = RateGate(100.0)
    _stream(_ticks(120.0, 2.0), gate)          # 前 2 秒 100Hz 写入
    gate.set_hz(60.0)
    tail = _ticks(120.0, 5.0)
    offset = tail[-1] + int(1e9 / 120.0)       # 时间轴接续前流末尾
    written = _stream([t + offset for t in tail], gate)   # 后 5 秒按 60Hz
    assert len(written) == pytest.approx(300, abs=25)


def test_low_rate_stream_never_dropped():
    """30Hz 输入流 + 100Hz 目标 → 全部写入。"""
    gate = RateGate(100.0)
    ticks = _ticks(30.0, 4.0)
    written = _stream(ticks, gate)
    assert len(written) == len(ticks)


def test_dropped_frames_do_not_advance_window():
    """同刻突发 50 帧只写 1 帧;被丢弃帧不推迟写满窗口,窗口重新打开后可写。"""
    gate = RateGate(100.0)
    t0 = 51_000_000                    # 距 last=0 超过一个周期,首帧可写
    burst = [t0] * 50                 # 同一时刻 50 帧 → 只写 1 帧
    assert sum(gate.should_write(t) for t in burst) == 1
    period = int(1e9 / 100.0)
    assert not gate.should_write(t0 + period // 2)   # 窗口未打开
    assert gate.should_write(t0 + period)            # 窗口重新打开


def test_hz_clamped_to_minimum():
    """非法频率钳制到 >=1Hz,不抛错。"""
    gate = RateGate(0.0)
    assert gate.hz == 1.0
    gate.set_hz(-5.0)
    assert gate.hz == 1.0


def test_reset_forces_next_frame_write():
    """reset 清空 last:新 take 开始第一帧立即写入(任意真实墙钟)。"""
    gate = RateGate(100.0)
    _stream(_ticks(120.0, 1.0), gate)
    gate.reset()
    assert gate.should_write(1_000_000_000)


def test_streams_have_independent_windows():
    """三路流(动捕/左右手)各自独立写窗口:互不挤占,每流都达目标频率。

    这是修复前的核心缺陷:单一共享窗口导致三流合计被限到目标频率,
    每流实际只有 1/3。
    """
    gate = RateGate(100.0)
    t0 = 1_000_000_000
    period = int(1e9 / 100.0)
    # 同一时刻三路流同时到达:每流都应写入(各自窗口从 0 开始)
    assert gate.should_write(t0, stream="mocap")
    assert gate.should_write(t0, stream="left")
    assert gate.should_write(t0, stream="right")
    # 一个周期内同流再写被拒,但其它流不受影响
    assert not gate.should_write(t0 + period // 2, stream="mocap")
    assert not gate.should_write(t0 + period // 2, stream="left")
    # 周期到达后各自独立重新打开
    assert gate.should_write(t0 + period, stream="mocap")
    assert gate.should_write(t0 + period, stream="right")


def test_default_stream_compat():
    """不带 stream 参数的行为保持旧语义(测试/单流场景兼容)。"""
    gate = RateGate(100.0)
    t0 = 51_000_000
    assert gate.should_write(t0)
    assert not gate.should_write(t0 + int(1e9 / 100.0) // 2)


def test_irregular_102hz_input_does_not_collapse_to_half_rate():
    gate = RateGate(100.0)
    pattern = (9_000_000, 10_300_000, 9_500_000, 10_180_000)
    times = [0]
    for index in range(1, 2300):
        times.append(times[-1] + pattern[index % len(pattern)])
    written = _stream(times, gate)
    output_hz = (len(written) - 1) / ((times[-1] - times[0]) / 1e9)
    assert output_hz == pytest.approx(100.0, abs=1.0)
    assert len(written) / len(times) > 0.95


def test_stats_distinguish_rate_limit_and_bad_timestamps():
    gate = RateGate(100.0)
    assert gate.should_write(1_000_000_000, stream="left")
    assert not gate.should_write(1_005_000_000, stream="left")
    assert not gate.should_write(1_005_000_000, stream="left")
    stats = gate.stats()["streams"]["left"]
    assert stats == {
        "input": 3,
        "kept": 1,
        "rate_limited": 1,
        "nonmonotonic": 1,
    }
