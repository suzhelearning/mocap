"""Tests for the mocap-viewer CLI and serve lifecycle (no browser, injected fakes)."""

from __future__ import annotations

import threading

import pytest

from natnet_zenoh.schema import FRAME_KEY

from mocap_viewer import main as main_module


def test_cli_defaults() -> None:
    """必须显式指定 listen 或 connect 端点之一。"""
    args = main_module.build_parser().parse_args(["--listen-endpoint", "tcp/0.0.0.0:7447"])
    assert args.listen_endpoint == "tcp/0.0.0.0:7447"
    assert args.connect_endpoint is None
    assert args.key == FRAME_KEY
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.fps == 30.0
    assert args.open is False
    assert args.queue_capacity == 8
    args = main_module.build_parser().parse_args(["--connect-endpoint", "tcp/127.0.0.1:7447"])
    assert args.connect_endpoint == "tcp/127.0.0.1:7447"
    assert args.listen_endpoint is None


def test_cli_requires_endpoint() -> None:
    """既不 listen 也不 connect 时必须拒绝(在 main 入口检查)。"""
    with pytest.raises(SystemExit):
        main_module.main([], zenoh_module=FakeZenohModule(),
                         source_factory=RecordingSource,
                         scene_factory=lambda *a: FakeViewer())


@pytest.mark.parametrize("argv", [["--fps", "0"], ["--fps", "nan"], ["--queue-capacity", "0"]])
def test_cli_rejects_bad_values(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        main_module.build_parser().parse_args(argv)


class FakeViewer:
    """sleep_forever 抛 KeyboardInterrupt，模拟 Ctrl-C。"""

    def __init__(self, server=None) -> None:
        self.server = server or FakeServer()
        self.closed = False

    def sleep_forever(self) -> None:
        raise KeyboardInterrupt

    def close(self) -> None:
        self.closed = True


class FakeServer:
    def __init__(self) -> None:
        self._port = 8080

    def get_port(self) -> int:
        return self._port

    def stop(self) -> None:
        pass


def test_serve_closes_on_keyboard_interrupt() -> None:
    viewer = FakeViewer()
    main_module.serve(viewer)
    assert viewer.closed


class RecordingSource:
    def __init__(self, *args, **kwargs) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeZenohModule:
    pass


def test_main_wires_source_scene_and_serve(capsys) -> None:
    viewer = FakeViewer()

    def scene_factory(source, host, port, fps):
        assert isinstance(source, RecordingSource)
        return viewer

    code = main_module.main(
        ["--port", "8080", "--connect-endpoint", "tcp/127.0.0.1:7447"],
        zenoh_module=FakeZenohModule(),
        source_factory=RecordingSource,
        scene_factory=scene_factory,
    )
    assert code == 0
    assert viewer.closed
    out = capsys.readouterr().out
    assert "NatNet live viewer: http://127.0.0.1:8080" in out


def test_main_returns_three_when_viser_port_busy() -> None:
    def scene_factory(source, host, port, fps):
        raise OSError("address already in use")

    code = main_module.main(
        ["--connect-endpoint", "tcp/127.0.0.1:7447"],
        zenoh_module=FakeZenohModule(),
        source_factory=RecordingSource,
        scene_factory=scene_factory,
    )
    assert code == 3
