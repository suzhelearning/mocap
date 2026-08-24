"""Real-Zenoh TCP loopback integration test for ZenohSource."""

from __future__ import annotations

import socket

import zenoh
from natnet_zenoh.schema import FRAME_KEY, encode_frame
from natnet_zenoh.zenoh_transport import build_peer_config

from mocap_viewer.zenoh_source import ZenohSource


def unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def valid_frame(number: int = 1) -> dict:
    return {
        "schema_version": 1,
        "frame_number": number,
        "motive_timestamp": 0.5,
        "publisher_received_time_ns": 123456789,
        "coordinate_system": "motive_x_forward_z_up_right_handed",
        "unit": "meter",
        "publisher_dropped_frames": 0,
        "markers": [],
        "rigid_bodies": [],
    }


def test_frame_crosses_real_tcp_to_source_queue() -> None:
    port = unused_tcp_port()
    source = ZenohSource(FRAME_KEY, listen_endpoint=f"tcp/127.0.0.1:{port}")
    source.start()
    try:
        config = build_peer_config(connect_endpoint=f"tcp/127.0.0.1:{port}")
        with zenoh.open(config) as session:
            publisher = session.declare_publisher(
                FRAME_KEY, encoding=zenoh.Encoding.APPLICATION_JSON
            )
            publisher.put(encode_frame(valid_frame(99)))
            frame = source.queue.get(timeout=5.0)
            assert frame["frame_number"] == 99
    finally:
        source.stop()
