"""Publish synthetic NatNet frames over Zenoh for local verification without Motive.

用法：先起接收端（pixi run view，监听 7447），再运行本脚本（connect 127.0.0.1:7447）。
帧内容：8 个绕原点旋转的 point_cloud 轨道点 + 1 active + 1 asset_member + 1 unknown
+ 1 个 occluded 点 + 1 个绕 y 轴旋转的 rigid body，60Hz。
"""

from __future__ import annotations

import math
import time

import numpy as np

import zenoh
from natnet_zenoh.schema import FRAME_KEY, encode_frame
from natnet_zenoh.zenoh_transport import build_peer_config

FPS = 60.0


def make_marker(raw_id: int, position, id_kind: str, occluded: bool = False) -> dict:
    return {
        "raw_id": raw_id,
        "model_id": 0,
        "member_id": raw_id,
        "id_kind": id_kind,
        "position": [float(v) for v in position],
        "size": 0.009,
        "residual_m_per_ray": 0.0002,
        "occluded": occluded,
        "point_cloud_solved": True,
        "model_filled": False,
        "has_model": False,
        "unlabeled": True,
        "active": False,
        "established": True,
        "measurement": False,
    }


def make_frame(number: int, t: float) -> dict:
    markers = []
    for i in range(8):  # 轨道点：绕 y 轴旋转的环
        angle = t * 0.8 + i * math.tau / 8
        radius = 0.18 + 0.03 * math.sin(t * 2.0 + i)
        y = 0.05 * math.sin(t * 1.5 + i * 0.7)
        markers.append(make_marker(100 + i, [radius * math.cos(angle), y, radius * math.sin(angle)], "point_cloud"))
    markers.append(make_marker(10, [0.05, 0.12, 0.02], "active"))
    markers.append(make_marker(20, [-0.05, 0.10, 0.0], "asset_member"))
    markers.append(make_marker(30, [0.0, 0.02, 0.06], "unknown"))
    markers.append(make_marker(40, [0.08, -0.05, -0.04], "point_cloud", occluded=(int(t) % 3 == 0)))

    angle = t * 0.5
    # 绕 y 轴的旋转四元数（xyzw）
    q = [0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)]
    rigid_bodies = [
        {
            "id": 1,
            "position": [0.0, 0.1, 0.0],
            "quaternion_xyzw": q,
            "mean_error": 0.0004,
            "tracking_valid": int(t) % 5 != 0,
        }
    ]
    return {
        "schema_version": 1,
        "frame_number": number,
        "motive_timestamp": t,
        "publisher_received_time_ns": time.time_ns(),
        "coordinate_system": "motive_y_up_right_handed",
        "unit": "meter",
        "publisher_dropped_frames": 0,
        "markers": markers,
        "rigid_bodies": rigid_bodies,
    }


def main() -> int:
    config = build_peer_config(connect_endpoint="tcp/127.0.0.1:7447")
    with zenoh.open(config) as session:
        publisher = session.declare_publisher(FRAME_KEY, encoding=zenoh.Encoding.APPLICATION_JSON)
        print(f"Publishing synthetic frames on {FRAME_KEY} at {FPS:.0f} Hz (Ctrl-C to stop)", flush=True)
        number = 0
        while True:
            publisher.put(encode_frame(make_frame(number, number / FPS)))
            number += 1
            time.sleep(1.0 / FPS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("demo publisher stopped")
