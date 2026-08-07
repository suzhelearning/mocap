"""Connect-mode Zenoh consumer for the mocap marker-frame stream.

与 viewer 不同（viewer 监听 7447 端口），本工具以 connect 模式加入已有
Zenoh 网络，不占用端口，适合任意消费者查看/记录数据。

用法：
    pixi run subscribe                          # 连接本地 7447，打印统计
    pixi run subscribe -- --output cap.jsonl    # 同时记录 JSONL
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from pathlib import Path

import zenoh

from natnet_zenoh.schema import FRAME_KEY, encode_frame
from natnet_zenoh.subscriber import FrameHandler, FrameStats
from natnet_zenoh.zenoh_transport import build_peer_config


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subscribe to the NatNet marker-frame Zenoh stream (connect mode)."
    )
    parser.add_argument("--connect-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--key", default=FRAME_KEY)
    parser.add_argument("--output", type=Path, help="optional JSONL output path")
    parser.add_argument("--stats-interval", type=_positive_float, default=2.0)
    return parser


class JsonlWriter:
    def __init__(self, path: Path, overwrite: bool = False) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite to replace it")
        self._file = path.open("w", encoding="utf-8")

    def write(self, frame: dict) -> None:
        self._file.write(encode_frame(frame) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = FrameStats()
    recorder = None
    if args.output is not None:
        try:
            recorder = JsonlWriter(args.output)
        except FileExistsError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 4

    def on_sample(sample: object) -> None:
        frame = FrameHandler(stats).handle_json(sample.payload.to_string())
        if frame is not None and recorder is not None:
            recorder.write(frame)

    config = build_peer_config(connect_endpoint=args.connect_endpoint)
    stop_event = threading.Event()
    print(
        f"Subscribing to {args.key} via {args.connect_endpoint} (Ctrl-C to stop)",
        flush=True,
    )
    try:
        with zenoh.open(config) as session:
            subscriber = session.declare_subscriber(args.key, on_sample)
            try:
                while not stop_event.wait(args.stats_interval):
                    s = stats.snapshot()
                    print(
                        f"frames={s['received_frames']} rate_hz={s['frame_rate_hz']:.1f} "
                        f"markers={s['last_marker_count']} missing={s['missing_frames']} "
                        f"publisher_dropped={s['publisher_dropped_frames']} "
                        f"invalid={s['invalid_messages']}",
                        flush=True,
                    )
            finally:
                subscriber.undeclare()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Zenoh connect to {args.connect_endpoint} failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if recorder is not None:
            recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
