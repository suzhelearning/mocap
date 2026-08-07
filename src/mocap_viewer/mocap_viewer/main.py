"""Command-line entry point for the live Zenoh-to-Viser viewer."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import webbrowser

import zenoh

from natnet_zenoh.schema import FRAME_KEY
from .live_scene import LiveScene
from .zenoh_source import ZenohSource


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize real-time NatNet marker frames from Zenoh in Viser."
    )
    parser.add_argument("--listen-endpoint", default=None,
                        help="监听端点(peer 直连模式,默认 tcp/0.0.0.0:7447)")
    parser.add_argument(
        "--connect-endpoint", default=None,
        help="连接端点(router 模式,如 tcp/127.0.0.1:7447);"
             "指定后不再 listen,不占用端口,可与 router 并存",
    )
    parser.add_argument("--key", default=FRAME_KEY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=_positive_float, default=30.0)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--queue-capacity", type=_positive_int, default=8)
    parser.add_argument(
        "--no-relay",
        action="store_true",
        help="不把收到的帧重新发布给 connect 模式的消费者（默认转发）",
    )
    return parser


def serve(scene) -> None:
    try:
        scene.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        scene.close()


def main(
    argv: list[str] | None = None,
    *,
    zenoh_module=zenoh,
    source_factory=ZenohSource,
    scene_factory=LiveScene,
) -> int:
    args = build_parser().parse_args(argv)
    if (args.listen_endpoint is None) == (args.connect_endpoint is None):
        raise SystemExit(
            "必须指定 --listen-endpoint(peer 模式)或 --connect-endpoint(router 模式)之一"
        )
    listen = args.listen_endpoint or "tcp/0.0.0.0:7447"
    source = source_factory(
        args.key,
        listen_endpoint=args.listen_endpoint,
        connect_endpoint=args.connect_endpoint,
        queue_capacity=args.queue_capacity,
        relay=not args.no_relay,
        zenoh_module=zenoh_module,
    )
    try:
        scene = scene_factory(source, args.host, args.port, args.fps)
    except OSError as exc:
        print(f"Cannot start Viser at http://{args.host}:{args.port}: {exc}", file=sys.stderr)
        return 3
    try:
        source.start()
    except Exception as exc:
        print(f"Zenoh connect/listen on {listen} failed: {exc}", file=sys.stderr)
        scene.close()
        return 3
    url = f"http://{args.host}:{scene.server.get_port()}"
    print(f"NatNet live viewer: {url}", flush=True)
    mode = f"listening on {listen}" if args.listen_endpoint else \
        f"connected to {args.connect_endpoint}"
    print(f"Zenoh: {mode} (key {args.key})", flush=True)
    if args.open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    serve(scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
