"""zenoh_topics.py — 列出 Zenoh 网络中当前有流量的 topic(ros2 topic list 风格)。

Zenoh 没有主题注册表/发现表(与 ROS2 DDS 不同),唯一纯网络手段是
被动订阅 ACL 放行的 key 空间并记录出现过的 key。因此:
- 只能看到「正在发数据」的 topic;声明了发布但从不发、或只订阅不发布的不可见。
- 低频消息需要足够观察窗口(本项目 rigid_body_names 周期 5s,edges 周期重发)。

用法:
    pixi run python scripts/zenoh_topics.py            # 观察 8 秒后一次性列出
    pixi run python scripts/zenoh_topics.py --seconds 15
    pixi run python scripts/zenoh_topics.py --watch    # 每 5 秒刷新一屏
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import zenoh

# ACL(config/zenohd_acl.yaml)放行的 key 空间 + liveliness(将来若启用)
WATCH_KEY_EXPRS = ("mocap/**", "manus/**", "liveliness/**")


def open_session(endpoint: str):
    cfg = zenoh.Config()
    cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    return zenoh.open(cfg)


def snapshot(session, duration: float) -> Counter:
    seen: Counter = Counter()

    def cb(sample):
        seen[str(sample.key_expr)] += 1

    subs = [session.declare_subscriber(k, cb) for k in WATCH_KEY_EXPRS]
    time.sleep(duration)
    for s in subs:
        s.undeclare()
    return seen


def render(seen: Counter, duration: float) -> str:
    if not seen:
        return "  (无任何消息 —— 当前没有发布端在线)"
    lines = []
    for key, n in seen.most_common():
        hz = n / duration
        lines.append(f"  {key:<40} {hz:6.1f} Hz  ({n} 条)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    ap.add_argument("--seconds", type=float, default=8.0, help="观察窗口(秒)")
    ap.add_argument("--watch", action="store_true", help="每 5 秒刷新一屏")
    args = ap.parse_args()

    session = open_session(args.endpoint)
    try:
        if not args.watch:
            print(f"[zenoh_topics] 观察 {args.seconds:.0f}s 内出现的 topic:")
            print(render(snapshot(session, args.seconds), args.seconds))
            return 0
        try:
            while True:
                print(f"\x1b[2J\x1b[H[zenoh_topics] 最近 {args.seconds:.0f}s 的 topic (Ctrl-C 退出):")
                print(render(snapshot(session, args.seconds), args.seconds))
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
