"""录制状态机(TakeController):IDLE → RECORDING → 保存/丢弃。

```
                 start键(创建 take)
     IDLE ──────────────────────────► RECORDING
       ▲                                │
       │  save键(保存)/discard键(丢弃)  │
       │◄───────────────────────────────┤
       └─────────── quit 键─────────────┘
```
quit:录制中先丢弃当前 take,再请求退出(quit_requested 置位)。
SAVING/DISCARDING 为瞬态阻塞(flush 落盘),完成后回 IDLE。

writer 生命周期完全由状态机管理:begin 在进入 RECORDING 时调用,
finalize_save/discard 在离开时调用。
"""

from __future__ import annotations

import enum
from datetime import datetime
import time
from collections.abc import Callable
from pathlib import Path

from .config import Config
from .recorder import EV_DISCARD, EV_SAVE, EV_START, TakeWriter

WriterFactory = Callable[[int, Path], TakeWriter]


class State(enum.Enum):
    IDLE = "空闲"
    RECORDING = "录制中"


class TakeController:
    """按键驱动状态机;handle(ch) 返回 True 表示状态发生了转移。"""

    def __init__(
        self,
        config: Config,
        writer_factory: WriterFactory,
        take_dir: Path | None = None,
    ) -> None:
        self._cfg = config
        self._writer_factory = writer_factory
        self._take_dir = take_dir or config.output_dir
        self.state = State.IDLE
        self.take_id = 0
        self.writer: TakeWriter | None = None
        self.quit_requested = False
        self.last_result = ""          # 最近一次保存/丢弃结果(状态栏显示)
        self._start_ns = 0

    # -- 主入口 -----------------------------------------------------------

    def handle(self, ch: str) -> bool:
        """处理一个按键;发生状态转移或退出请求返回 True。"""
        before = self.state
        if ch == self._cfg.keymap["quit"]:
            self._on_quit()
            return True                     # 退出请求即使无转移也要响应
        elif ch == self._cfg.keymap["start"]:
            self._on_start()
        elif ch == self._cfg.keymap["save"]:
            self._on_save()
        elif ch == self._cfg.keymap["discard"]:
            self._on_discard()
        else:
            return False
        return before != self.state

    def status_line(self) -> str:
        """状态栏文本(不含流速率,由 cli 拼接)。"""
        if self.state is State.RECORDING:
            dur = time.time() - self._start_ns / 1e9
            return f"● 录制中 take#{self.take_id} {dur:5.1f}s"
        return f"○ 空闲 ({self.last_result})"

    # -- 转移实现 ---------------------------------------------------------

    def _on_start(self) -> None:
        if self.state is not State.IDLE:
            return
        self.take_id += 1
        self._start_ns = time.time_ns()
        # 纳秒时间 + 进程内 take 序号，避免同秒多 take 文件名碰撞。
        stamp = datetime.fromtimestamp(self._start_ns / 1e9).strftime(
            "%Y%m%d_%H%M%S_%f")
        path = (self._take_dir / stamp[:8]
                / f"{stamp}_take{self.take_id:03d}.h5")
        self.writer = self._writer_factory(self.take_id, path)
        self.writer.begin(self.take_id, self._start_ns)
        self.writer.append_event(EV_START, "start")
        self.state = State.RECORDING
        self.last_result = ""

    def _on_save(self) -> None:
        if self.state is State.RECORDING:
            self.writer.append_event(EV_SAVE, "save")
            self.writer.finalize_save()
            n = self.writer.counts()
            self.last_result = (
                f"[saved] {self.writer.path.name} "
                f"({n['mocap']} mocap, {n['left']}/{n['right']} hand)"
            )
            self.writer = None
            self.state = State.IDLE

    def _on_discard(self) -> None:
        if self.state is State.RECORDING:
            self.writer.append_event(EV_DISCARD, "discard")
            self.writer.discard()
            self.last_result = f"[discarded] take#{self.take_id}"
            self.writer = None
            self.state = State.IDLE

    def _on_quit(self) -> None:
        if self.state is State.RECORDING:
            self._on_discard()
        self.quit_requested = True
