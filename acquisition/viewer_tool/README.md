# Data Viewer

可复制到任意工程的本地可视化工具。默认能力包含：

- viser 轨迹 recording 播放与同步
- metadata 展示
- 可通过项目 adapter 接入其他数据格式；本项目提供 HDF5 动捕 adapter
- MANO 网格回放：HDF5 adapter 提供 `mano` 播放模式，加载 `hands/<side>/mano_beta`
  形状参数驱动 MANO 网格（未写入 β 时使用中性形状），并展示各手 β 参数

viewer_tool 是单 HTTP 服务：控制页、viser client 和 recording 全部由同一端口提供，
默认使用 `8082`，不会再启动 `8083`。

## 运行

项目根目录提供快捷启动脚本:

```bash
cd /home/current/syz/mocap
./viewer.sh
```

无参数时自动打开 `$HOME/data/$(date +%Y%m%d)`，adapter 也会根据数据自动选择。
需要查看其他日期时直接传入目录:

```bash
./viewer.sh /home/current/data/20260811
```

也可以直接运行 viewer。省略 `--root` 时使用当天目录；省略 `--adapter` 时按
`viewer_adapter.py`、HDF5 adapter、默认 `.viser` adapter 的顺序自动选择:

```bash
cd acquisition/viewer_tool
pixi run serve
pixi run serve -- --root /home/current/data/20260811
```

打开:

```text
http://127.0.0.1:8082
```


如果项目根目录没有 `viewer_adapter.py`，工具会使用默认 adapter：扫描
`--root` 下的 `.viser` 文件，并读取同目录的 `metadata.json`、`metadata.yaml`、
`metadata.toml`、`metadata.txt`、`info.json` 或 `config.json`。

adapter 也可以显式指定，不必放在数据 root：

```bash
uv run data-viewer --root /path/to/data --adapter /path/to/viewer_adapter.py
```

## 项目适配

在目标工程根目录放一个 `viewer_adapter.py`：

```python
from pathlib import Path

from data_viewer.contracts import ProjectIndex, SampleDetail, ViewerAdapter, ViewerConfig, ViserPlayback


class MyAdapter(ViewerAdapter):
    def __init__(self, root: Path, config: ViewerConfig):
        self.root = root
        self.config = config

    def index(self) -> ProjectIndex:
        ...

    def metadata(self, sample_id: str) -> SampleDetail:
        ...

    def build_viser(self, sample_id: str, cache_dir: Path) -> ViserPlayback:
        ...


def create_adapter(root: Path, config: ViewerConfig) -> ViewerAdapter:
    return MyAdapter(root, config)
```

`build_viser` 应返回一个 `.viser` recording 路径。多个轨迹需要同步播放时，请在
adapter 内生成一个合并后的 `.viser` recording；核心工具只加载一个 playback。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```
