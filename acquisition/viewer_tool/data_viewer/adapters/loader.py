from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from data_viewer.contracts import ViewerAdapter, ViewerConfig


def load_adapter(root: str | Path, config: ViewerConfig | None = None) -> ViewerAdapter:
    root_path = Path(root).expanduser().resolve()
    viewer_config = config or ViewerConfig()
    adapter_path = _resolve_adapter_path(root_path, viewer_config.adapter_path)
    if adapter_path is None:
        from data_viewer.adapters.default import DefaultAdapter

        return DefaultAdapter(root_path, viewer_config)

    module = _load_module(adapter_path)
    create_adapter = getattr(module, "create_adapter", None)
    if create_adapter is None or not callable(create_adapter):
        raise AttributeError(f"{adapter_path} must define create_adapter(root, config)")
    return create_adapter(root_path, viewer_config)


def _resolve_adapter_path(root: Path, explicit_path: Path | None) -> Path | None:
    """按显式路径、本地项目 adapter、当前工程 HDF5 adapter 顺序选择。"""
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"adapter not found: {path}")
        return path

    project_adapter = root / "viewer_adapter.py"
    if project_adapter.is_file():
        return project_adapter

    if _contains_hdf5(root):
        h5_adapter = Path(__file__).resolve().parents[3] / "scripts" / "h5_viewer_adapter.py"
        if h5_adapter.is_file():
            return h5_adapter

    return None


def _contains_hdf5(root: Path) -> bool:
    if not root.is_dir():
        return False
    return next(root.rglob("*.h5"), None) is not None


def _load_module(path: Path):
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    name = f"_data_viewer_adapter_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
