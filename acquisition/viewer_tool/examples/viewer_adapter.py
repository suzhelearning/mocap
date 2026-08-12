from __future__ import annotations

from pathlib import Path

from data_viewer.adapters.default import DefaultAdapter
from data_viewer.contracts import ViewerConfig


def create_adapter(root: Path, config: ViewerConfig):
    """Minimal project adapter.

    Replace DefaultAdapter with project-specific indexing or recording generation
    when a project needs to build .viser files on demand.
    """
    return DefaultAdapter(root, config)
