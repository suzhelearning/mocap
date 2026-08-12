from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


@dataclass
class ViewerConfig:
    title: str | None = None
    adapter_path: Path | None = None
    hidden_names: list[str] = field(
        default_factory=lambda: [".git", ".venv", "__pycache__", ".pytest_cache"]
    )


@dataclass
class SampleRecord:
    id: str
    label: str
    path: Path
    group_key: str = ""
    group_label: str = ""
    variant_label: str = ""
    summary: str = ""
    facets: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


@dataclass
class ProjectIndex:
    title: str
    root: Path | str
    samples: list[SampleRecord]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "root": str(self.root),
            "sample_count": len(self.samples),
            "warnings": self.warnings,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass
class SampleDetail:
    sample: SampleRecord
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


@dataclass
class ViserPlayback:
    recording_path: Path
    label: str = ""
    warnings: list[str] = field(default_factory=list)


class ViewerAdapter:
    def index(self) -> ProjectIndex:
        raise NotImplementedError

    def metadata(self, sample_id: str) -> SampleDetail | dict[str, Any]:
        raise NotImplementedError

    def build_viser(self, sample_id: str, cache_dir: Path) -> ViserPlayback:
        raise NotImplementedError
