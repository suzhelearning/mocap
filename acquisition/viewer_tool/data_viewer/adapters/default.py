from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 fallback for environments that only use custom adapters.
    tomllib = None

from data_viewer.contracts import (
    ProjectIndex,
    SampleDetail,
    SampleRecord,
    ViewerConfig,
    ViserPlayback,
)


METADATA_NAMES = [
    "metadata.json",
    "metadata.yaml",
    "metadata.yml",
    "metadata.toml",
    "metadata.txt",
    "info.json",
    "config.json",
]
METADATA_SUFFIXES = [".json", ".yaml", ".yml", ".toml", ".txt"]
MAX_TEXT_BYTES = 120_000


class DefaultAdapter:
    """Fallback adapter for projects that already contain .viser recordings."""

    def __init__(self, root: str | Path, config: ViewerConfig | None = None):
        self.root = Path(root).expanduser().resolve()
        self.config = config or ViewerConfig()
        self._index: ProjectIndex | None = None
        self._samples_by_id: dict[str, SampleRecord] = {}

    def index(self) -> ProjectIndex:
        if self._index is not None:
            return self._index
        samples = [
            self._sample_for_viser(path)
            for path in sorted(self._iter_viser_files(), key=lambda item: item.as_posix())
        ]
        warnings = [] if samples else ["No .viser recordings were found."]
        self._index = ProjectIndex(
            title=self.config.title or self.root.name or str(self.root),
            root=self.root,
            samples=samples,
            warnings=warnings,
        )
        self._samples_by_id = {sample.id: sample for sample in samples}
        return self._index

    def metadata(self, sample_id: str) -> SampleDetail:
        sample = self._sample_by_id(sample_id)
        return SampleDetail(sample=sample, metadata=self._metadata_for(sample.path))

    def build_viser(self, sample_id: str, cache_dir: Path) -> ViserPlayback:
        sample = self._sample_by_id(sample_id)
        if not sample.path.is_file():
            raise FileNotFoundError(sample.path)
        return ViserPlayback(recording_path=sample.path, label=sample.label)

    def _sample_by_id(self, sample_id: str) -> SampleRecord:
        self.index()
        try:
            return self._samples_by_id[sample_id]
        except KeyError:
            raise KeyError(sample_id) from None

    def _sample_for_viser(self, path: Path) -> SampleRecord:
        rel = path.relative_to(self.root).as_posix()
        parent = path.parent.relative_to(self.root).as_posix()
        group_label = parent if parent != "." else path.stem
        return SampleRecord(
            id=rel,
            label=rel,
            path=path,
            group_key=group_label,
            group_label=group_label,
            variant_label=path.stem,
            summary=f"{path.stat().st_size} bytes",
            facets={"directory": group_label},
        )

    def _iter_viser_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        results: list[Path] = []
        for path in self.root.rglob("*.viser"):
            if path.is_file() and not self._is_hidden(path):
                results.append(path)
        return results

    def _is_hidden(self, path: Path) -> bool:
        rel_parts = path.relative_to(self.root).parts
        hidden = set(self.config.hidden_names)
        return any(part in hidden or part.endswith(".egg-info") for part in rel_parts)

    def _metadata_for(self, viser_path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        candidates = self._metadata_candidates(viser_path)
        for path in candidates:
            if not path.is_file() or path.name in metadata:
                continue
            metadata[path.name] = _load_metadata(path)
        return metadata

    def _metadata_candidates(self, viser_path: Path) -> list[Path]:
        candidates = [viser_path.with_suffix(suffix) for suffix in METADATA_SUFFIXES]
        candidates.extend(viser_path.parent / name for name in METADATA_NAMES)
        return candidates


def _load_metadata(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("Reading TOML metadata requires Python 3.11+ or tomli.")
        return tomllib.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return _simple_yaml(path.read_text(encoding="utf-8"))
    text = path.read_bytes()[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
    return {"text": text, "truncated": path.stat().st_size > MAX_TEXT_BYTES}


def _simple_yaml(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _coerce_scalar(value.strip())
    return values


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")
