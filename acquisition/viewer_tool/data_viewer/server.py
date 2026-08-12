from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import tempfile
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from data_viewer.adapters.loader import load_adapter
from data_viewer.contracts import (
    ProjectIndex,
    SampleDetail,
    ViewerAdapter,
    ViewerConfig,
    ViserPlayback,
    json_value,
)


STATIC_DIR = Path(__file__).parent / "static"
CHUNK_SIZE = 1024 * 1024


@dataclass
class ApiState:
    root: Path
    adapter: ViewerAdapter
    index: ProjectIndex | None = None
    cache_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "data_viewer_viser"
    )


@dataclass
class FileResponseInfo:
    path: Path
    status: int
    start: int
    end: int
    length: int
    size: int
    content_type: str
    headers: dict[str, str]


def make_api_response(
    state: ApiState, method: str, path: str, body: dict | None
) -> dict:
    parsed = urlparse(path)
    if method == "POST" and parsed.path == "/api/load":
        try:
            state.index = state.adapter.index()
            return {
                "kind": "project_root",
                "root": str(state.root),
                "index": state.index.to_dict(),
            }
        except Exception as exc:
            return {"error": "index_failed", "message": str(exc)}

    if method == "GET" and parsed.path == "/api/sample":
        sample_id = parse_qs(parsed.query).get("id", [""])[0]
        try:
            detail = state.adapter.metadata(sample_id)
            if isinstance(detail, SampleDetail):
                return detail.to_dict()
            sample = _sample_by_id(_ensure_index(state), sample_id)
            return SampleDetail(sample=sample, metadata=detail).to_dict()
        except KeyError:
            return {"error": "sample_not_found"}
        except Exception as exc:
            return {"error": "metadata_failed", "message": str(exc)}

    if method == "GET" and parsed.path == "/api/viser":
        query = parse_qs(parsed.query)
        sample_id = query.get("id", [""])[0]
        playback_mode = query.get("mode", [""])[0]
        try:
            state.cache_dir.mkdir(parents=True, exist_ok=True)
            if playback_mode and hasattr(state.adapter, "build_viser_with_mode"):
                playback = state.adapter.build_viser_with_mode(sample_id, state.cache_dir, playback_mode)
            else:
                playback = state.adapter.build_viser(sample_id, state.cache_dir)
            return _viser_payload(state, playback)
        except KeyError:
            return {"error": "sample_not_found"}
        except Exception as exc:
            message = str(exc)
            if playback_mode == "mano":
                message = f"MANO 模式: {message}"
            return {"error": "viser_failed", "message": message}

    return {"error": "not_found"}


def _ensure_index(state: ApiState) -> ProjectIndex:
    if state.index is None:
        state.index = state.adapter.index()
    return state.index


def _sample_by_id(index: ProjectIndex, sample_id: str):
    for sample in index.samples:
        if sample.id == sample_id:
            return sample
    raise KeyError(sample_id)


def _viser_payload(state: ApiState, playback: ViserPlayback) -> dict:
    recording_path = playback.recording_path.expanduser().resolve()
    root = state.root.resolve()
    cache = state.cache_dir.resolve()
    if _is_relative_to(recording_path, cache):
        path = f"/viser-rec?name={quote(recording_path.name)}"
    elif _is_relative_to(recording_path, root):
        rel = recording_path.relative_to(root).as_posix()
        path = f"/file?path={quote(rel)}"
    else:
        return {
            "error": "recording_outside_root",
            "message": f"{recording_path} is outside root/cache.",
        }
    return {
        "name": recording_path.name,
        "label": playback.label or recording_path.name,
        "path": path,
        "client": "/viser-client/",
        "warnings": playback.warnings,
    }


def file_response_info(path: Path, range_header: str | None = None) -> FileResponseInfo:
    size = path.stat().st_size
    start = 0
    end = max(size - 1, 0)
    status = HTTPStatus.OK
    headers: dict[str, str] = {"Accept-Ranges": "bytes"}

    if range_header and range_header.startswith("bytes=") and size > 0:
        parsed_range = _parse_byte_range(range_header, size)
        if parsed_range is None:
            status = HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
            start = 0
            end = -1
            headers["Content-Range"] = f"bytes */{size}"
        else:
            start, end = parsed_range
            status = HTTPStatus.PARTIAL_CONTENT

    length = max(end - start + 1, 0) if size else 0
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    headers["Content-Length"] = str(length)
    headers["Content-Type"] = content_type
    if status == HTTPStatus.PARTIAL_CONTENT:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return FileResponseInfo(path, int(status), start, end, length, size, content_type, headers)


def _parse_byte_range(range_header: str, size: int) -> tuple[int, int] | None:
    spec = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
    if "-" not in spec:
        return None
    left, right = spec.split("-", 1)
    try:
        if left == "":
            suffix_len = int(right)
            if suffix_len <= 0:
                return None
            return max(size - suffix_len, 0), size - 1
        start = int(left)
        end = int(right) if right else size - 1
    except ValueError:
        return None
    if start < 0 or end < start or start >= size:
        return None
    return start, min(end, size - 1)


def make_handler(state: ApiState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DataViewer/0.1"

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/api/sample", "/api/viser"}:
                self._send_json(make_api_response(state, "GET", self.path, None))
                return
            if parsed.path == "/file":
                self._send_root_file()
                return
            if parsed.path == "/viser-rec":
                self._send_cache_file()
                return
            if parsed.path == "/viser-client/" or parsed.path.startswith("/viser-client/"):
                self._send_viser_client()
                return
            self._send_static(parsed.path)

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(make_api_response(state, "POST", parsed.path, body))

        def log_message(self, fmt, *args):
            return

        def _send_json(self, payload: dict, status=HTTPStatus.OK):
            data = json.dumps(json_value(payload), indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_static(self, request_path: str):
            rel = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
            target = (STATIC_DIR / rel).resolve()
            if not _is_relative_to(target, STATIC_DIR.resolve()) or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._write_path(target)

        def _send_root_file(self):
            rel = parse_qs(urlparse(self.path).query).get("path", [""])[0]
            rel = unquote(rel)
            target = (state.root / rel).resolve()
            root = state.root.resolve()
            if not _is_relative_to(target, root) or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._write_path(target)

        def _send_cache_file(self):
            name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
            name = unquote(name)
            cache = state.cache_dir.resolve()
            target = (cache / name).resolve()
            if name.endswith(".viser") and _is_relative_to(target, cache) and target.is_file():
                self._write_path(target)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send_viser_client(self):
            index = _viser_client_index()
            if index is None:
                self.send_error(HTTPStatus.NOT_FOUND, "viser client not installed")
                return
            self._write_path(index)

        def _write_path(self, target: Path):
            info = file_response_info(target, self.headers.get("Range"))
            self.send_response(info.status)
            for key, value in info.headers.items():
                self.send_header(key, value)
            self.end_headers()
            with target.open("rb") as file:
                file.seek(info.start)
                remaining = info.length
                while remaining > 0:
                    chunk = file.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def _viser_client_index() -> Path | None:
    spec = importlib.util.find_spec("viser")
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = Path(spec.submodule_search_locations[0]) / "client" / "build" / "index.html"
    return candidate if candidate.is_file() else None


def default_data_root(today: date | None = None) -> Path:
    """返回当天录制目录,可用 MOCAP_DATA_DIR 覆盖数据根目录。"""
    day = today or date.today()
    base = Path(os.environ.get("MOCAP_DATA_DIR", str(Path.home() / "data"))).expanduser()
    return base / day.strftime("%Y%m%d")


def run_server(
    host: str,
    port: int,
    root: str | Path | None = None,
    open_browser: bool = False,
    title: str | None = None,
    adapter: str | Path | None = None,
):
    root_path = (
        default_data_root()
        if root is None
        else Path(root).expanduser().resolve()
    )
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"数据目录不存在: {root_path}。可用 --root 指定其他目录。"
        )
    adapter_path = Path(adapter).expanduser().resolve() if adapter is not None else None
    config = ViewerConfig(title=title, adapter_path=adapter_path)
    loaded_adapter = load_adapter(root_path, config)
    state = ApiState(root=root_path, adapter=loaded_adapter)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    url = f"http://{host}:{httpd.server_port}"
    print(f"Data viewer running at {url}")
    print(f"Loaded root: {root_path}")
    print(f"Adapter: {type(loaded_adapter).__name__}")
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the local viser data viewer.")
    parser.add_argument(
        "--root",
        default=None,
        help="数据目录;默认使用 ~/data/YYYYMMDD,可按需指定其他目录。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8082, type=int)
    parser.add_argument("--title", default=None, help="Optional UI title override.")
    parser.add_argument(
        "--adapter",
        default=None,
        help="可选 adapter 路径;省略时自动选择项目或 HDF5 adapter。",
    )
    parser.add_argument("--open", action="store_true", help="Open the viewer in a browser.")
    args = parser.parse_args()
    run_server(args.host, args.port, args.root, args.open, args.title, args.adapter)


if __name__ == "__main__":
    main()
