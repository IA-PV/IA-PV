"""Atomic storage and reproducibility metadata for experiment reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
from uuid import uuid4


def describe_source_artifact(path: str | Path) -> dict[str, object]:
    """Return a stable identity for a model/checkpoint used by an experiment."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Source artifact does not exist: {source}")
    return {
        "path": str(source.resolve()),
        "size_bytes": source.stat().st_size,
        "sha256": _file_sha256(source),
    }


def available_run_id(
    root: Path,
    categories: Sequence[str],
    started_at: datetime,
) -> str:
    base = started_at.strftime("%Y%m%dT%H%M%S.%f%z")
    candidate = base
    suffix = 2
    while any((root / category / candidate).exists() for category in categories):
        candidate = f"{base}-{suffix:02d}"
        suffix += 1
    return candidate


def agent_slug(agent_name: str) -> str:
    characters: list[str] = []
    for index, character in enumerate(agent_name):
        if character.isupper() and index and (
            agent_name[index - 1].islower()
            or (index + 1 < len(agent_name) and agent_name[index + 1].islower())
        ):
            characters.append("_")
        characters.append(character.lower() if character.isalnum() else "_")
    return "".join(characters).strip("_")


def publish_directory(final: Path, builder: Callable[[Path], None]) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(f".{final.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    try:
        builder(staging)
        staging.replace(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def artifact_manifest(directory: Path, names: Sequence[str]) -> dict[str, object]:
    return {
        name: {
            "size_bytes": (directory / name).stat().st_size,
            "sha256": _file_sha256(directory / name),
        }
        for name in names
    }


def runtime_metadata(report_root: Path) -> dict[str, object]:
    source_path = report_root.resolve()
    while not source_path.exists() and source_path != source_path.parent:
        source_path = source_path.parent
    try:
        package_version = importlib.metadata.version("tetris-ai")
    except importlib.metadata.PackageNotFoundError:
        package_version = "not-installed"
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "working_directory": str(Path.cwd().resolve()),
        "package_version": package_version,
        "source_control": _git_metadata(source_path),
    }


def record_as_dict(record: object) -> dict[str, object]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    raise TypeError("Training episode records must be dataclasses or mappings.")


def unique_mappings(values: Sequence[Mapping[str, object]]) -> list[object]:
    unique: dict[str, object] = {}
    for value in values:
        safe_value = json_safe(value)
        key = json.dumps(safe_value, sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, safe_value)
    return list(unique.values())


def json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def local_datetime(value: datetime | None) -> datetime:
    current = value or datetime.now().astimezone()
    if current.tzinfo is None:
        return current.astimezone()
    return current


def atomic_write_json(path: Path, payload: object) -> Path:
    return atomic_write_text(
        path,
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
    )


def atomic_write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def temporary_file(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{uuid4().hex}")


def _git_metadata(path: Path) -> dict[str, object] | None:
    try:
        repository = _run_git(path, "rev-parse", "--show-toplevel")
        commit = _run_git(path, "rev-parse", "HEAD")
        status = _run_git(path, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "repository_root": repository,
        "commit": commit,
        "working_tree_dirty": bool(status),
    }


def _run_git(path: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(path.resolve()), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "agent_slug",
    "artifact_manifest",
    "atomic_write_json",
    "atomic_write_text",
    "available_run_id",
    "describe_source_artifact",
    "json_safe",
    "local_datetime",
    "publish_directory",
    "record_as_dict",
    "runtime_metadata",
    "temporary_file",
    "unique_mappings",
]
