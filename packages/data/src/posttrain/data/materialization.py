"""Reproducible execution support for typed Python dataset builders."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, cast

from posttrain.common import ContractError, JsonValue

from .definitions import (
    BuiltDatasetSource,
    DatasetBuildInput,
    HuggingFaceDatasetInput,
    LocalDatasetInput,
    PackageResourceInput,
    ResolvedDatasetBuildInput,
)

MATERIALIZER_SCHEMA_VERSION = 2


def build_key(
    plan: Any,
    *,
    project_root: Path,
    code_snapshot_digest: str | None = None,
    dependency_lock_digest: str | None = None,
) -> str:
    """Return the cache identity for a selection and its executable recipe."""

    source = plan.source
    payload: dict[str, Any] = {
        "materializer_schema_version": MATERIALIZER_SCHEMA_VERSION,
        "selection": {
            "id": plan.id,
            "revision": plan.revision,
            "kind": plan.kind,
            "split": plan.split,
            "schema_version": plan.schema_version,
            "format": plan.format,
            "provenance": _jsonable(plan.provenance),
            "access": _jsonable(plan.access),
        },
        "source": _source_identity(source, project_root=project_root),
    }
    if isinstance(source, BuiltDatasetSource):
        builder = source.builder
        payload["code_snapshot_digest"] = (
            code_snapshot_digest or builder.code_digest or source_code_digest(project_root)
        )
        payload["dependency_lock_digest"] = (
            dependency_lock_digest or builder.dependency_lock_digest or lock_digest(project_root)
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_code_digest(project_root: Path) -> str:
    """Hash project Python sources deterministically for a local build.

    This intentionally covers all project Python files, including imported
    helpers.  Job packing can provide a narrower immutable source snapshot
    digest through ``code_snapshot_digest``.
    """

    root = project_root.resolve()
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if any(
            part in {".git", ".posttrain", ".venv", ".venvs", "__pycache__", "build", "dist"} for part in path.parts
        ):
            continue
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def lock_digest(project_root: Path) -> str | None:
    """Hash the first supported dependency lock, if the project has one."""

    for name in ("uv.lock", "poetry.lock", "Pipfile.lock", "requirements.lock", "requirements.txt"):
        path = project_root / name
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def run_typed_builder(
    source: BuiltDatasetSource,
    *,
    project_root: Path,
    workspace: Path,
    timeout_seconds: float = 300.0,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve declared inputs and execute a typed builder in a child process."""

    if timeout_seconds <= 0:
        raise ContractError("dataset builder timeout must be positive")
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ContractError(f"dataset project root is not a directory: {project_root}")
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, ResolvedDatasetBuildInput] = {}
    temporary_paths: list[Path] = []
    try:
        for name, item in source.inputs.items():
            resolved_item, temporary = _resolve_input(name, item, project_root=project_root, inputs_dir=inputs_dir)
            resolved[name] = resolved_item
            if temporary:
                temporary_paths.append(temporary)
        request_path = workspace / "builder-request.json"
        result_path = workspace / "builder-result.json"
        request_path.write_text(
            json.dumps(
                {
                    "project_root": str(project_root),
                    "workspace": str(workspace),
                    "target": source.builder.target,
                    "inputs": {
                        name: {
                            "kind": value.kind,
                            "path": str(value.path),
                            "digest": value.digest,
                            "metadata": dict(value.metadata),
                        }
                        for name, value in sorted(resolved.items())
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        pythonpath = [str(project_root)]
        src_root = project_root / "src"
        if src_root.is_dir():
            pythonpath.append(str(src_root))
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "posttrain.data.builder_runner", str(request_path), str(result_path)],
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ContractError(f"dataset builder timed out after {timeout_seconds:g}s") from error
        if not result_path.is_file():
            detail = (completed.stderr or completed.stdout).strip()
            raise ContractError(f"dataset builder failed without a result ({detail})")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            detail = result.get("error", "unknown builder failure") if isinstance(result, Mapping) else "invalid result"
            trace = result.get("traceback", "") if isinstance(result, Mapping) else ""
            output = "\n".join(
                value
                for value in (
                    str(result.get("stdout", "")).strip() if isinstance(result, Mapping) else "",
                    str(result.get("stderr", "")).strip() if isinstance(result, Mapping) else "",
                    completed.stdout.strip(),
                    completed.stderr.strip(),
                )
                if value
            )
            raise ContractError(
                f"dataset builder {source.builder.target!r} failed: {detail}\n{trace}\n{output}".strip()
            )
        rows = result.get("rows")
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ContractError("dataset builder returned an invalid result")
        if not rows:
            raise ContractError("dataset builder returned no rows")
        return tuple(cast(Mapping[str, Any], row) for row in rows)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def _resolve_input(
    name: str,
    item: DatasetBuildInput,
    *,
    project_root: Path,
    inputs_dir: Path,
) -> tuple[ResolvedDatasetBuildInput, Path | None]:
    if isinstance(item, LocalDatasetInput):
        path = _safe_local_path(project_root, item.path)
        return ResolvedDatasetBuildInput(name, item.kind, path, _digest_file(path), {"format": item.format}), None
    if isinstance(item, PackageResourceInput):
        package, _, resource = item.resource.partition(":")
        try:
            data = resource_files(package).joinpath(resource).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError) as error:
            raise ContractError(f"package resource input not found: {item.resource}") from error
        path = inputs_dir / f"{_safe_name(name)}-resource"
        path.write_bytes(data)
        return ResolvedDatasetBuildInput(name, item.kind, path, _digest_file(path), {"resource": item.resource}), path
    if isinstance(item, HuggingFaceDatasetInput):
        path = inputs_dir / f"{_safe_name(name)}-records.jsonl"
        rows = _load_huggingface_rows(item)
        path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        return (
            ResolvedDatasetBuildInput(
                name,
                item.kind,
                path,
                _digest_file(path),
                dict(item.identity()),
            ),
            path,
        )
    raise ContractError(f"unsupported dataset build input for {name!r}: {type(item).__name__}")


def input_identities(source: BuiltDatasetSource, *, project_root: Path) -> Mapping[str, JsonValue]:
    """Return declared input identities plus local content digests for manifests."""

    result: dict[str, JsonValue] = {}
    for name, item in source.inputs.items():
        identity = dict(item.identity())
        if isinstance(item, LocalDatasetInput):
            identity["content_sha256"] = _digest_file(_safe_local_path(project_root, item.path))
        elif isinstance(item, PackageResourceInput):
            package, _, resource = item.resource.partition(":")
            try:
                identity["content_sha256"] = hashlib.sha256(
                    resource_files(package).joinpath(resource).read_bytes()
                ).hexdigest()
            except (FileNotFoundError, ModuleNotFoundError) as error:
                raise ContractError(f"package resource input not found: {item.resource}") from error
        result[name] = identity
    return result


def _load_huggingface_rows(item: HuggingFaceDatasetInput) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ContractError("Hugging Face dataset inputs require posttrain-data[huggingface]") from error
    loaded = load_dataset(item.repo, item.config, revision=item.revision, split=item.split)
    return cast(Iterable[Mapping[str, Any]], loaded)


def _source_identity(source: object, *, project_root: Path) -> Mapping[str, JsonValue]:
    if isinstance(source, BuiltDatasetSource):
        return source.identity()
    if isinstance(source, Mapping):
        result = dict(source)
        if result.get("kind") == "built":
            builder = result.get("builder")
            if isinstance(builder, Mapping) and builder.get("kind") == "python":
                result["builder"] = dict(builder)
            inputs = result.get("inputs")
            if isinstance(inputs, list):
                result["input_digests"] = {
                    value: _digest_file(_safe_local_path(project_root, value))
                    for value in sorted(inputs)
                    if isinstance(value, str) and (project_root / value).is_file()
                }
        return cast(Mapping[str, JsonValue], result)
    raise ContractError("dataset source must be a mapping or BuiltDatasetSource")


def _jsonable(value: object) -> JsonValue:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return cast(JsonValue, asdict(cast(Any, value)))
    return cast(JsonValue, value)


def _safe_local_path(root: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute() or not configured or ".." in path.parts:
        raise ContractError("dataset build input path must be relative to the project root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ContractError(f"dataset build input file not found: {configured}")
    return resolved


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


__all__ = [
    "MATERIALIZER_SCHEMA_VERSION",
    "build_key",
    "input_identities",
    "lock_digest",
    "run_typed_builder",
    "source_code_digest",
]


def __getattr__(name: str) -> object:
    """Lazily expose the result value while keeping catalog compatibility."""

    if name == "DatasetMaterialization":
        from .catalog import DatasetMaterialization

        return DatasetMaterialization
    raise AttributeError(name)
