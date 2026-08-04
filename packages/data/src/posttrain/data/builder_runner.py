"""Internal child-process entry point for Python dataset builders.

The module is intentionally small and only consumes a JSON request.  Public
callers should use ``materialize_dataset``; this process boundary prevents
builder imports and temporary state from leaking into catalog loading.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import sys
import traceback
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from .definitions import DatasetBuildContext, ResolvedDatasetBuildInput


def run(request_path: Path, result_path: Path) -> int:
    """Execute one serialized builder request and write a structured result."""

    captured_stdout: list[str] = []
    captured_stderr: list[str] = []
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, Mapping):
            raise ValueError("builder request must be a JSON object")
        project_root = Path(_string(request, "project_root")).resolve()
        workspace = Path(_string(request, "workspace")).resolve()
        if not project_root.is_dir() or not workspace.is_dir():
            raise ValueError("builder project_root and workspace must be directories")
        sys.path.insert(0, str(project_root))
        src_root = project_root / "src"
        if src_root.is_dir():
            sys.path.insert(0, str(src_root))

        raw_inputs = request.get("inputs")
        if not isinstance(raw_inputs, Mapping):
            raise ValueError("builder request inputs must be an object")
        inputs: dict[str, ResolvedDatasetBuildInput] = {}
        for name, raw in raw_inputs.items():
            if not isinstance(name, str) or not isinstance(raw, Mapping):
                raise ValueError("builder input entries must be named objects")
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"builder input {name!r} metadata must be an object")
            inputs[name] = ResolvedDatasetBuildInput(
                name=name,
                kind=_string(raw, "kind", name=name),
                path=Path(_string(raw, "path", name=name)).resolve(),
                digest=_string(raw, "digest", name=name),
                metadata=cast(Mapping[str, Any], metadata),
            )

        target = _string(request, "target")
        module_name, separator, callable_name = target.partition(":")
        if not separator:
            raise ValueError("builder target must use module:callable syntax")
        module = importlib.import_module(module_name)
        factory = getattr(module, callable_name, None)
        if not callable(factory) or getattr(factory, "__qualname__", callable_name) != callable_name:
            raise ValueError(f"builder target is not a module-level callable: {target}")

        with (
            contextlib.redirect_stdout(_Capture(captured_stdout)),
            contextlib.redirect_stderr(_Capture(captured_stderr)),
        ):
            rows = factory(DatasetBuildContext(inputs=inputs, workspace=workspace))
            if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Iterable):
                raise TypeError("dataset builder must return an iterable of mapping rows")
            normalized: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise TypeError(f"dataset builder row {index} must be a mapping")
                normalized.append(dict(cast(Mapping[str, Any], row)))
                # Force serialization in the child so failures report here,
                # before any cache directory is promoted.
                json.dumps(normalized[-1], sort_keys=True)

        result_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "rows": normalized,
                    "stdout": "".join(captured_stdout),
                    "stderr": "".join(captured_stderr),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    except BaseException as error:  # noqa: BLE001 - serialize all child failures
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "traceback": traceback.format_exc(),
                    "stdout": "".join(captured_stdout),
                    "stderr": "".join(captured_stderr),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 1


class _Capture:
    def __init__(self, target: list[str]) -> None:
        self._target = target

    def write(self, value: str) -> int:
        self._target.append(value)
        return len(value)

    def flush(self) -> None:
        return None


def _string(value: Mapping[str, Any], key: str, *, name: str | None = None) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        prefix = f"builder input {name!r}" if name is not None else "builder request"
        raise ValueError(f"{prefix} requires a non-empty {key}")
    return item


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit("usage: python -m posttrain.data.builder_runner REQUEST.json RESULT.json")
    return run(Path(args[0]), Path(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
