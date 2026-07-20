"""Small local recovery record shared by every execution engine."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import yaml

from common.profiles import ResolvedProfile

RunKind = Literal[
    "model-onboarding",
    "serving-benchmark",
    "general-eval",
    "domain-eval",
    "sft",
    "dpo",
    "rl",
    "model-transformation",
    "comparison-report",
]
RUN_KINDS = frozenset(
    {
        "model-onboarding",
        "serving-benchmark",
        "general-eval",
        "domain-eval",
        "sft",
        "dpo",
        "rl",
        "model-transformation",
        "comparison-report",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "trackio",
        "verifiers",
        "vllm",
        "sglang",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def _gpu_summary() -> str | None:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


@dataclass(slots=True)
class RunContext:
    """Resolved inputs, recovery directories, and events for one execution."""

    root: Path
    run_kind: RunKind

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def recovery_dir(self) -> Path:
        return self.root / "recovery"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @classmethod
    def create(
        cls,
        root: Path,
        run_kind: RunKind,
        resolved_config: dict[str, Any],
        resolved_profile: ResolvedProfile | None = None,
    ) -> "RunContext":
        if run_kind not in RUN_KINDS:
            expected = ", ".join(sorted(RUN_KINDS))
            raise ValueError(f"unsupported run kind {run_kind!r}; expected one of: {expected}")
        context = cls(root=root.resolve(), run_kind=run_kind)
        for path in (
            context.root,
            context.artifacts_dir,
            context.recovery_dir,
            context.output_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        (context.root / "resolved-config.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=True),
            encoding="utf-8",
        )
        profile_metadata = None
        if resolved_profile is not None:
            (context.root / "resolved-profile.yaml").write_text(
                yaml.safe_dump(resolved_profile.data, sort_keys=True),
                encoding="utf-8",
            )
            profile_metadata = {
                "kind": resolved_profile.kind,
                "reference": resolved_profile.reference,
                "sources": [str(path) for path in resolved_profile.sources],
            }
        metadata = {
            "run_kind": run_kind,
            "created_at": _now(),
            "command": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": _gpu_summary(),
            "packages": _package_versions(),
            "profile": profile_metadata,
        }
        (context.root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        context.event("run_started")
        return context

    def event(self, name: str, **payload: Any) -> None:
        record = {"timestamp": _now(), "event": name, "run_kind": self.run_kind, **payload}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def update_metadata(self, **values: Any) -> None:
        """Merge final state into the local recovery metadata."""

        path = self.root / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(values)
        path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def complete(self) -> None:
        self.update_metadata(status="complete", finished_at=_now())
        self.event("run_completed", output=str(self.output_dir))

    def fail(self, error: BaseException) -> None:
        self.update_metadata(
            status="failed",
            finished_at=_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        self.event("run_failed", error_type=type(error).__name__, error=str(error))


__all__ = ["RUN_KINDS", "RunContext", "RunKind"]
