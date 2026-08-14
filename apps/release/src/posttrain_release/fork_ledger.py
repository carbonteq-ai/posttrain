"""Generate the maintained-fork closure consumed by a framework release.

The release manifest is deliberately not the selection authority.  Direct
package metadata and runtime profiles already select exact bytes; this module
cross-checks the human-audited release receipts against those executable
inputs.  It keeps an unpublished component candidate distinct from an actually
deployed upstream component, so a release is neither falsely cleared nor
needlessly blocked by unrelated local work.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_LEDGER = Path("release/forks.toml")
_TRACKIO = Path("packages/tracking-trackio/pyproject.toml")
_TRAIN = Path("packages/train/pyproject.toml")
_VERL_PROFILE = Path(
    "packages/runtime-images/src/posttrain/runtime_images/containers/"
    "posttrain-job-kinds/verl-py313/profile.toml"
)
_AUTOMATIONBENCH = Path("packages/eval/src/posttrain/eval/programs/automationbench.py")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ForkLedgerEntry:
    """One maintained component in the candidate's supply-chain closure."""

    id: str
    scope: str
    repository: str
    release_tag: str | None
    required: bool
    version: str | None
    revision: str | None
    artifacts: dict[str, str]
    selection_source: str
    deployed_image: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_fork_ledger(repository_root: Path) -> tuple[ForkLedgerEntry, ...]:
    """Return the audited release closure after checking every executable pin."""

    root = repository_root.resolve()
    declared = _declared_entries(root)
    expected = {entry["id"]: entry for entry in declared}
    if set(expected) != {"carbonteq-trackio", "trl", "verl", "vllm", "automationbench", "dstack"}:
        raise ValueError("release/forks.toml must declare the complete maintained-fork closure")

    trackio = _tool_metadata(root / _TRACKIO, "trackio")
    trl = _tool_metadata(root / _TRAIN, "trl")
    profile = _toml(root / _VERL_PROFILE)
    dependencies = _mapping(profile.get("dependencies"), "veRL profile dependencies")

    entries = (
        _package_entry(
            expected["carbonteq-trackio"],
            metadata=trackio,
            package="carbonteq-trackio",
            source=_TRACKIO.as_posix(),
        ),
        _package_entry(expected["trl"], metadata=trl, package="trl", source=_TRAIN.as_posix()),
        _verl_entry(expected["verl"], profile),
        _vllm_entry(expected["vllm"], dependencies),
        _automationbench_entry(expected["automationbench"], root),
        _dstack_entry(expected["dstack"]),
    )
    return entries


def render_fork_ledger(repository_root: Path) -> dict[str, object]:
    """Produce a stable, receipt-safe JSON value for CLI/readiness evidence."""

    return {
        "schema": "posttrain.fork-ledger.v1",
        "entries": [entry.to_dict() for entry in load_fork_ledger(repository_root)],
    }


def _declared_entries(root: Path) -> tuple[dict[str, Any], ...]:
    document = _toml(root / _LEDGER)
    if document.get("schema_version") != 1:
        raise ValueError("release/forks.toml has an unsupported schema version")
    raw_entries = document.get("fork")
    if not isinstance(raw_entries, list):
        raise ValueError("release/forks.toml must contain [[fork]] entries")
    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        entry = _mapping(raw, "fork ledger entry")
        identifier = _string(entry.get("id"), "fork ledger id")
        if any(existing["id"] == identifier for existing in entries):
            raise ValueError(f"release/forks.toml declares {identifier!r} more than once")
        _string(entry.get("scope"), f"fork {identifier!r} scope")
        _string(entry.get("repository"), f"fork {identifier!r} repository")
        if not isinstance(entry.get("required"), bool):
            raise ValueError(f"fork {identifier!r} required must be a boolean")
        release_tag = entry.get("release_tag")
        if not isinstance(release_tag, str):
            raise ValueError(f"fork {identifier!r} release_tag must be a string")
        if entry["required"] and not release_tag:
            raise ValueError(f"required fork {identifier!r} has no immutable release tag")
        entries.append(entry)
    return tuple(entries)


def _package_entry(
    declared: dict[str, Any], *, metadata: dict[str, str], package: str, source: str
) -> ForkLedgerEntry:
    version = _string(metadata.get("version"), f"{package} version")
    tag = _string(metadata.get("release_tag"), f"{package} release tag")
    revision = _revision(metadata.get("source_revision"), f"{package} source revision")
    _matches(declared, tag, f"{package} release tag")
    artifacts = {
        "wheel_sha256": _sha256(metadata.get("wheel_sha256"), f"{package} wheel SHA-256"),
        "sdist_sha256": _sha256(metadata.get("sdist_sha256"), f"{package} source SHA-256"),
    }
    return _entry(declared, version=version, revision=revision, artifacts=artifacts, selection_source=source)


def _verl_entry(declared: dict[str, Any], profile: dict[str, Any]) -> ForkLedgerEntry:
    tag = _string(profile.get("release_tag"), "veRL release tag")
    _matches(declared, tag, "veRL release tag")
    return _entry(
        declared,
        version=tag.removeprefix("carbonteq-v"),
        revision=_revision(profile.get("fork_revision"), "veRL fork revision"),
        artifacts={
            "wheel_sha256": _sha256(profile.get("release_wheel_sha256"), "veRL wheel SHA-256"),
            "sdist_sha256": _sha256(profile.get("release_sdist_sha256"), "veRL source SHA-256"),
        },
        selection_source=_VERL_PROFILE.as_posix(),
    )


def _vllm_entry(declared: dict[str, Any], dependencies: dict[str, Any]) -> ForkLedgerEntry:
    tag = _string(dependencies.get("vllm_release_tag"), "vLLM release tag")
    _matches(declared, tag, "vLLM release tag")
    return _entry(
        declared,
        version=_string(dependencies.get("vllm"), "vLLM version"),
        revision=_revision(dependencies.get("vllm_revision"), "vLLM revision"),
        artifacts={
            "source_archive_sha256": _sha256(
                dependencies.get("vllm_release_source_sha256"), "vLLM source archive SHA-256"
            ),
            "binary_base_wheel_sha256": _sha256(
                dependencies.get("vllm_binary_wheel_sha256"), "vLLM ABI wheel SHA-256"
            ),
        },
        selection_source=_VERL_PROFILE.as_posix(),
    )


def _automationbench_entry(declared: dict[str, Any], root: Path) -> ForkLedgerEntry:
    text = (root / _AUTOMATIONBENCH).read_text(encoding="utf-8")
    environment_revision = _python_constant(text, "AUTOMATIONBENCH_REVISION")
    if environment_revision != "b7bcb591facfcd2b073802f6d7496b24ab9c479e":
        raise ValueError("AutomationBench environment source changed; update the release ledger deliberately")
    return _entry(
        declared,
        version="1.0.5.post1",
        revision="908db2abd4a868acc37ab0850474bff653bea25c",
        artifacts={
            "wheel_sha256": "bd80b4947fbdd60706d9545e79635b79931d89dfc294ed45b01df6886c1f1509",
            "sdist_sha256": "04ccef85e2a83bd26777a10a08702b4fb6a47169352777ab8564fa1bbba9acf6",
            "environment_revision": environment_revision,
        },
        selection_source=_AUTOMATIONBENCH.as_posix(),
    )


def _dstack_entry(declared: dict[str, Any]) -> ForkLedgerEntry:
    if declared["required"]:
        raise ValueError("dstack must not be marked required while production selects the upstream image")
    image = _string(declared.get("deployed_image"), "deployed dstack image")
    if "@sha256:" not in image:
        raise ValueError("deployed dstack image must be digest pinned")
    return _entry(
        declared,
        version="0.20.29",
        revision=None,
        artifacts={},
        selection_source="docs/tooling/dstack/README.md",
        deployed_image=image,
    )


def _entry(
    declared: dict[str, Any],
    *,
    version: str | None,
    revision: str | None,
    artifacts: dict[str, str],
    selection_source: str,
    deployed_image: str | None = None,
) -> ForkLedgerEntry:
    raw_tag = declared.get("release_tag")
    if not isinstance(raw_tag, str):
        raise ValueError(f"fork {declared['id']!r} release tag must be a string")
    tag = raw_tag or None
    return ForkLedgerEntry(
        id=_string(declared.get("id"), "fork ledger id"),
        scope=_string(declared.get("scope"), "fork ledger scope"),
        repository=_string(declared.get("repository"), "fork ledger repository"),
        release_tag=tag,
        required=bool(declared["required"]),
        version=version,
        revision=revision,
        artifacts=artifacts,
        selection_source=selection_source,
        deployed_image=deployed_image,
    )


def _tool_metadata(path: Path, name: str) -> dict[str, str]:
    document = _toml(path)
    tool = _mapping(document.get("tool"), f"{path} tool metadata")
    posttrain = _mapping(tool.get("posttrain"), f"{path} posttrain metadata")
    metadata = _mapping(posttrain.get(name), f"{path} [tool.posttrain.{name}]")
    return {key.replace("-", "_"): value for key, value in metadata.items() if isinstance(value, str)}


def _toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required release input is missing: {path}")
    result = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"invalid TOML object: {path}")
    return result


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a table")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _revision(value: object, context: str) -> str:
    result = _string(value, context)
    if _REVISION.fullmatch(result) is None:
        raise ValueError(f"{context} must be a 40-character lowercase Git revision")
    return result


def _sha256(value: object, context: str) -> str:
    result = _string(value, context)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return result


def _matches(declared: dict[str, Any], actual: str, context: str) -> None:
    expected = _string(declared.get("release_tag"), f"fork {declared['id']!r} release tag")
    if actual != expected:
        raise ValueError(f"{context} {actual!r} does not match release ledger {expected!r}")


def _python_constant(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\s*=\s*\"([^\"]+)\"\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"AutomationBench program does not declare {name}")
    return _revision(match.group(1), name)


__all__ = ["ForkLedgerEntry", "load_fork_ledger", "render_fork_ledger"]
