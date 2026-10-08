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

from .artifacts import verify_index_artifacts

_LEDGER = Path("release/forks.toml")
_TRACKIO = Path("packages/tracking-trackio/pyproject.toml")
_TRAIN = Path("packages/train/pyproject.toml")
_VERL_PROFILE = Path(
    "packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/verl-py313/profile.toml"
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
    if set(expected) != {
        "carbonteq-trackio",
        "trl",
        "carbonteq-renderers",
        "causal-conv1d",
        "verl",
        "vllm",
        "automationbench",
        "dstack",
    }:
        raise ValueError("release/forks.toml must declare the complete maintained-fork closure")

    trackio = _tool_metadata(root / _TRACKIO, "trackio")
    trl = _tool_metadata(root / _TRAIN, "trl")
    renderers = _tool_metadata(root / _TRAIN, "renderers")
    causal_conv1d = _tool_metadata(root / _TRAIN, "causal-conv1d")
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
        _package_entry(
            expected["carbonteq-renderers"],
            metadata=renderers,
            package="carbonteq-renderers",
            source=_TRAIN.as_posix(),
        ),
        _wheel_only_package_entry(
            expected["causal-conv1d"],
            metadata=causal_conv1d,
            package="causal-conv1d",
            root=root,
            source=_TRAIN.as_posix(),
        ),
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


def verify_required_fork_index(repository_root: Path, simple_base_url: str) -> tuple[str, ...]:
    """Prove every required Python fork is byte-identical in one index.

    A pure-Python fork release is a universal wheel plus an sdist. A wheel-only
    release (a rebuilt CUDA extension) names its exact platform wheel and has
    no sdist; only that wheel is checked.
    """

    selected: list[ForkLedgerEntry] = []
    packages: list[str] = []
    artifacts: list[dict[str, str]] = []
    for entry in load_fork_ledger(repository_root):
        wheel_sha256 = entry.artifacts.get("wheel_sha256")
        sdist_sha256 = entry.artifacts.get("sdist_sha256")
        wheel_filename = entry.artifacts.get("wheel_filename")
        if not entry.required or entry.scope not in {"direct-package", "runtime-kind"} or wheel_sha256 is None:
            continue
        if sdist_sha256 is None and wheel_filename is None:
            continue
        if entry.version is None:
            raise ValueError(f"required Python fork {entry.id!r} has no version")
        distribution = entry.id.replace("-", "_")
        packages.append(entry.id)
        artifacts.append(
            {
                "filename": wheel_filename or f"{distribution}-{entry.version}-py3-none-any.whl",
                "sha256": wheel_sha256,
            }
        )
        if sdist_sha256 is not None:
            artifacts.append({"filename": f"{distribution}-{entry.version}.tar.gz", "sha256": sdist_sha256})
        selected.append(entry)
    verify_index_artifacts(packages, artifacts, simple_base_url)
    return tuple(entry.id for entry in selected)


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


def _package_entry(declared: dict[str, Any], *, metadata: dict[str, str], package: str, source: str) -> ForkLedgerEntry:
    version = _string(metadata.get("version"), f"{package} version")
    tag = _string(metadata.get("release_tag"), f"{package} release tag")
    revision = _revision(metadata.get("source_revision"), f"{package} source revision")
    _matches(declared, tag, f"{package} release tag")
    artifacts = {
        "wheel_sha256": _sha256(metadata.get("wheel_sha256"), f"{package} wheel SHA-256"),
        "sdist_sha256": _sha256(metadata.get("sdist_sha256"), f"{package} source SHA-256"),
    }
    return _entry(declared, version=version, revision=revision, artifacts=artifacts, selection_source=source)


def _wheel_only_package_entry(
    declared: dict[str, Any], *, metadata: dict[str, str], package: str, root: Path, source: str
) -> ForkLedgerEntry:
    """A fork released as one retained platform wheel, with no sdist."""

    version = _string(metadata.get("version"), f"{package} version")
    tag = _string(metadata.get("release_tag"), f"{package} release tag")
    if tag != f"carbonteq-v{version}":
        raise ValueError(f"{package} release tag {tag!r} does not match version {version!r}")
    _matches(declared, tag, f"{package} release tag")
    if "sdist_sha256" in metadata:
        raise ValueError(f"{package} is a wheel-only fork release and must not record an sdist")
    wheel_filename = _string(metadata.get("wheel_filename"), f"{package} wheel filename")
    distribution = package.replace("-", "_")
    if not wheel_filename.startswith(f"{distribution}-{version}-") or not wheel_filename.endswith(".whl"):
        raise ValueError(f"{package} wheel filename {wheel_filename!r} does not name version {version!r}")
    project = _mapping(_toml(root / source).get("project"), f"{source} project")
    extras = _mapping(project.get("optional-dependencies", {}), f"{source} optional dependencies")
    requirements = [*project.get("dependencies", []), *(item for group in extras.values() for item in group)]
    selected = f"{package}=={version}"
    if not any(isinstance(item, str) and item.split(";", 1)[0].strip() == selected for item in requirements):
        raise ValueError(f"{source} does not select {selected}")
    return _entry(
        declared,
        version=version,
        revision=_revision(metadata.get("source_revision"), f"{package} source revision"),
        artifacts={
            "wheel_sha256": _sha256(metadata.get("wheel_sha256"), f"{package} wheel SHA-256"),
            "wheel_filename": wheel_filename,
        },
        selection_source=source,
    )


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
            "binary_base_wheel_sha256": _sha256(dependencies.get("vllm_binary_wheel_sha256"), "vLLM ABI wheel SHA-256"),
        },
        selection_source=_VERL_PROFILE.as_posix(),
    )


def _automationbench_entry(declared: dict[str, Any], root: Path) -> ForkLedgerEntry:
    text = (root / _AUTOMATIONBENCH).read_text(encoding="utf-8")
    environment_revision = _python_constant(text, "AUTOMATIONBENCH_REVISION")
    if environment_revision != "d430dd87c8d6a37c0520acc3624d74f2de31659e":
        raise ValueError("AutomationBench environment source changed; update the release ledger deliberately")
    return _entry(
        declared,
        version="1.0.5.post3",
        revision="9bfbdd703bba4a8c06c831b35f55ce442347cf12",
        artifacts={
            "wheel_sha256": "682f9852280ae0bd75152f35b1a833437a7ac6bf28d53b36774fc7023f8ab870",
            "sdist_sha256": "036dc954f533a0809d731a922eb670d7e7b598ea4742b535d0146e7777bb3763",
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
