"""Materialize internal runtime-package receipts into the OCI constraint lock."""

from __future__ import annotations

import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WORKSPACE_LOCK = Path(
    "packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/locks/workspace.lock.txt"
)
_PROFILES = Path("packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/profiles")
_BASE_REQUIREMENTS = Path("packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-base/requirements.txt")
_RUNTIME_LOCKS = Path("packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/locks")
_REQUIREMENT_START = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?(?:==|\s+@\s+)")
_PROFILE_PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^ ;]+)$")
_HASH = re.compile(r"--hash=sha256:(?P<digest>[0-9a-f]{64})")
_DIRECT_HASH = re.compile(r"#sha256=(?P<digest>[0-9a-f]{64})")
_INTERNAL_INDEXES = (
    "https://pypi.lan/carbonteq/dev/",
    "https://pypi.lan/carbonteq/stable/",
)
_WORKSPACE_EXPORT_COMMAND = (
    "uv export --all-packages --all-extras --locked --no-emit-workspace "
    "--no-dev --format requirements-txt --emit-index-url --no-header"
)
_NARROW_LOCK_HEADER = (
    "# Generated from the repository's exact uv.lock and workspace.lock.txt. Do not hand edit.\n"
    "# The selected profile roots and their transitive lock graph are projected without re-resolving.\n\n"
)
_EXACT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?==(?P<version>[^ ;]+)$")
_GIT_REVISION = re.compile(r"git\+https://[^@\s]+@(?P<revision>[0-9a-f]{40})(?:#\S+)?$")


@dataclass(frozen=True, slots=True)
class RuntimeLockMaterialization:
    path: Path
    changed: bool
    packages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeProfileSynchronization:
    """Result of aligning authored profile pins with an isolated candidate lock."""

    changed_profiles: tuple[Path, ...]
    packages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeLockCompilation:
    """Result of regenerating the narrow runtime closures from ``uv.lock``."""

    changed_paths: tuple[Path, ...]
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RuntimeWorkspaceLockExport:
    """Result of exporting the runtime constraint authority from ``uv.lock``."""

    path: Path
    changed: bool


_NARROW_LOCK_INPUTS: dict[str, tuple[Path, ...]] = {
    "base": (_BASE_REQUIREMENTS,),
    "supervised": (_PROFILES / "common.txt", _PROFILES / "supervised.txt"),
    "online-rl-trl-py312": (
        _PROFILES / "common.txt",
        _PROFILES / "online-rl-trl-py312.txt",
    ),
    "online-rl-verl-py313": (
        _PROFILES / "common.txt",
        _PROFILES / "online-rl-verl-py313-control.txt",
    ),
    "eval": (_PROFILES / "common.txt", _PROFILES / "eval.txt"),
    "serve": (_PROFILES / "common.txt", _PROFILES / "serve.txt"),
}


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _profile_pins(root: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for profile in sorted((root / _PROFILES).glob("*.txt")):
        if profile.name in {"build-tools.in", "transform.txt"}:
            continue
        for raw_line in profile.read_text(encoding="utf-8").splitlines():
            match = _PROFILE_PIN.fullmatch(raw_line.strip())
            if match is None:
                continue
            name = _normalized(match.group("name"))
            version = match.group("version")
            prior = pins.setdefault(name, version)
            if prior != version:
                raise ValueError(f"runtime profiles pin {name!r} to both {prior!r} and {version!r}")
    return pins


def _internal_wheels(root: Path) -> dict[str, tuple[str, str, str]]:
    document = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    packages: dict[str, tuple[str, str, str]] = {}
    for package in document.get("package", []):
        if not isinstance(package, dict):
            continue
        source = package.get("source")
        registry = source.get("registry") if isinstance(source, dict) else None
        if not isinstance(registry, str) or not registry.startswith(_INTERNAL_INDEXES):
            continue
        name = package.get("name")
        version = package.get("version")
        wheels = package.get("wheels")
        if not isinstance(name, str) or not isinstance(version, str) or not isinstance(wheels, list):
            raise ValueError(f"internal uv.lock package has incomplete metadata: {name!r}")
        candidates = [wheel for wheel in wheels if isinstance(wheel, dict) and isinstance(wheel.get("url"), str)]
        universal = [wheel for wheel in candidates if str(wheel["url"]).endswith("-py3-none-any.whl")]
        selected: dict[str, Any] | None = (universal or candidates)[0] if candidates else None
        if selected is None:
            raise ValueError(f"internal runtime package {name!r} has no wheel receipt in uv.lock")
        url = selected["url"]
        digest = selected.get("hash")
        if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError(f"internal runtime package {name!r} has no SHA-256 wheel receipt")
        packages[_normalized(name)] = (version, url, digest.removeprefix("sha256:"))
    return packages


def render_runtime_lock(repository_root: Path) -> tuple[str, tuple[str, ...]]:
    """Render internal package entries without re-resolving the public closure."""

    root = repository_root.resolve()
    path = root / _WORKSPACE_LOCK
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _REQUIREMENT_START.match(line)
        if match is not None:
            starts.append((index, _normalized(match.group("name"))))

    pins = _profile_pins(root)
    internal = _internal_wheels(root)
    selected = {name: receipt for name, receipt in internal.items() if name in pins}
    for name, (version, _, _) in selected.items():
        if pins[name] != version:
            raise ValueError(
                f"runtime profile pins {name}=={pins[name]}, but uv.lock records the published package as {version}"
            )

    locations = {name: index for index, name in starts if name in selected}
    missing = set(selected) - set(locations)
    if missing:
        raise ValueError("runtime constraint lock omits internal profile packages: " + ", ".join(sorted(missing)))

    rendered = list(lines)
    for position in range(len(starts) - 1, -1, -1):
        start, name = starts[position]
        if name not in selected:
            continue
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        annotation = next((index for index in range(start + 1, end) if lines[index].lstrip().startswith("#")), end)
        _, url, digest = selected[name]
        rendered[start:annotation] = [f"{name} @ {url}#sha256={digest}\n"]

    return "".join(rendered), tuple(sorted(selected))


def materialize_runtime_lock(repository_root: Path, *, check: bool = False) -> RuntimeLockMaterialization:
    root = repository_root.resolve()
    path = root / _WORKSPACE_LOCK
    rendered, packages = render_runtime_lock(root)
    changed = rendered != path.read_text(encoding="utf-8")
    if changed and not check:
        path.write_text(rendered, encoding="utf-8")
    return RuntimeLockMaterialization(path=path, changed=changed, packages=packages)


def export_runtime_workspace_lock(
    repository_root: Path, *, check: bool = False
) -> RuntimeWorkspaceLockExport:
    """Export the authoritative workspace resolution for image-lock compilation.

    ``uv.lock`` is the only solver authority.  Do not rebuild this file by
    compiling one selected image profile: that can omit a transitive
    requirement (or select a newer artifact) used by another profile.
    Workspace packages are omitted because image closures install published
    internal wheels, not editable source paths.
    """

    root = repository_root.resolve()
    path = root / _WORKSPACE_LOCK
    completed = subprocess.run(
        [
            "uv",
            "export",
            "--all-packages",
            "--all-extras",
            "--locked",
            "--no-emit-workspace",
            "--no-dev",
            "--format",
            "requirements-txt",
            "--emit-index-url",
            "--no-header",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    exported = completed.stdout
    if not exported.strip():
        raise ValueError("uv export produced an empty runtime workspace lock")
    rendered = (
        "# Generated from the repository's exact uv.lock. Do not hand edit.\n"
        f"# {_WORKSPACE_EXPORT_COMMAND}\n\n"
        f"{exported.rstrip()}\n"
    )
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    changed = current != rendered
    if changed and not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    return RuntimeWorkspaceLockExport(path=path, changed=changed)


def compile_runtime_locks(repository_root: Path, *, check: bool = False) -> RuntimeLockCompilation:
    """Generate complete, per-image dependency closures from the workspace lock.

    The workspace lock remains the solver authority. The published base and
    kind images consume smaller closures, but this function *projects* the
    already resolved ``uv.lock`` graph rather than invoking a solver again.
    A constrained solve can still discover a later artifact for the same
    version (or a transitive dependency omitted by an older export), which is
    not an immutable image receipt.
    """

    root = repository_root.resolve()
    constraint = root / _WORKSPACE_LOCK
    if not constraint.is_file():
        raise ValueError(f"runtime workspace lock does not exist: {constraint}")
    workspace_lock = constraint.read_text(encoding="utf-8")
    lock_document = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    paths: list[Path] = []
    changed: list[Path] = []
    for name, inputs in _NARROW_LOCK_INPUTS.items():
        output = root / _RUNTIME_LOCKS / f"{name}.lock.txt"
        missing = [path for path in inputs if not (root / path).is_file()]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise ValueError(f"runtime lock {name!r} has missing input(s): {rendered}")
        rendered = _project_runtime_closure(
            root,
            inputs=inputs,
            workspace_lock=workspace_lock,
            lock_document=lock_document,
        )
        current = output.read_text(encoding="utf-8") if output.is_file() else ""
        if current != rendered:
            changed.append(output)
            if not check:
                output.write_text(rendered, encoding="utf-8")
        paths.append(output)
    return RuntimeLockCompilation(changed_paths=tuple(changed), paths=tuple(paths))


def _project_runtime_closure(
    root: Path,
    *,
    inputs: tuple[Path, ...],
    workspace_lock: str,
    lock_document: dict[str, Any],
) -> str:
    """Return a profile's exact Linux/Python 3.13 closure from the lock graph."""

    roots = _profile_roots(root, inputs)
    packages = lock_document.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock does not contain a package graph")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for raw_package in packages:
        if not isinstance(raw_package, dict) or not isinstance(raw_package.get("name"), str):
            continue
        by_name.setdefault(_normalized(raw_package["name"]), []).append(raw_package)

    selected: set[str] = set()
    pending: list[str] = []
    for name, expected in roots.items():
        candidates = by_name.get(name, [])
        if expected is not None:
            candidates = [candidate for candidate in candidates if candidate.get("version") == expected]
        if not candidates:
            detail = f"=={expected}" if expected is not None else ""
            raise ValueError(f"runtime profile root {name}{detail} is absent from uv.lock")
        pending.append(name)

    while pending:
        name = pending.pop()
        if name in selected:
            continue
        candidates = by_name.get(name, [])
        if not candidates:
            raise ValueError(f"uv.lock closure references missing package {name!r}")
        selected.add(name)
        for package in candidates:
            dependencies = package.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError(f"uv.lock package {name!r} has malformed dependencies")
            for dependency in dependencies:
                if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                    raise ValueError(f"uv.lock package {name!r} has malformed dependency")
                marker = dependency.get("marker")
                if isinstance(marker, str) and not _linux_python313_marker_applies(marker):
                    continue
                pending.append(_normalized(dependency["name"]))

    workspace_blocks = _requirement_blocks(workspace_lock)
    missing = sorted(selected - set(workspace_blocks))
    if missing:
        raise ValueError("workspace.lock omits packages selected by uv.lock: " + ", ".join(missing))

    index_lines = [
        line for line in workspace_lock.splitlines(keepends=True) if line.startswith(("--index-url ", "--extra-index-url "))
    ]
    selected_blocks: list[str] = []
    for block in _requirement_blocks_in_order(workspace_lock):
        requirement_name = _requirement_name(block)
        marker = _requirement_marker(block)
        if requirement_name in selected and (marker is None or _linux_python313_marker_applies(marker)):
            selected_blocks.append(block.rstrip() + "\n")
    if not selected_blocks:
        raise ValueError("runtime lock projection selected no installable requirements")
    return _NARROW_LOCK_HEADER + "".join(index_lines) + "\n" + "\n".join(selected_blocks)


def _profile_roots(root: Path, inputs: tuple[Path, ...]) -> dict[str, str | None]:
    """Read exact requirement roots and recursively include ``-r`` files."""

    roots: dict[str, str | None] = {}
    visited: set[Path] = set()

    def visit(relative: Path) -> None:
        path = (root / relative).resolve()
        if path in visited:
            return
        visited.add(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                visit(relative.parent / line.removeprefix("-r ").strip())
                continue
            match = _EXACT_PIN.fullmatch(line)
            if match is not None:
                name = _normalized(match.group("name"))
                version: str | None = match.group("version")
            else:
                requirement = _REQUIREMENT_START.match(line)
                if requirement is None:
                    raise ValueError(f"runtime profile {relative} contains an unsupported requirement: {line}")
                name = _normalized(requirement.group("name"))
                version = None
                revision = _GIT_REVISION.search(line)
                if revision is not None:
                    version = None
            prior = roots.setdefault(name, version)
            if prior != version:
                raise ValueError(f"runtime profiles select conflicting roots for {name!r}: {prior!r}, {version!r}")

    for input_path in inputs:
        visit(input_path)
    return roots


def _linux_python313_marker_applies(marker: str) -> bool:
    """Evaluate a locked dependency marker for the released kind platform."""

    from packaging.markers import Marker

    return Marker(marker).evaluate(
        {
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "python_full_version": "3.13.14",
            "python_version": "3.13",
            "sys_platform": "linux",
        }
    )


def _requirement_blocks_in_order(text: str) -> tuple[str, ...]:
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if _REQUIREMENT_START.match(line) is not None]
    return tuple(
        "".join(lines[start : starts[position + 1] if position + 1 < len(starts) else len(lines)])
        for position, start in enumerate(starts)
    )


def _requirement_name(block: str) -> str:
    first = block.lstrip().splitlines()[0]
    match = _REQUIREMENT_START.match(first)
    if match is None:
        raise ValueError(f"invalid requirement block: {first!r}")
    return _normalized(match.group("name"))


def _requirement_marker(block: str) -> str | None:
    first = block.lstrip().splitlines()[0].rstrip(" \\")
    _, separator, marker = first.partition(" ; ")
    return marker if separator else None


def _restrict_hashes_to_workspace(rendered: str, workspace_lock: str) -> str:
    """Keep only artifact hashes already admitted by the solver lock.

    ``uv pip compile --generate-hashes`` looks up every currently published
    artifact for a resolved version.  That can widen a narrow lock after the
    workspace resolution was made, which would let an image install a byte the
    authored lock never reviewed.  Versions and direct references are checked
    below too; this function limits hash-mode acceptance to the authoritative
    workspace receipt.
    """

    workspace = _requirement_blocks(workspace_lock)
    lines = rendered.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _REQUIREMENT_START.match(line)
        if match is not None:
            starts.append((index, _normalized(match.group("name"))))
    result: list[str] = []
    cursor = 0
    for position, (start, name) in enumerate(starts):
        result.extend(lines[cursor:start])
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]
        expected_candidates = workspace.get(name, ())
        if not expected_candidates:
            raise ValueError(f"narrow runtime lock resolves {name!r}, which is absent from workspace.lock")
        candidate_text = "".join(block)
        observed = _artifact_hashes(candidate_text)
        if " @ " in _requirement_identity(candidate_text):
            expected = next(
                (item for item in expected_candidates if observed & _artifact_hashes(item)),
                None,
            )
        else:
            expected = next(
                (item for item in expected_candidates if _requirement_identity(item) == _requirement_identity(candidate_text)),
                None,
            )
        if expected is None:
            raise ValueError(f"narrow runtime lock resolves {name!r} differently from workspace.lock")
        admitted = _artifact_hashes(expected)
        if observed and not admitted:
            raise ValueError(f"workspace.lock has no artifact hashes for narrow runtime package {name!r}")
        result.extend(
            line
            for line in block
            if not ((match := _HASH.search(line)) is not None and match.group("digest") not in admitted)
        )
        cursor = end
    result.extend(lines[cursor:])
    return "".join(result)


def _requirement_blocks(text: str) -> dict[str, tuple[str, ...]]:
    lines = text.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _REQUIREMENT_START.match(line)
        if match is not None:
            starts.append((index, _normalized(match.group("name"))))
    blocks: dict[str, list[str]] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        blocks.setdefault(name, []).append("".join(lines[start:end]))
    return {name: tuple(entries) for name, entries in blocks.items()}


def _artifact_hashes(block: str) -> set[str]:
    """Read both pip hash-mode and immutable direct-wheel receipts."""

    return set(_HASH.findall(block)) | set(_DIRECT_HASH.findall(block))


def _requirement_identity(block: str) -> str:
    """Compare name/version/direct URL without platform-only marker variance."""

    first = block.lstrip().splitlines()[0].rstrip(" \\")
    return first.split(" ;", 1)[0]


def synchronize_runtime_profile_pins(repository_root: Path) -> RuntimeProfileSynchronization:
    """Align internal runtime-profile pins with the already-resolved lock.

    This is deliberately for a disposable release-candidate checkout. Authored
    source keeps its stable consumer pins until promotion, while a development
    candidate needs profile constraints that agree with its dev-channel lock.
    """

    root = repository_root.resolve()
    internal = _internal_wheels(root)
    changed: list[Path] = []
    selected: set[str] = set()
    for profile in sorted((root / _PROFILES).glob("*.txt")):
        original = profile.read_text(encoding="utf-8")
        rendered: list[str] = []
        profile_changed = False
        for line in original.splitlines(keepends=True):
            match = _PROFILE_PIN.fullmatch(line.strip())
            name = _normalized(match.group("name")) if match is not None else None
            if name is None or name not in internal:
                rendered.append(line)
                continue
            assert match is not None
            version = internal[name][0]
            replacement = f"{match.group('name')}=={version}"
            newline = "\n" if line.endswith("\n") else ""
            rendered.append(replacement + newline)
            selected.add(name)
            profile_changed |= replacement != line.strip()
        if profile_changed:
            profile.write_text("".join(rendered), encoding="utf-8")
            changed.append(profile)
    return RuntimeProfileSynchronization(
        changed_profiles=tuple(changed), packages=tuple(sorted(selected))
    )


__all__ = [
    "RuntimeLockMaterialization",
    "RuntimeLockCompilation",
    "RuntimeWorkspaceLockExport",
    "compile_runtime_locks",
    "export_runtime_workspace_lock",
    "RuntimeProfileSynchronization",
    "materialize_runtime_lock",
    "render_runtime_lock",
    "synchronize_runtime_profile_pins",
]
