"""Materialize internal runtime-package receipts into the OCI constraint lock."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WORKSPACE_LOCK = Path(
    "packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/locks/workspace.lock.txt"
)
_PROFILES = Path("packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/profiles")
_REQUIREMENT_START = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?(?:==|\s+@\s+)")
_PROFILE_PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^ ;]+)$")
_INTERNAL_INDEXES = (
    "https://pypi.lan/carbonteq/dev/",
    "https://pypi.lan/carbonteq/stable/",
)


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
    "RuntimeProfileSynchronization",
    "materialize_runtime_lock",
    "render_runtime_lock",
    "synchronize_runtime_profile_pins",
]
