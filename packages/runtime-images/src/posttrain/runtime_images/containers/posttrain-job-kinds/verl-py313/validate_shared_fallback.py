"""Validate uv's partial backend sync against the inherited control environment."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import Distribution, distributions
from pathlib import Path
from typing import NoReturn


class FallbackError(ValueError):
    """Raised when the shared-package fallback is inconsistent or unsafe."""


@dataclass(frozen=True, slots=True)
class SharingPolicy:
    distributions: tuple[str, ...]


def normalize_distribution_name(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if not normalized or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", normalized) is None:
        raise FallbackError(f"invalid distribution name: {name!r}")
    return normalized


def load_policy(path: Path) -> SharingPolicy:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise FallbackError(f"cannot read sharing policy {path}: {exc}") from exc
    if document.get("schema_version") != 1:
        raise FallbackError("shared-heavy policy schema_version must be 1")
    raw_names = document.get("distributions")
    if not isinstance(raw_names, list) or not raw_names or not all(isinstance(name, str) for name in raw_names):
        raise FallbackError("shared-heavy policy distributions must be a non-empty string list")
    normalized = tuple(normalize_distribution_name(name) for name in raw_names)
    if normalized != tuple(sorted(set(normalized))) or tuple(raw_names) != normalized:
        raise FallbackError("shared-heavy policy names must be normalized, unique, and sorted")
    return SharingPolicy(distributions=normalized)


def _inventory(site_packages: Path) -> dict[str, Distribution]:
    try:
        root = site_packages.resolve(strict=True)
    except OSError as exc:
        raise FallbackError(f"site-packages root is unavailable: {site_packages}") from exc
    if not root.is_dir():
        raise FallbackError(f"site-packages root is not a directory: {site_packages}")
    found: dict[str, Distribution] = {}
    for distribution in distributions(path=[str(root)]):
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = normalize_distribution_name(raw_name)
        if name in found:
            raise FallbackError(f"duplicate installed distribution in {root}: {name}")
        found[name] = distribution
    return found


def _locked_versions(lock_path: Path, selected: tuple[str, ...]) -> dict[str, str]:
    try:
        document = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise FallbackError(f"cannot read backend lock {lock_path}: {exc}") from exc
    wanted = set(selected)
    found: dict[str, str] = {}
    for package in document.get("package", ()):
        if not isinstance(package, dict):
            continue
        raw_name = package.get("name")
        version = package.get("version")
        if not isinstance(raw_name, str) or not isinstance(version, str):
            continue
        name = normalize_distribution_name(raw_name)
        if name not in wanted:
            continue
        if name in found and found[name] != version:
            raise FallbackError(f"backend lock selects multiple versions for {name}")
        found[name] = version
    missing = sorted(wanted - found.keys())
    if missing:
        raise FallbackError(f"shared distributions are missing from backend lock: {', '.join(missing)}")
    return found


def validate_fallback(
    *,
    control_site: Path,
    backend_site: Path,
    backend_lock: Path,
    policy: SharingPolicy,
    fallback_file: Path,
) -> dict[str, object]:
    control_root = control_site.resolve(strict=True)
    backend_root = backend_site.resolve(strict=True)
    if control_root == backend_root:
        raise FallbackError("control and backend site-packages roots must be separate")
    expected_fallback = f"{control_root}\n"
    if fallback_file.read_text(encoding="utf-8") != expected_fallback:
        raise FallbackError("backend fallback .pth must contain only the control site-packages path")

    control = _inventory(control_root)
    backend = _inventory(backend_root)
    locked = _locked_versions(backend_lock, policy.distributions)
    packages: list[dict[str, str]] = []
    for name in policy.distributions:
        if name in backend:
            raise FallbackError(f"{name}: uv partial sync unexpectedly installed a backend copy")
        distribution = control.get(name)
        if distribution is None:
            raise FallbackError(f"{name}: inherited control distribution is missing")
        if distribution.version != locked[name]:
            raise FallbackError(
                f"{name}: inherited version {distribution.version!r} does not match backend lock {locked[name]!r}"
            )
        packages.append({"name": name, "version": distribution.version})
    return {"packages": packages, "schema_version": 1, "strategy": "uv-partial-sync-pth-fallback"}


def _die(message: str) -> NoReturn:
    print(f"shared-fallback: {message}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-site", type=Path, required=True)
    parser.add_argument("--backend-site", type=Path, required=True)
    parser.add_argument("--backend-lock", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--fallback-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_fallback(
            control_site=args.control_site,
            backend_site=args.backend_site,
            backend_lock=args.backend_lock,
            policy=load_policy(args.policy),
            fallback_file=args.fallback_file,
        )
        payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        args.report.write_text(payload, encoding="utf-8")
    except (FallbackError, OSError) as exc:
        _die(str(exc))
    selected = report["packages"]
    if not isinstance(selected, list):
        _die("internal report packages value is invalid")
    print(f"shared-fallback: packages={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
