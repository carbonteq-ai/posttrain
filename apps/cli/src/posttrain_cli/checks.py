"""Readiness checks reported by `posttrain doctor`.

Checks were previously built as dictionary literals inline in the command body,
which offered nowhere to express a condition that is reportable but not fatal.
A missing registry blocks submission while leaving validation entirely usable,
so it must be sayable without failing the command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from posttrain.common import ContractError
from posttrain.runtime_images.manifest import ManifestError, load_manifest

from .context import CliState
from .execution_config import (
    REGISTRY_ENVIRONMENT_VARIABLE,
    TRUST_BUNDLE_ENVIRONMENT_VARIABLE,
    WELL_KNOWN_TRUST_BUNDLE,
    configured_registry_prefix,
    load_local_execution_config,
    resolve_trust_bundle,
)

_CERTIFICATE_HEADER = "-----BEGIN CERTIFICATE-----"

type CheckStatus = Literal["ok", "warn", "error"]


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: CheckStatus
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "message": self.message}


def registry_check(state: CliState) -> Check:
    """Report whether this project has somewhere to publish its job images."""
    try:
        layout = state.layout()
    except (ContractError, OSError) as error:
        return Check("registry", "error", str(error))

    try:
        configuration = load_local_execution_config(layout)
    except (ContractError, OSError) as error:
        return Check("registry", "error", str(error))

    if configuration.registry is None:
        return Check(
            "registry",
            "warn",
            f"not configured; set {REGISTRY_ENVIRONMENT_VARIABLE} to an OCI registry "
            "prefix before submitting jobs (validation works without it)",
        )

    source = (
        f"{REGISTRY_ENVIRONMENT_VARIABLE}={configured_registry_prefix()}"
        if configured_registry_prefix() is not None
        else "execution configuration"
    )
    return Check("registry", "ok", f"{configuration.registry.repository} (from {source})")


def _in_system_trust(bundle: Path) -> bool | None:
    """Whether the host's own trust store already contains this authority.

    Returns None when the system store cannot be read, because absent evidence
    is not evidence of absence and reporting a problem would be a guess.
    """
    system = Path("/etc/ssl/certs/ca-certificates.crt")
    try:
        installed = system.read_text(encoding="utf-8", errors="ignore")
        declared = bundle.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    blocks = [block.strip() for block in declared.split(_CERTIFICATE_HEADER)[1:]]
    if not blocks:
        return None
    return all(block in installed for block in blocks)


def trust_check(state: CliState) -> Check:
    """Report which additional certificate authority jobs will be given.

    Host tools and job containers trust different things: `uv` and the Docker
    daemon read the machine's own store, while a container inherits nothing.
    A machine can therefore run jobs perfectly while failing to reach the same
    registry from the shell, so both are reported rather than one.
    """
    try:
        layout = state.layout()
        configuration = load_local_execution_config(layout)
        configured = configuration.local.trust_bundle if configuration.local is not None else None
        resolved = resolve_trust_bundle(configured)
    except (ContractError, OSError) as error:
        return Check("trust", "error", str(error))

    if resolved.path is None:
        return Check(
            "trust",
            "ok",
            "no internal certificate authority configured; jobs trust public authorities only "
            f"(install one at {WELL_KNOWN_TRUST_BUNDLE} or set {TRUST_BUNDLE_ENVIRONMENT_VARIABLE})",
        )

    origin = {
        "configured": "providers trust_bundle",
        "environment": TRUST_BUNDLE_ENVIRONMENT_VARIABLE,
        "convention": "well-known path",
    }[resolved.source]
    installed = _in_system_trust(resolved.path)
    if installed is False:
        return Check(
            "trust",
            "warn",
            f"{resolved.path} (from {origin}) will be added to jobs, but is not in this "
            "machine's own trust store, so host tools such as uv and docker will still reject it",
        )
    return Check("trust", "ok", f"{resolved.path} (from {origin}), added to what each job image already trusts")


def runtime_images_check(state: CliState) -> Check:
    """Fail when a configured job-kind image is not the one this release pins.

    This is a local comparison and makes no network request. Confirming that
    the registry's copy still carries the expected lock digest additionally
    requires `posttrain runtime images verify`.
    """
    try:
        layout = state.layout()
        configuration = load_local_execution_config(layout)
    except (ContractError, OSError) as error:
        return Check("runtime_images", "error", str(error))

    if configuration.registry is None:
        # Reported separately from the registry check on purpose. Staying silent
        # here would read as "images verified", when in fact no image identity
        # was checked at all.
        return Check(
            "runtime_images",
            "warn",
            "not verified: no registry is configured, so no job-kind image "
            "identity was checked against this framework release",
        )

    try:
        manifest = load_manifest()
    except ManifestError as error:
        return Check("runtime_images", "error", str(error))

    drifted: list[str] = []
    for variant, image in sorted(configuration.registry.kind_images.items()):
        published = manifest.kinds.get(variant)
        if published is None:
            # Not part of this release, so there is nothing to disagree with.
            continue
        configured_digest = image.value.rsplit("@", 1)[1]
        if configured_digest != published.digest:
            drifted.append(f"{variant}: configured {configured_digest}, release pins {published.digest}")

    if drifted:
        return Check(
            "runtime_images",
            "error",
            "configured job-kind images disagree with this framework release; "
            "qualification evidence produced on them is not comparable. " + "; ".join(drifted),
        )
    return Check(
        "runtime_images",
        "ok",
        f"{len(manifest.kinds)} variants match release {manifest.framework_version}",
    )


def catalog_overlay_check(state: CliState) -> Check:
    """Warn when YAML files under a catalog overlay are missing from layer.yaml."""
    try:
        layout = state.layout()
    except (ContractError, OSError) as error:
        return Check("catalog_overlays", "error", str(error))

    orphaned: list[str] = []
    for overlay in layout.catalog_overlays:
        if not overlay.is_dir():
            continue
        manifest = overlay / "layer.yaml"
        if not manifest.is_file():
            orphaned.append(f"{overlay}: missing layer.yaml")
            continue
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            orphaned.append(f"{overlay}: unreadable layer.yaml")
            continue
        declared = set()
        if isinstance(payload, dict) and isinstance(payload.get("files"), list):
            declared = {str(name) for name in payload["files"]}
        on_disk = {
            path.name
            for path in overlay.iterdir()
            if path.is_file() and path.suffix in {".yaml", ".yml"} and path.name != "layer.yaml"
        }
        missing = sorted(on_disk - declared)
        if missing:
            orphaned.append(f"{overlay.name}: {', '.join(missing)} not listed in layer.yaml")
    if orphaned:
        return Check(
            "catalog_overlays",
            "warn",
            "overlay YAML must be listed in layer.yaml or it is ignored: " + "; ".join(orphaned),
        )
    return Check("catalog_overlays", "ok", "overlay files match layer.yaml")


__all__ = [
    "Check",
    "CheckStatus",
    "catalog_overlay_check",
    "registry_check",
    "runtime_images_check",
    "trust_check",
]
