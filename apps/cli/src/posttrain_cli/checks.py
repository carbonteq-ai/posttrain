"""Readiness checks reported by `posttrain doctor`.

Checks were previously built as dictionary literals inline in the command body,
which offered nowhere to express a condition that is reportable but not fatal.
A missing registry blocks submission while leaving validation entirely usable,
so it must be sayable without failing the command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from posttrain.common import ContractError
from posttrain.runtime_images.manifest import ManifestError, load_manifest

from .context import CliState
from .execution_config import (
    REGISTRY_ENVIRONMENT_VARIABLE,
    configured_registry_prefix,
    load_local_execution_config,
)

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
            drifted.append(
                f"{variant}: configured {configured_digest}, release pins {published.digest}"
            )

    if drifted:
        return Check(
            "runtime_images",
            "error",
            "configured job-kind images disagree with this framework release; "
            "qualification evidence produced on them is not comparable. "
            + "; ".join(drifted),
        )
    return Check(
        "runtime_images",
        "ok",
        f"{len(manifest.kinds)} variants match release {manifest.framework_version}",
    )


__all__ = ["Check", "CheckStatus", "registry_check", "runtime_images_check"]
