"""Verify that configured runtime images match the installed framework.

A job-kind image is only valid for the dependency closure it was built from.
When a framework pin moves but its published image does not, every run against
that image is silently qualifying the wrong software. This module makes that
disagreement observable, and is shared by `runtime images verify`, `doctor`,
and the pre-submission check in `job run`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from posttrain.common import ContractError
from posttrain.runtime_images.manifest import (
    ManifestError,
    PublishedManifest,
    load_manifest,
)
from posttrain_execution_buildkit import (
    LOCK_DIGEST_LABEL,
    ImageInspector,
    RemoteImageNotFoundError,
    RuntimeImageInspector,
)

from .execution_config import RegistryBinding

type VerificationStatus = Literal["ok", "missing", "drifted", "unreachable"]

_JOB_KIND_LEVEL = "job-kind"


@dataclass(frozen=True, slots=True)
class VariantVerification:
    """The outcome of checking one variant against a registry."""

    variant: str
    reference: str
    expected_lock_digest: str
    status: VerificationStatus
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def expected_images(registry: RegistryBinding) -> Mapping[str, str]:
    """Return the digest-pinned reference this project will actually use."""
    return {variant: image.value for variant, image in sorted(registry.kind_images.items())}


def verify_variant(
    variant: str,
    reference: str,
    *,
    manifest: PublishedManifest,
    inspector: ImageInspector,
) -> VariantVerification:
    """Check one published image against what the installed framework expects."""
    try:
        expected = manifest.expected_lock_digest(variant)
    except ManifestError:
        # A locally declared variant the release does not publish, such as a
        # release-blocked backend. There is no shipped lock to compare against,
        # so this cannot be verified either way and must not be called drift.
        return VariantVerification(
            variant=variant,
            reference=reference,
            expected_lock_digest="",
            status="ok",
            detail="not published by this release; not verified",
        )

    try:
        facts = inspector.inspect(reference)
    except RemoteImageNotFoundError:
        return VariantVerification(
            variant=variant,
            reference=reference,
            expected_lock_digest=expected,
            status="missing",
            detail="not present in the registry",
        )
    except (RuntimeError, OSError) as error:
        return VariantVerification(
            variant=variant,
            reference=reference,
            expected_lock_digest=expected,
            status="unreachable",
            detail=f"registry could not be queried: {error}",
        )

    # Checked before the lock digest, and deliberately so: every level of the
    # image hierarchy is built from the same workspace lock, so a base image
    # pinned into a job-kind slot carries a lock digest that matches perfectly.
    # Only the level label distinguishes them.
    level = facts.image_level
    if level != _JOB_KIND_LEVEL:
        found = level or "unlabelled"
        return VariantVerification(
            variant=variant,
            reference=reference,
            expected_lock_digest=expected,
            status="drifted",
            detail=(
                f"expected a {_JOB_KIND_LEVEL} image but found a {found} image; "
                f"a {found} image cannot run a job even though its lock digest "
                f"may match"
            ),
        )

    observed = facts.lock_digest
    if observed is None:
        return VariantVerification(
            variant=variant,
            reference=reference,
            expected_lock_digest=expected,
            status="drifted",
            detail=f"image carries no {LOCK_DIGEST_LABEL} label",
        )
    if observed != expected:
        return VariantVerification(
            variant=variant,
            reference=reference,
            expected_lock_digest=expected,
            status="drifted",
            detail=(
                f"image was built from lock {observed}"
                + (f" at framework revision {facts.revision}" if facts.revision else "")
                + f", but this framework ships lock {expected}; "
                "the image must be republished"
            ),
        )
    revision = facts.revision
    provenance = f", built from framework revision {revision}" if revision else ""
    return VariantVerification(
        variant=variant,
        reference=reference,
        expected_lock_digest=expected,
        status="ok",
        detail=f"lock digest {expected} matches{provenance}",
    )


def verify_registry(
    registry: RegistryBinding,
    *,
    variants: Iterable[str] | None = None,
    manifest: PublishedManifest | None = None,
    inspector: ImageInspector | None = None,
) -> tuple[VariantVerification, ...]:
    """Verify every configured variant, or a named subset."""
    resolved_manifest = manifest or load_manifest()
    resolved_inspector = inspector or RuntimeImageInspector()
    images = expected_images(registry)
    selected = sorted(set(variants)) if variants is not None else sorted(images)
    unknown = [variant for variant in selected if variant not in images]
    if unknown:
        raise ContractError(
            "no runtime image is configured for: "
            + ", ".join(unknown)
            + "; configured variants are "
            + ", ".join(sorted(images))
        )
    return tuple(
        verify_variant(
            variant,
            images[variant],
            manifest=resolved_manifest,
            inspector=resolved_inspector,
        )
        for variant in selected
    )


def ensure_kind_image_ready(
    registry: RegistryBinding,
    variant: str,
    *,
    build_missing: bool = False,
    manifest: PublishedManifest | None = None,
    inspector: ImageInspector | None = None,
) -> VariantVerification:
    """Refuse to pack against a job-kind image that is absent or has drifted.

    Failing closed is the point. A stale job-kind image does not announce
    itself: the run succeeds, produces evidence, and that evidence silently
    describes different software than the framework claims to be running.
    """
    result = verify_variant(
        variant,
        expected_images(registry).get(variant, ""),
        manifest=manifest or load_manifest(),
        inspector=inspector or RuntimeImageInspector(),
    )
    if result.ok:
        return result

    if result.status == "unreachable":
        raise ContractError(
            f"the registry holding job-kind image {result.reference} could not be "
            f"queried, so its identity cannot be confirmed: {result.detail}"
        )

    if not build_missing:
        remedy = (
            "posttrain runtime images mirror --registry <your-registry>"
            if result.status == "missing"
            else "republish the job-kind images for this framework release"
        )
        raise ContractError(
            f"job-kind image for {variant} is {result.status}: {result.detail}. "
            f"Refusing to pack, because evidence produced against it would not "
            f"describe this framework. Remedy: {remedy}, or pass --build-missing "
            f"to rebuild from the shipped definitions."
        )

    from .runtime_image_builds import build_runtime_images

    built = build_runtime_images(registry, variants=[variant])
    rebuilt = built[0] if built else None
    verified = verify_variant(
        variant,
        expected_images(registry).get(variant, ""),
        manifest=manifest or load_manifest(),
        inspector=inspector or RuntimeImageInspector(),
    )
    if not verified.ok:
        rebuilt_reference = rebuilt.image if rebuilt is not None else "the rebuilt image"
        raise ContractError(
            f"rebuilt {variant} as {rebuilt_reference}, but the configured digest-pinned "
            f"reference {verified.reference} is still {verified.status}: {verified.detail}. "
            "Update [registry].kind_images (or publish the rebuilt digest) before packing; "
            "refusing to create evidence against the stale image."
        )
    return verified


__all__ = [
    "VariantVerification",
    "ensure_kind_image_ready",
    "VerificationStatus",
    "expected_images",
    "verify_registry",
    "verify_variant",
]
