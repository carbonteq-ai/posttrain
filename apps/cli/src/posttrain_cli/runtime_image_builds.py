"""Rebuild framework runtime images from the shipped definitions.

This exists for sites that can reach neither the framework's public registry
nor a mirror. It is not the normal path: a rebuilt image is not the published
image, and the difference is reported rather than hidden.

The lock digest passed to the build is computed here, from the lock shipped in
this distribution. It was previously typed by hand into a `--set` argument and
then transcribed a second time into machine-local configuration, which is the
mechanism by which a published image and its framework came to disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import ContractError
from posttrain.runtime_images import (
    BASE_BAKE_FILE,
    BASE_DEFINITION,
    KIND_BAKE_FILE,
    KIND_DEFINITION,
    cached_definition_root,
    constraint_lock,
    lock_digest,
)
from posttrain.runtime_images.manifest import load_manifest
from posttrain_execution_buildkit import (
    BuildKitRuntimeBuilder,
    RuntimeBuildRequest,
    digest_runtime_sources,
)

from .execution_config import RegistryBinding

_KIND_REPOSITORY_PREFIX = "posttrain-kind-"


@dataclass(frozen=True, slots=True)
class RuntimeImageBuild:
    """One locally rebuilt image and how it compares to the published one."""

    variant: str
    image: str
    lock_digest: str
    matches_published_digest: bool


def _registry_prefix(registry: RegistryBinding) -> str:
    if registry.mirror_prefix is not None:
        return registry.mirror_prefix.rstrip("/")
    prefix = registry.repository.rsplit("/", 1)[0]
    if not prefix:
        raise ContractError(
            f"cannot derive a registry prefix from repository {registry.repository!r}"
        )
    return prefix


def _receipt_root(registry: RegistryBinding) -> Path:
    root = registry.receipt_root
    if root is None:
        raise ContractError(
            "rebuilding runtime images needs [registry].receipt_root so a build "
            "receipt can be retained"
        )
    return root.resolve()


def _source_digest(root: Path) -> str:
    return digest_runtime_sources(root, [Path(BASE_DEFINITION), Path(KIND_DEFINITION)])


def _request(
    variant: str,
    *,
    registry: RegistryBinding,
    root: Path,
    source_digest: str,
) -> RuntimeBuildRequest:
    return RuntimeBuildRequest(
        profile=variant,
        bake_file=(root / KIND_BAKE_FILE).resolve(),
        context=root,
        target=f"{_KIND_REPOSITORY_PREFIX}{variant}",
        repository=f"{_registry_prefix(registry)}/{_KIND_REPOSITORY_PREFIX}{variant}",
        source_digest=source_digest,
        # Computed from the shipped lock, never restated by a human.
        lock_digest=lock_digest(constraint_lock(variant)),
        base_image=registry.universal_image,
        builder=registry.buildx_builder,
    )


def _base_request(
    *,
    registry: RegistryBinding,
    root: Path,
    source_digest: str,
) -> RuntimeBuildRequest:
    return RuntimeBuildRequest(
        profile="base",
        bake_file=(root / BASE_BAKE_FILE).resolve(),
        context=root,
        target="posttrain-base",
        repository=f"{_registry_prefix(registry)}/posttrain-base",
        source_digest=source_digest,
        lock_digest=lock_digest(),
        base_image=registry.universal_image,
        builder=registry.buildx_builder,
    )


def check_runtime_images(
    registry: RegistryBinding,
    *,
    variants: Sequence[str],
    builder: BuildKitRuntimeBuilder | None = None,
) -> tuple[str, ...]:
    """Resolve and validate each definition without producing an image."""
    root = cached_definition_root()
    source_digest = _source_digest(root)
    resolved = builder or BuildKitRuntimeBuilder(receipt_root=_receipt_root(registry))
    for variant in variants:
        resolved.check(
            _request(variant, registry=registry, root=root, source_digest=source_digest)
        )
    return tuple(variants)


def build_runtime_images(
    registry: RegistryBinding,
    *,
    variants: Sequence[str],
    builder: BuildKitRuntimeBuilder | None = None,
) -> tuple[RuntimeImageBuild, ...]:
    """Rebuild and publish each variant, reporting divergence from the release."""
    manifest = load_manifest()
    root = cached_definition_root()
    source_digest = _source_digest(root)
    resolved = builder or BuildKitRuntimeBuilder(receipt_root=_receipt_root(registry))

    built: list[RuntimeImageBuild] = []
    for variant in variants:
        if variant not in manifest.kinds:
            raise ContractError(
                f"this release does not publish runtime variant {variant!r}; "
                "published variants are " + ", ".join(sorted(manifest.kinds))
            )
        result = resolved.build(
            _request(variant, registry=registry, root=root, source_digest=source_digest)
        )
        published = manifest.kinds[variant].digest
        observed = result.image.value.rsplit("@", 1)[1]
        built.append(
            RuntimeImageBuild(
                variant=variant,
                image=result.image.value,
                lock_digest=result.lock_digest,
                matches_published_digest=observed == published,
            )
        )
    return tuple(built)


__all__ = [
    "RuntimeImageBuild",
    "build_runtime_images",
    "check_runtime_images",
]
