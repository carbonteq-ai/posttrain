"""Build, publish, and pin one framework release of the runtime images."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from posttrain.execution import RuntimeImageRef
from posttrain.runtime_images import (
    BASE_BAKE_FILE,
    BASE_DEFINITION,
    KIND_BAKE_FILE,
    KIND_DEFINITION,
    RUNTIME_VARIANTS,
    cached_definition_root,
    constraint_lock,
    lock_digest,
)
from posttrain.runtime_images.manifest import PublishedImage
from posttrain_execution_buildkit import (
    BuildKitRuntimeBuilder,
    RuntimeBuildRequest,
    digest_runtime_sources,
)

from .manifest_render import render_manifest

_BASE_REPOSITORY = "posttrain-base"
_KIND_REPOSITORY_PREFIX = "posttrain-kind-"


def _source_digest(root: Path) -> str:
    return digest_runtime_sources(root, [Path(BASE_DEFINITION), Path(KIND_DEFINITION)])


def publish_release(
    *,
    prefix: str,
    framework_version: str,
    builder: BuildKitRuntimeBuilder,
    variants: Sequence[str] = RUNTIME_VARIANTS,
    provided_packages: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Publish every image in this release and return the pinned manifest text.

    Each digest is the one the registry reports after the push, never one
    predicted locally, so the manifest can only describe images that exist.
    """
    root = cached_definition_root()
    source_digest = _source_digest(root)
    normalized = prefix.rstrip("/")
    supplied = provided_packages or {}

    base_result = builder.build(
        RuntimeBuildRequest(
            profile="base",
            bake_file=(root / BASE_BAKE_FILE).resolve(),
            context=root,
            target=_BASE_REPOSITORY,
            repository=f"{normalized}/{_BASE_REPOSITORY}",
            source_digest=source_digest,
            lock_digest=lock_digest(),
            base_image=RuntimeImageRef(f"scratch@sha256:{'0' * 64}"),
        )
    )
    base_image = PublishedImage(
        name="base",
        repository=_BASE_REPOSITORY,
        digest=base_result.image.value.rsplit("@", 1)[1],
        lock_digest=lock_digest(),
        constraint_lock=constraint_lock("supervised"),
    )

    kinds: dict[str, PublishedImage] = {}
    for variant in variants:
        lock = constraint_lock(variant)
        result = builder.build(
            RuntimeBuildRequest(
                profile=variant,
                bake_file=(root / KIND_BAKE_FILE).resolve(),
                context=root,
                target=f"{_KIND_REPOSITORY_PREFIX}{variant}",
                repository=f"{normalized}/{_KIND_REPOSITORY_PREFIX}{variant}",
                source_digest=source_digest,
                lock_digest=lock_digest(lock),
                base_image=base_result.image,
            )
        )
        kinds[variant] = PublishedImage(
            name=f"kinds.{variant}",
            repository=f"{_KIND_REPOSITORY_PREFIX}{variant}",
            digest=result.image.value.rsplit("@", 1)[1],
            lock_digest=lock_digest(lock),
            constraint_lock=lock,
            provided_packages=supplied.get(variant, ()),
        )

    return render_manifest(
        framework_version=framework_version,
        default_prefix=normalized,
        base=base_image,
        kinds=kinds,
    )


__all__ = ["publish_release"]
