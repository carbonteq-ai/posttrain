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

# Packages a job-kind image installs that a selected environment must not
# resolve again. Recording them by hand in the manifest is another transcription
# of something the profile already states, so they are derived from it.
_PROVIDABLE = ("verifiers",)


def _provided_packages(variant: str, root: Path) -> tuple[str, ...]:
    profile = root / KIND_DEFINITION / "profiles" / f"{variant}.txt"
    if not profile.is_file():
        return ()
    installed = set()
    for line in profile.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith(("#", "-r ")):
            continue
        installed.add(entry.split()[0].split("==")[0].split("@")[0].strip().lower())
    return tuple(name for name in _PROVIDABLE if name in installed)


_KIND_REPOSITORY_PREFIX = "posttrain-kind-"


def _source_digest(root: Path) -> str:
    return digest_runtime_sources(root, [Path(BASE_DEFINITION), Path(KIND_DEFINITION)])


def _bake_variables(
    *,
    created: str,
    revision: str,
    version: str,
    base_image: str | None = None,
) -> dict[str, str]:
    """Variables the shipped base and job-kind Bake files actually declare.

    `RuntimeBuildRequest` also emits BASE_IMAGE and SOURCE_DIGEST, which these
    Bake files do not declare and Bake therefore ignores. Those names belong to
    the superseded job-runtime definition the builder was originally written
    for; the parent image must be passed as POSTTRAIN_BASE_IMAGE or the kind
    build fails with a blank FROM.
    """
    variables = {"CREATED": created, "SOURCE_REVISION": revision, "VERSION": version}
    if base_image is not None:
        variables["POSTTRAIN_BASE_IMAGE"] = base_image
    return variables


def publish_release(
    *,
    prefix: str,
    framework_version: str,
    created: str,
    revision: str,
    default_prefix: str | None = None,
    builder: BuildKitRuntimeBuilder,
    variants: Sequence[str] = RUNTIME_VARIANTS,
    provided_packages: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Publish every image in this release and return the pinned manifest text.

    `prefix` is where the images are pushed. `default_prefix` is what the
    manifest records as the framework's release registry, and defaults to
    `prefix`. They differ when a release is staged through another registry
    first: digests are content-addressed, so the recorded identity stays true
    once the images are mirrored to the canonical location.
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
            variables=_bake_variables(created=created, revision=revision, version=framework_version),
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
                variables=_bake_variables(
                    created=created,
                    revision=revision,
                    version=framework_version,
                    base_image=base_result.image.value,
                ),
            )
        )
        kinds[variant] = PublishedImage(
            name=f"kinds.{variant}",
            repository=f"{_KIND_REPOSITORY_PREFIX}{variant}",
            digest=result.image.value.rsplit("@", 1)[1],
            lock_digest=lock_digest(lock),
            constraint_lock=lock,
            provided_packages=supplied.get(variant) or _provided_packages(variant, root),
        )

    return render_manifest(
        framework_version=framework_version,
        default_prefix=(default_prefix or normalized).rstrip("/"),
        base=base_image,
        kinds=kinds,
    )


__all__ = ["publish_release"]
