"""Build, publish, and pin one framework release of the runtime images."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from posttrain.runtime_images.manifest import ManifestError, PublishedImage, load_manifest
from posttrain_execution_buildkit import (
    BuildKitRuntimeBuilder,
    RemoteImageNotFoundError,
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


def _normalize_variants(variants: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(variants) if variants is not None else RUNTIME_VARIANTS
    unknown = sorted(set(selected) - set(RUNTIME_VARIANTS))
    if unknown:
        known = ", ".join(RUNTIME_VARIANTS)
        raise ValueError(f"unknown runtime variant(s) {unknown}; known: {known}")
    if not selected:
        raise ValueError("at least one runtime variant is required")
    # Preserve the canonical release order even when the caller passes a subset.
    return tuple(variant for variant in RUNTIME_VARIANTS if variant in set(selected))


def _prior_ref(prefix: str, repository: str, digest: str) -> str:
    return f"{prefix.rstrip('/')}/{repository}@{digest}"


def _prior_base_ref(prefix: str) -> str | None:
    try:
        previous = load_manifest(verify_locks=False, verify_variants=False).base
    except ManifestError:
        return None
    return _prior_ref(prefix, previous.repository, previous.digest)


def _prior_kind_ref(prefix: str, variant: str) -> str | None:
    try:
        previous = load_manifest(verify_locks=False, verify_variants=False).image(variant)
    except ManifestError:
        return None
    return _prior_ref(prefix, previous.repository, previous.digest)


def _reuse_unchanged_kind(variant: str, root: Path) -> PublishedImage:
    """Reuse a previously published kind when its constraint lock is unchanged.

    Lock-digest drift is what forces a rebuild. When the shipped lock still
    hashes to the digest recorded in the committed manifest, the image bytes
    the label binds to remain valid and rebuilding only burns cache.
    """
    previous = load_manifest(verify_locks=False, verify_variants=False).image(variant)
    expected = lock_digest(constraint_lock(variant))
    if previous.lock_digest != expected:
        raise ValueError(
            f"{variant}: published lock digest {previous.lock_digest} no longer "
            f"matches the shipped lock ({expected}); pass --variant {variant}"
        )
    return PublishedImage(
        name=f"kinds.{variant}",
        repository=previous.repository,
        digest=previous.digest,
        lock_digest=previous.lock_digest,
        constraint_lock=previous.constraint_lock,
        provided_packages=previous.provided_packages or _provided_packages(variant, root),
    )


def _reuse_unchanged_base(prefix: str, builder: BuildKitRuntimeBuilder) -> PublishedImage | None:
    """Pull the committed base when its lock digest still matches.

    Base is the expensive image and should almost never rebuild. When only
    kind-level pins moved the workspace lock, the operator must rebuild kinds
    — not base — but today's lock-digest label is the full workspace lock, so
    a true content-stable base still looks stale. Reuse is therefore gated on
    an exact lock match (or an explicit --base-image).
    """
    try:
        previous = load_manifest(verify_locks=False, verify_variants=False).base
    except ManifestError:
        return None
    expected = lock_digest()
    if previous.lock_digest != expected:
        return None
    image = RuntimeImageRef(_prior_ref(prefix, previous.repository, previous.digest))
    try:
        builder.verify_remote(image)
    except (RemoteImageNotFoundError, RuntimeError):
        return None
    return PublishedImage(
        name="base",
        repository=_BASE_REPOSITORY,
        digest=previous.digest,
        lock_digest=previous.lock_digest,
        constraint_lock=previous.constraint_lock,
    )


def publish_release(
    *,
    prefix: str,
    framework_version: str,
    created: str,
    revision: str,
    default_prefix: str | None = None,
    builder: BuildKitRuntimeBuilder,
    variants: Sequence[str] | None = None,
    base_image: RuntimeImageRef | None = None,
    parallel: bool = True,
    attestations: bool = False,
    compression_level: int = 1,
    force_compression: bool = False,
    provided_packages: dict[str, tuple[str, ...]] | None = None,
    trust_bundle: Path | None = None,
) -> str:
    """Publish images for this release and return the pinned manifest text.

    `prefix` is where the images are pushed. `default_prefix` is what the
    manifest records as the framework's release registry, and defaults to
    `prefix`. They differ when a release is staged through another registry
    first: digests are content-addressed, so the recorded identity stays true
    once the images are mirrored to the canonical location.

    `variants`, when set, rebuilds only those kinds. Other kinds are reused from
    the committed manifest when their lock digest is still current. `base_image`
    skips the base rebuild and parents every kind on the given registry digest.
    When omitted, an unchanged committed base is reused automatically; otherwise
    a rebuild seeds BuildKit from the previous registry digest via cache-from so
    a wiped local cache does not rebuild base layers from scratch.

    Push defaults skip provenance/SBOM and use light zstd without
    force-recompression. Pass `attestations=True` when policy requires them.
    """
    root = cached_definition_root()
    source_digest = _source_digest(root)
    normalized = prefix.rstrip("/")
    supplied = provided_packages or {}
    selected = _normalize_variants(variants)
    build_opts = {
        "attestations": attestations,
        "compression_level": compression_level,
        "force_compression": force_compression,
    }

    if base_image is not None:
        published_base = base_image
        base_lock = lock_digest()
        base = PublishedImage(
            name="base",
            repository=_BASE_REPOSITORY,
            digest=published_base.value.rsplit("@", 1)[1],
            lock_digest=base_lock,
            constraint_lock=constraint_lock("supervised"),
        )
    else:
        # A committed manifest does not record the machine trust-bundle digest,
        # so it cannot prove that a prior base contains the requested trust.
        # Supplying a bundle deliberately forces the content-addressed build
        # path; its byte digest is part of RuntimeBuildRequest.build_key.
        reused = None if trust_bundle is not None else _reuse_unchanged_base(normalized, builder)
        if reused is not None:
            published_base = RuntimeImageRef(f"{normalized}/{reused.repository}@{reused.digest}")
            base = reused
        else:
            prior_base = _prior_base_ref(normalized)
            cache_from = (prior_base,) if prior_base is not None else ()
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
                    cache_from=cache_from,
                    trust_bundle=trust_bundle,
                    **build_opts,
                )
            )
            published_base = base_result.image
            base = PublishedImage(
                name="base",
                repository=_BASE_REPOSITORY,
                digest=published_base.value.rsplit("@", 1)[1],
                lock_digest=lock_digest(),
                constraint_lock=constraint_lock("supervised"),
            )

    def _build_kind(variant: str) -> PublishedImage:
        lock = constraint_lock(variant)
        prior_kind = _prior_kind_ref(normalized, variant)
        cache_from = tuple(ref for ref in (published_base.value, prior_kind) if ref)
        result = builder.build(
            RuntimeBuildRequest(
                profile=variant,
                bake_file=(root / KIND_BAKE_FILE).resolve(),
                context=root,
                target=f"{_KIND_REPOSITORY_PREFIX}{variant}",
                repository=f"{normalized}/{_KIND_REPOSITORY_PREFIX}{variant}",
                source_digest=source_digest,
                lock_digest=lock_digest(lock),
                base_image=published_base,
                variables=_bake_variables(
                    created=created,
                    revision=revision,
                    version=framework_version,
                    base_image=published_base.value,
                ),
                cache_from=cache_from,
                **build_opts,
            )
        )
        return PublishedImage(
            name=f"kinds.{variant}",
            repository=f"{_KIND_REPOSITORY_PREFIX}{variant}",
            digest=result.image.value.rsplit("@", 1)[1],
            lock_digest=lock_digest(lock),
            constraint_lock=lock,
            provided_packages=supplied.get(variant) or _provided_packages(variant, root),
        )

    def _resolve_kind(variant: str) -> PublishedImage:
        if variant in selected:
            return _build_kind(variant)
        # Prefer a receipt for an already-built image from this release identity
        # (parallel helpers plant these). Only fall back to the committed
        # manifest when the lock is unchanged and nothing newer is cached.
        lock = constraint_lock(variant)
        request = RuntimeBuildRequest(
            profile=variant,
            bake_file=(root / KIND_BAKE_FILE).resolve(),
            context=root,
            target=f"{_KIND_REPOSITORY_PREFIX}{variant}",
            repository=f"{normalized}/{_KIND_REPOSITORY_PREFIX}{variant}",
            source_digest=source_digest,
            lock_digest=lock_digest(lock),
            base_image=published_base,
            variables=_bake_variables(
                created=created,
                revision=revision,
                version=framework_version,
                base_image=published_base.value,
            ),
            **build_opts,
        )
        if builder.has_receipt(request):
            return _build_kind(variant)
        return _reuse_unchanged_kind(variant, root)

    kinds: dict[str, PublishedImage] = {}
    if parallel and len(RUNTIME_VARIANTS) > 1:
        with ThreadPoolExecutor(max_workers=len(RUNTIME_VARIANTS)) as pool:
            futures = {pool.submit(_resolve_kind, variant): variant for variant in RUNTIME_VARIANTS}
            for future in as_completed(futures):
                kinds[futures[future]] = future.result()
    else:
        for variant in RUNTIME_VARIANTS:
            kinds[variant] = _resolve_kind(variant)

    return render_manifest(
        framework_version=framework_version,
        default_prefix=(default_prefix or normalized).rstrip("/"),
        base=base,
        kinds=kinds,
    )


__all__ = ["publish_release"]
