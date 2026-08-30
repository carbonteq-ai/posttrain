"""Build, publish, and pin one framework release of the runtime images."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from posttrain.common import ContractError
from posttrain.execution import RegistryManifestRef, RuntimeImageRef
from posttrain.runtime_images import (
    BASE_BAKE_FILE,
    BASE_DEFINITION,
    BASE_LOCK,
    KIND_BAKE_FILE,
    KIND_DEFINITION,
    RUNTIME_VARIANTS,
    backend_constraint_lock,
    backend_runtime_identity,
    cached_definition_root,
    constraint_lock,
    lock_digest,
)
from posttrain.runtime_images.manifest import ManifestError, PublishedImage, PublishedManifest, load_manifest
from posttrain_execution_buildkit import (
    BuildKitRuntimeBuilder,
    DistributionRegistryContentReader,
    RemoteImageNotFoundError,
    RuntimeBuildRequest,
    UrllibDistributionTransport,
    digest_runtime_sources,
)

from .image_plan import DesiredImage, ObservedImage, ReleaseImagePlan, plan_release_images
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
# Each kind build can concurrently resolve packages, pull base layers, and push
# multi-gigabyte output. Two workers preserve useful overlap without saturating
# the release runner, the internal package index, or the registry.
_MAX_PARALLEL_KIND_BUILDS = 2


def _base_source_digest(root: Path) -> str:
    return digest_runtime_sources(
        root,
        [
            Path(BASE_DEFINITION),
            Path(BASE_DEFINITION) / "requirements.txt",
            Path(BASE_LOCK),
        ],
    )


def _kind_source_paths(variant: str) -> tuple[Path, ...]:
    """Return only the definition files that can affect one kind target.

    Shared Docker/Bake inputs remain shared invalidation boundaries. Profile
    files and the dedicated veRL definition are variant-local, so changing one
    does not make every unrelated kind look stale.
    """

    common = (
        Path(KIND_BAKE_FILE),
        Path(KIND_DEFINITION) / "locks" / "build-tools.lock.txt",
        Path(KIND_DEFINITION) / "profiles" / "common.txt",
        Path(constraint_lock(variant)),
    )
    if variant == "online-rl-verl-py313":
        return (
            *common,
            Path(KIND_DEFINITION) / "profiles" / "online-rl-verl-py313-control.txt",
            Path(KIND_DEFINITION) / "verl-py313",
        )
    return (
        *common,
        Path(KIND_DEFINITION) / "Dockerfile",
        Path(KIND_DEFINITION) / "profiles" / f"{variant}.txt",
    )


def _kind_source_digest(root: Path, variant: str) -> str:
    return digest_runtime_sources(root, _kind_source_paths(variant))


def _trust_bundle_digest(bundle: Path | None) -> str | None:
    if bundle is None:
        return None
    digest = hashlib.sha256()
    with bundle.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bake_variables(
    *,
    created: str,
    revision: str,
    version: str,
    base_image: str | None = None,
    variant: str | None = None,
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
    if variant is not None:
        identity = backend_runtime_identity(variant)
        if identity is not None:
            variables.update(
                {
                    "DEPENDENCY_LOCK_SHA256": identity.dependency_lock_digest,
                    "FORK_REVISION": identity.source_revision,
                    "SOURCE_REPOSITORY": identity.source_repository,
                }
            )
    return variables


def _source_date_epoch(created: str) -> int:
    """Convert the release commit timestamp into BuildKit's stable epoch."""

    normalized = created.removesuffix("Z") + "+00:00" if created.endswith("Z") else created
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"release created timestamp is not ISO 8601: {created!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("release created timestamp must include a timezone")
    epoch = int(parsed.timestamp())
    if epoch <= 0:
        raise ValueError("release created timestamp must be after the Unix epoch")
    return epoch


def _normalize_variants(variants: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(variants) if variants is not None else ()
    unknown = sorted(set(selected) - set(RUNTIME_VARIANTS))
    if unknown:
        known = ", ".join(RUNTIME_VARIANTS)
        raise ValueError(f"unknown runtime variant(s) {unknown}; known: {known}")
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


def _desired_base(root: Path, trust_bundle: Path | None) -> DesiredImage:
    return DesiredImage(
        "base",
        _BASE_REPOSITORY,
        _base_source_digest(root),
        lock_digest(BASE_LOCK),
        trust_bundle_digest=_trust_bundle_digest(trust_bundle),
    )


def _desired_kind(root: Path, variant: str) -> DesiredImage:
    identity = backend_runtime_identity(variant)
    backend_lock = backend_constraint_lock(variant)
    return DesiredImage(
        f"kinds.{variant}",
        f"{_KIND_REPOSITORY_PREFIX}{variant}",
        _kind_source_digest(root, variant),
        lock_digest(constraint_lock(variant)),
        parent="base",
        backend_lock_digest=(lock_digest(backend_lock) if backend_lock is not None else None),
        backend_identity=(
            (identity.source_repository, identity.source_revision, identity.dependency_lock_digest)
            if identity is not None
            else None
        ),
    )


def _published_identity(name: str, image: PublishedImage, *, parent: str | None) -> str:
    """Reconstruct the semantic identity recorded by a prior manifest entry.

    Missing fields deliberately produce a different identity.  The first
    release after an identity strengthening must rebuild rather than pretend
    an older manifest proved inputs it did not record.
    """

    backend = image.backend_runtime_identity
    return DesiredImage(
        name,
        image.repository,
        image.runtime_source_digest or "",
        image.lock_digest,
        parent=parent,
        trust_bundle_digest=image.trust_bundle_digest,
        backend_lock_digest=image.backend_lock_digest,
        backend_identity=(
            (backend.source_repository, backend.source_revision, backend.dependency_lock_digest)
            if backend is not None
            else None
        ),
    ).identity


def _remote_observation(
    builder: BuildKitRuntimeBuilder,
    *,
    reference: str,
    digest: str,
    identity: str,
) -> ObservedImage:
    try:
        builder.verify_remote(RuntimeImageRef(reference))
    except RemoteImageNotFoundError:
        return ObservedImage("missing", detail="registry does not retain the digest")
    except RuntimeError as error:
        return ObservedImage("unreachable", detail=str(error))
    return ObservedImage("present", digest=digest, identity=identity)


def _release_image_plan(
    *,
    root: Path,
    prefix: str,
    builder: BuildKitRuntimeBuilder,
    trust_bundle: Path | None,
    variants: Sequence[str] | None,
) -> tuple[ReleaseImagePlan, PublishedManifest | None]:
    """Inspect current/published image evidence and produce one pure plan.

    The source is the prior manifest's canonical registry; the destination is
    the registry selected for this candidate.  They may be identical.  Every
    observation is digest-pinned, so an existing tag cannot make a different
    image look reusable.
    """

    try:
        prior = load_manifest(verify_locks=False, verify_variants=False)
    except ManifestError:
        prior = None
    desired = [_desired_base(root, trust_bundle)]
    selected = set(_normalize_variants(variants)) if variants is not None else set()
    desired.extend(replace(_desired_kind(root, variant), forced=variant in selected) for variant in RUNTIME_VARIANTS)
    source: dict[str, ObservedImage] = {}
    destination: dict[str, ObservedImage] = {}
    content_reader = (
        DistributionRegistryContentReader(UrllibDistributionTransport(trust_bundle=trust_bundle))
        if prior is not None and prior.default_prefix.rstrip("/") != prefix.rstrip("/")
        else None
    )
    if prior is not None:
        published = ((desired[0], prior.base),) + tuple(
            (node, prior.image(variant)) for node, variant in zip(desired[1:], RUNTIME_VARIANTS, strict=True)
        )
        for node, image in published:
            previous_identity = _published_identity(node.name, image, parent=node.parent)
            source_ref = _prior_ref(prior.default_prefix, image.repository, image.digest)
            destination_ref = _prior_ref(prefix, image.repository, image.digest)
            source[node.name] = _remote_observation(
                builder,
                reference=source_ref,
                digest=image.digest,
                identity=previous_identity,
            )
            if destination_ref == source_ref:
                destination[node.name] = source[node.name]
            else:
                destination[node.name] = _remote_observation(
                    builder,
                    reference=destination_ref,
                    digest=image.digest,
                    identity=previous_identity,
                )
            if content_reader is not None and source[node.name].status == "present":
                try:
                    missing_blob_bytes = content_reader.missing_blob_bytes(
                        RegistryManifestRef.parse(source_ref),
                        f"{prefix.rstrip('/')}/{image.repository}",
                    )
                except (ContractError, OSError, RuntimeError):
                    # The action remains correct without byte inventory.  The
                    # plan explicitly reports an unknown copy payload rather
                    # than turning an otherwise inspectable release into a
                    # blind build because one registry refuses blob HEADs.
                    pass
                else:
                    source[node.name] = replace(source[node.name], missing_blob_bytes=missing_blob_bytes)
    return plan_release_images(tuple(desired), source=source, destination=destination), prior


def _reuse_unchanged_kind(
    variant: str,
    root: Path,
    *,
    prefix: str,
    builder: BuildKitRuntimeBuilder,
    runtime_source_digest: str,
    base_digest: str,
) -> PublishedImage:
    """Reuse a verified kind only when every runtime input is unchanged.

    A framework distribution version is intentionally absent from this test:
    the packed job injects the versioned wheelhouse. Runtime source, lock,
    parent image, and registry presence are the image's real dependencies.
    """
    previous = load_manifest(verify_locks=False, verify_variants=False).image(variant)
    expected = lock_digest(constraint_lock(variant))
    if previous.lock_digest != expected:
        raise ValueError(
            f"{variant}: published lock digest {previous.lock_digest} no longer "
            f"matches the shipped lock ({expected}); pass --variant {variant}"
        )
    if previous.runtime_source_digest != runtime_source_digest:
        raise ValueError(f"{variant}: runtime source digest changed or is absent")
    if previous.base_digest != base_digest:
        raise ValueError(f"{variant}: base image digest changed or is absent")
    expected_backend = backend_constraint_lock(variant)
    if previous.backend_constraint_lock != expected_backend or (
        expected_backend is not None and previous.backend_lock_digest != lock_digest(expected_backend)
    ):
        raise ValueError(f"{variant}: backend constraint lock changed or is absent")
    if previous.backend_runtime_identity != backend_runtime_identity(variant):
        raise ValueError(f"{variant}: backend runtime identity changed or is absent")
    builder.verify_remote(RuntimeImageRef(_prior_ref(prefix, previous.repository, previous.digest)))
    return PublishedImage(
        name=f"kinds.{variant}",
        repository=previous.repository,
        digest=previous.digest,
        lock_digest=previous.lock_digest,
        constraint_lock=previous.constraint_lock,
        runtime_source_digest=previous.runtime_source_digest,
        base_digest=previous.base_digest,
        provided_packages=previous.provided_packages or _provided_packages(variant, root),
        backend_constraint_lock=previous.backend_constraint_lock,
        backend_lock_digest=previous.backend_lock_digest,
        backend_provided_packages=previous.backend_provided_packages,
        backend_runtime_identity=previous.backend_runtime_identity,
    )


def _reuse_unchanged_base(
    prefix: str,
    builder: BuildKitRuntimeBuilder,
    *,
    runtime_source_digest: str,
    trust_bundle_digest: str | None,
) -> PublishedImage | None:
    """Reuse the committed base when every runtime input still matches.

    The trust bundle is image content, so a different bundle must rebuild. A
    missing digest is intentionally incompatible with older manifests and
    establishes the stronger receipt on the first migrated release.
    """
    try:
        previous = load_manifest(verify_locks=False, verify_variants=False).base
    except ManifestError:
        return None
    expected = lock_digest(BASE_LOCK)
    if previous.lock_digest != expected:
        return None
    if previous.runtime_source_digest != runtime_source_digest:
        return None
    if previous.trust_bundle_digest != trust_bundle_digest:
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
        runtime_source_digest=previous.runtime_source_digest,
        trust_bundle_digest=previous.trust_bundle_digest,
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
    the committed manifest only when their runtime source, dependency locks,
    and parent base digest are still current. `base_image`
    skips the base rebuild and parents every kind on the given registry digest.
    When omitted, an unchanged committed base is reused automatically; otherwise
    a rebuild seeds BuildKit from the previous registry digest via cache-from so
    a wiped local cache does not rebuild base layers from scratch.

    Push defaults skip provenance/SBOM and use light zstd without
    force-recompression. Pass `attestations=True` when policy requires them.
    """
    root = cached_definition_root()
    base_source_digest = _base_source_digest(root)
    trust_digest = _trust_bundle_digest(trust_bundle)
    normalized = prefix.rstrip("/")
    supplied = provided_packages or {}
    selected = _normalize_variants(variants)
    build_opts = {
        "attestations": attestations,
        "compression_level": compression_level,
        "force_compression": force_compression,
    }

    release_plan: ReleaseImagePlan | None = None
    prior_manifest: PublishedManifest | None = None
    if base_image is None:
        release_plan, prior_manifest = _release_image_plan(
            root=root,
            prefix=normalized,
            builder=builder,
            trust_bundle=trust_bundle,
            variants=variants,
        )
        base_node = release_plan.node("base")
        if base_node.action == "blocked":
            raise RuntimeError(f"base image publication is blocked: {base_node.reason}")
        if base_node.action == "reuse-remote":
            assert prior_manifest is not None
            base = prior_manifest.base
            published_base = RuntimeImageRef(f"{normalized}/{base.repository}@{base.digest}")
        elif base_node.action == "copy":
            assert prior_manifest is not None and base_node.expected_digest is not None
            source_image = prior_manifest.base
            published_base = builder.copy_remote(
                RuntimeImageRef(
                    _prior_ref(prior_manifest.default_prefix, source_image.repository, source_image.digest)
                ),
                destination_repository=f"{normalized}/{source_image.repository}",
                expected_digest=base_node.expected_digest,
                tag=f"reuse-{base_node.desired.identity[:16]}",
            )
            base = source_image
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
                    source_digest=base_source_digest,
                    lock_digest=lock_digest(BASE_LOCK),
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
                lock_digest=lock_digest(BASE_LOCK),
                constraint_lock=BASE_LOCK,
                runtime_source_digest=base_source_digest,
                trust_bundle_digest=trust_digest,
            )
    elif base_image is not None:
        published_base = base_image
        base_lock = lock_digest(BASE_LOCK)
        base = PublishedImage(
            name="base",
            repository=_BASE_REPOSITORY,
            digest=published_base.value.rsplit("@", 1)[1],
            lock_digest=base_lock,
            constraint_lock=BASE_LOCK,
            runtime_source_digest=base_source_digest,
            trust_bundle_digest=trust_digest,
        )

    def _build_kind(variant: str) -> PublishedImage:
        lock = constraint_lock(variant)
        backend_lock = backend_constraint_lock(variant)
        kind_source_digest = _kind_source_digest(root, variant)
        prior_kind = _prior_kind_ref(normalized, variant)
        cache_from = tuple(ref for ref in (published_base.value, prior_kind) if ref)
        result = builder.build(
            RuntimeBuildRequest(
                profile=variant,
                bake_file=(root / KIND_BAKE_FILE).resolve(),
                context=root,
                target=f"{_KIND_REPOSITORY_PREFIX}{variant}",
                repository=f"{normalized}/{_KIND_REPOSITORY_PREFIX}{variant}",
                source_digest=kind_source_digest,
                lock_digest=lock_digest(lock),
                base_image=published_base,
                variables=_bake_variables(
                    created=created,
                    revision=revision,
                    version=framework_version,
                    base_image=published_base.value,
                    variant=variant,
                ),
                cache_from=cache_from,
                source_date_epoch=(_source_date_epoch(created) if variant == "online-rl-verl-py313" else None),
                **build_opts,
            )
        )
        return PublishedImage(
            name=f"kinds.{variant}",
            repository=f"{_KIND_REPOSITORY_PREFIX}{variant}",
            digest=result.image.value.rsplit("@", 1)[1],
            lock_digest=lock_digest(lock),
            constraint_lock=lock,
            runtime_source_digest=kind_source_digest,
            base_digest=published_base.value.rsplit("@", 1)[1],
            provided_packages=supplied.get(variant) or _provided_packages(variant, root),
            backend_constraint_lock=backend_lock,
            backend_lock_digest=lock_digest(backend_lock) if backend_lock is not None else None,
            backend_runtime_identity=backend_runtime_identity(variant),
        )

    def _resolve_kind(variant: str) -> PublishedImage:
        if release_plan is not None:
            node = release_plan.node(f"kinds.{variant}")
            if node.action == "blocked":
                raise RuntimeError(f"{variant} image publication is blocked: {node.reason}")
            if node.action == "reuse-remote":
                assert prior_manifest is not None
                image = prior_manifest.image(variant)
                return replace(
                    image,
                    name=f"kinds.{variant}",
                    provided_packages=supplied.get(variant)
                    or image.provided_packages
                    or _provided_packages(variant, root),
                )
            if node.action == "copy":
                assert prior_manifest is not None and node.expected_digest is not None
                image = prior_manifest.image(variant)
                copied = builder.copy_remote(
                    RuntimeImageRef(_prior_ref(prior_manifest.default_prefix, image.repository, image.digest)),
                    destination_repository=f"{normalized}/{image.repository}",
                    expected_digest=node.expected_digest,
                    tag=f"reuse-{node.desired.identity[:16]}",
                )
                if copied.value.rsplit("@", 1)[1] != image.digest:
                    raise RuntimeError(f"{variant}: registry copy returned a different digest")
                return replace(
                    image,
                    name=f"kinds.{variant}",
                    provided_packages=supplied.get(variant)
                    or image.provided_packages
                    or _provided_packages(variant, root),
                )
            return _build_kind(variant)

        kind_source_digest = _kind_source_digest(root, variant)
        if variant in selected:
            return _build_kind(variant)
        # Prefer a receipt for an already-built image from this release identity
        # (parallel helpers plant these). Only fall back to the committed
        # manifest only when every immutable runtime input still matches and
        # nothing newer is cached.
        lock = constraint_lock(variant)
        request = RuntimeBuildRequest(
            profile=variant,
            bake_file=(root / KIND_BAKE_FILE).resolve(),
            context=root,
            target=f"{_KIND_REPOSITORY_PREFIX}{variant}",
            repository=f"{normalized}/{_KIND_REPOSITORY_PREFIX}{variant}",
            source_digest=kind_source_digest,
            lock_digest=lock_digest(lock),
            base_image=published_base,
            variables=_bake_variables(
                created=created,
                revision=revision,
                version=framework_version,
                base_image=published_base.value,
                variant=variant,
            ),
            source_date_epoch=(_source_date_epoch(created) if variant == "online-rl-verl-py313" else None),
            **build_opts,
        )
        if builder.has_receipt(request):
            return _build_kind(variant)
        try:
            return _reuse_unchanged_kind(
                variant,
                root,
                prefix=normalized,
                builder=builder,
                runtime_source_digest=kind_source_digest,
                base_digest=published_base.value.rsplit("@", 1)[1],
            )
        except (RemoteImageNotFoundError, RuntimeError, ValueError):
            return _build_kind(variant)

    kinds: dict[str, PublishedImage] = {}
    if parallel and len(RUNTIME_VARIANTS) > 1:
        with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL_KIND_BUILDS, len(RUNTIME_VARIANTS))) as pool:
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
