"""The published runtime image manifest.

Parses `published.toml` into frozen values and verifies, against the locks
shipped in the same distribution, that each recorded lock digest still
describes the bytes it claims to describe.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

from . import (
    RUNTIME_VARIANTS,
    BackendRuntimeImageIdentity,
    backend_constraint_lock,
    backend_runtime_identity,
    lock_digest,
    read_resource,
)

_MANIFEST = "published.toml"
_SUPPORTED_SCHEMA = 1


class ManifestError(RuntimeError):
    """The shipped manifest is absent, malformed, or disagrees with the locks."""


@dataclass(frozen=True, slots=True)
class PublishedImage:
    """One immutable runtime dependency selected by a framework release."""

    name: str
    repository: str
    digest: str
    lock_digest: str
    constraint_lock: PurePosixPath
    # Framework distributions can reuse a runtime image only when its own
    # source, lock, parent, and installed trust material are unchanged. Older
    # manifests omit these fields and are rebuilt once rather than assumed
    # compatible.
    runtime_source_digest: str | None = None
    trust_bundle_digest: str | None = None
    base_digest: str | None = None
    provided_packages: tuple[str, ...] = ()
    # A job-kind image whose runtime has a second, separately locked Python
    # environment publishes that lock too.  The veRL image is the only current
    # case: its control venv is built from `constraint_lock` and its backend
    # venv from the veRL fork's own release lock.  Packaging an environment
    # into such an image needs both, and neither may be transcribed by hand.
    backend_constraint_lock: PurePosixPath | None = None
    backend_lock_digest: str | None = None
    backend_provided_packages: tuple[str, ...] = ()
    backend_runtime_identity: BackendRuntimeImageIdentity | None = None

    def reference(self, prefix: str) -> str:
        """Return the digest-pinned pull reference under `prefix`."""
        return f"{prefix.rstrip('/')}/{self.repository}@{self.digest}"


@dataclass(frozen=True, slots=True)
class PublishedManifest:
    """Every image published from this framework release.

    `default_prefix` is the framework's own public release registry, identical
    for every consumer. It is deliberately not a project's registry: a project
    pushes its actual-job images to `POSTTRAIN_REGISTRY`, which may separately
    serve as a mirror prefix for these images. Pass that prefix explicitly to
    `reference` rather than expecting it here.
    """

    schema_version: int
    framework_version: str
    default_prefix: str
    base: PublishedImage
    kinds: Mapping[str, PublishedImage]

    def image(self, variant: str) -> PublishedImage:
        try:
            return self.kinds[variant]
        except KeyError:
            known = ", ".join(sorted(self.kinds))
            raise ManifestError(f"no published job-kind image for variant {variant!r}; published: {known}") from None

    def reference(self, variant: str, *, prefix: str | None = None) -> str:
        return self.image(variant).reference(prefix or self.default_prefix)

    def expected_lock_digest(self, variant: str) -> str:
        """Return the lock digest a published `variant` image must carry.

        The value is recomputed from the shipped lock rather than read back
        from the manifest, so a lock edited without regenerating the manifest
        cannot masquerade as current.
        """
        image = self.image(variant)
        return lock_digest(image.constraint_lock)


def _image(name: str, payload: Mapping[str, object]) -> PublishedImage:
    def _text(key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ManifestError(f"{name}: {key!r} must be a non-empty string")
        return value

    digest = _text("digest")
    if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
        raise ManifestError(f"{name}: 'digest' must be a sha256 OCI digest, got {digest!r}")

    def _optional_digest(key: str, *, prefixed: bool = False) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ManifestError(f"{name}: {key!r} must be a string when present")
        if prefixed and not value.startswith("sha256:"):
            raise ManifestError(f"{name}: {key!r} must use the sha256: prefix")
        candidate = value.removeprefix("sha256:") if prefixed else value
        if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
            raise ManifestError(f"{name}: {key!r} must be a SHA-256 digest when present")
        return value

    provided = payload.get("provided_packages", ())
    if not isinstance(provided, list | tuple) or not all(isinstance(p, str) for p in provided):
        raise ManifestError(f"{name}: 'provided_packages' must be a list of strings")

    backend_lock = payload.get("backend_constraint_lock")
    backend_digest = payload.get("backend_lock_digest")
    if (backend_lock is None) != (backend_digest is None):
        raise ManifestError(f"{name}: 'backend_constraint_lock' and 'backend_lock_digest' must be declared together")
    if backend_lock is not None and (not isinstance(backend_lock, str) or not backend_lock):
        raise ManifestError(f"{name}: 'backend_constraint_lock' must be a non-empty string")
    if backend_digest is not None and (not isinstance(backend_digest, str) or not backend_digest):
        raise ManifestError(f"{name}: 'backend_lock_digest' must be a non-empty string")

    backend_provided = payload.get("backend_provided_packages", ())
    if not isinstance(backend_provided, list | tuple) or not all(isinstance(p, str) for p in backend_provided):
        raise ManifestError(f"{name}: 'backend_provided_packages' must be a list of strings")
    if backend_lock is None and backend_provided:
        raise ManifestError(f"{name}: 'backend_provided_packages' needs a 'backend_constraint_lock'")

    runtime_identity_fields = (
        "backend_source_repository",
        "backend_source_revision",
        "backend_dependency_lock_digest",
    )
    identity_values = tuple(payload.get(field) for field in runtime_identity_fields)
    if any(value is not None for value in identity_values) and not all(
        isinstance(value, str) and value for value in identity_values
    ):
        raise ManifestError(f"{name}: backend runtime identity fields must be declared together")
    identity: BackendRuntimeImageIdentity | None = None
    if all(isinstance(value, str) and value for value in identity_values):
        repository, revision, dependency_digest = identity_values
        assert isinstance(repository, str) and isinstance(revision, str) and isinstance(dependency_digest, str)
        if not repository.startswith("https://"):
            raise ManifestError(f"{name}: 'backend_source_repository' must use canonical HTTPS")
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ManifestError(f"{name}: 'backend_source_revision' must be a full commit")
        if len(dependency_digest) != 64 or any(character not in "0123456789abcdef" for character in dependency_digest):
            raise ManifestError(f"{name}: 'backend_dependency_lock_digest' must be a SHA-256 digest")
        identity = BackendRuntimeImageIdentity(repository, revision, dependency_digest)
    if (backend_lock is None) != (identity is None):
        raise ManifestError(f"{name}: backend runtime identity requires a backend constraint lock")

    return PublishedImage(
        name=name,
        repository=_text("repository"),
        digest=digest,
        lock_digest=_text("lock_digest"),
        constraint_lock=PurePosixPath(_text("constraint_lock")),
        runtime_source_digest=_optional_digest("runtime_source_digest"),
        trust_bundle_digest=_optional_digest("trust_bundle_digest"),
        base_digest=_optional_digest("base_digest", prefixed=True),
        provided_packages=tuple(provided),
        backend_constraint_lock=(PurePosixPath(backend_lock) if backend_lock is not None else None),
        backend_lock_digest=backend_digest,
        backend_provided_packages=tuple(backend_provided),
        backend_runtime_identity=identity,
    )


def _verify_lock(image: PublishedImage, lock: PurePosixPath, recorded: str, *, role: str) -> None:
    try:
        actual = lock_digest(lock)
    except (FileNotFoundError, OSError) as error:
        raise ManifestError(f"{image.name}: {role} lock {lock} is not shipped in this distribution") from error
    if actual != recorded:
        raise ManifestError(
            f"{image.name}: published image records {role} lock digest {recorded}, "
            f"but the shipped {lock} hashes to {actual}. The image "
            f"must be republished, or the manifest regenerated; a stale job-kind "
            f"image silently invalidates every qualification run against it."
        )


def _verify(image: PublishedImage) -> None:
    _verify_lock(image, image.constraint_lock, image.lock_digest, role="constraint")
    if image.backend_constraint_lock is not None:
        assert image.backend_lock_digest is not None
        _verify_lock(image, image.backend_constraint_lock, image.backend_lock_digest, role="backend constraint")


def _verify_directory_lock(
    image: PublishedImage,
    lock: PurePosixPath,
    recorded: str,
    *,
    role: str,
    directory: Path,
) -> None:
    if lock.is_absolute() or ".." in lock.parts:
        raise ManifestError(f"{image.name}: {role} lock must stay within the runtime-images package")
    path = directory / lock
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except (FileNotFoundError, OSError) as error:
        raise ManifestError(f"{image.name}: {role} lock {lock} is not shipped in this distribution") from error
    if actual != recorded:
        raise ManifestError(
            f"{image.name}: published image records {role} lock digest {recorded}, "
            f"but the shipped {lock} hashes to {actual}. The image "
            "must be republished, or the manifest regenerated; a stale job-kind "
            "image silently invalidates every qualification run against it."
        )


def _parse_manifest(
    raw: bytes,
    *,
    verify_locks: bool,
    verify_variants: bool,
    directory: Path | None = None,
) -> PublishedManifest:
    document = tomllib.loads(raw.decode("utf-8"))

    schema = document.get("schema_version")
    if schema != _SUPPORTED_SCHEMA:
        raise ManifestError(f"unsupported manifest schema_version {schema!r}; expected {_SUPPORTED_SCHEMA}")

    base = _image("base", document.get("base", {}))
    kinds = {variant: _image(f"kinds.{variant}", payload) for variant, payload in document.get("kinds", {}).items()}

    if verify_variants:
        missing = set(RUNTIME_VARIANTS) - set(kinds)
        if missing:
            raise ManifestError("published.toml is missing job-kind images for: " + ", ".join(sorted(missing)))
        unexpected = set(kinds) - set(RUNTIME_VARIANTS)
        if unexpected:
            raise ManifestError("published.toml publishes unreleased variants: " + ", ".join(sorted(unexpected)))
        for variant, image in kinds.items():
            expected_backend = backend_constraint_lock(variant)
            if image.backend_constraint_lock != expected_backend:
                raise ManifestError(
                    f"kinds.{variant}: backend constraint lock is "
                    f"{image.backend_constraint_lock}, expected {expected_backend}"
                )
            if image.backend_runtime_identity != backend_runtime_identity(variant):
                raise ManifestError(f"kinds.{variant}: backend runtime identity differs from its shipped profile")

    if verify_locks:
        if directory is None:
            _verify(base)
            for image in kinds.values():
                _verify(image)
        else:
            _verify_directory_lock(base, base.constraint_lock, base.lock_digest, role="constraint", directory=directory)
            for image in kinds.values():
                _verify_directory_lock(
                    image, image.constraint_lock, image.lock_digest, role="constraint", directory=directory
                )
                if image.backend_constraint_lock is not None:
                    assert image.backend_lock_digest is not None
                    _verify_directory_lock(
                        image,
                        image.backend_constraint_lock,
                        image.backend_lock_digest,
                        role="backend constraint",
                        directory=directory,
                    )

    return PublishedManifest(
        schema_version=schema,
        framework_version=str(document.get("framework_version", "")),
        default_prefix=str(document.get("default_prefix", "")),
        base=base,
        kinds=kinds,
    )


@cache
def load_manifest(
    *,
    verify_locks: bool = True,
    verify_variants: bool = True,
) -> PublishedManifest:
    """Load the shipped manifest.

    When `verify_locks` is true (the consumer default), every recorded lock
    digest must still match the shipped lock bytes. When `verify_variants` is
    true, the manifest must cover exactly the runtime variants shipped by this
    distribution. Release tooling disables both checks so it can read a prior
    manifest while adding or retiring a variant and selectively reuse the
    entries that still apply.
    """
    try:
        raw = read_resource(PurePosixPath(_MANIFEST))
    except (FileNotFoundError, OSError) as error:
        raise ManifestError("this distribution ships no published.toml") from error

    return _parse_manifest(raw, verify_locks=verify_locks, verify_variants=verify_variants)


def load_manifest_from_directory(
    directory: Path,
    *,
    verify_locks: bool = True,
    verify_variants: bool = True,
) -> PublishedManifest:
    """Load a staged runtime-images manifest, never the caller's environment.

    Release checks run against a candidate/staged checkout.  They must validate
    the locks beside that checkout's ``published.toml`` rather than whichever
    editable runtime-images package happens to be imported by the tool.
    """

    root = directory.resolve()
    try:
        raw = (root / _MANIFEST).read_bytes()
    except (FileNotFoundError, OSError) as error:
        raise ManifestError(f"runtime-images manifest is absent: {root / _MANIFEST}") from error
    return _parse_manifest(raw, verify_locks=verify_locks, verify_variants=verify_variants, directory=root)


__all__ = [
    "ManifestError",
    "PublishedImage",
    "PublishedManifest",
    "load_manifest_from_directory",
    "load_manifest",
]
