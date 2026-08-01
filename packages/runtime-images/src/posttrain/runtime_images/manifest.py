"""The published runtime image manifest.

Parses `published.toml` into frozen values and verifies, against the locks
shipped in the same distribution, that each recorded lock digest still
describes the bytes it claims to describe.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import PurePosixPath

from . import RUNTIME_VARIANTS, lock_digest, read_resource

_MANIFEST = "published.toml"
_SUPPORTED_SCHEMA = 1


class ManifestError(RuntimeError):
    """The shipped manifest is absent, malformed, or disagrees with the locks."""


@dataclass(frozen=True, slots=True)
class PublishedImage:
    """One image published from this framework release."""

    name: str
    repository: str
    digest: str
    lock_digest: str
    constraint_lock: PurePosixPath
    provided_packages: tuple[str, ...] = ()
    # A job-kind image whose runtime has a second, separately locked Python
    # environment publishes that lock too.  The veRL image is the only current
    # case: its control venv is built from `constraint_lock` and its backend
    # venv from the veRL fork's own release lock.  Packaging an environment
    # into such an image needs both, and neither may be transcribed by hand.
    backend_constraint_lock: PurePosixPath | None = None
    backend_lock_digest: str | None = None
    backend_provided_packages: tuple[str, ...] = ()

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

    return PublishedImage(
        name=name,
        repository=_text("repository"),
        digest=digest,
        lock_digest=_text("lock_digest"),
        constraint_lock=PurePosixPath(_text("constraint_lock")),
        provided_packages=tuple(provided),
        backend_constraint_lock=(PurePosixPath(backend_lock) if backend_lock is not None else None),
        backend_lock_digest=backend_digest,
        backend_provided_packages=tuple(backend_provided),
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

    if verify_locks:
        _verify(base)
        for image in kinds.values():
            _verify(image)

    return PublishedManifest(
        schema_version=schema,
        framework_version=str(document.get("framework_version", "")),
        default_prefix=str(document.get("default_prefix", "")),
        base=base,
        kinds=kinds,
    )


__all__ = [
    "ManifestError",
    "PublishedImage",
    "PublishedManifest",
    "load_manifest",
]
