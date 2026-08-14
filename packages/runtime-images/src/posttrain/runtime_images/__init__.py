"""Access to the framework's shipped container definitions.

The definitions live as package data so an installed distribution carries the
exact inputs that produced its runtime images, rather than a reference to a
source checkout it may not have.
"""

from __future__ import annotations

import atexit
import hashlib
import re
import tomllib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import cache
from importlib.resources import as_file
from importlib.resources import files as resource_files
from pathlib import Path, PurePosixPath

RUNTIME_VARIANTS: tuple[str, ...] = (
    "supervised",
    "online-rl-trl-py312",
    "online-rl-verl-py313",
    "eval",
    "serve",
    "transform",
)
"""Job-kind variants published once per framework release.
"""

_CONTAINERS = PurePosixPath("containers")
BASE_DEFINITION = _CONTAINERS / "posttrain-base"
KIND_DEFINITION = _CONTAINERS / "posttrain-job-kinds"
JOB_DEFINITION = _CONTAINERS / "posttrain-job"

BASE_BAKE_FILE = BASE_DEFINITION / "docker-bake.hcl"
KIND_BAKE_FILE = KIND_DEFINITION / "docker-bake.hcl"
JOB_BAKE_FILE = JOB_DEFINITION / "docker-bake.hcl"
PUBLISHED_MANIFEST = PurePosixPath("published.toml")

WORKSPACE_LOCK = KIND_DEFINITION / "locks" / "workspace.lock.txt"
"""Authoritative workspace resolution used to generate narrower runtime locks."""

BASE_LOCK = KIND_DEFINITION / "locks" / "base.lock.txt"
TRANSFORM_LOCK = KIND_DEFINITION / "locks" / "transform.lock.txt"
VERL_BACKEND_LOCK = KIND_DEFINITION / "verl-py313" / "release" / "backend-constraints.txt"
VERL_PROFILE = KIND_DEFINITION / "verl-py313" / "profile.toml"

VERL_DEPENDENCY_LOCK_LABEL = "org.carbonteq.posttrain.verl-dependency-lock-sha256"
VERL_SOURCE_REPOSITORY_LABEL = "org.carbonteq.posttrain.verl-source-repository"
VERL_SOURCE_REVISION_LABEL = "org.carbonteq.posttrain.verl-source-revision"

_BACKEND_CONSTRAINT_LOCKS = {
    "online-rl-verl-py313": VERL_BACKEND_LOCK,
}

_KIND_CONSTRAINT_LOCKS = {
    "supervised": KIND_DEFINITION / "locks" / "supervised.lock.txt",
    "online-rl-trl-py312": KIND_DEFINITION / "locks" / "online-rl-trl-py312.lock.txt",
    "online-rl-verl-py313": KIND_DEFINITION / "locks" / "online-rl-verl-py313.lock.txt",
    "eval": KIND_DEFINITION / "locks" / "eval.lock.txt",
    "serve": KIND_DEFINITION / "locks" / "serve.lock.txt",
    "transform": TRANSFORM_LOCK,
}


@dataclass(frozen=True, slots=True)
class BackendRuntimeImageIdentity:
    """Source and lock bytes baked into a backend runtime image."""

    source_repository: str
    source_revision: str
    dependency_lock_digest: str


def backend_runtime_identity(variant: str) -> BackendRuntimeImageIdentity | None:
    """Return the immutable backend identity baked into ``variant``, if any."""
    if variant not in RUNTIME_VARIANTS:
        raise ValueError(f"unknown runtime variant: {variant!r}")
    if variant != "online-rl-verl-py313":
        return None
    profile = tomllib.loads(read_resource(VERL_PROFILE).decode("utf-8"))
    repository = profile.get("source_repository")
    revision = profile.get("fork_revision")
    digest = profile.get("dependency_lock_sha256")
    if not isinstance(repository, str) or not repository.startswith("https://"):
        raise ValueError("veRL profile has no canonical source repository")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("veRL profile has no full fork revision")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("veRL profile has no dependency-lock digest")
    return BackendRuntimeImageIdentity(repository, revision, digest)


def backend_runtime_labels(
    variant: str,
    identity: BackendRuntimeImageIdentity | None = None,
) -> dict[str, str]:
    """Return the OCI labels that prove a backend image's immutable identity.

    The labels are part of the job-kind image contract, rather than merely
    build diagnostics. Consumers compare them before packing so a registry tag
    cannot silently substitute a different backend fork beneath a matching
    framework lock.
    """
    resolved = identity if identity is not None else backend_runtime_identity(variant)
    if resolved is None:
        return {}
    if variant == "online-rl-verl-py313":
        return {
            VERL_SOURCE_REPOSITORY_LABEL: resolved.source_repository,
            VERL_SOURCE_REVISION_LABEL: resolved.source_revision,
            VERL_DEPENDENCY_LOCK_LABEL: resolved.dependency_lock_digest,
        }
    raise ValueError(f"no OCI backend-identity label contract for runtime variant {variant!r}")


@contextmanager
def definition_root() -> Iterator[Path]:
    """Yield a real directory containing the shipped `containers/` tree.

    This is the only extraction point on purpose. Every shipped Dockerfile
    names its inputs as `containers/...`, and `docker buildx bake` resolves a
    target's `dockerfile` relative to its `context`; both must therefore be
    rooted at the same real directory. Extracting bake files and build inputs
    through separate `as_file` calls could place them under different
    temporary directories when the distribution is not a plain filesystem
    tree, silently breaking that relationship.

    Use the module-level `*_DEFINITION`, `*_BAKE_FILE`, and `*_LOCK` paths to
    address material inside the yielded root.
    """
    with as_file(resource_files("posttrain.runtime_images")) as root:
        yield Path(root)


@cache
def cached_definition_root() -> Path:
    """Return a definition root that stays valid for the life of the process.

    `definition_root` is the right shape for a scoped read, but a build hands
    the extracted paths to `docker buildx`, which reads them long after any
    `with` block would have closed. This variant extracts at most once and
    releases at interpreter exit. When the distribution is an ordinary
    filesystem tree, `as_file` returns the real directory and nothing is
    copied.
    """
    stack = ExitStack()
    atexit.register(stack.close)
    return Path(stack.enter_context(as_file(resource_files("posttrain.runtime_images"))))


def constraint_lock(variant: str) -> PurePosixPath:
    """Return the dependency lock that constrains `variant`'s environment compiles."""
    if variant not in RUNTIME_VARIANTS:
        raise ValueError(f"unknown runtime variant: {variant!r}")
    return _KIND_CONSTRAINT_LOCKS[variant]


def backend_constraint_lock(variant: str) -> PurePosixPath | None:
    """Return the separately locked backend environment for ``variant``."""
    if variant not in RUNTIME_VARIANTS:
        raise ValueError(f"unknown runtime variant: {variant!r}")
    return _BACKEND_CONSTRAINT_LOCKS.get(variant)


def read_resource(relative: PurePosixPath) -> bytes:
    """Read one shipped file without extracting the whole definition tree."""
    resource = resource_files("posttrain.runtime_images")
    for part in relative.parts:
        resource = resource.joinpath(part)
    return resource.read_bytes()


def published_manifest_digest() -> str:
    """Return the exact shipped release-manifest digest for service admission."""

    return hashlib.sha256(read_resource(PUBLISHED_MANIFEST)).hexdigest()


def read_lock(lock: PurePosixPath) -> bytes:
    """Read a shipped dependency lock file."""
    return read_resource(lock)


def lock_digest(lock: PurePosixPath = BASE_LOCK) -> str:
    """Return the lowercase SHA-256 of a shipped lock file.

    The selected lock's digest is the published image identity recorded in the
    `org.carbonteq.posttrain.lock-digest` label. Computing it here removes the
    hand transcription that previously let a stale job-kind image and a fixed
    framework pin disagree without detection.
    """
    return hashlib.sha256(read_lock(lock)).hexdigest()


__all__ = [
    "BASE_BAKE_FILE",
    "BASE_DEFINITION",
    "BASE_LOCK",
    "BackendRuntimeImageIdentity",
    "JOB_BAKE_FILE",
    "JOB_DEFINITION",
    "KIND_BAKE_FILE",
    "KIND_DEFINITION",
    "RUNTIME_VARIANTS",
    "TRANSFORM_LOCK",
    "VERL_BACKEND_LOCK",
    "VERL_DEPENDENCY_LOCK_LABEL",
    "VERL_PROFILE",
    "VERL_SOURCE_REPOSITORY_LABEL",
    "VERL_SOURCE_REVISION_LABEL",
    "backend_runtime_labels",
    "backend_runtime_identity",
    "WORKSPACE_LOCK",
    "cached_definition_root",
    "backend_constraint_lock",
    "constraint_lock",
    "definition_root",
    "lock_digest",
    "read_lock",
    "read_resource",
]
