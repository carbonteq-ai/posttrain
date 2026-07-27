"""Access to the framework's shipped container definitions.

The definitions live as package data so an installed distribution carries the
exact inputs that produced its runtime images, rather than a reference to a
source checkout it may not have.
"""

from __future__ import annotations

import atexit
import hashlib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import cache
from importlib.resources import as_file
from importlib.resources import files as resource_files
from pathlib import Path, PurePosixPath

RUNTIME_VARIANTS: tuple[str, ...] = (
    "supervised",
    "online-rl-trl-py312",
    "eval",
    "serve",
    "transform",
)
"""Job-kind variants published once per framework release.

`online-rl-verl-py313` is deliberately absent: it remains release-blocked and
is qualified out of band.
"""

_CONTAINERS = PurePosixPath("containers")
BASE_DEFINITION = _CONTAINERS / "posttrain-base"
KIND_DEFINITION = _CONTAINERS / "posttrain-job-kinds"
JOB_DEFINITION = _CONTAINERS / "posttrain-job"

BASE_BAKE_FILE = BASE_DEFINITION / "docker-bake.hcl"
KIND_BAKE_FILE = KIND_DEFINITION / "docker-bake.hcl"
JOB_BAKE_FILE = JOB_DEFINITION / "docker-bake.hcl"

WORKSPACE_LOCK = KIND_DEFINITION / "locks" / "workspace.lock.txt"
TRANSFORM_LOCK = KIND_DEFINITION / "locks" / "transform.lock.txt"


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
    return TRANSFORM_LOCK if variant == "transform" else WORKSPACE_LOCK


def read_resource(relative: PurePosixPath) -> bytes:
    """Read one shipped file without extracting the whole definition tree."""
    resource = resource_files("posttrain.runtime_images")
    for part in relative.parts:
        resource = resource.joinpath(part)
    return resource.read_bytes()


def read_lock(lock: PurePosixPath) -> bytes:
    """Read a shipped dependency lock file."""
    return read_resource(lock)


def lock_digest(lock: PurePosixPath = WORKSPACE_LOCK) -> str:
    """Return the lowercase SHA-256 of a shipped lock file.

    The workspace lock's digest is the published image identity recorded in the
    `org.carbonteq.posttrain.lock-digest` label. Computing it here removes the
    hand transcription that previously let a stale job-kind image and a fixed
    framework pin disagree without detection.
    """
    return hashlib.sha256(read_lock(lock)).hexdigest()


__all__ = [
    "BASE_BAKE_FILE",
    "BASE_DEFINITION",
    "JOB_BAKE_FILE",
    "JOB_DEFINITION",
    "KIND_BAKE_FILE",
    "KIND_DEFINITION",
    "RUNTIME_VARIANTS",
    "TRANSFORM_LOCK",
    "WORKSPACE_LOCK",
    "cached_definition_root",
    "constraint_lock",
    "definition_root",
    "lock_digest",
    "read_lock",
    "read_resource",
]
