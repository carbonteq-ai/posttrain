"""Obtain the framework's own distributions for an actual-job image.

The job-kind image already carries the framework's dependency closure, so an
actual-job image needs framework *code* and nothing else. When the framework is
installed rather than checked out, that code is available as built
distributions, and staging those wheels is both simpler and stricter than
copying a source tree: identity becomes a digest over the exact artifacts that
will be installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from posttrain.common import ContractError

# Distributions that carry framework code into a job. Kept as an explicit list
# because it defines what a job image contains, which is not something to infer.
FRAMEWORK_DISTRIBUTIONS = (
    "posttrain",
    "posttrain-catalog",
    "posttrain-common",
    "posttrain-data",
    "posttrain-environment",
    "posttrain-eval",
    "posttrain-execution",
    "posttrain-execution-buildkit",
    "posttrain-execution-dstack",
    "posttrain-execution-local",
    "posttrain-execution-pack",
    "posttrain-jobs",
    "posttrain-project",
    "posttrain-runtime",
    "posttrain-runtime-images",
    "posttrain-serve",
    "posttrain-tracking",
    "posttrain-tracking-trackio",
    "posttrain-train",
    "posttrain-work",
)

# Provides the actual-job image's entry point, so it must be in every job image
# even though nothing on a developer's machine imports it.
IMAGE_ONLY_DISTRIBUTION = "posttrain-runtime"

_WHEEL = re.compile(r"^[A-Za-z0-9._]+-[^-]+-.*\.whl$")


@dataclass(frozen=True, slots=True)
class FrameworkDistributions:
    """The framework artifacts that will be installed into a job image."""

    wheels: tuple[Path, ...]
    digest: str

    @property
    def filenames(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.wheels)


def installed_versions() -> dict[str, str]:
    """Return the framework distributions to stage, at the versions to stage.

    Most are read from the environment doing the packing, which is what makes
    the job image agree with the developer who built it. `posttrain-runtime` is
    the exception: it provides the image's entry point but runs only inside the
    image, so a consumer has no reason to install it and normally has not. It is
    therefore requested at the framework's own version rather than skipped,
    which would produce an image whose entry point does not exist.
    """
    found: dict[str, str] = {}
    for name in FRAMEWORK_DISTRIBUTIONS:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    if "posttrain" not in found:
        raise ContractError(
            "the posttrain distribution is not installed, so its framework code cannot be packed into a job image"
        )
    found.setdefault(IMAGE_ONLY_DISTRIBUTION, found["posttrain"])
    return found


def _digest(wheels: tuple[Path, ...]) -> str:
    """Digest the exact bytes that will be installed, not their names."""
    entries = []
    for wheel in sorted(wheels, key=lambda path: path.name):
        digest = hashlib.sha256()
        with wheel.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"name": wheel.name, "sha256": digest.hexdigest()})
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _index_arguments(environ: Mapping[str, str]) -> list[str]:
    """Forward the consumer's index to pip, which reads different variables.

    uv is configured through `UV_INDEX_URL`, and pip through `PIP_INDEX_URL`. A
    consumer who configured only uv, which is what the framework's own
    documentation tells them to do, would otherwise have their private index
    silently ignored here and fall back to PyPI, where these distributions do
    not exist.
    """
    if environ.get("PIP_INDEX_URL"):
        return []
    index = environ.get("UV_INDEX_URL") or environ.get("UV_DEFAULT_INDEX")
    return ["--index-url", index] if index else []


def materialize(
    destination: Path,
    *,
    uv: str = "uv",
    environ: Mapping[str, str] | None = None,
    wheelhouse: Path | None = None,
) -> FrameworkDistributions:
    """Download the installed framework distributions as wheels.

    Resolution uses whatever index the environment already configures, so a site
    with a private index needs no framework-specific setting: the wheels come
    from wherever the consumer installed them.

    uv has no download command, so this borrows pip through an ephemeral uv
    environment rather than requiring the consumer's own environment to carry
    pip, which a uv-created virtual environment does not.
    """
    environ = os.environ if environ is None else environ
    destination.mkdir(parents=True, exist_ok=True)
    # The destination is reused between packs. Wheels left by a previously
    # installed framework would otherwise be staged alongside the current ones,
    # and the image would be asked to install two versions of every package.
    for stale in destination.glob("*.whl"):
        stale.unlink()
    versions = installed_versions()
    if wheelhouse is not None:
        source = wheelhouse.expanduser().resolve()
        if not source.is_dir():
            raise ContractError(f"framework wheelhouse is not a directory: {source}")
        for name, version in sorted(versions.items()):
            normalized = name.replace("-", "_").lower()
            matches = tuple(sorted(source.glob(f"{normalized}-{version}-*.whl")))
            if len(matches) != 1:
                found = ", ".join(path.name for path in matches) or "none"
                raise ContractError(
                    f"framework wheelhouse must contain exactly one {name}=={version} wheel; found: {found}"
                )
            shutil.copy2(matches[0], destination / matches[0].name)
        return _validated_distributions(destination, versions)

    requirements = [f"{name}=={version}" for name, version in sorted(versions.items())]
    result = subprocess.run(
        [
            uv,
            "run",
            "--no-project",
            "--with",
            "pip",
            "python",
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--dest",
            str(destination),
            *_index_arguments(environ),
            *requirements,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1500:]
        raise ContractError(
            "framework distributions could not be obtained for packing. They are "
            "normally served by the same index the framework was installed from. "
            f"Requested: {' '.join(requirements)}. {detail}"
        )
    return _validated_distributions(destination, versions)


def _validated_distributions(destination: Path, versions: Mapping[str, str]) -> FrameworkDistributions:
    wheels = tuple(sorted(path for path in destination.glob("*.whl") if _WHEEL.match(path.name)))
    if not wheels:
        raise ContractError(f"no framework wheels were produced in {destination}")
    missing = []
    for name in versions:
        prefix = name.replace("-", "_").lower() + "-"
        matched = [path for path in wheels if path.name.lower().startswith(prefix)]
        if not matched:
            missing.append(name)
        elif len(matched) > 1:
            # One distribution resolving to several wheels means the image would
            # be told to install more than one version of the same package.
            raise ContractError(
                f"framework distribution {name} produced more than one wheel: "
                + ", ".join(sorted(path.name for path in matched))
            )
    if missing:
        raise ContractError("framework wheels are missing for: " + ", ".join(sorted(missing)))
    return FrameworkDistributions(wheels=wheels, digest=_digest(wheels))


__all__ = [
    "FRAMEWORK_DISTRIBUTIONS",
    "IMAGE_ONLY_DISTRIBUTION",
    "FrameworkDistributions",
    "installed_versions",
    "materialize",
]
