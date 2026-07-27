"""Release tooling, and the boundary that keeps it away from consumers."""

from __future__ import annotations

import tomllib
from pathlib import PurePosixPath

import pytest
from posttrain.runtime_images import RUNTIME_VARIANTS, constraint_lock, lock_digest
from posttrain.runtime_images.manifest import PublishedImage, load_manifest
from posttrain_release.manifest_render import render_manifest

_REPOSITORY_ROOT_DEPTH = 3


def _image(name: str, variant: str, digest_char: str) -> PublishedImage:
    lock = constraint_lock(variant)
    return PublishedImage(
        name=name,
        repository=f"posttrain-kind-{variant}",
        digest="sha256:" + digest_char * 64,
        lock_digest=lock_digest(lock),
        constraint_lock=lock,
    )


def _render_all() -> str:
    return render_manifest(
        framework_version="9.9.9",
        default_prefix="ghcr.io/carbonteq-ai",
        base=PublishedImage(
            name="base",
            repository="posttrain-base",
            digest="sha256:" + "a" * 64,
            lock_digest=lock_digest(),
            constraint_lock=PurePosixPath("containers/posttrain-job-kinds/locks/workspace.lock.txt"),
        ),
        kinds={
            variant: _image(f"kinds.{variant}", variant, str(index)) for index, variant in enumerate(RUNTIME_VARIANTS)
        },
    )


def test_rendered_manifest_parses_and_round_trips() -> None:
    document = tomllib.loads(_render_all())
    assert document["schema_version"] == 1
    assert document["framework_version"] == "9.9.9"
    assert document["default_prefix"] == "ghcr.io/carbonteq-ai"
    assert set(document["kinds"]) == set(RUNTIME_VARIANTS)


def test_rendered_lock_digests_come_from_the_shipped_locks() -> None:
    """The manifest can only ever claim what the shipped locks actually hash to."""
    document = tomllib.loads(_render_all())
    for variant in RUNTIME_VARIANTS:
        assert document["kinds"][variant]["lock_digest"] == lock_digest(constraint_lock(variant))


def test_transform_keeps_its_own_constraint_lock() -> None:
    document = tomllib.loads(_render_all())
    assert document["kinds"]["transform"]["constraint_lock"].endswith("transform.lock.txt")
    assert document["kinds"]["supervised"]["constraint_lock"].endswith("workspace.lock.txt")
    assert document["kinds"]["transform"]["lock_digest"] != document["kinds"]["supervised"]["lock_digest"]


def test_provided_packages_survive_rendering() -> None:
    lock = constraint_lock("eval")
    rendered = render_manifest(
        framework_version="1.0.0",
        default_prefix="ghcr.io/carbonteq-ai",
        base=PublishedImage(
            name="base",
            repository="posttrain-base",
            digest="sha256:" + "a" * 64,
            lock_digest=lock_digest(),
            constraint_lock=constraint_lock("supervised"),
        ),
        kinds={
            "eval": PublishedImage(
                name="kinds.eval",
                repository="posttrain-kind-eval",
                digest="sha256:" + "b" * 64,
                lock_digest=lock_digest(lock),
                constraint_lock=lock,
                provided_packages=("verifiers",),
            )
        },
    )
    document = tomllib.loads(rendered)
    assert document["kinds"]["eval"]["provided_packages"] == ["verifiers"]


def test_an_empty_release_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one job-kind image"):
        render_manifest(
            framework_version="1.0.0",
            default_prefix="ghcr.io/carbonteq-ai",
            base=PublishedImage(
                name="base",
                repository="posttrain-base",
                digest="sha256:" + "a" * 64,
                lock_digest=lock_digest(),
                constraint_lock=constraint_lock("supervised"),
            ),
            kinds={},
        )


def test_the_shipped_manifest_matches_what_the_renderer_would_produce() -> None:
    """The committed manifest must be generated output, not hand-edited.

    Digests differ because the shipped manifest records a real publication, so
    only the generated structure is compared.
    """
    shipped = load_manifest()
    rendered = tomllib.loads(_render_all())
    assert set(rendered["kinds"]) == set(shipped.kinds)
    for variant, image in shipped.kinds.items():
        assert rendered["kinds"][variant]["repository"] == image.repository
        assert rendered["kinds"][variant]["constraint_lock"] == image.constraint_lock.as_posix()
        assert rendered["kinds"][variant]["lock_digest"] == image.lock_digest


def test_the_consumer_distribution_does_not_depend_on_release_tooling() -> None:
    """Publishing must be unreachable from a consumer environment.

    `posttrain` is what a project developer installs. If it depended on this
    package, release authority would land in every consumer's environment.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[_REPOSITORY_ROOT_DEPTH]
    consumer = tomllib.loads((root / "apps/cli/pyproject.toml").read_text(encoding="utf-8"))
    declared = consumer["project"]["dependencies"]
    optional = [
        dependency for extra in consumer["project"].get("optional-dependencies", {}).values() for dependency in extra
    ]
    assert not any("posttrain-release" in item for item in [*declared, *optional])
