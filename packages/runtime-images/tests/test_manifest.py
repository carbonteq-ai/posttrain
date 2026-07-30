from __future__ import annotations

import tomllib
from pathlib import PurePosixPath

import pytest
from posttrain.runtime_images import (
    RUNTIME_VARIANTS,
    constraint_lock,
    lock_digest,
    read_resource,
)
from posttrain.runtime_images.manifest import (
    ManifestError,
    PublishedImage,
    _image,
    _verify,
    load_manifest,
)


def test_every_released_variant_is_published() -> None:
    manifest = load_manifest()
    assert set(manifest.kinds) == set(RUNTIME_VARIANTS)


def test_verl_is_published() -> None:
    assert "online-rl-verl-py313" in load_manifest().kinds


def test_recorded_lock_digest_matches_the_shipped_lock_bytes() -> None:
    """The check that would have caught the distillation gate failure.

    A job-kind image is only valid for the dependency closure it was built
    from. If a lock is edited without republishing, this fails.
    """
    manifest = load_manifest()
    for image in (manifest.base, *manifest.kinds.values()):
        assert image.lock_digest == lock_digest(image.constraint_lock), (
            f"{image.name} records a lock digest that no longer matches {image.constraint_lock}"
        )


def test_manifest_agrees_with_the_variant_constraint_lock_mapping() -> None:
    manifest = load_manifest()
    for variant in RUNTIME_VARIANTS:
        assert manifest.image(variant).constraint_lock == constraint_lock(variant)


def test_expected_lock_digest_is_recomputed_not_read_back() -> None:
    manifest = load_manifest()
    for variant in RUNTIME_VARIANTS:
        assert manifest.expected_lock_digest(variant) == lock_digest(constraint_lock(variant))


def test_published_digests_are_distinct_per_variant() -> None:
    manifest = load_manifest()
    digests = [manifest.image(v).digest for v in RUNTIME_VARIANTS]
    assert len(set(digests)) == len(digests), "two variants share one image digest"
    assert manifest.base.digest not in digests


def test_framework_publishes_to_a_public_registry_by_default() -> None:
    """`default_prefix` is the framework's release location, not a project's.

    A project's own registry is configured per site via POSTTRAIN_REGISTRY and
    must never leak into the shipped manifest.
    """
    assert load_manifest().default_prefix == "registry.lan/carbonteq"


def test_reference_is_digest_pinned_and_prefix_overridable() -> None:
    manifest = load_manifest()
    default = manifest.reference("supervised")
    assert default.startswith("registry.lan/carbonteq/")
    assert "@sha256:" in default

    # A site that cannot reach the public registry mirrors into its own.
    mirrored = manifest.reference("supervised", prefix="registry.internal/team/")
    assert mirrored == (
        f"registry.internal/team/{manifest.image('supervised').repository}@{manifest.image('supervised').digest}"
    )
    assert mirrored.rsplit("@", 1)[1] == default.rsplit("@", 1)[1], "a digest-level mirror must preserve image identity"


def test_provided_packages_are_carried_from_the_published_image() -> None:
    manifest = load_manifest()
    assert manifest.image("online-rl-trl-py312").provided_packages == ("verifiers",)
    assert manifest.image("eval").provided_packages == ("verifiers",)
    assert manifest.image("supervised").provided_packages == ()


def test_unknown_variant_reports_what_is_published() -> None:
    with pytest.raises(ManifestError, match="no published job-kind image"):
        load_manifest().image("does-not-exist")


def test_drifted_lock_digest_is_rejected() -> None:
    drifted = PublishedImage(
        name="kinds.supervised",
        repository="posttrain-kind-supervised",
        digest="sha256:" + "a" * 64,
        lock_digest="b" * 64,
        constraint_lock=PurePosixPath("containers/posttrain-job-kinds/locks/workspace.lock.txt"),
    )
    with pytest.raises(ManifestError, match="must be republished"):
        _verify(drifted)


def test_unshipped_constraint_lock_is_rejected() -> None:
    missing = PublishedImage(
        name="kinds.ghost",
        repository="posttrain-kind-ghost",
        digest="sha256:" + "a" * 64,
        lock_digest="b" * 64,
        constraint_lock=PurePosixPath("containers/posttrain-job-kinds/locks/nope.lock.txt"),
    )
    with pytest.raises(ManifestError, match="not shipped"):
        _verify(missing)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"repository": "r", "digest": "sha256:" + "a" * 64}, "lock_digest"),
        ({"repository": "r", "digest": "notadigest", "lock_digest": "b" * 64}, "sha256 OCI digest"),
        ({"repository": "", "digest": "sha256:" + "a" * 64}, "repository"),
    ],
)
def test_malformed_entries_are_rejected(payload: dict[str, object], expected: str) -> None:
    with pytest.raises(ManifestError, match=expected):
        _image("kinds.broken", payload)


def test_provided_packages_must_be_strings() -> None:
    with pytest.raises(ManifestError, match="provided_packages"):
        _image(
            "kinds.broken",
            {
                "repository": "r",
                "digest": "sha256:" + "a" * 64,
                "lock_digest": "b" * 64,
                "constraint_lock": "containers/posttrain-job-kinds/locks/workspace.lock.txt",
                "provided_packages": [1],
            },
        )


def test_shipped_manifest_declares_the_supported_schema() -> None:
    document = tomllib.loads(read_resource(PurePosixPath("published.toml")).decode())
    assert document["schema_version"] == 1
    assert load_manifest().schema_version == 1


def test_a_digest_cannot_be_a_mirror_destination() -> None:
    """A digest is derived from content, so it cannot be pushed to.

    Naming a mirror destination by digest is silently wrong: it looks correct
    because the source is addressed that way, and the registry rejects it only
    after auth succeeds and blob upload begins.
    """
    from posttrain_execution_buildkit import RuntimeImageInspector

    manifest = load_manifest()
    digest_reference = manifest.base.reference("ghcr.io/example")
    assert "@sha256:" in digest_reference

    with pytest.raises(ValueError, match="must be a tag, not a digest"):
        RuntimeImageInspector().copy("source:tag", digest_reference)
