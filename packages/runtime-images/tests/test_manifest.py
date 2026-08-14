from __future__ import annotations

import hashlib
import tomllib
from pathlib import PurePosixPath

import pytest
from posttrain.runtime_images import (
    RUNTIME_VARIANTS,
    backend_runtime_identity,
    constraint_lock,
    lock_digest,
    read_resource,
)
from posttrain.runtime_images.manifest import (
    ManifestError,
    PublishedImage,
    _image,
    _verify,
    load_manifest_from_directory,
)
from posttrain.runtime_images.manifest import (
    load_manifest as _load_manifest,
)


def _candidate_manifest():
    """Inspect declared candidate shape without accepting it as runnable."""

    return _load_manifest(verify_locks=False)


def test_every_released_variant_is_published() -> None:
    manifest = _candidate_manifest()
    assert set(manifest.kinds) == set(RUNTIME_VARIANTS)


def test_verl_is_published() -> None:
    assert "online-rl-verl-py313" in _candidate_manifest().kinds


def test_verl_manifest_carries_the_runtime_backend_identity() -> None:
    assert _candidate_manifest().image("online-rl-verl-py313").backend_runtime_identity == backend_runtime_identity(
        "online-rl-verl-py313"
    )


def test_candidate_manifest_declares_lock_digests_for_every_image() -> None:
    """Candidate shape remains inspectable while strict consumer load stays separate."""

    manifest = _candidate_manifest()
    for image in (manifest.base, *manifest.kinds.values()):
        assert len(image.lock_digest) == 64


def test_manifest_agrees_with_the_variant_constraint_lock_mapping() -> None:
    manifest = _candidate_manifest()
    for variant in RUNTIME_VARIANTS:
        # A candidate may still carry the preceding published image record
        # while generated current-release locks await image materialization.
        # It must nevertheless reference a shipped lock, and the current
        # variant mapping must also resolve to a shipped lock.
        assert read_resource(manifest.image(variant).constraint_lock)
        assert read_resource(constraint_lock(variant))


def test_expected_lock_digest_is_recomputed_not_read_back() -> None:
    manifest = _candidate_manifest()
    for variant in RUNTIME_VARIANTS:
        image = manifest.image(variant)
        assert manifest.expected_lock_digest(variant) == lock_digest(image.constraint_lock)


def test_published_digests_are_distinct_per_variant() -> None:
    manifest = _candidate_manifest()
    digests = [manifest.image(v).digest for v in RUNTIME_VARIANTS]
    assert len(set(digests)) == len(digests), "two variants share one image digest"
    assert manifest.base.digest not in digests


def test_framework_publishes_to_a_public_registry_by_default() -> None:
    """`default_prefix` is the framework's release location, not a project's.

    A project's own registry is configured per site via POSTTRAIN_REGISTRY and
    must never leak into the shipped manifest.
    """
    assert _candidate_manifest().default_prefix == "registry.lan/carbonteq"


def test_reference_is_digest_pinned_and_prefix_overridable() -> None:
    manifest = _candidate_manifest()
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
    manifest = _candidate_manifest()
    assert manifest.image("online-rl-trl-py312").provided_packages == ("verifiers",)
    assert manifest.image("eval").provided_packages == ("verifiers",)
    assert manifest.image("supervised").provided_packages == ()


def test_unknown_variant_reports_what_is_published() -> None:
    with pytest.raises(ManifestError, match="no published job-kind image"):
        _candidate_manifest().image("does-not-exist")


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


def test_release_check_can_validate_a_staged_manifest_without_importing_local_package_data(tmp_path) -> None:
    """A candidate checkout must not accidentally validate this worktree's manifest."""

    lock = tmp_path / "locks/base.lock.txt"
    lock.parent.mkdir()
    lock.write_text("torch==2.11.0\n", encoding="utf-8")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    (tmp_path / "published.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                'framework_version = "test"',
                'default_prefix = "registry.example/team"',
                "",
                "[base]",
                'repository = "posttrain-base"',
                'digest = "sha256:' + "a" * 64 + '"',
                f'lock_digest = "{digest}"',
                'constraint_lock = "locks/base.lock.txt"',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = load_manifest_from_directory(tmp_path, verify_variants=False)

    assert manifest.base.lock_digest == digest
    lock.write_text("torch==2.11.1\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="must be republished"):
        load_manifest_from_directory(tmp_path, verify_variants=False)


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
    assert _candidate_manifest().schema_version == 1


def test_a_digest_cannot_be_a_mirror_destination() -> None:
    """A digest is derived from content, so it cannot be pushed to.

    Naming a mirror destination by digest is silently wrong: it looks correct
    because the source is addressed that way, and the registry rejects it only
    after auth succeeds and blob upload begins.
    """
    from posttrain_execution_buildkit import RuntimeImageInspector

    manifest = _candidate_manifest()
    digest_reference = manifest.base.reference("ghcr.io/example")
    assert "@sha256:" in digest_reference

    with pytest.raises(ValueError, match="must be a tag, not a digest"):
        RuntimeImageInspector().copy("source:tag", digest_reference)
