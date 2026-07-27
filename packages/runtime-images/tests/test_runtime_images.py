from __future__ import annotations

import hashlib
import re

import pytest
from posttrain.runtime_images import (
    BASE_BAKE_FILE,
    JOB_BAKE_FILE,
    KIND_BAKE_FILE,
    RUNTIME_VARIANTS,
    TRANSFORM_LOCK,
    WORKSPACE_LOCK,
    constraint_lock,
    definition_root,
    lock_digest,
    read_lock,
)


def test_definition_root_exposes_all_three_image_levels() -> None:
    with definition_root() as root:
        for bake in (BASE_BAKE_FILE, KIND_BAKE_FILE, JOB_BAKE_FILE):
            assert (root / bake).is_file(), f"missing shipped bake file: {bake}"


def test_shipped_dockerfile_input_paths_resolve_against_the_definition_root() -> None:
    """Every `COPY containers/...` in a shipped Dockerfile must resolve.

    This is the property that lets the build context point at the definition
    root without editing any shipped Dockerfile or bake file.
    """
    copied = re.compile(r"^COPY\s+(containers/\S+)", re.MULTILINE)
    with definition_root() as root:
        for level in ("posttrain-base", "posttrain-job-kinds"):
            dockerfile = root / "containers" / level / "Dockerfile"
            referenced = copied.findall(dockerfile.read_text())
            assert referenced, f"expected context inputs in {level}/Dockerfile"
            for path in referenced:
                assert (root / path).exists(), f"{level}/Dockerfile copies missing {path}"


def test_runtime_variants_match_the_published_bake_targets() -> None:
    published = re.compile(r'^target "posttrain-kind-([a-z0-9-]+)" \{', re.MULTILINE)
    with definition_root() as root:
        targets = published.findall((root / KIND_BAKE_FILE).read_text())
    assert set(RUNTIME_VARIANTS) == {t for t in targets if not t.endswith("-smoke")}


@pytest.mark.parametrize("variant", RUNTIME_VARIANTS)
def test_every_variant_names_a_shipped_constraint_lock(variant: str) -> None:
    lock = constraint_lock(variant)
    assert read_lock(lock), f"{variant} constraint lock is empty"


def test_transform_is_the_only_variant_off_the_workspace_lock() -> None:
    off_workspace = {v for v in RUNTIME_VARIANTS if constraint_lock(v) != WORKSPACE_LOCK}
    assert off_workspace == {"transform"}
    assert constraint_lock("transform") == TRANSFORM_LOCK


def test_constraint_lock_rejects_unknown_variants() -> None:
    with pytest.raises(ValueError, match="unknown runtime variant"):
        constraint_lock("online-rl-verl-py313")


def test_lock_digest_is_computed_from_the_shipped_bytes() -> None:
    assert lock_digest() == hashlib.sha256(read_lock(WORKSPACE_LOCK)).hexdigest()
    assert lock_digest() == lock_digest(WORKSPACE_LOCK)
    assert lock_digest(TRANSFORM_LOCK) != lock_digest(WORKSPACE_LOCK)


def test_lock_digest_matches_the_published_image_identity() -> None:
    """Guards the transcription that let a stale kind image go undetected.

    The published `org.carbonteq.posttrain.lock-digest` label is the SHA-256 of
    the shipped workspace lock. If this value changes, the job-kind images must
    be republished; a fixed framework pin alone is not sufficient.
    """
    assert lock_digest() == "e8a833bf24f5fe5459ee69eb04d26a9ea5cfc49bd0b6dd8dc3b678c310fcfbbd"
