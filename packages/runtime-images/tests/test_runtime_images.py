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


def test_base_accepts_a_build_secret_ca_bundle_without_disabling_tls() -> None:
    with definition_root() as root:
        dockerfile = (root / "containers/posttrain-base/Dockerfile").read_text()

    assert "--mount=type=secret,id=posttrain_ca_bundle,required=false" in dockerfile
    assert "cat /run/secrets/posttrain_ca_bundle >>" in dockerfile
    assert "/etc/ssl/certs/ca-certificates.crt" in dockerfile
    assert "install -m 0644 /run/secrets/posttrain_ca_bundle" not in dockerfile
    assert "trusted-host" not in dockerfile
    assert "allow-insecure" not in dockerfile


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
        constraint_lock("unknown-runtime")


def test_lock_digest_is_computed_from_the_shipped_bytes() -> None:
    assert lock_digest() == hashlib.sha256(read_lock(WORKSPACE_LOCK)).hexdigest()
    assert lock_digest() == lock_digest(WORKSPACE_LOCK)
    assert lock_digest(TRANSFORM_LOCK) != lock_digest(WORKSPACE_LOCK)


def test_every_shipped_lock_has_a_distinct_stable_digest() -> None:
    """Identity comes from the bytes, never from a value restated in a test.

    An earlier version of this test asserted one hardcoded digest. That is the
    same hand transcription this package exists to remove: it has to be edited
    by a human every time a lock legitimately changes, and a wrong edit makes
    it agree with nothing. Whether a published image still matches its lock is
    checked against the manifest in test_manifest.py, from the shipped bytes.
    """
    digests = {lock_digest(WORKSPACE_LOCK), lock_digest(TRANSFORM_LOCK)}
    assert len(digests) == 2
    assert all(len(d) == 64 and d == d.lower() for d in digests)
    assert lock_digest(WORKSPACE_LOCK) == lock_digest(WORKSPACE_LOCK)
