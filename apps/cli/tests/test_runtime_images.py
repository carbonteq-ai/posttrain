"""Runtime image verification, and the drift that must never be walked past."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.runtime_images.manifest import load_manifest
from posttrain_cli.checks import runtime_images_check
from posttrain_cli.cli import main
from posttrain_cli.context import CliState
from posttrain_cli.execution_config import (
    REGISTRY_ENVIRONMENT_VARIABLE,
    load_local_execution_config,
)
from posttrain_cli.runtime_images import (
    ensure_kind_image_ready,
    verify_registry,
    verify_variant,
)
from posttrain_execution_buildkit import (
    IMAGE_LEVEL_LABEL,
    LOCK_DIGEST_LABEL,
    REVISION_LABEL,
    RemoteImageFacts,
    RemoteImageNotFoundError,
)

_MANIFEST = load_manifest()
_EXPECTED_LOCK = _MANIFEST.expected_lock_digest("supervised")


class _FakeInspector:
    """Stands in for a registry without needing one."""

    def __init__(
        self,
        *,
        lock_digest: str | None = _EXPECTED_LOCK,
        missing: bool = False,
        image_level: str | None = "job-kind",
        revision: str | None = None,
    ) -> None:
        self._lock_digest = lock_digest
        self._missing = missing
        self._image_level = image_level
        self._revision = revision
        self.inspected: list[str] = []

    def inspect(self, reference: str) -> RemoteImageFacts:
        self.inspected.append(reference)
        if self._missing:
            raise RemoteImageNotFoundError(reference)
        labels: dict[str, str] = {}
        if self._lock_digest is not None:
            labels[LOCK_DIGEST_LABEL] = self._lock_digest
        if self._image_level is not None:
            labels[IMAGE_LEVEL_LABEL] = self._image_level
        if self._revision is not None:
            labels[REVISION_LABEL] = self._revision
        return RemoteImageFacts(
            reference=reference,
            digest=reference.rsplit("@", 1)[1],
            labels=labels,
        )


class _UnreachableInspector:
    def inspect(self, reference: str) -> RemoteImageFacts:
        raise RuntimeError("connection refused")


def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")
    layout = CliState(project_root=project).layout()
    configuration = load_local_execution_config(layout)
    assert configuration.registry is not None
    return configuration.registry


def test_matching_lock_digest_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(tmp_path, monkeypatch)
    result = verify_variant(
        "supervised",
        registry.kind_images["supervised"].value,
        manifest=_MANIFEST,
        inspector=_FakeInspector(),
    )
    assert result.status == "ok"
    assert result.ok


def test_a_stale_image_is_reported_as_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distillation failure, made visible.

    The published image was built from an older dependency lock than the one
    the framework ships. Nothing detected it, and a GPU qualification ran to
    completion against the wrong software.
    """
    registry = _registry(tmp_path, monkeypatch)
    stale = hashlib.sha256(b"an older workspace lock").hexdigest()
    result = verify_variant(
        "supervised",
        registry.kind_images["supervised"].value,
        manifest=_MANIFEST,
        inspector=_FakeInspector(lock_digest=stale),
    )
    assert result.status == "drifted"
    assert stale in result.detail
    assert _EXPECTED_LOCK in result.detail
    assert "must be republished" in result.detail


def test_an_unlabelled_image_is_drift_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    result = verify_variant(
        "supervised",
        registry.kind_images["supervised"].value,
        manifest=_MANIFEST,
        inspector=_FakeInspector(lock_digest=None),
    )
    assert result.status == "drifted"
    assert LOCK_DIGEST_LABEL in result.detail


def test_absent_and_unreachable_are_distinguished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry that cannot be reached is not the same as a missing image."""
    registry = _registry(tmp_path, monkeypatch)
    reference = registry.kind_images["supervised"].value

    absent = verify_variant(
        "supervised", reference, manifest=_MANIFEST, inspector=_FakeInspector(missing=True)
    )
    assert absent.status == "missing"

    unreachable = verify_variant(
        "supervised", reference, manifest=_MANIFEST, inspector=_UnreachableInspector()
    )
    assert unreachable.status == "unreachable"


def test_verify_registry_covers_every_published_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    inspector = _FakeInspector()
    results = verify_registry(registry, manifest=_MANIFEST, inspector=inspector)
    assert {result.variant for result in results} == set(_MANIFEST.kinds)
    # transform is constrained by a different lock, so a single expected digest
    # would have silently passed it; each variant must be checked on its own.
    transform = next(r for r in results if r.variant == "transform")
    assert transform.expected_lock_digest == _MANIFEST.expected_lock_digest("transform")
    assert transform.expected_lock_digest != _EXPECTED_LOCK


def test_verify_registry_rejects_unknown_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="no runtime image is configured"):
        verify_registry(
            registry,
            variants=["nope"],
            manifest=_MANIFEST,
            inspector=_FakeInspector(),
        )


def test_packing_refuses_a_drifted_image_and_names_the_remedy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    stale = hashlib.sha256(b"an older workspace lock").hexdigest()
    with pytest.raises(ContractError) as raised:
        ensure_kind_image_ready(
            registry,
            "supervised",
            manifest=_MANIFEST,
            inspector=_FakeInspector(lock_digest=stale),
        )
    message = str(raised.value)
    assert "supervised" in message
    assert stale in message
    assert _EXPECTED_LOCK in message
    assert "--build-missing" in message


def test_packing_refuses_a_missing_image_and_suggests_mirroring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="runtime images mirror"):
        ensure_kind_image_ready(
            registry,
            "supervised",
            manifest=_MANIFEST,
            inspector=_FakeInspector(missing=True),
        )


def test_an_unreachable_registry_never_silently_proceeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--build-missing must not turn a network failure into a rebuild."""
    registry = _registry(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="could not be queried"):
        ensure_kind_image_ready(
            registry,
            "supervised",
            build_missing=True,
            manifest=_MANIFEST,
            inspector=_UnreachableInspector(),
        )


def test_verified_image_passes_without_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    result = ensure_kind_image_ready(
        registry,
        "supervised",
        build_missing=True,
        manifest=_MANIFEST,
        inspector=_FakeInspector(),
    )
    assert result.ok


def test_doctor_fails_when_a_configured_image_is_not_this_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """A pinned older image must stop the command, not warn quietly."""
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")

    older = "sha256:" + "9" * 64
    config = project / ".posttrain" / "state" / "execution.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "schema_version = 1\n\n[registry]\n\n[registry.kind_images]\n"
        f'supervised = "registry.internal/team/posttrain-kind-supervised@{older}"\n',
        encoding="utf-8",
    )
    config.chmod(0o600)

    assert main(["--json", "--project-root", str(project), "doctor"]) == 1
    payload = json.loads(capsys.readouterr().out)
    check = next(c for c in payload["checks"] if c["name"] == "runtime_images")
    assert check["status"] == "error"
    assert "supervised" in check["message"]
    assert "9" * 64 in check["message"]
    assert _MANIFEST.kinds["supervised"].digest.removeprefix("sha256:") in check["message"]


def test_runtime_images_check_ignores_unpublished_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release-blocked variant has nothing in the release to disagree with."""
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")
    lock = Path(__file__).resolve().parents[3] / (
        "packages/runtime-images/src/posttrain/runtime_images/containers/"
        "posttrain-job-kinds/locks/workspace.lock.txt"
    )
    config = project / ".posttrain" / "state" / "execution.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "schema_version = 1\n\n[registry]\n\n[registry.kind_images]\n"
        f'custom-backend = "registry.internal/team/custom@sha256:{"7" * 64}"\n\n'
        "[registry.constraint_profiles.custom-backend]\n"
        f'path = "{lock}"\n'
        f'sha256 = "{hashlib.sha256(lock.read_bytes()).hexdigest()}"\n',
        encoding="utf-8",
    )
    config.chmod(0o600)

    check = runtime_images_check(CliState(project_root=project))
    assert check.status == "ok"


def test_list_works_offline(tmp_path: Path, capsys) -> None:
    """`list` is a local read and must not require a registry."""
    assert main(["--json", "runtime", "images", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["variant"] for row in payload["kinds"]} == set(_MANIFEST.kinds)
    assert payload["default_prefix"] == _MANIFEST.default_prefix


def test_a_base_image_pinned_into_a_kind_slot_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lock digest alone cannot catch this.

    Every level of the image hierarchy is built from the same workspace lock,
    so the base image and every job-kind image share a lock digest. Pointing a
    job-kind slot at the base image therefore passes a digest comparison
    perfectly while producing an image that cannot run a job at all.
    """
    registry = _registry(tmp_path, monkeypatch)
    assert _MANIFEST.base.lock_digest == _MANIFEST.kinds["supervised"].lock_digest

    result = verify_variant(
        "supervised",
        registry.kind_images["supervised"].value,
        manifest=_MANIFEST,
        # A real base image: correct lock digest, wrong level.
        inspector=_FakeInspector(lock_digest=_EXPECTED_LOCK, image_level="base"),
    )
    assert result.status == "drifted"
    assert "job-kind" in result.detail
    assert "base" in result.detail


def test_an_actual_job_image_is_also_rejected_in_a_kind_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path, monkeypatch)
    result = verify_variant(
        "supervised",
        registry.kind_images["supervised"].value,
        manifest=_MANIFEST,
        inspector=_FakeInspector(image_level="actual-job"),
    )
    assert result.status == "drifted"
    assert "actual-job" in result.detail


def test_verification_reports_the_framework_revision_that_built_the_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance was previously recovered by a human reading OCI labels."""
    registry = _registry(tmp_path, monkeypatch)
    reference = registry.kind_images["supervised"].value

    verified = verify_variant(
        "supervised",
        reference,
        manifest=_MANIFEST,
        inspector=_FakeInspector(revision="abc123def456"),
    )
    assert verified.status == "ok"
    assert "abc123def456" in verified.detail

    drifted = verify_variant(
        "supervised",
        reference,
        manifest=_MANIFEST,
        inspector=_FakeInspector(lock_digest="f" * 64, revision="abc123def456"),
    )
    assert drifted.status == "drifted"
    assert "abc123def456" in drifted.detail
