from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PROFILE_ROOT = ROOT / "containers" / "posttrain-job-kinds" / "verl-py313"
MODULE_PATH = PROFILE_ROOT / "release_gate.py"
SPEC = importlib.util.spec_from_file_location("posttrain_verl_release_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_gate
SPEC.loader.exec_module(release_gate)


def test_candidate_definition_preserves_two_python_environments() -> None:
    profile = release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml")

    assert release_gate.validate_definition(profile) == ()
    assert profile.control_virtual_env == "/opt/posttrain/venv"
    assert profile.backend_virtual_env == "/opt/posttrain-verl"
    assert profile.control_python == "3.12"
    assert profile.backend_python == "3.13.12"
    assert profile.backend_projection_path == "/opt/posttrain-verl/projection"
    assert profile.backend_pythonpath_variable == "POSTTRAIN_VERL_PYTHONPATH"
    assert profile.backend_worker_module == "posttrain.train.backends.verl.worker"
    assert profile.control_environment_lock_path == (
        "locks/runtime.control.requirements.txt"
    )
    assert profile.backend_environment_lock_path == (
        "locks/runtime.backend.requirements.txt"
    )
    assert profile.worker_projection_packages == ("common", "data", "train")


def test_unpublished_candidate_fails_release_closed(tmp_path: Path) -> None:
    profile = release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml")

    blockers = release_gate.release_blockers(
        profile,
        lock_path=tmp_path / "uv.lock",
        source_checkout=None,
        verify_remote=False,
    )

    assert "profile release_status is not ready" in blockers
    assert "published CarbonTeq veRL fork_revision is missing" in blockers
    assert any("uv.lock is missing" in blocker for blocker in blockers)
    assert "a clean veRL source checkout is required for release" in blockers
    assert "release requires remote reachability verification" in blockers


def test_research_lock_is_rejected_as_kind_image_input() -> None:
    research_lock = ROOT.parent / "verl-upstream" / "runtime" / "turboquant-cu130" / "uv.lock"
    if not research_lock.is_file():
        pytest.skip("local veRL research lock is not available")
    candidate = release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml")
    profile = replace(
        candidate,
        release_status="ready",
        fork_revision=candidate.upstream_revision,
        dependency_lock_sha256=hashlib.sha256(research_lock.read_bytes()).hexdigest(),
    )

    errors = release_gate._validate_lock(profile, research_lock)

    assert "concrete environment package leaked into veRL kind lock: automation-bench" in errors
    assert "concrete environment package leaked into veRL kind lock: gsm8k-v1" in errors
    assert "local/editable source is forbidden in release lock: verl" in errors


def test_blocked_verl_variant_is_not_publishable_but_trl_variant_is_explicit() -> None:
    bake = (ROOT / "containers" / "posttrain-job-kinds" / "docker-bake.hcl").read_text()

    assert 'target "posttrain-kind-online-rl-trl-py312"' in bake
    assert 'target "posttrain-kind-online-rl-verl-py313"' not in bake
    assert 'target "posttrain-kind-online-rl" {' not in bake


def test_actual_job_definition_projects_worker_into_backend_environment() -> None:
    profile = release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml")
    dockerfile = (
        ROOT / "containers" / "posttrain-job" / "Dockerfile"
    )
    bake = ROOT / "containers" / "posttrain-job-kinds" / "docker-bake.hcl"
    assert release_gate.validate_repository_integration(
        profile,
        actual_job_dockerfile=dockerfile,
        kind_bake_file=bake,
    ) == ()
    contents = dockerfile.read_text(encoding="utf-8")

    assert 'POSTTRAIN_VERL_PYTHONPATH="/opt/posttrain-verl/projection"' in contents
    assert 'test -x "/opt/posttrain-verl/bin/python"' in contents
    assert '--python "/opt/posttrain-verl/bin/python"' in contents
    assert "for package in common data train" in contents
    assert (
        'source="sources/framework/packages/${package}/src/posttrain/${package}"'
        in contents
    )
    assert 'PYTHONPATH="${POSTTRAIN_VERL_PYTHONPATH}"' in contents
    assert '"/opt/posttrain-verl/bin/python" -s -c' in contents
    assert (
        "import posttrain.common, posttrain.data, posttrain.train, "
        "posttrain.train.backends.verl.worker"
    ) in contents


def test_ready_profile_requires_a_publication_target(tmp_path: Path) -> None:
    profile = replace(
        release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml"),
        release_status="ready",
    )
    bake = tmp_path / "docker-bake.hcl"
    bake.write_text('target "another-profile" {}\n', encoding="utf-8")

    errors = release_gate.validate_repository_integration(
        profile,
        actual_job_dockerfile=(
            ROOT / "containers" / "posttrain-job" / "Dockerfile"
        ),
        kind_bake_file=bake,
    )

    assert (
        "ready veRL profile is absent from the job-kind publication graph"
        in errors
    )


def test_ready_profile_requires_real_container_smoke_not_definition_strings(
    tmp_path: Path,
) -> None:
    profile = replace(
        release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml"),
        release_status="ready",
        fork_revision="a" * 40,
        dependency_lock_sha256="b" * 64,
    )

    blockers = release_gate.release_blockers(
        profile,
        lock_path=tmp_path / "missing.lock",
        source_checkout=None,
        verify_remote=False,
        execute_container_gate=False,
    )

    assert "release requires the real veRL Docker/Bake smoke gate" in blockers
