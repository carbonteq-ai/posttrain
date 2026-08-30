from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "posttrain" / "runtime_images"
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
    assert profile.control_python == "3.13.12"
    assert profile.backend_python == "3.13.12"
    assert profile.backend_projection_path == "/opt/posttrain-verl/projection"
    assert profile.backend_pythonpath_variable == "POSTTRAIN_VERL_PYTHONPATH"
    assert profile.backend_worker_module == "posttrain.train.backends.verl.worker"
    assert profile.control_environment_lock_path == ("locks/runtime.control.requirements.txt")
    assert profile.backend_environment_lock_path == ("locks/runtime.backend.requirements.txt")
    assert (
        profile.backend_constraints_sha256
        == hashlib.sha256((PROFILE_ROOT / "release" / "backend-constraints.txt").read_bytes()).hexdigest()
    )
    assert profile.worker_projection_packages == ("common", "data", "environment", "train")


def test_candidate_image_smokes_both_python_313_environments() -> None:
    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "import ray, torch, transformers, tensordict, verl, verifiers, vllm" in dockerfile
    assert "0.25.2.dev2+g7817d8457.precompiled" in dockerfile
    assert 'VLLM_VERSION_OVERRIDE="${VLLM_RUNTIME_VERSION}"' in dockerfile
    assert 'CUDA_HOME="/opt/posttrain-verl/lib/python3.13/site-packages/nvidia/cu13"' in dockerfile
    assert 'CPATH="/opt/posttrain/venv/lib/python3.13/site-packages/nvidia/cu13/include"' in dockerfile
    assert 'LIBRARY_PATH="/opt/posttrain/venv/lib/python3.13/site-packages/nvidia/cu13/lib"' in dockerfile
    assert "apt-get install --yes --no-install-recommends curl g++" in dockerfile
    assert 'ln -s lib "${CUDA_HOME}/lib64"' in dockerfile
    assert 'ln -s libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"' in dockerfile
    assert "ln -s /opt/posttrain-verl/bin/ninja /usr/local/bin/ninja" in dockerfile
    assert 'ln -s "${CUDA_HOME}/bin/nvcc" /usr/local/bin/nvcc' in dockerfile
    assert "command -v g++ >/dev/null" in dockerfile
    assert (
        '"import sys, hatchling, pydantic, yaml, verifiers; assert sys.version_info[:3] == (3, 13, 12)"'
    ) in dockerfile


def test_candidate_carries_pinned_cuda_compat_without_globally_activating_it() -> None:
    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    source_clone = dockerfile.index("RUN git clone")
    compat_copy = dockerfile.index("RUN --mount=from=cuda-compat")

    assert (
        "nvidia/cuda:13.0.2-base-ubuntu24.04"
        "@sha256:605fb0c8acf8674e164d822da8a8521f3a655056e569f0899e72ae940e1fe7dc"
    ) in dockerfile
    assert source_clone < compat_copy
    assert '"compat_path":"/usr/local/cuda-13.0/compat"' in dockerfile
    assert '"runtime_api_version":13000' in dockerfile
    assert '"schema_version":1' in dockerfile
    assert "payload_digest" in dockerfile
    assert dockerfile.count("/opt/posttrain/runtime/cuda-compat.json") == 2
    assert 'ENV LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat"' not in dockerfile


def test_candidate_keeps_release_labels_out_of_filesystem_cache_keys() -> None:
    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compat_layer = dockerfile.index("RUN --mount=from=cuda-compat")

    assert dockerfile.index("ARG RELEASE_CREATED", compat_layer) > compat_layer
    assert dockerfile.index("ARG RELEASE_SOURCE_REVISION", compat_layer) > compat_layer
    assert dockerfile.index("ARG RELEASE_VERSION", compat_layer) > compat_layer
    assert 'org.opencontainers.image.created="${RELEASE_CREATED}"' in dockerfile
    assert 'org.opencontainers.image.revision="${RELEASE_SOURCE_REVISION}"' in dockerfile


def test_candidate_uses_uv_partial_sync_with_a_validated_control_fallback() -> None:
    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    sync_position = dockerfile.index("UV_COMPILE_BYTECODE=0 uv sync ")
    sharing_position = dockerfile.index("validate_shared_fallback.py", sync_position)
    cleanup_position = dockerfile.index("rm -rf /opt/posttrain-verl-build", sharing_position)

    assert "\nRUN " not in dockerfile[sync_position:cleanup_position]
    assert "--control-site" in dockerfile[sharing_position:cleanup_position]
    assert "--backend-site" in dockerfile[sharing_position:cleanup_position]
    assert "--backend-lock" in dockerfile[sharing_position:cleanup_position]
    assert "--fallback-file" in dockerfile[sharing_position:cleanup_position]
    assert "--no-install-package torch" in dockerfile[sync_position:sharing_position]
    assert "--no-install-package triton" in dockerfile[sync_position:sharing_position]
    assert "--no-install-package nvidia-cuda-runtime" not in dockerfile[sync_position:sharing_position]
    assert "posttrain-control-fallback.pth" in dockerfile[sync_position:cleanup_position]
    assert "/opt/posttrain-verl/release/shared-heavy-report.json" in dockerfile
    assert "/opt/posttrain-verl/release/shared-heavy.toml" in dockerfile


def test_candidate_normalizes_generated_timestamps_without_rewriting_its_parent() -> None:
    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG SOURCE_DATE_EPOCH" in dockerfile
    assert 'SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}"' in dockerfile
    assert 'PYTHONHASHSEED="0"' in dockerfile
    assert 'UV_LINK_MODE="copy"' in dockerfile
    assert "UV_COMPILE_BYTECODE=0 uv sync" in dockerfile
    assert "UV_COMPILE_BYTECODE=0 uv python install" in dockerfile
    assert 'sysconfig.get_path("stdlib")' in dockerfile
    assert dockerfile.count("compileall --invalidation-mode checked-hash") == 2
    assert "find /tmp -maxdepth 1 -type f -name 'uv-*.lock' -delete" in dockerfile
    assert "/var/cache/ldconfig/aux-cache" in dockerfile
    assert "/var/log/dpkg.log" in dockerfile
    assert ".posttrain-source-revision" in dockerfile
    assert "rm -rf /opt/posttrain-verl/workdir/.git" in dockerfile


def test_candidate_orders_overlapping_cutlass_wheels_before_compilation() -> None:
    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    base_install = dockerfile.index(
        "UV_COMPILE_BYTECODE=0 uv pip install --python /opt/posttrain-verl --no-deps --reinstall"
    )
    cu13_install = dockerfile.index(
        "UV_COMPILE_BYTECODE=0 uv pip install --python /opt/posttrain-verl --no-deps --reinstall",
        base_install + 1,
    )
    compile_position = dockerfile.index("BACKEND_STDLIB=", cu13_install)

    assert "CUTLASS_BASE_WHEEL_SHA256" in dockerfile
    assert "CUTLASS_CU13_WHEEL_SHA256" in dockerfile
    assert "nvidia_cutlass_dsl_libs_base" in dockerfile[base_install:cu13_install]
    assert "nvidia_cutlass_dsl_libs_cu13" in dockerfile[cu13_install:compile_position]
    assert "sed -i '/uv_cache\\.json,/d'" in dockerfile[cu13_install:compile_position]
    assert "-type f -name uv_cache.json -delete" in dockerfile[cu13_install:compile_position]
    assert base_install < cu13_install < compile_position


def test_candidate_resolves_unnamespaced_build_backend_collision_before_compilation() -> None:
    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    restore_position = dockerfile.index("names=[name for name in archive.namelist()")
    compile_position = dockerfile.index("BACKEND_STDLIB=", restore_position)

    assert "TORCH_C_DLPACK_WHEEL_SHA256" in dockerfile
    assert "torch_c_dlpack_ext-0.1.5-cp313-cp313-manylinux_2_28_x86_64.whl" in dockerfile
    assert "names == ['build_backend.py']" in dockerfile[restore_position:compile_position]
    assert restore_position < compile_position


def test_ready_profile_still_fails_closed_without_release_inputs(tmp_path: Path) -> None:
    profile = release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml")

    blockers = release_gate.release_blockers(
        profile,
        lock_path=tmp_path / "uv.lock",
        source_checkout=None,
        verify_remote=False,
    )

    assert "profile release_status is not ready" not in blockers
    assert any("uv.lock is missing" in blocker for blocker in blockers)
    assert "a clean veRL source checkout is required for release" in blockers
    assert "release requires remote reachability verification" in blockers
    # The immutable fork and lock are recorded, but the release command still
    # requires a clean checkout, remote verification, and real container smoke.
    assert profile.fork_revision
    assert profile.dependency_lock_sha256
    assert (PROFILE_ROOT / "release" / "uv.lock").is_file()


def test_backend_constraints_are_an_exact_export_of_the_release_lock() -> None:
    errors = release_gate._validate_backend_constraints(
        lock_path=PROFILE_ROOT / "release" / "uv.lock",
        constraints_path=PROFILE_ROOT / "release" / "backend-constraints.txt",
    )

    assert errors == ()
    constraints = (PROFILE_ROOT / "release" / "backend-constraints.txt").read_text(encoding="utf-8")
    assert "antlr4-python3-runtime==4.9.3" in constraints
    assert "omegaconf==2.3.1" in constraints


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


def test_ready_verl_and_trl_variants_are_explicitly_publishable() -> None:
    bake = (ROOT / "containers" / "posttrain-job-kinds" / "docker-bake.hcl").read_text()

    assert 'target "posttrain-kind-online-rl-trl-py312"' in bake
    assert 'target "posttrain-kind-online-rl-verl-py313"' in bake
    assert 'target "posttrain-kind-online-rl" {' not in bake
    for argument in ("CREATED", "LOCK_DIGEST", "SOURCE_REVISION", "VERSION"):
        assert f"{argument} = {argument}" in bake

    dockerfile = (PROFILE_ROOT / "Dockerfile").read_text()
    assert 'org.opencontainers.image.revision="${RELEASE_SOURCE_REVISION}"' in dockerfile
    assert 'org.opencontainers.image.version="${RELEASE_VERSION}"' in dockerfile
    assert 'org.carbonteq.posttrain.lock-digest="${LOCK_DIGEST}"' in dockerfile


def test_actual_job_definition_projects_worker_into_backend_environment() -> None:
    profile = release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml")
    dockerfile = ROOT / "containers" / "posttrain-job" / "Dockerfile"
    bake = ROOT / "containers" / "posttrain-job-kinds" / "docker-bake.hcl"
    assert (
        release_gate.validate_repository_integration(
            profile,
            actual_job_dockerfile=dockerfile,
            kind_bake_file=bake,
        )
        == ()
    )
    contents = dockerfile.read_text(encoding="utf-8")

    assert 'POSTTRAIN_VERL_PYTHONPATH="/opt/posttrain-verl/projection"' in contents
    assert 'test -x "/opt/posttrain-verl/bin/python"' in contents
    assert '--python "/opt/posttrain-verl/bin/python"' in contents
    assert "for package in common data environment train" in contents
    assert 'source="sources/framework/packages/${package}/src/posttrain/${package}"' in contents
    assert 'PYTHONPATH="${POSTTRAIN_VERL_PYTHONPATH}"' in contents
    assert '"/opt/posttrain-verl/bin/python" -s -B -c' in contents
    assert (
        "import posttrain.common, posttrain.data, posttrain.train, posttrain.train.backends.verl.worker"
    ) in contents
    assert 'revision_marker = worktree / ".posttrain-source-revision"' in contents
    assert "veRL immutable source snapshot unexpectedly retains Git metadata" in contents
    assert "revision_marker.read_text" in contents
    assert "Compatibility for already-published kind images" in contents


def test_ready_profile_requires_a_publication_target(tmp_path: Path) -> None:
    profile = replace(
        release_gate.ReleaseProfile.read(PROFILE_ROOT / "profile.toml"),
        release_status="ready",
    )
    bake = tmp_path / "docker-bake.hcl"
    bake.write_text('target "another-profile" {}\n', encoding="utf-8")

    errors = release_gate.validate_repository_integration(
        profile,
        actual_job_dockerfile=(ROOT / "containers" / "posttrain-job" / "Dockerfile"),
        kind_bake_file=bake,
    )

    assert "ready veRL profile is absent from the job-kind publication graph" in errors


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
