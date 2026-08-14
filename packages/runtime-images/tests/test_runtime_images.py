from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

import pytest
from posttrain.runtime_images import (
    BASE_BAKE_FILE,
    BASE_LOCK,
    JOB_BAKE_FILE,
    KIND_BAKE_FILE,
    RUNTIME_VARIANTS,
    TRANSFORM_LOCK,
    VERL_BACKEND_LOCK,
    backend_constraint_lock,
    constraint_lock,
    definition_root,
    lock_digest,
    read_lock,
)

_WORKSPACE_LOCK = PurePosixPath("containers/posttrain-job-kinds/locks/workspace.lock.txt")


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


def test_actual_job_from_arguments_are_declared_in_global_scope() -> None:
    """Every variable expanded by FROM must precede the first build stage."""

    with definition_root() as root:
        dockerfile = (root / "containers/posttrain-job/Dockerfile").read_text()

    global_scope = dockerfile.split("\nFROM ", 1)[0]
    assert "ARG JOB_CONTEXT_REF" in global_scope
    assert "ARG POSTTRAIN_KIND_IMAGE" in global_scope


def test_eval_kind_installs_one_locked_runtime_and_marks_it_preinstalled() -> None:
    with definition_root() as root:
        dockerfile = (root / "containers/posttrain-job-kinds/Dockerfile").read_text()
    assert "null_harness_warmup.py" not in dockerfile
    assert "uv sync --script" not in dockerfile
    assert "--constraint /opt/posttrain/locks/eval.lock.txt" in dockerfile
    assert 'POSTTRAIN_VERIFIERS_PREINSTALLED="1"' in dockerfile


def test_base_accepts_a_build_secret_ca_bundle_without_disabling_tls() -> None:
    with definition_root() as root:
        dockerfile = (root / "containers/posttrain-base/Dockerfile").read_text()

    assert "--mount=type=secret,id=posttrain_ca_bundle,required=false" in dockerfile
    assert "cat /run/secrets/posttrain_ca_bundle >>" in dockerfile
    assert "/etc/ssl/certs/ca-certificates.crt" in dockerfile
    assert "install -m 0644 /run/secrets/posttrain_ca_bundle" not in dockerfile
    assert "trusted-host" not in dockerfile
    assert "allow-insecure" not in dockerfile


def test_base_package_cache_is_scoped_to_the_immutable_lock() -> None:
    with definition_root() as root:
        dockerfile = (root / "containers/posttrain-base/Dockerfile").read_text()

    assert "ARG LOCK_DIGEST" in dockerfile
    assert "id=posttrain-base-uv-${LOCK_DIGEST}" in dockerfile


def test_runtime_variants_match_the_published_bake_targets() -> None:
    published = re.compile(r'^target "posttrain-kind-([a-z0-9-]+)" \{', re.MULTILINE)
    with definition_root() as root:
        targets = published.findall((root / KIND_BAKE_FILE).read_text())
    assert set(RUNTIME_VARIANTS) == {t for t in targets if not t.endswith("-smoke")}


@pytest.mark.parametrize("variant", RUNTIME_VARIANTS)
def test_every_variant_names_a_shipped_constraint_lock(variant: str) -> None:
    lock = constraint_lock(variant)
    assert read_lock(lock), f"{variant} constraint lock is empty"


def test_every_kind_has_its_own_runtime_closure() -> None:
    locks = {variant: constraint_lock(variant) for variant in RUNTIME_VARIANTS}
    assert len(set(locks.values())) == len(RUNTIME_VARIANTS)
    assert locks["supervised"].name == "supervised.lock.txt"
    assert locks["online-rl-trl-py312"].name == "online-rl-trl-py312.lock.txt"
    assert locks["online-rl-verl-py313"].name == "online-rl-verl-py313.lock.txt"
    assert locks["eval"].name == "eval.lock.txt"
    assert locks["serve"].name == "serve.lock.txt"
    assert constraint_lock("transform") == TRANSFORM_LOCK


def test_verl_is_the_only_variant_with_separate_backend_constraints() -> None:
    selected = {variant: backend_constraint_lock(variant) for variant in RUNTIME_VARIANTS}
    assert selected == {
        "supervised": None,
        "online-rl-trl-py312": None,
        "online-rl-verl-py313": VERL_BACKEND_LOCK,
        "eval": None,
        "serve": None,
        "transform": None,
    }
    assert read_lock(VERL_BACKEND_LOCK)


def test_constraint_lock_rejects_unknown_variants() -> None:
    with pytest.raises(ValueError, match="unknown runtime variant"):
        constraint_lock("unknown-runtime")


def test_lock_digest_is_computed_from_the_shipped_bytes() -> None:
    assert lock_digest() == hashlib.sha256(read_lock(BASE_LOCK)).hexdigest()
    assert lock_digest() == lock_digest(BASE_LOCK)
    assert lock_digest(TRANSFORM_LOCK) != lock_digest(BASE_LOCK)


def test_every_shipped_lock_has_a_distinct_stable_digest() -> None:
    """Identity comes from the bytes, never from a value restated in a test.

    An earlier version of this test asserted one hardcoded digest. That is the
    same hand transcription this package exists to remove: it has to be edited
    by a human every time a lock legitimately changes, and a wrong edit makes
    it agree with nothing. Whether a published image still matches its lock is
    checked against the manifest in test_manifest.py, from the shipped bytes.
    """
    locks = {BASE_LOCK, *(constraint_lock(variant) for variant in RUNTIME_VARIANTS)}
    digests = {lock_digest(lock) for lock in locks}
    assert len(digests) == len(locks)
    assert all(len(d) == 64 and d == d.lower() for d in digests)
    assert lock_digest(BASE_LOCK) == lock_digest(BASE_LOCK)


def _logical_requirements(lock: PurePosixPath) -> dict[str, str]:
    """Read pip-compile blocks without treating comments as requirements."""

    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+(?:\s+@|==))", read_lock(lock).decode("utf-8"))
    requirements: dict[str, str] = {}
    for block in blocks:
        first = block.lstrip().splitlines()[0] if block.strip() else ""
        match = re.match(r"(?P<name>[A-Za-z0-9_.-]+)(?:\s+@|==)", first)
        if match is not None:
            requirements[match.group("name").lower().replace("_", "-")] = block
    return requirements


def test_narrow_runtime_locks_pin_every_profile_root_and_artifact() -> None:
    """Each image closure is independently installable and immutable.

    Git requirements cannot participate in pip's hash mode, so their full
    commit is the immutable receipt. Every wheel/sdist requirement must carry
    a SHA-256 hash selected from the authoritative workspace resolution.
    """

    expected_roots = {
        BASE_LOCK: {"torch", "triton"},
        constraint_lock("supervised"): {"carbonteq-trackio", "pydantic", "pyyaml", "trl"},
        constraint_lock("online-rl-trl-py312"): {"carbonteq-trackio", "trl", "vllm", "verifiers"},
        constraint_lock("online-rl-verl-py313"): {"carbonteq-trackio", "verifiers"},
        constraint_lock("eval"): {"carbonteq-trackio", "datasets", "vllm", "verifiers"},
        constraint_lock("serve"): {"carbonteq-trackio", "vllm", "torchvision", "torchaudio"},
        TRANSFORM_LOCK: {"datasets", "llmcompressor", "torch", "transformers"},
    }
    for lock, roots in expected_roots.items():
        requirements = _logical_requirements(lock)
        assert roots <= set(requirements), f"{lock} omits profile root(s) {sorted(roots - set(requirements))}"
        for name, requirement in requirements.items():
            if "git+https://" in requirement:
                assert re.search(r"git\+https://[^@\s]+@[0-9a-f]{40}(?:\s|$)", requirement), (lock, name)
            else:
                assert "--hash=sha256:" in requirement or "#sha256=" in requirement, (lock, name)


def test_base_runtime_lock_includes_cuda_dependencies_selected_by_torch_extras() -> None:
    """Torch's locked CUDA toolkit edge requests the cudart extra.

    The base lock is installed with ``--require-hashes`` and therefore cannot
    rely on the package installer to resolve an omitted optional dependency at
    image-build time.
    """

    requirements = _logical_requirements(BASE_LOCK)
    assert "cuda-toolkit" in requirements
    assert "nvidia-cuda-runtime" in requirements


def test_base_runtime_lock_retains_the_reviewed_mirrored_triton_artifact() -> None:
    requirements = _logical_requirements(BASE_LOCK)
    assert requirements["triton"].startswith("triton @ https://pypi.lan/root/pypi/+f/10c/7f76c6e72d2ef/")
    assert "#sha256=10c7f76c6e72d2ef08df639e3d0d30729112f47a56b0c81672edc05ee5116ac9" in requirements["triton"]


def test_narrow_runtime_locks_only_select_artifacts_from_the_workspace_resolution() -> None:
    """Per-kind compilation cannot silently resolve a different dependency.

    The base and non-transform kind locks project the framework workspace
    resolution. Transform remains intentionally separate because its profile
    is governed by ``tools/quantization/uv.lock``. The selected requirement
    identity and every retained artifact hash must nevertheless be present in
    the workspace resolution that the release lock names as authoritative.
    """

    workspace = _logical_requirement_blocks(_WORKSPACE_LOCK)
    narrow_locks = (
        BASE_LOCK,
        *(constraint_lock(variant) for variant in RUNTIME_VARIANTS if variant != "transform"),
    )
    for lock in narrow_locks:
        for name, requirement in _logical_requirements(lock).items():
            candidates = workspace.get(name, ())
            assert candidates, f"{lock} resolves {name} outside the workspace lock"
            observed = _requirement_hashes(requirement)
            if " @ " in _requirement_identity(requirement):
                workspace_requirement = (
                    next((item for item in candidates if observed & _requirement_hashes(item)), None)
                    if observed
                    else next(
                        (
                            item
                            for item in candidates
                            if _requirement_identity(requirement) == _requirement_identity(item)
                        ),
                        None,
                    )
                )
            else:
                workspace_requirement = next(
                    (item for item in candidates if _requirement_identity(requirement) == _requirement_identity(item)),
                    None,
                )
            assert workspace_requirement is not None, (lock, name)
            assert _requirement_hashes(requirement) <= _requirement_hashes(workspace_requirement), (lock, name)


def _requirement_identity(requirement: str) -> str:
    """Return the version/direct-reference line without comments or hashes."""

    return requirement.lstrip().splitlines()[0].rstrip(" \\").split(" ;", 1)[0]


def _requirement_hashes(requirement: str) -> set[str]:
    return set(re.findall(r"--hash=sha256:([0-9a-f]{64})", requirement)) | set(
        re.findall(r"#sha256=([0-9a-f]{64})", requirement)
    )


def _logical_requirement_blocks(lock: PurePosixPath) -> dict[str, tuple[str, ...]]:
    """Return every marker-specific workspace block for each package name."""

    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+(?:\s+@|==))", read_lock(lock).decode("utf-8"))
    requirements: dict[str, list[str]] = {}
    for block in blocks:
        first = block.lstrip().splitlines()[0] if block.strip() else ""
        match = re.match(r"(?P<name>[A-Za-z0-9_.-]+)(?:\s+@|==)", first)
        if match is not None:
            requirements.setdefault(match.group("name").lower().replace("_", "-"), []).append(block)
    return {name: tuple(values) for name, values in requirements.items()}
