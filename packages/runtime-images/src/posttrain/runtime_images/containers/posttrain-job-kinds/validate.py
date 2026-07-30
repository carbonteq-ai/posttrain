"""Static validation for the framework-owned runtime image hierarchy."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

DEFINITIONS = Path(__file__).resolve().parents[2]
BASE = DEFINITIONS / "containers" / "posttrain-base"
KINDS = DEFINITIONS / "containers" / "posttrain-job-kinds"


def _workspace_root() -> Path:
    """Locate the framework checkout that owns the resolved workspace locks.

    These definitions ship inside an installed wheel, where no workspace lock
    exists. Profile validation compares pins against those locks, so it is a
    framework-repo check by construction and must fail loudly rather than
    silently skip when run from an installed distribution.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "uv.lock").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise AssertionError(
        "runtime profile validation requires a framework checkout with a resolved "
        "uv.lock; it cannot run against an installed distribution"
    )


ROOT = _workspace_root()

RUNTIME_VARIANTS = (
    "supervised",
    "online-rl-trl-py312",
    "online-rl-verl-py313",
    "eval",
    "serve",
    "transform",
)
FORBIDDEN_ENVIRONMENTS = (
    "alphabet-sort",
    "alphabet_sort",
    "automationbench",
    "code-golf",
    "code_golf",
    "gsm8k",
    "reverse-text",
    "reverse_text",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _locked_versions(path: Path) -> dict[str, set[str]]:
    document = tomllib.loads(path.read_text())
    versions: dict[str, set[str]] = {}
    for package in document["package"]:
        versions.setdefault(package["name"].lower(), set()).add(package["version"])
    return versions


def _validate_profiles() -> None:
    workspace_versions = _locked_versions(ROOT / "uv.lock")
    transform_versions = _locked_versions(ROOT / "tools" / "quantization" / "uv.lock")
    version_pin = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;]+)")
    full_git_revision = re.compile(r"git\+https://[^@\s]+@[0-9a-f]{40}(?:#\S+)?$")

    for profile in sorted((KINDS / "profiles").glob("*.txt")):
        expected_versions = transform_versions if profile.name == "transform.txt" else workspace_versions
        for line in profile.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "-r ")):
                continue
            if "git+https://" in stripped:
                _require(bool(full_git_revision.search(stripped)), f"{profile}: Git source must use a full commit")
                continue
            match = version_pin.match(stripped)
            if match is None:
                raise AssertionError(f"{profile}: direct dependency must be exactly pinned: {stripped}")
            name, version = match.groups()
            locked = expected_versions.get(name.lower(), set())
            _require(version in locked, f"{profile}: {name}=={version} is not present in its authoritative uv.lock")

    build_tools = (KINDS / "locks" / "build-tools.lock.txt").read_text()
    _require(
        "hatchling==1.27.0 \\\n" in build_tools,
        "build-tools lock must pin Hatchling",
    )
    logical_requirements = re.split(r"\n(?=[A-Za-z0-9_.-]+==)", build_tools)
    for requirement in logical_requirements:
        if "==" not in requirement:
            continue
        _require(
            "--hash=sha256:" in requirement,
            "every build-tools requirement must be hash locked",
        )


def _validate_boundaries() -> None:
    inspected = [
        KINDS / "Dockerfile",
        *(KINDS / "profiles").glob("*.txt"),
        *(KINDS / "locks").glob("*.txt"),
    ]
    combined = "\n".join(path.read_text().lower() for path in inspected)
    for forbidden in FORBIDDEN_ENVIRONMENTS:
        _require(forbidden not in combined, f"concrete environment leaked into base/kind inputs: {forbidden}")

    base_dockerfile = (BASE / "Dockerfile").read_text()
    _require("torch==2.11.0+cu130" in base_dockerfile, "universal base must install locked CUDA PyTorch")
    for backend in ("trl", "vllm", "verifiers", "llmcompressor"):
        _require(
            f"import {backend}" not in base_dockerfile and f'"{backend}' not in base_dockerfile,
            f"universal base must not install or import {backend}",
        )

    kind_dockerfile = (KINDS / "Dockerfile").read_text()
    _require("ARG POSTTRAIN_BASE_IMAGE" in kind_dockerfile, "kind images need an explicit parent image")
    _require(
        "locks/build-tools.lock.txt" in kind_dockerfile and "--require-hashes" in kind_dockerfile,
        "kind images must install the hash-locked source build backend",
    )
    _require(
        "FROM kind-common AS vllm-kind-common" in kind_dockerfile
        and "apt-get install --yes --no-install-recommends g++" in kind_dockerfile,
        "vLLM kind images need a host C++ compiler for CUDA JIT extensions",
    )
    for profile in (
        "online-rl-trl-py312-dependencies",
        "eval-dependencies",
        "serve-dependencies",
    ):
        _require(
            f"FROM vllm-kind-common AS {profile}" in kind_dockerfile,
            f"{profile} must inherit the CUDA JIT host toolchain",
        )
    for forbidden in (
        "COPY packages/",
        "COPY apps/",
        "posttrain-runtime",
        "from posttrain.",
    ):
        _require(
            forbidden not in kind_dockerfile,
            f"framework code leaked into dependency-only kind image: {forbidden}",
        )
    for variant in RUNTIME_VARIANTS:
        variant_dockerfile = (
            (KINDS / "verl-py313" / "Dockerfile").read_text() if variant == "online-rl-verl-py313" else kind_dockerfile
        )
        _require(f" AS {variant}\n" in variant_dockerfile, f"missing runtime stage for {variant}")
        _require(f" AS {variant}-smoke\n" in variant_dockerfile, f"missing smoke stage for {variant}")
    _require(
        'ENV POSTTRAIN_QUANTIZATION_PYTHON="/opt/posttrain/venv/bin/python"' in kind_dockerfile,
        "transform image must select its locked quantization interpreter",
    )

    verl_profile = tomllib.loads((KINDS / "verl-py313" / "profile.toml").read_text())
    _require(
        verl_profile.get("profile_id") == "online-rl-verl-py313",
        "veRL candidate must use the exact runtime-variant id",
    )
    _require(
        verl_profile.get("control_virtual_env") == "/opt/posttrain/venv"
        and verl_profile.get("backend_virtual_env") == "/opt/posttrain-verl",
        "veRL candidate must preserve separate control and backend environments",
    )
    _require(
        verl_profile.get("control_python") == "3.13.12" and verl_profile.get("backend_python") == "3.13.12",
        "veRL control and backend must both run the qualified interpreter",
    )


def _validate_bake() -> None:
    for bake in (BASE / "docker-bake.hcl", KINDS / "docker-bake.hcl"):
        contents = bake.read_text()
        for required in (
            "compression=zstd",
            "compression-level=1",
            "force-compression=false",
            "oci-mediatypes=true",
        ):
            _require(required in contents, f"{bake}: missing {required}")
        for forbidden in (
            "type=provenance,mode=max",
            "type=sbom",
            "force-compression=true",
        ):
            _require(forbidden not in contents, f"{bake}: unexpected {forbidden}")

    kind_bake = (KINDS / "docker-bake.hcl").read_text()
    for variant in RUNTIME_VARIANTS:
        _require(
            f'target "posttrain-kind-{variant}"' in kind_bake,
            f"missing Bake publication target for {variant}",
        )
        _require(
            f'target "posttrain-kind-{variant}-smoke"' in kind_bake,
            f"missing Bake smoke target for {variant}",
        )

    verl_profile = tomllib.loads((KINDS / "verl-py313" / "profile.toml").read_text())
    if verl_profile.get("release_status") != "ready":
        _require(
            'target "posttrain-kind-online-rl-verl-py313"' not in kind_bake,
            "blocked veRL candidate must not appear in the publication graph",
        )
    _require(
        'target "posttrain-kind-online-rl" {' not in kind_bake,
        "generic online-RL target would hide backend/Python incompatibility",
    )


def main() -> None:
    _validate_profiles()
    _validate_boundaries()
    _validate_bake()
    print("framework image hierarchy: static validation passed")


if __name__ == "__main__":
    main()
