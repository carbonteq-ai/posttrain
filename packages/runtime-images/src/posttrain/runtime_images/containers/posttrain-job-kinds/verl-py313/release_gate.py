"""Fail-closed release gate for the isolated veRL Python 3.13 profile."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_PACKAGES = frozenset(
    {
        "alphabet-sort-v1",
        "automation-bench",
        "automationbench",
        "automationbench-v1",
        "code-golf-v1",
        "gsm8k-v1",
        "reverse-text-v1",
    }
)
REQUIRED_PROFILE = "online-rl-verl-py313"
REQUIRED_JOB_KIND = "online-rl"
REQUIRED_CONTROL_PYTHON = "3.13.12"
REQUIRED_BACKEND_PYTHON = "3.13.12"
REQUIRED_CONTROL_VENV = "/opt/posttrain/venv"
REQUIRED_BACKEND_VENV = "/opt/posttrain-verl"
REQUIRED_BACKEND_PROJECTION = "/opt/posttrain-verl/projection"
REQUIRED_BACKEND_PYTHONPATH_VARIABLE = "POSTTRAIN_VERL_PYTHONPATH"
REQUIRED_BACKEND_WORKER_MODULE = "posttrain.train.backends.verl.worker"
REQUIRED_PROJECTION_PACKAGES = ("common", "data", "train")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTUAL_JOB_DOCKERFILE = ROOT / "posttrain-job" / "Dockerfile"
DEFAULT_KIND_BAKE_FILE = ROOT / "posttrain-job-kinds" / "docker-bake.hcl"
DEFAULT_VERL_DOCKERFILE = Path(__file__).with_name("Dockerfile")
DEFAULT_VERL_BAKE_FILE = Path(__file__).with_name("docker-bake.hcl")
DEFAULT_RELEASE_PROJECT = Path(__file__).with_name("release") / "pyproject.toml"
DEFAULT_RELEASE_LOCK = Path(__file__).with_name("release") / "uv.lock"


@dataclass(frozen=True)
class ReleaseProfile:
    path: Path
    schema_version: int
    profile_id: str
    job_kind: str
    release_status: str
    control_python: str
    backend_python: str
    control_virtual_env: str
    backend_virtual_env: str
    backend_working_directory: str
    backend_projection_path: str
    backend_pythonpath_variable: str
    backend_worker_module: str
    control_environment_lock_path: str
    backend_environment_lock_path: str
    backend_dependency_lock_path: str
    source_repository: str
    upstream_revision: str
    fork_revision: str
    dependency_lock_sha256: str
    dependencies: dict[str, str]
    worker_projection_packages: tuple[str, ...]

    @classmethod
    def read(cls, path: Path) -> ReleaseProfile:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        dependencies = payload.get("dependencies")
        if not isinstance(dependencies, dict):
            raise ValueError("veRL profile dependencies must be a TOML table")
        worker_projection = payload.get("worker_projection")
        if not isinstance(worker_projection, dict):
            raise ValueError("veRL profile worker_projection must be a TOML table")
        values = {str(key): str(value) for key, value in dependencies.items()}
        return cls(
            path=path,
            schema_version=_integer(payload, "schema_version"),
            profile_id=_string(payload, "profile_id"),
            job_kind=_string(payload, "job_kind"),
            release_status=_string(payload, "release_status"),
            control_python=_string(payload, "control_python"),
            backend_python=_string(payload, "backend_python"),
            control_virtual_env=_string(payload, "control_virtual_env"),
            backend_virtual_env=_string(payload, "backend_virtual_env"),
            backend_working_directory=_string(payload, "backend_working_directory"),
            backend_projection_path=_string(payload, "backend_projection_path"),
            backend_pythonpath_variable=_string(
                payload,
                "backend_pythonpath_variable",
            ),
            backend_worker_module=_string(payload, "backend_worker_module"),
            control_environment_lock_path=_string(
                payload,
                "control_environment_lock_path",
            ),
            backend_environment_lock_path=_string(
                payload,
                "backend_environment_lock_path",
            ),
            backend_dependency_lock_path=_string(
                payload,
                "backend_dependency_lock_path",
            ),
            source_repository=_string(payload, "source_repository"),
            upstream_revision=_string(payload, "upstream_revision"),
            fork_revision=_string(payload, "fork_revision"),
            dependency_lock_sha256=_string(payload, "dependency_lock_sha256"),
            dependencies=values,
            worker_projection_packages=_string_tuple(
                worker_projection,
                "packages",
            ),
        )


def validate_definition(profile: ReleaseProfile) -> tuple[str, ...]:
    errors: list[str] = []
    if profile.schema_version != 1:
        errors.append("veRL profile schema_version must be 1")
    if profile.release_status not in {"blocked", "ready"}:
        errors.append("veRL profile release_status must be blocked or ready")
    expected = {
        "profile_id": (profile.profile_id, REQUIRED_PROFILE),
        "job_kind": (profile.job_kind, REQUIRED_JOB_KIND),
        "control_python": (profile.control_python, REQUIRED_CONTROL_PYTHON),
        "backend_python": (profile.backend_python, REQUIRED_BACKEND_PYTHON),
        "control_virtual_env": (profile.control_virtual_env, REQUIRED_CONTROL_VENV),
        "backend_virtual_env": (profile.backend_virtual_env, REQUIRED_BACKEND_VENV),
        "backend_projection_path": (
            profile.backend_projection_path,
            REQUIRED_BACKEND_PROJECTION,
        ),
        "backend_pythonpath_variable": (
            profile.backend_pythonpath_variable,
            REQUIRED_BACKEND_PYTHONPATH_VARIABLE,
        ),
        "backend_worker_module": (
            profile.backend_worker_module,
            REQUIRED_BACKEND_WORKER_MODULE,
        ),
        "control_environment_lock_path": (
            profile.control_environment_lock_path,
            "locks/runtime.control.requirements.txt",
        ),
        "backend_environment_lock_path": (
            profile.backend_environment_lock_path,
            "locks/runtime.backend.requirements.txt",
        ),
        "backend_dependency_lock_path": (
            profile.backend_dependency_lock_path,
            "/opt/posttrain-verl/release/uv.lock",
        ),
    }
    for name, (actual, required) in expected.items():
        if actual != required:
            errors.append(f"{name} must be {required!r}, got {actual!r}")
    if profile.control_virtual_env == profile.backend_virtual_env:
        errors.append("control and veRL backend virtual environments must be separate")
    if not profile.backend_working_directory.startswith(f"{profile.backend_virtual_env}/"):
        errors.append("veRL working directory must remain below its backend virtual environment")
    if not profile.backend_projection_path.startswith(f"{profile.backend_virtual_env}/"):
        errors.append("veRL worker projection must remain below its backend virtual environment")
    if profile.backend_working_directory == profile.backend_projection_path:
        errors.append("veRL worktree and worker projection must be separate")
    if profile.worker_projection_packages != REQUIRED_PROJECTION_PACKAGES:
        errors.append("veRL worker projection packages must be common, data, and train")
    if not profile.source_repository.startswith("https://github.com/"):
        errors.append("veRL source repository must be a secret-free HTTPS GitHub URL")
    if FULL_REVISION.fullmatch(profile.upstream_revision) is None:
        errors.append("veRL upstream revision must be a full commit")
    required_versions = {
        "torch": "2.11.0+cu130",
        "transformers": "5.14.1",
        "vllm": "0.25.1",
        "ray": "2.56.1",
        "tensordict": "0.10.0",
    }
    for package, version in required_versions.items():
        if profile.dependencies.get(package) != version:
            errors.append(f"{package} must remain exactly {version}")
    verifiers_revision = profile.dependencies.get("verifiers_revision", "")
    if FULL_REVISION.fullmatch(verifiers_revision) is None:
        errors.append("Verifiers core must use a full source revision")
    return tuple(errors)


def release_blockers(
    profile: ReleaseProfile,
    *,
    lock_path: Path,
    source_checkout: Path | None,
    verify_remote: bool,
    actual_job_dockerfile: Path = DEFAULT_ACTUAL_JOB_DOCKERFILE,
    kind_bake_file: Path = DEFAULT_KIND_BAKE_FILE,
    execute_container_gate: bool = False,
) -> tuple[str, ...]:
    blockers = list(validate_definition(profile))
    blockers.extend(
        validate_repository_integration(
            profile,
            actual_job_dockerfile=actual_job_dockerfile,
            kind_bake_file=kind_bake_file,
        )
    )
    if profile.release_status != "ready":
        blockers.append("profile release_status is not ready")
    if FULL_REVISION.fullmatch(profile.fork_revision) is None:
        blockers.append("published CarbonTeq veRL fork_revision is missing")
    if re.fullmatch(r"[0-9a-f]{64}", profile.dependency_lock_sha256) is None:
        blockers.append("dependency_lock_sha256 is missing")
    if not lock_path.is_file():
        blockers.append(f"dependency-only uv.lock is missing: {lock_path}")
    else:
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if digest != profile.dependency_lock_sha256:
            blockers.append("dependency-only uv.lock digest differs from profile.toml")
        blockers.extend(_validate_lock(profile, lock_path))
    if source_checkout is None:
        blockers.append("a clean veRL source checkout is required for release")
    else:
        blockers.extend(_validate_checkout(profile, source_checkout, verify_remote=verify_remote))
    if not verify_remote:
        blockers.append("release requires remote reachability verification")
    if profile.release_status == "ready":
        if not execute_container_gate:
            blockers.append("release requires the real veRL Docker/Bake smoke gate")
        else:
            blockers.extend(
                run_container_gate(
                    profile,
                    lock_path=lock_path,
                )
            )
    return tuple(dict.fromkeys(blockers))


def validate_repository_integration(
    profile: ReleaseProfile,
    *,
    actual_job_dockerfile: Path,
    kind_bake_file: Path,
) -> tuple[str, ...]:
    """Bind release metadata to the dormant kind and actual-job definitions."""

    errors: list[str] = []
    for label, path in (
        ("veRL kind Dockerfile", DEFAULT_VERL_DOCKERFILE),
        ("veRL kind Bake file", DEFAULT_VERL_BAKE_FILE),
    ):
        if not path.is_file():
            errors.append(f"{label} is missing: {path}")
    if profile.release_status == "ready" and not DEFAULT_RELEASE_PROJECT.is_file():
        errors.append(f"ready veRL profile lacks dependency-only pyproject: {DEFAULT_RELEASE_PROJECT}")
    if not actual_job_dockerfile.is_file():
        errors.append(f"actual-job Dockerfile is missing: {actual_job_dockerfile}")
    else:
        dockerfile = actual_job_dockerfile.read_text(encoding="utf-8")
        required_fragments = {
            "runtime variant": f'"{profile.profile_id}"',
            "backend interpreter": (f'"{profile.backend_virtual_env}/bin/python"'),
            "worker projection path": (f'{profile.backend_pythonpath_variable}="{profile.backend_projection_path}"'),
            "worker projection environment": (f'PYTHONPATH="${{{profile.backend_pythonpath_variable}}}"'),
            "worker module": profile.backend_worker_module,
            "control Python 3.12 closure": "locks/runtime.control.requirements.txt",
            "backend Python 3.13 closure": "locks/runtime.backend.requirements.txt",
            "module-origin isolation": "PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1",
        }
        for label, fragment in required_fragments.items():
            if fragment not in dockerfile:
                errors.append(f"actual-job Dockerfile omits veRL {label}: {fragment}")
        package_loop = "for package in " + " ".join(profile.worker_projection_packages)
        if package_loop not in dockerfile:
            errors.append("actual-job Dockerfile differs from the veRL worker projection")
        backend_install = f'--python "{profile.backend_virtual_env}/bin/python"'
        if backend_install not in dockerfile:
            errors.append(
                "actual-job Dockerfile does not install selected runtime wheels into the veRL backend environment"
            )

    target = f'target "posttrain-kind-{profile.profile_id}"'
    if not kind_bake_file.is_file():
        errors.append(f"job-kind Bake file is missing: {kind_bake_file}")
    else:
        bake = kind_bake_file.read_text(encoding="utf-8")
        if profile.release_status == "ready" and target not in bake:
            errors.append("ready veRL profile is absent from the job-kind publication graph")
        if profile.release_status == "blocked" and target in bake:
            errors.append("blocked veRL profile is present in the job-kind publication graph")
    return tuple(errors)


def run_container_gate(
    profile: ReleaseProfile,
    *,
    lock_path: Path,
) -> tuple[str, ...]:
    """Execute the real BuildKit graph and import smoke for a ready profile."""

    if profile.release_status != "ready":
        return ("veRL container smoke cannot run for a blocked profile",)
    environment = {
        "DEPENDENCY_LOCK_SHA256": profile.dependency_lock_sha256,
        "FORK_REVISION": profile.fork_revision,
        "POSTTRAIN_BASE_IMAGE": os.environ.get("POSTTRAIN_BASE_IMAGE", ""),
        "SOURCE_REPOSITORY": profile.source_repository,
    }
    if not environment["POSTTRAIN_BASE_IMAGE"]:
        return ("POSTTRAIN_BASE_IMAGE is required for the veRL container smoke",)
    if lock_path.resolve() != DEFAULT_RELEASE_LOCK.resolve():
        return ("veRL container smoke requires the canonical release lock",)
    command = [
        "docker",
        "buildx",
        "bake",
        "--file",
        str(DEFAULT_VERL_BAKE_FILE),
        "posttrain-kind-online-rl-verl-py313-smoke",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT.parent,
            env={**os.environ, **environment},
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ("docker buildx is unavailable for the veRL container smoke",)
    if completed.returncode != 0:
        return ("real veRL Docker/Bake import smoke failed",)
    return ()


def _validate_lock(profile: ReleaseProfile, lock_path: Path) -> tuple[str, ...]:
    errors: list[str] = []
    payload = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = payload.get("package")
    if not isinstance(packages, list):
        return ("dependency lock has no package records",)
    by_name: dict[str, dict[str, Any]] = {}
    for raw in packages:
        if not isinstance(raw, dict):
            errors.append("dependency lock contains a malformed package record")
            continue
        name = str(raw.get("name", "")).lower()
        source = raw.get("source")
        if name in FORBIDDEN_PACKAGES:
            errors.append(f"concrete environment package leaked into veRL kind lock: {name}")
        if isinstance(source, dict):
            if "editable" in source or "directory" in source or "path" in source:
                errors.append(f"local/editable source is forbidden in release lock: {name}")
            source_text = " ".join(str(value).lower() for value in source.values())
            if "git" in source and name not in {"verifiers", "verl"}:
                errors.append(f"unexpected Git package in dependency-only kind lock: {name}")
            if "subdirectory=environments" in source_text or "subdirectory=environments%2f" in source_text:
                errors.append(f"concrete environment Git subdirectory leaked into kind lock: {name}")
        by_name[name] = raw
    for package, expected in (
        ("torch", profile.dependencies["torch"]),
        ("transformers", profile.dependencies["transformers"]),
        ("vllm", profile.dependencies["vllm"]),
        ("ray", profile.dependencies["ray"]),
        ("tensordict", profile.dependencies["tensordict"]),
    ):
        record = by_name.get(package)
        if record is None or str(record.get("version")) != expected:
            errors.append(f"dependency lock must contain {package}=={expected}")
    verl = by_name.get("verl")
    if verl is None:
        errors.append("dependency lock must contain veRL")
    else:
        source = verl.get("source")
        git = str(source.get("git", "")) if isinstance(source, dict) else ""
        if profile.source_repository not in git:
            errors.append("veRL lock source differs from the CarbonTeq repository")
        if profile.fork_revision not in git:
            errors.append("veRL lock source is not pinned to fork_revision")
    verifiers = by_name.get("verifiers")
    if verifiers is None:
        errors.append("dependency lock must contain Verifiers core")
    else:
        source = verifiers.get("source")
        git = str(source.get("git", "")) if isinstance(source, dict) else ""
        if profile.dependencies["verifiers_revision"] not in git:
            errors.append("Verifiers core is not pinned to the profile revision")
    return tuple(errors)


def _validate_checkout(
    profile: ReleaseProfile,
    checkout: Path,
    *,
    verify_remote: bool,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not (checkout / ".git").exists():
        return (f"veRL source checkout is not a Git worktree: {checkout}",)
    try:
        head = _git(checkout, "rev-parse", "HEAD")
        status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        origin = _git(checkout, "remote", "get-url", "origin")
    except RuntimeError as error:
        return (str(error),)
    if head != profile.fork_revision:
        errors.append("veRL checkout HEAD differs from fork_revision")
    if status:
        errors.append("veRL checkout is dirty")
    if _canonical_repository(origin) != _canonical_repository(profile.source_repository):
        errors.append("veRL checkout origin differs from source_repository")
    if FULL_REVISION.fullmatch(profile.fork_revision) is not None:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", profile.upstream_revision, profile.fork_revision],
            cwd=checkout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestor.returncode != 0:
            errors.append("veRL fork revision does not descend from the recorded upstream revision")
    if verify_remote and FULL_REVISION.fullmatch(profile.fork_revision) is not None:
        remote = subprocess.run(
            ["git", "ls-remote", "origin"],
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
        )
        if remote.returncode != 0:
            errors.append("veRL origin could not be queried for publication")
        elif not any(line.split(maxsplit=1)[0] == profile.fork_revision for line in remote.stdout.splitlines()):
            errors.append("veRL fork revision is not published on origin")
    return tuple(errors)


def _canonical_repository(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("git@github.com:"):
        normalized = f"https://github.com/{normalized.removeprefix('git@github.com:')}"
    return normalized.removesuffix(".git").removesuffix("/").lower()


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"veRL profile field {key!r} must be a string")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"veRL profile field {key!r} must be an integer")
    return value


def _string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"veRL profile field {key!r} must be an array of strings")
    parsed = tuple(value)
    if len(set(parsed)) != len(parsed) or tuple(sorted(parsed)) != parsed:
        raise ValueError(f"veRL profile field {key!r} must be unique and sorted")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("profile.toml"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).with_name("release") / "uv.lock",
    )
    parser.add_argument("--source-checkout", type=Path)
    parser.add_argument("--verify-remote", action="store_true")
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args(argv)
    profile = ReleaseProfile.read(arguments.profile)
    definition_errors = validate_definition(profile)
    if definition_errors:
        for error in definition_errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if not arguments.release:
        print(f"{profile.profile_id}: definition valid; release status={profile.release_status}")
        return 0
    blockers = release_blockers(
        profile,
        lock_path=arguments.lock,
        source_checkout=arguments.source_checkout,
        verify_remote=arguments.verify_remote,
        execute_container_gate=True,
    )
    if blockers:
        for blocker in blockers:
            print(f"blocked: {blocker}", file=sys.stderr)
        return 1
    print(f"{profile.profile_id}: release gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
