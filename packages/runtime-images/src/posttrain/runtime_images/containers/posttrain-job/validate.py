"""Statically validate the actual-job Docker contract and staged context."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
DEFAULT_CONTEXT = HERE / "fixtures" / "minimal-context"
EXPECTED_TOP_LEVEL = {
    "config",
    "datasets",
    "locks",
    "package.json",
    "sources",
    "wheels",
}
FORBIDDEN_MANIFEST_FIELDS = {
    "attempt",
    "credentials",
    "final_image",
    "image",
    "model_weights",
    "mounts",
    "provider",
    "run_id",
    "secrets",
    "target",
}
FORBIDDEN_FILE_NAMES = {
    ".env",
    ".netrc",
    "credentials",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets",
    "token",
}
FORBIDDEN_WEIGHT_SUFFIXES = {
    ".ckpt",
    ".gguf",
    ".safetensors",
}
REQUIRED_PATHS = (
    "package.json",
    "locks/runtime.requirements.txt",
    "locks/code.requirements.txt",
    "wheels/environments",
    "sources/framework",
    "sources/project",
    "config/resolved.json",
    "config/project",
    "datasets",
)
SHA256_IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
SHA256_VALUE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_definition() -> None:
    dockerfile = (HERE / "Dockerfile").read_text(encoding="utf-8")
    bake = (HERE / "docker-bake.hcl").read_text(encoding="utf-8")
    for required in (
        "ARG POSTTRAIN_KIND_IMAGE",
        "FROM ${POSTTRAIN_KIND_IMAGE} AS runtime",
        "COPY --from=job-context",
        "--require-hashes",
        "--no-build-isolation",
        "[tool.uv.workspace]",
        "code workspace members must be non-empty and unique",
        'test -x "${VIRTUAL_ENV}/bin/python"',
        'POSTTRAIN_VERL_PYTHONPATH="/opt/posttrain-verl/projection"',
        '"${RUNTIME_VARIANT}" = "online-rl-verl-py313"',
        'test -x "/opt/posttrain-verl/bin/python"',
        '--python "/opt/posttrain-verl/bin/python"',
        "posttrain.train.backends.verl.worker",
        'ENTRYPOINT ["posttrain-runtime"]',
        'CMD ["execute", "--manifest", "/opt/posttrain/job/package.json"]',
        "posttrain-runtime qualify --manifest /opt/posttrain/job/package.json",
        "ARG RUNTIME_DEPENDENCIES_DIGEST",
        "ARG CODE_REQUIREMENTS_DIGEST",
        "ARG RESOLVED_CONFIG_DIGEST",
        "ARG RUNTIME_VARIANT",
    ):
        _require(required in dockerfile, f"Dockerfile is missing {required}")
    runtime_lock_copy = "COPY --from=job-context /locks/ locks/"
    # Framework code is copied either as a source tree or as staged wheels, so
    # the ordering assertion anchors on the code copy rather than on one route.
    code_copy = "COPY --from=job-context /sources/ sources/"
    _require(
        dockerfile.index(runtime_lock_copy) < dockerfile.index(code_copy),
        "external dependencies must be installed before source code",
    )
    runtime_install = dockerfile[dockerfile.index(runtime_lock_copy) : dockerfile.index(code_copy)]
    _require(
        "--require-hashes" in runtime_install and "--no-deps" in runtime_install,
        "the complete runtime lock must install explicitly with hashes and without metadata re-resolution",
    )
    _require(
        dockerfile.index("/sources/") < dockerfile.index("/datasets/"),
        "source code must precede datasets in the cache graph",
    )
    for forbidden in ("--mount=type=secret", "--mount=type=ssh"):
        _require(
            forbidden not in dockerfile,
            f"actual-job build must not require credential mounts: {forbidden}",
        )
    _require(
        "COPY --link --from=job-context" not in dockerfile,
        "mutable named-context inputs must not use stale-prone linked COPY layers",
    )
    package_barrier = 'RUN test -n "${PACKAGE_KEY}"'
    first_context_copy = "COPY --from=job-context"
    _require(
        package_barrier in dockerfile and dockerfile.index(package_barrier) < dockerfile.index(first_context_copy),
        "actual-job layers must bind PACKAGE_KEY before copying mutable context",
    )
    for required in (
        "compression=zstd",
        "force-compression=true",
        "oci-mediatypes=true",
        "type=provenance,mode=max",
        "type=sbom",
        "job-context = STAGED_CONTEXT",
        "RUNTIME_VARIANT = RUNTIME_VARIANT",
        'target "posttrain-job-smoke"',
    ):
        _require(required in bake, f"Bake graph is missing {required}")


def _validate_tree(root: Path) -> None:
    _require(root.is_dir(), f"staged context is not a directory: {root}")
    observed_top_level = {path.name for path in root.iterdir()}
    _require(
        observed_top_level in {frozenset(EXPECTED_TOP_LEVEL), frozenset((*EXPECTED_TOP_LEVEL, "environment-resources"))},
        "staged context top-level layout differs from the contract",
    )
    for relative in REQUIRED_PATHS:
        _require((root / relative).exists(), f"staged context is missing {relative}")

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        mode = path.lstat().st_mode
        _require(not path.is_symlink(), f"staged context contains a symlink: {relative}")
        _require(
            stat.S_ISDIR(mode) or stat.S_ISREG(mode),
            f"staged context contains a special file: {relative}",
        )
        name = path.name.lower()
        _require(
            name not in FORBIDDEN_FILE_NAMES,
            f"staged context contains a secret-like filename: {relative}",
        )
        _require(
            path.suffix.lower() not in FORBIDDEN_WEIGHT_SUFFIXES,
            f"staged context contains a model-weight file: {relative}",
        )
        _require(
            name not in {"pytorch_model.bin", "model.bin"},
            f"staged context contains a model-weight file: {relative}",
        )


def _logical_requirements(path: Path) -> tuple[str, ...]:
    logical: list[str] = []
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = f"{current} {stripped}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical.append(current)
        current = ""
    _require(not current, f"{path.name} ends with an unterminated continuation")
    return tuple(logical)


def _validate_locks(root: Path) -> None:
    runtime_lock = _logical_requirements(root / "locks/runtime.requirements.txt")
    for requirement in runtime_lock:
        _require(
            "--hash=sha256:" in requirement,
            f"runtime requirement is not hash locked: {requirement}",
        )
        _require(
            "git+" not in requirement and "://" not in requirement,
            f"runtime lock must not fetch mutable or credentialed sources: {requirement}",
        )

    code_lock = _logical_requirements(root / "locks/code.requirements.txt")
    _require(
        bool(code_lock),
        "code requirements must install at least one source tree",
    )
    for requirement in code_lock:
        _require(
            requirement in {"./sources/framework", "./sources/project"}
            or requirement.startswith(("./sources/framework/", "./sources/project/", "./wheels/framework/")),
            f"code requirement escapes staged source roots: {requirement}",
        )
        _require(
            not any(character.isspace() for character in requirement),
            f"code requirement must be one normalized local path: {requirement}",
        )
        relative = PurePosixPath(requirement)
        _require(
            ".." not in relative.parts and not relative.is_absolute(),
            f"code requirement is not normalized: {requirement}",
        )
        _require(
            (root / requirement).is_dir(),
            f"code requirement does not name a source directory: {requirement}",
        )
        _require(
            (root / requirement / "pyproject.toml").is_file(),
            f"code requirement has no pyproject.toml: {requirement}",
        )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": _file_digest(path),
                    "executable": bool(mode & 0o111),
                }
            )
        else:
            raise AssertionError(f"tree contains a special file: {relative}")
    _require(bool(entries), f"tree cannot be empty: {root}")
    return hashlib.sha256(
        json.dumps(
            {"entries": entries},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_manifest(root: Path) -> None:
    payload = json.loads((root / "package.json").read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "package manifest must be a JSON object")
    _require(
        payload.get("schema") == "posttrain.job-package.v1",
        "package manifest schema is unsupported",
    )
    leaked = sorted(FORBIDDEN_MANIFEST_FIELDS.intersection(payload))
    _require(not leaked, f"run/launch fields leaked into package manifest: {leaked}")
    for field in (
        "resolved_inputs_digest",
        "framework_source_digest",
        "project_source_digest",
        "project_config_digest",
        "runtime_dependencies_digest",
        "code_requirements_digest",
        "resolved_config_digest",
    ):
        _require(
            isinstance(payload.get(field), str) and SHA256_VALUE.fullmatch(payload[field]) is not None,
            f"package manifest {field} must be a SHA-256 digest",
        )
    expected_file_digests = {
        "runtime_dependencies_digest": root / "locks/runtime.requirements.txt",
        "code_requirements_digest": root / "locks/code.requirements.txt",
        "resolved_config_digest": root / "config/resolved.json",
    }
    for field, path in expected_file_digests.items():
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == payload[field],
            f"package manifest {field} differs from staged bytes",
        )
    _require(
        isinstance(payload.get("kind_image"), str) and SHA256_IMAGE.fullmatch(payload["kind_image"]) is not None,
        "package manifest kind_image must be digest pinned",
    )
    _require(
        isinstance(payload.get("universal_image"), str)
        and SHA256_IMAGE.fullmatch(payload["universal_image"]) is not None,
        "package manifest universal_image must be digest pinned",
    )
    runtime_variant = payload.get("runtime_variant")
    _require(
        isinstance(runtime_variant, str) and bool(runtime_variant.strip()),
        "package manifest runtime_variant must be a non-empty identity",
    )
    for field, source_root in (
        ("framework_source_digest", root / "sources/framework"),
        ("project_source_digest", root / "sources/project"),
    ):
        _require(
            _tree_digest(source_root) == payload[field],
            f"package manifest {field} differs from staged bytes",
        )

    datasets = payload.get("datasets")
    _require(isinstance(datasets, list), "package manifest datasets must be a list")
    expected_dataset_files: set[str] = set()
    for dataset in datasets:
        _require(isinstance(dataset, dict), "dataset lock must be an object")
        for field in ("package_path", "manifest_path"):
            package_path = dataset.get(field)
            _require(
                isinstance(package_path, str),
                f"dataset {field} must be a string",
            )
            relative = PurePosixPath(package_path)
            _require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and bool(relative.parts)
                and relative.parts[0] == "datasets",
                f"dataset {field} must stay below datasets/: {package_path}",
            )
            _require(
                (root / package_path).is_file(),
                f"dataset {field} is missing from context: {package_path}",
            )
            expected_dataset_files.add(package_path)
        data_path = root / dataset["package_path"]
        _require(
            data_path.stat().st_size == dataset.get("size_bytes"),
            "dataset size differs from its lock",
        )
        _require(
            _file_digest(data_path) == dataset.get("digest"),
            "dataset digest differs from its lock",
        )
    observed_dataset_files = {
        path.relative_to(root).as_posix()
        for path in (root / "datasets").rglob("*")
        if path.is_file() and path.name != ".keep"
    }
    _require(
        observed_dataset_files == expected_dataset_files,
        "staged dataset files differ from package locks",
    )

    packages = payload.get("environment_packages")
    _require(
        isinstance(packages, list),
        "package manifest environment_packages must be a list",
    )
    expected_wheels: set[str] = set()
    installed_packages: set[str] = set()
    for package in packages:
        _require(isinstance(package, dict), "environment lock must be an object")
        filename = package.get("wheel_filename")
        _require(
            isinstance(filename, str) and PurePosixPath(filename).name == filename and filename.endswith(".whl"),
            "environment wheel filename is invalid",
        )
        expected_wheels.add(filename)
        installed = package.get("package")
        _require(
            isinstance(installed, str),
            "environment package identity is invalid",
        )
        installed_packages.add(installed)
        wheel = root / "wheels/environments" / filename
        _require(wheel.is_file(), f"environment wheel is missing: {filename}")
        _require(
            wheel.stat().st_size == package.get("wheel_size_bytes"),
            f"environment wheel size differs from its lock: {filename}",
        )
        _require(
            _file_digest(wheel) == package.get("wheel_digest"),
            f"environment wheel digest differs from its lock: {filename}",
        )
    observed_wheels = {
        path.name for path in (root / "wheels/environments").iterdir() if path.is_file() and path.name != ".keep"
    }
    _require(
        observed_wheels == expected_wheels,
        "staged environment wheels differ from package locks",
    )
    activations = payload.get("environment_activations")
    _require(
        isinstance(activations, list),
        "package manifest environment_activations must be a list",
    )
    _require(
        all(
            isinstance(activation, dict) and activation.get("package") in installed_packages
            for activation in activations
        ),
        "environment activation references a missing package",
    )

    config = json.loads((root / "config/resolved.json").read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "resolved config must be a JSON object")
    _require(
        config.get("schema") == "posttrain.resolved-job.v1",
        "resolved config schema is unsupported",
    )
    for field in (
        "project_id",
        "work_package_id",
        "job_id",
        "job_definition_id",
        "runtime_variant",
    ):
        _require(
            config.get(field) == payload.get(field),
            f"resolved config {field} differs from the package manifest",
        )
    resolved_inputs = config.get("resolved_inputs")
    _require(
        isinstance(resolved_inputs, dict),
        "resolved config resolved_inputs must be an object",
    )
    resolved_inputs_digest = hashlib.sha256(
        json.dumps(
            resolved_inputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _require(
        resolved_inputs_digest == payload["resolved_inputs_digest"],
        "resolved config inputs differ from resolved_inputs_digest",
    )
    project_manifest = config.get("project_manifest")
    selected_work_package = config.get("selected_work_package")
    for field, value in (
        ("project_manifest", project_manifest),
        ("selected_work_package", selected_work_package),
    ):
        _require(
            isinstance(value, str) and value.startswith("project/"),
            f"resolved config {field} must stay below config/project",
        )
    project_root = root / "config/project"
    _require(
        project_root.is_dir(),
        "staged context is missing its closed project configuration",
    )
    files = []
    for path in sorted(project_root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    project_payload = {
        "files": files,
        "project_manifest": project_manifest.removeprefix("project/"),
        "selected_work_package": selected_work_package.removeprefix("project/"),
    }
    project_digest = hashlib.sha256(
        json.dumps(
            project_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _require(
        project_digest == payload["project_config_digest"],
        "package manifest project_config_digest differs from staged bytes",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    args = parser.parse_args()
    _validate_definition()
    _validate_tree(args.context.resolve())
    _validate_locks(args.context.resolve())
    _validate_manifest(args.context.resolve())
    print("actual-job image definition and staged context: static validation passed")


if __name__ == "__main__":
    main()
