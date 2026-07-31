"""Provider-neutral assembly of one immutable actual-job build context."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast

from posttrain.common import ContractError, JsonValue
from posttrain.execution import (
    BackendRuntimeLock,
    DatasetPackageLock,
    EnvironmentActivationLock,
    EnvironmentPackageLock,
    JobPackageManifest,
    RuntimeDependencyLock,
)

from .contracts import (
    DatasetPackager,
    EnvironmentPackager,
    MaterializedEnvironments,
    MaterializedRuntimeDependency,
    SourcePackage,
)
from .planning import JobPackPlan

_FORBIDDEN_SOURCE_NAMES = {
    ".env",
    ".git",
    ".netrc",
    "credentials",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets",
    "token",
}
_FORBIDDEN_WEIGHT_SUFFIXES = {".ckpt", ".gguf", ".safetensors"}
_MAX_PROJECT_CONFIG_FILES = 10_000
_MAX_PROJECT_CONFIG_BYTES = 64 * 1024 * 1024
_SECRET_TEXT_KEY = re.compile(
    r"^\s*(?:-\s*)?[\"']?"
    r"(?:access[_-]?token|api[_-]?key|credential|credentials|password|secret|token)"
    r"[\"']?\s*[:=]",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"https?://[^/\s:@]+(?::[^@/\s]*)?@")
_SECRET_CONFIG_KEYS = {
    "access_token",
    "api_key",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


class _FrozenDict(dict[str, object]):
    """JSON-compatible dict whose retained snapshot cannot be mutated."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("resolved configuration is immutable")

    __delitem__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable  # pyright: ignore[reportAssignmentType]
    setdefault = _immutable  # pyright: ignore[reportAssignmentType]
    update = _immutable  # pyright: ignore[reportAssignmentType]


class _FrozenList(list[object]):
    """JSON-compatible list whose retained snapshot cannot be mutated."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("resolved configuration is immutable")

    __delitem__ = _immutable
    __iadd__ = _immutable  # pyright: ignore[reportAssignmentType]
    __imul__ = _immutable  # pyright: ignore[reportAssignmentType]
    __setitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable  # pyright: ignore[reportAssignmentType]


@dataclass(frozen=True, slots=True)
class ProjectConfigBundle:
    """Closed, non-secret project configuration required to rebuild one job."""

    files: Mapping[str, bytes]
    selected_work_package: str
    project_manifest: str = ".posttrain/project.toml"

    def __post_init__(self) -> None:
        files = dict(self.files)
        ordered: dict[str, bytes] = {}
        total_bytes = 0
        if len(files) > _MAX_PROJECT_CONFIG_FILES:
            raise ContractError("project configuration exceeds its file-count limit")
        for path, contents in sorted(files.items()):
            normalized = _relative_config_path(path)
            if normalized != path:
                raise ContractError("project configuration paths must be normalized")
            if not isinstance(contents, bytes):
                raise ContractError("project configuration contents must be bytes")
            if PurePosixPath(path).name.lower() in _FORBIDDEN_SOURCE_NAMES:
                raise ContractError(f"project configuration contains a forbidden file: {path}")
            if not PurePosixPath(path).is_relative_to(PurePosixPath(".posttrain")):
                raise ContractError("project configuration files must stay below .posttrain")
            total_bytes += len(contents)
            if total_bytes > _MAX_PROJECT_CONFIG_BYTES:
                raise ContractError("project configuration exceeds its byte limit")
            _reject_secret_config_file(path, contents)
            ordered[path] = contents
        manifest = _relative_config_path(self.project_manifest)
        work_package = _relative_config_path(self.selected_work_package)
        if manifest not in ordered:
            raise ContractError("project configuration does not contain its project manifest")
        if work_package not in ordered:
            raise ContractError("project configuration does not contain the selected work package")
        _validate_project_config_closure(
            ordered,
            project_manifest=manifest,
            selected_work_package=work_package,
        )
        object.__setattr__(self, "files", MappingProxyType(ordered))
        object.__setattr__(self, "project_manifest", manifest)
        object.__setattr__(self, "selected_work_package", work_package)

    @property
    def digest(self) -> str:
        """Content identity of the exact project configuration closure."""

        return _project_config_digest(self)


@dataclass(frozen=True, slots=True)
class JobPackInputs:
    """Local source and configuration inputs for an already-derived plan."""

    framework_source: SourcePackage | None
    project_source: SourcePackage
    resolved_inputs: Mapping[str, JsonValue]
    project_config: ProjectConfigBundle
    # Set when the framework is installed rather than checked out. The job-kind
    # image already holds the framework's dependencies, so these install with
    # --no-deps and no resolution happens inside the image build.
    framework_wheels: tuple[Path, ...] = ()
    activation_resource_sources: Mapping[tuple[str, str], Path] = field(default_factory=dict)
    project_environment_sources: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selected = dict(self.resolved_inputs)
        _json_bytes(selected, pretty=False)
        _reject_secret_config(selected)
        payload = _freeze_json(selected)
        object.__setattr__(
            self,
            "resolved_inputs",
            cast(Mapping[str, JsonValue], payload),
        )
        sources = dict(self.activation_resource_sources)
        if any(not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(item, str) for item in key) for key in sources):
            raise ContractError("activation resource source keys must be (environment id, resource name)")
        if any(not isinstance(path, Path) for path in sources.values()):
            raise ContractError("activation resource sources must be paths")
        object.__setattr__(self, "activation_resource_sources", MappingProxyType(sources))
        project_sources = dict(self.project_environment_sources)
        if any(not isinstance(path, str) or not isinstance(source, Path) for path, source in project_sources.items()):
            raise ContractError("project environment sources must map declared paths to local paths")
        object.__setattr__(self, "project_environment_sources", MappingProxyType(project_sources))


@dataclass(frozen=True, slots=True)
class PackedJobContext:
    """A reusable content-addressed context ready for an OCI builder."""

    root: Path
    manifest: JobPackageManifest
    context_digest: str
    publication_key: str

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ContractError("packed job context root must be absolute")
        for label, value in (
            ("context", self.context_digest),
            ("publication", self.publication_key),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ContractError(f"packed job {label} digest must be SHA-256")


class JobPackService:
    """Materialize, verify, and atomically retain one actual-job context."""

    def __init__(
        self,
        *,
        output_root: Path,
        dataset_packager: DatasetPackager,
        environment_packager: EnvironmentPackager | None = None,
    ) -> None:
        if not output_root.is_absolute():
            raise ContractError("job package output root must be absolute")
        self._output_root = output_root
        self._dataset_packager = dataset_packager
        self._environment_packager = environment_packager

    def pack(
        self,
        plan: JobPackPlan,
        inputs: JobPackInputs,
    ) -> PackedJobContext:
        """Build a deterministic context without selecting an execution provider."""

        resolved_inputs = cast(
            dict[str, JsonValue],
            _thaw_json(inputs.resolved_inputs),
        )
        resolved_inputs_digest = _semantic_digest(resolved_inputs)
        if resolved_inputs_digest != plan.spec.resolved_inputs_digest:
            raise ContractError("resolved configuration differs from the planned inputs")
        if inputs.framework_source is not None:
            framework_digest = digest_source_package(inputs.framework_source)
        else:
            framework_digest = _framework_wheel_digest(inputs.framework_wheels)
        project_digest = digest_source_package(inputs.project_source)
        if framework_digest != plan.spec.framework_source_digest:
            raise ContractError("framework code differs from the job-pack plan")
        if project_digest != plan.spec.project_source_digest:
            raise ContractError("project source tree differs from the job-pack plan")
        project_payload = tomllib.loads(inputs.project_config.files[inputs.project_config.project_manifest].decode())
        if project_payload.get("project_id") != plan.spec.project_id:
            raise ContractError("project manifest identity differs from the job-pack plan")

        self._output_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".job-context-stage-", dir=self._output_root))
        work = Path(tempfile.mkdtemp(prefix=".job-context-work-", dir=self._output_root))
        try:
            _create_layout(stage)
            if inputs.framework_source is not None:
                _copy_source_package(
                    inputs.framework_source,
                    stage / "sources" / "framework",
                )
            else:
                _stage_framework_wheels(inputs.framework_wheels, stage)
            _copy_source_package(
                inputs.project_source,
                stage / "sources" / "project",
            )
            _stage_activation_resources(
                stage,
                plan.spec.environment_activations,
                inputs.activation_resource_sources,
            )
            code_requirements = _code_requirements(inputs)
            code_requirements_path = stage / "locks" / "code.requirements.txt"
            code_requirements_path.write_bytes(code_requirements)
            code_requirements_digest = _bytes_digest(code_requirements)

            _stage_project_config(inputs.project_config, stage / "config" / "project")
            project_config_digest = _project_config_digest(inputs.project_config)
            resolved_config = {
                "schema": "posttrain.resolved-job.v1",
                "project_id": plan.spec.project_id,
                "work_package_id": plan.spec.work_package_id,
                "job_id": plan.spec.job_id,
                "job_definition_id": plan.spec.job_definition_id,
                "runtime_variant": plan.spec.runtime_variant,
                "project_root": "project",
                "project_manifest": (f"project/{inputs.project_config.project_manifest}"),
                "selected_work_package": (f"project/{inputs.project_config.selected_work_package}"),
                "resolved_inputs": resolved_inputs,
                "family_registry_lock": dict(plan.spec.family_registry_lock),
            }
            config_bytes = _json_bytes(resolved_config, pretty=True)
            (stage / "config" / "resolved.json").write_bytes(config_bytes)
            resolved_config_digest = _bytes_digest(config_bytes)

            datasets = self._dataset_packager.package(
                plan.spec.datasets,
                output_root=stage,
            )
            if datasets.root.resolve() != stage.resolve():
                raise ContractError("dataset packager returned a different context root")
            dataset_locks = tuple(sorted(datasets.locks, key=lambda lock: lock.seat_name))
            _validate_dataset_packages(stage, dataset_locks)

            environments = self._materialize_environments(plan, inputs, work)
            _stage_environment_packages(environments, stage)
            environment_locks = tuple(package.lock for package in environments.packages)
            runtime_dependency_locks = _stage_runtime_dependencies(
                environments,
                stage,
                runtime_variant=plan.spec.runtime_variant,
                environment_locks=environment_locks,
            )
            control_lock = next(lock for lock in runtime_dependency_locks if lock.role == "control")
            _validate_environment_selection(plan, environment_locks)
            if plan.spec.runtime_variant == "online-rl-verl-py313" and inputs.framework_source is None:
                raise ContractError(
                    "the veRL capsule projects framework source into its backend "
                    "environment, so it cannot be packed from installed "
                    "distributions; configure registry.framework_source_root"
                )
            backend_runtime = _backend_runtime_lock(
                runtime_variant=plan.spec.runtime_variant,
                resolved_inputs=resolved_inputs,
                framework_source=inputs.framework_source,
                work=work,
            )

            manifest = JobPackageManifest(
                project_id=plan.spec.project_id,
                work_package_id=plan.spec.work_package_id,
                job_id=plan.spec.job_id,
                job_definition_id=plan.spec.job_definition_id,
                job_kind=plan.spec.job_kind,
                resolved_inputs_digest=plan.spec.resolved_inputs_digest,
                framework_source_digest=framework_digest,
                project_source_digest=project_digest,
                runtime_dependencies_digest=control_lock.requirements_digest,
                code_requirements_digest=code_requirements_digest,
                resolved_config_digest=resolved_config_digest,
                project_config_digest=project_config_digest,
                universal_image=plan.spec.universal_image,
                kind_image=plan.spec.kind_image,
                runtime_variant=plan.spec.runtime_variant,
                runtime_dependency_locks=runtime_dependency_locks,
                backend_runtime=backend_runtime,
                environment_packages=environment_locks,
                environment_activations=plan.spec.environment_activations,
                datasets=dataset_locks,
                expected_artifact_roles=plan.spec.expected_artifact_roles,
                worker_contract_version=plan.spec.worker_contract_version,
            )
            (stage / "package.json").write_bytes(manifest.to_bytes())
            _normalize_tree_metadata(stage)
            _verify_staged_context(stage, manifest)
            context_digest = _tree_digest(stage)
            destination = self._retain(
                stage,
                package_key=manifest.package_key,
                context_digest=context_digest,
            )
            publication_key = _semantic_digest(
                {
                    "package_key": manifest.package_key,
                    "publication": plan.publication.to_payload(),
                }
            )
            return PackedJobContext(
                root=destination,
                manifest=manifest,
                context_digest=context_digest,
                publication_key=publication_key,
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)

    def _materialize_environments(
        self,
        plan: JobPackPlan,
        inputs: JobPackInputs,
        work: Path,
    ) -> MaterializedEnvironments:
        if not plan.spec.environment_wheels and not plan.spec.project_environment_sources:
            runtime_requirements = work / "runtime.requirements.txt"
            runtime_requirements.write_bytes(b"")
            return MaterializedEnvironments(
                packages=(),
                runtime_requirements=runtime_requirements,
                runtime_dependencies_digest=_bytes_digest(b""),
            )
        if self._environment_packager is None:
            raise ContractError("selected environments require an environment packager")
        return self._environment_packager.package(
            git_sources=plan.spec.git_sources,
            wheel_requests=plan.spec.environment_wheels,
            project_sources=inputs.project_environment_sources,
            project_requests=plan.spec.project_environment_sources,
            kind_profile=plan.spec.runtime_variant,
            output_root=work,
        )

    def _retain(
        self,
        stage: Path,
        *,
        package_key: str,
        context_digest: str,
    ) -> Path:
        destination = self._output_root / package_key
        if destination.exists():
            _verify_retained_context(
                destination,
                package_key=package_key,
                context_digest=context_digest,
            )
            return destination
        try:
            stage.replace(destination)
        except FileExistsError:
            _verify_retained_context(
                destination,
                package_key=package_key,
                context_digest=context_digest,
            )
        return destination


def digest_source_package(source: SourcePackage) -> str:
    """Compute the portable digest expected by a semantic job-pack plan."""

    _validate_source_package(source)
    return _tree_digest(source.root)


def _create_layout(root: Path) -> None:
    for relative in (
        "locks",
        "wheels/environments",
        "wheels/framework",
        "sources/framework",
        "sources/project",
        "config",
        "datasets",
        "environment-resources",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _stage_activation_resources(
    root: Path,
    activations: tuple[EnvironmentActivationLock, ...],
    sources: Mapping[tuple[str, str], Path],
) -> None:
    expected = {
        (activation.environment_id, name): resource
        for activation in activations
        for name, resource in activation.resources.items()
    }
    if set(sources) != set(expected):
        raise ContractError("activation resource inputs differ from the job-pack plan")
    for key, resource in expected.items():
        source = sources[key]
        contents = _read_regular_file(source, f"activation resource {key[0]}/{key[1]}")
        if len(contents) != resource.size_bytes or _bytes_digest(contents) != resource.digest:
            raise ContractError(f"activation resource differs from its planned lock: {key[0]}/{key[1]}")
        destination = root.joinpath(*PurePosixPath(resource.staged_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)


def _stage_project_config(
    config: ProjectConfigBundle,
    destination: Path,
) -> None:
    destination.mkdir()
    for relative, contents in config.files.items():
        target = destination.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)


def _project_config_digest(config: ProjectConfigBundle) -> str:
    return _semantic_digest(
        {
            "files": [
                {
                    "path": path,
                    "sha256": _bytes_digest(contents),
                    "size_bytes": len(contents),
                }
                for path, contents in config.files.items()
            ],
            "project_manifest": config.project_manifest,
            "selected_work_package": config.selected_work_package,
        }
    )


def _relative_config_path(value: str) -> str:
    if not value or value != value.strip() or "\\" in value:
        raise ContractError("project configuration path must be a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError("project configuration path must stay inside the staged project")
    return value


def _validate_project_config_closure(
    files: Mapping[str, bytes],
    *,
    project_manifest: str,
    selected_work_package: str,
) -> None:
    if project_manifest != ".posttrain/project.toml":
        raise ContractError("project manifest must retain the .posttrain/project.toml layout")
    try:
        payload = tomllib.loads(files[project_manifest].decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ContractError("project manifest is invalid TOML") from error
    if not isinstance(payload, dict):
        raise ContractError("project manifest must be a TOML table")
    _reject_secret_config(payload)
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2}:
        raise ContractError("project manifest schema_version must be 1 or 2")
    control = PurePosixPath(".posttrain")

    configured_work_packages = payload.get("work_packages", "work_packages")
    if not isinstance(configured_work_packages, str):
        raise ContractError("project work_packages path must be a string")
    work_package_root = _manifest_source_path(
        control,
        configured_work_packages,
        "work_packages",
    )
    selected = PurePosixPath(selected_work_package)
    if selected == work_package_root or not selected.is_relative_to(work_package_root):
        raise ContractError("selected work package must be a file below the declared work_packages path")
    packed_work_packages = tuple(path for path in files if PurePosixPath(path).is_relative_to(work_package_root))
    if packed_work_packages != (selected_work_package,):
        raise ContractError("project configuration must contain only the selected work package")

    configured_overlays = payload.get("catalog_overlays", ["catalog"])
    if not isinstance(configured_overlays, list) or not all(isinstance(value, str) for value in configured_overlays):
        raise ContractError("project catalog_overlays must be a string list")
    overlay_roots: list[PurePosixPath] = []
    for configured in configured_overlays:
        overlay = _manifest_source_path(
            control,
            configured,
            "catalog overlay",
        )
        overlay_roots.append(overlay)
        if not any(PurePosixPath(path).is_relative_to(overlay) and PurePosixPath(path) != overlay for path in files):
            raise ContractError(f"project configuration omits declared catalog overlay {configured}")

    project_brief = payload.get("project_brief")
    if schema_version == 1 and project_brief is not None:
        raise ContractError("project schema_version 1 cannot declare project_brief")
    if schema_version == 2 and project_brief is None:
        raise ContractError("project schema_version 2 requires project_brief")
    brief_path: PurePosixPath | None = None
    if project_brief is not None:
        if not isinstance(project_brief, str):
            raise ContractError("project_brief path must be a string")
        brief_path = _manifest_source_path(
            control,
            project_brief,
            "project brief",
        )
        if brief_path.as_posix() not in files:
            raise ContractError("project configuration omits its declared project brief")

    configured_state = payload.get("state", "state")
    if not isinstance(configured_state, str):
        raise ContractError("project state path must be a string")
    reserved_state_root = control / "state"
    state_roots = [reserved_state_root]
    if not PurePosixPath(configured_state).is_absolute():
        state_roots.append(_manifest_source_path(control, configured_state, "state"))
    if any(PurePosixPath(path).is_relative_to(state_root) for path in files for state_root in state_roots):
        raise ContractError("machine-local project state cannot be packed as configuration")

    exact_files = {PurePosixPath(project_manifest), selected}
    if brief_path is not None:
        exact_files.add(brief_path)
    unexpected = tuple(
        path
        for path in files
        if PurePosixPath(path) not in exact_files
        and not any(PurePosixPath(path).is_relative_to(root) and PurePosixPath(path) != root for root in overlay_roots)
    )
    if unexpected:
        raise ContractError(
            "project configuration contains files outside its declared closure: " + ", ".join(unexpected)
        )


def _manifest_source_path(
    control: PurePosixPath,
    configured: str,
    label: str,
) -> PurePosixPath:
    if not configured or configured != configured.strip() or "\\" in configured:
        raise ContractError(f"project {label} path must be normalized")
    path = PurePosixPath(configured)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"project {label} path must remain relative to .posttrain")
    return control / path


def _validate_source_package(source: SourcePackage) -> None:
    if not source.root.is_dir() or source.root.is_symlink() or not any(source.root.iterdir()):
        raise ContractError("source package root must be a non-empty directory")
    _tree_digest(source.root)
    for configured in source.install_roots:
        selected = source.root if configured == "." else source.root.joinpath(*configured.split("/"))
        if (
            not selected.is_dir()
            or selected.is_symlink()
            or not (selected / "pyproject.toml").is_file()
            or (selected / "pyproject.toml").is_symlink()
        ):
            raise ContractError(f"source install root has no regular pyproject.toml: {configured}")
        if any(character.isspace() for character in configured):
            raise ContractError("source install roots cannot contain whitespace")


def _copy_source_package(source: SourcePackage, destination: Path) -> None:
    _validate_source_package(source)
    for child in sorted(source.root.iterdir(), key=lambda path: path.name):
        _copy_entry(child, destination / child.name)
    if _tree_digest(destination) != _tree_digest(source.root):
        raise ContractError("staged source tree differs from its selected source")


def _copy_entry(source: Path, destination: Path) -> None:
    mode = source.lstat().st_mode
    relative_name = source.name.lower()
    if source.is_symlink():
        raise ContractError(f"source packages do not accept symlinks: {source}")
    if relative_name in _FORBIDDEN_SOURCE_NAMES:
        raise ContractError(f"source package contains a forbidden filename: {source.name}")
    if source.suffix.lower() in _FORBIDDEN_WEIGHT_SUFFIXES or relative_name in {
        "model.bin",
        "pytorch_model.bin",
    }:
        raise ContractError(f"source package contains model weights: {source.name}")
    if stat.S_ISDIR(mode):
        destination.mkdir()
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            _copy_entry(child, destination / child.name)
    elif stat.S_ISREG(mode):
        shutil.copyfile(source, destination)
        destination.chmod(0o755 if mode & 0o111 else 0o644)
    else:
        raise ContractError(f"source package contains a special file: {source.name}")


def _code_requirements(inputs: JobPackInputs) -> bytes:
    requirements: list[str] = []
    sources: list[tuple[str, SourcePackage]] = [("project", inputs.project_source)]
    if inputs.framework_source is not None:
        sources.insert(0, ("framework", inputs.framework_source))
    else:
        requirements.extend(f"./wheels/framework/{wheel.name}" for wheel in inputs.framework_wheels)
    for namespace, source in sources:
        for install_root in source.install_roots:
            suffix = "" if install_root == "." else f"/{install_root}"
            requirements.append(f"./sources/{namespace}{suffix}")
    ordered = sorted(set(requirements))
    if not ordered:
        raise ContractError("at least one framework or project source package must be installed")
    return ("".join(f"{requirement}\n" for requirement in ordered)).encode()


def _framework_wheel_digest(wheels: tuple[Path, ...]) -> str:
    """Digest the framework wheels that will be installed into the image.

    Mirrors the CLI-side computation so the plan and the pack agree on identity
    without either trusting the other's arithmetic.
    """
    if not wheels:
        raise ContractError("framework code requires either a source tree or built wheels")
    entries = [
        {"name": wheel.name, "sha256": _file_digest(wheel)} for wheel in sorted(wheels, key=lambda path: path.name)
    ]
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _staged_framework_digest(root: Path) -> str:
    """Digest whichever way framework code was staged into the context.

    A source tree and a set of wheels are alternatives, never both, so finding
    both staged means the context was assembled twice over and its identity
    would depend on which route the reader happened to check.
    """
    wheels = tuple(sorted(path for path in (root / "wheels/framework").glob("*.whl") if path.is_file()))
    source = root / "sources/framework"
    has_source = source.is_dir() and any(path.is_file() for path in source.rglob("*"))
    if wheels and has_source:
        raise ContractError("framework code is staged both as a source tree and as wheels")
    if wheels:
        return _framework_wheel_digest(wheels)
    if has_source:
        return _tree_digest(source)
    raise ContractError("no framework code is staged in the job context")


def _stage_framework_wheels(wheels: tuple[Path, ...], stage: Path) -> None:
    destination = stage / "wheels" / "framework"
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        shutil.copy2(wheel, destination / wheel.name)


def _stage_environment_packages(
    environments: MaterializedEnvironments,
    context: Path,
) -> None:
    destination = context / "wheels" / "environments"
    seen: set[str] = set()
    for package in environments.packages:
        lock = package.lock
        if lock.wheel_filename in seen:
            raise ContractError(f"environment wheel filename collides: {lock.wheel_filename}")
        seen.add(lock.wheel_filename)
        contents = _read_regular_file(package.path, "environment wheel")
        if len(contents) != lock.wheel_size_bytes:
            raise ContractError(f"environment wheel size differs from its lock: {lock.package}")
        if _bytes_digest(contents) != lock.wheel_digest:
            raise ContractError(f"environment wheel digest differs from its lock: {lock.package}")
        (destination / lock.wheel_filename).write_bytes(contents)


def _stage_runtime_dependencies(
    environments: MaterializedEnvironments,
    context: Path,
    *,
    runtime_variant: str,
    environment_locks: tuple[EnvironmentPackageLock, ...],
) -> tuple[RuntimeDependencyLock, ...]:
    dependencies = environments.runtime_dependencies
    if not dependencies:
        contents = _read_regular_file(
            environments.runtime_requirements,
            "runtime requirements",
        )
        digest = _bytes_digest(contents)
        if digest != environments.runtime_dependencies_digest:
            raise ContractError("runtime requirements differ from their dependency digest")
        dependencies = (
            MaterializedRuntimeDependency(
                path=environments.runtime_requirements,
                lock=RuntimeDependencyLock(
                    role="control",
                    python_version="3.13.12",
                    python_executable="/opt/posttrain/venv/bin/python",
                    requirements_path="locks/runtime.control.requirements.txt",
                    requirements_digest=digest,
                    resolution_digest=_semantic_digest(
                        {
                            "role": "control",
                            "python_version": "3.13.12",
                            "requirements_digest": digest,
                        }
                    ),
                ),
            ),
        )
    locks = tuple(item.lock for item in dependencies)
    roles = tuple(lock.role for lock in locks)
    required_roles = ("backend", "control") if runtime_variant == "online-rl-verl-py313" else ("control",)
    if roles != required_roles:
        raise ContractError(f"runtime variant {runtime_variant} requires dependency roles " + ", ".join(required_roles))
    for dependency in dependencies:
        contents = _read_regular_file(
            dependency.path,
            f"{dependency.lock.role} runtime requirements",
        )
        if _bytes_digest(contents) != dependency.lock.requirements_digest:
            raise ContractError(f"{dependency.lock.role} runtime requirements differ from their digest")
        _validate_runtime_requirements(
            contents,
            environment_locks=environment_locks,
        )
        destination = context.joinpath(*PurePosixPath(dependency.lock.requirements_path).parts)
        destination.write_bytes(contents)
    # Keep the established control-lock path during the manifest-v1 migration.
    control = next(item for item in dependencies if item.lock.role == "control")
    control_contents = _read_regular_file(
        control.path,
        "control runtime requirements",
    )
    (context / "locks" / "runtime.requirements.txt").write_bytes(control_contents)
    return locks


def _backend_runtime_lock(
    *,
    runtime_variant: str,
    resolved_inputs: Mapping[str, JsonValue],
    framework_source: SourcePackage | None,
    work: Path,
) -> BackendRuntimeLock | None:
    if runtime_variant != "online-rl-verl-py313":
        return None
    if framework_source is None:
        # Guarded by the caller; restated here so the projection below can never
        # be reached without the source it reads.
        raise ContractError("the veRL capsule requires packed framework source")
    training = resolved_inputs.get("training")
    if not isinstance(training, Mapping):
        raise ContractError("veRL capsule requires a resolved training selection")
    resolved = training.get("resolved")
    if not isinstance(resolved, Mapping):
        raise ContractError("veRL capsule requires resolved training facts")
    backend = resolved.get("backend")
    options = resolved.get("backend_options")
    if not isinstance(backend, str) or not backend.startswith("verl@"):
        raise ContractError("veRL runtime variant requires a veRL training backend")
    if not isinstance(options, Mapping):
        raise ContractError("veRL training backend options must be an object")
    required = {
        "python_executable": "/opt/posttrain-verl/bin/python",
        "working_directory": "/opt/posttrain-verl/workdir",
    }
    for option_name, expected in required.items():
        value = options.get(option_name)
        if value != expected:
            raise ContractError(f"veRL {option_name} must use capsule-owned path {expected}")
    if options.get("source_dirty") not in {None, False}:
        raise ContractError("veRL capsule cannot select a dirty source worktree")
    source_revision = options.get("source_revision")
    dependency_digest = options.get("dependency_lock_sha256")
    if not isinstance(source_revision, str) or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ContractError("veRL capsule requires a full source revision")
    if not isinstance(dependency_digest, str) or re.fullmatch(r"[0-9a-f]{64}", dependency_digest) is None:
        raise ContractError("veRL capsule requires a dependency lock digest")
    projection = work / "verl-worker-projection"
    (projection / "posttrain").mkdir(parents=True)
    for package in ("common", "data", "train"):
        source = framework_source.root / "packages" / package / "src" / "posttrain" / package
        if not source.is_dir() or source.is_symlink():
            raise ContractError(f"framework snapshot lacks veRL worker projection package: {package}")
        _copy_entry(source, projection / "posttrain" / package)
    _normalize_tree_metadata(projection)
    return BackendRuntimeLock(
        backend="verl",
        source_repository="https://github.com/carbonteq-ai/verl.git",
        source_revision=source_revision,
        dependency_lock_path="/opt/posttrain-verl/release/uv.lock",
        dependency_lock_digest=dependency_digest,
        working_directory="/opt/posttrain-verl/workdir",
        projection_path="/opt/posttrain-verl/projection",
        projection_digest=_tree_digest(projection),
        worker_module="posttrain.train.backends.verl.worker",
    )


def _validate_environment_selection(
    plan: JobPackPlan,
    locks: tuple[EnvironmentPackageLock, ...],
) -> None:
    observed_git = tuple(
        (lock.package, lock.repository, lock.revision, lock.subdirectory)
        for lock in locks
        if lock.source_kind == "git"
    )
    expected_git = tuple(
        (
            request.package,
            request.repository,
            request.revision,
            request.subdirectory,
        )
        for request in plan.spec.environment_wheels
    )
    observed_project = tuple(
        (lock.package, lock.project_path, lock.tree_digest)
        for lock in locks
        if lock.source_kind == "project-path"
    )
    expected_project = tuple(
        (source.package, source.path, source.tree_digest)
        for source in plan.spec.project_environment_sources
    )
    if observed_git != expected_git or observed_project != expected_project:
        raise ContractError("materialized environment packages differ from the job-pack plan")


def _validate_runtime_requirements(
    contents: bytes,
    *,
    environment_locks: tuple[EnvironmentPackageLock, ...],
) -> None:
    try:
        lines = contents.decode().splitlines()
    except UnicodeDecodeError as error:
        raise ContractError("runtime requirements must be UTF-8") from error
    current = ""
    expected_wheels = {f"./wheels/environments/{lock.wheel_filename}" for lock in environment_locks}
    observed_wheels: set[str] = set()
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = f"{current} {stripped}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        if "--hash=sha256:" not in current:
            raise ContractError("every runtime requirement must be hash locked")
        if "git+" in current or "://" in current:
            raise ContractError("runtime requirements cannot fetch remote source URLs")
        head = current.partition(" --hash=sha256:")[0].strip()
        if head.startswith("./"):
            if head not in expected_wheels:
                raise ContractError("runtime requirements reference an unselected local wheel")
            observed_wheels.add(head)
        elif (
            "/" in head
            or "\\" in head
            or "@" in head
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*"
                r"(?:\[[A-Za-z0-9,._-]+\])?==[^\s;]+"
                r"(?:\s*;\s*.+)?",
                head,
            )
            is None
        ):
            raise ContractError("runtime requirements must use pinned packages or selected wheels")
        current = ""
    if current:
        raise ContractError("runtime requirements end with an unterminated continuation")
    if observed_wheels != expected_wheels:
        raise ContractError("runtime requirements omit one or more selected environment wheels")


def _validate_dataset_packages(
    root: Path,
    locks: tuple[DatasetPackageLock, ...],
) -> None:
    expected_files: set[PurePosixPath] = set()
    for value in locks:
        package_path = PurePosixPath(value.package_path)
        manifest_path = PurePosixPath(value.manifest_path)
        if (
            not package_path.is_relative_to(PurePosixPath("datasets"))
            or not manifest_path.is_relative_to(PurePosixPath("datasets"))
            or package_path == PurePosixPath("datasets")
            or manifest_path == PurePosixPath("datasets")
        ):
            raise ContractError("dataset package paths must stay below datasets/")
        if package_path in expected_files or manifest_path in expected_files:
            raise ContractError("dataset package paths must be unique")
        expected_files.update((package_path, manifest_path))
        data = root.joinpath(*package_path.parts)
        manifest = root.joinpath(*manifest_path.parts)
        if not data.is_file() or data.is_symlink() or not manifest.is_file() or manifest.is_symlink():
            raise ContractError("dataset package files are missing")
        if data.stat().st_size != value.size_bytes:
            raise ContractError("dataset package size differs from its lock")
        if _file_digest(data) != value.digest:
            raise ContractError("dataset package digest differs from its lock")

    observed_files = {path.relative_to(root) for path in (root / "datasets").rglob("*") if path.is_file()}
    if observed_files != {Path(*relative.parts) for relative in expected_files}:
        raise ContractError("dataset packager emitted files outside its declared locks")


def _verify_staged_context(
    root: Path,
    manifest: JobPackageManifest,
) -> None:
    expected_top_level = {
        "config",
        "datasets",
        "environment-resources",
        "locks",
        "package.json",
        "sources",
        "wheels",
    }
    if {path.name for path in root.iterdir()} != expected_top_level:
        raise ContractError("staged job context top-level layout differs from the contract")
    retained = JobPackageManifest.from_bytes(_read_regular_file(root / "package.json", "job package manifest"))
    if retained != manifest:
        raise ContractError("staged job package manifest differs from memory")
    for observed, expected, label in (
        (
            _file_digest(root / "locks/runtime.requirements.txt"),
            manifest.runtime_dependencies_digest,
            "runtime requirements",
        ),
        (
            _file_digest(root / "locks/code.requirements.txt"),
            manifest.code_requirements_digest,
            "code requirements",
        ),
        (
            _file_digest(root / "config/resolved.json"),
            manifest.resolved_config_digest,
            "resolved configuration",
        ),
        (
            _staged_framework_digest(root),
            manifest.framework_source_digest,
            "framework code",
        ),
        (
            _tree_digest(root / "sources/project"),
            manifest.project_source_digest,
            "project source",
        ),
    ):
        if observed != expected:
            raise ContractError(f"staged {label} differs from its digest")

    try:
        resolved = json.loads((root / "config/resolved.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("resolved job configuration is invalid") from error
    if not isinstance(resolved, dict) or resolved.get("schema") != ("posttrain.resolved-job.v1"):
        raise ContractError("resolved job configuration schema is unsupported")
    expected_identities = {
        "project_id": manifest.project_id,
        "work_package_id": manifest.work_package_id,
        "job_id": manifest.job_id,
        "job_definition_id": manifest.job_definition_id,
        "runtime_variant": manifest.runtime_variant,
    }
    if any(resolved.get(field) != expected for field, expected in expected_identities.items()):
        raise ContractError("resolved job configuration identities differ from the package")
    resolved_inputs = resolved.get("resolved_inputs")
    if not isinstance(resolved_inputs, dict):
        raise ContractError("resolved job configuration inputs must be an object")
    _reject_secret_config(resolved_inputs)
    if _semantic_digest(resolved_inputs) != manifest.resolved_inputs_digest:
        raise ContractError("resolved job configuration inputs differ from their digest")

    project_manifest = _resolved_project_path(
        resolved.get("project_manifest"),
        "project manifest",
    )
    selected_work_package = _resolved_project_path(
        resolved.get("selected_work_package"),
        "selected work package",
    )
    project_root = root / "config" / "project"
    config_files = {
        path.relative_to(project_root).as_posix(): path.read_bytes()
        for path in sorted(project_root.rglob("*"))
        if path.is_file()
    }
    config = ProjectConfigBundle(
        files=config_files,
        project_manifest=project_manifest,
        selected_work_package=selected_work_package,
    )
    if _project_config_digest(config) != manifest.project_config_digest:
        raise ContractError("staged project configuration differs from its digest")
    expected_config_files = {
        Path("config/resolved.json"),
        *(Path("config/project").joinpath(*PurePosixPath(path).parts) for path in config_files),
    }
    observed_config_files = {path.relative_to(root) for path in (root / "config").rglob("*") if path.is_file()}
    if observed_config_files != expected_config_files:
        raise ContractError("staged configuration contains files outside its declared closure")

    _validate_dataset_packages(root, manifest.datasets)
    expected_wheels = {package.wheel_filename for package in manifest.environment_packages}
    wheel_root = root / "wheels" / "environments"
    observed_wheels = {path.name for path in wheel_root.iterdir() if path.is_file()}
    if observed_wheels != expected_wheels:
        raise ContractError("staged environment wheels differ from the package manifest")
    for package in manifest.environment_packages:
        wheel = wheel_root / package.wheel_filename
        if (
            wheel.is_symlink()
            or wheel.stat().st_size != package.wheel_size_bytes
            or _file_digest(wheel) != package.wheel_digest
        ):
            raise ContractError(f"staged environment wheel differs from its lock: {package.package}")
    _validate_activation_resources(root, manifest.environment_activations)
    _validate_runtime_requirements(
        (root / "locks/runtime.requirements.txt").read_bytes(),
        environment_locks=manifest.environment_packages,
    )
    for lock in manifest.runtime_dependency_locks:
        path = root.joinpath(*PurePosixPath(lock.requirements_path).parts)
        contents = _read_regular_file(
            path,
            f"{lock.role} runtime requirements",
        )
        if _bytes_digest(contents) != lock.requirements_digest:
            raise ContractError(f"staged {lock.role} runtime requirements differ from manifest")
        _validate_runtime_requirements(
            contents,
            environment_locks=manifest.environment_packages,
        )


def _validate_activation_resources(
    root: Path,
    activations: tuple[EnvironmentActivationLock, ...],
) -> None:
    resource_root = root / "environment-resources"
    if not resource_root.is_dir() or resource_root.is_symlink():
        raise ContractError("staged activation resource directory is missing")
    expected = {
        resource.staged_path: resource
        for activation in activations
        for resource in activation.resources.values()
    }
    observed = {
        path.relative_to(root).as_posix()
        for path in resource_root.rglob("*")
        if path.is_file()
    }
    if observed != set(expected):
        raise ContractError("staged activation resources differ from the package manifest")
    for staged_path, resource in expected.items():
        contents = _read_regular_file(root.joinpath(*PurePosixPath(staged_path).parts), "activation resource")
        if len(contents) != resource.size_bytes or _bytes_digest(contents) != resource.digest:
            raise ContractError(f"staged activation resource differs from its lock: {resource.name}")


def _resolved_project_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("project/"):
        raise ContractError(f"resolved job {label} must stay below config/project")
    return _relative_config_path(value.removeprefix("project/"))


def _verify_retained_context(
    destination: Path,
    *,
    package_key: str,
    context_digest: str,
) -> None:
    if not destination.is_dir() or destination.is_symlink() or _tree_digest(destination) != context_digest:
        raise ContractError("retained job context contains dirty filesystem drift")
    retained = JobPackageManifest.from_bytes((destination / "package.json").read_bytes())
    if retained.package_key != package_key:
        raise ContractError("retained job context has a conflicting package manifest")


def _read_regular_file(path: Path, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be a regular file")
    return path.read_bytes()


def _normalize_tree_metadata(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError("packed job context cannot contain symlinks")
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        else:
            raise ContractError("packed job context cannot contain special files")
        os.utime(path, (0, 0), follow_symlinks=False)
    root.chmod(0o755)
    os.utime(root, (0, 0), follow_symlinks=False)


def _tree_digest(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ContractError("tree digest root must be a regular directory")
    entries: list[dict[str, JsonValue]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if path.is_symlink():
            raise ContractError(f"tree contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(mode):
            name = path.name.lower()
            if name in _FORBIDDEN_SOURCE_NAMES:
                raise ContractError(f"tree contains a forbidden filename: {relative}")
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": _file_digest(path),
                    "executable": bool(mode & 0o111),
                }
            )
        else:
            raise ContractError(f"tree contains a special file: {relative}")
    if not entries:
        raise ContractError("tree cannot be empty")
    return _semantic_digest({"entries": entries})


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_secret_config(
    value: object,
    path: PurePosixPath | None = None,
) -> None:
    path = path or PurePosixPath()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractError("resolved configuration keys must be strings")
            normalized = key.lower().replace("-", "_")
            if normalized in _SECRET_CONFIG_KEYS:
                location = (path / key).as_posix()
                raise ContractError(f"resolved configuration cannot contain secret field {location}")
            _reject_secret_config(child, path / key)
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _reject_secret_config(child, path / str(index))
    elif isinstance(value, str):
        if (
            _URL_USERINFO.search(value)
            or "-----BEGIN OPENSSH PRIVATE KEY-----" in value
            or "-----BEGIN PRIVATE KEY-----" in value
        ):
            location = path.as_posix() or "<root>"
            raise ContractError(f"resolved configuration contains secret material at {location}")


def _reject_secret_config_file(path: str, contents: bytes) -> None:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"project configuration must be UTF-8 text: {path}") from error
    if (
        _URL_USERINFO.search(text)
        or "-----BEGIN OPENSSH PRIVATE KEY-----" in text
        or "-----BEGIN PRIVATE KEY-----" in text
    ):
        raise ContractError(f"project configuration contains secret material: {path}")
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".yaml", ".yml"} and any(
        _SECRET_TEXT_KEY.match(line) for line in text.splitlines() if not line.lstrip().startswith("#")
    ):
        raise ContractError(f"project configuration contains a secret field: {path}")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ContractError(f"project configuration contains invalid JSON: {path}") from error
        _reject_secret_config(payload)
    elif suffix == ".toml":
        try:
            payload = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise ContractError(f"project configuration contains invalid TOML: {path}") from error
        _reject_secret_config(payload)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ContractError("resolved configuration keys must be strings")
        return _FrozenDict({str(key): _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list | tuple):
        return _FrozenList(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw_json(child) for child in value]
    return value


def _json_bytes(value: object, *, pretty: bool) -> bytes:
    try:
        if pretty:
            encoded = (
                json.dumps(
                    value,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            encoded = json.dumps(
                value,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
    except (TypeError, ValueError) as error:
        raise ContractError("resolved configuration must contain only JSON values") from error
    return encoded.encode()


def _semantic_digest(value: object) -> str:
    return _bytes_digest(_json_bytes(value, pretty=False))


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
