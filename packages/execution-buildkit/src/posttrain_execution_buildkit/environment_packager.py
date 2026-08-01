"""Compose immutable environment sources, wheels, and runtime dependency locks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from posttrain.common import ContractError
from posttrain.execution import EnvironmentPackageLock, RuntimeDependencyLock
from posttrain.execution_pack import (
    EnvironmentWheelRequest,
    GitSourceRequest,
    MaterializedEnvironmentPackage,
    MaterializedEnvironments,
    MaterializedRuntimeDependency,
    ProjectEnvironmentSourceRequest,
)

from .environment_dependencies import (
    DependencyCompileGateway,
    ImmutableEnvironmentDependencyCompiler,
    KindDependencyConstraints,
)
from .environment_wheels import (
    EnvironmentWheelLock,
    ImmutableEnvironmentWheelBuilder,
    MaterializedEnvironmentWheels,
    WheelBuildGateway,
)
from .git_sources import GitGateway, ImmutableGitSourcePacker


@dataclass(frozen=True, slots=True)
class EnvironmentPackagerCacheRoots:
    """Explicit host-local caches that never become part of package identity."""

    git_sources: Path
    wheels: Path
    dependencies: Path

    def __post_init__(self) -> None:
        roots = (self.git_sources, self.wheels, self.dependencies)
        if any(not root.is_absolute() for root in roots):
            raise ContractError("environment packager cache roots must be absolute")
        if len({root.resolve() for root in roots}) != len(roots):
            raise ContractError("environment packager cache roots must be distinct")
        resolved = tuple(root.resolve() for root in roots)
        if any(
            first.is_relative_to(second) or second.is_relative_to(first)
            for index, first in enumerate(resolved)
            for second in resolved[index + 1 :]
        ):
            raise ContractError("environment packager cache roots must not overlap")


class ImmutableEnvironmentPackager:
    """Implement the environment-packaging port with immutable local adapters."""

    def __init__(
        self,
        *,
        cache_roots: EnvironmentPackagerCacheRoots,
        kind_constraints: Mapping[str, KindDependencyConstraints],
        backend_kind_constraints: Mapping[str, KindDependencyConstraints] | None = None,
        git_gateway: GitGateway | None = None,
        wheel_gateway: WheelBuildGateway | None = None,
        dependency_gateway: DependencyCompileGateway | None = None,
    ) -> None:
        selected_constraints: dict[str, KindDependencyConstraints] = {}
        for profile, constraints in sorted(kind_constraints.items()):
            if profile != constraints.profile:
                raise ContractError("kind dependency constraint key must match its profile")
            selected_constraints[profile] = constraints
        if not selected_constraints:
            raise ContractError("environment packager requires at least one kind constraint profile")

        selected_backend_constraints: dict[str, KindDependencyConstraints] = {}
        for profile, constraints in sorted((backend_kind_constraints or {}).items()):
            if profile != constraints.profile:
                raise ContractError("backend kind dependency constraint key must match its profile")
            if profile not in selected_constraints:
                raise ContractError("backend kind dependency constraints require a control profile")
            selected_backend_constraints[profile] = constraints

        self._kind_constraints = MappingProxyType(selected_constraints)
        self._backend_kind_constraints = MappingProxyType(selected_backend_constraints)
        self._cache_roots = cache_roots
        self._source_packer = ImmutableGitSourcePacker(
            cache_root=cache_roots.git_sources,
            gateway=git_gateway,
        )
        self._wheel_builder = ImmutableEnvironmentWheelBuilder(
            output_root=cache_roots.wheels,
            gateway=wheel_gateway,
        )
        self._dependency_compiler = ImmutableEnvironmentDependencyCompiler(
            output_root=cache_roots.dependencies,
            gateway=dependency_gateway,
        )

    def package(
        self,
        *,
        git_sources: tuple[GitSourceRequest, ...],
        wheel_requests: tuple[EnvironmentWheelRequest, ...],
        project_sources: Mapping[str, Path] = {},
        project_requests: tuple[ProjectEnvironmentSourceRequest, ...] = (),
        kind_profile: str,
        output_root: Path,
    ) -> MaterializedEnvironments:
        """Build every selected environment and resolve them as one dependency set."""

        if not output_root.is_absolute():
            raise ContractError("environment package output root must be absolute")
        constraints = self._kind_constraints.get(kind_profile)
        if constraints is None:
            raise ContractError(f"environment package kind profile has no exact constraints: {kind_profile}")
        if not wheel_requests and not project_requests:
            raise ContractError("environment packaging requires at least one selected source")
        output = output_root.resolve()
        for cache in (
            self._cache_roots.git_sources.resolve(),
            self._cache_roots.wheels.resolve(),
            self._cache_roots.dependencies.resolve(),
        ):
            if output.is_relative_to(cache) or cache.is_relative_to(output):
                raise ContractError("environment package work root must not overlap persistent caches")
        _validate_source_closure(git_sources, wheel_requests)

        # The service owns this per-pack work root. Persistent, content-addressed
        # source/wheel/dependency outputs live in the explicitly configured caches.
        output_root.mkdir(parents=True, exist_ok=True)
        git_wheels = ()
        if git_sources:
            sources = self._source_packer.materialize(git_sources)
            git_wheels = self._wheel_builder.build(sources, wheel_requests).wheels
        elif wheel_requests:
            raise ContractError("environment wheel requests require their Git sources")
        project_wheels = (
            self._wheel_builder.build_project_sources(project_sources, project_requests).wheels
            if project_requests
            else ()
        )
        wheels = MaterializedEnvironmentWheels(
            wheels=tuple(sorted((*git_wheels, *project_wheels), key=lambda item: item.lock.package)),
            lock=EnvironmentWheelLock(
                tuple(item.lock for item in sorted((*git_wheels, *project_wheels), key=lambda item: item.lock.package))
            ),
        )
        closures = _runtime_closure_constraints(
            constraints,
            self._backend_kind_constraints.get(kind_profile),
        )
        dependencies = tuple(self._dependency_compiler.compile(wheels, closure) for closure in closures)

        packages = tuple(
            MaterializedEnvironmentPackage(
                path=wheel.path,
                lock=EnvironmentPackageLock(
                    package=wheel.lock.package,
                    repository=wheel.lock.repository,
                    revision=wheel.lock.revision,
                    subdirectory=wheel.lock.subdirectory,
                    tree_digest=wheel.lock.source_tree_digest,
                    wheel_filename=wheel.lock.wheel_filename,
                    wheel_digest=wheel.lock.wheel_sha256,
                    wheel_size_bytes=wheel.lock.wheel_size_bytes,
                    source_kind=cast(Literal["git", "project-path"], wheel.lock.source_kind),
                    project_path=wheel.lock.project_path,
                ),
            )
            for wheel in wheels.wheels
        )
        runtime_dependencies: list[MaterializedRuntimeDependency] = []
        for dependency in dependencies:
            requirements_digest = hashlib.sha256(dependency.path.read_bytes()).hexdigest()
            if requirements_digest != dependency.lock.requirements_sha256:
                raise ContractError("materialized dependency lock differs from its portable identity")
            runtime_dependencies.append(
                MaterializedRuntimeDependency(
                    path=dependency.path,
                    lock=RuntimeDependencyLock(
                        role=cast(
                            Literal["control", "backend"],
                            dependency.lock.role,
                        ),
                        python_version=dependency.lock.python_version,
                        python_executable=dependency.lock.python_executable,
                        requirements_path=(f"locks/{dependency.lock.requirements_filename}"),
                        requirements_digest=requirements_digest,
                        resolution_digest=dependency.lock.digest,
                    ),
                )
            )
        control = next(item for item in runtime_dependencies if item.lock.role == "control")
        return MaterializedEnvironments(
            packages=packages,
            runtime_requirements=control.path,
            runtime_dependencies_digest=control.lock.requirements_digest,
            runtime_dependencies=tuple(sorted(runtime_dependencies, key=lambda item: item.lock.role)),
        )


def _validate_source_closure(
    git_sources: tuple[GitSourceRequest, ...],
    wheel_requests: tuple[EnvironmentWheelRequest, ...],
) -> None:
    selected_roots = {
        (source.repository, source.revision, subdirectory)
        for source in git_sources
        for subdirectory in source.subdirectories
    }
    requested_roots = {(wheel.repository, wheel.revision, wheel.subdirectory) for wheel in wheel_requests}
    if selected_roots != requested_roots:
        raise ContractError("environment Git sources must exactly cover selected package roots")


def _runtime_closure_constraints(
    constraints: KindDependencyConstraints,
    backend_constraints: KindDependencyConstraints | None,
) -> tuple[KindDependencyConstraints, ...]:
    control_python = "3.13.12" if constraints.profile == "online-rl-verl-py313" else "3.12"
    control = replace(
        constraints,
        role="control",
        python_version=control_python,
        python_executable="/opt/posttrain/venv/bin/python",
        requirements_filename="runtime.control.requirements.txt",
    )
    if constraints.profile != "online-rl-verl-py313":
        if backend_constraints is not None:
            raise ContractError("backend dependency constraints are supported only by the veRL runtime profile")
        return (control,)
    if backend_constraints is None:
        raise ContractError("veRL environment packaging requires exact backend kind constraints")
    backend = replace(
        backend_constraints,
        role="backend",
        python_version="3.13.12",
        python_executable="/opt/posttrain-verl/bin/python",
        requirements_filename="runtime.backend.requirements.txt",
    )
    return (backend, control)


__all__ = [
    "EnvironmentPackagerCacheRoots",
    "ImmutableEnvironmentPackager",
]
