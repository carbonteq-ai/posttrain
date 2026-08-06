from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain_execution_buildkit import (
    DependencyResolutionError,
    EnvironmentPackagerCacheRoots,
    EnvironmentWheelRequest,
    GitSourceRequest,
    ImmutableEnvironmentPackager,
    KindDependencyConstraints,
)

_REVISION = "a" * 40
_REPOSITORY = "https://github.com/example/environments"
_OTHER_REPOSITORY = "https://github.com/example/other-environments"


class FakeGitGateway:
    def __init__(
        self,
        trees: Mapping[tuple[str, str], Mapping[str, str]],
    ) -> None:
        self.trees = trees
        self.calls: list[tuple[str, ...]] = []
        self.repositories: dict[Path, str] = {}
        self.revisions: dict[Path, str] = {}

    def invoke(self, arguments: Sequence[str]) -> str:
        call = tuple(arguments)
        self.calls.append(call)
        if call[:2] == ("init", "--quiet"):
            (Path(call[2]) / ".git").mkdir(parents=True)
            return ""

        assert call[0] == "-C"
        root = Path(call[1])
        command = call[2]
        if call[2:5] == ("remote", "add", "origin"):
            self.repositories[root] = call[5]
            return ""
        if command == "fetch":
            self.revisions[root] = call[-1]
            return ""
        if command == "checkout":
            repository = self.repositories[root]
            revision = self.revisions[root]
            for relative, contents in self.trees[(repository, revision)].items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents, encoding="utf-8")
            return ""
        if command == "rev-parse":
            revision = self.revisions.get(root)
            if revision is None:
                revisions = set(self.revisions.values())
                assert len(revisions) == 1
                revision = revisions.pop()
            return revision + "\n"
        if command == "status":
            return ""
        raise AssertionError(f"unexpected Git invocation: {call}")


class FakeWheelGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def build(self, package_root: Path, output_directory: Path) -> None:
        document = (package_root / "pyproject.toml").read_text(encoding="utf-8")
        package = document.split("name = '", maxsplit=1)[1].split("'", maxsplit=1)[0]
        distribution = package.replace("-", "_")
        self.calls.append(package)
        (output_directory / f"{distribution}-1.0.0-py3-none-any.whl").write_bytes(f"{package}-wheel".encode())


class FakeDependencyGateway:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.constraints: list[str] = []

    def compile(self, **arguments: object) -> None:
        self.calls.append(arguments)
        constraints = arguments["constraints"]
        assert isinstance(constraints, Path)
        self.constraints.append(constraints.read_text(encoding="utf-8"))
        if self.error is not None:
            raise self.error

        requirements = arguments["requirements"]
        output = arguments["output"]
        working_directory = arguments["working_directory"]
        assert isinstance(requirements, Path)
        assert isinstance(output, Path)
        assert isinstance(working_directory, Path)
        blocks: list[str] = [f"shared-runtime==1.0.0 \\\n    --hash=sha256:{'f' * 64}\n"]
        for relative in requirements.read_text(encoding="utf-8").splitlines():
            wheel = working_directory / relative.removeprefix("./")
            blocks.append(f"{relative} \\\n    --hash=sha256:{_file_digest(wheel)}\n")
        output.write_text("".join(blocks), encoding="utf-8")


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_roots(tmp_path: Path) -> EnvironmentPackagerCacheRoots:
    return EnvironmentPackagerCacheRoots(
        git_sources=(tmp_path / "cache/git").absolute(),
        wheels=(tmp_path / "cache/wheels").absolute(),
        dependencies=(tmp_path / "cache/dependencies").absolute(),
    )


def _git_gateway(
    *,
    include_other_repository: bool = False,
) -> FakeGitGateway:
    trees: dict[tuple[str, str], Mapping[str, str]] = {
        (_REPOSITORY, _REVISION): {
            "README.md": "environment monorepo",
            "environments/math/pyproject.toml": ("[project]\nname = 'math-env'\nversion = '1.0.0'\n"),
            "environments/math/environment.py": "def load(): ...\n",
            "environments/text/pyproject.toml": ("[project]\nname = 'text-env'\nversion = '1.0.0'\n"),
            "environments/text/environment.py": "def load(): ...\n",
        }
    }
    if include_other_repository:
        trees[(_OTHER_REPOSITORY, _REVISION)] = {
            "pyproject.toml": ("[project]\nname = 'other-env'\nversion = '1.0.0'\n"),
            "environment.py": "def load(): ...\n",
        }
    return FakeGitGateway(trees)


def _one_repository_requests() -> tuple[
    tuple[GitSourceRequest, ...],
    tuple[EnvironmentWheelRequest, ...],
]:
    sources = (
        GitSourceRequest(
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectories=("environments/math", "environments/text"),
        ),
    )
    wheels = (
        EnvironmentWheelRequest(
            package="math-env",
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectory="environments/math",
        ),
        EnvironmentWheelRequest(
            package="text-env",
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectory="environments/text",
        ),
    )
    return sources, wheels


def _packager(
    tmp_path: Path,
    *,
    git_gateway: FakeGitGateway,
    dependency_gateway: FakeDependencyGateway,
    wheel_gateway: FakeWheelGateway | None = None,
) -> ImmutableEnvironmentPackager:
    return ImmutableEnvironmentPackager(
        cache_roots=_cache_roots(tmp_path),
        kind_constraints={
            "online-rl": KindDependencyConstraints(
                "online-rl",
                "shared-runtime==1.0.0\n",
            )
        },
        git_gateway=git_gateway,
        wheel_gateway=wheel_gateway or FakeWheelGateway(),
        dependency_gateway=dependency_gateway,
    )


def test_packages_two_environments_from_one_checkout_with_portable_locks(
    tmp_path: Path,
) -> None:
    git_gateway = _git_gateway()
    wheel_gateway = FakeWheelGateway()
    dependency_gateway = FakeDependencyGateway()
    sources, wheels = _one_repository_requests()

    result = _packager(
        tmp_path,
        git_gateway=git_gateway,
        wheel_gateway=wheel_gateway,
        dependency_gateway=dependency_gateway,
    ).package(
        git_sources=sources,
        wheel_requests=wheels,
        kind_profile="online-rl",
        output_root=(tmp_path / "work").absolute(),
    )

    assert [package.lock.package for package in result.packages] == [
        "math-env",
        "text-env",
    ]
    assert wheel_gateway.calls == ["math-env", "text-env"]
    assert sum(call[2] == "fetch" for call in git_gateway.calls if call[0] == "-C") == 1
    assert dependency_gateway.constraints == ["shared-runtime==1.0.0\n"]
    assert dependency_gateway.calls[0]["python_version"] == "3.13.12"
    assert result.runtime_dependencies_digest == _file_digest(result.runtime_requirements)
    requirements = result.runtime_requirements.read_text(encoding="utf-8")
    assert "./wheels/environments/math_env-1.0.0-py3-none-any.whl" in requirements
    assert "./wheels/environments/text_env-1.0.0-py3-none-any.whl" in requirements
    locks = json.dumps(
        [package.lock.to_payload() for package in result.packages],
        sort_keys=True,
    )
    assert str(tmp_path) not in locks
    assert "credential" not in locks


def test_verl_profile_resolves_isolated_python313_closures(
    tmp_path: Path,
) -> None:
    git_gateway = _git_gateway()
    dependency_gateway = FakeDependencyGateway()
    sources, wheels = _one_repository_requests()
    packager = ImmutableEnvironmentPackager(
        cache_roots=_cache_roots(tmp_path),
        kind_constraints={
            "online-rl-verl-py313": KindDependencyConstraints(
                "online-rl-verl-py313",
                "shared-runtime==1.0.0\n",
            )
        },
        backend_kind_constraints={
            "online-rl-verl-py313": KindDependencyConstraints(
                "online-rl-verl-py313",
                "backend-runtime==0.9.0\n",
                ("backend-runtime",),
            )
        },
        git_gateway=git_gateway,
        wheel_gateway=FakeWheelGateway(),
        dependency_gateway=dependency_gateway,
    )

    result = packager.package(
        git_sources=sources,
        wheel_requests=wheels,
        kind_profile="online-rl-verl-py313",
        output_root=(tmp_path / "work").absolute(),
    )

    assert [
        (
            item.lock.role,
            item.lock.python_version,
            item.lock.python_executable,
            item.lock.requirements_path,
        )
        for item in result.runtime_dependencies
    ] == [
        (
            "backend",
            "3.13.12",
            "/opt/posttrain-verl/bin/python",
            "locks/runtime.backend.requirements.txt",
        ),
        (
            "control",
            "3.13.12",
            "/opt/posttrain/venv/bin/python",
            "locks/runtime.control.requirements.txt",
        ),
    ]
    assert [call["python_version"] for call in dependency_gateway.calls] == [
        "3.13.12",
        "3.13.12",
    ]
    assert dependency_gateway.constraints == [
        "backend-runtime==0.9.0\n",
        "shared-runtime==1.0.0\n",
    ]
    assert dependency_gateway.calls[0]["provided_packages"] == ("backend-runtime",)
    assert (
        result.runtime_dependencies[0].lock.resolution_digest != result.runtime_dependencies[1].lock.resolution_digest
    )


def test_verl_profile_requires_exact_backend_kind_constraints(
    tmp_path: Path,
) -> None:
    sources, wheels = _one_repository_requests()
    packager = ImmutableEnvironmentPackager(
        cache_roots=_cache_roots(tmp_path),
        kind_constraints={
            "online-rl-verl-py313": KindDependencyConstraints(
                "online-rl-verl-py313",
                "shared-runtime==1.0.0\n",
            )
        },
        git_gateway=_git_gateway(),
        wheel_gateway=FakeWheelGateway(),
        dependency_gateway=FakeDependencyGateway(),
    )

    with pytest.raises(
        ContractError,
        match="requires exact backend kind constraints",
    ):
        packager.package(
            git_sources=sources,
            wheel_requests=wheels,
            kind_profile="online-rl-verl-py313",
            output_root=(tmp_path / "work").absolute(),
        )


def test_packages_environments_from_multiple_repositories(tmp_path: Path) -> None:
    git_gateway = _git_gateway(include_other_repository=True)
    dependency_gateway = FakeDependencyGateway()
    sources, wheels = _one_repository_requests()
    sources = (
        *sources,
        GitSourceRequest(_OTHER_REPOSITORY, _REVISION, (".",)),
    )
    wheels = (
        *wheels,
        EnvironmentWheelRequest(
            "other-env",
            _OTHER_REPOSITORY,
            _REVISION,
            ".",
        ),
    )

    result = _packager(
        tmp_path,
        git_gateway=git_gateway,
        dependency_gateway=dependency_gateway,
    ).package(
        git_sources=tuple(
            sorted(
                sources,
                key=lambda item: (
                    item.repository,
                    item.revision,
                    item.subdirectories,
                ),
            )
        ),
        wheel_requests=tuple(
            sorted(
                wheels,
                key=lambda item: (
                    item.package,
                    item.repository,
                    item.revision,
                    item.subdirectory,
                ),
            )
        ),
        kind_profile="online-rl",
        output_root=(tmp_path / "work").absolute(),
    )

    assert [package.lock.package for package in result.packages] == [
        "math-env",
        "other-env",
        "text-env",
    ]
    assert sum(call[2] == "fetch" for call in git_gateway.calls if call[0] == "-C") == 2


def test_propagates_combined_dependency_conflict_without_a_runtime_lock(
    tmp_path: Path,
) -> None:
    gateway = FakeDependencyGateway(DependencyResolutionError("selected environments are incompatible"))
    sources, wheels = _one_repository_requests()

    with pytest.raises(
        DependencyResolutionError,
        match="selected environments are incompatible",
    ):
        _packager(
            tmp_path,
            git_gateway=_git_gateway(),
            dependency_gateway=gateway,
        ).package(
            git_sources=sources,
            wheel_requests=wheels,
            kind_profile="online-rl",
            output_root=(tmp_path / "work").absolute(),
        )

    assert len(gateway.calls) == 1
    assert not list((tmp_path / "cache/dependencies").glob("*/environment-dependencies.lock.txt"))


def test_rejects_unknown_or_mismatched_profiles_before_fetch(
    tmp_path: Path,
) -> None:
    git_gateway = _git_gateway()
    sources, wheels = _one_repository_requests()
    packager = _packager(
        tmp_path,
        git_gateway=git_gateway,
        dependency_gateway=FakeDependencyGateway(),
    )

    with pytest.raises(ContractError, match="no exact constraints"):
        packager.package(
            git_sources=sources,
            wheel_requests=wheels,
            kind_profile="eval",
            output_root=(tmp_path / "work").absolute(),
        )
    assert git_gateway.calls == []

    with pytest.raises(ContractError, match="must match"):
        ImmutableEnvironmentPackager(
            cache_roots=_cache_roots(tmp_path / "mismatch"),
            kind_constraints={
                "eval": KindDependencyConstraints(
                    "online-rl",
                    "shared-runtime==1.0.0\n",
                )
            },
        )


def test_requires_separate_absolute_cache_and_output_roots(
    tmp_path: Path,
) -> None:
    absolute = tmp_path.absolute()
    with pytest.raises(ContractError, match="absolute"):
        EnvironmentPackagerCacheRoots(
            Path("git"),
            absolute / "wheels",
            absolute / "dependencies",
        )
    with pytest.raises(ContractError, match="distinct"):
        EnvironmentPackagerCacheRoots(
            absolute / "shared",
            absolute / "shared",
            absolute / "dependencies",
        )
    with pytest.raises(ContractError, match="must not overlap"):
        EnvironmentPackagerCacheRoots(
            absolute / "cache",
            absolute / "cache/wheels",
            absolute / "dependencies",
        )

    sources, wheels = _one_repository_requests()
    with pytest.raises(ContractError, match="output root must be absolute"):
        _packager(
            tmp_path,
            git_gateway=_git_gateway(),
            dependency_gateway=FakeDependencyGateway(),
        ).package(
            git_sources=sources,
            wheel_requests=wheels,
            kind_profile="online-rl",
            output_root=Path("relative"),
        )


def test_rejects_unused_or_missing_git_roots_before_fetch(
    tmp_path: Path,
) -> None:
    git_gateway = _git_gateway()
    sources, wheels = _one_repository_requests()
    extra = GitSourceRequest(
        repository=_OTHER_REPOSITORY,
        revision=_REVISION,
        subdirectories=(".",),
    )
    packager = _packager(
        tmp_path,
        git_gateway=git_gateway,
        dependency_gateway=FakeDependencyGateway(),
    )

    with pytest.raises(ContractError, match="exactly cover"):
        packager.package(
            git_sources=(*sources, extra),
            wheel_requests=wheels,
            kind_profile="online-rl",
            output_root=(tmp_path / "work-extra").absolute(),
        )
    with pytest.raises(ContractError, match="exactly cover"):
        packager.package(
            git_sources=sources,
            wheel_requests=wheels[:-1],
            kind_profile="online-rl",
            output_root=(tmp_path / "work-missing").absolute(),
        )
    assert git_gateway.calls == []


def test_rejects_work_root_overlapping_a_persistent_cache(
    tmp_path: Path,
) -> None:
    sources, wheels = _one_repository_requests()
    packager = _packager(
        tmp_path,
        git_gateway=_git_gateway(),
        dependency_gateway=FakeDependencyGateway(),
    )
    with pytest.raises(ContractError, match="must not overlap"):
        packager.package(
            git_sources=sources,
            wheel_requests=wheels,
            kind_profile="online-rl",
            output_root=(tmp_path / "cache/git/work").absolute(),
        )
