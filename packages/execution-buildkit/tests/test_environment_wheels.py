from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import ContractError
from posttrain.execution_pack import ProjectEnvironmentSourceRequest
from posttrain_execution_buildkit import (
    EnvironmentWheelRequest,
    GitSourceLock,
    ImmutableEnvironmentWheelBuilder,
    LockedGitSource,
    LockedGitSubdirectory,
    MaterializedGitSource,
    MaterializedGitSources,
    UvWheelBuildCli,
)
from posttrain_execution_buildkit.git_sources import _tree_digest

_REPOSITORY = "https://github.com/PrimeIntellect-ai/verifiers"
_REVISION = "a" * 40


class FakeWheelGateway:
    def __init__(self, outputs: Mapping[str, Mapping[str, bytes]]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[Path, Path]] = []
        self.mutate_source = False

    def build(self, package_root: Path, output_directory: Path) -> None:
        self.calls.append((package_root, output_directory))
        for filename, content in self.outputs[package_root.name].items():
            (output_directory / filename).write_bytes(content)
        if self.mutate_source:
            (package_root / "generated.txt").write_text("drift")


def _sources(tmp_path: Path) -> MaterializedGitSources:
    root = (tmp_path / "source").absolute()
    (root / ".git").mkdir(parents=True)
    for subdirectory, name in (
        ("environments/gsm8k_v1", "gsm8k-v1"),
        ("environments/reverse_text_v1", "reverse-text-v1"),
    ):
        package_root = root / subdirectory
        package_root.mkdir(parents=True)
        (package_root / "pyproject.toml").write_text(f"[project]\nname = '{name}'\nversion = '1.0.0'\n")
        (package_root / "environment.py").write_text("def load(): ...\n")
    locked_subdirectories = tuple(
        LockedGitSubdirectory(
            path=subdirectory,
            tree_digest=_tree_digest(root / subdirectory),
        )
        for subdirectory in (
            "environments/gsm8k_v1",
            "environments/reverse_text_v1",
        )
    )
    locked_source = LockedGitSource(
        repository=_REPOSITORY,
        revision=_REVISION,
        source_tree_digest=_tree_digest(root),
        subdirectories=locked_subdirectories,
    )
    materialized = MaterializedGitSource(root=root, lock=locked_source)
    return MaterializedGitSources(
        sources=(materialized,),
        lock=GitSourceLock(sources=(locked_source,)),
    )


def _requests() -> list[EnvironmentWheelRequest]:
    return [
        EnvironmentWheelRequest(
            package="reverse-text-v1",
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectory="environments/reverse_text_v1",
        ),
        EnvironmentWheelRequest(
            package="gsm8k-v1",
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectory="environments/gsm8k_v1",
        ),
    ]


def test_builds_multiple_wheels_from_one_checkout_with_deterministic_lock(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    gateway = FakeWheelGateway(
        {
            "gsm8k_v1": {"gsm8k_v1-1.0.0-py3-none-any.whl": b"gsm8k-wheel"},
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"reverse-wheel"},
        }
    )
    builder = ImmutableEnvironmentWheelBuilder(
        output_root=(tmp_path / "wheels").absolute(),
        gateway=gateway,
    )

    result = builder.build(sources, _requests())

    assert [wheel.lock.package for wheel in result.wheels] == [
        "gsm8k-v1",
        "reverse-text-v1",
    ]
    assert len({root for root, _output in gateway.calls}) == 2
    assert all(wheel.path.is_file() for wheel in result.wheels)
    assert all(wheel.path.parent.name == wheel.lock.wheel_sha256 for wheel in result.wheels)
    assert str(tmp_path) not in result.lock.to_json()
    assert result.lock.as_dict()["schema"] == "posttrain.environment-wheel-lock.v1"
    assert len(result.lock.digest) == 64


def test_deduplicates_exact_requests(tmp_path: Path) -> None:
    gateway = FakeWheelGateway(
        {
            "gsm8k_v1": {"gsm8k_v1-1.0.0-py3-none-any.whl": b"wheel"},
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"unused"},
        }
    )
    builder = ImmutableEnvironmentWheelBuilder(
        output_root=(tmp_path / "wheels").absolute(),
        gateway=gateway,
    )
    request = _requests()[1]

    result = builder.build(_sources(tmp_path), [request, request])

    assert len(result.wheels) == 1
    assert len(gateway.calls) == 1


def test_builds_a_project_snapshot_without_git_metadata(tmp_path: Path) -> None:
    root = (tmp_path / "toy_env").absolute()
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "toy-env"\nversion = "1.0.0"\n')
    gateway = FakeWheelGateway({"toy_env": {"toy_env-1.0.0-py3-none-any.whl": b"wheel"}})
    builder = ImmutableEnvironmentWheelBuilder(output_root=(tmp_path / "wheels").absolute(), gateway=gateway)
    request = ProjectEnvironmentSourceRequest("toy-env", "environments/toy_env", _tree_digest(root))

    result = builder.build_project_sources({request.path: root}, (request,))

    lock = result.wheels[0].lock
    assert lock.source_kind == "project-path"
    assert lock.repository is None and lock.revision is None
    assert lock.project_path == request.path


def test_accepts_uv_generated_output_gitignore(tmp_path: Path) -> None:
    gateway = FakeWheelGateway(
        {
            "gsm8k_v1": {
                ".gitignore": b"*",
                "gsm8k_v1-1.0.0-py3-none-any.whl": b"wheel",
            },
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"unused"},
        }
    )

    result = ImmutableEnvironmentWheelBuilder(
        output_root=(tmp_path / "wheels").absolute(),
        gateway=gateway,
    ).build(_sources(tmp_path), [_requests()[1]])

    assert len(result.wheels) == 1


@pytest.mark.parametrize(
    "requests",
    [
        [
            EnvironmentWheelRequest(
                "same",
                _REPOSITORY,
                _REVISION,
                "environments/gsm8k_v1",
            ),
            EnvironmentWheelRequest(
                "same",
                _REPOSITORY,
                _REVISION,
                "environments/reverse_text_v1",
            ),
        ],
        [
            EnvironmentWheelRequest(
                "first",
                _REPOSITORY,
                _REVISION,
                "environments/gsm8k_v1",
            ),
            EnvironmentWheelRequest(
                "second",
                _REPOSITORY,
                _REVISION,
                "environments/gsm8k_v1",
            ),
        ],
    ],
)
def test_rejects_package_and_source_root_conflicts(
    tmp_path: Path,
    requests: list[EnvironmentWheelRequest],
) -> None:
    with pytest.raises(ContractError, match="multiple source roots|conflicting"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "wheels").absolute(),
            gateway=FakeWheelGateway({}),
        ).build(_sources(tmp_path), requests)


def test_rejects_unselected_or_missing_package_root(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    request = EnvironmentWheelRequest(
        package="not-selected",
        repository=_REPOSITORY,
        revision=_REVISION,
        subdirectory="environments/not_selected",
    )
    with pytest.raises(ContractError, match="was not selected"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "wheels").absolute(),
            gateway=FakeWheelGateway({}),
        ).build(sources, [request])

    pyproject = sources.sources[0].root / "environments/gsm8k_v1/pyproject.toml"
    pyproject.unlink()
    locked = sources.sources[0].lock
    drifted_source = MaterializedGitSource(
        root=sources.sources[0].root,
        lock=LockedGitSource(
            repository=locked.repository,
            revision=locked.revision,
            source_tree_digest=_tree_digest(sources.sources[0].root),
            subdirectories=(
                LockedGitSubdirectory(
                    path="environments/gsm8k_v1",
                    tree_digest=_tree_digest(sources.sources[0].root / "environments/gsm8k_v1"),
                ),
            ),
        ),
    )
    drifted_sources = MaterializedGitSources(
        sources=(drifted_source,),
        lock=GitSourceLock(sources=(drifted_source.lock,)),
    )
    with pytest.raises(ContractError, match="no regular pyproject"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "other-wheels").absolute(),
            gateway=FakeWheelGateway({}),
        ).build(drifted_sources, [_requests()[1]])


def test_rejects_source_drift_before_build(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    (sources.sources[0].root / "environments/gsm8k_v1/environment.py").write_text("changed")
    gateway = FakeWheelGateway(
        {
            "gsm8k_v1": {"gsm8k_v1-1.0.0-py3-none-any.whl": b"wheel"},
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"unused"},
        }
    )

    with pytest.raises(ContractError, match="filesystem drift"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "wheels").absolute(),
            gateway=gateway,
        ).build(sources, [_requests()[1]])

    assert gateway.calls == []


@pytest.mark.parametrize(
    "outputs",
    [
        {},
        {
            "one-1.0.0-py3-none-any.whl": b"one",
            "two-1.0.0-py3-none-any.whl": b"two",
        },
        {
            "one-1.0.0-py3-none-any.whl": b"one",
            "unexpected.txt": b"extra",
        },
    ],
)
def test_rejects_unbounded_or_ambiguous_build_output(
    tmp_path: Path,
    outputs: Mapping[str, bytes],
) -> None:
    gateway = FakeWheelGateway(
        {
            "gsm8k_v1": outputs,
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"unused"},
        }
    )
    with pytest.raises(ContractError, match="exactly one"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "wheels").absolute(),
            gateway=gateway,
        ).build(_sources(tmp_path), [_requests()[1]])


def test_rejects_oversized_wheel_and_isolates_source_from_build_mutation(
    tmp_path: Path,
) -> None:
    oversized = FakeWheelGateway(
        {
            "gsm8k_v1": {"gsm8k_v1-1.0.0-py3-none-any.whl": b"too-large"},
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"unused"},
        }
    )
    with pytest.raises(ContractError, match="exceeds 4 bytes"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "large-wheels").absolute(),
            gateway=oversized,
            max_wheel_bytes=4,
        ).build(_sources(tmp_path / "large"), [_requests()[1]])

    mutating = FakeWheelGateway(
        {
            "gsm8k_v1": {"gsm8k_v1-1.0.0-py3-none-any.whl": b"wheel"},
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"unused"},
        }
    )
    mutating.mutate_source = True
    sources = _sources(tmp_path / "mutating")
    result = ImmutableEnvironmentWheelBuilder(
        output_root=(tmp_path / "mutating-wheels").absolute(),
        gateway=mutating,
    ).build(sources, [_requests()[1]])

    assert len(result.wheels) == 1
    assert not (sources.sources[0].root / "environments/gsm8k_v1/generated.txt").exists()


def test_rejects_invalid_limits_and_package_identity(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="identity"):
        EnvironmentWheelRequest(
            package="../escape",
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectory="environments/gsm8k_v1",
        )
    with pytest.raises(ContractError, match="limits"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "wheels").absolute(),
            gateway=FakeWheelGateway({}),
            max_packages=0,
        )


def test_rejects_declared_and_built_package_name_mismatches(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path / "declared")
    pyproject = sources.sources[0].root / "environments/gsm8k_v1/pyproject.toml"
    pyproject.write_text("[project]\nname = 'another-package'\nversion = '1.0.0'\n")
    locked = sources.sources[0].lock
    selected = tuple(
        LockedGitSubdirectory(
            path=item.path,
            tree_digest=_tree_digest(sources.sources[0].root.joinpath(*item.path.split("/"))),
        )
        for item in locked.subdirectories
    )
    relocked = LockedGitSource(
        repository=locked.repository,
        revision=locked.revision,
        source_tree_digest=_tree_digest(sources.sources[0].root),
        subdirectories=selected,
    )
    relocked_sources = MaterializedGitSources(
        sources=(MaterializedGitSource(root=sources.sources[0].root, lock=relocked),),
        lock=GitSourceLock(sources=(relocked,)),
    )
    with pytest.raises(ContractError, match="does not match pyproject"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "declared-wheels").absolute(),
            gateway=FakeWheelGateway({}),
        ).build(relocked_sources, [_requests()[1]])

    gateway = FakeWheelGateway(
        {
            "gsm8k_v1": {"wrong_name-1.0.0-py3-none-any.whl": b"wheel"},
            "reverse_text_v1": {"reverse_text_v1-1.0.0-py3-none-any.whl": b"unused"},
        }
    )
    with pytest.raises(ContractError, match="filename does not match"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "filename-wheels").absolute(),
            gateway=gateway,
        ).build(_sources(tmp_path / "filename"), [_requests()[1]])


def test_rejects_distribution_names_that_collide_after_normalization(
    tmp_path: Path,
) -> None:
    requests = [
        EnvironmentWheelRequest(
            "same-name",
            _REPOSITORY,
            _REVISION,
            "environments/gsm8k_v1",
        ),
        EnvironmentWheelRequest(
            "same_name",
            _REPOSITORY,
            _REVISION,
            "environments/reverse_text_v1",
        ),
    ]
    with pytest.raises(ContractError, match="after normalization"):
        ImmutableEnvironmentWheelBuilder(
            output_root=(tmp_path / "wheels").absolute(),
            gateway=FakeWheelGateway({}),
        ).build(_sources(tmp_path), requests)


def test_uv_wheel_gateway_uses_sanitized_non_shell_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("UV_INDEX_URL", "https://user:secret@example.test")

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=1,
            stdout="https://user:secret@example.test",
            stderr="private-path",
        )

    monkeypatch.setattr(
        "posttrain_execution_buildkit.environment_wheels.subprocess.run",
        fake_run,
    )
    root = tmp_path.absolute()
    with pytest.raises(RuntimeError) as caught:
        UvWheelBuildCli("/opt/uv").build(root / "package", root / "output")

    arguments = observed["arguments"]
    assert isinstance(arguments, list)
    assert arguments[:2] == ["/opt/uv", "build"]
    assert "--no-config" in arguments
    assert "--no-sources" in arguments
    assert "--no-python-downloads" in arguments
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert "shell" not in kwargs
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert "UV_INDEX_URL" not in environment
    assert "secret" not in str(caught.value)
    assert "private-path" not in str(caught.value)
