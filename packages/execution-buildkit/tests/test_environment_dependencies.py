from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import ContractError
from posttrain_execution_buildkit import (
    DependencyResolutionError,
    EnvironmentWheelLock,
    ImmutableEnvironmentDependencyCompiler,
    KindDependencyConstraints,
    LockedEnvironmentWheel,
    MaterializedEnvironmentWheel,
    MaterializedEnvironmentWheels,
    UvDependencyCompileCli,
)


class FakeCompileGateway:
    def __init__(self, output: str | None) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []
        self.requirement_inputs: list[str] = []
        self.error: Exception | None = None
        self.mutate: Path | None = None

    def compile(self, **arguments: object) -> None:
        self.calls.append(arguments)
        requirements = arguments["requirements"]
        assert isinstance(requirements, Path)
        self.requirement_inputs.append(requirements.read_text())
        if self.error is not None:
            raise self.error
        if self.mutate is not None:
            self.mutate.write_bytes(b"drift")
        if self.output is not None:
            output = arguments["output"]
            assert isinstance(output, Path)
            output.write_text(self.output, encoding="utf-8")


def _digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _wheels(
    tmp_path: Path,
    packages: Mapping[str, tuple[str, bytes]] | None = None,
) -> MaterializedEnvironmentWheels:
    selected = packages or {
        "alpha-env": ("alpha_env-1.0.0-py3-none-any.whl", b"alpha"),
        "beta-env": ("beta_env-2.0.0-py3-none-any.whl", b"beta"),
    }
    materialized: list[MaterializedEnvironmentWheel] = []
    for package, (filename, contents) in sorted(selected.items()):
        digest = _digest(contents)
        root = (tmp_path / "wheels" / digest).absolute()
        root.mkdir(parents=True, exist_ok=True)
        path = root / filename
        path.write_bytes(contents)
        lock = LockedEnvironmentWheel(
            package=package,
            repository="https://github.com/example/environments",
            revision="a" * 40,
            subdirectory=f"environments/{package}",
            source_tree_digest="b" * 64,
            wheel_filename=filename,
            wheel_sha256=digest,
            wheel_size_bytes=len(contents),
        )
        materialized.append(MaterializedEnvironmentWheel(path=path, lock=lock))
    result = tuple(materialized)
    return MaterializedEnvironmentWheels(
        wheels=result,
        lock=EnvironmentWheelLock(packages=tuple(wheel.lock for wheel in result)),
    )


def _combined_output(wheels: MaterializedEnvironmentWheels) -> str:
    alpha, beta = wheels.wheels
    return (
        "urllib3==2.6.3 \\\n"
        f"    --hash=sha256:{'f' * 64} \\\n"
        f"    --hash=sha256:{'e' * 64}\n"
        f"./wheels/environments/{beta.lock.wheel_filename} \\\n"
        f"    --hash=sha256:{beta.lock.wheel_sha256}\n"
        f"./wheels/environments/{alpha.lock.wheel_filename} \\\n"
        f"    --hash=sha256:{alpha.lock.wheel_sha256}\n"
    )


def test_compiles_all_wheels_together_into_a_portable_deterministic_lock(
    tmp_path: Path,
) -> None:
    wheels = _wheels(tmp_path)
    first_gateway = FakeCompileGateway(_combined_output(wheels))
    first = ImmutableEnvironmentDependencyCompiler(
        output_root=(tmp_path / "dependencies").absolute(),
        gateway=first_gateway,
    ).compile(
        wheels,
        KindDependencyConstraints(
            "online-rl",
            "urllib3==2.6.3\r\n",
            ("Verifiers",),
        ),
    )

    beta = wheels.wheels[1]
    alpha = wheels.wheels[0]
    second_gateway = FakeCompileGateway(
        f"./wheels/environments/{alpha.lock.wheel_filename} \\\n"
        f"    --hash=sha256:{alpha.lock.wheel_sha256}\n"
        f"./wheels/environments/{beta.lock.wheel_filename} \\\n"
        f"    --hash=sha256:{beta.lock.wheel_sha256}\n"
        "urllib3==2.6.3 \\\n"
        f"    --hash=sha256:{'e' * 64} \\\n"
        f"    --hash=sha256:{'f' * 64}\n"
    )
    second = ImmutableEnvironmentDependencyCompiler(
        output_root=(tmp_path / "dependencies").absolute(),
        gateway=second_gateway,
    ).compile(
        wheels,
        KindDependencyConstraints(
            "online-rl",
            "urllib3==2.6.3\n",
            ("verifiers",),
        ),
    )

    assert first == second
    assert first.path.is_file()
    contents = first.path.read_text()
    assert str(tmp_path) not in contents
    assert "./wheels/environments/alpha_env-1.0.0-py3-none-any.whl" in contents
    assert "./wheels/environments/beta_env-2.0.0-py3-none-any.whl" in contents
    assert first.lock.as_dict() == {
        "schema": "posttrain.environment-dependency-lock.v2",
        "kind_profile": "online-rl",
        "kind_constraints_sha256": hashlib.sha256(b"urllib3==2.6.3\n").hexdigest(),
        "constraint_profile_sha256": KindDependencyConstraints(
            "online-rl",
            "urllib3==2.6.3\n",
            ("verifiers",),
        ).digest,
        "provided_packages": ["verifiers"],
        "role": "control",
        "environment_wheel_lock_sha256": wheels.lock.digest,
        "python_version": "3.12",
        "python_platform": "x86_64-unknown-linux-gnu",
        "python_executable": "/opt/posttrain/venv/bin/python",
        "wheel_directory": "wheels/environments",
        "requirements_filename": "environment-dependencies.lock.txt",
        "requirements_sha256": hashlib.sha256(first.path.read_bytes()).hexdigest(),
        "requirements_size_bytes": first.path.stat().st_size,
        "requirement_count": 3,
    }
    assert len(first.lock.digest) == 64
    call = first_gateway.calls[0]
    assert call["python_version"] == "3.12"
    assert call["python_platform"] == "x86_64-unknown-linux-gnu"
    assert call["provided_packages"] == ("verifiers",)
    assert first_gateway.requirement_inputs[0].splitlines() == [
        "./wheels/environments/alpha_env-1.0.0-py3-none-any.whl",
        "./wheels/environments/beta_env-2.0.0-py3-none-any.whl",
    ]


def test_propagates_combined_resolution_conflicts_without_retaining_a_lock(
    tmp_path: Path,
) -> None:
    gateway = FakeCompileGateway(None)
    gateway.error = DependencyResolutionError("incompatible requirements")

    with pytest.raises(DependencyResolutionError, match="incompatible"):
        ImmutableEnvironmentDependencyCompiler(
            output_root=(tmp_path / "dependencies").absolute(),
            gateway=gateway,
        ).compile(_wheels(tmp_path), KindDependencyConstraints("eval", "foo==1\n"))

    assert not list((tmp_path / "dependencies").glob("*/environment-dependencies.lock.txt"))


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("unsafe==1.0\n", "sha256 hash"),
        (
            f"unsafe>=1.0 \\\n    --hash=sha256:{'a' * 64}\n",
            "unpinned",
        ),
        (
            "unsafe @ git+https://github.com/example/unsafe@" + "a" * 40 + "\n",
            "sha256 hash|mutable",
        ),
        (
            f"/tmp/unsafe.whl \\\n    --hash=sha256:{'a' * 64}\n",
            "mutable or non-portable",
        ),
        (
            f'unsafe==1.0 ; implementation_name == "/tmp/python" \\\n    --hash=sha256:{"a" * 64}\n',
            "non-portable path",
        ),
        (
            "unsafe==1.0 \\\n    --index-url=https://user:password@example.test/simple\n",
            "unsupported or unhashed",
        ),
    ],
)
def test_rejects_unhashed_mutable_and_nonportable_resolver_output(
    tmp_path: Path,
    output: str,
    message: str,
) -> None:
    wheels = _wheels(
        tmp_path,
        {"alpha-env": ("alpha_env-1.0.0-py3-none-any.whl", b"alpha")},
    )
    gateway = FakeCompileGateway(
        output
        + f"./wheels/environments/{wheels.wheels[0].lock.wheel_filename} \\\n"
        + f"    --hash=sha256:{wheels.wheels[0].lock.wheel_sha256}\n"
    )
    with pytest.raises(DependencyResolutionError, match=message):
        ImmutableEnvironmentDependencyCompiler(
            output_root=(tmp_path / "dependencies").absolute(),
            gateway=gateway,
        ).compile(wheels, KindDependencyConstraints("eval", "foo==1\n"))


def test_rejects_omitted_unknown_or_mismatched_wheels(tmp_path: Path) -> None:
    wheels = _wheels(
        tmp_path,
        {"alpha-env": ("alpha_env-1.0.0-py3-none-any.whl", b"alpha")},
    )
    filename = wheels.wheels[0].lock.wheel_filename
    for output, message in (
        (
            f"other==1.0 \\\n    --hash=sha256:{'c' * 64}\n",
            "omitted",
        ),
        (
            f"./wheels/environments/unknown.whl \\\n    --hash=sha256:{'c' * 64}\n",
            "unselected",
        ),
        (
            f"./wheels/environments/{filename} \\\n    --hash=sha256:{'c' * 64}\n",
            "does not match",
        ),
    ):
        with pytest.raises(DependencyResolutionError, match=message):
            ImmutableEnvironmentDependencyCompiler(
                output_root=(tmp_path / f"dependencies-{message}").absolute(),
                gateway=FakeCompileGateway(output),
            ).compile(wheels, KindDependencyConstraints("eval", "foo==1\n"))


@pytest.mark.parametrize(
    "contents",
    [
        "--index-url https://user:password@example.test/simple\nfoo==1\n",
        "--extra-index-url https://example.test/simple?token=value\nfoo==1\n",
        "-r inherited.txt\n",
        "--constraint inherited.txt\n",
        "/tmp/constraints.txt\n",
        "foo @ file:///tmp/foo.whl\n",
        "foo @ git+https://github.com/example/foo@main\n",
    ],
)
def test_rejects_secret_nonportable_or_mutable_kind_constraints(
    contents: str,
) -> None:
    with pytest.raises(ContractError):
        KindDependencyConstraints("online-rl", contents)


def test_provided_packages_are_normalized_validated_and_digest_bound() -> None:
    selected = KindDependencyConstraints(
        "online-rl",
        "foo==1\n",
        ("Verifiers", "typing_extensions"),
    )

    assert selected.provided_packages == ("typing-extensions", "verifiers")
    assert selected.constraints_sha256 == hashlib.sha256(b"foo==1\n").hexdigest()
    assert (
        selected.digest
        != KindDependencyConstraints(
            "online-rl",
            "foo==1\n",
        ).digest
    )

    for packages in (
        ("verifiers[dev]",),
        ("https://example.test/verifiers",),
        ("verifiers", "Verifiers"),
    ):
        with pytest.raises(ContractError, match="provided packages"):
            KindDependencyConstraints("online-rl", "foo==1\n", packages)


def test_constraint_package_names_are_normalized_and_unique() -> None:
    selected = KindDependencyConstraints(
        "online-rl-verl-py313",
        (
            "antlr4-python3-runtime==4.9.3\n"
            "cffi==2.0.0 ; implementation_name == 'pypy'\n"
            "Verifiers @ git+https://github.com/example/verifiers.git@"
            f"{'a' * 40}\n"
        ),
    )

    assert selected.constrained_packages == (
        "antlr4-python3-runtime",
        "cffi",
        "verifiers",
    )

    with pytest.raises(ContractError, match="at most once"):
        _ = KindDependencyConstraints(
            "online-rl-verl-py313",
            "foo-bar==1\nfoo_bar==1\n",
        ).constrained_packages


def test_rejects_wheel_drift_before_and_after_resolution(tmp_path: Path) -> None:
    wheels = _wheels(
        tmp_path,
        {"alpha-env": ("alpha_env-1.0.0-py3-none-any.whl", b"alpha")},
    )
    wheel = wheels.wheels[0]
    wheel.path.write_bytes(b"before")
    with pytest.raises(ContractError, match="digest|size"):
        ImmutableEnvironmentDependencyCompiler(
            output_root=(tmp_path / "before").absolute(),
            gateway=FakeCompileGateway(""),
        ).compile(wheels, KindDependencyConstraints("eval", "foo==1\n"))

    wheels = _wheels(
        tmp_path / "after",
        {"alpha-env": ("alpha_env-1.0.0-py3-none-any.whl", b"alpha")},
    )
    wheel = wheels.wheels[0]
    gateway = FakeCompileGateway(
        f"./wheels/environments/{wheel.lock.wheel_filename} \\\n    --hash=sha256:{wheel.lock.wheel_sha256}\n"
    )
    gateway.mutate = wheel.path
    with pytest.raises(ContractError, match="digest|size"):
        ImmutableEnvironmentDependencyCompiler(
            output_root=(tmp_path / "after-dependencies").absolute(),
            gateway=gateway,
        ).compile(wheels, KindDependencyConstraints("eval", "foo==1\n"))


def test_uv_gateway_uses_one_non_shell_compile_with_fixed_safety_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "posttrain_execution_buildkit.environment_dependencies.subprocess.run",
        fake_run,
    )
    root = tmp_path.absolute()
    requirements = root / "requirements.in"
    constraints = root / "constraints.txt"
    output = root / "requirements.txt"

    UvDependencyCompileCli("/opt/uv").compile(
        requirements=requirements,
        constraints=constraints,
        output=output,
        working_directory=root,
        python_version="3.12",
        python_platform="x86_64-unknown-linux-gnu",
        provided_packages=("datasets", "verifiers"),
    )

    arguments = observed["arguments"]
    assert isinstance(arguments, list)
    assert arguments[:3] == ["/opt/uv", "pip", "compile"]
    assert "--generate-hashes" in arguments
    assert "--no-config" in arguments
    assert arguments[arguments.index("--index-strategy") + 1] == "unsafe-best-match"
    assert arguments[arguments.index("--python-version") + 1] == "3.12"
    assert arguments[arguments.index("--python-platform") + 1] == "x86_64-unknown-linux-gnu"
    assert [arguments[index + 1] for index, argument in enumerate(arguments) if argument == "--no-emit-package"] == [
        "datasets",
        "verifiers",
    ]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert "shell" not in kwargs
    assert kwargs["cwd"] == root


def test_uv_gateway_uses_only_explicit_credential_free_index_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("UV_INDEX_URL", "https://ambient-user:ambient-secret@example.test/simple")

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "posttrain_execution_buildkit.environment_dependencies.subprocess.run",
        fake_run,
    )
    root = tmp_path.absolute()
    UvDependencyCompileCli(
        "/opt/uv",
        index_environment={
            "UV_INDEX_URL": "https://pypi.example.test/simple/",
            "UV_INDEX_USERNAME": "reader",
            "UV_INDEX_PASSWORD": "secret",
        },
    ).compile(
        requirements=root / "requirements.in",
        constraints=root / "constraints.txt",
        output=root / "requirements.txt",
        working_directory=root,
        python_version="3.12",
        python_platform="x86_64-unknown-linux-gnu",
        provided_packages=(),
    )

    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["UV_INDEX_URL"] == "https://pypi.example.test/simple/"
    assert environment["UV_INDEX_USERNAME"] == "reader"
    assert environment["UV_INDEX_PASSWORD"] == "secret"
    assert "ambient-secret" not in str(observed["arguments"])


@pytest.mark.parametrize(
    "index_environment",
    [
        {"PIP_INDEX_URL": "https://example.test/simple/"},
        {"UV_INDEX_URL": "https://user:secret@example.test/simple/"},
        {"UV_INDEX_URL": "https://example.test/simple/?token=secret"},
    ],
)
def test_uv_gateway_rejects_unsupported_or_secret_bearing_index_bindings(
    index_environment: dict[str, str],
) -> None:
    with pytest.raises(ContractError, match="dependency-index|unsupported"):
        UvDependencyCompileCli(index_environment=index_environment)


def test_uv_gateway_reports_conflict_without_echoing_resolver_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(_arguments, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stderr="https://user:password@example.test/simple conflict",
            stdout="/private/path",
        )

    monkeypatch.setattr(
        "posttrain_execution_buildkit.environment_dependencies.subprocess.run",
        fake_run,
    )
    root = tmp_path.absolute()
    with pytest.raises(DependencyResolutionError) as caught:
        UvDependencyCompileCli().compile(
            requirements=root / "requirements.in",
            constraints=root / "constraints.txt",
            output=root / "requirements.txt",
            working_directory=root,
            python_version="3.12",
            python_platform="x86_64-unknown-linux-gnu",
            provided_packages=(),
        )

    assert "password" not in str(caught.value)
    assert "/private/path" not in str(caught.value)
