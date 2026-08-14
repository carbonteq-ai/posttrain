from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from posttrain.catalog import ProjectExecutionDefaults, load_project_layout
from posttrain.common import ContractError
from posttrain.execution import (
    ExecutionEvidenceSource,
    ExecutionSubmission,
    ExecutionSubmissionStore,
)
from posttrain.runtime_images.manifest import load_manifest as _load_manifest
from posttrain_cli.execution_config import (
    ADMISSION_ROOT_ENVIRONMENT_VARIABLE,
    REGISTRY_ENVIRONMENT_VARIABLE,
    TRUST_BUNDLE_ENVIRONMENT_VARIABLE,
    ExecutionOverrides,
    LocalExecutionConfig,
    LocalProviderBinding,
    MachineConfig,
    MachineServicesBinding,
    derived_local_registry,
    load_execution_environment,
    load_local_execution_config,
    provider_binding_fingerprint,
    resolve_admission_state_root,
    resolve_execution_settings,
    resolve_job_builder,
    resolve_trust_bundle,
)
from posttrain_cli.execution_planning import _with_registry_override
from posttrain_cli.execution_provider import (
    create_execution_provider,
    evidence_source_for_project,
    evidence_source_for_run,
    provider_source_for_project,
)


@pytest.fixture(autouse=True)
def _candidate_manifest_for_configuration_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep configuration tests independent of candidate image publication.

    These tests exercise registry derivation and machine precedence.  A source
    candidate deliberately retains the last published image manifest until its
    matching OCI graph is built, which must make the *real* consumer manifest
    loader fail closed.  Unit tests here therefore use the structural manifest
    only; strict manifest integrity is covered by runtime-image/release tests.
    """

    manifest = _load_manifest(verify_locks=False)
    monkeypatch.setattr("posttrain_cli.execution_config.load_manifest", lambda: manifest)


def _candidate_manifest():
    return _load_manifest(verify_locks=False)


def _layout(tmp_path: Path):
    control = tmp_path / ".posttrain"
    control.mkdir()
    (control / "project.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                'project_id = "execution-config-tests"',
                "catalog_overlays = []",
                'state = "state"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return load_project_layout(tmp_path)


def _write_runtime_environment(path: Path, values: str) -> None:
    path.write_text(values, encoding="utf-8")
    path.chmod(0o600)


def test_explicit_registry_override_changes_only_the_publication_destination(tmp_path: Path) -> None:
    registry = derived_local_registry()
    configuration = LocalExecutionConfig(path=tmp_path / "config.toml", registry=registry)

    overridden = _with_registry_override(configuration, "registry.example/team")

    assert overridden.registry is not None
    assert overridden.registry.repository == "registry.example/team/posttrain-job"
    assert overridden.registry.universal_image == registry.universal_image
    assert overridden.registry.kind_images == registry.kind_images
    assert overridden.registry.constraint_profiles == registry.constraint_profiles
    assert configuration.registry == registry


def test_project_runtime_environment_is_authoritative_over_shell_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(
        tmp_path / "posttrain.env",
        "POSTTRAIN_REGISTRY=registry.project.example/posttrain\n",
    )
    monkeypatch.setenv("POSTTRAIN_REGISTRY", "registry.shell.example/posttrain")

    configuration = load_local_execution_config(layout)

    assert configuration.environment_file == (tmp_path / "posttrain.env").resolve()
    assert configuration.registry is not None
    assert configuration.registry.repository == "registry.project.example/posttrain/posttrain-job"


def test_machine_keeps_provider_ownership_but_allows_project_candidate_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(
        tmp_path / "posttrain.env",
        "POSTTRAIN_REGISTRY=registry.machine.example/posttrain\n",
    )
    lock = layout.state / "candidate.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("candidate receipt\n", encoding="utf-8")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    execution = layout.state / "execution.toml"
    execution.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[registry]",
                'repository = "registry.candidate.example/posttrain-job"',
                "",
                "[registry.kind_images]",
                'supervised = "registry.candidate.example/posttrain-kind-supervised@sha256:' + "a" * 64 + '"',
                "",
                "[registry.constraint_profiles.supervised]",
                'path = "candidate.lock"',
                f'sha256 = "{digest}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    execution.chmod(0o600)
    machine = MachineConfig(
        name="test-machine",
        path=tmp_path / "machine.toml",
        projects=(),
        defaults=ExecutionOverrides(),
        local=LocalProviderBinding(),
        dstack=None,
        tracking=None,
        huggingface=None,
        services=MachineServicesBinding(),
        credentials={},
    )
    monkeypatch.setattr("posttrain_cli.execution_config.load_machine_config", lambda: machine)

    configuration = load_local_execution_config(layout)

    assert configuration.machine is machine
    assert configuration.registry is not None
    assert configuration.registry.repository == "registry.candidate.example/posttrain-job"
    assert configuration.registry.kind_images["supervised"].value.endswith("sha256:" + "a" * 64)


def test_explicit_runtime_environment_replaces_the_project_file(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(
        tmp_path / "posttrain.env",
        "POSTTRAIN_REGISTRY=registry.project.example/posttrain\n",
    )
    override = tmp_path / "alternate.env"
    _write_runtime_environment(
        override,
        "POSTTRAIN_REGISTRY=registry.override.example/posttrain\n",
    )

    configuration = load_local_execution_config(layout, env_file=override)

    assert configuration.environment_file == override.resolve()
    assert configuration.registry is not None
    assert configuration.registry.repository == "registry.override.example/posttrain/posttrain-job"


def test_run_evidence_locator_survives_project_configuration_drift(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    source = ExecutionEvidenceSource(
        provider="trackio",
        source_id="trackio-original",
        project="original-project",
        endpoint="https://original-trackio.example",
    )
    ExecutionSubmissionStore(layout.state).save(
        ExecutionSubmission(
            run_id="run-drift",
            provider="dstack",
            provider_id="provider-run",
            idempotency_key="run-drift-attempt-1",
            job_image=f"registry.lan/posttrain@sha256:{'a' * 64}",
            submitted_at=datetime.now(UTC),
            evidence_source=source,
        )
    )
    layout.manifest.write_text(
        layout.manifest.read_text(encoding="utf-8") + '\ntracking = "none"\n',
        encoding="utf-8",
    )
    changed_layout = load_project_layout(layout.root)

    assert evidence_source_for_run(changed_layout, "run-drift") == source


def test_new_run_with_tracking_disabled_stays_disabled_after_config_drift(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    ExecutionSubmissionStore(layout.state).save(
        ExecutionSubmission(
            run_id="run-disabled",
            provider="local-docker",
            provider_id="provider-run",
            idempotency_key="run-disabled-attempt-1",
            job_image=f"registry.lan/posttrain@sha256:{'a' * 64}",
            submitted_at=datetime.now(UTC),
            evidence_source=None,
        )
    )

    with pytest.raises(RuntimeError, match="submitted with tracking disabled"):
        evidence_source_for_run(layout, "run-disabled")


def test_local_configuration_is_mode_checked_and_parsed(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    job_environment = layout.state / "job.env"
    job_environment.write_text(
        "POSTTRAIN_TRACKIO_SERVER_URL=https://trackio.example\n",
        encoding="utf-8",
    )
    job_environment.chmod(0o600)
    local_trust_bundle = layout.state / "local-ca.pem"
    local_trust_bundle.write_text("local ca\n", encoding="utf-8")
    constraints = layout.state / "constraints.txt"
    constraints.write_text("pydantic==2.12.5\n", encoding="utf-8")
    constraints_digest = hashlib.sha256(constraints.read_bytes()).hexdigest()
    backend_constraints = layout.state / "backend-constraints.txt"
    backend_constraints.write_text(
        "antlr4-python3-runtime==4.9.3\nomegaconf==2.3.1\n",
        encoding="utf-8",
    )
    backend_constraints_digest = hashlib.sha256(backend_constraints.read_bytes()).hexdigest()
    image = f"registry.lan/carbonteq/posttrain@sha256:{'1' * 64}"
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'environment_file = "job.env"',
                "",
                "[defaults]",
                'provider = "dstack"',
                'target = "targets/remote@1"',
                "timeout_seconds = 7200",
                "",
                "[providers.local.storage]",
                'run_root = "local-runs"',
                'model_cache = "local-model-cache"',
                "",
                "[providers.local]",
                'canonical_hostname = "POP-OS.LAN."',
                'dns_servers = ["192.0.2.53", "2001:db8::53"]',
                'trust_bundle = "local-ca.pem"',
                "",
                "[providers.dstack]",
                'project = "main"',
                'python = "dstack-venv/bin/python"',
                'environment_file = "dstack.env"',
                'trust_bundle = "/etc/posttrain/trust/internal-ca.pem"',
                "capacity_wait_seconds = 86400",
                "",
                "[providers.dstack.storage]",
                'run_root = "/var/lib/posttrain/runs"',
                'model_cache = "/var/lib/posttrain/cache/huggingface"',
                'compile_cache = "/var/lib/posttrain/cache/compile"',
                "",
                "[registry]",
                'repository = "registry.lan/carbonteq/posttrain"',
                f'universal_image = "{image}"',
                'buildx_builder = "posttrain-builder"',
                'receipt_root = "runtime-builds"',
                'bake_file = "../../../packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job/docker-bake.hcl"',
                'framework_source_root = "../../.."',
                "",
                "[registry.kind_images]",
                *(
                    f'{profile} = "{image}"'
                    for profile in (
                        "supervised",
                        "online-rl-trl-py312",
                        "online-rl-verl-py313",
                        "eval",
                        "serve",
                        "transform",
                    )
                ),
                "",
                *(
                    line
                    for profile in (
                        "supervised",
                        "online-rl-trl-py312",
                        "online-rl-verl-py313",
                        "eval",
                        "serve",
                        "transform",
                    )
                    for line in (
                        f"[registry.constraint_profiles.{profile}]",
                        'path = "constraints.txt"',
                        f'sha256 = "{constraints_digest}"',
                        *(
                            (
                                'backend_path = "backend-constraints.txt"',
                                f'backend_sha256 = "{backend_constraints_digest}"',
                                'backend_source_repository = "https://github.com/carbonteq-ai/verl.git"',
                                f'backend_source_revision = "{"a" * 40}"',
                                f'backend_dependency_lock_sha256 = "{"b" * 64}"',
                            )
                            if profile == "online-rl-verl-py313"
                            else ()
                        ),
                        *(('provided_packages = ["verifiers"]',) if profile in {"online-rl-trl-py312", "eval"} else ()),
                        "",
                    )
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    configuration = load_local_execution_config(layout)

    assert configuration.defaults.provider == "dstack"
    assert configuration.defaults.target == "targets/remote@1"
    assert configuration.dstack is not None
    assert configuration.dstack.project == "main"
    assert configuration.dstack.python == (layout.state / "dstack-venv/bin/python").resolve()
    assert configuration.local is not None
    assert configuration.local.canonical_hostname == "pop-os.lan"
    assert configuration.local.dns_servers == ("192.0.2.53", "2001:db8::53")
    assert configuration.local.storage is not None
    assert configuration.local.storage.run_root == (layout.state / "local-runs").resolve()
    assert configuration.local.trust_bundle == local_trust_bundle.resolve()
    assert configuration.dstack.storage is not None
    assert configuration.dstack.storage.run_root == Path("/var/lib/posttrain/runs")
    assert configuration.dstack.trust_bundle == Path("/etc/posttrain/trust/internal-ca.pem")
    assert configuration.dstack.capacity_wait_seconds == 86_400
    assert configuration.registry is not None
    assert configuration.registry.repository == "registry.lan/carbonteq/posttrain"
    assert configuration.registry.universal_image.value == image
    assert configuration.registry.kind_images["online-rl-trl-py312"].value == image
    eval_constraints = configuration.registry.constraint_profiles["eval"]
    assert eval_constraints.contents_digest == constraints_digest
    assert eval_constraints.provided_packages == ("verifiers",)
    assert eval_constraints.digest != constraints_digest
    verl_constraints = configuration.registry.constraint_profiles["online-rl-verl-py313"]
    assert verl_constraints.backend_contents_digest == backend_constraints_digest
    assert verl_constraints.backend_provided_packages == (
        "antlr4-python3-runtime",
        "omegaconf",
    )
    assert verl_constraints.backend_digest is not None
    assert verl_constraints.backend_source_repository == "https://github.com/carbonteq-ai/verl.git"
    assert verl_constraints.backend_source_revision == "a" * 40
    assert verl_constraints.backend_dependency_lock_digest == "b" * 64
    assert configuration.registry.receipt_root == (layout.state / "runtime-builds").resolve()
    assert configuration.environment_file == job_environment


def test_local_configuration_rejects_group_readable_file(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    path.write_text("schema_version = 1\n", encoding="utf-8")
    path.chmod(0o640)

    with pytest.raises(ContractError, match="group or others"):
        load_local_execution_config(layout)


def test_local_configuration_rejects_missing_local_trust_bundle(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[providers.local]",
                'trust_bundle = "missing.pem"',
                "",
            )
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    with pytest.raises(ContractError, match="trust bundle is missing"):
        load_local_execution_config(layout)


def test_registry_accepts_an_incremental_exact_runtime_variant_set(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    constraints = layout.state / "constraints.txt"
    constraints.write_text("pydantic==2.12.5\n", encoding="utf-8")
    digest = hashlib.sha256(constraints.read_bytes()).hexdigest()
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[registry]",
                'repository = "registry.lan/carbonteq/jobs"',
                f'universal_image = "registry.lan/base@sha256:{"1" * 64}"',
                "",
                "[registry.kind_images]",
                f'supervised = "registry.lan/kind@sha256:{"2" * 64}"',
                "",
                "[registry.constraint_profiles.supervised]",
                'path = "constraints.txt"',
                f'sha256 = "{digest}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    manifest = _candidate_manifest()

    # The declared entry overrides only itself.
    assert loaded.registry.kind_images["supervised"].value == (f"registry.lan/kind@sha256:{'2' * 64}")
    # Every other published variant still resolves, from the release manifest
    # rather than from this file.
    assert set(loaded.registry.kind_images) == set(manifest.kinds)
    assert loaded.registry.kind_images["eval"].value == manifest.reference("eval")


def test_registry_constraint_profile_provided_packages_are_validated_and_digest_bound(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    constraints = layout.state / "constraints.txt"
    constraints.write_text("pydantic==2.12.5\n", encoding="utf-8")
    digest = hashlib.sha256(constraints.read_bytes()).hexdigest()
    image = f"registry.lan/kind@sha256:{'2' * 64}"
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[registry]",
                'repository = "registry.lan/carbonteq/jobs"',
                f'universal_image = "registry.lan/base@sha256:{"1" * 64}"',
                "",
                "[registry.kind_images]",
                f'online-rl-trl-py312 = "{image}"',
                "",
                "[registry.constraint_profiles.online-rl-trl-py312]",
                'path = "constraints.txt"',
                f'sha256 = "{digest}"',
                'provided_packages = ["Verifiers"]',
                "",
            )
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    selected = loaded.registry.constraint_profiles["online-rl-trl-py312"]
    assert selected.contents_digest == digest
    assert selected.provided_packages == ("verifiers",)
    assert selected.digest != digest

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'provided_packages = ["Verifiers"]',
            'provided_packages = ["verifiers", "Verifiers"]',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="provided packages"):
        load_local_execution_config(layout)


def test_registry_requires_matching_image_and_constraint_variant_keys(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    state = layout.state
    state.mkdir(parents=True)
    constraints = state / "constraints.txt"
    constraints.write_text("pydantic==2.12.5\n", encoding="utf-8")
    digest = hashlib.sha256(constraints.read_bytes()).hexdigest()
    image = f"registry.lan/kind@sha256:{'2' * 64}"
    path = state / "execution.toml"
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[registry]",
                'repository = "registry.lan/carbonteq/jobs"',
                f'universal_image = "{image}"',
                "",
                "[registry.kind_images]",
                # A variant the release does not publish has nothing to inherit
                # a constraint profile from, so it must declare its own.
                f'custom-backend = "{image}"',
                "",
                "[registry.constraint_profiles.supervised]",
                'path = "constraints.txt"',
                f'sha256 = "{digest}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    with pytest.raises(ContractError, match="must define the same runtime variants"):
        load_local_execution_config(layout)


def test_registry_rejects_constraint_file_drift(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    state = layout.state
    state.mkdir(parents=True)
    constraints = state / "constraints.txt"
    constraints.write_text("pydantic==2.12.5\n", encoding="utf-8")
    image = f"registry.lan/kind@sha256:{'2' * 64}"
    lines = [
        "schema_version = 1",
        "",
        "[registry]",
        'repository = "registry.lan/carbonteq/jobs"',
        f'universal_image = "{image}"',
        "",
        "[registry.kind_images]",
        *(
            f'{profile} = "{image}"'
            for profile in (
                "supervised",
                "online-rl-trl-py312",
                "online-rl-verl-py313",
                "eval",
                "serve",
                "transform",
            )
        ),
        "",
    ]
    for profile in (
        "supervised",
        "online-rl-trl-py312",
        "online-rl-verl-py313",
        "eval",
        "serve",
        "transform",
    ):
        lines.extend(
            (
                f"[registry.constraint_profiles.{profile}]",
                'path = "constraints.txt"',
                f'sha256 = "{"f" * 64}"',
                "",
            )
        )
    path = state / "execution.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ContractError, match="differs from its exact digest"):
        load_local_execution_config(layout)


def test_resolution_precedence_and_provenance_are_visible() -> None:
    resolved = resolve_execution_settings(
        ProjectExecutionDefaults(
            provider="local",
            target="targets/project@1",
            timeout_seconds=1800,
            environment_names=("TRACKIO_SERVER_URL",),
        ),
        local=ExecutionOverrides(
            provider="dstack",
            target="targets/local-config@1",
            max_attempts=2,
        ),
        cli=ExecutionOverrides(
            target="targets/cli@1",
            priority=9,
        ),
        job=ExecutionOverrides(
            runtime_profile="training/verl@1",
            timeout_seconds=600,
            max_attempts=1,
            priority=0,
            environment_names=(),
        ),
    )

    assert resolved.provider == "dstack"
    assert resolved.target == "targets/cli@1"
    assert resolved.runtime_profile == "training/verl@1"
    assert resolved.timeout_seconds == 1800
    assert resolved.max_attempts == 2
    assert resolved.priority == 9
    assert resolved.environment_names == ("TRACKIO_SERVER_URL",)
    assert resolved.sources == {
        "provider": "local",
        "target": "cli",
        "runtime_profile": "job",
        "timeout_seconds": "project",
        "max_attempts": "local",
        "priority": "cli",
        "environment_names": "project",
    }


def test_execution_environment_names_are_additive_and_deduplicated() -> None:
    resolved = resolve_execution_settings(
        ProjectExecutionDefaults(
            environment_names=("POSTTRAIN_TRACKIO_SERVER_URL",),
        ),
        local=ExecutionOverrides(
            environment_names=("REQUESTS_CA_BUNDLE", "POSTTRAIN_TRACKIO_SERVER_URL"),
        ),
        cli=ExecutionOverrides(
            environment_names=("HF_TOKEN", "REQUESTS_CA_BUNDLE"),
        ),
        job=ExecutionOverrides(
            provider="local",
            runtime_profile="framework/supervised@1",
            timeout_seconds=600,
            max_attempts=1,
            priority=0,
            environment_names=("TRACKIO_WRITE_TOKEN",),
        ),
    )

    assert resolved.environment_names == (
        "TRACKIO_WRITE_TOKEN",
        "POSTTRAIN_TRACKIO_SERVER_URL",
        "REQUESTS_CA_BUNDLE",
        "HF_TOKEN",
    )
    assert resolved.sources["environment_names"] == "cli"


def test_local_provider_factory_uses_project_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    local = load_local_execution_config(layout)
    calls: list[tuple[Path, dict[str, str], tuple[str, ...]]] = []

    class FakeLocalProvider:
        def __init__(
            self,
            *,
            state_root: Path,
            environment: dict[str, str],
            dns_servers: tuple[str, ...],
            trust_bundle: Path | None,
        ) -> None:
            assert trust_bundle is None
            calls.append((state_root, environment, dns_servers))

    class FakeModule:
        LocalDockerExecutionProvider = FakeLocalProvider

    monkeypatch.setattr(
        "posttrain_cli.execution_provider.importlib.import_module",
        lambda name: FakeModule(),
    )
    settings = resolve_execution_settings(ProjectExecutionDefaults())

    provider_name, _ = create_execution_provider(layout, settings, local)

    assert provider_name == "local-docker"
    assert calls == [(layout.state, {}, ())]


def test_dstack_factory_requires_protected_binding(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    settings = resolve_execution_settings(ProjectExecutionDefaults(provider="dstack"))

    with pytest.raises(RuntimeError, match=r"\[providers.dstack\]"):
        create_execution_provider(
            layout,
            settings,
            load_local_execution_config(layout),
        )


def test_dstack_python_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    environment = layout.state / "dstack-venv"
    executable = environment / "bin" / "python"
    executable.parent.mkdir(parents=True)
    target = layout.state / "base-python"
    target.write_text("", encoding="utf-8")
    executable.symlink_to(target)
    config = layout.state / "execution.toml"
    config.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[providers.dstack]",
                'project = "main"',
                'python = "dstack-venv/bin/python"',
                "",
            )
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)

    loaded = load_local_execution_config(layout)

    assert loaded.dstack is not None
    assert loaded.dstack.python == executable.absolute()
    assert loaded.dstack.python.is_symlink()


def test_dstack_factory_uses_only_protected_binding_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    config_path = layout.state / "execution.toml"
    python = layout.state / "dstack/bin/python"
    environment_file = layout.state / "dstack.env"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    environment_file.write_text("DSTACK_TOKEN=not-read-by-config-loader\n", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[providers.dstack]",
                'project = "main"',
                f'python = "{python}"',
                f'environment_file = "{environment_file}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    _write_runtime_environment(
        tmp_path / "posttrain.env",
        "TRACKIO_WRITE_TOKEN=from-posttrain-env\n",
    )
    calls: list[dict[str, object]] = []

    class FakeDstackProvider:
        @classmethod
        def from_sdk_environment(cls, **kwargs):
            calls.append(kwargs)
            return cls()

    class FakeModule:
        DstackExecutionProvider = FakeDstackProvider

    monkeypatch.setattr(
        "posttrain_cli.execution_provider.importlib.import_module",
        lambda name: FakeModule(),
    )
    settings = resolve_execution_settings(ProjectExecutionDefaults(provider="dstack"))

    provider_name, _ = create_execution_provider(
        layout,
        settings,
        load_local_execution_config(layout),
    )

    assert provider_name == "dstack"
    assert calls == [
        {
            "project": "main",
            "python": python.resolve(),
            "environment_file": environment_file.resolve(),
            "runtime_environment": {"TRACKIO_WRITE_TOKEN": "from-posttrain-env"},
            "trust_bundle": None,
            "capacity_wait_seconds": 0,
        }
    ]


def test_provider_binding_fingerprint_ignores_secret_rotation_but_not_identity(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    python = layout.state / "dstack/bin/python"
    environment_file = layout.state / "dstack.env"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    environment_file.write_text("DSTACK_TOKEN=first\n", encoding="utf-8")
    environment_file.chmod(0o600)
    config = layout.state / "execution.toml"
    config.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[providers.dstack]",
                'project = "main"',
                f'python = "{python}"',
                f'environment_file = "{environment_file}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    first = provider_binding_fingerprint(
        load_local_execution_config(layout),
        "dstack",
    )
    source = provider_source_for_project(layout, "dstack", load_local_execution_config(layout))

    environment_file.write_text("DSTACK_TOKEN=rotated\n", encoding="utf-8")
    rotated = provider_binding_fingerprint(
        load_local_execution_config(layout),
        "dstack",
    )
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'project = "main"',
            'project = "other"',
        ),
        encoding="utf-8",
    )
    changed = provider_binding_fingerprint(
        load_local_execution_config(layout),
        "dstack",
    )

    assert rotated == first
    assert changed != first
    assert source.provider == "dstack"
    assert source.endpoint_scope == "main"
    assert source.adapter_python == python.resolve()
    assert source.credential_file == environment_file.resolve()
    assert source.binding_fingerprint == first


def test_registry_resolves_from_the_environment_with_no_configuration_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project with no machine binding is still fully usable.

    The release pins every framework image, so the only thing a consumer must
    supply is the registry for their own actual-job images.
    """
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_REGISTRY=registry.internal/team\n")

    loaded = load_local_execution_config(layout)
    assert not loaded.path.exists()
    assert loaded.registry is not None

    manifest = _candidate_manifest()
    assert loaded.registry.repository == "registry.internal/team/posttrain-job"
    assert set(loaded.registry.kind_images) == set(manifest.kinds)
    assert loaded.registry.universal_image.value == manifest.base.reference(manifest.default_prefix)
    for variant, image in manifest.kinds.items():
        assert loaded.registry.kind_images[variant].value == image.reference(manifest.default_prefix)


def test_framework_images_do_not_follow_the_project_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSTTRAIN_REGISTRY is the project's registry, not a mirror by default.

    A site that can reach the public registry must keep pulling framework
    images from it even when its own job images go somewhere private.
    """
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_REGISTRY=registry.internal/team\n")

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    assert loaded.registry.kind_images["supervised"].value.startswith(_candidate_manifest().default_prefix + "/")


def test_trailing_slashes_in_the_environment_prefix_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_REGISTRY=  registry.internal/team/  \n")

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    assert loaded.registry.repository == "registry.internal/team/posttrain-job"


def test_no_registry_anywhere_yields_no_binding_rather_than_a_guess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    monkeypatch.delenv(REGISTRY_ENVIRONMENT_VARIABLE, raising=False)

    loaded = load_local_execution_config(layout)
    assert loaded.registry is None


def test_declared_registry_without_repository_names_both_remedies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    monkeypatch.delenv(REGISTRY_ENVIRONMENT_VARIABLE, raising=False)
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    path.write_text("schema_version = 1\n\n[registry]\n", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ContractError) as raised:
        load_local_execution_config(layout)
    message = str(raised.value)
    assert REGISTRY_ENVIRONMENT_VARIABLE in message
    assert "[registry].repository" in message


def test_mirror_prefix_moves_framework_images_without_changing_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirroring is a prefix change only; digests are content-addressed."""
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_REGISTRY=registry.internal/team\n")
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        'schema_version = 1\n\n[registry]\nmirror_prefix = "registry.internal/mirror"\n',
        encoding="utf-8",
    )
    path.chmod(0o600)

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    manifest = _candidate_manifest()
    for variant, image in manifest.kinds.items():
        resolved = loaded.registry.kind_images[variant].value
        assert resolved.startswith("registry.internal/mirror/")
        assert resolved.rsplit("@", 1)[1] == image.digest
    assert loaded.registry.universal_image.value.rsplit("@", 1)[1] == manifest.base.digest


def test_derived_constraint_profiles_carry_published_provided_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_REGISTRY=registry.internal/team\n")

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    profiles = loaded.registry.constraint_profiles
    assert profiles["online-rl-trl-py312"].provided_packages == ("verifiers",)
    assert profiles["supervised"].provided_packages == ()
    assert profiles["supervised"].contents_digest == _candidate_manifest().kinds["supervised"].lock_digest


def test_released_verl_variant_is_derived_from_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_REGISTRY=registry.internal/team\n")

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    assert loaded.registry.kind_images["online-rl-verl-py313"].value == _candidate_manifest().reference(
        "online-rl-verl-py313"
    )


def test_an_execution_file_without_a_registry_still_uses_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing execution.toml for one setting must not drop another.

    The registry is configured through the project-owned posttrain.env. A
    project that adds execution.toml for an unrelated reason, such as the
    local provider's canonical hostname, must retain that runtime value.
    """
    layout = _layout(tmp_path)
    layout.state.mkdir(parents=True, exist_ok=True)
    configured = layout.state / "execution.toml"
    configured.write_text(
        'schema_version = 1\n\n[providers.local]\ncanonical_hostname = "example-host"\n',
        encoding="utf-8",
    )
    configured.chmod(0o600)
    _write_runtime_environment(
        tmp_path / "posttrain.env",
        "POSTTRAIN_REGISTRY=registry.example.invalid/team\n",
    )

    loaded = load_local_execution_config(layout)

    assert loaded.local is not None
    assert loaded.local.canonical_hostname == "example-host"
    assert loaded.registry is not None


def test_tracking_endpoint_is_ignored_when_only_the_process_environment_has_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambient endpoint must not select a job or reconciliation destination."""
    layout = _layout(tmp_path)
    monkeypatch.setenv("POSTTRAIN_TRACKIO_SERVER_URL", "https://tracking.example.invalid")

    source = evidence_source_for_project(layout)

    assert source is not None
    assert source.endpoint is None


def test_machine_config_example_supplies_every_project_with_shared_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    config_home = tmp_path / "config"
    config_dir = config_home / "posttrain"
    config_dir.mkdir(parents=True)
    state_home = tmp_path / "state"
    config_path = config_dir / "config.toml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'projects = ["{layout.root}"]',
                "",
                "[tracking]",
                'kind = "trackio"',
                'endpoint = "https://trackio.lan"',
                "",
                "[trust]",
                'ca_bundle = "/etc/ssl/certs/ca-certificates.crt"',
                "",
                "[storage]",
                'run_root = "runs"',
                'model_cache = "cache/huggingface"',
                'compile_cache = "cache/compile"',
                "",
                "[providers.local]",
                'dns_servers = ["192.0.2.53", "2001:db8::53"]',
                "",
                "[cache]",
                "total_budget_bytes = 123456",
                "minimum_free_bytes = 7890",
                "reusable_max_age_seconds = 3600",
                "failed_debug_max_age_seconds = 120",
                "retain_failed_debug = true",
                "",
            )
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o644)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    loaded = load_local_execution_config(layout)

    assert loaded.machine is not None
    assert loaded.machine.projects == (layout.root,)
    assert loaded.path == config_path
    assert loaded.defaults.provider == "local"
    assert loaded.local is not None
    assert loaded.local.dns_servers == ("192.0.2.53", "2001:db8::53")
    assert loaded.local.storage is not None
    assert loaded.local.storage.run_root == state_home / "posttrain" / "runs"
    assert loaded.local.storage.model_cache == state_home / "posttrain" / "cache" / "huggingface"
    assert loaded.local.storage.compile_cache == state_home / "posttrain" / "cache" / "compile"
    assert loaded.machine.cache.total_budget_bytes == 123456
    assert loaded.machine.cache.minimum_free_bytes == 7890
    assert loaded.machine.cache.reusable_max_age_seconds == 3600
    assert loaded.machine.cache.failed_debug_max_age_seconds == 120
    assert loaded.machine.cache.retain_failed_debug
    assert load_execution_environment(loaded)["POSTTRAIN_TRACKIO_SERVER_URL"] == "https://trackio.lan"
    evidence = evidence_source_for_project(layout)
    assert evidence is not None
    assert evidence.endpoint == "https://trackio.lan"


def test_machine_config_extends_defaults_without_owning_dstack_worker_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_REGISTRY=project.example/jobs\n")
    config_home = tmp_path / "config"
    config_dir = config_home / "posttrain"
    config_dir.mkdir(parents=True)
    credentials = tmp_path / "dstack.env"
    credentials.write_text("DSTACK_TOKEN=redacted\n", encoding="utf-8")
    credentials.chmod(0o600)
    (config_dir / "config.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                'machine_name = "rtx-pro-96gb.lan"',
                'default_provider = "dstack"',
                "",
                "[services]",
                'python_index_url = "https://pypi.lan/simple/"',
                'job_registry = "registry.lan/posttrain"',
                "",
                "[providers.dstack]",
                'project = "main"',
                f'python = "{Path(sys.executable)}"',
                'credentials = "dstack-default"',
                "capacity_wait_seconds = 60",
                "",
                "[credentials.dstack-default]",
                f'file = "{credentials}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    loaded = load_local_execution_config(layout)

    assert loaded.machine is not None
    assert loaded.machine.name == "rtx-pro-96gb.lan"
    assert loaded.dstack is not None
    assert loaded.dstack.storage is None
    assert loaded.defaults.provider == "dstack"
    environment = load_execution_environment(loaded)
    assert environment["UV_INDEX_URL"] == "https://pypi.lan/simple/"
    assert environment["POSTTRAIN_REGISTRY"] == "project.example/jobs"


def test_machine_config_loads_remote_job_builder_from_protected_machine_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    config_home = tmp_path / "config"
    config_dir = config_home / "posttrain"
    config_dir.mkdir(parents=True)
    credentials = tmp_path / "job-builder.env"
    credentials.write_text("POSTTRAIN_JOB_BUILDER_TOKEN=redacted\n", encoding="utf-8")
    credentials.chmod(0o600)
    (config_dir / "config.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                'machine_name = "developer"',
                'default_provider = "local"',
                "",
                "[services.job_builder]",
                'mode = "remote"',
                'endpoint = "https://job-builder.lan"',
                'credentials = "job-builder"',
                "request_timeout_seconds = 45",
                "upload_concurrency = 3",
                "",
                "[credentials.job-builder]",
                f'file = "{credentials}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    loaded = load_local_execution_config(layout)

    assert loaded.machine is not None
    assert loaded.machine.services.job_builder.mode == "remote"
    assert loaded.machine.services.job_builder.endpoint == "https://job-builder.lan"
    assert loaded.machine.services.job_builder.upload_concurrency == 3
    assert load_execution_environment(loaded)["POSTTRAIN_JOB_BUILDER_TOKEN"] == "redacted"
    assert resolve_job_builder(loaded.machine, cli_override=None).mode == "remote"
    assert resolve_job_builder(loaded.machine, cli_override=None).source == "machine"
    assert resolve_job_builder(loaded.machine, cli_override="local").source == "cli"


def test_remote_job_builder_override_requires_machine_remote_configuration() -> None:
    with pytest.raises(ContractError, match="--builder remote requires machine"):
        resolve_job_builder(None, cli_override="remote")

    local = resolve_job_builder(None, cli_override="local")
    assert local.mode == "local"
    assert local.source == "cli"


def test_machine_config_rejects_local_dns_hostnames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    config_home = tmp_path / "config"
    config_dir = config_home / "posttrain"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'schema_version = 1\n\n[providers.local]\ndns_servers = ["dns.lan"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    with pytest.raises(ContractError, match="only literal IP addresses"):
        load_local_execution_config(layout)


def test_removed_profile_selector_fails_with_a_migration_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    _write_runtime_environment(tmp_path / "posttrain.env", "POSTTRAIN_PROFILE=rtx96\n")
    config_home = tmp_path / "config"
    config_dir = config_home / "posttrain"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    with pytest.raises(ContractError, match="POSTTRAIN_PROFILE was removed"):
        load_local_execution_config(layout)


def test_prepared_evidence_source_uses_the_explicit_resolved_runtime_environment(tmp_path: Path) -> None:
    layout = _layout(tmp_path)

    source = evidence_source_for_project(
        layout,
        environment={"POSTTRAIN_TRACKIO_SERVER_URL": "https://tracking.example.invalid"},
    )

    assert source is not None
    assert source.endpoint == "https://tracking.example.invalid"


def test_an_explicit_trust_bundle_wins_over_every_other_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = tmp_path / "declared.pem"
    declared.write_text("-----BEGIN CERTIFICATE-----\nX\n", encoding="utf-8")
    other = tmp_path / "other.pem"
    other.write_text("-----BEGIN CERTIFICATE-----\nY\n", encoding="utf-8")
    monkeypatch.setenv(TRUST_BUNDLE_ENVIRONMENT_VARIABLE, str(other))

    resolved = resolve_trust_bundle(declared)

    assert resolved.path == declared.resolve()
    assert resolved.source == "configured"


def test_the_environment_supplies_a_bundle_when_nothing_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = tmp_path / "from-env.pem"
    declared.write_text("-----BEGIN CERTIFICATE-----\nX\n", encoding="utf-8")
    monkeypatch.setenv(TRUST_BUNDLE_ENVIRONMENT_VARIABLE, str(declared))

    resolved = resolve_trust_bundle(None)

    assert resolved.path == declared.resolve()
    assert resolved.source == "environment"


def test_a_machine_with_no_internal_authority_needs_no_configuration() -> None:
    """The convention path is the only source allowed to be absent.

    Its absence is how a machine says it has no internal authority, so jobs
    simply trust what their image already trusts. The shared fixture points
    that path somewhere empty, so this holds wherever the tests run.
    """
    resolved = resolve_trust_bundle(None)

    assert resolved.path is None
    assert resolved.source == "none"


def test_a_named_bundle_that_does_not_exist_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Substituting a different authority for the one asked for is worse than failing."""
    with pytest.raises(ContractError, match="trust_bundle does not exist"):
        resolve_trust_bundle(tmp_path / "absent.pem")

    monkeypatch.setenv(TRUST_BUNDLE_ENVIRONMENT_VARIABLE, str(tmp_path / "also-absent.pem"))
    with pytest.raises(ContractError, match="does not name a file"):
        resolve_trust_bundle(None)


def test_admission_root_prefers_an_explicit_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "ledger"
    monkeypatch.setenv(ADMISSION_ROOT_ENVIRONMENT_VARIABLE, str(root))

    assert resolve_admission_state_root() == root.resolve()


def test_admission_root_rejects_a_relative_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ADMISSION_ROOT_ENVIRONMENT_VARIABLE, "relative/ledger")

    with pytest.raises(ContractError, match="must be an absolute path"):
        resolve_admission_state_root()


def test_admission_root_falls_back_to_xdg_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ADMISSION_ROOT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(
        "posttrain_cli.execution_config._WORKER_ADMISSION_ROOT",
        tmp_path / "missing-worker",
    )
    xdg = tmp_path / "xdg-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))

    assert resolve_admission_state_root() == (xdg / "posttrain").resolve()
