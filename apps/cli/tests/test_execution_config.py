from __future__ import annotations

import hashlib
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
from posttrain_cli.execution_config import (
    ExecutionOverrides,
    load_local_execution_config,
    provider_binding_fingerprint,
    resolve_execution_settings,
)
from posttrain_cli.execution_provider import (
    create_execution_provider,
    evidence_source_for_run,
)


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
        layout.manifest.read_text(encoding="utf-8")
        + '\ntracking = "none"\n',
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
                'trust_bundle = "local-ca.pem"',
                "",
                "[providers.dstack]",
                'project = "main"',
                'python = "dstack-venv/bin/python"',
                'environment_file = "dstack.env"',
                'trust_bundle = "/etc/posttrain/trust/internal-ca.pem"',
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
                            ('provided_packages = ["verifiers"]',)
                            if profile in {"online-rl-trl-py312", "eval"}
                            else ()
                        ),
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
    assert configuration.local.storage is not None
    assert configuration.local.storage.run_root == (layout.state / "local-runs").resolve()
    assert configuration.local.trust_bundle == local_trust_bundle.resolve()
    assert configuration.dstack.storage is not None
    assert configuration.dstack.storage.run_root == Path("/var/lib/posttrain/runs")
    assert configuration.dstack.trust_bundle == Path(
        "/etc/posttrain/trust/internal-ca.pem"
    )
    assert configuration.registry is not None
    assert configuration.registry.repository == "registry.lan/carbonteq/posttrain"
    assert configuration.registry.universal_image.value == image
    assert configuration.registry.kind_images["online-rl-trl-py312"].value == image
    eval_constraints = configuration.registry.constraint_profiles["eval"]
    assert eval_constraints.contents_digest == constraints_digest
    assert eval_constraints.provided_packages == ("verifiers",)
    assert eval_constraints.digest != constraints_digest
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
    assert set(loaded.registry.kind_images) == {"supervised"}


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
                f'supervised = "{image}"',
                f'eval = "{image}"',
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
    calls: list[tuple[Path, dict[str, str]]] = []

    class FakeLocalProvider:
        def __init__(
            self,
            *,
            state_root: Path,
            environment: dict[str, str],
            trust_bundle: Path | None,
        ) -> None:
            assert trust_bundle is None
            calls.append((state_root, environment))

    class FakeModule:
        LocalDockerExecutionProvider = FakeLocalProvider

    monkeypatch.setattr(
        "posttrain_cli.execution_provider.importlib.import_module",
        lambda name: FakeModule(),
    )
    settings = resolve_execution_settings(ProjectExecutionDefaults())

    provider_name, _ = create_execution_provider(layout, settings, local)

    assert provider_name == "local-docker"
    assert calls == [(layout.state, {})]


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
            "job_environment_file": None,
            "trust_bundle": None,
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
