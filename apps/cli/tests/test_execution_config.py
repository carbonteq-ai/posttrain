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
from posttrain.runtime_images.manifest import load_manifest
from posttrain_cli.execution_config import (
    ADMISSION_ROOT_ENVIRONMENT_VARIABLE,
    REGISTRY_ENVIRONMENT_VARIABLE,
    TRUST_BUNDLE_ENVIRONMENT_VARIABLE,
    ExecutionOverrides,
    load_local_execution_config,
    provider_binding_fingerprint,
    resolve_admission_state_root,
    resolve_execution_settings,
    resolve_trust_bundle,
)
from posttrain_cli.execution_provider import (
    create_execution_provider,
    evidence_source_for_project,
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
    assert configuration.local.storage is not None
    assert configuration.local.storage.run_root == (layout.state / "local-runs").resolve()
    assert configuration.local.trust_bundle == local_trust_bundle.resolve()
    assert configuration.dstack.storage is not None
    assert configuration.dstack.storage.run_root == Path("/var/lib/posttrain/runs")
    assert configuration.dstack.trust_bundle == Path("/etc/posttrain/trust/internal-ca.pem")
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
    manifest = load_manifest()

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


def test_registry_resolves_from_the_environment_with_no_configuration_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project with no machine binding is still fully usable.

    The release pins every framework image, so the only thing a consumer must
    supply is the registry for their own actual-job images.
    """
    layout = _layout(tmp_path)
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")

    loaded = load_local_execution_config(layout)
    assert not loaded.path.exists()
    assert loaded.registry is not None

    manifest = load_manifest()
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
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    assert loaded.registry.kind_images["supervised"].value.startswith(load_manifest().default_prefix + "/")


def test_trailing_slashes_in_the_environment_prefix_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(tmp_path)
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "  registry.internal/team/  ")

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
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")
    path = layout.state / "execution.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        'schema_version = 1\n\n[registry]\nmirror_prefix = "registry.internal/mirror"\n',
        encoding="utf-8",
    )
    path.chmod(0o600)

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    manifest = load_manifest()
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
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    profiles = loaded.registry.constraint_profiles
    assert profiles["online-rl-trl-py312"].provided_packages == ("verifiers",)
    assert profiles["supervised"].provided_packages == ()
    assert profiles["supervised"].contents_digest == load_manifest().kinds["supervised"].lock_digest


def test_release_blocked_variant_is_never_derived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard that refuses an unpublished variant must stay reachable."""
    layout = _layout(tmp_path)
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.internal/team")

    loaded = load_local_execution_config(layout)
    assert loaded.registry is not None
    assert "online-rl-verl-py313" not in loaded.registry.kind_images


def test_an_execution_file_without_a_registry_still_uses_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing execution.toml for one setting must not drop another.

    The registry is configured through POSTTRAIN_REGISTRY. A project that adds
    execution.toml for an unrelated reason, such as the local provider's
    canonical hostname, previously lost it: the environment was consulted only
    when no execution configuration existed at all.
    """
    layout = _layout(tmp_path)
    layout.state.mkdir(parents=True, exist_ok=True)
    configured = layout.state / "execution.toml"
    configured.write_text(
        'schema_version = 1\n\n[providers.local]\ncanonical_hostname = "example-host"\n',
        encoding="utf-8",
    )
    configured.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENVIRONMENT_VARIABLE, "registry.example.invalid/team")

    loaded = load_local_execution_config(layout)

    assert loaded.local is not None
    assert loaded.local.canonical_hostname == "example-host"
    assert loaded.registry is not None


def test_tracking_endpoint_is_recorded_from_the_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence must be read back from where the job actually wrote it.

    The job container receives POSTTRAIN_TRACKIO_SERVER_URL from the process
    environment, so a run configured through the shell writes to the remote
    server. Recording no endpoint made reconciliation read a local store
    instead, where the run does not exist, and a succeeded run could never
    produce retained evidence.
    """
    layout = _layout(tmp_path)
    monkeypatch.setenv("POSTTRAIN_TRACKIO_SERVER_URL", "https://tracking.example.invalid")

    source = evidence_source_for_project(layout)

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
