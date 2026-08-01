"""Provider factories for resolved CLI execution settings."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from posttrain.catalog import ProjectLayout, load_project_layout
from posttrain.execution import (
    AdmissionEntry,
    CancelledTrackingWriter,
    ExecutionAdmissionService,
    ExecutionEvidenceSource,
    ExecutionPlan,
    ExecutionProvider,
    ExecutionProviderSource,
    ExecutionSubmissionStore,
    JobExecutionService,
    ProjectControlLocator,
)
from posttrain.tracking import RunDataSource

from .execution_config import (
    DstackBinding,
    ExecutionOverrides,
    LocalExecutionConfig,
    LocalProviderBinding,
    ResolvedExecutionSettings,
    load_execution_environment,
    load_local_execution_config,
    provider_binding_fingerprint,
    resolve_admission_state_root,
    resolve_execution_settings,
    resolve_trust_bundle,
)
from .tracking_config import project_tracking_environment


def create_execution_provider(
    layout: ProjectLayout,
    settings: ResolvedExecutionSettings,
    local_config: LocalExecutionConfig,
) -> tuple[str, ExecutionProvider]:
    """Construct the selected optional adapter without leaking it into neutral packages."""

    if settings.provider == "local":
        module = _import_provider(
            "posttrain_execution_local",
            extra="local execution support is not installed",
        )
        provider_type = getattr(module, "LocalDockerExecutionProvider", None)
        if provider_type is None:
            raise RuntimeError("installed local execution package has no provider")
        provider = provider_type(
            state_root=layout.state,
            environment=load_execution_environment(local_config),
            trust_bundle=resolve_trust_bundle(
                local_config.local.trust_bundle if local_config.local is not None else None
            ).path,
        )
        return "local-docker", cast(ExecutionProvider, provider)

    if settings.provider == "dstack":
        binding = local_config.dstack
        if binding is None:
            raise RuntimeError(f"dstack execution requires [providers.dstack] in {local_config.path}")
        _require_file(binding.python, "dstack Python")
        if binding.environment_file is not None:
            _require_file(binding.environment_file, "dstack environment file")
        module = _import_provider(
            "posttrain_execution_dstack",
            extra="dstack execution support is not installed",
        )
        provider_type = getattr(module, "DstackExecutionProvider", None)
        if provider_type is None:
            raise RuntimeError("installed dstack execution package has no provider")
        provider = provider_type.from_sdk_environment(
            project=binding.project,
            python=binding.python,
            environment_file=binding.environment_file,
            runtime_environment=load_execution_environment(local_config),
            trust_bundle=resolve_trust_bundle(binding.trust_bundle).path,
            capacity_wait_seconds=binding.capacity_wait_seconds,
        )
        return "dstack", cast(ExecutionProvider, provider)

    raise RuntimeError(f"unsupported execution provider: {settings.provider}")


def provider_source_for_project(
    layout: ProjectLayout,
    provider_name: str,
    local_config: LocalExecutionConfig | None = None,
) -> ExecutionProviderSource:
    """Freeze secret-free adapter identity while leaving credentials rotatable."""

    local = local_config or load_local_execution_config(layout)
    profile_id = local.machine.name if local.machine is not None else f"project:{layout.project_id}"
    fingerprint = provider_binding_fingerprint(local, provider_name)
    if provider_name == "local-docker":
        binding = local.local or LocalProviderBinding()
        return ExecutionProviderSource(
            provider=provider_name,
            profile_id=profile_id,
            binding_fingerprint=fingerprint,
            trust_bundle=binding.trust_bundle,
            canonical_hostname=binding.canonical_hostname,
        )
    if provider_name == "dstack":
        binding = local.dstack
        if binding is None:
            raise RuntimeError("dstack provider binding is unavailable")
        return ExecutionProviderSource(
            provider=provider_name,
            profile_id=profile_id,
            binding_fingerprint=fingerprint,
            endpoint_scope=binding.project,
            adapter_python=binding.python,
            credential_file=binding.environment_file,
            trust_bundle=binding.trust_bundle,
            capacity_wait_seconds=binding.capacity_wait_seconds,
        )
    return ExecutionProviderSource(
        provider=provider_name,
        profile_id=profile_id,
        binding_fingerprint=fingerprint,
    )


def _configuration_for_provider_source(
    owner: ProjectLayout,
    source: ExecutionProviderSource,
) -> LocalExecutionConfig:
    current = load_local_execution_config(owner)
    if source.provider == "local-docker":
        return replace(
            current,
            local=LocalProviderBinding(
                canonical_hostname=source.canonical_hostname,
                storage=(current.local.storage if current.local is not None else None),
                trust_bundle=source.trust_bundle,
            ),
        )
    if source.provider == "dstack":
        if source.endpoint_scope is None or source.adapter_python is None:
            raise RuntimeError("recorded dstack provider source is incomplete")
        return replace(
            current,
            dstack=DstackBinding(
                project=source.endpoint_scope,
                python=source.adapter_python,
                environment_file=source.credential_file,
                storage=(current.dstack.storage if current.dstack is not None else None),
                trust_bundle=source.trust_bundle,
                capacity_wait_seconds=source.capacity_wait_seconds,
            ),
        )
    raise RuntimeError(f"recorded provider source is unsupported: {source.provider!r}")


def execution_service_for_run(
    layout: ProjectLayout,
    run_id: str,
) -> JobExecutionService:
    """Recover the provider named by durable submission state, not current defaults."""

    store = ExecutionSubmissionStore(layout.state)
    submission = store.load(run_id)
    providers = {"local-docker": "local", "dstack": "dstack"}
    try:
        configured_provider = providers[submission.provider]
    except KeyError as error:
        raise RuntimeError(f"execution run uses unsupported provider {submission.provider!r}") from error
    local = (
        _configuration_for_provider_source(layout, submission.provider_source)
        if submission.provider_source_recorded and submission.provider_source is not None
        else load_local_execution_config(layout)
    )
    settings = resolve_execution_settings(
        layout.execution,
        local=local.defaults,
        cli=ExecutionOverrides(provider=configured_provider),
    )
    provider_name, provider = create_execution_provider(layout, settings, local)
    if provider_name != submission.provider:
        raise RuntimeError(f"execution provider resolved as {provider_name!r}, expected {submission.provider!r}")
    return JobExecutionService(
        provider,
        store,
        provider_name=provider_name,
        evidence_source=submission.evidence_source,
        provider_source=submission.provider_source,
    )


def execution_admission_service(
    layout: ProjectLayout,
) -> ExecutionAdmissionService:
    """Create the durable singular admission controller for this project."""

    def service_factory(
        provider_name: str,
        evidence_source: ExecutionEvidenceSource | None,
    ) -> JobExecutionService:
        configured = {"local-docker": "local", "dstack": "dstack"}
        try:
            provider_override = configured[provider_name]
        except KeyError as error:
            raise RuntimeError(f"execution admission uses unsupported provider {provider_name!r}") from error
        local = load_local_execution_config(layout)
        settings = resolve_execution_settings(
            layout.execution,
            local=local.defaults,
            cli=ExecutionOverrides(provider=provider_override),
        )
        resolved_name, provider = create_execution_provider(layout, settings, local)
        if resolved_name != provider_name:
            raise RuntimeError(f"execution provider resolved as {resolved_name!r}, expected {provider_name!r}")
        return JobExecutionService(
            provider,
            ExecutionSubmissionStore(layout.state),
            provider_name=resolved_name,
            evidence_source=evidence_source,
        )

    def owner_layout(entry: AdmissionEntry) -> tuple[ProjectLayout, ProjectControlLocator]:
        locator = entry.control_locator
        if locator is None:
            raise RuntimeError(
                f"admission run {entry.run_id} predates project ownership locators; "
                "reconcile it from its owning project"
            )
        owner = load_project_layout(locator.project_root)
        if owner.project_id != locator.project_id:
            raise RuntimeError(
                f"admission run {entry.run_id} names project {locator.project_id!r}, "
                f"but {locator.project_root} currently identifies {owner.project_id!r}"
            )
        if owner.state.resolve() != locator.control_store:
            raise RuntimeError(f"admission run {entry.run_id} control store no longer matches its owning project")
        return owner, locator

    def entry_service_factory(entry: AdmissionEntry) -> JobExecutionService:
        owner, locator = owner_layout(entry)
        configured = {"local-docker": "local", "dstack": "dstack"}
        try:
            provider_override = configured[entry.plan.provider]
        except KeyError as error:
            raise RuntimeError(f"execution admission uses unsupported provider {entry.plan.provider!r}") from error
        local = (
            _configuration_for_provider_source(owner, entry.provider_source)
            if entry.provider_source is not None
            else load_local_execution_config(owner)
        )
        settings = resolve_execution_settings(
            owner.execution,
            local=local.defaults,
            cli=ExecutionOverrides(provider=provider_override),
        )
        resolved_name, provider = create_execution_provider(owner, settings, local)
        if resolved_name != entry.plan.provider:
            raise RuntimeError(f"execution provider resolved as {resolved_name!r}, expected {entry.plan.provider!r}")
        return JobExecutionService(
            provider,
            ExecutionSubmissionStore(locator.control_store),
            provider_name=resolved_name,
            evidence_source=entry.evidence_source,
            provider_source=entry.provider_source,
        )

    def provider_binding_factory(provider_name: str) -> str:
        return provider_binding_fingerprint(
            load_local_execution_config(layout),
            provider_name,
        )

    def entry_provider_binding_factory(entry: AdmissionEntry) -> str:
        owner, _locator = owner_layout(entry)
        return provider_binding_fingerprint(
            load_local_execution_config(owner),
            entry.plan.provider,
        )

    def physical_host_factory(plan: ExecutionPlan) -> str | None:
        if plan.provider != "local-docker":
            return None
        local = load_local_execution_config(layout).local
        return local.canonical_hostname if local is not None else None

    return ExecutionAdmissionService(
        resolve_admission_state_root(),
        service_factory,
        entry_service_factory=entry_service_factory,
        provider_binding_factory=provider_binding_factory,
        entry_provider_binding_factory=entry_provider_binding_factory,
        physical_host_factory=physical_host_factory,
    )


def tracking_source_for_project(layout: ProjectLayout) -> RunDataSource:
    """Construct the project's normalized evidence reader from protected settings."""

    source = evidence_source_for_project(layout)
    if source is None:
        raise RuntimeError(
            "run reconciliation requires project tracking; set tracking to "
            "'trackio' or 'wandb' in .posttrain/project.toml"
        )
    return _tracking_source(layout, source)


def evidence_source_for_project(
    layout: ProjectLayout,
    *,
    environment: Mapping[str, str] | None = None,
) -> ExecutionEvidenceSource | None:
    """Resolve the secret-free evidence destination recorded at submission."""

    resolved_environment = dict(environment) if environment is not None else project_tracking_environment(layout)
    if layout.tracking == "trackio":
        project = resolved_environment.get("POSTTRAIN_TRACKIO_PROJECT") or layout.project_id
        return ExecutionEvidenceSource(
            provider="trackio",
            source_id=f"trackio-{layout.project_id}",
            project=project,
            endpoint=resolved_environment.get("POSTTRAIN_TRACKIO_SERVER_URL"),
        )
    if layout.tracking == "wandb":
        entity = resolved_environment.get("WANDB_ENTITY")
        if not entity:
            raise RuntimeError("W&B execution requires WANDB_ENTITY")
        project = resolved_environment.get("POSTTRAIN_WANDB_PROJECT") or layout.project_id
        return ExecutionEvidenceSource(
            provider="wandb",
            source_id=f"wandb-{layout.project_id}",
            project=project,
            endpoint=resolved_environment.get("WANDB_BASE_URL"),
            scope=entity,
        )
    if layout.tracking == "none":
        return None
    raise RuntimeError(f"unsupported project tracking provider: {layout.tracking}")


def evidence_source_for_run(
    layout: ProjectLayout,
    run_id: str,
) -> ExecutionEvidenceSource:
    """Load the submitted destination, falling back only for legacy receipts."""

    submission = ExecutionSubmissionStore(layout.state).load(run_id)
    if submission.evidence_source_recorded:
        if submission.evidence_source is None:
            raise RuntimeError(f"execution run {run_id} was submitted with tracking disabled")
        return submission.evidence_source
    warnings.warn(
        f"execution run {run_id} predates durable evidence locators; "
        "using the project's current tracking configuration",
        RuntimeWarning,
        stacklevel=2,
    )
    source = evidence_source_for_project(layout)
    if source is None:
        raise RuntimeError(
            f"execution run {run_id} has no recorded evidence locator and project tracking is currently disabled"
        )
    return source


def tracking_source_for_run(
    layout: ProjectLayout,
    run_id: str,
) -> RunDataSource:
    """Construct a reader for the immutable evidence destination of one run."""

    return _tracking_source(layout, evidence_source_for_run(layout, run_id))


def reconciliation_source_for_run(
    layout: ProjectLayout,
    run_id: str,
) -> RunDataSource | None:
    """Return the submitted reader, or the explicit no-tracking waiver."""

    submission = ExecutionSubmissionStore(layout.state).load(run_id)
    if submission.evidence_source_recorded and submission.evidence_source is None:
        return None
    return tracking_source_for_run(layout, run_id)


def _tracking_source(
    layout: ProjectLayout,
    source: ExecutionEvidenceSource,
) -> RunDataSource:
    project_tracking_environment(layout)
    if source.provider == "trackio":
        module = _import_provider(
            "posttrain_tracking_trackio",
            extra="Trackio reconciliation support is not installed",
        )
        source_type = getattr(module, "TrackioDataSource", None)
        if source_type is None:
            raise RuntimeError("installed Trackio package has no normalized reader")
        return cast(
            RunDataSource,
            source_type(source.project, server_url=source.endpoint),
        )

    if source.provider == "wandb":
        module = _import_provider(
            "posttrain_tracking_wandb",
            extra="W&B reconciliation support is not installed",
        )
        source_type = getattr(module, "WandbDataSource", None)
        settings_type = getattr(module, "WandbSettings", None)
        if source_type is None or settings_type is None:
            raise RuntimeError("installed W&B package has no normalized reader")
        if source.scope is None:
            raise RuntimeError("recorded W&B evidence source has no entity")
        return cast(
            RunDataSource,
            source_type(
                settings_type(
                    entity=source.scope,
                    project=source.project,
                    base_url=source.endpoint,
                    mode="online",
                )
            ),
        )

    raise RuntimeError(f"unsupported evidence provider: {source.provider}")


def cancelled_tracking_writer_for_project(
    layout: ProjectLayout,
) -> CancelledTrackingWriter:
    """Construct the explicit Trackio cancellation-recovery writer."""

    if layout.tracking != "trackio":
        raise RuntimeError("tracking cancellation recovery currently requires Trackio")
    environment = project_tracking_environment(layout)
    module = _import_provider(
        "posttrain_tracking_trackio",
        extra="Trackio cancellation recovery support is not installed",
    )
    writer_type = getattr(module, "TrackioCancelledRunRecovery", None)
    settings_type = getattr(module, "TrackioSettings", None)
    if writer_type is None or settings_type is None:
        raise RuntimeError("installed Trackio package has no cancellation recovery writer")
    project = environment.get(
        "POSTTRAIN_TRACKIO_PROJECT",
        layout.project_id,
    )
    server_url = environment.get("POSTTRAIN_TRACKIO_SERVER_URL")
    return cast(
        CancelledTrackingWriter,
        writer_type(
            settings_type(
                project=project,
                server_url=server_url,
            ),
            write_token=environment.get("TRACKIO_WRITE_TOKEN"),
        ),
    )


def cancelled_tracking_writer_for_run(
    layout: ProjectLayout,
    run_id: str,
) -> CancelledTrackingWriter:
    """Construct a cancellation writer for the run's recorded destination."""

    source = evidence_source_for_run(layout, run_id)
    if source.provider != "trackio":
        raise RuntimeError("tracking cancellation recovery currently requires Trackio")
    environment = project_tracking_environment(layout)
    module = _import_provider(
        "posttrain_tracking_trackio",
        extra="Trackio cancellation recovery support is not installed",
    )
    writer_type = getattr(module, "TrackioCancelledRunRecovery", None)
    settings_type = getattr(module, "TrackioSettings", None)
    if writer_type is None or settings_type is None:
        raise RuntimeError("installed Trackio package has no cancellation recovery writer")
    return cast(
        CancelledTrackingWriter,
        writer_type(
            settings_type(
                project=source.project,
                server_url=source.endpoint,
            ),
            write_token=environment.get("TRACKIO_WRITE_TOKEN"),
        ),
    )


def _import_provider(module: str, *, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise RuntimeError(extra) from error


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")


__all__ = [
    "cancelled_tracking_writer_for_project",
    "cancelled_tracking_writer_for_run",
    "create_execution_provider",
    "evidence_source_for_project",
    "evidence_source_for_run",
    "execution_service_for_run",
    "reconciliation_source_for_run",
    "tracking_source_for_project",
    "tracking_source_for_run",
]
