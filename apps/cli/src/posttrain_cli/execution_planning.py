"""Plan, pack, publish, and launch one immutable actual-job OCI image."""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import yaml
from posttrain.catalog import FamilyRegistryLock, ProjectLayout
from posttrain.common import Catalog, CatalogRef, ContractError, ExecutionTarget, StoredArtifactRef
from posttrain.data import DatasetLoadPlan, project_dataset_input_paths
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    BackendRuntimeIdentity,
    ExecutionEvidenceSource,
    ExecutionMount,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionProviderSource,
    ExecutionRequest,
    ExecutionSubmissionStore,
    JobExecutionService,
    JobPackageManifest,
    RuntimeImageRef,
)
from posttrain.execution_pack import (
    ImagePublicationSpec,
    ImmutableDatasetPackager,
    ImmutableSourceSnapshotter,
    JobImagePublicationRequest,
    JobImagePublisher,
    JobImageResolutionRequest,
    JobPackInputs,
    JobPackPlan,
    JobPackService,
    LocalDaemonJobImage,
    LocalPublishedJobImage,
    PackageMaterializationStore,
    PackedJobContext,
    ProjectConfigBundle,
    PublishedJobImage,
    SourceSnapshotInspection,
    SourceSnapshotRequest,
    activation_resource_sources,
    environment_bindings,
    plan_job_pack,
    publication_key_for,
)
from posttrain.project import JobIntent, load_project_pack_config
from posttrain.runtime_images import JOB_BAKE_FILE, cached_definition_root, published_manifest_digest
from posttrain.tracking import ArtifactInput, ArtifactLink, RunSpec
from posttrain.work import (
    PreparedWorkPackageJob,
    load_work_package,
    override_job_execution_target,
    prepare_work_package_job,
)
from posttrain_execution_buildkit import (
    BuildKitJobImagePublisher,
    EnvironmentPackagerCacheRoots,
    ImmutableEnvironmentPackager,
    KindDependencyConstraints,
    UvDependencyCompileCli,
    job_build_definition_digest,
)
from posttrain_execution_job_builder import RemoteJobBuilderConfig, RemoteJobImagePublisher

from .context import CliState
from .execution_config import (
    REGISTRY_ENVIRONMENT_VARIABLE,
    ExecutionOverrides,
    ExecutionStorageBinding,
    LaunchOverrides,
    LocalExecutionConfig,
    PackageOverrides,
    RegistryBinding,
    ResolvedExecutionSettings,
    SettingSource,
    derived_local_registry,
    derived_registry,
    load_execution_environment,
    load_local_execution_config,
    resolve_execution_settings,
    resolve_job_builder,
)
from .execution_provider import create_execution_provider, evidence_source_for_project, provider_source_for_project
from .framework_distributions import FrameworkDistributions
from .framework_distributions import materialize as materialize_framework_distributions
from .selection_resolve import resolve_selection
from .state_layout import cache_path
from .work_runtime import load_work_package_bundle, runtime_context

_EMPTY_OVERRIDES = ExecutionOverrides()
_EMPTY_PACKAGE_OVERRIDES = PackageOverrides()
_EMPTY_LAUNCH_OVERRIDES = LaunchOverrides()
_PACKAGE_KEY = re.compile(r"^[0-9a-f]{64}$")
_FRAMEWORK_INSTALL_ROOTS = (
    "apps/runtime",
    "packages/catalog",
    "packages/common",
    "packages/data",
    "packages/environment",
    "packages/eval",
    "packages/execution",
    "packages/jobs",
    "packages/serve",
    "packages/tracking",
    "packages/tracking-trackio",
    "packages/tracking-wandb",
    "packages/train",
    "packages/work",
)
_FRAMEWORK_SOURCE_INCLUDES = ("pyproject.toml", *_FRAMEWORK_INSTALL_ROOTS)


@dataclass(frozen=True, slots=True)
class PlannedJobPackage:
    """Read-only immutable capsule plan, independent of launch infrastructure."""

    layout: ProjectLayout
    catalog: Catalog
    work_package_path: Path
    prepared: PreparedWorkPackageJob
    local_config: LocalExecutionConfig
    pack_plan: JobPackPlan
    framework_source_request: SourceSnapshotRequest | None
    framework_distributions: FrameworkDistributions | None
    project_source_request: SourceSnapshotRequest
    project_config_digest: str
    target: ExecutionTarget
    runtime_profile: str
    target_source: SettingSource
    runtime_profile_source: SettingSource
    project_source_inspection: SourceSnapshotInspection
    framework_source_inspection: SourceSnapshotInspection | None = None
    dataset_source_estimates: tuple[dict[str, object], ...] = ()
    builder_override: str | None = None

    def materialize(self) -> PackedJobContext:
        """Materialize the immutable job context without publishing an image."""

        registry = _registry(self.local_config)
        source_root = cache_path(self.layout, "pack", "sources")
        snapshotter = ImmutableSourceSnapshotter(cache_root=source_root)
        project_source = snapshotter.materialize(self.project_source_request)
        project_environment_sources: dict[str, Path] = {}
        for request in self.pack_plan.spec.project_environment_sources:
            snapshot = snapshotter.materialize(
                SourceSnapshotRequest(
                    root=self.layout.root,
                    includes=(request.path,),
                    install_roots=(request.path,),
                )
            )
            if snapshot.digest != request.tree_digest:
                raise ContractError("project environment source changed after planning; run job plan again")
            project_environment_sources[request.path] = snapshot.package.root / request.path
        if self.framework_source_request is not None:
            framework_source = snapshotter.materialize(self.framework_source_request)
            framework_package = framework_source.package
            framework_wheels: tuple[Path, ...] = ()
            framework_digest = framework_source.digest
        else:
            assert self.framework_distributions is not None
            framework_package = None
            framework_wheels = self.framework_distributions.wheels
            framework_digest = self.framework_distributions.digest
        if (
            framework_digest != self.pack_plan.spec.framework_source_digest
            or project_source.digest != self.pack_plan.spec.project_source_digest
        ):
            raise ContractError("source bytes changed after planning; run job plan again")

        cache_root = cache_path(self.layout, "pack", "cache")
        constraints = {
            profile: KindDependencyConstraints(
                profile,
                binding.path.read_text(encoding="utf-8"),
                binding.provided_packages,
            )
            for profile, binding in registry.constraint_profiles.items()
        }
        backend_constraints = {
            profile: KindDependencyConstraints(
                profile,
                binding.backend_path.read_text(encoding="utf-8"),
                binding.backend_provided_packages,
                role="backend",
                python_version="3.13.12",
                python_executable="/opt/posttrain-verl/bin/python",
                requirements_filename="runtime.backend.requirements.txt",
            )
            for profile, binding in registry.constraint_profiles.items()
            if binding.backend_path is not None
        }
        for profile, selected in constraints.items():
            binding = registry.constraint_profiles[profile]
            if selected.constraints_sha256 != binding.contents_digest or selected.digest != binding.digest:
                raise ContractError(f"job-kind constraint profile changed after configuration load: {profile}")
            selected_backend = backend_constraints.get(profile)
            if (selected_backend is None) != (binding.backend_digest is None) or (
                selected_backend is not None
                and (
                    selected_backend.constraints_sha256 != binding.backend_contents_digest
                    or selected_backend.digest != binding.backend_digest
                )
            ):
                raise ContractError(f"job-kind backend constraint profile changed after configuration load: {profile}")
        execution_environment = load_execution_environment(self.local_config)
        dependency_index_environment = {
            name: execution_environment[name]
            for name in ("UV_INDEX_PASSWORD", "UV_INDEX_URL", "UV_INDEX_USERNAME")
            if name in execution_environment
        }
        environment_packager = ImmutableEnvironmentPackager(
            cache_roots=EnvironmentPackagerCacheRoots(
                git_sources=cache_root / "git",
                wheels=cache_root / "wheels",
                dependencies=cache_root / "dependencies",
            ),
            kind_constraints=constraints,
            backend_kind_constraints=backend_constraints,
            dependency_gateway=UvDependencyCompileCli(
                index_environment=dependency_index_environment,
            ),
        )
        pack_service = JobPackService(
            output_root=cache_path(self.layout, "pack", "contexts"),
            record_root=(self.layout.state / "packages" / "materializations").resolve(),
            lease_root=cache_path(self.layout, "pack", "leases"),
            dataset_packager=ImmutableDatasetPackager(
                state_dir=cache_path(self.layout, "datasets"),
                project_root=project_source.package.root,
                input_root=self.layout.root,
                code_snapshot_digest=project_source.digest,
            ),
            environment_packager=environment_packager,
        )
        project_config = _project_config_bundle(
            self.layout,
            self.work_package_path,
            self.prepared,
            self.catalog,
        )
        if project_config.digest != self.project_config_digest:
            raise ContractError("selected project configuration changed after planning; run job plan again")
        context = pack_service.pack(
            self.pack_plan,
            JobPackInputs(
                framework_source=framework_package,
                framework_wheels=framework_wheels,
                project_source=project_source.package,
                resolved_inputs=dict(self.prepared.spec.resolved_inputs),
                project_config=project_config,
                activation_resource_sources=activation_resource_sources(
                    environment_bindings(self.prepared.seats),
                    project_root=self.layout.root,
                ),
                project_environment_sources=project_environment_sources,
            ),
        )
        return context

    def _local_publisher(self) -> BuildKitJobImagePublisher:
        registry = _registry(self.local_config)
        execution_environment = load_execution_environment(self.local_config)
        return BuildKitJobImagePublisher(
            bake_file=_bake_file(registry),
            receipt_root=(registry.receipt_root or (self.layout.state / "publications").resolve()),
            local_layout_root=cache_path(self.layout, "pack", "local-layouts"),
            builder=registry.buildx_builder,
            python_index_url=execution_environment.get("UV_INDEX_URL"),
        )

    def _publisher(self) -> JobImagePublisher:
        """Compose a developer publisher without changing package identity.

        This deliberately has no release-builder option: maintained forks are
        released manually outside Posttrain.  ``remote`` means only the
        optional developer actual-job build service.
        """

        choice = resolve_job_builder(self.local_config.machine, cli_override=self.builder_override)
        if choice.mode == "local":
            return self._local_publisher()
        machine = self.local_config.machine
        if machine is None:
            raise ContractError("remote job builder requires machine [services.job_builder] configuration")
        binding = machine.services.job_builder
        if binding.endpoint is None:
            raise ContractError("remote job builder requires machine [services.job_builder] configuration")
        environment = load_execution_environment(self.local_config)
        token = environment.get("POSTTRAIN_JOB_BUILDER_TOKEN")
        if token is None:
            raise ContractError("remote job builder credentials did not provide POSTTRAIN_JOB_BUILDER_TOKEN")
        bake_file = _bake_file(_registry(self.local_config))
        return RemoteJobImagePublisher(
            RemoteJobBuilderConfig(
                endpoint=binding.endpoint,
                token=token,
                release_manifest_digest=published_manifest_digest(),
                build_definition_digest=job_build_definition_digest(bake_file),
                receipt_root=(self.layout.state / "publications").resolve(),
                ca_bundle=machine.local.trust_bundle,
                request_timeout_seconds=float(binding.request_timeout_seconds),
                upload_concurrency=binding.upload_concurrency,
            )
        )

    def pack(self, *, allow_deferred_qualification: bool = False) -> PackedJobPackage:
        """Materialize the exact inputs and publish or reuse the actual-job image."""

        records = PackageMaterializationStore(
            (self.layout.state / "packages" / "materializations").resolve()
        ).resolve_all(self.pack_plan.plan_key)
        record = next(
            (
                candidate
                for candidate in records
                if candidate.publication_key
                == _publication_key(candidate.manifest, self.pack_plan.publication)
            ),
            None,
        )
        if record is not None:
            resolver = getattr(self._publisher(), "resolve", None)
            if resolver is not None:
                image = resolver(
                    JobImageResolutionRequest(
                        manifest=record.manifest,
                        publication=self.pack_plan.publication,
                        publication_key=record.publication_key,
                        allow_deferred_qualification=allow_deferred_qualification,
                    )
                )
                if image is not None:
                    context = PackedJobContext(
                        root=cache_path(self.layout, "pack", "contexts", record.package_key),
                        manifest=record.manifest,
                        context_digest=record.context_digest,
                        publication_key=record.publication_key,
                    )
                    return PackedJobPackage(self, context, image)

        context = self.materialize()
        try:
            image = self._publisher().publish(
                JobImagePublicationRequest(
                    manifest=context.manifest,
                    staged_context=context.root,
                    publication=self.pack_plan.publication,
                    allow_deferred_qualification=allow_deferred_qualification,
                    source_context_digest=context.context_digest,
                )
            )
            if image.publication_key != context.publication_key:
                raise ContractError("published image identity conflicts with the retained job context")
            return PackedJobPackage(self, context, image)
        finally:
            context.close()
            _discard_materialized_context(self.layout, context.root)

    def pack_local(
        self,
        *,
        allow_deferred_qualification: bool = False,
        local_output: Path | None = None,
    ) -> LocalPackedJobPackage:
        """Export a qualified OCI layout without publishing to a registry."""

        context = self.materialize()
        try:
            image = self._local_publisher().publish_local(
                JobImagePublicationRequest(
                    manifest=context.manifest,
                    staged_context=context.root,
                    publication=self.pack_plan.publication,
                    allow_deferred_qualification=allow_deferred_qualification,
                    local_output=local_output,
                    source_context_digest=context.context_digest,
                )
            )
            if image.publication_key != context.publication_key:
                raise ContractError("local image identity conflicts with the retained job context")
            return LocalPackedJobPackage(self, context, image)
        finally:
            context.close()
            _discard_materialized_context(self.layout, context.root)

    def pack_local_daemon(
        self,
        *,
        allow_deferred_qualification: bool = False,
        run_id: str | None = None,
    ) -> LocalDaemonPackedJobPackage:
        """Load one single-platform image into the local daemon for execution."""

        context = self.materialize()
        try:
            publisher = self._local_publisher()
            publish_local_daemon = getattr(publisher, "publish_local_daemon", None)
            if publish_local_daemon is None:
                raise ContractError("configured job image publisher does not support local daemon loading")
            image = publish_local_daemon(
                JobImagePublicationRequest(
                    manifest=context.manifest,
                    staged_context=context.root,
                    publication=self.pack_plan.publication,
                    allow_deferred_qualification=allow_deferred_qualification,
                    local_tag=(
                        f"posttrain-local:{context.publication_key}-{hashlib.sha256(run_id.encode()).hexdigest()[:16]}"
                        if run_id is not None
                        else None
                    ),
                    source_context_digest=context.context_digest,
                )
            )
            if image.publication_key != context.publication_key:
                raise ContractError("local daemon image identity conflicts with the retained job context")
            return LocalDaemonPackedJobPackage(self, context, image)
        finally:
            context.close()
            _discard_materialized_context(self.layout, context.root)


def _publication_key(manifest: JobPackageManifest, publication: ImagePublicationSpec) -> str:
    """Compute a publication key for a compact-record cache lookup."""

    return publication_key_for(manifest, publication)


def _discard_materialized_context(layout: ProjectLayout, root: Path) -> None:
    """Remove only a framework-owned retained context after publication."""

    contexts = cache_path(layout, "pack", "contexts").resolve()
    candidate = root.resolve(strict=False)
    if candidate.parent != contexts or not _PACKAGE_KEY.fullmatch(candidate.name):
        return
    if root.is_symlink() or not root.is_dir():
        return
    shutil.rmtree(root)


@dataclass(frozen=True, slots=True)
class PackedJobPackage:
    """Published immutable capsule with no provider or scheduling identity."""

    planned: PlannedJobPackage
    context: PackedJobContext
    image: PublishedJobImage


@dataclass(frozen=True, slots=True)
class LocalPackedJobPackage:
    """Qualified local OCI layout; deliberately not launchable by a provider."""

    planned: PlannedJobPackage
    context: PackedJobContext
    image: LocalPublishedJobImage


@dataclass(frozen=True, slots=True)
class LocalDaemonPackedJobPackage:
    """Daemon-loaded image handle used only by a local execution."""

    planned: PlannedJobPackage
    context: PackedJobContext
    image: LocalDaemonJobImage


@dataclass(frozen=True, slots=True)
class PlannedJobExecution:
    """Launch plan composed over one immutable capsule plan."""

    package: PlannedJobPackage
    launch: PlannedJobLaunch

    @property
    def target(self) -> ExecutionTarget:
        return self.package.target

    @property
    def settings(self) -> ResolvedExecutionSettings:
        return self.launch.settings

    @property
    def mounts(self) -> tuple[ExecutionMount, ...]:
        return self.launch.mounts

    def pack(self, *, allow_deferred_qualification: bool = False) -> PackedJobExecution:
        publisher_supports_daemon = self.settings.provider == "local" and hasattr(
            self.package._publisher(), "publish_local_daemon"
        )
        if publisher_supports_daemon:
            packed = self.package.pack_local_daemon(
                allow_deferred_qualification=allow_deferred_qualification,
                run_id=self.launch.run_spec.run_id,
            )
        else:
            packed = self.package.pack(allow_deferred_qualification=allow_deferred_qualification)
        return PackedJobExecution(self, packed.context, packed.image)


@dataclass(frozen=True, slots=True)
class PlannedJobLaunch:
    """One run identity and provider policy composed over a capsule plan."""

    run_spec: RunSpec
    settings: ResolvedExecutionSettings
    mounts: tuple[ExecutionMount, ...]


@dataclass(frozen=True, slots=True)
class PackedJobExecution:
    planned: PlannedJobExecution
    context: PackedJobContext
    image: PublishedJobImage | LocalDaemonJobImage

    def prepare_submission(self) -> PreparedJobSubmission:
        return _prepared_submission(self)


@dataclass(frozen=True, slots=True)
class PreparedJobSubmission:
    packed: PackedJobExecution
    request: ExecutionRequest
    provider_plan: ExecutionPlan
    service: JobExecutionService
    evidence_source: ExecutionEvidenceSource | None
    provider_source: ExecutionProviderSource


def with_recovery_checkpoint(
    planned: PlannedJobExecution,
    *,
    source_run_id: str,
    artifact: ArtifactLink,
) -> PlannedJobExecution:
    """Bind one immutable recovery artifact to a new training run."""

    spec = planned.launch.run_spec
    if not spec.job_kind.startswith("train."):
        raise ContractError("recovery checkpoints can only resume training jobs")
    if source_run_id == spec.run_id:
        raise ContractError("a recovery run must use a new run identity")
    if artifact.direction != "output" or artifact.kind != "training-checkpoint":
        raise ContractError("resume input must be an output training-checkpoint artifact")
    if "recovery_checkpoint" in spec.artifacts:
        raise ContractError("run already has a selected recovery checkpoint")
    stored = artifact.artifact
    reference = StoredArtifactRef(
        provider=stored.provider,
        namespace=stored.namespace,
        name=stored.name,
        version=stored.version,
        digest=stored.digest,
        provider_metadata=stored.provider_metadata,
    )
    rebound_spec = replace(
        spec,
        artifacts={**dict(spec.artifacts), "recovery_checkpoint": ArtifactInput(reference, artifact.kind)},
        resolved_inputs={
            **dict(spec.resolved_inputs),
            "recovery_checkpoint": {
                "source_run_id": source_run_id,
                "logical_name": artifact.logical_name,
                "provider": stored.provider,
                "namespace": stored.namespace,
                "name": stored.name,
                "version": stored.version,
                "digest": stored.digest,
            },
        },
    )
    return replace(planned, launch=replace(planned.launch, run_spec=rebound_spec))


def with_model_checkpoint(
    planned: PlannedJobExecution,
    *,
    source_run_id: str,
    artifact: ArtifactLink,
    model_seat: str = "model",
    replace_existing: bool = False,
) -> PlannedJobExecution:
    """Bind one immutable loadable model view to a new train/eval/serve run.

    An explicit ``--model-from-run`` selection may replace the package's
    catalog model input, but callers must opt into that replacement. Keeping
    the default strict protects library callers from silently rebinding a
    planned run while still giving the CLI an intentional override path.
    """

    spec = planned.launch.run_spec
    if not spec.job_kind.startswith(("train.", "eval.", "serve.")):
        raise ContractError("model sources can only start train, eval, or serve jobs")
    if source_run_id == spec.run_id:
        raise ContractError("a model-source run must use a new run identity")
    if not model_seat.strip():
        raise ContractError("model source seat cannot be empty")
    if artifact.direction != "output" or artifact.kind not in {"model-adapter", "model-weights"}:
        raise ContractError("model source must be an output model-adapter or model-weights artifact")
    stored = artifact.artifact
    if stored.digest is None:
        raise ContractError("model source must have a committed content digest")
    input_name = "model_adapter" if artifact.kind == "model-adapter" else "model_weights"
    if input_name in spec.artifacts and not replace_existing:
        raise ContractError(f"run already has a selected {input_name} model source")
    reference = StoredArtifactRef(
        provider=stored.provider,
        namespace=stored.namespace,
        name=stored.name,
        version=stored.version,
        digest=stored.digest,
        provider_metadata=stored.provider_metadata,
    )
    rebound_spec = replace(
        spec,
        artifacts={**dict(spec.artifacts), input_name: ArtifactInput(reference, artifact.kind)},
        resolved_inputs={
            **dict(spec.resolved_inputs),
            "model_source": {
                "source_run_id": source_run_id,
                "logical_name": artifact.logical_name,
                "kind": artifact.kind,
                "provider": stored.provider,
                "namespace": stored.namespace,
                "name": stored.name,
                "version": stored.version,
                "digest": stored.digest,
                "checkpoint_step": stored.provider_metadata.get("checkpoint_step"),
                "model_seat": model_seat,
            },
        },
    )
    return replace(planned, launch=replace(planned.launch, run_spec=rebound_spec))


def plan_job_execution(
    state: CliState,
    path: Path,
    *,
    job: str,
    overrides: ExecutionOverrides = _EMPTY_OVERRIDES,
    registry_prefix: str | None = None,
    run_id: str | None = None,
    host: str | None = None,
    entry: str | None = None,
    project_packages: tuple[str, ...] | None = None,
    source_includes: tuple[str, ...] | None = None,
    intent: JobIntent | None = None,
    env_file: Path | None = None,
    framework_wheelhouse: Path | None = None,
    builder: str | None = None,
) -> PlannedJobExecution:
    """Resolve and hash one job without materializing or submitting it."""

    package = plan_job_package(
        state,
        path,
        job=job,
        overrides=PackageOverrides(
            target=overrides.target,
            runtime_profile=overrides.runtime_profile,
            registry_prefix=registry_prefix,
        ),
        host=host,
        entry=entry,
        project_packages=project_packages,
        source_includes=source_includes,
        intent=intent,
        env_file=env_file,
        framework_wheelhouse=framework_wheelhouse,
        builder=builder,
    )
    return PlannedJobExecution(
        package=package,
        launch=plan_job_launch(
            package,
            overrides=LaunchOverrides(
                provider=overrides.provider,
                timeout_seconds=overrides.timeout_seconds,
                max_attempts=overrides.max_attempts,
                priority=overrides.priority,
                environment_names=overrides.environment_names,
            ),
            run_id=run_id,
        ),
    )


def plan_job_package(
    state: CliState,
    path: Path,
    *,
    job: str,
    overrides: PackageOverrides = _EMPTY_PACKAGE_OVERRIDES,
    host: str | None = None,
    entry: str | None = None,
    project_packages: tuple[str, ...] | None = None,
    source_includes: tuple[str, ...] | None = None,
    intent: JobIntent | None = None,
    env_file: Path | None = None,
    local_publication: bool = False,
    framework_wheelhouse: Path | None = None,
    builder: str | None = None,
) -> PlannedJobPackage:
    """Resolve capsule bytes without requiring a provider or worker storage."""

    if intent is not None:
        if intent.job_id != job:
            raise ContractError("public job intent does not match the requested job")
        return _plan_job_package_from_intent(
            intent,
            overrides=overrides.as_execution_overrides(),
            registry_prefix=overrides.registry_prefix,
            project_packages=project_packages,
            source_includes=source_includes,
            env_file=env_file,
            local_publication=local_publication,
            framework_wheelhouse=framework_wheelhouse,
            builder=builder,
        )
    return _plan_job_package(
        state,
        path,
        job=job,
        overrides=overrides.as_execution_overrides(),
        registry_prefix=overrides.registry_prefix,
        host=host,
        entry=entry,
        project_packages=project_packages,
        source_includes=source_includes,
        env_file=env_file,
        local_publication=local_publication,
        framework_wheelhouse=framework_wheelhouse,
        builder=builder,
    )


def plan_job_launch(
    package: PlannedJobPackage,
    *,
    overrides: LaunchOverrides = _EMPTY_LAUNCH_OVERRIDES,
    run_id: str | None = None,
) -> PlannedJobLaunch:
    """Compose scheduling policy and a fresh run ID over immutable job meaning."""

    base = resolve_execution_settings(
        package.layout.execution,
        local=package.local_config.defaults,
        cli=overrides.as_execution_overrides(),
        job=_job_defaults(
            package.layout,
            job_kind=package.prepared.recipe_job.kind,
            runtime_variant=package.pack_plan.spec.runtime_variant,
        ),
    )
    sources = dict(base.sources)
    sources["target"] = package.target_source
    sources["runtime_profile"] = package.runtime_profile_source
    settings = ResolvedExecutionSettings(
        provider=base.provider,
        target=package.target.id,
        runtime_profile=package.runtime_profile,
        timeout_seconds=base.timeout_seconds,
        max_attempts=base.max_attempts,
        priority=base.priority,
        environment_names=base.environment_names,
        sources=sources,
    )
    _validate_remote_training_timeout(package.prepared.recipe_job.kind, settings)
    selected_run_id = run_id or str(uuid.uuid4())
    run_spec = replace(package.prepared.spec, run_id=selected_run_id)
    storage = _storage(package.layout, package.local_config, settings.provider)
    return PlannedJobLaunch(
        run_spec=run_spec,
        settings=settings,
        mounts=_mounts(selected_run_id, storage),
    )


def _validate_remote_training_timeout(job_kind: str, settings: ResolvedExecutionSettings) -> None:
    """Prevent expensive remote training from silently inheriting the one-hour fallback."""

    if (
        settings.provider != "local"
        and job_kind.startswith("train.")
        and settings.sources.get("timeout_seconds") == "job"
    ):
        raise ContractError(
            "remote training requires an explicit provider wall-clock timeout; set "
            "[execution].timeout_seconds in .posttrain/project.toml or pass --timeout-seconds"
        )


def _plan_job_package(
    state: CliState,
    path: Path,
    *,
    job: str,
    overrides: ExecutionOverrides,
    registry_prefix: str | None,
    host: str | None,
    entry: str | None,
    project_packages: tuple[str, ...] | None,
    source_includes: tuple[str, ...] | None,
    env_file: Path | None,
    local_publication: bool,
    framework_wheelhouse: Path | None,
    builder: str | None,
) -> PlannedJobPackage:
    layout, catalog, work_package_path, package = load_work_package_bundle(state, path)
    context = runtime_context(
        layout=layout,
        catalog=catalog,
        path=work_package_path,
        host=host,
        entry=entry,
        activate=False,
    )
    prepared = prepare_work_package_job(context, package, job)
    return _plan_job_package_from_intent(
        JobIntent(
            layout=layout,
            catalog=catalog,
            work_package_path=work_package_path,
            work_package=package,
            job_id=job,
            context=context,
            prepared=prepared,
        ),
        overrides=overrides,
        registry_prefix=registry_prefix,
        project_packages=project_packages,
        source_includes=source_includes,
        env_file=env_file,
        local_publication=local_publication,
        framework_wheelhouse=framework_wheelhouse,
        builder=builder,
    )


def _plan_job_package_from_intent(
    intent: JobIntent,
    *,
    overrides: ExecutionOverrides,
    registry_prefix: str | None,
    project_packages: tuple[str, ...] | None,
    source_includes: tuple[str, ...] | None,
    env_file: Path | None,
    local_publication: bool,
    framework_wheelhouse: Path | None,
    builder: str | None,
) -> PlannedJobPackage:
    layout = intent.layout
    local_config = _with_registry_override(
        load_local_execution_config(layout, env_file=env_file),
        registry_prefix,
        project_id=layout.project_id,
    )
    if local_publication and local_config.registry is None:
        local_config = replace(local_config, registry=derived_local_registry())
    catalog = intent.catalog
    work_package_path = intent.work_package_path
    package = intent.work_package
    context = intent.context
    job = intent.job_id
    prepared = intent.prepared
    profile = _kind_profile(prepared.recipe_job.kind)
    inferred_variant = _runtime_variant_from_backend(prepared, profile)
    settings = resolve_execution_settings(
        layout.execution,
        local=local_config.defaults,
        cli=overrides,
        job=_job_defaults(
            layout,
            job_kind=prepared.recipe_job.kind,
            runtime_variant=inferred_variant,
        ),
    )
    if settings.target is not None:
        resolved_target = resolve_selection(catalog, "target", settings.target).value
        if not isinstance(resolved_target, ExecutionTarget):
            raise ContractError(f"execution target override did not resolve to an ExecutionTarget: {settings.target}")
        package = override_job_execution_target(
            context,
            package,
            job,
            resolved_target,
            allow_unchanged=settings.sources["target"] != "cli",
        )
        prepared = prepare_work_package_job(
            context,
            package,
            job,
        )
    registry = _registry(local_config)
    project_config = load_project_pack_config(
        layout,
        project_packages=project_packages,
        source_includes=source_includes,
    )
    _reject_dataset_inputs_in_project_source(project_config.source_includes, catalog)
    project_source_request = project_config.source_request(layout.root)
    # A supplied wheelhouse is an explicit request to pack the staged framework
    # distributions.  In particular, a maintainer qualifying a nested project
    # from inside this monorepo must not silently capture the checkout merely
    # because the CLI happens to be importable from it.
    framework_source_request = (
        None if framework_wheelhouse is not None else _framework_source_request(registry.framework_source_root)
    )
    inspector = ImmutableSourceSnapshotter(cache_root=cache_path(layout, "pack", "sources"))
    if framework_source_request is not None:
        framework_inspection = inspector.inspect_details(framework_source_request)
        framework_digest = framework_inspection.digest
        framework_distributions = None
    else:
        # No checkout: the framework is installed, so its own distributions are
        # the code that goes into the image, and their bytes are its identity.
        framework_distributions = materialize_framework_distributions(
            cache_path(layout, "pack", "framework-wheels"),
            environ=load_execution_environment(local_config),
            wheelhouse=framework_wheelhouse,
        )
        framework_digest = framework_distributions.digest
        framework_inspection = None
    project_inspection = inspector.inspect_details(project_source_request)
    project_digest = project_inspection.digest
    runtime_variant = _runtime_variant(
        inferred_variant,
        profile,
        settings.runtime_profile,
    )
    backend_runtime_identity = _backend_runtime_identity(registry, runtime_variant)
    _validate_backend_runtime_selection(prepared, runtime_variant, backend_runtime_identity)
    if not isinstance(catalog.family_registry_lock, FamilyRegistryLock):
        raise ContractError("opened project catalog has no family registry lock")
    pack_plan = plan_job_pack(
        prepared,
        framework_source_digest=framework_digest,
        project_source_digest=project_digest,
        universal_image=registry.universal_image,
        kind_image=_kind_image(registry, runtime_variant),
        publication=ImagePublicationSpec(registry.repository),
        runtime_variant=runtime_variant,
        family_registry_lock=catalog.family_registry_lock.to_payload(),
        project_root=layout.root,
        backend_runtime_identity=backend_runtime_identity,
    )
    target = _execution_target(prepared)
    if settings.runtime_profile is None:
        raise ContractError("execution runtime profile could not be resolved")
    project_config_digest = _project_config_bundle(
        layout,
        work_package_path,
        prepared,
        catalog,
    ).digest
    return PlannedJobPackage(
        layout=layout,
        catalog=catalog,
        work_package_path=work_package_path,
        prepared=prepared,
        local_config=local_config,
        pack_plan=pack_plan,
        framework_source_request=framework_source_request,
        framework_distributions=framework_distributions,
        project_source_request=project_source_request,
        project_config_digest=project_config_digest,
        target=target,
        runtime_profile=settings.runtime_profile,
        target_source=settings.sources.get("target", "job"),
        runtime_profile_source=settings.sources["runtime_profile"],
        project_source_inspection=project_inspection,
        framework_source_inspection=framework_inspection,
        dataset_source_estimates=_dataset_source_estimates(layout.root, pack_plan),
        builder_override=_validate_builder_override(builder),
    )


def _validate_builder_override(builder: str | None) -> str | None:
    if builder is not None and builder not in {"local", "remote"}:
        raise ContractError("job builder must be 'local' or 'remote'")
    return builder


def _dataset_source_estimates(layout_root: Path, pack_plan: JobPackPlan) -> tuple[dict[str, object], ...]:
    estimates: list[dict[str, object]] = []
    for request in pack_plan.spec.datasets:
        paths = project_dataset_input_paths(request.selection)
        byte_count = 0
        for relative in paths:
            candidate = (layout_root / relative).resolve()
            if not candidate.is_relative_to(layout_root.resolve()) or candidate.is_symlink():
                continue
            if candidate.is_file():
                byte_count += candidate.stat().st_size
            elif candidate.is_dir():
                byte_count += sum(
                    path.stat().st_size for path in candidate.rglob("*") if path.is_file() and not path.is_symlink()
                )
        estimates.append(
            {
                "seat_name": request.seat_name,
                "selection_id": request.selection.id,
                "selection_revision": request.selection.revision,
                "paths": list(paths),
                "byte_count": byte_count,
                "materialization_estimate": "source-inputs" if paths else "generated-or-remote",
            }
        )
    return tuple(estimates)


def _prepared_submission(packed: PackedJobExecution) -> PreparedJobSubmission:
    planned = packed.planned
    package = planned.package
    context = packed.context
    image = packed.image
    request = ExecutionRequest(
        run_spec=planned.launch.run_spec,
        job_definition_id=package.prepared.definition.id,
        image=image.image,
        local_image=(image.tag if isinstance(image, LocalDaemonJobImage) else None),
        target=planned.target,
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key=_idempotency_key(
            planned.launch.run_spec.run_id,
            context.manifest.package_key,
            image.image.value,
        ),
        policy=ExecutionPolicy(
            planned.settings.timeout_seconds,
            max_attempts=planned.settings.max_attempts,
            priority=planned.settings.priority,
        ),
        environment_names=planned.settings.environment_names,
        mounts=planned.mounts,
    )
    provider_name, provider = create_execution_provider(
        package.layout,
        planned.settings,
        package.local_config,
    )
    evidence_source = evidence_source_for_project(
        package.layout,
        environment=load_execution_environment(package.local_config),
    )
    provider_source = provider_source_for_project(package.layout, provider_name, package.local_config)
    service = JobExecutionService(
        provider,
        ExecutionSubmissionStore(package.layout.state),
        provider_name=provider_name,
        evidence_source=evidence_source,
        provider_source=provider_source,
    )
    provider_plan = service.plan(request)
    return PreparedJobSubmission(packed, request, provider_plan, service, evidence_source, provider_source)


def _registry(local_config: LocalExecutionConfig) -> RegistryBinding:
    registry = local_config.registry
    if registry is None:
        raise ContractError(
            f"job packing requires [registry] with exact image and constraint identities in {local_config.path}"
        )
    return registry


def _with_registry_override(
    local_config: LocalExecutionConfig,
    registry_prefix: str | None,
    *,
    project_id: str | None = None,
) -> LocalExecutionConfig:
    """Apply the explicit, one-invocation publication destination override."""

    if registry_prefix is None:
        return local_config
    override = derived_registry(
        environ={REGISTRY_ENVIRONMENT_VARIABLE: registry_prefix},
        project_id=project_id,
    )
    if override is None:
        raise ContractError("--registry must name a non-empty OCI registry prefix")
    registry = local_config.registry
    return replace(
        local_config,
        registry=(override if registry is None else replace(registry, repository=override.repository)),
    )


def _discover_framework_source_root() -> Path | None:
    """Return a framework checkout if one encloses this installation.

    Absence is not an error. A checkout means framework source is packed, which
    is what a framework developer wants; without one the framework is installed
    as distributions and those are staged instead.
    """
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / relative / "pyproject.toml").is_file() for relative in _FRAMEWORK_INSTALL_ROOTS):
            return candidate
    return None


def _framework_source_request(configured_root: Path | None) -> SourceSnapshotRequest | None:
    root = configured_root or _discover_framework_source_root()
    if root is None:
        return None
    missing = [relative for relative in _FRAMEWORK_INSTALL_ROOTS if not (root / relative / "pyproject.toml").is_file()]
    if missing:
        raise ContractError("framework source root does not contain required packages: " + ", ".join(missing))
    return SourceSnapshotRequest(
        root=root.resolve(),
        includes=tuple(sorted(_FRAMEWORK_SOURCE_INCLUDES)),
        install_roots=tuple(sorted(_FRAMEWORK_INSTALL_ROOTS)),
    )


def _bake_file(registry: RegistryBinding) -> Path:
    """Locate the actual-job BuildKit definition.

    The framework ships this definition as package data, so an installed
    distribution carries it and no source checkout has to be discovered.
    `registry.framework_source_root` is deliberately not consulted here: it
    selects which framework *source* gets packed into the job image, which is
    a separate question from where the image definition itself lives.
    `registry.bake_file` remains the explicit override.
    """
    selected = registry.bake_file
    if selected is None:
        selected = (cached_definition_root() / JOB_BAKE_FILE).resolve()
    if not selected.is_file():
        raise ContractError(f"actual-job BuildKit definition is missing: {selected}")
    return selected


def _project_config_bundle(
    layout: ProjectLayout,
    work_package_path: Path,
    prepared: PreparedWorkPackageJob,
    catalog: Catalog,
) -> ProjectConfigBundle:
    selected: set[Path] = {layout.manifest, work_package_path}
    if layout.project_brief is not None:
        selected.add(layout.project_brief)
    files = {_project_relative(layout, path): path.read_bytes() for path in sorted(selected)}
    roots = {seat.ref for seat in prepared.resolved.seats.values() if seat.ref is not None}
    # Execution-target overrides replace a catalog-backed training or
    # inference selection with an equivalent inline value.  The prepared
    # seat then quite correctly has no ``ref``, but the worker still resolves
    # the original work package from the packaged project catalog.  Retain
    # every source binding as a closure root so that an override cannot
    # silently omit the catalog entry the worker needs at activation time.
    source_package = load_work_package(work_package_path)
    roots.update(binding for binding in source_package.bindings.values() if isinstance(binding, CatalogRef))
    if isinstance(source_package.recipe, CatalogRef):
        roots.add(source_package.recipe)
    roots.update(catalog.refs_for_values(prepared.seats.values()))
    recipe = prepared.resolved.snapshot.get("recipe")
    if isinstance(recipe, dict) and isinstance((ref := recipe.get("ref")), dict):
        family, identifier = ref.get("family"), ref.get("id")
        if isinstance(family, str) and isinstance(identifier, str):
            roots.add(CatalogRef(family, identifier))
    closure = catalog.transitive_refs(roots)
    selected_by_overlay: dict[str, set[CatalogRef]] = {}
    for ref in closure:
        resolved = catalog.resolve(ref)
        if resolved.source_layer == "overlay":
            assert resolved.overlay_id is not None
            selected_by_overlay.setdefault(resolved.overlay_id, set()).add(ref)
    for overlay in layout.catalog_overlays:
        if not overlay.is_dir():
            raise ContractError(f"project catalog overlay is missing: {overlay}")
        manifest_path = overlay / "layer.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance((layer_id := manifest.get("layer_id")), str):
            raise ContractError(f"project catalog overlay has an invalid layer manifest: {manifest_path}")
        selected_refs = selected_by_overlay.get(layer_id, set())
        generated_files: list[str] = []
        for filename in manifest.get("files", []):
            if not isinstance(filename, str):
                raise ContractError(f"project catalog overlay has an invalid file name: {manifest_path}")
            source_path = overlay / filename
            document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ContractError(f"project catalog document is invalid: {source_path}")
            retained = {
                family: {
                    identifier: value
                    for identifier, value in entries.items()
                    if any(
                        getattr(ref, "family", None) == family and getattr(ref, "id", None) == identifier
                        for ref in selected_refs
                    )
                }
                for family, entries in document.items()
                if isinstance(family, str) and isinstance(entries, dict)
            }
            retained = {family: entries for family, entries in retained.items() if entries}
            if retained:
                generated_files.append(filename)
                files[_project_relative(layout, source_path)] = yaml.safe_dump(
                    retained, sort_keys=True, allow_unicode=True
                ).encode()
        files[_project_relative(layout, manifest_path)] = yaml.safe_dump(
            {"schema_version": 1, "layer_id": layer_id, "files": generated_files},
            sort_keys=False,
        ).encode()
    return ProjectConfigBundle(
        files=files,
        selected_work_package=_project_relative(layout, work_package_path),
    )


def _project_relative(layout: ProjectLayout, path: Path) -> str:
    try:
        return path.resolve().relative_to(layout.root).as_posix()
    except ValueError as error:
        raise ContractError(f"packed project configuration is outside the project: {path}") from error


def _reject_dataset_inputs_in_project_source(
    source_includes: tuple[str, ...],
    catalog: Catalog,
) -> None:
    """Keep every catalog-declared local dataset out of generic code source."""

    conflicts: list[tuple[str, str, str]] = []
    includes = tuple(PurePosixPath(value) for value in source_includes)
    for ref in catalog.list("dataset"):
        value = catalog.resolve(ref).value
        if not isinstance(value, DatasetLoadPlan):
            continue
        for configured in project_dataset_input_paths(value):
            path = PurePosixPath(configured)
            covering = next(
                (
                    include.as_posix()
                    for include in includes
                    if include == PurePosixPath(".") or include == path or include in path.parents
                ),
                None,
            )
            if covering is not None:
                conflicts.append((ref.id, configured, covering))
    if conflicts:
        examples = ", ".join(f"{selection} input {path} via {include}" for selection, path, include in conflicts[:3])
        extra = f" (+{len(conflicts) - 3} more)" if len(conflicts) > 3 else ""
        raise ContractError(
            "project source includes catalog dataset inputs; datasets must be "
            "packaged only through the selected job's dataset seats. Remove the "
            f"covering source include(s): {examples}{extra}"
        )


def _job_defaults(
    layout: ProjectLayout,
    *,
    job_kind: str | None = None,
    runtime_variant: str | None = None,
) -> ExecutionOverrides:
    environment_names: tuple[str, ...] = ()
    if layout.tracking == "trackio":
        environment_names = (
            "POSTTRAIN_TRACKIO_SERVER_URL",
            "TRACKIO_WRITE_TOKEN",
        )
    elif layout.tracking == "wandb":
        environment_names = ("WANDB_API_KEY", "WANDB_ENTITY")
    return ExecutionOverrides(
        provider="local",
        runtime_profile=_runtime_profile_for_job_kind(
            job_kind,
            runtime_variant=runtime_variant,
        ),
        timeout_seconds=3600,
        max_attempts=1,
        priority=0,
        environment_names=environment_names,
    )


def _runtime_profile_for_job_kind(
    job_kind: str | None,
    *,
    runtime_variant: str | None = None,
) -> str:
    selected = runtime_variant or (_kind_profile(job_kind) if job_kind is not None else "supervised")
    return f"framework/{selected}@1"


def _kind_profile(job_kind: str | None) -> str:
    if job_kind in {"train.grpo", "train.sampo", "train.distill"}:
        return "online-rl"
    if job_kind in {"eval.general", "eval.domain"}:
        return "eval"
    if job_kind in {"serve.benchmark", "serve.smoke"}:
        return "serve"
    if job_kind == "model.transform":
        return "transform"
    return "supervised"


def _runtime_variant_from_backend(
    prepared: PreparedWorkPackageJob,
    kind_profile: str,
) -> str:
    if kind_profile != "online-rl":
        return kind_profile
    training = prepared.seats.get("training")
    backend = getattr(training, "backend", None)
    if not isinstance(backend, str):
        raise ContractError("online-RL job packing requires a resolved training backend")
    product = backend.partition("@")[0].lower()
    if product == "trl":
        return "online-rl-trl-py312"
    if product == "verl":
        return "online-rl-verl-py313"
    raise ContractError(f"online-RL backend has no qualified runtime variant: {backend}")


def _runtime_variant(
    inferred: str,
    kind_profile: str,
    runtime_profile: str | None,
) -> str:
    if runtime_profile is None:
        raise ContractError("execution runtime profile could not be resolved")
    prefix = "framework/"
    suffix = "@1"
    if not runtime_profile.startswith(prefix) or not runtime_profile.endswith(suffix):
        raise ContractError("runtime profile must name a framework runtime variant such as framework/online-rl-trl@1")
    selected = runtime_profile[len(prefix) : -len(suffix)]
    if not (selected == kind_profile or selected.startswith(f"{kind_profile}-")):
        raise ContractError(f"runtime profile {runtime_profile} does not implement {kind_profile}")
    if selected != inferred:
        raise ContractError(f"runtime profile {runtime_profile} conflicts with the resolved backend variant {inferred}")
    return selected


def _kind_image(
    registry: RegistryBinding,
    runtime_variant: str,
) -> RuntimeImageRef:
    image = registry.kind_images.get(runtime_variant)
    if image is None:
        available = ", ".join(sorted(registry.kind_images)) or "none"
        raise ContractError(
            f"runtime variant {runtime_variant} is not published in this execution "
            f"configuration (available: {available})"
        )
    return image


def _backend_runtime_identity(
    registry: RegistryBinding,
    runtime_variant: str,
) -> BackendRuntimeIdentity | None:
    if runtime_variant != "online-rl-verl-py313":
        return None
    binding = registry.constraint_profiles.get(runtime_variant)
    if binding is None:
        raise ContractError(f"veRL runtime variant {runtime_variant} has no published constraint profile")
    values = (
        binding.backend_source_repository,
        binding.backend_source_revision,
        binding.backend_dependency_lock_digest,
    )
    if any(value is None for value in values):
        raise ContractError(
            "veRL runtime image has no immutable backend source identity; update the framework release before packing"
        )
    repository, revision, dependency_digest = values
    assert isinstance(repository, str) and isinstance(revision, str) and isinstance(dependency_digest, str)
    return BackendRuntimeIdentity(repository, revision, dependency_digest)


def _validate_backend_runtime_selection(
    prepared: PreparedWorkPackageJob,
    runtime_variant: str,
    identity: BackendRuntimeIdentity | None,
) -> None:
    if runtime_variant != "online-rl-verl-py313":
        return
    assert identity is not None
    training = prepared.seats.get("training")
    backend = getattr(training, "backend", None)
    options = getattr(training, "backend_options", None)
    if backend != f"verl@{identity.source_revision}":
        raise ContractError("veRL training selection differs from the immutable kind image source revision")
    if not isinstance(options, Mapping):
        raise ContractError("veRL training selection has no backend options")
    if options.get("source_revision") != identity.source_revision:
        raise ContractError("veRL training selection differs from the immutable kind image source revision")
    if options.get("dependency_lock_sha256") != identity.dependency_lock_digest:
        raise ContractError("veRL training selection differs from the immutable kind image dependency lock")


def _execution_target(prepared: PreparedWorkPackageJob) -> ExecutionTarget:
    direct = [value for value in prepared.seats.values() if isinstance(value, ExecutionTarget)]
    training = [
        target
        for name, value in prepared.seats.items()
        if name == "training" and isinstance((target := getattr(value, "target", None)), ExecutionTarget)
    ]
    candidates = training or direct
    if not candidates:
        candidates = [
            target
            for value in prepared.seats.values()
            if isinstance((target := getattr(value, "target", None)), ExecutionTarget)
        ]
    unique: list[ExecutionTarget] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) != 1:
        raise ContractError("detached execution requires one unambiguous primary ExecutionTarget")
    return unique[0]


def _storage(
    layout: ProjectLayout,
    local_config: LocalExecutionConfig,
    provider: str,
) -> ExecutionStorageBinding:
    if provider == "local":
        configured = local_config.local.storage if local_config.local is not None else None
        return configured or ExecutionStorageBinding(
            run_root=cache_path(layout, "runs"),
            model_cache=(layout.state / "cache" / "huggingface").resolve(),
            compile_cache=(layout.state / "cache" / "compile").resolve(),
        )
    if provider == "dstack":
        configured = local_config.dstack.storage if local_config.dstack is not None else None
        if configured is not None:
            warnings.warn(
                "providers.dstack.storage is deprecated; dstack worker storage "
                "is owned by the execution-dstack worker contract",
                DeprecationWarning,
                stacklevel=2,
            )
            return configured
        try:
            from posttrain_execution_dstack import (
                DSTACK_WORKER_COMPILE_CACHE,
                DSTACK_WORKER_MODEL_CACHE,
                DSTACK_WORKER_RUN_ROOT,
            )
        except ImportError as error:
            raise ContractError("dstack execution support is not installed; install posttrain[dstack]") from error
        return ExecutionStorageBinding(
            run_root=DSTACK_WORKER_RUN_ROOT,
            model_cache=DSTACK_WORKER_MODEL_CACHE,
            compile_cache=DSTACK_WORKER_COMPILE_CACHE,
        )
    raise ContractError(f"unsupported execution provider: {provider}")


def _mounts(run_id: str, storage: ExecutionStorageBinding) -> tuple[ExecutionMount, ...]:
    run_container = Path("/opt/posttrain/run") / run_id
    mounts = [ExecutionMount(storage.run_root / run_id, run_container, "run-workspace")]
    if storage.model_cache is not None:
        mounts.append(
            ExecutionMount(
                storage.model_cache,
                Path("/root/.cache/huggingface"),
                "model-cache",
            )
        )
    if storage.compile_cache is not None:
        mounts.append(
            ExecutionMount(
                storage.compile_cache,
                Path("/root/.cache/posttrain/compile"),
                "compile-cache",
            )
        )
    return tuple(mounts)


def _idempotency_key(run_id: str, package_key: str, image: str) -> str:
    payload = f"{run_id}\0{package_key}\0{image}".encode()
    return f"posttrain-{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "LocalPackedJobPackage",
    "PackedJobPackage",
    "PackedJobExecution",
    "PlannedJobPackage",
    "PlannedJobLaunch",
    "PlannedJobExecution",
    "PreparedJobSubmission",
    "plan_job_package",
    "plan_job_launch",
    "plan_job_execution",
    "with_recovery_checkpoint",
    "with_model_checkpoint",
]
