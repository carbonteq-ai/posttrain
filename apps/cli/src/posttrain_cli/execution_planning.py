"""Plan, pack, publish, and launch one immutable actual-job OCI image."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from posttrain.catalog import FamilyRegistryLock, ProjectLayout
from posttrain.common import Catalog, CatalogRef, ContractError, ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    ExecutionEvidenceSource,
    ExecutionMount,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionRequest,
    ExecutionSubmissionStore,
    JobExecutionService,
    RuntimeImageRef,
)
from posttrain.execution_pack import (
    ImagePublicationSpec,
    ImmutableDatasetPackager,
    ImmutableSourceSnapshotter,
    JobImagePublicationRequest,
    JobPackInputs,
    JobPackPlan,
    JobPackService,
    LocalPublishedJobImage,
    PackedJobContext,
    ProjectConfigBundle,
    PublishedJobImage,
    SourceSnapshotRequest,
    activation_resource_sources,
    environment_bindings,
    plan_job_pack,
)
from posttrain.project import JobIntent, load_project_pack_config
from posttrain.runtime_images import JOB_BAKE_FILE, cached_definition_root
from posttrain.tracking import RunSpec
from posttrain.work import (
    PreparedWorkPackageJob,
    override_job_execution_target,
    prepare_work_package_job,
)
from posttrain_execution_buildkit import (
    BuildKitJobImagePublisher,
    EnvironmentPackagerCacheRoots,
    ImmutableEnvironmentPackager,
    KindDependencyConstraints,
)

from .context import CliState
from .execution_config import (
    ExecutionOverrides,
    ExecutionStorageBinding,
    LaunchOverrides,
    LocalExecutionConfig,
    PackageOverrides,
    RegistryBinding,
    ResolvedExecutionSettings,
    SettingSource,
    derived_local_registry,
    load_execution_environment,
    load_local_execution_config,
    resolve_execution_settings,
)
from .execution_provider import create_execution_provider, evidence_source_for_project
from .framework_distributions import FrameworkDistributions
from .framework_distributions import materialize as materialize_framework_distributions
from .selection_resolve import resolve_selection
from .work_runtime import load_work_package_bundle, runtime_context

_EMPTY_OVERRIDES = ExecutionOverrides()
_EMPTY_PACKAGE_OVERRIDES = PackageOverrides()
_EMPTY_LAUNCH_OVERRIDES = LaunchOverrides()
_FRAMEWORK_INSTALL_ROOTS = (
    "apps/runtime",
    "packages/catalog",
    "packages/common",
    "packages/data",
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

    def materialize(self) -> PackedJobContext:
        """Materialize the immutable job context without publishing an image."""

        registry = _registry(self.local_config)
        source_root = (self.layout.state / "pack" / "sources").resolve()
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

        cache_root = (self.layout.state / "pack" / "cache").resolve()
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
        environment_packager = ImmutableEnvironmentPackager(
            cache_roots=EnvironmentPackagerCacheRoots(
                git_sources=cache_root / "git",
                wheels=cache_root / "wheels",
                dependencies=cache_root / "dependencies",
            ),
            kind_constraints=constraints,
            backend_kind_constraints=backend_constraints,
        )
        pack_service = JobPackService(
            output_root=(self.layout.state / "pack" / "contexts").resolve(),
            dataset_packager=ImmutableDatasetPackager(
                state_dir=(self.layout.state / "datasets").resolve(),
                project_root=self.layout.root,
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

    def _publisher(self) -> BuildKitJobImagePublisher:
        registry = _registry(self.local_config)
        return BuildKitJobImagePublisher(
            bake_file=_bake_file(registry),
            receipt_root=(registry.receipt_root or (self.layout.state / "pack" / "publications").resolve()),
            builder=registry.buildx_builder,
        )

    def pack(self, *, allow_deferred_qualification: bool = False) -> PackedJobPackage:
        """Materialize the exact inputs and publish or reuse the actual-job image."""

        context = self.materialize()
        image = self._publisher().publish(
            JobImagePublicationRequest(
                manifest=context.manifest,
                staged_context=context.root,
                publication=self.pack_plan.publication,
                allow_deferred_qualification=allow_deferred_qualification,
            )
        )
        if image.publication_key != context.publication_key:
            raise ContractError("published image identity conflicts with the retained job context")
        return PackedJobPackage(self, context, image)

    def pack_local(self, *, allow_deferred_qualification: bool = False) -> LocalPackedJobPackage:
        """Export a qualified OCI layout without publishing to a registry."""

        context = self.materialize()
        image = self._publisher().publish_local(
            JobImagePublicationRequest(
                manifest=context.manifest,
                staged_context=context.root,
                publication=self.pack_plan.publication,
                allow_deferred_qualification=allow_deferred_qualification,
            )
        )
        if image.publication_key != context.publication_key:
            raise ContractError("local image identity conflicts with the retained job context")
        return LocalPackedJobPackage(self, context, image)


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
    image: PublishedJobImage

    def prepare_submission(self) -> PreparedJobSubmission:
        return _prepared_submission(self)


@dataclass(frozen=True, slots=True)
class PreparedJobSubmission:
    packed: PackedJobExecution
    request: ExecutionRequest
    provider_plan: ExecutionPlan
    service: JobExecutionService
    evidence_source: ExecutionEvidenceSource | None


def plan_job_execution(
    state: CliState,
    path: Path,
    *,
    job: str,
    overrides: ExecutionOverrides = _EMPTY_OVERRIDES,
    run_id: str | None = None,
    host: str | None = None,
    entry: str | None = None,
    project_packages: tuple[str, ...] | None = None,
    source_includes: tuple[str, ...] | None = None,
    intent: JobIntent | None = None,
    env_file: Path | None = None,
    framework_wheelhouse: Path | None = None,
) -> PlannedJobExecution:
    """Resolve and hash one job without materializing or submitting it."""

    package = plan_job_package(
        state,
        path,
        job=job,
        overrides=PackageOverrides(
            target=overrides.target,
            runtime_profile=overrides.runtime_profile,
        ),
        host=host,
        entry=entry,
        project_packages=project_packages,
        source_includes=source_includes,
        intent=intent,
        env_file=env_file,
        framework_wheelhouse=framework_wheelhouse,
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
) -> PlannedJobPackage:
    """Resolve capsule bytes without requiring a provider or worker storage."""

    if intent is not None:
        if intent.job_id != job:
            raise ContractError("public job intent does not match the requested job")
        return _plan_job_package_from_intent(
            intent,
            overrides=overrides.as_execution_overrides(),
            project_packages=project_packages,
            source_includes=source_includes,
            env_file=env_file,
            local_publication=local_publication,
            framework_wheelhouse=framework_wheelhouse,
        )
    return _plan_job_package(
        state,
        path,
        job=job,
        overrides=overrides.as_execution_overrides(),
        host=host,
        entry=entry,
        project_packages=project_packages,
        source_includes=source_includes,
        env_file=env_file,
        local_publication=local_publication,
        framework_wheelhouse=framework_wheelhouse,
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
    selected_run_id = run_id or str(uuid.uuid4())
    run_spec = replace(package.prepared.spec, run_id=selected_run_id)
    storage = _storage(package.layout, package.local_config, settings.provider)
    return PlannedJobLaunch(
        run_spec=run_spec,
        settings=settings,
        mounts=_mounts(selected_run_id, storage),
    )


def _plan_job_package(
    state: CliState,
    path: Path,
    *,
    job: str,
    overrides: ExecutionOverrides,
    host: str | None,
    entry: str | None,
    project_packages: tuple[str, ...] | None,
    source_includes: tuple[str, ...] | None,
    env_file: Path | None,
    local_publication: bool,
    framework_wheelhouse: Path | None,
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
        project_packages=project_packages,
        source_includes=source_includes,
        env_file=env_file,
        local_publication=local_publication,
        framework_wheelhouse=framework_wheelhouse,
    )


def _plan_job_package_from_intent(
    intent: JobIntent,
    *,
    overrides: ExecutionOverrides,
    project_packages: tuple[str, ...] | None,
    source_includes: tuple[str, ...] | None,
    env_file: Path | None,
    local_publication: bool,
    framework_wheelhouse: Path | None,
) -> PlannedJobPackage:
    layout = intent.layout
    local_config = load_local_execution_config(layout, env_file=env_file)
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
    project_source_request = project_config.source_request(layout.root)
    # A supplied wheelhouse is an explicit request to pack the staged framework
    # distributions.  In particular, a maintainer qualifying a nested project
    # from inside this monorepo must not silently capture the checkout merely
    # because the CLI happens to be importable from it.
    framework_source_request = (
        None if framework_wheelhouse is not None else _framework_source_request(registry.framework_source_root)
    )
    inspector = ImmutableSourceSnapshotter(cache_root=(layout.state / "pack" / "sources").resolve())
    if framework_source_request is not None:
        framework_digest = inspector.inspect(framework_source_request)
        framework_distributions = None
    else:
        # No checkout: the framework is installed, so its own distributions are
        # the code that goes into the image, and their bytes are its identity.
        framework_distributions = materialize_framework_distributions(
            (layout.state / "pack" / "framework-wheels").resolve(),
            wheelhouse=framework_wheelhouse,
        )
        framework_digest = framework_distributions.digest
    project_digest = inspector.inspect(project_source_request)
    runtime_variant = _runtime_variant(
        inferred_variant,
        profile,
        settings.runtime_profile,
    )
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
    )


def _prepared_submission(packed: PackedJobExecution) -> PreparedJobSubmission:
    planned = packed.planned
    package = planned.package
    context = packed.context
    image = packed.image
    request = ExecutionRequest(
        run_spec=planned.launch.run_spec,
        job_definition_id=package.prepared.definition.id,
        image=image.image,
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
    service = JobExecutionService(
        provider,
        ExecutionSubmissionStore(package.layout.state),
        provider_name=provider_name,
        evidence_source=evidence_source,
    )
    provider_plan = service.plan(request)
    return PreparedJobSubmission(packed, request, provider_plan, service, evidence_source)


def _registry(local_config: LocalExecutionConfig) -> RegistryBinding:
    registry = local_config.registry
    if registry is None:
        raise ContractError(
            f"job packing requires [registry] with exact image and constraint identities in {local_config.path}"
        )
    return registry


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
            run_root=(layout.state / "runs").resolve(),
            model_cache=(layout.state / "cache" / "huggingface").resolve(),
            compile_cache=(layout.state / "cache" / "compile").resolve(),
        )
    if provider == "dstack":
        configured = local_config.dstack.storage if local_config.dstack is not None else None
        if configured is None:
            raise ContractError(f"dstack execution requires [providers.dstack.storage] in {local_config.path}")
        return configured
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
]
