"""Thin resolve, validate, and execute helper for one work package."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

from posttrain.common import (
    Catalog,
    CatalogRef,
    ContractError,
    ExecutionTarget,
    HubModelRef,
    InferenceBinding,
    JsonValue,
    LocalArtifactRef,
    ModelVariant,
    Resolved,
    RunContext,
    StoredArtifactRef,
    TrackioArtifactRef,
    Workload,
)
from posttrain.common.selections import Selection
from posttrain.data import DatasetDescriptor
from posttrain.environment import EnvironmentBinding
from posttrain.eval import EvaluationPlan
from posttrain.train import (
    DPOSettings,
    GRPOSettings,
    OnPolicyDistillationSettings,
    QuantizationPlan,
    SFTSettings,
    TrainingBinding,
    parameter_update_digest,
)

from .contracts import (
    JobDefinition,
    Recipe,
    RecipeJob,
    ResolvedSeats,
    WorkPackage,
    WorkPackageJobResult,
    WorkPackageResult,
)
from .execution import ArtifactInput, FinalizedRunResult, RunSpec, execute_run
from .project_brief import ProjectBrief, project_brief_snapshot

type RunExecutor = Callable[[RunSpec, Callable[[RunContext], object]], object]
type SeatResolver = Callable[[ResolvedSeat], Selection]


@dataclass(frozen=True, slots=True)
class ResolvedSeat:
    name: str
    value: Selection
    ref: CatalogRef | None
    source_layer: str
    overlay_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedWorkPackage:
    definition: WorkPackage
    recipe: Recipe
    seats: Mapping[str, ResolvedSeat]
    snapshot: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seats", MappingProxyType(dict(self.seats)))
        object.__setattr__(self, "snapshot", MappingProxyType(dict(self.snapshot)))

    def seat[SelectionT: Selection](self, name: str, expected: type[SelectionT]) -> SelectionT:
        try:
            value = self.seats[name].value
        except KeyError as error:
            raise ContractError(f"work package has no resolved seat {name!r}") from error
        if not isinstance(value, expected):
            raise ContractError(f"work-package seat {name!r} has the wrong selection type")
        return value


@dataclass(frozen=True, slots=True)
class PreparedWorkPackageJob:
    """Statically validated job meaning ready for local or remote activation."""

    resolved: ResolvedWorkPackage
    recipe_job: RecipeJob
    definition: JobDefinition
    seats: ResolvedSeats
    spec: RunSpec


@dataclass(frozen=True, slots=True)
class WorkPackageContext:
    catalog: Catalog
    definitions: Mapping[str, JobDefinition]
    project_brief: ProjectBrief | None = None
    source_metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    executor: RunExecutor = execute_run
    seat_resolver: SeatResolver | None = None

    def __post_init__(self) -> None:
        if len(self.definitions) != len(set(self.definitions)):
            raise ContractError("job-definition ids must be unique")
        for key, definition in self.definitions.items():
            if key != definition.id:
                raise ContractError("job-definition registry keys must match definition ids")
        object.__setattr__(self, "definitions", MappingProxyType(dict(self.definitions)))
        object.__setattr__(self, "source_metadata", MappingProxyType(dict(self.source_metadata)))


@dataclass(frozen=True, slots=True)
class WorkPackageHostRequest:
    """Resolved project values passed to an explicitly selected execution host."""

    project_id: str
    project_root: Path
    state_dir: Path
    work_package_path: Path
    catalog: Catalog
    project_brief: ProjectBrief | None = None

    def __post_init__(self) -> None:
        if self.catalog.scope != self.project_id:
            raise ContractError("host request catalog scope must match the project id")
        if not self.project_root.is_absolute() or not self.state_dir.is_absolute():
            raise ContractError("host request paths must be absolute")
        if not self.work_package_path.is_absolute():
            raise ContractError("host request work-package path must be absolute")


type WorkPackageHostFactory = Callable[[WorkPackageHostRequest], WorkPackageContext]


def resolve_work_package(catalog: Catalog, package: WorkPackage) -> ResolvedWorkPackage:
    """Resolve recipe and bound seats without starting any operation."""

    if catalog.scope != package.project_id:
        raise ContractError("catalog scope must match the work-package project")
    recipe, recipe_snapshot = _resolve_recipe(catalog, package.recipe)
    if recipe.stage != package.stage:
        raise ContractError("work-package and recipe stages must match")
    _enabled_jobs(recipe, package.enabled_optional_jobs)
    extra = set(package.bindings) - set(recipe.seats)
    if extra:
        raise ContractError(f"work package binds undeclared recipe seats: {', '.join(sorted(extra))}")

    seats: dict[str, ResolvedSeat] = {}
    snapshot: dict[str, JsonValue] = {
        "catalog": {
            "base_id": catalog.base_id,
            "overlay_ids": list(catalog.overlay_ids),
        },
        "recipe": recipe_snapshot,
    }
    for name, binding in package.bindings.items():
        family = recipe.seats[name]
        if isinstance(binding, CatalogRef):
            if binding.family != family:
                raise ContractError(f"work-package seat {name!r} expects {family}, got {binding.family}")
            resolved = catalog.resolve(binding)
            seat = ResolvedSeat(
                name,
                resolved.value,
                resolved.ref,
                resolved.source_layer,
                resolved.overlay_id,
            )
        else:
            seat = ResolvedSeat(name, binding, None, "inline")
        seats[name] = seat
        snapshot[name] = _seat_snapshot(seat)
    execution_targets = _execution_target_snapshot(seats)
    if execution_targets:
        snapshot["execution_targets"] = {
            "schema_version": 1,
            "targets": execution_targets,
        }
    snapshot["work_package"] = {
        "project_id": package.project_id,
        "work_package_id": package.work_package_id,
        "stage": package.stage,
        "description": package.description,
        "metadata": dict(package.metadata),
    }
    evaluation_plan = next(
        (seat.value for seat in seats.values() if isinstance(seat.value, EvaluationPlan)),
        None,
    )
    evaluation_environment = next(
        (seat.value for seat in seats.values() if isinstance(seat.value, EnvironmentBinding)),
        None,
    )
    if evaluation_plan is not None:
        snapshot["evaluation"] = _evaluation_contract_snapshot(evaluation_plan, evaluation_environment)
    return ResolvedWorkPackage(package, recipe, seats, snapshot)


def run_work_package(context: WorkPackageContext, package: WorkPackage) -> WorkPackageResult:
    """Run required and explicitly enabled optional jobs in recipe order."""

    resolved, enabled, prepared = _prepare_work_package(context, package)

    results: list[WorkPackageJobResult] = []
    enabled_ids = {job.id for job in enabled}
    for job in resolved.recipe.jobs:
        if job.id not in enabled_ids:
            results.append(WorkPackageJobResult(job.id, job.kind, job.definition, "not_run"))
            continue
        _, definition, seats = prepared[job.id]
        resolved_inputs = _run_snapshot(resolved, context.project_brief)
        resolved_inputs["job_definition"] = {
            "id": definition.id,
            "kind": definition.kind,
            "description": definition.description,
        }
        spec = RunSpec(
            project_id=package.project_id,
            work_package_id=package.work_package_id,
            stage=package.stage,
            job_kind=job.kind,
            job_definition_version=definition.id,
            resolved_inputs=resolved_inputs,
            source_metadata=context.source_metadata,
            artifacts=_artifact_inputs(seats),
            required_artifact_roles=definition.required_artifact_roles,
        )
        execution_value = context.executor(
            spec,
            lambda run_context, operation=definition.operation, bound=seats: operation(run_context, bound),
        )
        if isinstance(execution_value, FinalizedRunResult):
            value = execution_value.value
            published_artifacts = execution_value.published_artifacts
        else:
            value = execution_value
            published_artifacts = ()
        results.append(
            WorkPackageJobResult(
                job.id,
                job.kind,
                job.definition,
                "succeeded",
                spec.run_id,
                value,
                published_artifacts,
            )
        )
    return WorkPackageResult(package.project_id, package.work_package_id, tuple(results))


def run_work_package_job(
    context: WorkPackageContext,
    package: WorkPackage,
    job_id: str,
    *,
    run_id: str | None = None,
    prepared: PreparedWorkPackageJob | None = None,
) -> WorkPackageResult:
    """Run one enabled job, optionally preserving a prepared run binding.

    ``prepared`` is used by execution workers when a provider launch carries
    an explicit model or recovery artifact override.  The packaged work
    package remains the source of truth for static selections; the prepared
    object lets the run-scoped binding survive into the executor.
    """

    if prepared is None:
        prepared = prepare_work_package_job(
            context,
            package,
            job_id,
            run_id=run_id,
        )
    else:
        if prepared.recipe_job.id != job_id:
            raise ContractError("prepared job does not match the requested job")
        if run_id is not None and prepared.spec.run_id != run_id:
            raise ContractError("prepared job run identity does not match the requested run")
    execution_value = context.executor(
        prepared.spec,
        lambda run_context: prepared.definition.operation(run_context, prepared.seats),
    )
    if isinstance(execution_value, FinalizedRunResult):
        value = execution_value.value
        published_artifacts = execution_value.published_artifacts
    else:
        value = execution_value
        published_artifacts = ()
    result = WorkPackageJobResult(
        prepared.recipe_job.id,
        prepared.recipe_job.kind,
        prepared.recipe_job.definition,
        "succeeded",
        prepared.spec.run_id,
        value,
        published_artifacts,
    )
    return WorkPackageResult(package.project_id, package.work_package_id, (result,))


def prepare_work_package_job(
    context: WorkPackageContext,
    package: WorkPackage,
    job_id: str,
    *,
    run_id: str | None = None,
) -> PreparedWorkPackageJob:
    """Resolve and statically validate one job without activating its runtime."""

    resolved = resolve_work_package(context.catalog, package)
    enabled = {job.id: job for job in _enabled_jobs(resolved.recipe, package.enabled_optional_jobs)}
    try:
        job = enabled[job_id]
    except KeyError as error:
        raise ContractError(f"work-package job is not enabled: {job_id}") from error
    try:
        definition = context.definitions[job.definition]
    except KeyError as error:
        raise ContractError(f"job definition is not registered: {job.definition}") from error
    if definition.kind != job.kind:
        raise ContractError(
            f"recipe job {job.id!r} kind {job.kind!r} conflicts with definition kind {definition.kind!r}"
        )
    seats = _job_seats(resolved, definition, context.seat_resolver)
    if context.seat_resolver is None and definition.static_validator is not None:
        definition.static_validator(seats)
    _preflight_job(context.project_brief, job, seats)
    resolved_inputs = _run_snapshot(resolved, context.project_brief)
    resolved_inputs["job_definition"] = {
        "id": definition.id,
        "kind": definition.kind,
        "description": definition.description,
    }
    spec = RunSpec(
        project_id=package.project_id,
        work_package_id=package.work_package_id,
        stage=package.stage,
        **({"run_id": run_id} if run_id is not None else {}),
        job_kind=job.kind,
        job_definition_version=definition.id,
        resolved_inputs=resolved_inputs,
        source_metadata=context.source_metadata,
        artifacts=_artifact_inputs(seats),
        required_artifact_roles=definition.required_artifact_roles,
    )
    return PreparedWorkPackageJob(resolved, job, definition, seats, spec)


def override_job_execution_target(
    context: WorkPackageContext,
    package: WorkPackage,
    job_id: str,
    target: ExecutionTarget,
    *,
    allow_unchanged: bool = False,
) -> WorkPackage:
    """Replace the selected job's one unambiguous primary execution target."""

    prepared = prepare_work_package_job(context, package, job_id)
    training_roles = {name: value for name, value in prepared.seats.items() if isinstance(value, TrainingBinding)}
    direct_roles = {name: value for name, value in prepared.seats.items() if isinstance(value, ExecutionTarget)}
    inference_roles = {name: value for name, value in prepared.seats.items() if isinstance(value, InferenceBinding)}

    replacements: dict[str, Selection]
    if training_roles:
        # Training and rollout targets may intentionally differ. The training
        # binding is the scheduler-facing primary target for a training job.
        primary = _one_override_target(
            (value.target for value in training_roles.values()),
            "training bindings",
        )
        replacements = {name: replace(value, target=target) for name, value in training_roles.items()}
    elif direct_roles:
        # Eval and serve definitions expose an explicit target and a colocated
        # inference binding. Preserve that equality when changing the target.
        primary = _one_override_target(
            direct_roles.values(),
            "explicit target seats",
        )
        conflicting_inference = [name for name, value in inference_roles.items() if value.target != primary]
        if conflicting_inference:
            raise ContractError(
                "execution-target override is ambiguous: explicit target seats "
                "conflict with nested inference targets "
                f"({', '.join(sorted(conflicting_inference))})"
            )
        replacements = {name: target for name in direct_roles}
        replacements.update({name: replace(value, target=target) for name, value in inference_roles.items()})
    elif inference_roles:
        primary = _one_override_target(
            (value.target for value in inference_roles.values()),
            "inference bindings",
        )
        replacements = {name: replace(value, target=target) for name, value in inference_roles.items()}
    else:
        raise ContractError("selected job does not expose a supported execution target to override")

    if target == primary:
        if allow_unchanged:
            return package
        raise ContractError(
            f"execution-target override is a no-op: selected job already uses {target.id}@{target.revision}"
        )

    return replace(
        package,
        bindings={**package.bindings, **replacements},
    )


def _one_override_target(
    targets: Iterable[ExecutionTarget],
    role: str,
) -> ExecutionTarget:
    unique: list[ExecutionTarget] = []
    for target in targets:
        if target not in unique:
            unique.append(target)
    if len(unique) != 1:
        raise ContractError(f"execution-target override is ambiguous across {role}")
    return unique[0]


def validate_work_package(
    context: WorkPackageContext,
    package: WorkPackage,
) -> ResolvedWorkPackage:
    """Resolve and statically validate every enabled job without starting a run."""

    resolved, _, _ = _prepare_work_package(context, package)
    return resolved


def _prepare_work_package(
    context: WorkPackageContext,
    package: WorkPackage,
) -> tuple[
    ResolvedWorkPackage,
    tuple[RecipeJob, ...],
    dict[str, tuple[RecipeJob, JobDefinition, ResolvedSeats]],
]:
    resolved = resolve_work_package(context.catalog, package)
    enabled = _enabled_jobs(resolved.recipe, package.enabled_optional_jobs)
    prepared: dict[str, tuple[RecipeJob, JobDefinition, ResolvedSeats]] = {}
    for job in enabled:
        try:
            definition = context.definitions[job.definition]
        except KeyError as error:
            raise ContractError(f"job definition is not registered: {job.definition}") from error
        if definition.kind != job.kind:
            raise ContractError(
                f"recipe job {job.id!r} kind {job.kind!r} conflicts with definition kind {definition.kind!r}"
            )
        seats = _job_seats(resolved, definition, context.seat_resolver)
        if context.seat_resolver is None and definition.static_validator is not None:
            definition.static_validator(seats)
        _preflight_job(context.project_brief, job, seats)
        prepared[job.id] = (job, definition, seats)

    return resolved, enabled, prepared


def _run_snapshot(
    resolved: ResolvedWorkPackage,
    project_brief: ProjectBrief | None,
) -> dict[str, JsonValue]:
    snapshot = dict(resolved.snapshot)
    if project_brief is not None:
        snapshot["project_brief"] = project_brief_snapshot(project_brief)
    return snapshot


def _preflight_job(
    project_brief: ProjectBrief | None,
    job: RecipeJob,
    seats: ResolvedSeats,
) -> None:
    """Apply project policy only at the composition boundary."""

    environment = seats.get("environment")
    inference = seats.get("evaluation_inference", seats.get("rollout_inference"))
    if isinstance(environment, EnvironmentBinding) and isinstance(inference, InferenceBinding):
        missing_capabilities = sorted(
            set(environment.required_inference_capabilities).difference(inference.capabilities)
        )
        if missing_capabilities:
            raise ContractError(
                f"{job.kind} inference binding is missing environment capabilities: " + ", ".join(missing_capabilities)
            )

    requirements = project_brief.serving if project_brief is not None else None
    if job.kind != "serve.benchmark" or requirements is None:
        return

    inferences = [value for value in seats.values() if isinstance(value, InferenceBinding)]
    workloads = [value for value in seats.values() if isinstance(value, Workload)]
    targets = [value for value in seats.values() if isinstance(value, ExecutionTarget)]
    models = [value for value in seats.values() if isinstance(value, ModelVariant)]
    if len(inferences) != 1 or len(workloads) != 1:
        raise ContractError("serve.benchmark preflight requires one inference binding and one workload")

    inference = inferences[0]
    workload = workloads[0]
    if targets and any(target != inference.target for target in targets):
        raise ContractError("serve.benchmark target conflicts with its inference binding")
    if models and any(model != inference.model for model in models):
        raise ContractError("serve.benchmark model conflicts with its inference binding")

    context_window = workload.requests.get("context_window")
    if not isinstance(context_window, int) or isinstance(context_window, bool) or context_window < 1:
        raise ContractError("serve.benchmark workload context_window must be a positive integer")
    if context_window < requirements.required_context_tokens:
        raise ContractError("serve.benchmark workload context_window is below the project serving requirement")
    if inference.model.capabilities.native_context_window < requirements.required_context_tokens:
        raise ContractError("serve.benchmark model native context window is below the project serving requirement")


def _resolve_recipe(catalog: Catalog, selection: CatalogRef | Recipe) -> tuple[Recipe, JsonValue]:
    if isinstance(selection, Recipe):
        return selection, {
            "selection_id": selection.id,
            "revision": selection.revision,
            "source_layer": "inline",
            "overlay_id": None,
        }
    resolved = catalog.resolve(selection)
    if not isinstance(resolved.value, Recipe):
        raise ContractError("recipe catalog ref did not resolve to a Recipe")
    return resolved.value, _resolved_snapshot(resolved)


def _enabled_jobs(recipe: Recipe, requested: tuple[str, ...]) -> tuple[RecipeJob, ...]:
    jobs = {job.id: job for job in recipe.jobs}
    unknown = set(requested) - set(jobs)
    if unknown:
        raise ContractError(f"unknown optional job ids: {', '.join(sorted(unknown))}")
    non_optional = [job_id for job_id in requested if not jobs[job_id].optional]
    if non_optional:
        raise ContractError(f"enabled_optional_jobs contains required jobs: {', '.join(sorted(non_optional))}")
    requested_set = set(requested)
    return tuple(job for job in recipe.jobs if not job.optional or job.id in requested_set)


def _job_seats(
    package: ResolvedWorkPackage,
    definition: JobDefinition,
    resolver: SeatResolver | None = None,
) -> ResolvedSeats:
    result: dict[str, Selection] = {}
    for name, expected in definition.seats.items():
        family = package.recipe.seats.get(name)
        if family is None:
            raise ContractError(f"job definition {definition.id!r} requires undeclared recipe seat {name!r}")
        try:
            seat = package.seats[name]
        except KeyError as error:
            raise ContractError(f"enabled job definition {definition.id!r} requires unbound seat {name!r}") from error
        value = resolver(seat) if resolver is not None else seat.value
        static_expected = definition.selection_seats.get(name, expected)
        accepted = expected if resolver is not None or static_expected is expected else (expected, static_expected)
        if not isinstance(value, accepted):
            raise ContractError(f"enabled job definition {definition.id!r} received the wrong type for seat {name!r}")
        result[name] = value
    return MappingProxyType(result)


def _seat_snapshot(seat: ResolvedSeat) -> dict[str, JsonValue]:
    value = seat.value
    entry: dict[str, JsonValue] = {
        "ref": ({"family": seat.ref.family, "id": seat.ref.id} if seat.ref is not None else None),
        "selection_id": value.id,
        "revision": getattr(value, "revision", None),
        "source_layer": seat.source_layer,
        "overlay_id": seat.overlay_id,
    }
    if isinstance(value, TrainingBinding):
        entry["parameter_update_kind"] = value.update.kind
        entry["parameter_update_digest"] = parameter_update_digest(value.update)
    if isinstance(value, QuantizationPlan):
        entry["recipe"] = value.recipe
        entry["recipe_digest"] = value.recipe_digest
    details = _selection_details(value)
    if details:
        entry["resolved"] = details
    return entry


def _execution_target_snapshot(
    seats: Mapping[str, ResolvedSeat],
) -> list[JsonValue]:
    """Record every binding target once, retaining the roles that reference it."""

    by_identity: dict[tuple[str, str], tuple[ExecutionTarget, list[str]]] = {}
    for role, seat in seats.items():
        value = seat.value
        if isinstance(value, ExecutionTarget):
            target = value
        elif isinstance(value, (InferenceBinding, TrainingBinding)):
            target = value.target
        else:
            continue
        identity = (target.id, target.revision)
        if identity not in by_identity:
            by_identity[identity] = (target, [])
        by_identity[identity][1].append(role)

    result: list[JsonValue] = []
    for target, roles in by_identity.values():
        serialized_roles: list[JsonValue] = []
        for role in sorted(roles):
            serialized_roles.append(role)
        entry: dict[str, JsonValue] = {
            "selection_id": target.id,
            "revision": target.revision,
            "roles": serialized_roles,
            "device_class": target.device_class,
            "memory_gb": target.memory_gb,
            "placement": dict(target.placement),
            "host_constraints": dict(target.host_constraints),
        }
        result.append(entry)
    return result


def _selection_details(value: Selection) -> dict[str, JsonValue]:
    if isinstance(value, ModelVariant):
        artifact: dict[str, JsonValue]
        if isinstance(value.artifact, HubModelRef):
            artifact = {"kind": "huggingface", "repo_id": value.artifact.repo_id, "revision": value.artifact.revision}
        elif isinstance(value.artifact, StoredArtifactRef):
            artifact = {
                "kind": "stored",
                "provider": value.artifact.provider,
                "namespace": value.artifact.namespace,
                "name": value.artifact.name,
                "version": value.artifact.version,
                "digest": value.artifact.digest,
            }
        elif isinstance(value.artifact, TrackioArtifactRef):
            artifact = {
                "kind": "trackio",
                "project": value.artifact.project,
                "name": value.artifact.name,
                "version": value.artifact.version,
            }
        elif isinstance(value.artifact, LocalArtifactRef):
            artifact = {"kind": "local", "digest": value.artifact.digest}
        else:
            artifact = {"kind": "unknown"}
        return {
            "artifact": artifact,
            "form": value.form,
            "weight_precision": value.weight_precision,
            "family": value.family,
            "renderer": value.renderer.id,
            "parent": value.parent,
        }
    if isinstance(value, ExecutionTarget):
        return {
            "device_class": value.device_class,
            "memory_gb": value.memory_gb,
            "placement": dict(value.placement),
            "host_constraints": dict(value.host_constraints),
        }
    if isinstance(value, InferenceBinding):
        return {
            "model_variant_id": value.model.id,
            "backend": value.backend,
            "renderer": value.renderer,
            "engine": dict(value.engine),
            "sampling": dict(value.sampling),
            "target_id": value.target.id,
            "purpose": list(value.purpose),
            "startup_timeout_seconds": value.startup_timeout_seconds,
        }
    if isinstance(value, TrainingBinding):
        return {
            "backend": value.backend,
            "backend_options": dict(value.backend_options),
            "renderer": value.renderer.id,
            "parameter_update_kind": value.update.kind,
            "parameter_update": asdict(value.update),
            "parameter_update_digest": parameter_update_digest(value.update),
            "target_id": value.target.id,
            "parallelism": {
                "tensor_parallel_size": value.parallelism.tensor_parallel_size,
                "context_parallel_size": value.parallelism.context_parallel_size,
                "expert_parallel_size": value.parallelism.expert_parallel_size,
                "sequence_length_divisor": value.parallelism.sequence_length_divisor,
            },
            "runtime": asdict(value.runtime),
        }
    if isinstance(value, (SFTSettings, DPOSettings, GRPOSettings, OnPolicyDistillationSettings)):
        loop = value.loop
        details: dict[str, JsonValue] = {
            "max_steps": loop.max_steps,
            "max_length": loop.max_length,
            "per_device_batch_size": loop.per_device_batch_size,
            "gradient_accumulation_steps": loop.gradient_accumulation_steps,
            "learning_rate": loop.learning_rate,
            "seed": loop.seed,
        }
        if isinstance(value, DPOSettings):
            details.update({"beta": value.beta, "loss_kernel": value.loss_kernel})
        if isinstance(value, GRPOSettings):
            details.update(
                {
                    "beta": value.beta,
                    "num_generations": value.num_generations,
                    "max_prompt_length": value.max_prompt_length,
                    "max_completion_length": value.max_completion_length,
                    "importance_sampling_mode": value.importance_sampling_mode,
                }
            )
        if isinstance(value, OnPolicyDistillationSettings):
            details.update(
                {
                    "temperature": value.temperature,
                    "num_generations": value.num_generations,
                    "max_prompt_length": value.max_prompt_length,
                    "max_completion_length": value.max_completion_length,
                    "divergence": "sampled-token-reverse-kl",
                }
            )
        return details
    descriptor = getattr(value, "descriptor", None)
    if isinstance(descriptor, DatasetDescriptor):
        return {
            "kind": descriptor.kind,
            "schema_version": descriptor.schema_version,
            "num_examples": descriptor.num_examples,
            "metadata": dict(descriptor.metadata),
        }
    if isinstance(value, EnvironmentBinding):
        from posttrain.environment import environment_source_payload

        return {
            "category": value.category,
            "package": value.source.package,
            "source": cast(JsonValue, environment_source_payload(value.source)),
            "source_revision": value.revision,
            "activation": value.activation.to_payload(),
            "activation_digest": value.activation.digest,
            "sampling": {
                "max_tokens": value.sampling.max_tokens,
                "temperature": value.sampling.temperature,
                "top_p": value.sampling.top_p,
                "reasoning_effort": value.sampling.reasoning_effort,
            },
            "num_tasks": value.num_tasks,
            "num_rollouts": value.num_rollouts,
            "max_concurrent": value.max_concurrent,
            "parameters": dict(value.parameters),
            "reward_components": list(value.reward_components),
            "observation": {
                "primary_metric": value.observation.primary_metric,
                "primary_metric_label": value.observation.primary_metric_label,
                "pass_rate_metric": value.observation.pass_rate_metric,
                "facets": [
                    {
                        "field": facet.field,
                        "dimension": facet.dimension,
                        "label": facet.label,
                        "transform": facet.transform,
                    }
                    for facet in value.observation.facets
                ],
            },
        }
    if isinstance(value, EvaluationPlan):
        return {
            "kind": value.kind,
            "environment_ids": [environment.id for environment in value.environments],
            "contract": {
                "id": "posttrain.eval.verifiers-observation",
                "schema_version": 3,
            },
            "inference_requirements": dict(value.inference_requirements),
            "metrics_and_slices": list(value.metrics_and_slices),
            "success": {
                environment_id: {
                    "id": definition.id,
                    "label": definition.label,
                    "source": {
                        "namespace": definition.source.namespace,
                        "name": definition.source.name,
                    },
                    "predicate": {
                        "operator": definition.predicate.operator,
                        "value": definition.predicate.value,
                        "upper": definition.predicate.upper,
                        "tolerance": definition.predicate.tolerance,
                    },
                    "missing": definition.missing,
                }
                for environment_id, definition in value.success.items()
            },
            "breakdowns": {
                environment_id: [
                    {
                        "id": definition.id,
                        "label": definition.label,
                        "dimensions": list(definition.dimensions),
                        "presentation": definition.presentation,
                        "multi_value": definition.multi_value,
                        "missing": definition.missing,
                    }
                    for definition in definitions
                ]
                for environment_id, definitions in value.breakdowns.items()
            },
            "aggregation": dict(value.aggregation),
            "comparison": dict(value.comparison),
        }
    if isinstance(value, Workload):
        return {
            "requests": dict(value.requests),
            "concurrency": list(value.concurrency),
            "warmup_repetitions": value.warmup_repetitions,
            "measured_repetitions": value.measured_repetitions,
            "required_measures": list(value.required_measures),
            "plateau_improvement_ratio": value.plateau_improvement_ratio,
            "plateau_intervals": value.plateau_intervals,
            "max_consecutive_point_failures": value.max_consecutive_point_failures,
        }
    return {}


def _evaluation_contract_snapshot(
    plan: EvaluationPlan,
    environment: EnvironmentBinding | None,
) -> dict[str, JsonValue]:
    """Materialize the eval contract that Observatory must use for this run."""

    environment_revision = environment.revision if environment is not None else None
    environment_id = environment.id if environment is not None else None
    manifest: dict[str, JsonValue] = {
        "schema_version": "evaluation-signals/v1",
        "source": "catalog",
        "environment_id": environment_id,
        "environment_revision": environment_revision,
        "reward_components": list(environment.reward_components) if environment is not None else [],
        "observation": (
            {
                "primary_metric": environment.observation.primary_metric,
                "primary_metric_label": environment.observation.primary_metric_label,
                "pass_rate_metric": environment.observation.pass_rate_metric,
                "facets": [
                    {
                        "field": facet.field,
                        "dimension": facet.dimension,
                        "label": facet.label,
                        "transform": facet.transform,
                    }
                    for facet in environment.observation.facets
                ],
            }
            if environment is not None
            else {}
        ),
    }
    activation = environment.activation.to_payload() if environment is not None else {}
    activation_config = activation.get("config") if isinstance(activation, Mapping) else None
    taskset = activation_config.get("taskset") if isinstance(activation_config, Mapping) else None
    taskset_payload = taskset if isinstance(taskset, Mapping) else {}
    dataset_id = taskset_payload.get("dataset_repo") or taskset_payload.get("repository")
    dataset_revision = taskset_payload.get("dataset_revision") or taskset_payload.get("revision")
    population: dict[str, JsonValue] = {
        "taskset": cast(JsonValue, taskset_payload),
        "dataset": cast(
            JsonValue,
            {
                "id": dataset_id if isinstance(dataset_id, str) else None,
                "revision": dataset_revision if isinstance(dataset_revision, str) else None,
                "split": taskset_payload.get("split") if isinstance(taskset_payload.get("split"), str) else None,
            },
        ),
        "num_tasks": environment.num_tasks if environment is not None else 0,
        "num_rollouts": environment.num_rollouts if environment is not None else 0,
        "max_concurrent": environment.max_concurrent if environment is not None else 0,
        "parameters": dict(environment.parameters) if environment is not None else {},
    }
    success = plan.success_for(environment_id) if environment_id is not None else None
    breakdowns = plan.breakdowns_for(environment_id) if environment_id is not None else ()
    return {
        "contract": {
            "id": "posttrain.eval.verifiers-observation",
            "schema_version": 3,
        },
        "plan": {
            "id": plan.id,
            "revision": plan.revision,
            "kind": plan.kind,
            "metrics_and_slices": list(plan.metrics_and_slices),
            "success": (
                {
                    "id": success.id,
                    "label": success.label,
                    "source": {
                        "namespace": success.source.namespace,
                        "name": success.source.name,
                    },
                    "predicate": {
                        "operator": success.predicate.operator,
                        "value": success.predicate.value,
                        "upper": success.predicate.upper,
                        "tolerance": success.predicate.tolerance,
                    },
                    "missing": success.missing,
                }
                if success is not None
                else None
            ),
            "breakdowns": [
                {
                    "id": definition.id,
                    "label": definition.label,
                    "dimensions": list(definition.dimensions),
                    "presentation": definition.presentation,
                    "multi_value": definition.multi_value,
                    "missing": definition.missing,
                }
                for definition in breakdowns
            ],
            "aggregation": dict(plan.aggregation),
            "comparison": dict(plan.comparison),
        },
        "environment": {
            "id": environment_id,
            "revision": environment_revision,
            "package": environment.source.package if environment is not None else None,
            "category": environment.category if environment is not None else None,
            "source_revision": environment.revision if environment is not None else None,
        },
        "population": population,
        "signal_manifest": manifest,
        "native_evidence": {
            "schema_id": "verifiers.trace",
            "schema_version": "v1",
        },
    }


def _artifact_inputs(seats: ResolvedSeats) -> dict[str, ArtifactInput]:
    inputs: dict[str, ArtifactInput] = {}
    for value in seats.values():
        if isinstance(value, ModelVariant) and isinstance(value.artifact, (StoredArtifactRef, TrackioArtifactRef)):
            reference = (
                value.artifact
                if isinstance(value.artifact, StoredArtifactRef)
                else StoredArtifactRef(
                    provider="trackio",
                    namespace=value.artifact.project,
                    name=value.artifact.name,
                    version=value.artifact.version,
                    provider_metadata={"alias": value.artifact.alias},
                )
            )
            if value.form in {"adapter", "peft-adapter"}:
                inputs["model_adapter"] = ArtifactInput(reference, "model-adapter")
            else:
                inputs["model_weights"] = ArtifactInput(reference, "model-weights")
    return inputs


def _resolved_snapshot(resolved: Resolved[Selection]) -> dict[str, JsonValue]:
    return {
        "ref": {"family": resolved.ref.family, "id": resolved.ref.id},
        "selection_id": resolved.value.id,
        "revision": getattr(resolved.value, "revision", None),
        "source_layer": resolved.source_layer,
        "overlay_id": resolved.overlay_id,
    }


__all__ = [
    "ResolvedSeat",
    "ResolvedWorkPackage",
    "RunExecutor",
    "WorkPackageContext",
    "WorkPackageHostFactory",
    "WorkPackageHostRequest",
    "override_job_execution_target",
    "prepare_work_package_job",
    "resolve_work_package",
    "run_work_package",
    "run_work_package_job",
    "validate_work_package",
]
