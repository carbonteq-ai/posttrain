"""Qualification scenario CLI: compose exact selections into work packages and runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

from posttrain.common import (
    Catalog,
    CatalogRef,
    ExecutionTarget,
    InferenceBinding,
    LocalArtifactRef,
    ModelVariant,
    StoredArtifactRef,
    TrackioArtifactRef,
    Workload,
)
from posttrain.common.selections import Selection, SelectionFamily
from posttrain.data import (
    PreferencePairSource,
    ScoredContinuation,
    SupervisedPartitionPlan,
    partition_supervised_dataset,
)
from posttrain.eval import EnvironmentBinding, EvaluationBudget, EvaluationPlan, EvaluationResult
from posttrain.serve import BenchmarkResult, GenerationResult
from posttrain.tracking import TrackingBackend
from posttrain.train import (
    GRPOSettings,
    LoRAUpdate,
    QuantizationPlan,
    TrainingBinding,
    TrainingResult,
    TrainingRuntime,
    TransformResult,
)
from posttrain_tracking_trackio import TrackioBackend, TrackioSettings
from posttrain_tracking_wandb import WandbBackend, WandbSettings, wandb_artifact_name

from .catalog import (
    AUTOMATIONBENCH_ZAPIER_GRPO,
    LFM25_DPO,
    LFM25_SFT,
    LFM25_TRL_LORA,
    LFM25_TRL_QLORA,
    QWEN35_AUTOMATIONBENCH_GRPO_MTP,
    QWEN35_AUTOMATIONBENCH_TRL_LORA_THINKING,
    QWEN35_DISTILL,
    QWEN35_DISTILL_TRL_LORA,
    QWEN35_DPO,
    QWEN35_GRPO,
    QWEN35_GRPO_MTP,
    QWEN35_SFT_PEFT_COMPARISON,
    QWEN35_SFT_VALIDATED_SMOKE,
    QWEN35_TRL_LORA,
    QWEN35_TRL_QLORA,
    open_project_catalog,
)
from .data import GSM8KSupervisedSource, HalcyonGraphQLSupervisedSource, SmolSmolTalkSupervisedSource
from .execution import execute_run, execute_run_tracked
from .gemma4_halcyon import (
    GEMMA4_HALCYON_CANARY,
    GEMMA4_HALCYON_LORA,
    GEMMA4_HALCYON_LORA_FULL,
    GEMMA4_HALCYON_SFT,
    GEMMA_4_12B_IT,
)
from .jobs import (
    GSM8K_LFM_TRAINING_ROLLOUTS,
    GSM8K_TRAINING_ROLLOUTS,
    run_distillation,
    run_dpo_materialized,
    run_grpo_materialized,
    run_managed_evaluation,
    run_noop,
    run_online_smoke,
    run_quantization_transform,
    run_screen_benchmark,
    run_sft,
)
from .project import ProjectLayout, discover_project
from .source import resolve_git_source
from .tracking import trackio_artifact_name, verifiers_rollout
from .work_packages import (
    JobDefinition,
    Recipe,
    RecipeJob,
    WorkPackage,
    WorkPackageContext,
    distillation_definition,
    dpo_definition,
    grpo_definition,
    load_work_package,
    managed_evaluation_definition,
    model_transform_definition,
    run_work_package,
    serve_benchmark_definition,
    serve_smoke_definition,
    sft_definition,
)
from .work_packages.contracts import Stage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrain-lab")
    parser.add_argument(
        "job",
        choices=(
            "noop",
            "foundation-qwen-smoke",
            "foundation-lfm-smoke",
            "foundation-lfm-online-smoke",
            "foundation-qwen-gsm8k",
            "foundation-lfm-gsm8k",
            "gsm8k-qwen-sft-smoke",
            "gemma4-halcyon-graphql-sft-canary",
            "gemma4-halcyon-graphql-sft",
            "smoltalk-qwen-sft-validated-smoke",
            "gsm8k-lfm-sft-smoke",
            "gsm8k-qwen-preference-rollouts",
            "gsm8k-lfm-preference-rollouts",
            "gsm8k-qwen-dpo-smoke",
            "gsm8k-lfm-dpo-smoke",
            "gsm8k-qwen-grpo-smoke",
            "gsm8k-qwen-grpo-mtp-smoke",
            "gsm8k-qwen-0.8b-grpo-mtp-smoke",
            "automationbench-zapier-qwen-0.8b-grpo-mtp-smoke",
            "gsm8k-qwen-distill-smoke",
            "gsm8k-qwen-peft-eval",
            "qwen-awq-transform",
            "qwen-awq-eval",
            "qwen-rtn-transform",
            "qwen-rtn-eval",
        ),
    )
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument(
        "--tracking-backend",
        choices=("trackio", "wandb"),
        help="enable tracking with this backend; --tracked alone defaults to trackio",
    )
    parser.add_argument("--tracking-server-url")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-base-url", default=os.environ.get("WANDB_BASE_URL"))
    parser.add_argument("--project", default="posttrain-platform")
    parser.add_argument(
        "--project-root",
        type=Path,
        help="portable project root containing .posttrain/project.toml; otherwise discover upward",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        help="legacy repository root containing top-level catalog/ and work_packages/",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="disk-backed parent for ephemeral run workspaces",
    )
    parser.add_argument("--rollout-run-id")
    parser.add_argument("--rollout-project")
    parser.add_argument("--rejected-trace-id")
    parser.add_argument("--adapter-backend", choices=("trackio", "wandb"))
    parser.add_argument("--adapter-run-id")
    parser.add_argument("--adapter-version", default="v0")
    parser.add_argument("--artifact-version", default="v0")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--peft", choices=("lora", "qlora"), default="qlora")
    parser.add_argument("--training-backend", choices=("trl", "verl"), default="trl")
    parser.add_argument("--verl-python-executable", default=os.environ.get("POSTTRAIN_VERL_PYTHON"))
    parser.add_argument("--verl-working-directory", default=os.environ.get("POSTTRAIN_VERL_WORKTREE"))
    parser.add_argument("--verl-source-revision", default=os.environ.get("POSTTRAIN_VERL_REVISION"))
    parser.add_argument(
        "--verl-dependency-lock",
        type=Path,
        default=(Path(value) if (value := os.environ.get("POSTTRAIN_VERL_DEPENDENCY_LOCK")) else None),
        help="path to the isolated veRL environment lockfile recorded in run lineage",
    )
    return parser


def _selection[SelectionT: Selection](
    catalog: Catalog,
    family: SelectionFamily,
    identifier: str,
    expected: type[SelectionT],
) -> SelectionT:
    value = catalog.resolve(CatalogRef(family, identifier)).value
    if not isinstance(value, expected):
        raise TypeError(f"catalog entry {family}/{identifier} has the wrong type")
    return value


def _one_job_package(
    *,
    project_id: str,
    work_package_id: str,
    stage: Stage,
    job_id: str,
    definition: JobDefinition,
    bindings: dict[str, tuple[SelectionFamily, Selection | CatalogRef]],
) -> WorkPackage:
    return WorkPackage(
        project_id=project_id,
        work_package_id=work_package_id,
        stage=stage,
        recipe=Recipe(
            id=f"recipes/{job_id}@1",
            revision="1",
            stage=stage,
            seats={name: family for name, (family, _) in bindings.items()},
            jobs=(RecipeJob(job_id, definition.kind, definition.id),),
        ),
        bindings={name: value for name, (_, value) in bindings.items()},
        description=definition.description,
    )


def _gemma4_halcyon_canary(project_id: str) -> tuple[WorkPackage, JobDefinition]:
    """Compose the fixed lab-only Gemma 4 GraphQL SFT canary."""

    train = HalcyonGraphQLSupervisedSource("train")
    validation = HalcyonGraphQLSupervisedSource("test")
    definition = sft_definition(
        run_sft,
        definition_id="train/trl-sft-gemma4-halcyon-canary@1",
        with_validation=True,
    )
    package = _one_job_package(
        project_id=project_id,
        work_package_id="train/gemma4-12b/halcyon-graphql-sft-canary",
        stage="train",
        job_id="sft",
        definition=definition,
        bindings={
            "model": ("model", GEMMA_4_12B_IT),
            "dataset": ("dataset", train),
            "validation_dataset": ("dataset", validation),
            "settings": ("training", GEMMA4_HALCYON_CANARY),
            "training": ("training", GEMMA4_HALCYON_LORA),
        },
    )
    return (
        replace(
            package,
            metadata={
                "corpus_stage": "query_generation",
                "dataset_revision": train.revision,
                "labels": ["gemma4", "graphql", "sft", "tool-use", "stage1", "canary"],
            },
        ),
        definition,
    )


def _gemma4_halcyon_full_sft(project_id: str) -> tuple[WorkPackage, JobDefinition]:
    """Compose the fixed lab-only two-pass Gemma 4 GraphQL SFT run."""

    train = HalcyonGraphQLSupervisedSource("train")
    validation = HalcyonGraphQLSupervisedSource("test")
    definition = sft_definition(
        run_sft,
        definition_id="train/trl-sft-gemma4-halcyon@1",
        with_validation=True,
    )
    package = _one_job_package(
        project_id=project_id,
        work_package_id="train/gemma4-12b/halcyon-graphql-sft",
        stage="train",
        job_id="sft",
        definition=definition,
        bindings={
            "model": ("model", GEMMA_4_12B_IT),
            "dataset": ("dataset", train),
            "validation_dataset": ("dataset", validation),
            "settings": ("training", GEMMA4_HALCYON_SFT),
            "training": ("training", GEMMA4_HALCYON_LORA_FULL),
        },
    )
    return (
        replace(
            package,
            metadata={
                "corpus_stage": "query_generation",
                "dataset_revision": train.revision,
                "labels": ["gemma4", "graphql", "sft", "tool-use", "stage1", "full"],
            },
        ),
        definition,
    )


def _adapter_variant(
    model: ModelVariant,
    reference: StoredArtifactRef | TrackioArtifactRef,
    update_kind: str,
) -> ModelVariant:
    return replace(
        model,
        id=f"{model.id}/sft-{update_kind}-{reference.version}",
        artifact=reference,
        form="peft-adapter",
        revision=reference.version,
        digest=None,
        parent=model.id,
        provenance={
            "operation": "sft",
            "parameter_update_kind": update_kind,
            "base_model_repo_id": model.base.repo_id,
            "base_model_revision": model.base.revision,
        },
    )


def _adapter_reference(args: argparse.Namespace, model: ModelVariant) -> StoredArtifactRef | TrackioArtifactRef:
    logical_name = f"training/{model.id}/sft/{args.peft}/adapter"
    provider = args.adapter_backend or args.tracking_backend or "trackio"
    if provider == "wandb":
        if not args.wandb_entity:
            raise SystemExit("W&B adapter inputs require --wandb-entity or WANDB_ENTITY")
        if not args.adapter_run_id:
            raise SystemExit("W&B adapter inputs require --adapter-run-id")
        return StoredArtifactRef(
            provider="wandb",
            namespace=f"{args.wandb_entity}/{args.project}",
            name=wandb_artifact_name(logical_name, args.adapter_run_id),
            version=args.adapter_version,
        )
    return TrackioArtifactRef(
        args.project,
        trackio_artifact_name(logical_name),
        args.adapter_version,
    )


def _grpo_training_binding(
    args: argparse.Namespace,
    *,
    base: TrainingBinding | None = None,
) -> TrainingBinding:
    base = base or (QWEN35_TRL_LORA if args.peft == "lora" else QWEN35_TRL_QLORA)
    if args.training_backend == "trl":
        return base
    if not isinstance(base.update, LoRAUpdate):
        raise SystemExit("the qualified veRL GRPO path requires a LoRA update selection")
    missing = [
        name
        for name, value in (
            ("--verl-python-executable", args.verl_python_executable),
            ("--verl-working-directory", args.verl_working_directory),
            ("--verl-source-revision", args.verl_source_revision),
            ("--verl-dependency-lock", args.verl_dependency_lock),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"veRL GRPO requires {', '.join(missing)}")
    worktree = Path(args.verl_working_directory).resolve()
    source = resolve_git_source(worktree)
    if source.revision != args.verl_source_revision:
        raise SystemExit(f"veRL worktree is at {source.revision}, expected {args.verl_source_revision}")
    lockfile = Path(args.verl_dependency_lock).resolve()
    if not lockfile.is_file():
        raise SystemExit(f"veRL dependency lock does not exist: {lockfile}")
    source_state: dict[str, object] = {"source_dirty": source.dirty}
    if source.dirty_digest is not None:
        source_state["source_dirty_digest"] = source.dirty_digest
    return replace(
        base,
        id="training/qwen3.5-0.8b-verl-lora-mtp-qualified@1",
        backend=f"verl@{source.revision[:7]}",
        update=replace(
            base.update,
            target_modules=r".*[.](o_proj|down_proj)$",
        ),
        runtime=TrainingRuntime(
            global_batch_size=base.runtime.global_batch_size,
            nodes=1,
            devices_per_node=1,
            parameter_offload=True,
            optimizer_offload=True,
            timeout_seconds=900,
        ),
        backend_options={
            "python_executable": args.verl_python_executable,
            "working_directory": str(worktree),
            "source_revision": source.revision,
            "dependency_lock_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest(),
            "attention_implementation": "sdpa",
            **source_state,
            "hydra_overrides": [
                "actor_rollout_ref.rollout.agent.num_workers=2",
                "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=16",
                "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true",
                "actor_rollout_ref.actor.entropy_from_logits_chunk_size=256",
                "actor_rollout_ref.actor.fsdp_config.use_torch_compile=false",
                "actor_rollout_ref.model.use_fused_kernels=true",
                "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch",
                "reward.num_workers=2",
                "data.dataloader_num_workers=1",
            ],
        },
    )


def _grpo_settings_with_loop_overrides(
    settings: GRPOSettings,
    *,
    max_steps: int | None,
    max_length: int | None,
) -> GRPOSettings:
    if max_steps is None and max_length is None:
        return settings
    return replace(
        settings,
        loop=replace(
            settings.loop,
            max_steps=settings.loop.max_steps if max_steps is None else max_steps,
            max_length=settings.loop.max_length if max_length is None else max_length,
        ),
    )


def _quantized_variant(
    model: ModelVariant,
    plan: QuantizationPlan,
    reference: TrackioArtifactRef,
    output_id: str,
) -> ModelVariant:
    return replace(
        model,
        id=output_id,
        artifact=reference,
        form="weight-quantized",
        weight_precision=plan.output_weight_precision,
        revision=reference.version,
        digest=None,
        quantization={
            "method": plan.method,
            "weight_format": plan.weight_format,
            **plan.output_quantization,
        },
        parent=model.id,
        provenance={
            "operation": "transform",
            "quantization_plan_id": plan.id,
            "recipe_digest": plan.recipe_digest,
        },
    )


def _tracking_backend(args: argparse.Namespace) -> TrackingBackend | None:
    selected = args.tracking_backend or ("trackio" if args.tracked else None)
    if selected is None:
        return None
    if selected == "trackio":
        return TrackioBackend(
            TrackioSettings(
                project=args.project,
                server_url=args.tracking_server_url,
                auto_log_gpu=True,
                auto_log_cpu=True,
            )
        )
    if not args.wandb_entity:
        raise SystemExit("W&B tracking requires --wandb-entity or WANDB_ENTITY")
    return WandbBackend(
        WandbSettings(
            entity=args.wandb_entity,
            project=args.project,
            base_url=args.wandb_base_url,
        )
    )


def _project_layout(args: argparse.Namespace, project_id: str) -> ProjectLayout:
    if args.repository is not None:
        return ProjectLayout.legacy(args.repository, project_id)
    return discover_project(Path.cwd(), explicit_root=args.project_root)


def _execute_package(
    args: argparse.Namespace,
    package: WorkPackage,
    definition: JobDefinition,
    *,
    layout: ProjectLayout | None = None,
) -> object:
    selected_layout = _project_layout(args, package.project_id) if layout is None else layout
    source = resolve_git_source(selected_layout.root)
    backend = _tracking_backend(args)
    scratch_root = args.scratch_root.resolve() if args.scratch_root is not None else (selected_layout.state / "scratch")
    if scratch_root is not None:
        scratch_root.mkdir(parents=True, exist_ok=True)
    executor = (
        partial(execute_run_tracked, backend=backend, scratch_root=scratch_root)
        if backend is not None
        else partial(execute_run, scratch_root=scratch_root)
    )
    result = run_work_package(
        WorkPackageContext(
            catalog=open_project_catalog(selected_layout, scope=package.project_id),
            definitions={definition.id: definition},
            source_metadata=source.metadata(),
            executor=executor,
        ),
        package,
    )
    return result.jobs[0].value


def main() -> None:
    args = _parser().parse_args()
    project = args.project
    layout = _project_layout(args, project)
    catalog = open_project_catalog(layout, scope=project)
    qwen = _selection(catalog, "model", "models/qwen3.5-2b@bf16", ModelVariant)
    qwen_student = _selection(catalog, "model", "models/qwen3.5-0.8b@bf16", ModelVariant)
    lfm = _selection(catalog, "model", "models/lfm2.5-1.2b-thinking@bf16", ModelVariant)
    target = _selection(catalog, "target", "targets/local-cuda-8gb", ExecutionTarget)
    workload = _selection(catalog, "workload", "workloads/foundation-smoke-v1@1", Workload)

    if args.job == "foundation-qwen-smoke":
        package = load_work_package(layout.work_packages / "foundation_screen.yaml")
        definition = serve_benchmark_definition(run_screen_benchmark)
    elif args.job == "noop":
        definition = JobDefinition(
            "platform/noop@1", "data.prepare", {"target": type(target)}, lambda context, _: run_noop(context)
        )
        package = _one_job_package(
            project_id=project,
            work_package_id="screen/noop",
            stage="screen",
            job_id="noop",
            definition=definition,
            bindings={"target": ("target", target)},
        )
    elif args.job == "gemma4-halcyon-graphql-sft-canary":
        package, definition = _gemma4_halcyon_canary(project)
    elif args.job == "gemma4-halcyon-graphql-sft":
        package, definition = _gemma4_halcyon_full_sft(project)
    elif args.job in {"gsm8k-qwen-sft-smoke", "gsm8k-lfm-sft-smoke"}:
        model, settings, lora, qlora = (
            (qwen, QWEN35_SFT_PEFT_COMPARISON, QWEN35_TRL_LORA, QWEN35_TRL_QLORA)
            if args.job == "gsm8k-qwen-sft-smoke"
            else (lfm, LFM25_SFT, LFM25_TRL_LORA, LFM25_TRL_QLORA)
        )
        training = lora if args.peft == "lora" else qlora
        data = GSM8KSupervisedSource(count=2)
        definition = sft_definition(run_sft)
        package = _one_job_package(
            project_id=project,
            work_package_id=f"train/{model.id}/sft-{args.peft}-smoke",
            stage="train",
            job_id="sft",
            definition=definition,
            bindings={
                "model": ("model", model),
                "dataset": ("dataset", data),
                "settings": ("training", settings),
                "training": ("training", training),
            },
        )
    elif args.job == "smoltalk-qwen-sft-validated-smoke":
        source = SmolSmolTalkSupervisedSource(count=64)
        partitions = partition_supervised_dataset(
            source.load(),
            SupervisedPartitionPlan(
                id="smol-smoltalk/sft-smoke-partitions-v1",
                revision="1",
                validation_fraction=0.20,
                reserve_fraction=0.10,
                seed=42,
                stratify_by="source",
            ),
        )
        if partitions.validation is None:
            raise AssertionError("validated SFT partition plan produced no validation population")
        training = QWEN35_TRL_LORA if args.peft == "lora" else QWEN35_TRL_QLORA
        definition = sft_definition(
            run_sft,
            definition_id="train/trl-sft-validated@1",
            with_validation=True,
        )
        package = _one_job_package(
            project_id=project,
            work_package_id=f"train/{qwen.id}/smoltalk-sft-{args.peft}-validated-smoke",
            stage="train",
            job_id="sft",
            definition=definition,
            bindings={
                "model": ("model", qwen),
                "dataset": ("dataset", partitions.train),
                "validation_dataset": ("dataset", partitions.validation),
                "settings": ("training", QWEN35_SFT_VALIDATED_SMOKE),
                "training": ("training", training),
            },
        )
        package = replace(
            package,
            metadata={"dataset_partition_manifest": partitions.manifest.as_dict()},
        )
    elif args.job in {"gsm8k-qwen-dpo-smoke", "gsm8k-lfm-dpo-smoke"}:
        if args.rollout_run_id is None or args.rejected_trace_id is None:
            raise SystemExit("DPO smoke requires --rollout-run-id and --rejected-trace-id")
        foundation, settings, lora, qlora = (
            (qwen, QWEN35_DPO, QWEN35_TRL_LORA, QWEN35_TRL_QLORA)
            if args.job == "gsm8k-qwen-dpo-smoke"
            else (lfm, LFM25_DPO, LFM25_TRL_LORA, LFM25_TRL_QLORA)
        )
        training = lora if args.peft == "lora" else qlora
        if args.max_length is not None or args.max_steps is not None:
            settings = replace(
                settings,
                loop=replace(
                    settings.loop,
                    max_length=settings.loop.max_length if args.max_length is None else args.max_length,
                    max_steps=settings.loop.max_steps if args.max_steps is None else args.max_steps,
                ),
            )
        rollout_project = args.rollout_project or project
        rollout = verifiers_rollout(rollout_project, args.rollout_run_id, args.rejected_trace_id)
        data = PreferencePairSource(
            demonstrations=GSM8KSupervisedSource(count=1, offset=rollout.task_index),
            candidates=(
                ScoredContinuation(
                    example_id=f"train/{rollout.task_index:06d}",
                    messages=({"role": "assistant", "content": rollout.response},),
                    score=rollout.reward,
                    trace_id=rollout.trace_id,
                    metadata={
                        "source_project": rollout_project,
                        "source_run_id": args.rollout_run_id,
                        "source_trace_id": rollout.trace_id,
                    },
                ),
            ),
            id_suffix="trace-preferences",
        )
        remote = _adapter_reference(args, foundation)
        model = _adapter_variant(foundation, remote, args.peft)
        definition = dpo_definition(run_dpo_materialized)
        package = _one_job_package(
            project_id=project,
            work_package_id=f"train/{foundation.id}/dpo-{args.peft}-smoke",
            stage="train",
            job_id="dpo",
            definition=definition,
            bindings={
                "model": ("model", model),
                "dataset": ("dataset", data),
                "settings": ("training", settings),
                "training": ("training", training),
            },
        )
    elif args.job in {
        "gsm8k-qwen-grpo-smoke",
        "gsm8k-qwen-grpo-mtp-smoke",
        "gsm8k-qwen-0.8b-grpo-mtp-smoke",
        "automationbench-zapier-qwen-0.8b-grpo-mtp-smoke",
    }:
        automationbench = args.job == "automationbench-zapier-qwen-0.8b-grpo-mtp-smoke"
        small_model = automationbench or args.job == "gsm8k-qwen-0.8b-grpo-mtp-smoke"
        if args.training_backend == "verl" and not small_model:
            raise SystemExit(
                "the local 8 GiB veRL qualification is restricted to "
                "gsm8k-qwen-0.8b-grpo-mtp-smoke or "
                "automationbench-zapier-qwen-0.8b-grpo-mtp-smoke"
            )
        if automationbench:
            model = qwen_student
            settings = QWEN35_AUTOMATIONBENCH_GRPO_MTP
            inference = _selection(
                catalog,
                "inference",
                "inference/qwen3.5-0.8b-vllm-automationbench-rollout-mtp@1",
                InferenceBinding,
            )
            training_base = QWEN35_AUTOMATIONBENCH_TRL_LORA_THINKING
            environment: EnvironmentBinding | CatalogRef = CatalogRef("environment", AUTOMATIONBENCH_ZAPIER_GRPO.id)
        elif small_model:
            model = qwen_student
            settings = replace(
                QWEN35_GRPO_MTP,
                loop=replace(QWEN35_GRPO_MTP.loop, max_length=640),
                max_prompt_length=256,
                max_completion_length=384,
            )
            inference = _selection(
                catalog,
                "inference",
                "inference/qwen3.5-0.8b-vllm-distill-rollout@1",
                InferenceBinding,
            )
            inference = replace(
                inference,
                model=model,
                engine={
                    **inference.engine,
                    "disable_log_stats": False,
                    **(
                        {
                            "gpu_memory_utilization": 0.55,
                            "max_num_batched_tokens": 640,
                            "max_num_seqs": 2,
                            "free_cache_engine": True,
                            "sleep_during_optimization": True,
                            "enforce_eager": True,
                            "enable_chunked_prefill": True,
                        }
                        if args.training_backend == "verl"
                        else {}
                    ),
                    "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
                },
            )
            if args.training_backend == "verl":
                inference = replace(
                    inference,
                    engine={name: value for name, value in inference.engine.items() if name != "kv_cache_memory_bytes"},
                )
            training_base = replace(
                QWEN35_DISTILL_TRL_LORA,
                update=replace(
                    QWEN35_DISTILL_TRL_LORA.update,
                    target_modules=".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
                ),
            )
            environment = replace(
                GSM8K_TRAINING_ROLLOUTS.environment("gsm8k-train-candidates"),
                num_tasks=1,
                num_rollouts=settings.num_generations,
                parameters={"sampling_seed": 0},
                reward_components=("reward", "final_answer_conciseness"),
            )
        else:
            remote = _adapter_reference(args, qwen)
            model = _adapter_variant(qwen, remote, args.peft)
            settings = QWEN35_GRPO if args.job == "gsm8k-qwen-grpo-smoke" else QWEN35_GRPO_MTP
            inference_id = (
                "inference/qwen3.5-2b-vllm-rollout@1"
                if args.job == "gsm8k-qwen-grpo-smoke"
                else "inference/qwen3.5-2b-vllm-rollout-mtp@1"
            )
            inference = replace(_selection(catalog, "inference", inference_id, InferenceBinding), model=model)
            training_base = None
            environment = replace(
                GSM8K_TRAINING_ROLLOUTS.environment("gsm8k-train-candidates"),
                num_tasks=1,
                num_rollouts=settings.num_generations,
                parameters={"sampling_seed": 0},
                reward_components=("reward", "final_answer_conciseness"),
            )
        settings = _grpo_settings_with_loop_overrides(
            settings,
            max_steps=args.max_steps,
            max_length=args.max_length,
        )
        training = _grpo_training_binding(args, base=training_base)
        definition = grpo_definition(run_grpo_materialized)
        package = _one_job_package(
            project_id=project,
            work_package_id=(
                f"train/{qwen_student.id if small_model else qwen.id}/"
                f"grpo-{training.update.kind}-{'automationbench-zapier' if automationbench else 'gsm8k'}-smoke"
            ),
            stage="train",
            job_id="grpo",
            definition=definition,
            bindings={
                "model": ("model", model),
                "environment": ("environment", environment),
                "settings": ("training", settings),
                "training": ("training", training),
                "rollout_inference": ("inference", inference),
            },
        )
    elif args.job == "gsm8k-qwen-distill-smoke":
        rollout_inference = _selection(
            catalog,
            "inference",
            "inference/qwen3.5-0.8b-vllm-distill-rollout@1",
            InferenceBinding,
        )
        teacher_inference = _selection(
            catalog,
            "inference",
            "inference/qwen3.5-2b-vllm-teacher-score@1",
            InferenceBinding,
        )
        environment = _selection(
            catalog,
            "environment",
            "gsm8k-distill-train",
            EnvironmentBinding,
        )
        definition = distillation_definition(run_distillation)
        package = _one_job_package(
            project_id=project,
            work_package_id=f"train/{qwen_student.id}/distill-from-{qwen.id}-smoke",
            stage="train",
            job_id="distill",
            definition=definition,
            bindings={
                "student": ("model", qwen_student),
                "teacher": ("model", qwen),
                "environment": ("environment", environment),
                "settings": ("training", QWEN35_DISTILL),
                "training": ("training", QWEN35_DISTILL_TRL_LORA),
                "rollout_inference": ("inference", rollout_inference),
                "teacher_inference": ("inference", teacher_inference),
            },
        )
    elif args.job == "gsm8k-qwen-peft-eval":
        remote = TrackioArtifactRef(
            project,
            trackio_artifact_name(f"training/{qwen.id}/sft/{args.peft}/adapter"),
            args.adapter_version,
        )
        model = _adapter_variant(qwen, remote, args.peft)
        inference = replace(
            _selection(catalog, "inference", "inference/qwen3.5-2b-vllm-eval@1", InferenceBinding),
            id=f"inference/qwen3.5-2b-{args.peft}-vllm-eval@1",
            model=model,
        )
        evaluation_plan = _selection(catalog, "evaluation", "general-smoke-v1", EvaluationPlan)
        environment = evaluation_plan.environment("math-gsm8k")
        definition = managed_evaluation_definition(
            run_managed_evaluation,
            context_window=8_192,
            budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
            kind="eval.general",
            definition_id="eval/verifiers-managed-general@1",
        )
        package = _one_job_package(
            project_id=project,
            work_package_id=f"qualify/qwen3.5-2b/sft-{args.peft}",
            stage="qualify",
            job_id="managed-eval",
            definition=definition,
            bindings={
                "model": ("model", model),
                "evaluation_inference": ("inference", inference),
                "target": ("target", target),
                "evaluation_plan": ("evaluation", evaluation_plan),
                "environment": ("environment", environment),
            },
        )
    elif args.job == "qwen-awq-transform":
        quantization_plan = _selection(catalog, "quantization", "qwen3.5-2b/awq-4bit-v1", QuantizationPlan)
        output_id = "models/qwen3.5-2b@awq-int4-v1"
        definition = model_transform_definition(run_quantization_transform, output_id=output_id)
        package = _one_job_package(
            project_id=project,
            work_package_id="train/qwen3.5-2b/awq-int4-v1",
            stage="train",
            job_id="transform",
            definition=definition,
            bindings={
                "model": ("model", qwen),
                "quantization": ("quantization", quantization_plan),
                "target": ("target", target),
            },
        )
    elif args.job == "qwen-rtn-transform":
        quantization_plan = _selection(catalog, "quantization", "qwen3.5-2b/rtn-w4a16-v3", QuantizationPlan)
        output_id = "models/qwen3.5-2b@rtn-w4a16-v3"
        definition = model_transform_definition(run_quantization_transform, output_id=output_id)
        package = _one_job_package(
            project_id=project,
            work_package_id="train/qwen3.5-2b/rtn-w4a16-v3",
            stage="train",
            job_id="transform",
            definition=definition,
            bindings={
                "model": ("model", qwen),
                "quantization": ("quantization", quantization_plan),
                "target": ("target", target),
            },
        )
    elif args.job == "qwen-awq-eval":
        quantization_plan = _selection(catalog, "quantization", "qwen3.5-2b/awq-4bit-v1", QuantizationPlan)
        output_id = "models/qwen3.5-2b@awq-int4-v1"
        remote = TrackioArtifactRef(
            project,
            trackio_artifact_name(f"training/{output_id}/weights"),
            args.artifact_version,
        )
        model = _quantized_variant(qwen, quantization_plan, remote, output_id)
        inference = replace(
            _selection(catalog, "inference", "inference/qwen3.5-2b-vllm-eval@1", InferenceBinding),
            id="inference/qwen3.5-2b-awq-vllm-eval@1",
            model=model,
        )
        evaluation_plan = _selection(catalog, "evaluation", "general-smoke-v1", EvaluationPlan)
        environment = evaluation_plan.environment("math-gsm8k")
        definition = managed_evaluation_definition(
            run_managed_evaluation,
            context_window=8_192,
            budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
            kind="eval.general",
            definition_id="eval/verifiers-managed-general@1",
        )
        package = _one_job_package(
            project_id=project,
            work_package_id="qualify/qwen3.5-2b/awq-int4-v1",
            stage="qualify",
            job_id="managed-eval",
            definition=definition,
            bindings={
                "model": ("model", model),
                "evaluation_inference": ("inference", inference),
                "target": ("target", target),
                "evaluation_plan": ("evaluation", evaluation_plan),
                "environment": ("environment", environment),
            },
        )
    elif args.job == "qwen-rtn-eval":
        quantization_plan = _selection(catalog, "quantization", "qwen3.5-2b/rtn-w4a16-v3", QuantizationPlan)
        output_id = "models/qwen3.5-2b@rtn-w4a16-v3"
        remote = TrackioArtifactRef(
            project,
            trackio_artifact_name(f"training/{output_id}/weights"),
            args.artifact_version,
        )
        model = _quantized_variant(qwen, quantization_plan, remote, output_id)
        inference = replace(
            _selection(catalog, "inference", "inference/qwen3.5-2b-vllm-eval@1", InferenceBinding),
            id="inference/qwen3.5-2b-rtn-vllm-eval@1",
            model=model,
        )
        evaluation_plan = _selection(catalog, "evaluation", "general-smoke-v1", EvaluationPlan)
        environment = evaluation_plan.environment("math-gsm8k")
        definition = managed_evaluation_definition(
            run_managed_evaluation,
            context_window=8_192,
            budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
            kind="eval.general",
            definition_id="eval/verifiers-managed-general@1",
        )
        package = _one_job_package(
            project_id=project,
            work_package_id="qualify/qwen3.5-2b/rtn-w4a16-v3",
            stage="qualify",
            job_id="managed-eval",
            definition=definition,
            bindings={
                "model": ("model", model),
                "evaluation_inference": ("inference", inference),
                "target": ("target", target),
                "evaluation_plan": ("evaluation", evaluation_plan),
                "environment": ("environment", environment),
            },
        )
    elif args.job == "foundation-lfm-online-smoke":
        inference = _selection(catalog, "inference", "inference/lfm2.5-1.2b-vllm-screen@1", InferenceBinding)
        definition = serve_smoke_definition(run_online_smoke)
        package = _one_job_package(
            project_id=project,
            work_package_id="screen/lfm2.5-online-smoke",
            stage="screen",
            job_id="serve-smoke",
            definition=definition,
            bindings={"inference": ("inference", inference)},
        )
    elif args.job == "foundation-lfm-smoke":
        inference = _selection(catalog, "inference", "inference/lfm2.5-1.2b-vllm-screen@1", InferenceBinding)
        definition = serve_benchmark_definition(run_screen_benchmark)
        package = _one_job_package(
            project_id=project,
            work_package_id="screen/lfm2.5-benchmark-smoke",
            stage="screen",
            job_id="benchmark",
            definition=definition,
            bindings={
                "model": ("model", lfm),
                "screen_inference": ("inference", inference),
                "workload": ("workload", workload),
                "target": ("target", target),
            },
        )
    else:
        is_rollout = args.job in {"gsm8k-qwen-preference-rollouts", "gsm8k-lfm-preference-rollouts"}
        model = qwen if "qwen" in args.job else lfm
        inference_id = "inference/qwen3.5-2b-vllm-eval@1" if model is qwen else "inference/lfm2.5-1.2b-vllm-eval@1"
        inference = _selection(catalog, "inference", inference_id, InferenceBinding)
        plan: EvaluationPlan = (
            GSM8K_TRAINING_ROLLOUTS
            if args.job == "gsm8k-qwen-preference-rollouts"
            else GSM8K_LFM_TRAINING_ROLLOUTS
            if args.job == "gsm8k-lfm-preference-rollouts"
            else _selection(catalog, "evaluation", "general-smoke-v1", EvaluationPlan)
        )
        environment = plan.environment("gsm8k-train-candidates" if is_rollout else "math-gsm8k")
        definition = managed_evaluation_definition(
            run_managed_evaluation,
            context_window=8_192,
            budget=EvaluationBudget(num_tasks=None if is_rollout else 1, max_concurrent=1),
            kind="eval.domain" if is_rollout else "eval.general",
            definition_id=("eval/verifiers-managed-domain@1" if is_rollout else "eval/verifiers-managed-general@1"),
        )
        package = _one_job_package(
            project_id=project,
            work_package_id=f"{'train' if is_rollout else 'qualify'}/{model.id}/gsm8k",
            stage="train" if is_rollout else "qualify",
            job_id="managed-eval",
            definition=definition,
            bindings={
                "model": ("model", model),
                "evaluation_inference": ("inference", inference),
                "target": ("target", target),
                "evaluation_plan": ("evaluation", plan),
                "environment": ("environment", environment),
            },
        )

    result = _execute_package(args, package, definition, layout=layout)
    if isinstance(result, BenchmarkResult):
        print(json.dumps(result.as_json(), indent=2, sort_keys=True))
    elif isinstance(result, GenerationResult):
        print(json.dumps(result.summary(), indent=2, sort_keys=True))
    elif isinstance(result, EvaluationResult):
        print(
            json.dumps(
                {
                    "evaluation_plan_id": result.plan_id,
                    "environment_id": result.environment_id,
                    "model_variant_id": result.model_id,
                    "trace_ids": result.trace_ids,
                    "trace_sync_complete": result.synchronization.complete,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif isinstance(result, TransformResult):
        if not isinstance(result.artifact.reference, LocalArtifactRef):
            raise TypeError("model transforms must return a local artifact before promotion")
        print(
            json.dumps(
                {
                    "source_model_variant_id": result.source_model.id,
                    "output_model_variant_id": result.model.id,
                    "artifact_name": result.artifact.name,
                    "artifact_digest": result.artifact.reference.digest,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif isinstance(result, TrainingResult):
        print(
            json.dumps(
                {
                    "technique": result.technique,
                    "model_variant_id": result.model.id,
                    "model_form": result.model.form,
                    "global_step": result.summary.global_step,
                    "train_loss": result.summary.train_loss,
                    "runtime_seconds": result.summary.runtime_seconds,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(result)


if __name__ == "__main__":
    main()
