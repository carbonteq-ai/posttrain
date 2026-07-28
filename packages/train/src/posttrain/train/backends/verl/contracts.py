"""Versioned JSON contracts shared by the veRL launcher and isolated worker."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from posttrain.common import JsonValue
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VerlContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class VerlHubArtifact(VerlContract):
    kind: Literal["hub"] = "hub"
    repo_id: str
    revision: str


class VerlLocalArtifact(VerlContract):
    kind: Literal["local"] = "local"
    path: Path
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


type VerlModelArtifact = Annotated[
    VerlHubArtifact | VerlLocalArtifact,
    Field(discriminator="kind"),
]


class VerlModel(VerlContract):
    id: str
    family: str
    form: str
    artifact: VerlModelArtifact
    tokenizer_fingerprint: str | None
    renderer_contract: str


class VerlTarget(VerlContract):
    id: str
    world_size: int = Field(gt=0)


class VerlInference(VerlContract):
    id: str
    backend: str
    engine: dict[str, JsonValue]
    sampling: dict[str, JsonValue]
    target: VerlTarget


class VerlEnvironmentExample(VerlContract):
    id: str
    prompt: str
    metadata: dict[str, JsonValue]


class VerlEnvironment(VerlContract):
    id: str
    revision: str
    dataset_id: str
    dataset_revision: str
    bridge_snapshot: Path
    examples: tuple[VerlEnvironmentExample, ...]


class VerlRenderer(VerlContract):
    id: str
    implementation: str
    reasoning_mode: str


class VerlFullUpdate(VerlContract):
    kind: Literal["full"] = "full"


class VerlLoRAUpdate(VerlContract):
    kind: Literal["lora"] = "lora"
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: str


type VerlUpdate = Annotated[
    VerlFullUpdate | VerlLoRAUpdate,
    Field(discriminator="kind"),
]


class VerlLoop(VerlContract):
    max_steps: int = Field(gt=0)
    per_device_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    warmup_steps: int = Field(ge=0)
    max_grad_norm: float = Field(gt=0, allow_inf_nan=False)
    checkpoint_steps: int = Field(gt=0)
    checkpoint_limit: int = Field(gt=0)
    seed: int
    gradient_checkpointing: bool


class VerlParallelism(VerlContract):
    tensor_parallel_size: int = Field(gt=0)
    context_parallel_size: int = Field(gt=0)
    expert_parallel_size: int = Field(gt=0)


class VerlRuntime(VerlContract):
    global_batch_size: int | None = Field(default=None, gt=0)
    nodes: int = Field(default=1, gt=0)
    devices_per_node: int | None = Field(default=None, gt=0)
    parameter_offload: bool = False
    optimizer_offload: bool = False


class VerlTraining(VerlContract):
    binding_id: str
    renderer: VerlRenderer
    update: VerlUpdate
    loop: VerlLoop
    parallelism: VerlParallelism
    target: VerlTarget
    runtime: VerlRuntime
    backend_options: dict[str, JsonValue]


class VerlAlgorithm(VerlContract):
    advantage_estimator: Literal["grpo", "sampo"] = "grpo"
    num_prompts_per_step: int = Field(gt=0)
    num_generations: int = Field(gt=0)
    max_prompt_length: int = Field(gt=0)
    max_completion_length: int = Field(gt=0)
    beta: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    loss_mode: Literal["k1"] | None = None
    use_policy_gradient: bool | None = None
    use_task_rewards: bool | None = None
    temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    online_rl_algorithm: Literal["grpo", "dapo", "sampo"] | None = None
    clip_epsilon_low: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    clip_epsilon_high: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    dynamic_sampling: bool | None = None
    dynamic_sampling_max_candidate_batches: int | None = Field(default=None, gt=0)
    mask_truncated_completions: bool | None = None
    overlong_buffer_tokens: int | None = Field(default=None, gt=0)
    overlong_penalty_factor: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    discount_gamma: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    step_advantage_weight: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    advantage_normalization: Literal["mean", "mean_std"] | None = None


class VerlPayload(VerlContract):
    policy: VerlModel | None = None
    reference: VerlModel | None = None
    student: VerlModel | None = None
    teacher: VerlModel | None = None
    algorithm: VerlAlgorithm
    rollout: VerlInference
    teacher_scoring: VerlInference | None = None
    environment: VerlEnvironment
    training: VerlTraining
    resume_from: Path | None = None


class VerlLaunchManifest(VerlContract):
    """Complete, validated input to one isolated veRL worker process."""

    schema_version: Literal[3] = 3
    operation: Literal["grpo", "sampo", "distill"]
    backend: str
    backend_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    recipe_source_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    python_executable: Path
    working_directory: Path
    recipe_working_directory: Path | None = None
    output_directory: Path
    result_file: Path
    payload: VerlPayload

    @field_validator(
        "python_executable",
        "working_directory",
        "output_directory",
        "result_file",
    )
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("veRL process contract paths must be absolute")
        return value

    @field_validator("recipe_working_directory")
    @classmethod
    def _optional_absolute_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("veRL recipe process contract paths must be absolute")
        return value

    @model_validator(mode="after")
    def _operation_roles(self) -> Self:
        payload = self.payload
        if not self.result_file.is_relative_to(self.output_directory):
            raise ValueError("veRL result_file must remain inside output_directory")
        if not payload.environment.bridge_snapshot.is_relative_to(self.output_directory):
            raise ValueError("veRL bridge_snapshot must remain inside output_directory")
        if self.operation in {"grpo", "sampo"}:
            if payload.policy is None or payload.student is not None or payload.teacher is not None:
                raise ValueError("online-RL manifest requires policy and forbids student and teacher")
            if payload.teacher_scoring is not None:
                raise ValueError("online-RL manifest forbids teacher_scoring")
            if payload.algorithm.beta is None:
                raise ValueError("online-RL manifest requires algorithm.beta")
            algorithm = payload.algorithm
            if (
                algorithm.online_rl_algorithm is None
                or algorithm.clip_epsilon_low is None
                or algorithm.clip_epsilon_high is None
                or algorithm.dynamic_sampling is None
                or algorithm.mask_truncated_completions is None
                or algorithm.overlong_penalty_factor is None
            ):
                raise ValueError("online-RL manifest requires its policy objective settings")
            if self.operation == "grpo" and algorithm.advantage_estimator != "grpo":
                raise ValueError("GRPO manifest requires the GRPO advantage estimator")
            if self.operation == "sampo" and (
                algorithm.advantage_estimator != "sampo"
                or algorithm.online_rl_algorithm != "sampo"
                or algorithm.discount_gamma is None
                or algorithm.step_advantage_weight is None
                or algorithm.advantage_normalization is None
            ):
                raise ValueError("SAMPO manifest requires hierarchical advantage settings")
            uses_dynamic_recipe = algorithm.dynamic_sampling and algorithm.online_rl_algorithm in {
                "dapo",
                "sampo",
            }
            if uses_dynamic_recipe and (
                algorithm.dynamic_sampling_max_candidate_batches is None
                or self.recipe_source_revision is None
                or self.recipe_working_directory is None
            ):
                raise ValueError("veRL dynamic sampling requires a pinned recipe source")
            if not uses_dynamic_recipe and (
                self.recipe_source_revision is not None or self.recipe_working_directory is not None
            ):
                raise ValueError("veRL recipe source is only valid for dynamic online RL")
        else:
            if self.recipe_source_revision is not None or self.recipe_working_directory is not None:
                raise ValueError("distillation manifests forbid a DAPO recipe source")
            if payload.student is None or payload.teacher is None or payload.policy is not None:
                raise ValueError("distillation manifest requires student and teacher and forbids policy")
            if payload.reference is not None or payload.teacher_scoring is None:
                raise ValueError("distillation manifest requires teacher_scoring and forbids reference")
            algorithm = payload.algorithm
            if (
                algorithm.loss_mode is None
                or algorithm.use_policy_gradient is None
                or algorithm.use_task_rewards is None
                or algorithm.temperature is None
            ):
                raise ValueError("distillation manifest requires its loss settings")
        return self

    @property
    def command(self) -> tuple[str, ...]:
        return (
            str(self.python_executable),
            "-m",
            "posttrain.train.backends.verl.worker",
            str(self.output_directory / "posttrain-verl-launch.json"),
        )

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class VerlTrainingSummary(VerlContract):
    global_step: int = Field(ge=0)
    train_loss: float = Field(allow_inf_nan=False)
    runtime_seconds: float = Field(ge=0, allow_inf_nan=False)
    samples_per_second: float = Field(ge=0, allow_inf_nan=False)
    steps_per_second: float = Field(ge=0, allow_inf_nan=False)


class VerlWorkerResult(VerlContract):
    """Validated output written by the isolated veRL worker."""

    schema_version: Literal[1] = 1
    summary: VerlTrainingSummary
    model_dir: Path
    recovery_checkpoint: Path | None
    metrics_file: Path
    retention_manifest: Path | None = None

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "VerlAlgorithm",
    "VerlEnvironment",
    "VerlEnvironmentExample",
    "VerlFullUpdate",
    "VerlHubArtifact",
    "VerlInference",
    "VerlLaunchManifest",
    "VerlLoRAUpdate",
    "VerlLocalArtifact",
    "VerlLoop",
    "VerlModel",
    "VerlParallelism",
    "VerlPayload",
    "VerlRenderer",
    "VerlRuntime",
    "VerlTarget",
    "VerlTraining",
    "VerlTrainingSummary",
    "VerlWorkerResult",
]
