"""Compose the resolved native lifecycle with ordinary synchronous TRL jobs."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from posttrain.common import RunContext

from ...online_rl import policy_sampling_from_mapping
from ...requests import CAPORequest, GDPORequest, GRPORequest, SAMPORequest
from ...update_objectives import resolve_objective_term
from ...update_plan import ExecutionCapabilities
from ...update_records import InvalidPolicyUpdate, PolicyVersions
from ...update_recovery import inspect_update_recovery
from ...update_resolution import resolve_policy_population
from ...update_sampler_correction import recipe_sampler_correction_weights
from ...update_transport import decode_population_payload
from ..policy_update_admission import AdmittedNativePopulation
from .policy_rollouts import collect_active_resolved_population, collect_resolved_population
from .policy_updates import ResolvedTRLPopulation, ResolvedTRLRun
from .update_totals import RolloutUpdateTotals

SCORE_CONTRACT = "posttrain.causal-text-tempered-logsoftmax-fp32@1"


def numerical_execution_identity() -> Mapping[str, object]:
    """Bind effective arithmetic controls as well as the selected recipe."""
    import os

    import torch

    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_LAUNCH_BLOCKING",
                "FLASH_ATTENTION_DETERMINISTIC",
            )
        },
    }


def validate_resolved_job(request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest) -> None:
    """Reject unqualified host composition before model loading or collection."""
    settings = request.settings
    if settings.policy_updates is None:
        raise InvalidPolicyUpdate("resolved job requires an explicit policy update selection")
    if request.training.runtime.nodes != 1 or request.training.runtime.devices_per_node != 1:
        raise InvalidPolicyUpdate("resolved TRL job has not qualified distributed native execution")
    if request.inference.backend.split("@", 1)[0] == "vllm":
        raise InvalidPolicyUpdate("resolved TRL job has not qualified production sampler correction")
    sampling = policy_sampling_from_mapping(getattr(request.inference, "sampling", {}), settings.max_completion_length)
    if (
        sampling.top_p != 1
        or sampling.top_k != 0
        or sampling.min_p not in {None, 0}
        or sampling.repetition_penalty != 1
        or sampling.presence_penalty != 0
    ):
        raise InvalidPolicyUpdate("resolved TRL job has not qualified correction for a warped sampling distribution")
    if (
        (getattr(settings, "active_sampling", None) is not None and not isinstance(request, SAMPORequest))
        or getattr(settings, "dynamic_sampling", None) is not None
        or getattr(settings, "adaptive_curriculum", None) is not None
    ):
        raise InvalidPolicyUpdate(
            "resolved TRL job has not qualified production filtering/refill/curriculum composition"
        )
    if settings.mask_truncated_completions or (
        settings.policy_updates.objective_variant == "semantic-spans" and not isinstance(request, SAMPORequest)
    ):
        raise InvalidPolicyUpdate("resolved TRL job requires qualified native support for selected masks")
    if request.training.backend_options.get("use_liger_kernel", False):
        raise InvalidPolicyUpdate("resolved TRL job does not support the Liger loss bypass")
    if request.training.backend_options.get("post_update_probe_rows", 0):
        raise InvalidPolicyUpdate("resolved TRL job has not qualified legacy flattened-row probes")
    if getattr(request.bridge, "policy_update_context_contract", None) != "causal-text@1" or not callable(
        getattr(request.bridge, "retain_population", None)
    ):
        raise InvalidPolicyUpdate("resolved TRL job requires original causal-text native collection")
    if isinstance(request, GRPORequest) and request.settings.algorithm not in {"grpo", "dapo"}:
        raise InvalidPolicyUpdate("resolved TRL job does not support this GRPO algorithm")


def job_identity(request: Any, tokenizer: Any, native_trainer: type) -> tuple[str, str]:
    """Bind selected job meaning and actual framework/native trainer source.

    Source hashing permits qualification of dirty candidate code without calling
    it a published release. A release still needs immutable fork/runtime adoption.
    Local checkpoint/output paths are intentionally absent from this identity.
    """
    import inspect

    def identity_value(value: Any) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            return {item.name: getattr(value, item.name) for item in fields(value)}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"resolved job identity cannot serialize {type(value).__name__}")

    template = hashlib.sha256(
        json.dumps(
            {"template": tokenizer.chat_template, "renderer": request.training.renderer},
            default=identity_value,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    train_root = Path(__file__).resolve().parents[2]
    sources = {
        str(path.relative_to(train_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(train_root.rglob("*.py"))
    }
    trainer_file = inspect.getsourcefile(native_trainer)
    if trainer_file is None:
        raise InvalidPolicyUpdate("resolved native trainer lacks inspectable source identity")
    sources["native-trainer"] = hashlib.sha256(Path(trainer_file).read_bytes()).hexdigest()
    # Checkpoint restoration and optimizer lifecycle are inherited. Hashing
    # only grpo_trainer.py would admit a changed native base implementation
    # under the same recovery identity (as exposed by the mixed PEFT layout).
    for ancestor in native_trainer.__mro__[1:]:
        if ancestor is object:
            continue
        ancestor_file = inspect.getsourcefile(ancestor)
        if ancestor_file is None:
            raise InvalidPolicyUpdate("resolved native trainer ancestor lacks inspectable source identity")
        sources[f"native-ancestor/{ancestor.__module__}.{ancestor.__qualname__}"] = hashlib.sha256(
            Path(ancestor_file).read_bytes()
        ).hexdigest()
    import importlib.metadata

    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "accelerate", "peft")
    }
    payload = {
        "schema": "posttrain.resolved-trl-job@1",
        "policy": request.policy,
        "settings": request.settings,
        "training": request.training,
        "inference": request.inference,
        "environment": request.environment,
        "template": template,
        "sources": sources,
        "versions": versions,
        "numerical_execution": numerical_execution_identity(),
    }
    runtime = hashlib.sha256(
        json.dumps(payload, default=identity_value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return f"resolved-trl-job/{runtime}", f"renderer/{template}"


def _decode_native(evidence: bytes) -> Mapping[str, Any]:
    from ...integrations.verifiers_population_artifact import decode_native_population

    if not evidence or not evidence.endswith(b"\n"):
        raise InvalidPolicyUpdate("resolved job recovery requires complete native JSONL evidence")
    first = json.loads(evidence.splitlines()[0])
    if not isinstance(first, dict):
        raise InvalidPolicyUpdate("resolved job recovery requires native record objects")
    format = "verifiers-native-episodes" if "traces" in first else "verifiers-native-traces"
    return decode_native_population(evidence, format=format)


@dataclass
class ResolvedTRLJob:
    context: RunContext
    request: GRPORequest | SAMPORequest | GDPORequest | CAPORequest
    tokenizer: Any
    rows: list[dict[str, Any]]
    runtime_identity: str
    template_revision: str
    score_temperature: float
    totals: RolloutUpdateTotals
    max_overflow_retries: int = 0
    trainer: Any = field(default=None, init=False, repr=False)
    run: ResolvedTRLRun = field(init=False)
    capabilities: ExecutionCapabilities = field(init=False)
    _observed_applied: int = field(default=0, init=False)

    def observe_applied_update(self, step: int) -> None:
        """Report the evaluated objective once at the committed native boundary.

        Native Trainer's logged loss is a windowed total, not policy loss.
        Prepared-credit statistics describe selected actions, without changing
        their objective weights or claiming that a nonzero gradient was applied.
        """
        if step == self._observed_applied:
            return
        population = self.run.active()
        evaluation = population.last_evaluation
        if (
            step != self._observed_applied + 1
            or step != population.global_applied_updates
            or population.next_update < 1
            or evaluation is None
        ):
            raise InvalidPolicyUpdate("resolved observation requires the committed applied boundary")
        update = population.updates[population.next_update - 1]
        term = resolve_objective_term(
            update, population.spec, population.credit, parameter_version=evaluation.parameter_version
        )
        if term.digest != evaluation.term_digest:
            raise InvalidPolicyUpdate("resolved observation differs from evaluated objective")
        values = {
            "train/rl/loss": float(evaluation.loss.detach()),
            "train/rl/policy_loss": float(evaluation.policy_loss.detach()),
            "train/rl/kl_loss": float(evaluation.kl_loss.detach()),
            "train/rl/applied_optimizer_updates": step,
            "train/rl/optimizer_attempts": population.global_attempts,
            "train/rl/selected_policy_actions": len(term.policy_weights),
            "train/rl/selected_kl_actions": len(term.kl_weights),
        }
        if population.spec.beta:
            values["train/rl/kl"] = values["train/rl/kl_loss"] / population.spec.beta
        if term.policy_weights:
            credit = {item.action: item.advantage for item in population.credit.values}
            advantages = [credit[item.action] for item in term.policy_weights]
            mean = sum(advantages) / len(advantages)
            values.update(
                {
                    "train/rl/advantage_mean": mean,
                    "train/rl/advantage_abs_mean": sum(abs(value) for value in advantages) / len(advantages),
                    "train/rl/advantage_std": (sum((value - mean) ** 2 for value in advantages) / len(advantages))
                    ** 0.5,
                    "train/rl/advantage_nonzero_fraction": sum(value != 0 for value in advantages) / len(advantages),
                    "train/rl/clip_fraction": len(evaluation.clipped_actions) / len(term.policy_weights),
                }
            )
        self.context.metrics(values, step=step, attributes={"measurement_scope": "resolved-applied-update"})
        self.context.event(
            "resolved_policy_update_applied",
            {
                "applied_update": step,
                "optimizer_attempts": population.global_attempts,
                "objective_digest": term.digest,
                "credit_digest": term.credit_digest,
                "update_digest": term.update_digest,
                "parameter_version": term.parameter_version,
                "policy_denominators": dict(term.policy_denominators),
                "kl_denominators": dict(term.kl_denominators),
            },
        )
        self._observed_applied = step

    def observation_callback(self, imports: Mapping[str, Any]) -> Any:
        job = self

        class ResolvedObservationCallback(imports["TrainerCallback"]):
            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                job.observe_applied_update(int(state.global_step))

        return ResolvedObservationCallback()

    def __post_init__(self) -> None:
        validate_resolved_job(self.request)
        AdmittedNativePopulation._validate_counters(0, 0, self.max_overflow_retries)  # noqa: SLF001
        if len({row["example_id"] for row in self.rows}) != len(self.rows):
            raise InvalidPolicyUpdate("resolved collection inventory contains duplicate tasks")
        if len(self.rows) < self.reservation:
            raise InvalidPolicyUpdate("resolved collection inventory cannot fill distinct complete groups")
        self.capabilities = ExecutionCapabilities(
            ("grpo@1", "dapo@1", "sampo@1", "sampo-spans@1", "gdpo@1", "capo@1"),
            ("sampled-logp", "old-logp", "reference-logp"),
            self.request.settings.max_prompt_length + self.request.settings.max_completion_length,
            True,
        )
        self.run = ResolvedTRLRun(self.collect, self.restore)

    @property
    def reservation(self) -> int:
        """Distinct tasks one update reserves: complete groups, times active candidate batches."""
        active = getattr(self.request.settings, "active_sampling", None)
        return self.request.settings.num_prompts_per_step * (active.max_candidate_batches if active else 1)

    def versions(self, applied: int) -> PolicyVersions:
        current = f"{self.runtime_identity}/actor-{applied}"
        return PolicyVersions(
            current, current, current, f"{self.runtime_identity}/reference" if self.request.settings.beta else None
        )

    def collect(self, applied: int, attempts: int) -> ResolvedTRLPopulation:
        if self.trainer is None or self.trainer.state.global_step != applied:
            raise InvalidPolicyUpdate("resolved job collector is not bound at the native applied boundary")
        ordered = list(self.rows)
        if self.request.settings.shuffle_prompts:
            random.Random(self.request.settings.loop.seed + applied).shuffle(ordered)
        active = isinstance(self.request, SAMPORequest) and self.request.settings.active_sampling is not None
        start = (applied * self.reservation if active else applied) % len(ordered)
        selected = [ordered[(start + index) % len(ordered)] for index in range(self.reservation)]
        selection: dict[str, object] = {
            "schema": "posttrain.resolved-task-selection@1",
            "applied": applied,
            "tasks": [row["example_id"] for row in selected],
            "seed": self.request.settings.loop.seed,
            "shuffle": self.request.settings.shuffle_prompts,
        }
        if active:
            assert isinstance(self.request, SAMPORequest)
            selection["active_sampling"] = asdict(self.request.settings.active_sampling)
        selector = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
        common: dict[str, Any] = dict(
            population_id=f"{self.context.run_id}/population-at-{applied}",
            template_revision=self.template_revision,
            versions=self.versions(applied),
            selector_digest=selector,
            attempt_offset=attempts,
            max_overflow_retries=self.max_overflow_retries,
            totals=self.totals,
            process_credit=getattr(self.request, "process_credit", None),
        )
        if active:
            assert isinstance(self.request, SAMPORequest)
            admitted = collect_active_resolved_population(
                self.context,
                self.request,
                self.tokenizer,
                self.trainer,
                selected,
                self.capabilities,
                evidence_directory=Path(self.trainer.args.output_dir).parent / "collection-evidence",
                **common,
            )
        else:
            rows = [row for row in selected for _ in range(self.request.settings.num_generations)]
            admitted = collect_resolved_population(
                self.context, self.request, self.tokenizer, self.trainer, rows, self.capabilities, **common
            )
        population = ResolvedTRLPopulation.from_admitted(
            admitted, score_temperature=self.score_temperature, score_contract=SCORE_CONTRACT, sampler_correction=None
        )
        sampled = admitted.read_input.sampling_log_scores(admitted.resolved.snapshot)
        population.prepare_sampler_correction = lambda old: recipe_sampler_correction_weights(
            self.request.settings, admitted.resolved.snapshot, old, sampled
        )
        if population.spec.beta:
            population.reference = self._reference_scores(population)
        return population

    def _reference_scores(self, population: ResolvedTRLPopulation) -> Any:
        from ..policy_update_scoring import freeze_population_scores

        snapshot = population.updates[0].population
        kwargs: dict[str, Any] = dict(
            read_input=population.read_input,
            device=self.trainer.accelerator.device,
            policy_version=snapshot.versions.reference,
            score_contract=SCORE_CONTRACT,
            score_temperature=self.score_temperature,
        )
        if self.trainer.ref_model is not None:
            return freeze_population_scores(self.trainer.ref_model, snapshot, **kwargs)
        from trl.trainer.utils import use_adapter

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        from .policy_config import kl_reference

        if (
            kl_reference(self.request) == "start"
            and self.request.policy.form in {"adapter", "peft-adapter"}
            and "ref" not in model.peft_config
        ):
            raise InvalidPolicyUpdate("native trainer lacks the selected frozen starting-adapter KL reference")
        with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
            return freeze_population_scores(self.trainer.model, snapshot, **kwargs)

    def restore(self, checkpoint: Path) -> ResolvedTRLPopulation:
        from ..policy_update_recovery import POPULATION_FILENAME, load_sampler_correction, population_recovery_identity

        state = inspect_update_recovery(checkpoint)
        identity = state.identity
        if (
            identity.runtime_identity != self.runtime_identity
            or identity.world_size != 1
            or identity.score_contract != SCORE_CONTRACT
            or identity.score_temperature != self.score_temperature
        ):
            raise InvalidPolicyUpdate("resolved job checkpoint differs from selected runtime or scoring contract")
        retained = decode_population_payload(json.loads((checkpoint / POPULATION_FILENAME).read_text()))
        correction = load_sampler_correction(checkpoint, identity)
        if correction is None:
            raise InvalidPolicyUpdate("selected ordinary recipe requires retained sampler correction")
        selected = resolve_policy_population(
            retained.resolved.snapshot, retained.resolved.credit, self.request.settings, self.capabilities
        )
        # Sampling provenance belongs to the authenticated retained population,
        # not to the executor performing replay. Frozen correction binds its
        # sampled scores to the old policy; keep executor/old/reference versions
        # exact while retaining (rather than rewriting) the original sampler.
        expected_versions = replace(
            self.versions(retained.applied_update_offset), sampler=selected.snapshot.versions.sampler
        )
        if (
            selected != retained.resolved
            or retained.max_overflow_retries != self.max_overflow_retries
            or selected.snapshot.versions != expected_versions
            or any(view.template_revision != self.template_revision for view in selected.snapshot.conditioning)
        ):
            raise InvalidPolicyUpdate("resolved job checkpoint differs from selected population contracts")
        # Constructing an expected identity uses authenticated records and the
        # independently selected recipe above; never re-estimate saved credit.
        candidate = SimpleNamespace(
            updates=selected.updates,
            credit=selected.credit,
            spec=selected.spec,
            execution=selected.execution,
            capabilities=selected.capabilities,
            max_overflow_retries=self.max_overflow_retries,
            sampler_correction=correction,
            score_contract=SCORE_CONTRACT,
            score_temperature=self.score_temperature,
            applied_update_offset=retained.applied_update_offset,
            attempt_offset=retained.attempt_offset,
        )
        if population_recovery_identity(candidate, runtime_identity=self.runtime_identity, world_size=1) != identity:
            raise InvalidPolicyUpdate("resolved job checkpoint identity differs from selected recovery")
        admitted = AdmittedNativePopulation.from_checkpoint(
            checkpoint, identity, sampler_correction=correction, decode=_decode_native
        )
        self._observed_applied = state.native_applied_updates
        return ResolvedTRLPopulation.from_admitted(
            admitted,
            score_temperature=self.score_temperature,
            score_contract=SCORE_CONTRACT,
            sampler_correction=correction,
        )
