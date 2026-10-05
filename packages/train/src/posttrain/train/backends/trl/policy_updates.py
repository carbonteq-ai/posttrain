"""Native TRL lifecycle adapter for pre-resolved policy populations.

The existing GRPOTrainer owns backward, optimizer, precision and LR scheduling.
Resolved populations supply optimizer boundaries, so native generation repetition
and slicing are bypassed. Public launch remains gated pending live qualification,
overflow retry, recovery and distributed execution.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from posttrain.environment.verifiers_conditioning import NativeConditioningInput

from ...update_credit import PreparedCredit
from ...update_objectives import ObjectiveSpec, resolve_objective_term
from ...update_plan import ExecutionCapabilities, PolicyExecutionBudget, ResolvedUpdate, plan_packs
from ...update_records import ActionRef, ConditioningView, InvalidPolicyUpdate
from ...update_resolution import ResolvedPolicyPopulation
from ..policy_update_admission import AdmittedNativePopulation
from ..policy_update_lifecycle import ResolvedPolicyRun


@dataclass
class ResolvedTRLPopulation:
    """One frozen population reused across its declared resolved occurrences."""

    updates: tuple[ResolvedUpdate, ...]
    credit: PreparedCredit
    spec: ObjectiveSpec
    execution: PolicyExecutionBudget
    capabilities: ExecutionCapabilities
    read_input: Callable[[ConditioningView], NativeConditioningInput]
    score_temperature: float
    score_contract: str
    sampler_correction: Mapping[ActionRef, float] | None
    reference: Any = None
    prepare_sampler_correction: Callable[[Mapping[ActionRef, float]], Mapping[ActionRef, float]] | None = None
    next_update: int = 0
    applied_updates: int = 0
    attempts: int = 0
    applied_update_offset: int = 0
    attempt_offset: int = 0
    max_overflow_retries: int = 0
    # The sampler's own log scores for each original action, retained for
    # sampler-gap observation only. Correction weights are frozen separately.
    sampled_scores: Mapping[ActionRef, float] | None = field(default=None, init=False, repr=False)
    old: Any = field(default=None, init=False)
    last_evaluation: Any = field(default=None, init=False)
    _pending: int | None = field(default=None, init=False)
    _scale_before: float | None = field(default=None, init=False)
    _retryable: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.prepare_sampler_correction is not None and self.sampler_correction is not None:
            raise InvalidPolicyUpdate("correction preparation cannot replace supplied frozen weights")
        if any(type(value) is not int or value < 0 for value in (self.applied_update_offset, self.attempt_offset)) or (
            self.attempt_offset < self.applied_update_offset
        ):
            raise InvalidPolicyUpdate("native TRL population offsets require coherent prior applied work and attempts")
        if type(self.max_overflow_retries) is not int or self.max_overflow_retries < 0:
            raise InvalidPolicyUpdate("native overflow retries require an explicit nonnegative limit")
        if not self.updates or len({update.population.digest for update in self.updates}) != 1:
            raise InvalidPolicyUpdate("native TRL population requires one frozen population and resolved updates")
        if len({update.digest for update in self.updates}) != len(self.updates):
            raise InvalidPolicyUpdate("native TRL occurrences must identify distinct resolved updates")
        if (
            isinstance(self.score_temperature, bool)
            or not math.isfinite(self.score_temperature)
            or self.score_temperature <= 0
        ):
            raise InvalidPolicyUpdate("native TRL score temperature must be finite and positive")
        if not isinstance(self.score_contract, str) or not self.score_contract.strip():
            raise InvalidPolicyUpdate("native TRL score arithmetic requires an explicit contract identity")
        for update in self.updates:
            resolve_objective_term(update, self.spec, self.credit)
            plan_packs(update, self.execution, self.capabilities)

    @classmethod
    def from_resolved(
        cls,
        resolved: ResolvedPolicyPopulation,
        *,
        read_input: Callable[[ConditioningView], NativeConditioningInput],
        score_temperature: float,
        score_contract: str,
        sampler_correction: Mapping[ActionRef, float] | None,
        reference: Any = None,
        max_overflow_retries: int = 0,
        applied_update_offset: int = 0,
        attempt_offset: int = 0,
    ) -> ResolvedTRLPopulation:
        """Consume the shared validated population without rebuilding its recipe."""
        return cls(
            resolved.updates,
            resolved.credit,
            resolved.spec,
            resolved.execution,
            resolved.capabilities,
            read_input,
            score_temperature,
            score_contract,
            sampler_correction,
            reference=reference,
            max_overflow_retries=max_overflow_retries,
            applied_update_offset=applied_update_offset,
            attempt_offset=attempt_offset,
        )

    @classmethod
    def from_admitted(
        cls,
        admitted: AdmittedNativePopulation,
        *,
        score_temperature: float,
        score_contract: str,
        sampler_correction: Mapping[ActionRef, float] | None,
        reference: Any = None,
    ) -> ResolvedTRLPopulation:
        return cls.from_resolved(
            admitted.resolved,
            read_input=admitted.read_input,
            score_temperature=score_temperature,
            score_contract=score_contract,
            sampler_correction=sampler_correction,
            reference=reference,
            max_overflow_retries=admitted.max_overflow_retries,
            applied_update_offset=admitted.applied_update_offset,
            attempt_offset=admitted.attempt_offset,
        )

    @property
    def global_applied_updates(self) -> int:
        return self.applied_update_offset + self.applied_updates

    @property
    def global_attempts(self) -> int:
        return self.attempt_offset + self.attempts

    def loss(self, model: Any, index: int, device: Any, *, before_current: Callable[[], None] | None = None) -> Any:
        from ..policy_update_execution import compute_resolved_loss
        from ..policy_update_scoring import freeze_population_scores

        if type(index) is not int or index != self.next_update or not 0 <= index < len(self.updates):
            raise InvalidPolicyUpdate("native TRL dataloader changed resolved occurrence order")
        if self._pending is not None:
            raise InvalidPolicyUpdate("resolved native update cannot be evaluated twice before optimizer completion")
        update = self.updates[index]
        if self.old is None:
            self.old = freeze_population_scores(
                model,
                update.population,
                read_input=self.read_input,
                device=device,
                policy_version=update.population.versions.old_score,
                score_contract=self.score_contract,
                score_temperature=self.score_temperature,
            )
        if self.prepare_sampler_correction is not None:
            from types import MappingProxyType

            correction = self.prepare_sampler_correction(
                {action: float(value) for action, value in self.old.values.items()}
            )
            if set(correction) != {record.action for record in update.population.actions} or any(
                type(value) not in (float, int) or not math.isfinite(value) or value < 0
                for value in correction.values()
            ):
                raise InvalidPolicyUpdate("prepared correction requires complete detached finite action weights")
            self.sampler_correction = MappingProxyType(dict(correction))
            self.prepare_sampler_correction = None
        term = resolve_objective_term(
            update,
            self.spec,
            self.credit,
            parameter_version=f"{update.population.versions.current}/applied-{self.applied_updates}",
        )
        if before_current is not None:
            # The first old-score forward can consume randomness. Retries skip
            # it, so capture current-score state after freezing, not before it.
            before_current()
        self.last_evaluation = compute_resolved_loss(
            model,
            update,
            term,
            self.credit,
            plan_packs(update, self.execution, self.capabilities),
            old=self.old,
            reference=self.reference,
            read_input=self.read_input,
            device=device,
            score_temperature=self.score_temperature,
            score_contract=self.score_contract,
            sampler_correction=self.sampler_correction,
        )
        if not self.last_evaluation.loss.requires_grad:
            raise InvalidPolicyUpdate("empty resolved native update cannot count as an applied optimizer step")
        self._pending = index
        self.attempts += 1
        return self.last_evaluation.loss

    def before_step(self, optimizer: Any) -> None:
        if self._pending is None:
            raise InvalidPolicyUpdate("native optimizer has no resolved pending update")
        scaler = getattr(optimizer, "scaler", None)
        self._scale_before = float(scaler.get_scale()) if scaler is not None else None
        self._retryable = False

    def complete_step(self, optimizer: Any, *, allow_retry: bool = False) -> bool:
        scaler = getattr(optimizer, "scaler", None)
        overflow = bool(getattr(optimizer, "step_was_skipped", False)) or (
            scaler is not None and self._scale_before is not None and float(scaler.get_scale()) < self._scale_before
        )
        self._scale_before = None
        if self._pending is None:
            raise InvalidPolicyUpdate("native optimizer completed without resolved pending evidence")
        if overflow:
            # Keep the occurrence pending: an exhausted retry is not a boundary.
            self._retryable = True
            if allow_retry:
                return False
            raise InvalidPolicyUpdate("native resolved overflow exhausted retry limit without advancing policy")
        self.next_update += 1
        self.applied_updates += 1
        self._pending = None
        self._retryable = False
        return True

    def prepare_retry(self) -> None:
        if not self._retryable or self._pending != self.next_update or self._scale_before is not None:
            raise InvalidPolicyUpdate("native overflow retry requires the same completed skipped attempt")
        self._pending = None
        self._retryable = False


class ResolvedTRLRun(ResolvedPolicyRun[ResolvedTRLPopulation]):
    """Collect frozen populations at applied boundaries inside one native train loop.

    Dataloader records contain run-global update slots only. Collection happens
    while computing a slot, never during dataloader prefetch. The host restores
    the retained population and conditioning inputs before native state loading;
    its returned identity is then checked against the sealed checkpoint.
    """


def resolved_policy_trainer_type(
    parent: type,
    population: ResolvedTRLPopulation | ResolvedTRLRun,
    *,
    recovery_runtime_identity: str | None = None,
) -> type:
    """Adapt native GRPOTrainer without replacing its optimizer lifecycle."""

    def active_population() -> ResolvedTRLPopulation:
        return population.active() if isinstance(population, ResolvedTRLRun) else population

    class ResolvedPolicyTrainer(parent):
        def _before_resolved_step(self, optimizer):
            import torch

            if getattr(optimizer, "scaler", None) is None:
                if any(
                    not bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ):
                    raise InvalidPolicyUpdate("native unscaled nonfinite gradients cannot mutate the resolved policy")
            active_population().before_step(optimizer)

        def _capture_current_rng(self):
            import random

            import numpy as np
            import torch

            if self._resolved_retry_context is None:
                raise InvalidPolicyUpdate("native current scoring lacks a retained retry context")
            model, inputs, count, _ = self._resolved_retry_context
            rng = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            )
            self._resolved_retry_context = (model, inputs, count, rng)

        def training_step(self, model, inputs, num_items_in_batch=None):
            if self._resolved_retry_context is not None:
                raise InvalidPolicyUpdate("native unresolved attempt cannot consume another dataloader occurrence")
            self._resolved_retry_context = (model, inputs, num_items_in_batch, None)
            return super().training_step(model, inputs, num_items_in_batch)

        def _finish_resolved_step(self, optimizer):
            import random

            import numpy as np
            import torch

            active = active_population()
            if active.complete_step(optimizer, allow_retry=active.max_overflow_retries > 0):
                self._resolved_retry_context = None
                return
            if self._resolved_retry_context is None or getattr(optimizer, "scaler", None) is None:
                raise InvalidPolicyUpdate("native resolved retry requires retained inputs and native loss scaler")
            model, inputs, count, rng = self._resolved_retry_context
            if rng is None:
                raise InvalidPolicyUpdate("native resolved retry lacks current-score stochastic state")
            for _ in range(active.max_overflow_retries):
                # Retry native backward at unchanged parameters; old scores and
                # dataloader membership remain frozen. The scaler keeps its
                # reduced scale, while stochastic inputs replay the first try.
                optimizer.zero_grad(set_to_none=True)
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                if rng[3] is not None:
                    torch.cuda.set_rng_state_all(rng[3])
                active.prepare_retry()
                super().training_step(model, inputs, count)
                grad_norm = self._clip_grad_norm(model) if self.args.max_grad_norm > 0 else None
                self._get_grad_norm(model, grad_norm=grad_norm)
                self._before_resolved_step(optimizer)
                optimizer.step()
                if active.complete_step(optimizer, allow_retry=True):
                    self._resolved_retry_context = None
                    return
            raise InvalidPolicyUpdate("native resolved overflow exhausted retry limit without advancing policy")

        def _recovery_identity(self):
            from ..policy_update_recovery import population_recovery_identity

            if recovery_runtime_identity is None:
                raise InvalidPolicyUpdate("resolved checkpointing requires an immutable runtime identity")
            return population_recovery_identity(
                active_population(), runtime_identity=recovery_runtime_identity, world_size=1
            )

        def _save_checkpoint(self, *args, **kwargs):
            from ..policy_update_recovery import save_population_recovery

            identity = self._recovery_identity()
            active = active_population()
            if (
                self._resolved_retry_context is not None
                or active._pending is not None
                or self.state.global_step != active.global_applied_updates
            ):
                raise InvalidPolicyUpdate("native checkpoint is not at the resolved applied boundary")
            trial = kwargs.get("trial", args[1] if len(args) > 1 else None)
            checkpoint = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
            if (checkpoint / "posttrain-resolved-update.json").exists():
                save_population_recovery(
                    active,
                    checkpoint,
                    runtime_identity=identity.runtime_identity,
                    world_size=1,
                    native_applied_updates=self.state.global_step,
                    native_components=(),
                )
                return
            super()._save_checkpoint(*args, **kwargs)
            required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
            if self.accelerator.scaler is not None:
                required += ("scaler.pt",)
            if any(not (checkpoint / relative).is_file() for relative in required):
                raise InvalidPolicyUpdate("native TRL checkpoint lacks complete optimizer/RNG/scaler state")
            weights = [path for pattern in ("*model*.safetensors", "*model*.bin") for path in checkpoint.glob(pattern)]
            if not weights:
                raise InvalidPolicyUpdate("native TRL checkpoint lacks retained model weights")
            files = tuple(sorted(str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*") if path.is_file()))
            save_population_recovery(
                active,
                checkpoint,
                runtime_identity=identity.runtime_identity,
                world_size=1,
                native_applied_updates=self.state.global_step,
                native_components=files,
            )

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            from ...update_recovery import load_update_recovery

            if isinstance(population, ResolvedTRLRun):
                population.retain_checkpoint_population(Path(resume_from_checkpoint))
            load_update_recovery(Path(resume_from_checkpoint), self._recovery_identity())
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def _load_optimizer_and_scheduler(self, checkpoint):
            import json

            from ..policy_update_recovery import restore_population_recovery

            super()._load_optimizer_and_scheduler(checkpoint)
            if checkpoint is not None:
                path = Path(checkpoint)
                native_step = json.loads((path / "trainer_state.json").read_text())["global_step"]
                restore_population_recovery(
                    active_population(),
                    path,
                    runtime_identity=self._recovery_identity().runtime_identity,
                    world_size=1,
                    native_applied_updates=native_step,
                    device=self.accelerator.device,
                )

        def get_train_dataloader(self):
            return self._get_dataloader(
                dataset=self.train_dataset,
                description="Training",
                batch_size=1,
                sampler_fn=self._get_train_sampler,
                is_training=True,
            )

        def _get_train_sampler(self, dataset=None):
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset if dataset is None else dataset)

        def _prepare_inputs(self, rows):
            if len(rows) != 1 or set(rows[0]) != {"resolved_update"}:
                raise InvalidPolicyUpdate("native resolved dataloader must deliver exactly one declared update")
            return rows[0]

        def _compute_loss(self, model, inputs):
            active, index = (
                population.occurrence(inputs["resolved_update"], self.state.global_step)
                if isinstance(population, ResolvedTRLRun)
                else (population, inputs["resolved_update"])
            )
            if self.state.global_step != active.global_applied_updates:
                raise InvalidPolicyUpdate("native TRL run-global counter differs from the active population boundary")
            return active.loss(model, index, self.accelerator.device, before_current=self._capture_current_rng)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._resolved_retry_context = None
            if self.args.gradient_accumulation_steps != 1 or self.num_iterations != 1:
                raise InvalidPolicyUpdate(
                    "resolved native updates replace native accumulation and iteration scheduling"
                )
            if self.args.per_device_train_batch_size != 1 or self.args.remove_unused_columns:
                raise InvalidPolicyUpdate(
                    "resolved native dataloader requires one payload per batch and retained columns"
                )
            if self.use_liger_kernel:
                raise InvalidPolicyUpdate("resolved native loss does not support the Liger compute-loss bypass")
            if self.accelerator.num_processes != 1:
                raise InvalidPolicyUpdate("resolved native distributed execution has not been qualified")
            if self.args.save_only_model or self.args.ignore_data_skip:
                raise InvalidPolicyUpdate("resolved native recovery requires full state and exact occurrence skipping")
            from transformers import TrainerCallback

            trainer = self

            class ResolvedUpdateCallback(TrainerCallback):
                def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
                    trainer._before_resolved_step(optimizer)

                def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
                    trainer._finish_resolved_step(optimizer)

            self.add_callback(ResolvedUpdateCallback())

    return ResolvedPolicyTrainer
