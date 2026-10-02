"""Resolved update adapter over native veRL inference and FSDP train_batch.

Uses native model scoring, microbatch backward, optimizer/scaler and scheduler.
Complete score adjoints preserve cross-turn dependencies without retaining all
turn graphs. Public launch remains gated pending qualification and recovery.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from posttrain.environment.verifiers_conditioning import NativeConditioningInput

from ...update_credit import PreparedCredit
from ...update_objectives import ObjectiveSpec, resolve_objective_term
from ...update_plan import (
    ExecutionCapabilities,
    PolicyExecutionBudget,
    ResolvedUpdate,
    execution_context_tokens,
    plan_packs,
    population_context_width,
)
from ...update_records import ActionRef, ConditioningView, InvalidPolicyUpdate
from ...update_resolution import ResolvedPolicyPopulation
from ..policy_update_admission import AdmittedNativePopulation
from ..policy_update_lifecycle import ResolvedPolicyRun


def selected_native_outputs(output: Any, output_args: Mapping[str, Any], data: Any) -> Mapping[str, Any]:
    """Project full-context dense logits on declared sampled prediction positions.

    This is Posttrain probability arithmetic, not veRL's stock probability
    kernel. The native engine still owns model forward, backward and stepping.
    Callers qualify the dense causal-text model path and this score contract.
    """
    import torch
    from verl.utils import tensordict_utils as tu  # pyright: ignore[reportMissingImports]

    if tu.get_non_tensor_data(data, "use_remove_padding", default=True) or tu.get_non_tensor_data(
        data, "use_fused_kernels", default=False,
    ):
        raise InvalidPolicyUpdate("resolved veRL scoring requires qualified dense unfused model logits")
    if tu.get_non_tensor_data(data, "calculate_entropy", default=False):
        raise InvalidPolicyUpdate("score-only replay does not support native entropy terms")
    logits = output.logits
    inputs = data["input_ids"].unbind()
    temperatures = output_args["temperature"]
    if logits.ndim != 3 or logits.shape[0] != len(inputs) or not logits.is_floating_point():
        raise InvalidPolicyUpdate("resolved veRL logits must preserve original dense context rows")
    rows = []
    for index, ids in enumerate(inputs):
        length = len(ids)
        if logits.shape[1] < length:
            raise InvalidPolicyUpdate("resolved veRL logits truncate original conditioning")
        mask = data["loss_mask"][index]
        if bool(((mask != 0) & (mask != 1)).any()) or bool(mask[length - 1:].any()):
            raise InvalidPolicyUpdate("resolved veRL mask exceeds eligible causal prediction positions")
        positions = mask[:length - 1].nonzero().flatten()
        if not len(positions):
            raise InvalidPolicyUpdate("resolved veRL context has no required sampled scores")
        selected = logits[index, positions]
        if selected.dtype in (torch.bfloat16, torch.float16):
            selected = selected.float()
        temperature = temperatures[index]
        if not bool(torch.isfinite(temperature)) or float(temperature) <= 0 or not bool(torch.isfinite(selected).all()):
            raise InvalidPolicyUpdate("resolved veRL scores require finite logits and positive temperature")
        targets = ids[positions + 1]
        if bool(((targets < 0) | (targets >= logits.shape[-1])).any()):
            raise InvalidPolicyUpdate("resolved veRL sampled action is outside model vocabulary")
        values = (selected / temperature).log_softmax(-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        # Only addressed entries are read by _scores. Unselected placeholders
        # carry no gradient or objective weight; full context remains in forward.
        rows.append(torch.zeros(length, dtype=values.dtype, device=values.device).scatter(0, positions, values))
    return {"log_probs": torch.nested.as_nested_tensor(rows, layout=torch.jagged)}


def resolved_verl_engine_type(native_engine_type: type) -> type:
    """Explicit engine subclass using Posttrain's resolved sampled-score contract."""
    class ResolvedEngine(native_engine_type):
        def prepare_model_inputs(self, micro_batch):
            import torch
            from verl.utils import tensordict_utils as tu  # pyright: ignore[reportMissingImports]

            width = tu.get_non_tensor_data(micro_batch, "resolved_dense_width", default=None)
            budget = tu.get_non_tensor_data(micro_batch, "resolved_context_budget", default=None)
            if budget is not None and (type(budget) is not int or budget < 1):
                raise InvalidPolicyUpdate("native padded context requires a positive physical execution budget")
            if width is not None:
                if type(width) is not int or width < 1:
                    raise InvalidPolicyUpdate("resolved dense padding requires an explicit positive width")
                if budget is None or micro_batch.batch_size[0] * width > budget:
                    raise InvalidPolicyUpdate("native padded context exceeds the physical execution budget")
            model_inputs, output_args = super().prepare_model_inputs(micro_batch)
            if width is not None:
                for key in ("input_ids", "attention_mask", "position_ids"):
                    value = model_inputs[key]
                    if value.ndim != 2 or value.shape[-1] > width:
                        raise InvalidPolicyUpdate("resolved dense padding requires bounded causal-text inputs")
                    model_inputs[key] = torch.nn.functional.pad(value, (0, width - value.shape[-1]), value=0)
            if budget is not None:
                actual = model_inputs["input_ids"].numel()
                if actual > budget:
                    raise InvalidPolicyUpdate("native padded context exceeds the physical execution budget")
            return model_inputs, output_args

        def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
            return selected_native_outputs(output, output_args, micro_batch)

    return ResolvedEngine


@dataclass
class ResolvedVeRLPopulation:
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
    replay_absolute_tolerance: float = 0.0
    replay_relative_tolerance: float = 0.0
    max_overflow_retries: int = 0
    next_update: int = 0
    applied_updates: int = 0
    attempts: int = 0
    applied_update_offset: int = 0
    attempt_offset: int = 0
    old: Any = field(default=None, init=False)
    last_adjoints: Any = field(default=None, init=False)
    last_output: Any = field(default=None, init=False)
    _pending: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.prepare_sampler_correction is not None and self.sampler_correction is not None:
            raise InvalidPolicyUpdate("correction preparation cannot replace supplied frozen weights")
        if any(type(value) is not int or value < 0 for value in (self.applied_update_offset, self.attempt_offset)) or (
            self.attempt_offset < self.applied_update_offset
        ):
            raise InvalidPolicyUpdate("native veRL population offsets require coherent prior applied work and attempts")
        if not self.updates or len({update.population.digest for update in self.updates}) != 1:
            raise InvalidPolicyUpdate("native veRL adapter requires one frozen population")
        if self.execution.records > 1 and self.capabilities.context_layout == "ragged":
            raise InvalidPolicyUpdate("dense native veRL packs require a declared physical context layout")
        if type(self.max_overflow_retries) is not int or self.max_overflow_retries < 0:
            raise InvalidPolicyUpdate("native overflow retries require an explicit nonnegative limit")
        if isinstance(self.score_temperature, bool) or not math.isfinite(self.score_temperature) or self.score_temperature <= 0:
            raise InvalidPolicyUpdate("native veRL score temperature must be finite and positive")
        if not isinstance(self.score_contract, str) or not self.score_contract.strip():
            raise InvalidPolicyUpdate("native veRL score arithmetic requires an explicit contract identity")
        if len({update.digest for update in self.updates}) != len(self.updates):
            raise InvalidPolicyUpdate("native veRL occurrences must identify distinct resolved updates")
        for update in self.updates:
            resolve_objective_term(update, self.spec, self.credit)
            plan_packs(update, self.execution, self.capabilities)

    @classmethod
    def from_resolved(
        cls, resolved: ResolvedPolicyPopulation, *,
        read_input: Callable[[ConditioningView], NativeConditioningInput],
        score_temperature: float, score_contract: str,
        sampler_correction: Mapping[ActionRef, float] | None,
        reference: Any = None, max_overflow_retries: int = 0,
        applied_update_offset: int = 0, attempt_offset: int = 0,
    ) -> ResolvedVeRLPopulation:
        """Consume the same objective, credit and schedule as the TRL adapter."""
        return cls(resolved.updates, resolved.credit, resolved.spec, resolved.execution, resolved.capabilities,
                   read_input, score_temperature, score_contract, sampler_correction,
                   reference=reference, max_overflow_retries=max_overflow_retries,
                   applied_update_offset=applied_update_offset, attempt_offset=attempt_offset)

    @classmethod
    def from_admitted(
        cls, admitted: AdmittedNativePopulation, *, score_temperature: float,
        score_contract: str, sampler_correction: Mapping[ActionRef, float] | None,
        reference: Any = None,
    ) -> ResolvedVeRLPopulation:
        return cls.from_resolved(
            admitted.resolved, read_input=admitted.read_input, score_temperature=score_temperature,
            score_contract=score_contract, sampler_correction=sampler_correction, reference=reference,
            max_overflow_retries=admitted.max_overflow_retries,
            applied_update_offset=admitted.applied_update_offset, attempt_offset=admitted.attempt_offset,
        )

    @property
    def global_applied_updates(self) -> int:
        return self.applied_update_offset + self.applied_updates

    @property
    def global_attempts(self) -> int:
        return self.attempt_offset + self.attempts

    def _rows(self, actions: tuple[ActionRef, ...]):
        snapshot = self.updates[0].population
        records = {record.action: record for record in snapshot.actions}
        contexts = {view.id: view for view in snapshot.conditioning}
        if not actions or len(set(actions)) != len(actions) or any(action not in records for action in actions):
            raise InvalidPolicyUpdate("native veRL score support requires unique admitted actions")
        groups: dict[str, list[ActionRef]] = {}
        for action in actions:
            groups.setdefault(records[action].conditioning_id, []).append(action)
        rows = []
        for context_id, members in groups.items():
            view = contexts[context_id]
            inputs = self.read_input(view)
            if inputs.record.context_contract != "causal-text@1" or inputs.record.input_digest != view.digest or (
                len(inputs.token_ids) != view.context_tokens or inputs.record.context_tokens != view.context_tokens
            ):
                raise InvalidPolicyUpdate("native veRL input differs from frozen original context")
            positions = dict(inputs.action_positions)
            if any(action.token_index not in positions for action in members):
                raise InvalidPolicyUpdate("native veRL view lost original action positions")
            if len({action.turn_id for action in members}) != 1 or len({action.token_index for action in members}) != len(members):
                raise InvalidPolicyUpdate("native veRL view must identify one original sampled turn")
            if any(not 1 <= positions[action.token_index] < len(inputs.token_ids) for action in members):
                raise InvalidPolicyUpdate("native veRL action lacks its preceding causal position")
            rows.append((inputs, tuple((action, positions[action.token_index] - 1) for action in members)))
        return tuple(rows)

    def _pack_sizes(self, rows):
        """Preserve ordered original views under both execution capacities."""
        sizes, pending = [], []
        snapshot = self.updates[0].population
        contexts = {record.action: record.conditioning_id for record in snapshot.actions}
        for _, members in rows:
            context = contexts[members[0][0]]
            if execution_context_tokens(snapshot, (context,), self.capabilities) > self.execution.context_tokens:
                raise InvalidPolicyUpdate("native original context exceeds execution pack capacity")
            cost = execution_context_tokens(snapshot, (*pending, context), self.capabilities)
            if pending and (len(pending) == self.execution.records or cost > self.execution.context_tokens):
                sizes.append(len(pending))
                pending = []
            pending.append(context)
        if pending:
            sizes.append(len(pending))
        return tuple(sizes)

    def _batch(self, rows):
        import torch
        from verl import DataProto  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
        from verl.utils import tensordict_utils as tu  # pyright: ignore[reportMissingImports]

        width = max(len(inputs.token_ids) for inputs, _ in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention = torch.zeros_like(ids)
        mask = torch.zeros((len(rows), width - 1), dtype=torch.long)
        for index, (inputs, actions) in enumerate(rows):
            ids[index, :len(inputs.token_ids)] = torch.tensor(inputs.token_ids)
            attention[index, :len(inputs.token_ids)] = 1
            for _, position in actions:
                mask[index, position] = 1
        values = dict(input_ids=ids, attention_mask=attention, position_ids=torch.arange(width).expand(len(rows), width),
                      prompts=ids[:, :1], responses=ids[:, 1:], response_mask=mask,
                      resolved_context_index=torch.arange(len(rows)).unsqueeze(1))
        data = DataProto.from_single_dict(values, meta_info={
            "temperature": self.score_temperature,
        }).to_tensordict()
        # Construct native jagged views directly. Dense SDPA does not need the
        # FlashAttention unpad helper used by the generic padding converter.
        data["input_ids"] = torch.nested.as_nested_tensor(
            [ids[index, :len(inputs.token_ids)] for index, (inputs, _) in enumerate(rows)], layout=torch.jagged,
        )
        data["position_ids"] = torch.nested.as_nested_tensor(
            [torch.arange(len(inputs.token_ids)) for inputs, _ in rows], layout=torch.jagged,
        )
        data["loss_mask"] = mask
        tu.assign_non_tensor(data, max_seq_len=width, max_response_len=width - 1,
                             indices=attention.flatten().nonzero().flatten())
        tu.assign_non_tensor(data, global_batch_size=len(rows), use_remove_padding=False, use_dynamic_bsz=False,
                             micro_batch_size_per_gpu=1, max_token_len_per_gpu=width, use_fused_kernels=False,
                             calculate_entropy=False, return_model_output=True)
        tu.assign_non_tensor(data, resolved_context_budget=self.execution.context_tokens)
        if self.capabilities.context_layout == "dense-population":
            tu.assign_non_tensor(data, resolved_dense_width=population_context_width(self.updates[0].population))
        if self.execution.records > 1:
            tu.assign_non_tensor(data, micro_batch_sizes=self._pack_sizes(rows))
        return data

    @staticmethod
    def _scores(output, data, rows):
        scores = {}
        for local_index, row_index in enumerate(data["resolved_context_index"].flatten().tolist()):
            probabilities = output["log_probs"][local_index]
            for action, position in rows[row_index][1]:
                scores[action] = probabilities[position]
        return scores

    def _infer(self, engine, rows):
        with engine.eval_mode():
            data = self._batch(rows)
            output = engine.infer_batch(data)
            return self._scores(output["model_output"], data, rows)

    def freeze_old_scores(self, engine) -> Any:
        """Score complete original support once at the population's initial actor.

        Collection may call this before preparing sampler correction. Subsequent
        updates and resumed populations retain these scores without rescoring.
        """
        from ..policy_update_scoring import FrozenPopulationScores

        snapshot = self.updates[0].population
        if self.old is None:
            if self.next_update != 0 or self.applied_updates != 0 or self._pending is not None or (
                engine.lr_scheduler.last_epoch != self.applied_update_offset
            ):
                raise InvalidPolicyUpdate("old scores must freeze at the initial population actor boundary")
            old = self._infer(engine, self._rows(tuple(record.action for record in snapshot.actions)))
            self.old = FrozenPopulationScores(snapshot.digest, snapshot.versions.old_score,
                self.score_contract, self.score_temperature,
                MappingProxyType({action: value.detach().clone() for action, value in old.items()}))
        self.old.validate(snapshot, policy_version=snapshot.versions.old_score,
                          score_contract=self.score_contract, score_temperature=self.score_temperature)
        return self.old

    def _checkpoint_contract(self, engine):
        if engine.get_data_parallel_size() != 1:
            raise InvalidPolicyUpdate("native veRL checkpoint recovery is qualified per world size")
        manager = engine.checkpoint_manager
        for operation in ("save", "load"):
            if not all(getattr(manager, f"should_{operation}_{component}") for component in ("model", "optimizer", "extra")):
                raise InvalidPolicyUpdate("resolved veRL checkpoints require full native model/optimizer/extra state")
        scaler = getattr(engine, "scaler", None)
        if scaler is not None and getattr(manager, "grad_scaler", None) is not scaler:
            raise InvalidPolicyUpdate("native veRL checkpoint manager does not retain the active FP16 scaler")

    def save_native_checkpoint(self, engine, checkpoint: Path, *, runtime_identity: str):
        from ..policy_update_recovery import save_population_recovery

        self._checkpoint_contract(engine)
        if self._pending is not None or engine.lr_scheduler.last_epoch != self.global_applied_updates:
            raise InvalidPolicyUpdate("native veRL checkpoint is not at a complete applied/scheduler boundary")
        if (checkpoint / "posttrain-resolved-update.json").exists():
            return save_population_recovery(self, checkpoint, runtime_identity=runtime_identity, world_size=1,
                native_applied_updates=engine.lr_scheduler.last_epoch, native_components=())
        engine.save_checkpoint(local_path=str(checkpoint), global_step=self.global_applied_updates)
        required = tuple(f"{name}_world_size_1_rank_0.pt" for name in ("model", "optim", "extra_state"))
        if any(not (checkpoint / filename).is_file() for filename in required):
            raise InvalidPolicyUpdate("native veRL checkpoint lacks full retained state")
        files = tuple(sorted(str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*") if path.is_file()))
        return save_population_recovery(self, checkpoint, runtime_identity=runtime_identity, world_size=1,
            native_applied_updates=engine.lr_scheduler.last_epoch, native_components=files)

    def load_native_checkpoint(self, engine, checkpoint: Path, *, runtime_identity: str):
        from ...update_recovery import load_update_recovery
        from ..policy_update_recovery import population_recovery_identity, restore_population_recovery

        self._checkpoint_contract(engine)
        identity = population_recovery_identity(self, runtime_identity=runtime_identity, world_size=1)
        load_update_recovery(checkpoint, identity)
        engine.load_checkpoint(local_path=str(checkpoint), del_local_after_load=False)
        return restore_population_recovery(self, checkpoint, runtime_identity=runtime_identity, world_size=1,
            native_applied_updates=engine.lr_scheduler.last_epoch, device=next(engine.module.parameters()).device)

    def run_update(self, engine: Any, index: int) -> Any:
        from ..policy_update_math import ScoreBundle
        from ..policy_update_replay import prepare_score_adjoints

        if engine.get_data_parallel_size() != 1 or index != self.next_update or not 0 <= index < len(self.updates):
            raise InvalidPolicyUpdate("native veRL requires qualified single-rank ordered resolved updates")
        if self._pending is not None:
            raise InvalidPolicyUpdate("failed native veRL update requires recovery before another attempt")
        if engine.lr_scheduler.last_epoch != self.global_applied_updates:
            raise InvalidPolicyUpdate("native veRL run-global counter differs from the active population boundary")
        update = self.updates[index]
        self.freeze_old_scores(engine)
        if self.prepare_sampler_correction is not None:
            correction = self.prepare_sampler_correction({action: float(value) for action, value in self.old.values.items()})
            if set(correction) != {record.action for record in update.population.actions} or any(
                type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in correction.values()
            ):
                raise InvalidPolicyUpdate("prepared correction requires complete detached finite action weights")
            self.sampler_correction = MappingProxyType(dict(correction))
            self.prepare_sampler_correction = None
        rows = self._rows(update.dependencies)
        sizes = self._pack_sizes(rows)
        planned = plan_packs(update, self.execution, self.capabilities)
        offset = 0
        for size, pack in zip(sizes, planned, strict=True):
            actions = tuple(action for _, members in rows[offset:offset + size] for action, _ in members)
            if set(actions) != set(pack.actions):
                raise InvalidPolicyUpdate("native context packing differs from resolved objective packs")
            offset += size
        current = self._infer(engine, rows)
        term = resolve_objective_term(update, self.spec, self.credit,
                                      parameter_version=f"{update.population.versions.current}/applied-{self.applied_updates}")
        if term.kl_weights:
            if self.reference is None or update.population.versions.reference is None:
                raise InvalidPolicyUpdate("native veRL KL requires frozen reference evidence")
            self.reference.validate(update.population, policy_version=update.population.versions.reference,
                                    score_contract=self.score_contract, score_temperature=self.score_temperature)
        self.last_adjoints = prepare_score_adjoints(
            term, self.credit, ScoreBundle(current, self.old.values, term.parameter_version,
                                          self.reference.values if self.reference is not None else {}, self.sampler_correction),
        )
        self._pending = index
        for retry in range(self.max_overflow_retries + 1):
            seen: set[ActionRef] = set()
            next_context = 0
            next_pack = 0

            def loss_function(model_output, data, dp_group=None, seen=seen):
                nonlocal next_context, next_pack
                indices = data["resolved_context_index"].flatten().tolist()
                if next_pack >= len(sizes) or indices != list(range(next_context, next_context + sizes[next_pack])):
                    raise InvalidPolicyUpdate("native veRL replay differs from declared original context packs")
                scores = self._scores(model_output, data, rows)
                if seen.intersection(scores):
                    raise InvalidPolicyUpdate("native veRL replay duplicated original action derivatives")
                seen.update(scores)
                next_context += sizes[next_pack]
                next_pack += 1
                if next_context == len(rows) and seen != set(update.dependencies):
                    raise InvalidPolicyUpdate("native veRL replay lost dependency coverage before optimizer step")
                carrier = self.last_adjoints.carrier(scores, parameter_version=term.parameter_version,
                    absolute_tolerance=self.replay_absolute_tolerance, relative_tolerance=self.replay_relative_tolerance)
                return carrier, {}

            self.attempts += 1
            with engine.train_mode():
                self.last_output = engine.train_batch(self._batch(rows), loss_function)
            skipped = bool(engine.last_loss_scale_metrics.get("optimizer_step_skipped", 0)) or not math.isfinite(
                float(self.last_output["metrics"]["grad_norm"])
            )
            if skipped:
                if retry == self.max_overflow_retries:
                    raise InvalidPolicyUpdate("native veRL overflow exhausted resolved retry limit without advancing policy")
                continue
            # The native engine's scalar losses are VJP carriers. Expose them
            # separately so observation never reports a carrier as the objective.
            self.last_output["replay_carrier_losses"] = self.last_output.pop("loss", None)
            evaluation = self.last_adjoints.evaluation
            self.last_output["loss"] = float(evaluation.loss)
            self.last_output["policy_loss"] = float(evaluation.policy_loss)
            self.last_output["kl_loss"] = float(evaluation.kl_loss)
            engine.lr_scheduler_step()
            self.applied_updates += 1
            self.next_update += 1
            self._pending = None
            return self.last_output
        raise AssertionError("bounded native retry loop failed to return or raise")


class ResolvedVeRLRun(ResolvedPolicyRun[ResolvedVeRLPopulation]):
    """Apply successive populations on one retained native model/optimizer engine."""

    def run_update(self, engine: Any) -> Any:
        active, index = self.occurrence(engine.lr_scheduler.last_epoch, engine.lr_scheduler.last_epoch)
        return active.run_update(engine, index)

    def save_native_checkpoint(self, engine: Any, checkpoint: Path, *, runtime_identity: str) -> Any:
        return self.active().save_native_checkpoint(engine, checkpoint, runtime_identity=runtime_identity)

    def load_native_checkpoint(self, engine: Any, checkpoint: Path, *, runtime_identity: str) -> Any:
        self.retain_checkpoint_population(checkpoint)
        return self.active().load_native_checkpoint(engine, checkpoint, runtime_identity=runtime_identity)
