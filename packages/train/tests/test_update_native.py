"""Optional native trainer-loop checks, distinct from pretrained GPU gates."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from datasets import Dataset  # noqa: E402
from posttrain.train.backends.trl.policy_updates import (  # noqa: E402
    ResolvedTRLPopulation,
    ResolvedTRLRun,
    resolved_policy_trainer_type,
)
from posttrain.train.profiles import SAMPOSettings, TrainingLoop  # noqa: E402
from posttrain.train.update_plan import (  # noqa: E402
    PolicyExecutionBudget,
    PolicyUpdateSchedule,
    PolicyUpdateSettings,
    resolve_updates,
)
from posttrain.train.update_resolution import resolve_policy_population  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # noqa: E402
from trl.trainer.grpo_config import GRPOConfig  # noqa: E402
from trl.trainer.grpo_trainer import GRPOTrainer  # noqa: E402

from .test_update_execution import resolved  # noqa: E402


def retained_identity(retained) -> tuple:
    """Population-sized records compare by digest; counters and small records compare structurally."""
    resolved = retained.resolved
    return (
        resolved.snapshot.digest,
        resolved.credit.digest,
        resolved.spec,
        tuple(update.digest for update in resolved.updates),
        resolved.packs,
        resolved.execution,
        resolved.capabilities,
        retained.max_overflow_retries,
        retained.applied_update_offset,
        retained.attempt_offset,
    )


@pytest.mark.parametrize("minibatch", [1, 2])
def test_native_trl_train_applies_each_declared_update_once(tmp_path, minibatch):
    torch.manual_seed(19)
    snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
    # Two declared passes: either one or two minibatches per frozen population.
    updates = resolve_updates(snapshot, PolicyUpdateSchedule("episode", minibatch, epochs=2), update.objective)
    selection = SAMPOSettings(
        id="native-resolved",
        loop=TrainingLoop(max_steps=len(updates), per_device_batch_size=1),
        clip_epsilon_low=spec.clip_low,
        clip_epsilon_high=spec.clip_high,
        policy_updates=PolicyUpdateSettings(
            PolicyUpdateSchedule("episode", minibatch, epochs=2), PolicyExecutionBudget(1, 100, 10000)
        ),
    )
    prepared = resolve_policy_population(snapshot, credit, selection, capabilities)
    population = ResolvedTRLPopulation.from_resolved(
        prepared,
        read_input=lambda view: source,
        score_temperature=0.7,
        score_contract="causal@1",
        sampler_correction=None,
    )
    assert [item.digest for item in population.updates] == [item.digest for item in updates]
    assert population.credit is prepared.credit
    config = GPT2Config(
        vocab_size=7,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        bos_token_id=0,
        eos_token_id=6,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{index}": index for index in range(7)}, unk_token="t0")),
        pad_token="t0",
        eos_token="t6",
    )
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        generation_batch_size=2,
        num_generations=2,
        num_iterations=1,
        max_steps=len(updates),
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        beta=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",
        disable_tqdm=True,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer_type = resolved_policy_trainer_type(GRPOTrainer, population)
    trainer = trainer_type(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"resolved_update": list(range(len(updates)))}),
        reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
    )
    trainer.train()
    assert trainer.state.global_step == population.applied_updates == population.next_update == len(updates)
    assert population.attempts == len(updates)
    assert population.old.policy_version == snapshot.versions.old_score
    assert population.old.values.shape == (snapshot.size,) and not population.old.values.requires_grad
    assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())
    assert population.last_evaluation.parameter_version.endswith(f"applied-{len(updates) - 1}")
    # The kept evaluation is evidence only: a graph kept past the trainer's step
    # would hold the update's leaves (and their gradients) on the device.
    evaluation = population.last_evaluation
    assert all(
        value.grad_fn is None and not value.requires_grad
        for value in (evaluation.loss, evaluation.policy_loss, evaluation.kl_loss)
    )


def test_resolved_adapter_rejects_native_double_scheduling():
    snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
    population = ResolvedTRLPopulation(
        (update,),
        credit,
        spec,
        PolicyExecutionBudget(1, 100, 10000),
        capabilities,
        lambda view: source,
        1.0,
        "causal@1",
        None,
    )
    from types import SimpleNamespace

    class Parent:
        def __init__(self):
            self.args = SimpleNamespace(gradient_accumulation_steps=2)
            self.num_iterations = 1

    with pytest.raises(ValueError, match="replace native accumulation"):
        resolved_policy_trainer_type(Parent, population)()


def test_overflow_does_not_advance_resolved_applied_cursor():
    from types import SimpleNamespace

    from .test_update_scoring import CausalModel

    snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
    population = ResolvedTRLPopulation(
        (update,),
        credit,
        spec,
        PolicyExecutionBudget(1, 100, 10000),
        capabilities,
        lambda view: source,
        1.0,
        "causal@1",
        None,
    )
    population.loss(CausalModel(), 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=True)
    population.before_step(optimizer)
    with pytest.raises(ValueError, match="exhausted retry limit"):
        population.complete_step(optimizer)
    assert population.next_update == population.applied_updates == 0
    assert population.attempts == 1
    assert population._pending == 0


def test_retry_requires_a_completed_skipped_attempt():
    from types import SimpleNamespace

    from .test_update_scoring import CausalModel

    snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
    population = ResolvedTRLPopulation(
        (update,),
        credit,
        spec,
        PolicyExecutionBudget(1, 100, 10000),
        capabilities,
        lambda view: source,
        1.0,
        "causal@1",
        None,
    )
    model = CausalModel()
    population.loss(model, 0, torch.device("cpu"))
    with pytest.raises(ValueError, match="completed skipped"):
        population.prepare_retry()
    optimizer = SimpleNamespace(step_was_skipped=True)
    population.before_step(optimizer)
    assert population.complete_step(optimizer, allow_retry=True) is False
    population.prepare_retry()
    with pytest.raises(ValueError, match="completed skipped"):
        population.prepare_retry()
    population.loss(model, 0, torch.device("cpu"))
    optimizer.step_was_skipped = False
    population.before_step(optimizer)
    assert population.complete_step(optimizer) is True
    assert population.applied_updates == population.next_update == 1 and population.attempts == 2


def test_native_trl_checkpoint_resume_matches_uninterrupted_update(tmp_path):
    from transformers import TrainerCallback

    class StopAfterFirst(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True

    def make(folder, *, stop=False):
        torch.manual_seed(31)
        snapshot, source, credit, spec, update, capabilities = resolved("sampo@1")
        updates = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2, epochs=2), update.objective)
        population = ResolvedTRLPopulation(
            updates,
            credit,
            spec,
            PolicyExecutionBudget(1, 100, 10000),
            capabilities,
            lambda view: source,
            0.7,
            "causal@1",
            None,
        )
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=7,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                bos_token_id=0,
                eos_token_id=6,
                pad_token_id=0,
            )
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({f"t{index}": index for index in range(7)}, unk_token="t0")),
            pad_token="t0",
            eos_token="t6",
        )
        args = GRPOConfig(
            output_dir=str(folder),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            generation_batch_size=2,
            num_generations=2,
            num_iterations=1,
            max_steps=2,
            learning_rate=1e-3,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            beta=0,
            use_cpu=True,
            bf16=False,
            fp16=False,
            report_to="none",
            disable_tqdm=True,
            save_strategy="steps" if stop else "no",
            save_steps=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        trainer = resolved_policy_trainer_type(
            GRPOTrainer, population, recovery_runtime_identity="native-cpu-fixture@1"
        )(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"resolved_update": [0, 1]}),
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            callbacks=[StopAfterFirst()] if stop else [],
        )
        return trainer, population

    uninterrupted, expected = make(tmp_path / "uninterrupted")
    uninterrupted.train()
    interrupted, first = make(tmp_path / "interrupted", stop=True)
    interrupted.train()
    assert first.next_update == first.applied_updates == 1
    checkpoint = tmp_path / "interrupted" / "checkpoint-1"
    assert (checkpoint / "posttrain-resolved-update.json").is_file()
    resumed, actual = make(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == actual.next_update == actual.applied_updates == actual.attempts == 2
    for left, right in zip(uninterrupted.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(expected.old.values, actual.old.values, rtol=0, atol=0)


@pytest.mark.parametrize("stop_step", [2, 3])
def test_native_trl_continuous_populations_resume_without_recollection_or_optimizer_reset(tmp_path, stop_step):
    import json
    from dataclasses import replace

    from posttrain.train.backends.policy_update_recovery import load_retained_population, population_recovery_identity
    from posttrain.train.update_objectives import objective_population
    from posttrain.train.update_transport import decode_population_payload
    from transformers import TrainerCallback

    from .test_update_scoring import inputs as native_inputs

    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == stop_step:
                control.should_training_stop = True

    def make(folder, stop=False):
        collected, restored = [], []

        def build(offset, attempts):
            snapshot, source, credit, spec, _, capabilities = resolved("sampo@1")
            snapshot = replace(
                snapshot,
                id=f"population-{offset}",
                versions=replace(
                    snapshot.versions,
                    sampler=f"sampler-{offset}",
                    old_score=f"old-{offset}",
                    current=f"policy-{offset}",
                ),
            )
            credit = replace(credit, population_digest=snapshot.digest)
            objective = objective_population(snapshot, spec, credit)
            updates = resolve_updates(snapshot, PolicyUpdateSchedule("episode", 2, epochs=2), objective)
            return ResolvedTRLPopulation(
                updates,
                credit,
                spec,
                PolicyExecutionBudget(1, 100, 10000),
                capabilities,
                lambda view: source,
                0.7,
                "causal@1",
                None,
                applied_update_offset=offset,
                attempt_offset=attempts,
            )

        def collect(offset, attempts):
            assert trainer.state.global_step == offset
            assert len(collected) == 0 or collected[-1].next_update == 2
            runtime = build(offset, attempts)
            collected.append(runtime)
            return runtime

        def restore(checkpoint):
            payload = json.loads((checkpoint / "posttrain-update-population.json").read_text())
            retained = decode_population_payload(payload)
            source = native_inputs()
            runtime = ResolvedTRLPopulation.from_resolved(
                retained.resolved,
                read_input=lambda view: source,
                score_temperature=0.7,
                score_contract="causal@1",
                sampler_correction=None,
                max_overflow_retries=retained.max_overflow_retries,
                applied_update_offset=retained.applied_update_offset,
                attempt_offset=retained.attempt_offset,
            )
            identity = population_recovery_identity(runtime, runtime_identity="native-cpu-fixture@1", world_size=1)
            loaded = load_retained_population(checkpoint, identity, sampler_correction=None)
            assert retained_identity(loaded) == retained_identity(retained)
            restored.append(runtime)
            return runtime

        run = ResolvedTRLRun(collect, restore)
        torch.manual_seed(31)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=7,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                bos_token_id=0,
                eos_token_id=6,
                pad_token_id=0,
            )
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({f"t{index}": index for index in range(7)}, unk_token="t0")),
            pad_token="t0",
            eos_token="t6",
        )
        args = GRPOConfig(
            output_dir=str(folder),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            generation_batch_size=2,
            num_generations=2,
            num_iterations=1,
            max_steps=4,
            learning_rate=1e-3,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            beta=0,
            use_cpu=True,
            bf16=False,
            fp16=False,
            report_to="none",
            disable_tqdm=True,
            save_strategy="steps" if stop else "no",
            save_steps=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )
        trainer = resolved_policy_trainer_type(GRPOTrainer, run, recovery_runtime_identity="native-cpu-fixture@1")(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"resolved_update": list(range(4))}),
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            callbacks=[Stop()] if stop else [],
        )
        return trainer, run, collected, restored

    uninterrupted, expected, collected, _ = make(tmp_path / "full")
    uninterrupted.train()
    assert [p.applied_update_offset for p in collected] == [0, 2]
    assert collected[0].old.policy_version == "old-0" and collected[1].old.policy_version == "old-2"
    assert not torch.equal(collected[0].old.values, collected[1].old.values)
    interrupted, _, _, _ = make(tmp_path / "interrupted", stop=True)
    interrupted.train()
    checkpoint = tmp_path / "interrupted" / f"checkpoint-{stop_step}"
    resumed, actual, recollected, restored = make(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert [p.applied_update_offset for p in restored] == [0 if stop_step == 2 else 2]
    assert [p.applied_update_offset for p in recollected] == ([2] if stop_step == 2 else [])
    assert resumed.state.global_step == actual.active().global_applied_updates == actual.active().global_attempts == 4
    assert resumed.lr_scheduler.last_epoch == uninterrupted.lr_scheduler.last_epoch == 4
    for left, right in zip(uninterrupted.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    left_state, right_state = uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    for parameter, values in left_state["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, right_state["state"][parameter][key], rtol=0, atol=0)
    torch.testing.assert_close(expected.active().old.values, actual.active().old.values, rtol=0, atol=0)
