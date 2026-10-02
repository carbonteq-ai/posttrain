"""Native CPU GradScaler skips exercise bounded retry, distinct from GPU gates."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from datasets import Dataset  # noqa: E402
from posttrain.train.backends.trl.policy_updates import (  # noqa: E402
    ResolvedTRLPopulation,
    resolved_policy_trainer_type,
)
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, resolve_updates  # noqa: E402
from posttrain.train.update_records import InvalidPolicyUpdate  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # noqa: E402
from trl.trainer.grpo_config import GRPOConfig  # noqa: E402
from trl.trainer.grpo_trainer import GRPOTrainer  # noqa: E402

from .test_update_execution import resolved  # noqa: E402


def run_native(folder, *, overflow=False, retries=0, always=False, scaled=True):
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
        "cpu-scaled@1",
        None,
        max_overflow_retries=retries,
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
            bos_token_id=0,
            eos_token_id=6,
            pad_token_id=0,
        )
    )
    hook_calls = []

    def inject(value):
        hook_calls.append(value.detach().clone())
        return torch.full_like(value, float("inf")) if overflow and (always or len(hook_calls) == 1) else value

    next(model.parameters()).register_hook(inject)
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
        lr_scheduler_type="linear",
        optim="adamw_torch",
        beta=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
        disable_dropout=False,
        report_to="none",
        disable_tqdm=True,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = resolved_policy_trainer_type(GRPOTrainer, population)(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"resolved_update": [0, 1]}),
        reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
    )
    # Exercise actual torch GradScaler and AcceleratedOptimizer on CPU. This
    # explicit fixture is not a claim that CPU fp16 is a supported user profile.
    previous = trainer.accelerator.state._mixed_precision
    if scaled:
        trainer.accelerator.state._mixed_precision = "fp16"
        trainer.accelerator.native_amp = True
        trainer.accelerator.scaler = torch.amp.GradScaler("cpu", init_scale=8 if overflow else 4)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    try:
        if always:
            with pytest.raises(InvalidPolicyUpdate, match="exhausted retry limit" if scaled else "unscaled nonfinite"):
                trainer.train()
        else:
            trainer.train()
    finally:
        trainer.accelerator.state._mixed_precision = previous
    return trainer, population, before, hook_calls


def test_native_scaler_retry_matches_control_with_stochastic_replay(tmp_path):
    control, expected, _, control_calls = run_native(tmp_path / "control")
    retried, actual, _, calls = run_native(tmp_path / "retry", overflow=True, retries=1)
    assert retried.state.global_step == actual.next_update == actual.applied_updates == 2
    assert actual.attempts == 3 and len(calls) == 3
    assert retried.lr_scheduler.last_epoch == control.lr_scheduler.last_epoch == 2
    assert retried.accelerator.scaler.get_scale() == control.accelerator.scaler.get_scale() == 4
    for left, right in zip(control.model.parameters(), retried.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for action in expected.old.values:
        torch.testing.assert_close(expected.old.values[action], actual.old.values[action], rtol=0, atol=0)
    # Compare at the same native scale. Inverse-scaling a different FP16
    # backward need not be bit-exact because intermediate rounding differs.
    torch.testing.assert_close(control_calls[0], calls[1], rtol=0, atol=0)
    torch.testing.assert_close(control_calls[1], calls[2], rtol=0, atol=0)


def test_native_scaler_exhaustion_keeps_occurrence_pending(tmp_path):
    trainer, population, before, calls = run_native(tmp_path, overflow=True, retries=2, always=True)
    assert trainer.state.global_step == population.next_update == population.applied_updates == 0
    assert population.attempts == len(calls) == 3 and population._pending == 0
    assert trainer.lr_scheduler.last_epoch == 0
    for name, value in trainer.model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_native_unscaled_nonfinite_gradient_cannot_mutate_policy(tmp_path):
    trainer, population, before, calls = run_native(tmp_path, overflow=True, retries=1, always=True, scaled=False)
    assert trainer.state.global_step == population.applied_updates == population.next_update == 0
    assert population.attempts == len(calls) == 1 and population._pending == 0
    assert trainer.lr_scheduler.last_epoch == 0
    for name, value in trainer.model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
