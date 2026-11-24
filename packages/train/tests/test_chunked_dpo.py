"""Chunked DPO equals TRL's full-logits DPO (loss, gradients, reference scores) on a tiny Qwen3.5 with LoRA."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
trl_dpo = pytest.importorskip("trl.trainer.dpo_trainer")

from posttrain.train.backends.trl import chunked_dpo  # noqa: E402
from posttrain.train.profiles import DPOSettings, TrainingLoop  # noqa: E402

VOCAB = 97


def tiny_qwen35(seed: int = 0, lm_head_adapter: bool = False):
    config = transformers.Qwen3_5Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "vocab_size": VOCAB,
            "tie_word_embeddings": True,
            "eos_token_id": 1,
        },
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32},
        tie_word_embeddings=True,
    )
    torch.manual_seed(seed)
    model = transformers.Qwen3_5ForConditionalGeneration(config).float()
    targets = ["lm_head", "q_proj"] if lm_head_adapter else "all-linear"
    lora = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=targets, exclude_modules=r".*visual.*", init_lora_weights=False
    )
    return peft.get_peft_model(model, lora)


def batch(seed: int = 1, pairs: int = 2, length: int = 14):
    """TRL's collated layout: chosen rows then rejected rows, right-padded, completion mask with holes."""

    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, VOCAB, (2 * pairs, length), generator=generator)
    attention = torch.ones_like(ids)
    attention[1, -3:] = 0  # padding on one chosen row
    ids[1, -3:] = 0
    completion = torch.zeros_like(ids)
    completion[:, 5:] = 1  # prompt is the first 5 tokens
    completion[:, 8:10] = 0  # a tool result inside the completion
    completion[attention == 0] = 0
    return {"input_ids": ids, "attention_mask": attention, "completion_mask": completion}


def fake_trainer(model, loss_types, loss_weights, ref_model=None, precompute=True):
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        gather=lambda tensor: tensor,
        gather_for_metrics=lambda tensor: tensor,
        unwrap_model=lambda module: module,
    )
    return SimpleNamespace(
        model=model,
        ref_model=ref_model,
        accelerator=accelerator,
        args=SimpleNamespace(gradient_checkpointing_kwargs=None),
        aux_loss_enabled=False,
        ld_alpha=None,
        use_weighting=False,
        precompute_ref_logps=precompute,
        f_divergence_type="reverse_kl",
        loss_types=loss_types,
        loss_weights=loss_weights,
        beta=0.1,
        label_smoothing=0.0,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=0,
    )


def lora_grads(model):
    return {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}


def test_chunked_sequence_logps_equal_full_log_softmax_values_and_gradients():
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(3, 9, 8, generator=generator, requires_grad=True)
    weight = torch.randn(VOCAB, 8, generator=generator, requires_grad=True)
    labels = torch.randint(0, VOCAB, (3, 9), generator=generator)
    mask = (torch.rand(3, 9, generator=generator) > 0.3).long()

    full = torch.log_softmax(hidden @ weight.t(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected = (full * mask).sum(dim=1)
    expected.sum().backward()
    expected_grads = hidden.grad.clone(), weight.grad.clone()
    hidden.grad, weight.grad = None, None

    sums, counts = chunked_dpo.chunked_sequence_logps(hidden, weight, None, labels, mask, chunk_size=4)
    sums.sum().backward()
    assert torch.allclose(sums, expected, atol=1e-5)
    assert counts.tolist() == mask.sum(dim=1).tolist()
    assert torch.allclose(hidden.grad, expected_grads[0], atol=1e-5)
    assert torch.allclose(weight.grad, expected_grads[1], atol=1e-5)


def test_chunked_logps_ignore_masked_labels_and_see_trained_ones():
    """Both ways: changing the label at a mask-0 position (a tool-result token) leaves the score unchanged; changing
    the label at a trained position moves it. Hidden states are fixed, so only the label read differs."""

    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(2, 10, 8, generator=generator)
    weight = torch.randn(VOCAB, 8, generator=generator)
    labels = torch.randint(0, VOCAB, (2, 10), generator=generator)
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[0, 4:6] = 0
    base, _ = chunked_dpo.chunked_sequence_logps(hidden, weight, None, labels, mask, chunk_size=3)
    masked, trained = labels.clone(), labels.clone()
    masked[0, 4] = (masked[0, 4] + 1) % VOCAB
    trained[0, 7] = (trained[0, 7] + 1) % VOCAB
    after_masked, _ = chunked_dpo.chunked_sequence_logps(hidden, weight, None, masked, mask, chunk_size=3)
    after_trained, _ = chunked_dpo.chunked_sequence_logps(hidden, weight, None, trained, mask, chunk_size=3)
    assert torch.equal(after_masked, base)
    assert not torch.isclose(after_trained[0], base[0]) and torch.equal(after_trained[1], base[1])


@pytest.mark.parametrize(("loss_types", "loss_weights"), [(["sigmoid"], [1.0]), (["sigmoid", "sft"], [1.0, 1.0])])
def test_chunked_loss_and_gradients_equal_trl_full_logits_loss(loss_types, loss_weights):
    model = tiny_qwen35()
    model.train()
    inputs = batch()
    inputs["ref_chosen_logps"] = torch.tensor([-60.0, -55.0])
    inputs["ref_rejected_logps"] = torch.tensor([-58.0, -61.0])

    reference_trainer = fake_trainer(model, loss_types, loss_weights)
    expected = trl_dpo.DPOTrainer._compute_loss(reference_trainer, model, dict(inputs), False)
    expected.backward()
    expected_grads = lora_grads(model)
    model.zero_grad(set_to_none=True)

    trainer = fake_trainer(model, loss_types, loss_weights)
    loss = chunked_dpo.compute_loss(trainer, model, dict(inputs), False)
    loss.backward()
    grads = lora_grads(model)

    assert torch.allclose(loss, expected, atol=1e-5), (loss.item(), expected.item())
    assert expected_grads and set(grads) == set(expected_grads)
    for name, grad in grads.items():
        assert torch.allclose(grad, expected_grads[name], atol=1e-5, rtol=1e-4), name
    for key in ("rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins", "logps/chosen"):
        assert trainer._metrics["train"][key] == pytest.approx(reference_trainer._metrics["train"][key], abs=1e-5)


def test_chunked_reference_scores_equal_trl_reference_scores():
    model = tiny_qwen35()
    reference = tiny_qwen35(seed=5).merge_and_unload()  # a merged frozen adapter, as load_frozen_adapter_reference
    reference.eval()
    inputs = batch()
    trainer = fake_trainer(model, ["sigmoid"], [1.0], ref_model=reference)
    expected = trl_dpo.DPOTrainer.compute_ref_log_probs(trainer, reference, dict(inputs))
    got = chunked_dpo.reference_logps(trainer, reference, dict(inputs))
    for mine, theirs in zip(got, expected, strict=True):
        assert torch.allclose(mine, theirs, atol=1e-5)


def test_live_reference_forward_equals_precomputed_scores():
    model = tiny_qwen35()
    reference = tiny_qwen35(seed=5).merge_and_unload()
    reference.eval()
    inputs = batch()
    chosen, rejected = chunked_dpo.reference_logps(fake_trainer(model, ["sigmoid"], [1.0]), reference, dict(inputs))
    live = chunked_dpo.compute_loss(
        fake_trainer(model, ["sigmoid"], [1.0], ref_model=reference, precompute=False), model, dict(inputs), False
    )
    stored = chunked_dpo.compute_loss(
        fake_trainer(model, ["sigmoid"], [1.0]),
        model,
        {**inputs, "ref_chosen_logps": chosen, "ref_rejected_logps": rejected},
        False,
    )
    assert torch.allclose(live, stored, atol=1e-6)


def test_chunked_path_refuses_an_adapter_on_the_output_projection():
    model = tiny_qwen35(lm_head_adapter=True)
    with pytest.raises(ValueError, match="output projection"):
        chunked_dpo.completion_logps(model, batch())


def test_chunked_path_refuses_losses_it_does_not_implement():
    model = tiny_qwen35()
    inputs = {**batch(), "ref_chosen_logps": torch.zeros(2), "ref_rejected_logps": torch.zeros(2)}
    with pytest.raises(ValueError, match="sigmoid and sft"):
        chunked_dpo.compute_loss(fake_trainer(model, ["ipo"], [1.0]), model, inputs, False)


def test_dpo_settings_accept_chunked_sigmoid_with_sft_and_refuse_other_losses():
    loop = TrainingLoop(max_steps=1)
    assert DPOSettings("tests/dpo", loop, loss_kernel="chunked", sft_weight=1.0).loss_kernel == "chunked"
    with pytest.raises(ValueError, match="chunked DPO loss kernel"):
        DPOSettings("tests/dpo", loop, loss_kernel="chunked", preference_loss="sigmoid_norm")


def test_an_adapter_saved_on_the_image_text_class_attaches_only_under_that_class(tmp_path):
    """Both ways: a trained Qwen3.5 adapter saved from the image-text class loads under that class (non-zero B) and
    is refused under the causal-LM class, where PEFT would otherwise leave every B at zero (the base model)."""

    from posttrain.train.backends.trl.common import require_loaded_adapter

    trained = tiny_qwen35()  # init_lora_weights=False: non-zero B, like a trained adapter
    trained.save_pretrained(tmp_path / "adapter")
    config = trained.get_base_model().config

    torch.manual_seed(0)
    same_class = transformers.Qwen3_5ForConditionalGeneration(config).float()
    loaded = peft.PeftModel.from_pretrained(same_class, tmp_path / "adapter")
    require_loaded_adapter(loaded, tmp_path / "adapter")

    torch.manual_seed(0)
    text_only = transformers.Qwen3_5ForCausalLM(config.text_config).float()
    with pytest.warns(UserWarning):
        mismatched = peft.PeftModel.from_pretrained(text_only, tmp_path / "adapter")
    with pytest.raises(ValueError, match="did not attach"):
        require_loaded_adapter(mismatched, tmp_path / "adapter")
