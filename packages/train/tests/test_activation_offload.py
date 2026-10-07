"""Overflow offload keeps resolved-update gradients exact while moving saved activations to host."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from posttrain.train.backends.activation_offload import offload_overflow  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")]


def _model():
    torch.manual_seed(0)
    config = transformers.Lfm2Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        max_position_embeddings=8192,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Lfm2ForCausalLM(config).half().cuda()
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    return model


def _gradients(budget):
    model = _model()
    decoder = model.get_decoder()
    torch.manual_seed(1)
    covers = [torch.randint(0, 512, (1, n), device="cuda") for n in (3000, 2000, 2500)]
    import contextlib

    context = offload_overflow(model, budget, min_bytes=1 << 16) if budget is not None else contextlib.nullcontext()
    with context as stats:
        loss = 0
        for ids in covers:  # graphs retained across packs, one backward, as resolved updates do
            hidden = decoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
            loss = loss + hidden.float().pow(2).mean()
    loss.backward()
    grads = {name: p.grad.detach().float().clone() for name, p in model.named_parameters() if p.grad is not None}
    return float(loss), grads, stats


def test_offloaded_activations_give_identical_loss_and_gradients():
    loss, grads, _ = _gradients(None)
    offloaded_loss, offloaded, stats = _gradients(0)
    assert stats.tensors > 0 and stats.bytes > 0
    assert offloaded_loss == loss
    assert offloaded.keys() == grads.keys()
    assert all(torch.equal(offloaded[name], grads[name]) for name in grads)


def test_a_budget_above_the_peak_moves_nothing():
    _, _, stats = _gradients(1 << 40)
    assert stats.tensors == 0 and stats.bytes == 0


def test_budget_must_be_valid():
    with pytest.raises(ValueError, match="nonnegative"):
        with offload_overflow(torch.nn.Linear(1, 1), -1):
            pass
