"""Chunked sampled-token log-probabilities equal full log-softmax autograd."""

import pytest

torch = pytest.importorskip("torch")

from posttrain.train.backends.policy_update_logprobs import selected_token_logprobs  # noqa: E402
from posttrain.train.update_records import InvalidPolicyUpdate  # noqa: E402


def reference(hidden, weight, bias, targets, temperature, softcap):
    logits = torch.nn.functional.linear(hidden, weight, bias).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    log_probabilities = (logits / temperature).log_softmax(dim=-1)
    values = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    entropies = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
    return values, entropies.detach()


@pytest.mark.parametrize("softcap", [None, 3.0])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("chunk_rows", [1, 4, 64])
def test_values_entropies_and_gradients_match_full_log_softmax(softcap, temperature, chunk_rows):
    torch.manual_seed(5)
    rows, width, vocabulary = 9, 6, 11
    leaves = [
        torch.randn(rows, width, dtype=torch.float64, requires_grad=True),
        torch.randn(vocabulary, width, dtype=torch.float64, requires_grad=True),
        torch.randn(vocabulary, dtype=torch.float64, requires_grad=True),
    ]
    targets = torch.randint(0, vocabulary, (rows,))
    upstream = torch.randn(rows, dtype=torch.float32)
    values, entropies = selected_token_logprobs(
        *leaves, targets, temperature=temperature, softcap=softcap, chunk_rows=chunk_rows
    )
    expected_values, expected_entropies = reference(*leaves, targets, temperature, softcap)
    torch.testing.assert_close(values, expected_values.float(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropies, expected_entropies.float(), rtol=1e-6, atol=1e-6)
    assert not entropies.requires_grad
    gradients = torch.autograd.grad((values * upstream).sum(), leaves)
    expected = torch.autograd.grad((expected_values * upstream.double()).sum(), leaves)
    for actual, wanted in zip(gradients, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-6)


def test_frozen_head_receives_no_gradient_and_nonfinite_logits_reject():
    hidden = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4)
    values, _ = selected_token_logprobs(hidden, weight, None, torch.tensor([0, 1, 4]), temperature=1.0)
    (gradient,) = torch.autograd.grad(values.sum(), (hidden,))
    assert gradient.shape == hidden.shape
    with pytest.raises(InvalidPolicyUpdate, match="non-finite"):
        selected_token_logprobs(torch.full((2, 4), float("nan")), weight, None, torch.tensor([0, 1]), temperature=1.0)
