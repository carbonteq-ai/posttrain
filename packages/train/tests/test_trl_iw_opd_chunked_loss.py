"""The chunked local IW-OPD loss equals the full-vocabulary loss, in value, gradient and normalization."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from posttrain.train.backends.trl.distillation import (  # noqa: E402
    _compute_local_iw_opd_loss_chunked,
    _scores_local_iw_opd,
)

VOCABULARY, WIDTH, PROMPT_LENGTH, COMPLETION_LENGTH = 17, 8, 4, 6
GAMMA, EPSILON = 0.5, 1e-6


class _Decoder(torch.nn.Module):
    def __init__(self, embedding: Any, body: Any) -> None:
        super().__init__()
        self.embedding, self.body = embedding, body

    def forward(self, input_ids, attention_mask, use_cache):  # noqa: ARG002
        return SimpleNamespace(last_hidden_state=torch.tanh(self.body(self.embedding(input_ids))))


class _TinyCausalLM(torch.nn.Module):
    """The surface the loss reads: ``get_decoder``, ``get_output_embeddings`` and ``config``."""

    def __init__(self, seed: int, softcap: float | None) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = torch.nn.Embedding(VOCABULARY, WIDTH)
        self.body = torch.nn.Linear(WIDTH, WIDTH)
        self.lm_head = torch.nn.Linear(WIDTH, VOCABULARY, bias=True)
        self.config = SimpleNamespace(final_logit_softcapping=softcap)
        self.decoder = _Decoder(self.embedding, self.body)

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head


def _trainer(student, teacher, *, temperature: float):
    return SimpleNamespace(
        accelerator=SimpleNamespace(unwrap_model=lambda module: module),
        teacher_model=teacher,
        temperature=temperature,
        iw_opd_epsilon=EPSILON,
        iw_opd_gamma=GAMMA,
        model=SimpleNamespace(training=True),
        _metrics=defaultdict(lambda: defaultdict(list)),
        _compute_prompt_length=lambda inputs: inputs["prompt_length"],
        _raise_if_local_teacher_tokenizer_mismatch=lambda: None,
    )


def _inputs(*, with_rollout_logprobs: bool = True):
    torch.manual_seed(11)
    batch = 3
    input_ids = torch.randint(0, VOCABULARY, (batch, PROMPT_LENGTH + COMPLETION_LENGTH))
    labels = input_ids.clone()
    labels[:, :PROMPT_LENGTH] = -100
    # Right-padded completions of different lengths, as an on-policy window produces.
    for row, length in enumerate((COMPLETION_LENGTH, 4, 2)):
        labels[row, PROMPT_LENGTH + length :] = -100
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "prompt_length": PROMPT_LENGTH,
    }
    if with_rollout_logprobs:
        inputs["rollout_logprobs"] = -torch.rand(input_ids.shape) * 4
    return inputs


def _reference_loss(student, teacher, inputs, *, temperature: float, softcap: float | None, denominator):
    """The IW-OPD objective from full-vocabulary log-softmax, written per sequence and per token."""

    def selected(model):
        hidden = model.get_decoder()(inputs["input_ids"], inputs["attention_mask"], False).last_hidden_state
        logits = model.lm_head(hidden[:, PROMPT_LENGTH - 1 : -1])
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        log_probabilities = (logits / temperature).log_softmax(dim=-1)
        return log_probabilities.gather(-1, inputs["input_ids"][:, PROMPT_LENGTH:].unsqueeze(-1)).squeeze(-1)

    student_logprobs = selected(student)
    with torch.no_grad():
        teacher_logprobs = selected(teacher)
    labels = inputs["labels"][:, PROMPT_LENGTH:]
    rollout = inputs["rollout_logprobs"][:, PROMPT_LENGTH:]
    total = student_logprobs.new_zeros(())
    for row in range(labels.shape[0]):
        positions = [t for t in range(labels.shape[1]) if labels[row, t] != -100]
        advantages = {t: (teacher_logprobs[row, t] - rollout[row, t]).detach() for t in positions}
        absolute_total = max(sum(abs(float(a)) for a in advantages.values()), EPSILON)
        seen = 0.0
        for t in positions:
            weight = 1.0 + GAMMA * (1.0 - seen / absolute_total)
            total = total - student_logprobs[row, t] * weight * advantages[t]
            seen += abs(float(advantages[t]))
    return total / denominator


def _gradients(model):
    return {name: parameter.grad.clone() for name, parameter in model.named_parameters()}


@pytest.mark.parametrize("softcap", [None, 3.0])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("chunk_rows", [1, 4, 256])
def test_chunked_loss_and_gradients_match_the_full_vocabulary_loss(softcap, temperature, chunk_rows):
    inputs = _inputs()
    teacher = _TinyCausalLM(seed=2, softcap=softcap)
    chunked_student = _TinyCausalLM(seed=1, softcap=softcap)
    reference_student = _TinyCausalLM(seed=1, softcap=softcap)
    valid_tokens = int((inputs["labels"] != -100).sum())

    loss = _compute_local_iw_opd_loss_chunked(
        _trainer(chunked_student, teacher, temperature=temperature),
        chunked_student,
        inputs,
        chunk_rows=chunk_rows,
        return_outputs=False,
        num_items_in_batch=None,
    )
    loss.backward()
    expected = _reference_loss(
        reference_student, teacher, inputs, temperature=temperature, softcap=softcap, denominator=valid_tokens
    )
    expected.backward()

    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    expected_gradients = _gradients(reference_student)
    for name, gradient in _gradients(chunked_student).items():
        torch.testing.assert_close(gradient, expected_gradients[name], rtol=1e-4, atol=1e-6)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_the_post_generation_count_in_the_inputs_is_the_loss_denominator():
    inputs = _inputs()
    student, teacher = _TinyCausalLM(seed=1, softcap=None), _TinyCausalLM(seed=2, softcap=None)
    trainer = _trainer(student, teacher, temperature=1.0)
    valid_tokens = float((inputs["labels"] != -100).sum())

    def loss_with(*, argument, stamped):
        return _compute_local_iw_opd_loss_chunked(
            trainer,
            student,
            {**inputs, **({"num_items_in_batch": stamped} if stamped is not None else {})},
            chunk_rows=4,
            return_outputs=False,
            num_items_in_batch=argument,
        )

    per_valid_token = loss_with(argument=None, stamped=None)
    # A window whose Trainer-level count was taken before generation can be zero; the stamped count must win.
    stamped = torch.tensor(2 * valid_tokens)
    torch.testing.assert_close(loss_with(argument=torch.tensor(0.0), stamped=stamped), per_valid_token / 2)
    torch.testing.assert_close(loss_with(argument=stamped, stamped=None), per_valid_token / 2)


def test_non_finite_rollout_logprobs_at_a_required_position_are_rejected():
    inputs = _inputs()
    inputs["rollout_logprobs"][0, PROMPT_LENGTH] = float("nan")
    student, teacher = _TinyCausalLM(seed=1, softcap=None), _TinyCausalLM(seed=2, softcap=None)
    with pytest.raises(ValueError, match="Rollout logprobs are non-finite or missing"):
        _compute_local_iw_opd_loss_chunked(
            _trainer(student, teacher, temperature=1.0),
            student,
            inputs,
            chunk_rows=4,
            return_outputs=False,
            num_items_in_batch=None,
        )


@pytest.mark.parametrize(
    ("use_teacher_server", "objective", "use_liger_loss", "expected"),
    [
        (False, "iw_opd", False, True),
        (True, "iw_opd", False, False),
        (False, "jsd", False, False),
        (False, "iw_opd", True, False),
    ],
)
def test_only_the_local_teacher_iw_opd_loss_without_liger_is_chunked(
    use_teacher_server, objective, use_liger_loss, expected
):
    trainer = SimpleNamespace(
        use_teacher_server=use_teacher_server, distillation_objective=objective, use_liger_loss=use_liger_loss
    )
    assert _scores_local_iw_opd(trainer) is expected
