"""Numerical tests for the bounded-memory Gemma OPD loss."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from posttrain.train.backends.trl.distillation import (  # noqa: E402
    _memory_safe_server_sparse_loss,
)


class _TextBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        assert attention_mask.shape == input_ids.shape
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Gemma4ForConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TextBackbone()
        self.lm_head = nn.Linear(8, 32, bias=False)
        text_config = SimpleNamespace(final_logit_softcapping=30.0)
        self.config = SimpleNamespace(get_text_config=lambda: text_config)

    def get_output_embeddings(self):
        return self.lm_head


class _Accelerator:
    num_processes = 1

    @staticmethod
    def unwrap_model(model):
        return model


def _trainer(teacher_result):
    from trl.experimental.distillation import DistillationTrainer

    class FakeTrainer:
        use_teacher_server = True
        beta = 1.0
        loss_top_k = 1
        reverse_kl_top_1_mode = "sampled"
        loss_add_tail = True
        temperature = 1.0
        accelerator = _Accelerator()

        @staticmethod
        def _compute_prompt_length(inputs):
            return int(inputs["prompt_length"])

        @staticmethod
        def _get_teacher_token_logprobs_from_server(inputs, prompt_length):
            del inputs, prompt_length
            return teacher_result

        _compute_sparse_top_1_divergence_loss = (
            DistillationTrainer._compute_sparse_top_1_divergence_loss
        )
        _reduce_divergence_loss = staticmethod(DistillationTrainer._reduce_divergence_loss)

    return FakeTrainer()


def test_chunked_sparse_loss_matches_full_logits_loss_and_gradients() -> None:
    torch.manual_seed(7)
    chunked_model = Gemma4ForConditionalGeneration()
    full_model = Gemma4ForConditionalGeneration()
    full_model.load_state_dict(chunked_model.state_dict())
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4, 5, 6]]),
        "prompt_length": 2,
    }
    actual_teacher = torch.tensor([[-1.2, -0.7, -2.0, -1.5]])
    teacher_result = {
        "actual_logprobs": actual_teacher,
        "topk_logprobs": actual_teacher.unsqueeze(-1),
        "topk_token_ids": inputs["input_ids"][:, 2:].unsqueeze(-1),
    }

    trainer = _trainer(teacher_result)
    chunked_loss = _memory_safe_server_sparse_loss(
        trainer,
        chunked_model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        chunk_size=2,
    )
    chunked_loss.backward()

    hidden = full_model.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[:, 1:-1, :]
    logits = full_model.lm_head(hidden)
    logits = torch.tanh(logits / 30.0) * 30.0
    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    full_loss = trainer._compute_sparse_top_1_divergence_loss(
        student_log_probs=student_log_probs,
        teacher_top1_token_ids=teacher_result["topk_token_ids"].squeeze(-1),
        teacher_top1_logprobs=teacher_result["topk_logprobs"].squeeze(-1),
        reverse_token_ids=inputs["input_ids"][:, 2:],
        reverse_teacher_logprobs=teacher_result["actual_logprobs"],
        labels=inputs["labels"][:, 2:],
        num_items_in_batch=torch.tensor(4),
    )
    full_loss.backward()

    assert chunked_loss.item() == pytest.approx(full_loss.item(), rel=1e-6, abs=1e-7)
    assert torch.allclose(
        chunked_model.lm_head.weight.grad,
        full_model.lm_head.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        chunked_model.model.embedding.weight.grad,
        full_model.model.embedding.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
