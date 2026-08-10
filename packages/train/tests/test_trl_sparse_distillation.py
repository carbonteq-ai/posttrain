"""Numerical tests for the bounded-memory Gemma IW-OPD loss."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from posttrain.train.backends.trl.distillation import (  # noqa: E402
    _buffered_selected_token_count,
    _generate_heterogeneous_colocated_iw_opd_turns,
    _memory_safe_server_iw_opd_loss,
)


class _TextBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(48, 8)

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        assert attention_mask.shape == input_ids.shape
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Gemma4ForConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TextBackbone()
        self.lm_head = nn.Linear(8, 48, bias=False)
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
    from trl.experimental.iw_opd import IWOPDTrainer

    class FakeTrainer:
        use_teacher_server = True
        distillation_objective = "iw_opd"
        temperature = 1.0
        iw_opd_gamma = 0.5
        iw_opd_epsilon = 1e-8
        accelerator = _Accelerator()
        _metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }

        @staticmethod
        def _compute_prompt_length(inputs):
            return int(inputs["prompt_length"])

        @staticmethod
        def _get_teacher_token_logprobs_from_server(inputs, prompt_length):
            del inputs, prompt_length
            return teacher_result

        _compute_iw_opd_loss = IWOPDTrainer._compute_iw_opd_loss

    return FakeTrainer()


def _inputs(*, batch_size: int = 1):
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]] * batch_size)
    labels = torch.tensor([[-100, -100, 3, 4, 5, 6]] * batch_size)
    rollout_logprobs = torch.tensor([[0.0, 0.0, -1.4, -1.0, -1.8, -1.2]] * batch_size)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "rollout_logprobs": rollout_logprobs,
        "prompt_length": 2,
    }


def _full_loss(trainer, model, inputs, teacher_actual, denominator):
    trainer.model = model
    hidden = model.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[:, 1:-1, :]
    logits = model.lm_head(hidden)
    logits = torch.tanh(logits / 30.0) * 30.0
    return trainer._compute_iw_opd_loss(
        student_logits=logits,
        completion_tokens=inputs["input_ids"][:, 2:],
        labels=inputs["labels"][:, 2:],
        teacher_actual_logprobs=teacher_actual,
        rollout_logprobs=inputs["rollout_logprobs"][:, 2:],
        num_items_in_batch=denominator,
    )


def test_chunked_iw_opd_matches_full_logits_loss_and_gradients() -> None:
    torch.manual_seed(7)
    chunked_model = Gemma4ForConditionalGeneration()
    full_model = Gemma4ForConditionalGeneration()
    full_model.load_state_dict(chunked_model.state_dict())
    inputs = _inputs()
    teacher_actual = torch.tensor([[-1.2, -0.7, -2.0, -1.5]])
    teacher_result = {"actual_logprobs": teacher_actual}
    trainer = _trainer(teacher_result)

    chunked_loss = _memory_safe_server_iw_opd_loss(
        trainer,
        chunked_model,
        inputs,
        return_outputs=False,
        num_items_in_batch=torch.tensor(4),
        chunk_size=2,
    )
    chunked_loss.backward()
    full_loss = _full_loss(trainer, full_model, inputs, teacher_actual, torch.tensor(4))
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


def test_four_physical_slices_match_one_logical_iw_opd_batch() -> None:
    torch.manual_seed(11)
    sliced_model = Gemma4ForConditionalGeneration()
    full_model = Gemma4ForConditionalGeneration()
    full_model.load_state_dict(sliced_model.state_dict())
    inputs = _inputs(batch_size=4)
    inputs["labels"][1, -1] = -100
    inputs["labels"][2, -2:] = -100
    inputs["labels"][3, -3:] = -100
    teacher_actual = torch.tensor(
        [
            [-1.2, -0.7, -2.0, -1.5],
            [-0.9, -1.1, -1.7, float("-inf")],
            [-1.0, -1.4, float("-inf"), float("-inf")],
            [-0.8, float("-inf"), float("-inf"), float("-inf")],
        ]
    )
    denominator = inputs["labels"].ne(-100).sum()

    full_trainer = _trainer({"actual_logprobs": teacher_actual})
    full_loss = _full_loss(full_trainer, full_model, inputs, teacher_actual, denominator)
    full_loss.backward()

    sliced_loss = torch.zeros(())
    buffered = []
    for row in range(4):
        row_inputs = {
            key: value[row : row + 1] if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        buffered.append(row_inputs)
        trainer = _trainer({"actual_logprobs": teacher_actual[row : row + 1]})
        loss = _memory_safe_server_iw_opd_loss(
            trainer,
            sliced_model,
            row_inputs,
            return_outputs=False,
            num_items_in_batch=denominator,
            chunk_size=2,
        )
        sliced_loss = sliced_loss + loss.detach()
        loss.backward()

    assert _buffered_selected_token_count(SimpleNamespace(_buffered_inputs=buffered)) == denominator
    assert sliced_loss.item() == pytest.approx(full_loss.item(), rel=1e-6, abs=1e-7)
    assert torch.allclose(
        sliced_model.lm_head.weight.grad,
        full_model.lm_head.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        sliced_model.model.embedding.weight.grad,
        full_model.model.embedding.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_iw_opd_rejects_missing_teacher_logprob_for_selected_token() -> None:
    inputs = _inputs()
    teacher_actual = torch.tensor([[-1.2, float("-inf"), -2.0, -1.5]])
    with pytest.raises(ValueError, match="teacher logprobs are missing"):
        _memory_safe_server_iw_opd_loss(
            _trainer({"actual_logprobs": teacher_actual}),
            Gemma4ForConditionalGeneration(),
            inputs,
            return_outputs=False,
            num_items_in_batch=torch.tensor(4),
            chunk_size=2,
        )


def test_heterogeneous_iw_opd_turns_use_one_aligned_vllm_request() -> None:
    class FakeLlm:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, prompts, *, sampling_params, use_tqdm, lora_request):
            self.calls.append((prompts, sampling_params, use_tqdm, lora_request))
            values = []
            for index, _prompt in enumerate(prompts):
                item = SimpleNamespace(rank=1, logprob=-0.1 - index)
                output = SimpleNamespace(token_ids=[10 + index], logprobs=[{10 + index: item}])
                values.append(SimpleNamespace(outputs=[output]))
            return values

        def wake_up(self, *, tags):
            assert tags == ["kv_cache"]

    class FakeGeneration:
        mode = "colocate"
        tensor_parallel_size = 1
        repetition_penalty = 1.0
        temperature = 1.0
        top_p = 1.0
        top_k = 0
        min_p = 0.0
        logprobs = 0
        generation_kwargs = {"seed": 7}
        enable_sleep_mode = True
        max_num_seqs = 4
        _kv_cache_peak_tracker = None
        _lora_request = None
        last_generation_metrics = {}

        def __init__(self) -> None:
            self.llm = FakeLlm()
            self.sync_count = 0

        def sync_weights(self):
            self.sync_count += 1

        def _wake_weights_for_generation(self):
            return None

        def _collect_generation_metrics(self):
            self.last_generation_metrics = {}

        def _sleep_colocated_engine(self):
            return None

    class FakeContext:
        def __init__(self) -> None:
            self.values = {}

        def metric(self, name, value):
            self.values[name] = value

    generation = FakeGeneration()
    trainer = SimpleNamespace(
        use_vllm=True,
        vllm_generation=generation,
        state=SimpleNamespace(global_step=0),
        _last_vllm_sync_step=-1,
        vllm_sync_frequency=1,
        model=SimpleNamespace(training=True),
        _metrics={"train": {}, "eval": {}},
    )
    context = FakeContext()

    completion_ids, logprobs = _generate_heterogeneous_colocated_iw_opd_turns(
        context,
        trainer,
        [[1, 2], [3, 4]],
        [
            {"max_tokens": 7, "structured_outputs": {"json": {"const": "first"}}},
            {"max_tokens": 11, "structured_outputs": {"json": {"const": "second"}}},
        ],
    )

    assert completion_ids == [[10], [11]]
    assert logprobs == [[-0.1], [-1.1]]
    assert generation.sync_count == 1
    assert len(generation.llm.calls) == 1
    prompts, parameters, use_tqdm, lora_request = generation.llm.calls[0]
    assert prompts == [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3, 4]}]
    assert [item.max_tokens for item in parameters] == [7, 11]
    assert [item.structured_outputs.json for item in parameters] == [
        {"const": "first"},
        {"const": "second"},
    ]
    assert use_tqdm is False
    assert lora_request is None
    assert context.values == {
        "train/rollout/request_batch_size": 2,
        "train/rollout/resident_wave_size": 2,
    }
