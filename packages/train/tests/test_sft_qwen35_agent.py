"""Qwen3.5 agent SFT: no silent truncation, thinking-off targets, per-epoch checkpoints, and
adapters an image-text evaluation runtime loads (Phase 2A, branch sft-phase2a)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from posttrain.common.variants import QWEN_35_08B
from posttrain.data import SupervisedDataset, SupervisedExample
from posttrain.train import QWEN35_RENDERER, QWEN35_THINKING_RENDERER, LoRAUpdate, SFTSettings, TrainingLoop, render_supervised
from posttrain.train.backends.trl.common import model_class_option, trainable_model_factory
from posttrain.train.backends.trl.sft import epoch_loop, steps_per_epoch
from posttrain.train.bindings import parameter_update_digest
from posttrain.train.catalog_schema import decode_training_selection

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "slack_list_channels",
            "description": "List channels.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    }
]


def _tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("renderers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            QWEN_35_08B.base.repo_id, revision=QWEN_35_08B.base.revision, local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3.5-0.8B tokenizer is not cached")


def _agent_example(identifier: str = "agent/one", *, reasoning: str | None = None) -> SupervisedExample:
    first: dict[str, Any] = {
        "role": "assistant",
        "content": "Let me check Slack first.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "slack_list_channels", "arguments": {"limit": 20}},
            }
        ],
    }
    if reasoning is not None:
        first["reasoning_content"] = reasoning
    return SupervisedExample(
        identifier,
        (
            {"role": "system", "content": "You are a workflow automation agent."},
            {"role": "user", "content": "Check the AP channel."},
            first,
            {"role": "tool", "tool_call_id": "call-1", "content": '{"channels": ["C_AP"]}'},
            {"role": "assistant", "content": "The AP channel is C_AP."},
        ),
        (2, 4),
        tools=tuple(TOOLS),
    )


def _dataset(*examples: SupervisedExample) -> SupervisedDataset:
    return SupervisedDataset("agent-sft-test", "a" * 40, examples)


def test_overlong_example_is_rejected_with_its_id_instead_of_truncated() -> None:
    tokenizer = _tokenizer()
    dataset = _dataset(_agent_example("agent/short"), _agent_example("agent/long"))
    full = render_supervised(tokenizer, QWEN_35_08B, dataset, QWEN35_RENDERER, max_length=100_000)
    length = len(full[0].input_ids)
    assert all(sample.source_length == len(sample.input_ids) for sample in full)

    # exactly the rendered length fits
    fitting = render_supervised(tokenizer, QWEN_35_08B, dataset, QWEN35_RENDERER, max_length=length)
    assert [len(sample.input_ids) for sample in fitting] == [length, length]

    with pytest.raises(ValueError, match="max_length") as error:
        render_supervised(tokenizer, QWEN_35_08B, dataset, QWEN35_RENDERER, max_length=length - 1)
    assert "'agent/short'" in str(error.value) and "'agent/long'" in str(error.value)
    assert f"({length} tokens)" in str(error.value)


def test_qwen35_thinking_off_targets_are_exactly_the_sampled_turns() -> None:
    tokenizer = _tokenizer()
    sample = render_supervised(tokenizer, QWEN_35_08B, _dataset(_agent_example()), QWEN35_RENDERER, max_length=4096)[0]
    ids, labels = list(sample.input_ids), list(sample.labels)

    trained_runs: list[str] = []
    current: list[int] = []
    for token, label in zip(ids, labels, strict=True):
        if label != -100:
            current.append(token)
        elif current:
            trained_runs.append(tokenizer.decode(current))
            current = []
    if current:
        trained_runs.append(tokenizer.decode(current))

    # the generation prompt prefills "<think>\n\n</think>\n\n" with thinking off: never a target
    assert trained_runs == [
        "Let me check Slack first.\n\n<tool_call>\n<function=slack_list_channels>\n<parameter=limit>\n20\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>",
        "The AP channel is C_AP.<|im_end|>",
    ]
    # the sequence ends at the last target (no untrained trailing newline)
    assert labels[-1] != -100 and tokenizer.decode(ids[-1:]) == "<|im_end|>"
    # the prompt part is what the runner's generation prompt renders
    prompt = tokenizer.apply_chat_template(
        [dict(message) for message in _agent_example().messages[:2]],
        tools=TOOLS,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    first_target = labels.index(next(label for label in labels if label != -100))
    assert tokenizer.decode(ids[:first_target]) == prompt


def test_qwen35_thinking_off_rejects_reasoning_targets() -> None:
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="empty thinking wrapper"):
        render_supervised(
            tokenizer,
            QWEN_35_08B,
            _dataset(_agent_example(reasoning="hidden plan")),
            QWEN35_RENDERER,
            max_length=4096,
        )


@pytest.mark.parametrize(
    ("examples", "batch", "accumulation", "expected"),
    ((37, 1, 16, 3), (32, 1, 16, 2), (33, 1, 16, 3), (16, 1, 16, 1), (5, 2, 2, 2), (180, 1, 16, 12)),
)
def test_steps_per_epoch_matches_trainer_counting(examples: int, batch: int, accumulation: int, expected: int) -> None:
    loop = TrainingLoop(max_steps=1, per_device_batch_size=batch, gradient_accumulation_steps=accumulation)
    assert steps_per_epoch(examples, loop) == expected


def test_epoch_loop_derives_steps_and_checkpoints_and_respects_bound() -> None:
    loop = TrainingLoop(max_steps=1000, gradient_accumulation_steps=16, checkpoint_steps=0, checkpoint_limit=1)
    derived = epoch_loop(loop, 5, 180)
    assert (derived.max_steps, derived.checkpoint_steps, derived.checkpoint_limit) == (60, 12, 5)
    assert epoch_loop(loop, None, 180) is loop
    with pytest.raises(ValueError, match="upper|bound"):
        epoch_loop(replace(loop, max_steps=59), 5, 180)


def test_catalog_decodes_sft_epochs_cosine_and_lora_exclusion() -> None:
    settings = decode_training_selection(
        _ref("training", "qwen3.5-4b/agent-sft-test"),
        {
            "selection_type": "sft-settings",
            "id": "qwen3.5-4b/agent-sft-test",
            "epochs": 5,
            "loop": {"max_steps": 2000, "lr_scheduler_type": "cosine"},
        },
        {},
    )
    assert isinstance(settings, SFTSettings)
    assert settings.epochs == 5 and settings.loop.lr_scheduler_type == "cosine"
    update = LoRAUpdate(16, 32, 0.0, "all-linear", exclude_modules=r".*visual.*")
    assert parameter_update_digest(LoRAUpdate(16, 32)) != parameter_update_digest(update)
    with pytest.raises(ValueError, match="regular expression"):
        LoRAUpdate(exclude_modules="(")
    assert model_class_option({"model_class": "image-text-to-text"}) == "image-text-to-text"
    with pytest.raises(ValueError, match="model_class"):
        model_class_option({"model_class": "vision"})


def test_existing_update_digest_is_unchanged_without_exclusion() -> None:
    # sha256 of the pre-change payload {"alpha":16,"dropout":0.0,"kind":"lora","rank":8,"target_modules":"all-linear"}
    import hashlib

    payload = '{"alpha":16,"dropout":0.0,"kind":"lora","rank":8,"target_modules":"all-linear"}'
    assert parameter_update_digest(LoRAUpdate()) == hashlib.sha256(payload.encode()).hexdigest()


def _ref(family: str, identifier: str) -> Any:
    from posttrain.common import CatalogRef

    return CatalogRef(family, identifier)


# ---------------------------------------------------------------- per-epoch checkpoints (Transformers trainer)


def test_epoch_loop_checkpoints_land_on_epoch_boundaries(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from posttrain.train.backends.trl.common import trainer_arguments

    examples, epochs = 37, 3
    loop = epoch_loop(
        TrainingLoop(max_steps=1000, gradient_accumulation_steps=16, learning_rate=1e-3, checkpoint_steps=0),
        epochs,
        examples,
    )
    arguments = trainer_arguments(loop, tmp_path / "out")
    accepted = set(transformers.TrainingArguments.__dataclass_fields__)
    arguments = {key: value for key, value in arguments.items() if key in accepted}
    arguments.update(bf16=False, use_cpu=True, gradient_checkpointing=False, dataloader_pin_memory=False)

    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    rows = [{"input_ids": [1, 2 + index % 50, 3], "labels": [1, 2 + index % 50, 3]} for index in range(examples)]

    saves: list[tuple[int, float]] = []

    class Record(transformers.TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            saves.append((state.global_step, state.epoch))
            return control

    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(**arguments),
        train_dataset=rows,
        callbacks=[Record()],
    )
    trainer.train()
    per_epoch = steps_per_epoch(examples, loop)
    assert saves == [(per_epoch * (index + 1), float(index + 1)) for index in range(epochs)]
    assert sorted(path.name for path in (tmp_path / "out").iterdir() if path.name.startswith("checkpoint-")) == [
        f"checkpoint-{per_epoch * (index + 1)}" for index in range(epochs)
    ]


# ---------------------------------------------------------------- adapter compatibility with the image-text runner


def _tiny_qwen35(tmp_path: Path) -> Path:
    """A random two-layer Qwen3.5 image-text checkpoint with the 4B's architecture switches."""

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    text = {
        "model_type": "qwen3_5_text",
        "attn_output_gate": True,
        "full_attention_interval": 2,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 512,
        "tie_word_embeddings": True,
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    }
    vision = {
        "model_type": "qwen3_5",
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 64,
        "num_position_embeddings": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "deepstack_visual_indexes": [],
    }
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": text,
                "vision_config": vision,
                "image_token_id": 500,
                "video_token_id": 501,
                "vision_start_token_id": 502,
                "vision_end_token_id": 503,
            }
        )
    )
    torch.manual_seed(0)
    config = transformers.AutoConfig.from_pretrained(root)
    transformers.AutoModelForImageTextToText.from_config(config).to(torch.bfloat16).save_pretrained(root)
    return root


def _framework_adapter(tmp_path: Path, base: Path, update: LoRAUpdate, model_class: str | None) -> Path:
    """Build the adapter the framework's SFT path trains and save it as ``trainer.save_model`` does."""

    from posttrain.train.backends.trl.common import framework_imports, load_trainable_model

    imports = dict(framework_imports())

    def local(factory: Any) -> Any:
        class Local:
            @staticmethod
            def from_pretrained(repo_id: str, **options: Any) -> Any:
                del repo_id
                options.pop("device_map", None)
                options.pop("revision", None)
                return factory.from_pretrained(base, **options)

        return Local

    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        imports[name] = local(imports[name])
    loop = TrainingLoop(max_steps=1, gradient_checkpointing=False)
    model = load_trainable_model(QWEN_35_08B, update, loop, imports, model_class=model_class)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data.normal_()  # a trained adapter: nonzero updates
    directory = tmp_path / f"adapter-{model_class}"
    model.save_pretrained(directory)
    return directory


def _runner_model(base: Path, rank: int = 16, alpha: int = 32) -> Any:
    """``Runtime.start`` of the evaluation runner (wt-base3 fd207334 scripts/qualification/sdpo_ab_runner.py
    lines 955-958): the image-text model with LoRA all-linear excluding the vision tower."""

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
            exclude_modules=r".*visual.*",
            task_type="CAUSAL_LM",
        ),
    )


def _load_into_runner(runner: Any, adapter: Path) -> Any:
    """``phase_collect_only --adapter`` (fd207334 lines 1250-1254)."""

    from peft import load_peft_weights, set_peft_model_state_dict

    result = set_peft_model_state_dict(runner, load_peft_weights(str(adapter)), adapter_name="default")
    runner.set_adapter("default")
    return result


def _verify_adapter_config(path: Path, rank: int, alpha: int) -> dict[str, Any]:
    """The runner's ``verify_adapter_config`` (fd207334 lines 2166-2174), verbatim in effect."""

    config = json.loads(path.read_text())
    if config.get("r") != rank or config.get("lora_alpha") != alpha:
        raise ValueError(f"adapter has r={config.get('r')} alpha={config.get('lora_alpha')}")
    return config


def test_image_text_lora_profile_adapter_loads_fully_into_the_evaluation_runner(tmp_path: Path) -> None:
    import torch
    from peft import get_peft_model_state_dict, load_peft_weights

    base = _tiny_qwen35(tmp_path)
    update = LoRAUpdate(16, 32, 0.0, "all-linear", exclude_modules=r".*visual.*")
    adapter = _framework_adapter(tmp_path, base, update, "image-text-to-text")

    config = _verify_adapter_config(adapter / "adapter_config.json", 16, 32)
    assert config["lora_dropout"] == 0.0 and config["exclude_modules"] == r".*visual.*"
    saved = load_peft_weights(str(adapter))
    assert not [key for key in saved if "visual" in key]

    runner = _runner_model(base)
    expected = get_peft_model_state_dict(runner, adapter_name="default")
    assert set(saved) == set(expected)
    assert {key: tuple(value.shape) for key, value in saved.items()} == {
        key: tuple(value.shape) for key, value in expected.items()
    }
    result = _load_into_runner(runner, adapter)
    assert not result.unexpected_keys
    loaded = get_peft_model_state_dict(runner, adapter_name="default")
    assert all(torch.equal(loaded[key], saved[key].to(loaded[key].dtype)) for key in saved)
    lora_b = [name for name, _ in runner.named_parameters() if "lora_B" in name]
    assert lora_b and all(
        parameter.abs().sum() > 0 for name, parameter in runner.named_parameters() if "lora_B" in name
    )


def test_text_model_adapter_does_not_load_into_the_evaluation_runner(tmp_path: Path) -> None:
    """Regression record: the default text-model load names modules ``model.layers``; the runner's
    image-text model names them ``model.language_model.layers`` and silently keeps a zero adapter."""

    base = _tiny_qwen35(tmp_path)
    adapter = _framework_adapter(tmp_path, base, LoRAUpdate(16, 32, 0.0, "all-linear"), None)
    runner = _runner_model(base)
    result = _load_into_runner(runner, adapter)
    assert result.unexpected_keys
    assert all(parameter.abs().sum() == 0 for name, parameter in runner.named_parameters() if "lora_B" in name)


def test_image_text_model_class_requires_an_image_text_variant() -> None:
    text_only = replace(
        QWEN_35_08B,
        capabilities=replace(QWEN_35_08B.capabilities, modalities=("text",)),
    )
    with pytest.raises(ValueError, match="not an image-text model"):
        trainable_model_factory(text_only, {}, "image-text-to-text")


def test_qwen35_thinking_on_targets_start_after_the_prefilled_opener() -> None:
    """Thinking on: the generation prompt ends with "<think>\\n", so each target starts with the reasoning text;
    the reasoning, "</think>", the visible text, the calls and the final reply are trained."""
    tokenizer = _tokenizer()
    base = _agent_example(reasoning="Need the channel list first.")
    final = {**base.messages[4], "reasoning_content": "The list answers it."}
    example = SupervisedExample(base.id, (*base.messages[:4], final), base.trainable_message_indices, tools=tuple(TOOLS))
    sample = render_supervised(tokenizer, QWEN_35_08B, _dataset(example), QWEN35_THINKING_RENDERER, max_length=4096)[0]
    ids, labels = list(sample.input_ids), list(sample.labels)
    runs: list[str] = []
    current: list[int] = []
    for token, label in zip(ids, labels, strict=True):
        if label != -100:
            current.append(token)
        elif current:
            runs.append(tokenizer.decode(current))
            current = []
    if current:
        runs.append(tokenizer.decode(current))
    assert runs[0].startswith("Need the channel list first.\n</think>\n\nLet me check Slack first.")
    assert runs[0].endswith("</tool_call><|im_end|>") and runs[1] == "The list answers it.\n</think>\n\nThe AP channel is C_AP.<|im_end|>"
    assert not any(run.startswith("<think>") for run in runs)
    prompt = tokenizer.apply_chat_template(
        [dict(message) for message in example.messages[:2]],
        tools=TOOLS,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    first_target = labels.index(next(label for label in labels if label != -100))
    assert tokenizer.decode(ids[:first_target]) == prompt


def test_qwen35_thinking_on_rejects_a_turn_without_reasoning() -> None:
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="without reasoning"):
        render_supervised(tokenizer, QWEN_35_08B, _dataset(_agent_example(reasoning="Only the first turn.")),
                          QWEN35_THINKING_RENDERER, max_length=4096)
