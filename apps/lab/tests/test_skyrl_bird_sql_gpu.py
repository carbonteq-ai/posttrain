from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

MODEL_ID = "google/gemma-4-12B-it"
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
TARGET_PATTERN = r".*[.]language_model[.].*[.](q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def test_transformers_maps_gemma4_unified_to_the_supported_conditional_generation_class() -> None:
    transformers = pytest.importorskip("transformers")
    mapping = transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert mapping["gemma4_unified"] == "Gemma4UnifiedForConditionalGeneration"


@pytest.mark.gpu
def test_pinned_gemma4_text_only_lora_update_and_reload(tmp_path: Path) -> None:
    if os.environ.get("POSTTRAIN_GEMMA4_GPU_TEST") != "1":
        pytest.skip("set POSTTRAIN_GEMMA4_GPU_TEST=1 on the 96 GB GPU release-gate pod")
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN is required to load the gated pinned Gemma 4 model")

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        token=os.environ["HF_TOKEN"],
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        token=os.environ["HF_TOKEN"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    assert type(model).__name__ == "Gemma4UnifiedForConditionalGeneration"
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    assert model.config.use_cache is False
    assert model.is_gradient_checkpointing
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}

    config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=TARGET_PATTERN,
    )
    model = peft.get_peft_model(model, config)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert sum(parameter.numel() for _, parameter in trainable) < sum(
        parameter.numel() for parameter in model.parameters()
    ) / 10
    projection = re.compile(TARGET_PATTERN)
    for name, _parameter in trainable:
        assert ".language_model." in name
        assert "vision" not in name.casefold()
        assert "audio" not in name.casefold()
        module_name = name.split(".lora_", 1)[0]
        assert projection.fullmatch(module_name)

    messages = [
        {"role": "system", "content": "Use the SQL protocol."},
        {"role": "user", "content": "Return one row."},
        {"role": "assistant", "content": "<think>answer</think><solution>SELECT 1</solution>"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    batch = tokenizer(rendered, return_tensors="pt").to("cuda")
    output = model(**batch, labels=batch["input_ids"])
    assert torch.isfinite(output.loss)
    output.loss.backward()
    optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=5e-5)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    del output, optimizer, model
    torch.cuda.empty_cache()

    reloaded_base = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        token=os.environ["HF_TOKEN"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    reloaded = peft.PeftModel.from_pretrained(reloaded_base, adapter)
    prompt = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    generation_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated = reloaded.generate(**generation_inputs, max_new_tokens=4, do_sample=False)
    assert generated.shape[-1] > generation_inputs["input_ids"].shape[-1]
