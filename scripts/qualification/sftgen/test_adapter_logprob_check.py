"""CPU unit tests for adapter_logprob_check on a tiny random causal LM (wt-sft .venv: torch, transformers, peft).

  CAPPED_MEM=4G .../bin/capped .venv/bin/python -m pytest scripts/qualification/sftgen/test_adapter_logprob_check.py -q
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import adapter_logprob_check as alc  # noqa: E402

TOKENIZER = alc.Path(os.environ.get("SFTGEN_TOKENIZER", "missing-tokenizer"))  # local folder; skips without it


def tiny_base(path: Path, vocab: int = 512):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)
    return path


def save_adapter(base: Path, out: Path, *, rank: int = 16, alpha: int = 32, train: bool = True,
                 rename: tuple[str, str] | None = None) -> Path:
    from peft import get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base), alc.runner_lora_config(rank, alpha))
    if train:
        torch.manual_seed(1)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.data.normal_(0, 0.05)
    model.save_pretrained(out)
    if rename:
        file = out / "adapter_model.safetensors"
        tensors = load_file(file)
        save_file({k.replace(*rename): v for k, v in tensors.items()}, file)
    return out


def run_check(base: Path, adapter: Path, rank: int = 16) -> dict:
    model, report = alc.attach_adapter(alc.load_base(str(base), "causal-lm", torch.float32, "cpu"), str(adapter),
                                       rank, 32)
    ids = list(range(5, 45))
    return alc.check(model, ids, report, "cpu", 0.0)


def test_trained_adapter_changes_logprobs(tmp_path):
    base = tiny_base(tmp_path / "base")
    result = run_check(base, save_adapter(base, tmp_path / "adapter"))
    assert result["passed"], result
    assert result["effect"]["max_abs_delta"] > 0 and result["load"]["unexpected_count"] == 0
    assert result["load"]["adapter_tensors"] == result["load"]["model_lora_tensors"] > 0


def test_untrained_adapter_is_identical_and_fails(tmp_path):
    base = tiny_base(tmp_path / "base")
    result = run_check(base, save_adapter(base, tmp_path / "adapter", train=False))
    assert not result["passed"]
    assert result["effect"]["identical"] and result["load"]["zero_lora_B_count"] > 0


def test_mismatched_key_names_are_silently_ignored_by_the_runner_path_and_fail_here(tmp_path):
    base = tiny_base(tmp_path / "base")
    adapter = save_adapter(base, tmp_path / "adapter", rename=("model.layers", "model.language_model.layers"))
    result = run_check(base, adapter)
    assert not result["passed"]
    assert result["effect"]["identical"]  # what the runner would have evaluated: the base model
    assert result["load"]["unexpected_count"] > 0 and result["load"]["missing_count"] > 0


def test_cli_with_tokenizer_writes_result(tmp_path):
    if not TOKENIZER.exists():
        pytest.skip("Qwen3.5-4B tokenizer files not present")
    from transformers import AutoTokenizer

    vocab = len(AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True))
    base = tiny_base(tmp_path / "base", vocab=vocab)
    adapter = save_adapter(base, tmp_path / "adapter")
    out = tmp_path / "result.json"
    code = alc.main(["--model", str(base), "--adapter", str(adapter), "--tokenizer", str(TOKENIZER),
                     "--model-class", "causal-lm", "--device", "cpu", "--dtype", "float32", "--out", str(out)])
    result = json.loads(out.read_text())
    assert code == 0 and result["passed"] and result["effect"]["tokens"] > 20


def test_nonfinite_adapter_fails(tmp_path):
    from safetensors.torch import load_file, save_file

    base = tiny_base(tmp_path / "base")
    adapter = save_adapter(base, tmp_path / "adapter")
    file = adapter / "adapter_model.safetensors"
    tensors = load_file(file)
    name = next(k for k in tensors if "lora_A" in k)
    tensors[name][0, 0] = float("nan")
    save_file(tensors, file)
    result = run_check(base, adapter)
    assert not result["passed"] and result["load"]["nonfinite"] == [name]
