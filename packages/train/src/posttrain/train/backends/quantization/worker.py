"""Standalone LLM Compressor worker executed in the isolated environment."""

from __future__ import annotations

import gc
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: worker.py REQUEST_JSON")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        import torch
        from llmcompressor import oneshot  # pyright: ignore[reportMissingImports]
        from llmcompressor.modifiers.quantization import QuantizationModifier  # pyright: ignore[reportMissingImports]
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("the isolated quantization environment requires llmcompressor and transformers") from error

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_config = AutoConfig.from_pretrained(
        request["model_id"],
        revision=request["model_revision"],
        trust_remote_code=False,
    )
    model_factory = AutoModelForImageTextToText if source_config.model_type == "qwen3_5" else AutoModelForCausalLM
    model = model_factory.from_pretrained(
        request["model_id"],
        revision=request["model_revision"],
        dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        request["model_id"],
        revision=request["model_revision"],
        trust_remote_code=False,
    )
    quantization = QuantizationModifier(
        ignore=list(dict.fromkeys(request["ignored_modules"])),
        scheme=request["scheme"],
        targets=["Linear"],
    )
    method = request["method"]
    dataset = None
    recipe: Any = quantization
    oneshot_options: dict[str, Any] = {}
    if method == "awq":
        from datasets import load_dataset
        from llmcompressor.modifiers.transform.awq import AWQModifier  # pyright: ignore[reportMissingImports]

        split = f"{request['split']}[:{request['sample_count']}]"
        dataset = load_dataset(
            request["dataset_id"],
            request["dataset_config"],
            revision=request["dataset_revision"],
            split=split,
        )
        text_column = request["text_column"]
        if text_column not in dataset.column_names:
            raise ValueError(f"calibration dataset has no {text_column!r} column")
        dataset = dataset.map(lambda row: {"text": str(row[text_column])}, remove_columns=dataset.column_names)
        recipe = [AWQModifier(), quantization]
        oneshot_options = {
            "num_calibration_samples": request["sample_count"],
            "max_seq_length": request["sequence_length"],
            "batch_size": request["batch_size"],
        }
    elif method != "rtn":
        raise ValueError(f"unsupported LLM Compressor quantization method: {method!r}")
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        output_dir=str(output_dir),
        **oneshot_options,
    )
    if not any(output_dir.iterdir()):
        model.save_pretrained(output_dir, save_compressed=True)
        tokenizer.save_pretrained(output_dir)
    serialized = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))["quantization_config"]
    groups = serialized.get("config_groups", {})
    if not groups:
        raise RuntimeError("serialized model has no quantization config groups")
    expected = {
        "bits": request["expected_bits"],
        "group_size": request["expected_group_size"],
        "zero_point": request["expected_zero_point"],
    }
    for name, group in groups.items():
        weights = group.get("weights") or {}
        observed = {
            "bits": weights.get("num_bits"),
            "group_size": weights.get("group_size"),
            "zero_point": not bool(weights.get("symmetric")),
        }
        if observed != expected:
            raise RuntimeError(f"serialized quantization group {name!r} is {observed}, expected {expected}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded: Any = model_factory.from_pretrained(
        output_dir,
        dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )
    device = next(reloaded.parameters()).device
    inputs = tokenizer("Reply with one token.", return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = reloaded.generate(**inputs, max_new_tokens=1, do_sample=False)
    if generated.shape[-1] != inputs.input_ids.shape[-1] + 1:
        raise RuntimeError("serialized quantized model failed the one-token reload validation")
    metrics: dict[str, Any] = {"runtime_seconds": time.perf_counter() - started}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
    runtime_versions = {
        "python": sys.version.split()[0],
        "torch": version("torch"),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "llmcompressor": version("llmcompressor"),
        "cuda": str(torch.version.cuda),
    }
    (output_dir / "posttrain-quantization-summary.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "runtime_versions": runtime_versions,
                "validation": {
                    "model_class": type(reloaded).__name__,
                    "model_type": reloaded.config.model_type,
                    "reload_generate_tokens": 1,
                    "serialized_quantization": expected,
                    "status": "passed",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
