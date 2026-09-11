"""Reload a completed toy's LoRA export against its pinned base and generate."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--adapter", type=Path, help="Explicit separately re-exported adapter; original export is preserved."
    )
    parser.add_argument(
        "--allow-zero-update",
        action="store_true",
        help="Verify an exactly reloadable zero-signal export without claiming that learning occurred.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    assert (root / "result.json").is_file()
    selection = json.loads((root / "selection.json").read_text())
    algorithm = selection["settings"].get("algorithm")
    if algorithm is None:
        algorithm = "gdpo" if "component_names" in selection["settings"] else "capo"
    base_dir = root / "training" / algorithm
    adapter = (
        base_dir / "trainer/model/lora_adapter"
        if selection["training"]["backend"].startswith("verl@")
        else base_dir / "adapter"
    )
    if args.adapter is not None:
        adapter = args.adapter.resolve()
    tensors = load_file(adapter / "adapter_model.safetensors")
    assert tensors and all(torch.isfinite(value).all() for value in tensors.values())
    b_norm = sum(float(value.float().square().sum()) for name, value in tensors.items() if "lora_B" in name) ** 0.5
    if not args.allow_zero_update:
        assert b_norm > 0
    artifact = selection["policy"]["artifact"]
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(artifact["repo_id"], revision=artifact["revision"], local_files_only=True)
    # veRL trains the native conditional-generation wrapper; TRL's text-only
    # loader uses a different prefix. Loading the wrong wrapper can warn about
    # missing adapters yet still generate from the unmodified base model.
    loader = (
        AutoModelForImageTextToText if selection["training"]["backend"].startswith("verl@") else AutoModelForCausalLM
    )
    base = loader.from_pretrained(
        artifact["repo_id"],
        revision=artifact["revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter, local_files_only=True).to(args.device).eval()
    restored = get_peft_model_state_dict(model)
    assert set(restored) == set(tensors), "export adapter keys did not match the reloaded model"
    assert all(torch.equal(restored[name].cpu(), saved) for name, saved in tensors.items()), (
        "adapter weights changed on load"
    )
    template_kwargs = (
        {"enable_thinking": False} if selection["policy"]["family"] == "qwen3.5" else {"preserve_thinking": False}
    )
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one word: ready."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **template_kwargs,
    ).to(args.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    completion = generated[0, inputs["input_ids"].shape[-1] :].tolist()
    assert completion
    report = {
        "adapter": str(adapter),
        "lora_B_norm": b_norm,
        "device": args.device,
        "completion_ids": completion,
        "text": tokenizer.decode(completion),
        "exact_adapter_reload": True,
        "nonzero_update": b_norm > 0,
        "scope": "exact-export-reload-generation; not training-resume equivalence",
    }
    (root / "export-reload.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
