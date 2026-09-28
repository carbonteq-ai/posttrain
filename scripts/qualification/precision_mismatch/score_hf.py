"""Trainer-side scoring with the Transformers model, loaded as the TRL trainer loads it.

usage: score_hf.py DTYPE [--only SET[,SET]]   score samples_*.json; hook every module
       score_hf.py float32 --generate          sample the prompt set with Transformers
                                                 generate in fp32 (vLLM cannot run
                                                 Qwen3.5 Gated-DeltaNet in fp32)

Scoring writes /work/scores_<DTYPE>[_<sets>].json with, per sampled token, the
trainer log-probability three ways (see analyze.py) and, per hooked module
output, the largest finite |value|, its token position, and inf/NaN counts with
the first non-finite position.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

MODEL, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
DTYPE = sys.argv[1]
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
CHUNK = 128


def _generate(model: Any) -> None:
    prompts = json.loads(Path("/work/prompts.json").read_text())
    records, started, generated = [], time.perf_counter(), 0
    for index, prompt in enumerate(prompts):
        torch.manual_seed(1000 + index)
        ids = torch.tensor([prompt["prompt_ids"]], device="cuda")
        with torch.no_grad():
            output = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=prompt["max_tokens"],
                output_logits=True,
                return_dict_in_generate=True,
            )
        completion = output.sequences[0, ids.shape[1] :].tolist()
        logps = [
            float(torch.log_softmax(step[0].float(), -1)[token])
            for step, token in zip(output.logits, completion, strict=True)
        ]
        generated += len(completion)
        records.append(
            {
                "id": prompt["id"],
                "set": prompt["set"],
                "prompt_ids": prompt["prompt_ids"],
                "completion_ids": completion,
                "sampler_logps": logps,
                "finish_reason": "hf",
            }
        )
    summary = {
        "dtype": "float32",
        "sampler": "hf-generate",
        "seconds": time.perf_counter() - started,
        "generated_tokens": generated,
    }
    print(json.dumps(summary))
    Path("/work/samples_float32.json").write_text(json.dumps({"summary": summary, "records": records}))


def _install_hooks(model: Any, current: dict[str, str | None]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"max": 0.0, "max_pos": -1, "inf": 0, "nan": 0, "first_bad_pos": None}
    )

    def hook(name: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            outputs = output if isinstance(output, tuple | list) else (output,)
            for index, tensor in enumerate(outputs):
                if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.dim() < 2:
                    continue
                if tensor.numel() == 0:
                    continue
                entry = stats[f"{name}#{index}" if index else name]
                bad = ~torch.isfinite(tensor)
                if bool(bad.any()):
                    entry["nan"] += int(torch.isnan(tensor).sum())
                    entry["inf"] += int(torch.isinf(tensor).sum())
                    if entry["first_bad_pos"] is None:
                        entry["first_bad_pos"] = {
                            "sample": current["sample"],
                            "index": bad.nonzero()[0].tolist(),
                            "shape": list(tensor.shape),
                        }
                magnitude = tensor.detach().float().abs().masked_fill(bad, 0)
                largest = float(magnitude.max())
                if largest > entry["max"]:
                    flat = int(magnitude.argmax())
                    inner = int(torch.tensor(tensor.shape[2:]).prod()) if tensor.dim() >= 3 else 1
                    entry["max"] = largest
                    entry["max_pos"] = (flat // inner) % tensor.shape[1] if tensor.dim() >= 3 else flat // inner
                    entry["max_sample"] = current["sample"]

        return record

    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(hook(name))
    return stats


def _score(model: Any) -> None:
    head = model.get_output_embeddings()
    head_weight32 = head.weight.detach().float()
    current: dict[str, str | None] = {"sample": None}
    stats = _install_hooks(model, current)
    results = {}
    for path in sorted(glob.glob("/work/samples_*.json")):
        sampler = Path(path).stem.removeprefix("samples_")
        if ONLY and sampler not in ONLY:
            continue
        loaded = json.loads(Path(path).read_text())
        temperature = loaded["summary"].get("temperature", 1.0)
        scored = []
        for record in loaded["records"]:
            current["sample"] = f"{sampler}:{record['id']}"
            start = len(record["prompt_ids"])
            ids = torch.tensor([record["prompt_ids"] + record["completion_ids"]], device="cuda")
            completion = ids[0, start:]
            native, upcast, head32 = [], [], []
            with torch.no_grad():
                hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, start - 1 : -1]
                for offset in range(0, hidden.shape[0], CHUNK):
                    target = completion[offset : offset + CHUNK, None]
                    logits = head(hidden[offset : offset + CHUNK]) / temperature  # model dtype, as the trainer
                    native.append(torch.log_softmax(logits, -1).gather(-1, target)[:, 0].float())
                    upcast.append(torch.log_softmax(logits.float(), -1).gather(-1, target)[:, 0])
                    full = torch.nn.functional.linear(hidden[offset : offset + CHUNK].float(), head_weight32)
                    head32.append(torch.log_softmax(full / temperature, -1).gather(-1, target)[:, 0])
            scored.append(
                {
                    "id": record["id"],
                    "set": record["set"],
                    "native_logps": torch.cat(native).tolist(),
                    "fp32_logps": torch.cat(upcast).tolist(),
                    "head32_logps": torch.cat(head32).tolist(),
                }
            )
        results[sampler] = scored
        print("scored", sampler, len(scored), flush=True)
    suffix = "_" + "_".join(ONLY) if ONLY else ""
    Path(f"/work/scores_{DTYPE}{suffix}.json").write_text(
        json.dumps(
            {
                "dtype": DTYPE,
                "scores": results,
                "hooks": {name: dict(values) for name, values in stats.items()},
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
    )


def main() -> None:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[DTYPE]
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype, attn_implementation="sdpa")
    model = model.cuda().eval()
    print(type(model).__name__, next(model.parameters()).dtype, flush=True)
    if "--generate" in sys.argv:
        _generate(model)
    else:
        _score(model)


if __name__ == "__main__":
    main()
