"""Qualify the Qwen3.5 Gated DeltaNet fast path (fla-core + causal-conv1d) inside a kind image.

Run inside a Posttrain TRL kind image with a GPU, the Hugging Face cache mounted read-only and the
working directory mounted at /work (see docs/plan/qwen35-fast-gdn-kernels.md for the docker command):

    python qwen35_gdn_kernels.py kernels                 # fla/causal-conv1d vs the Transformers torch reference
    python qwen35_gdn_kernels.py actor bfloat16 TAG      # LoRA actor forward+backward at 2K/3K/4K tokens
    python qwen35_gdn_kernels.py score bfloat16 TAG      # trainer log-probs for retained vLLM samples
    python qwen35_gdn_kernels.py gap                     # |logp_vLLM - logp_HF| per token, before vs after
    python qwen35_gdn_kernels.py train-step bfloat16     # no-PEFT fwd+bwd (veRL backend / transform images)

`actor` and `score` write `<command>_<TAG>_<dtype>.json`; `score` and `gap` read `samples_<dtype>.json`
(records with prompt_ids, completion_ids and the vLLM sampler_logps of every completion token).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DTYPES = ("bfloat16", "float16")


def _dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def kernels() -> bool:
    import torch
    import torch.nn.functional as F
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    print(
        "device",
        torch.cuda.get_device_name(),
        torch.cuda.get_device_capability(),
        "fast path",
        qm.is_fast_path_available,
    )
    ok = qm.is_fast_path_available
    generator = torch.Generator("cuda").manual_seed(0)
    batch, tokens, heads, dim = 1, 2048, 16, 128
    for name in ("bfloat16", "float16", "float32"):
        dt = _dtype(name)
        base = [torch.randn(batch, tokens, heads, dim, device="cuda", generator=generator).to(dt) for _ in range(3)]
        gate = -torch.rand(batch, tokens, heads, device="cuda", generator=generator) * 0.5
        beta = torch.rand(batch, tokens, heads, device="cuda", generator=generator).to(dt).sigmoid()
        probe = torch.randn(batch, tokens, heads, dim, device="cuda", generator=generator)
        results = []
        for fn in (chunk_gated_delta_rule, qm.torch_chunk_gated_delta_rule):
            leaves = [t.clone().requires_grad_(True) for t in (*base, gate, beta)]
            out, _ = fn(*leaves[:3], g=leaves[3], beta=leaves[4], use_qk_l2norm_in_kernel=True)
            # A sum against a fixed probe keeps gradients O(1); a mean over 4M outputs underflows fp16.
            (out.float() * probe).sum().backward()
            results.append([out.detach(), *[leaf.grad for leaf in leaves]])
        errors = [_rel(a, b) for a, b in zip(*results, strict=True)]
        finite = all(bool(torch.isfinite(t).all()) for t in results[0])
        print(f"gated delta rule {name}: out/dq/dk/dv/dg/dbeta rel err " + " ".join(f"{e:.1e}" for e in errors), finite)
        ok &= finite and max(errors) < 5e-2

        conv_x = torch.randn(2, 6144, 3072, device="cuda", generator=generator).to(dt)
        conv_w = (torch.randn(6144, 4, device="cuda", generator=generator) * 0.3).to(dt)
        conv_probe = torch.randn(2, 6144, 3072, device="cuda", generator=generator)
        conv = []
        for kind in ("cuda", "torch"):
            x, w = conv_x.clone().requires_grad_(True), conv_w.clone().requires_grad_(True)
            if kind == "cuda":
                y = causal_conv1d_fn(x=x, weight=w, bias=None, activation="silu")
            else:
                y = F.silu(F.conv1d(x, w.unsqueeze(1), padding=3, groups=6144)[..., : x.shape[-1]])
            (y.float() * conv_probe).sum().backward()
            conv.append([y.detach(), x.grad, w.grad])
        conv_errors = [_rel(a, b) for a, b in zip(*conv, strict=True)]
        state = conv_x[:, :, -5:-1].contiguous()
        update = causal_conv1d_update(conv_x[:, :, -1:].clone(), state.clone(), conv_w, None, "silu")
        reference = qm.torch_causal_conv1d_update(conv_x[:, :, -1:].clone(), state.clone(), conv_w, None, "silu")
        conv_errors.append(_rel(update, reference))
        print(f"causal conv1d {name}: out/dx/dw/update rel err " + " ".join(f"{e:.1e}" for e in conv_errors))
        ok &= max(conv_errors) < 2e-2 and all(bool(torch.isfinite(t).all()) for t in conv[0])

    norm = FusedRMSNormGated(128, eps=1e-6, activation="silu").cuda()
    reference_norm = qm.Qwen3_5RMSNormGated(128, eps=1e-6).cuda()
    x = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    z = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    norm_error = _rel(norm(x, z).detach(), reference_norm(x, z).detach())
    print(f"gated RMSNorm bf16 rel err {norm_error:.1e}")
    ok &= norm_error < 2e-2
    print("PASS" if ok else "FAIL")
    return ok


def _load_model(dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=_dtype(dtype_name), attn_implementation="sdpa"
    ).to(torch.device("cuda"))


def actor(dtype_name: str, tag: str) -> None:
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint

    print("fast path", qm.is_fast_path_available, "fla", qm.chunk_gated_delta_rule is not None, flush=True)
    model = _load_model(dtype_name)
    model.config.use_cache = False
    model.train()  # Transformers applies gradient checkpointing only in training mode.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    core = model.base_model.model
    backbone, lm_head = core.model, core.lm_head
    records = json.loads(Path("samples_bfloat16.json").read_text())["records"]
    stream = [token for record in records for token in record["prompt_ids"] + record["completion_ids"]]
    params = [p for p in model.parameters() if p.requires_grad]

    def token_logps(hidden, targets):
        return torch.log_softmax(lm_head(hidden).float(), -1).gather(-1, targets[:, None])[:, 0]

    def step(ids):
        hidden = backbone(input_ids=ids).last_hidden_state[0, :-1]
        targets = ids[0, 1:]
        chunks = [
            checkpoint(token_logps, hidden[s : s + 512], targets[s : s + 512], use_reentrant=False)
            for s in range(0, hidden.shape[0], 512)
        ]
        logps = torch.cat(chunks)
        (-(logps * 0.01).sum()).backward()  # policy-gradient-shaped weighted log-likelihood
        return logps.detach()

    result: dict[str, object] = {"tag": tag, "dtype": dtype_name, "fast_path": qm.is_fast_path_available}
    for tokens in (2048, 3072, 4096):
        ids = torch.tensor([stream[tokens : 2 * tokens]], device="cuda")
        for _ in range(2):  # Triton compilation and autotuning happen here.
            model.zero_grad(set_to_none=True)
            step(ids)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        seconds = []
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logps = step(ids)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
        grads = [p.grad for p in params if p.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        row = {
            "step_s_median": sorted(seconds)[2],
            "step_s": seconds,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "grads": len(grads),
            "grads_finite": finite,
            "mean_logp": float(logps.mean()),
        }
        result[str(tokens)] = row
        print(f"T={tokens} {dtype_name} step {row['step_s_median']:.3f}s peak {row['peak_gib']:.2f}GiB finite {finite}")
    Path(f"actor_{tag}_{dtype_name}.json").write_text(json.dumps(result, indent=1))


def score(dtype_name: str, tag: str) -> None:
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm

    model = _load_model(dtype_name).eval()
    lm_head = model.get_output_embeddings()
    scores: dict[str, list[dict[str, object]]] = {}
    for sampler in DTYPES:
        loaded = json.loads(Path(f"samples_{sampler}.json").read_text())
        temperature = loaded["summary"].get("temperature", 1.0)
        rows = []
        for record in loaded["records"]:
            prompt = len(record["prompt_ids"])
            ids = torch.tensor([record["prompt_ids"] + record["completion_ids"]], device="cuda")
            completion = ids[0, prompt:]
            with torch.no_grad():
                hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, prompt - 1 : -1]
                logps = torch.cat(
                    [
                        torch.log_softmax((lm_head(hidden[s : s + 128]) / temperature).float(), -1).gather(
                            -1, completion[s : s + 128, None]
                        )[:, 0]
                        for s in range(0, hidden.shape[0], 128)
                    ]
                )
            rows.append({"id": record["id"], "set": record["set"], "fp32_logps": logps.tolist()})
        scores[sampler] = rows
    document = {"dtype": dtype_name, "tag": tag, "fast_path": qm.is_fast_path_available, "scores": scores}
    Path(f"score_{tag}_{dtype_name}.json").write_text(json.dumps(document))


def gap(tags: tuple[str, ...] = ("before", "after")) -> None:
    import numpy as np

    samples = {
        name: {r["id"]: r for r in json.loads(Path(f"samples_{name}.json").read_text())["records"]} for name in DTYPES
    }
    for sampler in DTYPES:
        for trainer in DTYPES:
            for tag in tags:
                path = Path(f"score_{tag}_{trainer}.json")
                if not path.is_file():
                    continue
                document = json.loads(path.read_text())
                deltas = np.concatenate(
                    [
                        np.abs(np.array(samples[sampler][row["id"]]["sampler_logps"]) - np.array(row["fp32_logps"]))
                        for row in document["scores"][sampler]
                    ]
                )
                print(
                    f"vLLM {sampler} vs HF {trainer} {tag}: mean {deltas.mean():.4f} "
                    f"p99 {np.quantile(deltas, 0.99):.3f} max {deltas.max():.2f} tokens {deltas.size}"
                )


def train_step(dtype_name: str) -> bool:
    """Decoder-parameter forward+backward that needs only torch and Transformers (fits an 8 GB GPU)."""

    import io
    import logging

    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    from transformers.utils import logging as hf_logging

    captured = io.StringIO()
    hf_logging.get_logger("transformers").addHandler(logging.StreamHandler(captured))
    model = _load_model(dtype_name)
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.get_input_embeddings().weight.requires_grad_(False)  # tied 248K-vocab embedding and head
    model.enable_input_require_grads()
    warned = "fast path is not available" in captured.getvalue()
    records = json.loads(Path("samples_bfloat16.json").read_text())["records"]
    stream = [token for record in records for token in record["prompt_ids"] + record["completion_ids"]]
    ids = torch.tensor([stream[2048:4096]], device="cuda")
    seconds = []
    for _ in range(4):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        hidden = model.model(input_ids=ids).last_hidden_state[0, -257:-1]
        loss = torch.nn.functional.cross_entropy(model.lm_head(hidden).float(), ids[0, -256:])
        loss.backward()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    print(
        f"fast path {qm.is_fast_path_available} warning {warned} T=2048 {dtype_name} grads {len(grads)} "
        f"finite {finite} step {sorted(seconds[1:])[1]:.3f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
    )
    ok = qm.is_fast_path_available and not warned and finite
    print("PASS" if ok else "FAIL")
    return ok


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    if command == "kernels":
        return 0 if kernels() else 1
    if command in {"actor", "score"} and len(argv) == 4 and argv[2] in DTYPES:
        (actor if command == "actor" else score)(argv[2], argv[3])
        return 0
    if command == "train-step" and len(argv) == 3 and argv[2] in DTYPES:
        return 0 if train_step(argv[2]) else 1
    if command == "gap":
        gap()
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
