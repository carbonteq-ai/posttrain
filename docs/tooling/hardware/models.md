# Models

## Short answer: yes, Qwen3.5 has small models

Qwen3.5 “small” line (Mar 2026):

| Model | Params | Notes |
| --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` | 0.8B | Smallest 3.5 |
| `Qwen/Qwen3.5-2B` | **2B** | Best size match for this lab |
| `Qwen/Qwen3.5-4B` | 4B | Stretch on 8 GB with QLoRA |
| `Qwen/Qwen3.5-9B` | 9B | Too large for comfortable local RL here |

Also exists: larger 3.5 MoE / dense (27B, 35B-A3B, …) — out of scope for this box.

## Primary target

**`Qwen/Qwen3.5-2B`** (Instruct / chat-tuned variant when training from instructions; Base if SFT warm-up first).

### Tooling notes (not blockers)

Qwen3.5 is a **hybrid Gated DeltaNet + Gated Attention** early-fusion VLM. Plan for:

- Recent `transformers` + `vLLM` (often `>=0.17` for 3.5)
- Text-only RL first (skip vision tower until the loop is stable)
- If TRL+vLLM colocate misbehaves, fall back to Transformers / Unsloth-style generate (`fast_inference=False` pattern)

No Qwen3 ladder — go straight to 3.5-2B.

## Other popular newer small models (alternatives)

| Family | ~Size | Fit here | Comment |
| --- | --- | --- | --- |
| **Qwen3** | 0.6B / 1.7B / 4B | Fallback only | Dense; use if 3.5 tooling blocks |
| **Qwen3.5** | 0.8B / 2B / 4B | Best “new 2B” | Hybrid + multimodal; pin fresh stacks |
| **SmolLM3** | 3B | Possible | Fully open recipe; a bit larger than 2B |
| **Gemma 4 E2B** | ~2.3B effective | Possible | Multimodal; heavier than plain text 2B |
| **Llama 3.2** | 1B / 3B | Possible | Mature, older than Qwen3.5 |
| **Phi-4-mini** | ~3.8B | Tight | Strong quality; larger than ideal |
| **Nemotron 3 Nano** | 4B | Inference first | Hybrid Mamba; prefer not for day-1 GRPO |

**Default pick for this lab:** **`Qwen/Qwen3.5-2B`**.

## Memory expectations (8 GB RTX 3070 Ti)

These are order-of-magnitude, not guarantees:

| Setup | Likely |
| --- | --- |
| Qwen3.5-2B QLoRA GRPO, short ctx / completions | Feasible |
| Same + vLLM colocate + long rollouts | Tight / OOM risk |
| Full BF16 + long context serve | Not the plan on this card |

Always cap `max_model_len` / completion length; do not use 128k–262k defaults for local RL.
