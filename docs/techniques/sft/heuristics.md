# SFT heuristics (local 8 GB)

- Prefer **QLoRA** (`load_in_4bit`) on RTX 3070 Ti; full FT is out of scope here.
- Start with `max_length` ≤ 1024; raise only if VRAM allows.
- `batch_size=1` + `grad_accum` 8–16 is the usual effective-batch lever.
- Text-only Qwen3.5-2B checkpoint saves vision-tower VRAM vs full multimodal.
- Smoke with `--max-steps 20 --max-samples 128` before long runs.
- If loss is NaN: lower LR, check bf16/4bit, shrink sequence length.
