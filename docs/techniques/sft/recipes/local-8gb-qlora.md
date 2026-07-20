# Recipe target: local 8 GB QLoRA SFT

Status: Qwen3.5-2B path verified; LFM2.5 qualification remains pending.

The reusable recipe will be a typed definition shipped with `packages/train`
and will define memory-safe QLoRA defaults for the local RTX 3070 Ti. A job
supplies its model profile, dataset reference, explicit overrides, and promotion
decision.

The Qwen smoke uses NF4 QLoRA, the model-owned renderer, two GSM8K examples,
two optimizer steps, and explicit checkpoint/adapter publication. Canonical
Trackio run `b549afa7241942bfa6ed31cc4fdacffd` completed from clean revision
`77dace9` with final loss `0.9181`, gradient norm `4.6875`, adapter `v0`, a
step-2 recovery checkpoint, and a training-summary artifact.

```bash
uv run --package posttrain-lab --extra gpu-posttrain \
  posttrain-lab gsm8k-qwen-sft-smoke \
  --tracked --project posttrain-platform
```

Remaining validation before treating the recipe as cross-model:

- the LFM2.5-1.2B-Thinking profile completes the same bounded smoke;
- model-specific LoRA targets are explicit;
- resolved profile/config snapshots are retained;
- the selected adapter is logged as a descendant artifact;
- recovery checkpoints do not automatically become derived profiles.

Do not infer LFM compatibility from the Qwen run; the renderer and adapter
targets are model-family concerns even though the trainer operation is shared.
