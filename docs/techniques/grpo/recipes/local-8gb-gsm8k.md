# Recipe target: local 8 GB QLoRA GRPO

Status: verified on the local RTX 3070 Ti 8 GB.

The reusable recipe will be a typed definition shipped with `packages/train`.
It will consume a model profile plus a qualified Verifiers environment package
reference. There is no local reward-callback compatibility layer.

The v2 smoke consumes Trackio SFT adapter `training-qwen3.5-2b-sft-adapter:v0`
and uses one native GSM8K task with two sampled trajectories. Verifiers owns the
episode and asks a backend-neutral policy client for model turns; the private
TRL adapter serves those turns from its already-loaded colocated-vLLM policy.
vLLM uses the immutable 4-bit base, dynamic LoRA sync, level-1 sleep, a 64 MiB
KV cache, a 640-token window, and sequence-level TIS bounded to `[0.1, 3.0]`.

Canonical Trackio run `07984dfc3feb44e1b34dcd5b92e2d850` from clean
revision `e7babfc` proved:

- global step `1`, loss `-3.997e-06`, and gradient norm `0.05420`;
- two correct, naturally terminated native episodes at 251 and 232 sampled
  tokens;
- reward `1.05148`, reward standard deviation `0.00139`, and token clipping
  ratio `0`;
- mean train/inference per-token log-probability difference `0.06902`;
- the sequence-level importance ratio reached the configured `0.1` lower
  bound, which remains a profile-tuning signal rather than a bridge failure;
- native model identity, token IDs, train masks, rewards, and stop state in both
  queryable Trackio traces and the retained Verifiers JSONL artifact;
- input lineage to SFT adapter `v0`, plus GRPO adapter, step-1 recovery
  checkpoint, and training summary outputs;
- clean distributed shutdown with no retained GPU compute process.

The first-step scalar loss is exactly zero because group-relative advantages
are centered and the optimizer ratio starts at one. That does not mean the
update is empty: the nonzero gradient norm is the acceptance signal.

```bash
uv run --package posttrain-lab --extra gpu-posttrain \
  posttrain-lab gsm8k-qwen-grpo-smoke \
  --tracked --project posttrain-platform --adapter-version v0
```

Acceptance conditions for future changes:

- the environment package can run independently through Verifiers;
- the trainer adapter remains independent of task and environment semantics;
- Verifiers owns episode execution and receives model turns from the trainer's
  already-loaded policy;
- bounded rollouts fit the local RTX 3070 Ti and terminate without clipping;
- reward components and native traces are retained;
- the resulting adapter is logged with its exact parent artifact.

MTP remains a separately versioned profile for a larger GPU; its extra drafter
embedding does not fit beside both local training and rollout representations.
