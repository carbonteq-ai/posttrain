# Recipe target: local 8 GB QLoRA GRPO

Status: verified on the local RTX 3070 Ti 8 GB.

The reusable recipe will be a typed definition shipped with `packages/train`.
It will consume a model profile plus a qualified Verifiers environment package
reference. There is no local reward-callback compatibility layer.

The v2 smoke consumes Trackio SFT adapter `training-qwen3.5-2b-sft-adapter:v0`
and uses one native GSM8K task with two sampled completions. TRL owns generation
through colocated vLLM; the reusable Verifiers bridge owns scoring and native
trace construction. vLLM uses the immutable 4-bit base, dynamic LoRA sync,
level-1 sleep, a 64 MiB KV cache, a 640-token window, and sequence-level TIS
bounded to `[0.1, 3.0]`.

Canonical Trackio run `17b7f95710a14e359f7c4706f2925690` from clean
revision `8a2edce` proved:

- global step `1` and gradient norm `0.08447`;
- two correct, naturally terminated rollouts at 164 and 273 tokens;
- clipped ratio `0` and reward standard deviation `0.00888`;
- mean train/inference per-token log-probability difference `0.08950`;
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
- the TRL bridge declares which task semantics it supports;
- bounded rollouts fit the local RTX 3070 Ti and terminate without clipping;
- reward components and native traces are retained;
- the resulting adapter is logged with its exact parent artifact.

MTP remains a separately versioned profile for a larger GPU; its extra drafter
embedding does not fit beside both local training and rollout representations.
