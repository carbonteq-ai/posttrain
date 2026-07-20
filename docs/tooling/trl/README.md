# TRL

TRL is the execution library behind `packages/train`.

The rebuilt `train` package will expose reusable SFT, DPO, and RL operations.
TRL is an internal adapter selected by a typed TRL config, not the object other
projects must compose directly. PEFT/QLoRA, checkpoint behavior, public result
semantics, and instrumentation hooks belong to the `train` package boundary.
The lab's injected observation context maps those hooks to Trackio. Datasets,
rewards, and Verifiers environment implementations remain independently owned.

The workspace uses the `carbonteq-ai/trl` fork pinned to immutable commit
`b6976fde8391afc8cd638b476d30dddc2e365c01`. The fork preserves TRL 1.8.0 and
adds the upstream-validated vLLM 0.24/0.25 dependency support plus regression
coverage. It also keeps the trainer runtime compatible with `datasets 4.6.1`
so the application can install Verifiers v1 and TRL together. It does not
contain project-specific trainers or environment logic.
Its entropy metrics also preserve chunked-memory behavior for non-contiguous
sequence slices, which is required for DPO on large-vocabulary models.
The fork also exposes colocated vLLM's engine-level speculative configuration
through `GRPOConfig`. This lets a typed Qwen rollout profile enable native MTP
without bypassing TRL's weight synchronization, importance-sampling correction,
or sleep lifecycle. The generic change was merged in
[`carbonteq-ai/trl#5`](https://github.com/carbonteq-ai/trl/pull/5).
It also accepts non-conflicting colocated engine arguments while protecting
TRL-controlled weight-sync and lifecycle options. Text-only runs of multimodal
models use this to skip an irrelevant dummy vision profiling pass. This was
merged in [`carbonteq-ai/trl#6`](https://github.com/carbonteq-ai/trl/pull/6).
Composite vLLM implementations may retain a namespace around a text-only
training model. The fork therefore exposes an explicit weight-name prefix at
the synchronization boundary instead of placing model-name rewrites in a job.
Qwen3.5 uses `language_model.` while vLLM's `language_model_only` mode omits the
vision tower. The generic change was merged in
[`carbonteq-ai/trl#7`](https://github.com/carbonteq-ai/trl/pull/7).
DPO kernel choice is model-specific and recorded as `dpo_loss_kernel`. Liger's
fused DPO loss can reduce projection memory for moderate vocabularies, but its
current backward path creates a full FP32 LM-head gradient even when that head
is frozen. The Qwen3.5 profile therefore uses the Torch loss with expandable
CUDA allocator segments; LFM2.5 uses Liger. This is a measured profile choice,
not a universal backend default.
See [ADR 0007](../../decisions/0007-trl-vllm-025-fork.md) for the provenance and
upgrade policy.

The fork's colocated vLLM path has been exercised on the local RTX 3070 Ti with
a 0.5B Qwen smoke through engine creation, CUDA graph capture, weight sync,
generation, and token-logprob extraction. That compatibility smoke does not
replace SFT, DPO, or GRPO acceptance for the two foundation profiles.

The code-defined `posttrain-lab` entrypoint now composes typed training requests
with job-owned data. Reusable trainers remain callable directly from Python.
The generic `VerifiersGRPOBridge` scores completions and records native traces;
it does not initialize a model. Transformers and colocated-vLLM rollouts are
explicit training-profile choices rather than behavior hidden in job code.
