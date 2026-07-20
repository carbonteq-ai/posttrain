# ADR 0007 — Pin a TRL 1.8 compatibility fork

**Status:** Accepted  
**Date:** 2026-07-20

## Context

The training package uses TRL 1.8.0, while the validated serving runtime uses
vLLM 0.25.1. Upstream TRL 1.8.0 declares vLLM 0.16.0 through 0.23.0 as its
supported range. Leaving `train[vllm]` on an unbounded `vllm>=0.25` dependency
would combine an unsupported trainer/runtime pair and make future lock refreshes
non-reproducible.

Upstream subsequently validated vLLM 0.24 and 0.25.1 without changing trainer,
generation, or weight-synchronization behavior. Those changes exist after the
1.8.0 release and the 0.25.1 change is still awaiting upstream merge.

## Decision

- Maintain `carbonteq-ai/trl` as an upstream fork.
- Keep `release/1.8-carbonteq` rooted at upstream tag `v1.8.0`, commit
  `95809b942eb5d11d0b06d749510d88be99230b73`.
- Apply upstream vLLM 0.24 support commit `c1fdca18f0cc56fb60726d879d73f0cbd344e91f`
  and vLLM 0.25 support commit `68d7cb1a4228f91d832c2dc7ced80674d2c46c56`.
- Pin the workspace to merged fork commit
  `a0b4bca78eeeb02abb050abfa04624f952d5f633` and pin vLLM to `0.25.1`.
- Keep the trainer-runtime dependency on `datasets>=4.6.1,<4.7`. TRL's runtime
  paths use APIs available in 4.6.1; 4.7-only `Json` dtype helpers belong to
  repository dataset-authoring scripts. This makes TRL 1.8 and Verifiers v1
  installable in the same application environment.
- Preserve TRL's package name, public imports, and version `1.8.0`.
- Treat the fork as a dependency compatibility boundary. Project-specific
  training behavior remains in `packages/train`, not in the fork.
- Expose vLLM's engine-level speculative configuration through GRPO's colocated
  generation adapter. Rollout profiles may use native MTP while retaining TRL's
  weight synchronization and sleep lifecycle.
- Keep vLLM's native composite model when that is how the Hub checkpoint is
  stored, omit unused modality towers through its native text-only mode, and
  expose a declarative weight-name prefix for compatible full-weight sync.
- For QLoRA policies, use vLLM's dynamic-LoRA path: retain the healthy
  quantized base representation and synchronize only the active PEFT adapter.
  Never pass packed `Linear4bit` parameter storage to the ordinary dense-weight
  update path. With sleep enabled, preserve the immutable base through vLLM
  level-1 CPU backup; level 2 discards it and vLLM cannot reload bitsandbytes
  checkpoints in place.

## Consequences

The serving and online-training paths can share one validated vLLM release
without moving the training API to TRL 1.9 development code. The workspace
must explicitly validate and raise the ceiling for every future vLLM upgrade.
The application can also compose Verifiers environments with TRL trainers;
neither reusable package imports the other.
An upstream TRL release containing the same support may replace the fork after
the training acceptance matrix passes unchanged.

## Validation evidence

On 2026-07-20, the pinned pair completed an RTX 3070 Ti colocated smoke using
`Qwen/Qwen2.5-0.5B-Instruct`: TRL created the vLLM engine, captured CUDA graphs,
synchronized trainer weights, generated a two-token completion, and returned
two aligned token log-probability records. Peak PyTorch allocation was
3,120.7 MiB. This proves the dependency and internal API contract; model-profile
GRPO acceptance remains a separate training milestone.

The datasets-bound change resolved `datasets 4.6.1`, TRL 1.8, and Verifiers v1
together, passed 26 focused upstream SFT/DPO/GRPO contract tests, and passed
upstream tiny-model SFT and DPO training tests. It was merged in
[`carbonteq-ai/trl#3`](https://github.com/carbonteq-ai/trl/pull/3).

The fork also avoids flattening non-contiguous sequence logits before chunked
entropy calculation. On large-vocabulary DPO models, the old reshape could
allocate a full logits copy before chunking. The regression fix and 33 entropy
tests were merged in [`carbonteq-ai/trl#4`](https://github.com/carbonteq-ai/trl/pull/4).

Colocated GRPO speculative configuration forwarding and its focused config and
engine tests were merged in
[`carbonteq-ai/trl#5`](https://github.com/carbonteq-ai/trl/pull/5).

Safe forwarding of non-conflicting colocated vLLM engine arguments was merged
in [`carbonteq-ai/trl#6`](https://github.com/carbonteq-ai/trl/pull/6).

Declarative vLLM weight namespaces were merged in
[`carbonteq-ai/trl#7`](https://github.com/carbonteq-ai/trl/pull/7). The Qwen3.5
full-weight compatibility probe used `language_model.`; the generic TRL adapter
applies the prefix once at its weight-update boundary.

Native PEFT adapter synchronization was merged in
[`carbonteq-ai/trl#8`](https://github.com/carbonteq-ai/trl/pull/8). Its
quantized-base sleep lifecycle was corrected in
[`carbonteq-ai/trl#9`](https://github.com/carbonteq-ai/trl/pull/9): level-1
sleep backed up 1.82 GiB of Qwen3.5 base weights on the local RTX 3070 Ti,
wake restored them without `reload_weights`, and the real SFT adapter produced
a coherent completion.

Aligned raw dataset rows and authoritative custom-rollout truncation were
merged in [`carbonteq-ai/trl#10`](https://github.com/carbonteq-ai/trl/pull/10).
These additive fields let native environment-driven rollouts preserve stable
task identity and termination semantics without encoding metadata into prompts.

## Maintenance

For each fork update, fetch `huggingface/trl`, rebase the maintenance branch on
the intended released TRL tag, record the upstream commits here, run upstream
vLLM client/server tests, and run the lab's import and GPU rollout smokes. Never
replace the immutable workspace pin with a branch or floating version range.

## Revision History

- 2026-07-20: Advanced the immutable pin to PR 10 for aligned custom-rollout
  inputs and authoritative truncation state.
- 2026-07-20: Accepted the TRL 1.8 compatibility fork and recorded vLLM 0.25.1,
  speculative decoding, engine arguments, weight namespaces, and native LoRA
  synchronization support.
