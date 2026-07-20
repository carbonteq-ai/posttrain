# ADR 0007 — Pin a TRL 1.8 fork with vLLM 0.25 support

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
  `935060f640f5195fe62f1acc300c16db327a32b9` and pin vLLM to `0.25.1`.
- Preserve TRL's package name, public imports, and version `1.8.0`.
- Treat the fork as a dependency compatibility boundary. Project-specific
  training behavior remains in `packages/train`, not in the fork.

## Consequences

The serving and online-training paths can share one validated vLLM release
without moving the training API to TRL 1.9 development code. The workspace
must explicitly validate and raise the ceiling for every future vLLM upgrade.
An upstream TRL release containing the same support may replace the fork after
the training acceptance matrix passes unchanged.

## Validation evidence

On 2026-07-20, the pinned pair completed an RTX 3070 Ti colocated smoke using
`Qwen/Qwen2.5-0.5B-Instruct`: TRL created the vLLM engine, captured CUDA graphs,
synchronized trainer weights, generated a two-token completion, and returned
two aligned token log-probability records. Peak PyTorch allocation was
3,120.7 MiB. This proves the dependency and internal API contract; model-profile
GRPO acceptance remains a separate training milestone.

## Maintenance

For each fork update, fetch `huggingface/trl`, rebase the maintenance branch on
the intended released TRL tag, record the upstream commits here, run upstream
vLLM client/server tests, and run the lab's import and GPU rollout smokes. Never
replace the immutable workspace pin with a branch or floating version range.
