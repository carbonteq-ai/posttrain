# TRL

TRL is the execution library behind `packages/train`.

The rebuilt `train` package will expose reusable SFT, DPO, and RL operations.
TRL is an internal adapter selected by a typed TRL config, not the object other
projects must compose directly. PEFT/QLoRA, checkpoint behavior, public result
semantics, and instrumentation hooks belong to the `train` package boundary.
The lab's injected observation context maps those hooks to Trackio. Datasets,
rewards, and Verifiers environment implementations remain independently owned.

The workspace uses the `carbonteq-ai/trl` fork pinned to immutable commit
`d726190b2f0a399e5a13f69584617efd0e7fcf00`. The fork preserves TRL 1.8.0 and
adds the upstream-validated vLLM 0.24/0.25 dependency support plus regression
coverage. It also keeps the trainer runtime compatible with `datasets 4.6.1`
so the application can install Verifiers v1 and TRL together. It does not
contain project-specific trainers or environment logic.
See [ADR 0007](../../decisions/0007-trl-vllm-025-fork.md) for the provenance and
upgrade policy.

The fork's colocated vLLM path has been exercised on the local RTX 3070 Ti with
a 0.5B Qwen smoke through engine creation, CUDA graph capture, weight sync,
generation, and token-logprob extraction. That compatibility smoke does not
replace SFT, DPO, or GRPO acceptance for the two foundation profiles.

The code-defined `posttrain-lab` entrypoint now composes typed training requests
with job-owned data. Reusable trainers remain callable directly from Python.
