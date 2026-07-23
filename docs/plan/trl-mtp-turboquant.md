# Add native MTP and TurboQuant rollout controls to the TRL backend

This ExecPlan is a living document maintained according to `docs/templates/PLAN.md`. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must stay current while implementation proceeds.

## Purpose / Big Picture

After this change, a framework `InferenceBinding` can select native Qwen MTP speculative decoding or a vLLM KV-cache dtype such as TurboQuant K8V4 for TRL-backed GRPO and on-policy distillation without embedding vLLM settings in algorithm configuration. Colocated TRL rollouts will also expose normalized per-generation MTP acceptance metrics. MTP remains rollout acceleration rather than an auxiliary training loss, and TurboQuant remains experimental until its Qwen 3.5 long-context correctness gate passes.

## Progress

- [x] (2026-07-22 16:03Z) Inspected the framework TRL adapters, immutable dependency pin, TRL fork generation abstraction, GRPO configuration, and experimental distillation trainer.
- [x] (2026-07-22 16:03Z) Confirmed GRPO already forwards `speculative_config`, while `kv_cache_dtype`, distillation forwarding, and normalized MTP telemetry are absent.
- [x] (2026-07-22) Extended the TRL fork's shared colocated generation runtime with step-local speculative-decoding metrics and exposed engine/speculative options to distillation.
- [x] (2026-07-22) Extended framework translation and validation for GRPO and on-policy distillation while preserving backend-neutral inference selections.
- [x] (2026-07-22) Added regression tests and updated the TRL consumer documentation plus fork ledger.
- [x] (2026-07-22) Qualified two-step Qwen 3.5 0.8B native-MTP GRPO on the 8 GB GPU with four complete AutomationBench trajectories, post-update synchronization, two backward cycles, checkpoints, and adapter export.
- [x] (2026-07-22) Requalified normalized step-total MTP observability: 237 drafted tokens, 204 accepted tokens, and 86.08% weighted acceptance in a complete rollout/backward run.
- [ ] Publish the TRL fork commit, update the immutable consumer pin, and run a real on-policy distillation qualification; these require repository publication and a separately viable teacher-scoring topology.

## Surprises & Discoveries

- Observation: CarbonTeq TRL already has the generic GRPO MTP argument path.
  Evidence: `GRPOConfig.vllm_speculative_config` reaches `VLLMGeneration(speculative_config=...)`, and the framework's GRPO adapter already selects it from `InferenceBinding.engine.speculative_config`.
- Observation: the active framework pin is newer than the prose in `docs/tooling/trl/README.md`.
  Evidence: `packages/train/pyproject.toml` and `uv.lock` pin commit `9c01e51243aed3c88d1f726bd6dab59843b62a9b`, while the consumer page still names `b30d820a160ee39a2294a2755fd2d96fe3ac57b0`.
- Observation: TurboQuant cannot yet be described as Qwen 3.5 quality-qualified on this machine.
  Evidence: the existing matched 32K probe gave about 2.67 times more KV capacity but K8V4 missed the beginning-of-context recall target at 8K, 16K, 24K, and 32.7K where normal KV passed.
- Observation: a 32K Qwen 3.5 0.8B MTP engine needs more than a token constructor smoke implies.
  Evidence: vLLM rejected a 256 MiB explicit cache because one full-length request requires 0.49 GiB; 640 MiB passed engine initialization.
- Observation: vLLM sleep does not discard CUDA graph private pools on this topology.
  Evidence: graph-enabled rollout left about 1.05 GiB in private pools and the actor backward OOMed. Eager rollout removed graph capture, but physical batch two still exceeded memory.
- Observation: normalized effective batch semantics are required to make the real 8 GB schedule fit.
  Evidence: physical microbatch one with gradient accumulation two preserved the logical two-generation GRPO group and completed backward with gradient norm 0.1378.
- Observation: TRL's default metric buffer averages multiple agent-turn counter samples.
  Evidence: the first bridge showed mean counts; the corrected accumulator reports step totals and weighted rates. Run `-10` recorded 237 drafted and 204 accepted tokens at 86.08%.

## Decision Log

- Decision: Treat native MTP and TurboQuant as independent rollout-engine capabilities.
  Rationale: MTP changes token proposal and verification; TurboQuant changes KV-cache storage. Neither changes the GRPO or distillation optimizer objective.
  Date/Author: 2026-07-22 / Codex
- Decision: Support only colocated MTP through trainer configuration in this slice.
  Rationale: TRL constructs and owns the colocated vLLM engine. Server mode must be configured when launching the external server and cannot truthfully consume these in-process constructor arguments.
  Date/Author: 2026-07-22 / Codex
- Decision: Forward generic `kv_cache_dtype` but label `turboquant_k8v4` experimental for Qwen 3.5.
  Rationale: engine wiring and memory benefit are established, while long-context semantic correctness is not.
  Date/Author: 2026-07-22 / Codex
- Decision: Normalize MTP observability at `rollout/spec_*` and compute per-generation deltas from vLLM lifetime counters.
  Rationale: the names then match veRL evidence, and deltas prevent cumulative counters from being misreported as one training step.
  Date/Author: 2026-07-22 / Codex
- Decision: Define algorithm batch validation using physical microbatch times gradient accumulation.
  Rationale: this is the backend-neutral effective batch, preserves one GRPO group, and lets constrained backends schedule identical logical work without pretending the physical microbatch is the algorithm batch.
  Date/Author: 2026-07-22 / Codex
- Decision: Use eager vLLM rollout execution in the qualified 8 GiB TRL profile.
  Rationale: CUDA graph private pools survive rollout sleep and compete with actor backward; eager mode trades some rollout throughput for the required phase-shared memory release.
  Date/Author: 2026-07-22 / Codex

## Outcomes & Retrospective

TRL GRPO now has a qualified native-MTP path for Qwen 3.5 0.8B at a 32K engine window. Two optimizer steps completed over four original AutomationBench trajectories, including post-update adapter synchronization and MTP rollout. The generic fork and framework translations also cover on-policy distillation, while its real teacher-scored qualification remains open. TurboQuant K8V4 is wired and guarded but remains experimental because long-context Qwen correctness failed. The fork must be committed and pushed before the framework pin can move from `9c01e51243aed3c88d1f726bd6dab59843b62a9b`.

## Context and Orientation

The framework repository is `/home/hammad/projects/rl`; its private TRL translations live under `packages/train/src/posttrain/train/backends/trl`. The maintained trainer fork is `/home/hammad/projects/trl`, currently at commit `9c01e51243aed3c88d1f726bd6dab59843b62a9b`. `trl/generation/vllm_generation.py` owns colocated engine construction, weight synchronization, sleep/wake, and generation. `trl/trainer/grpo_trainer.py` and `trl/experimental/distillation/distillation_trainer.py` consume that shared runtime.

Native multi-token prediction, abbreviated MTP, uses a model's bundled draft head to propose tokens which the base model verifies. TurboQuant K8V4 stores attention keys in FP8 and values in four-bit form. In this plan both operate only during rollout generation. They do not quantize trainable actor weights and do not add an MTP loss.

The frozen product baseline already places speculative decoding and KV-cache settings on `InferenceBinding.engine`, so this work does not change product meaning and requires no baseline amendment.

## Plan of Work

In the TRL fork, add a defensive metric snapshot helper to `VLLMGeneration`. It should read available vLLM speculative counters immediately after colocated generation and before sleep, subtract the prior snapshot, and expose normalized values for the trainer that initiated the call. Add `vllm_speculative_config` and `vllm_engine_kwargs` to the experimental distillation config and pass them to the shared generation runtime. GRPO and distillation should append the returned normalized values to their existing trainer metric buffers.

In the framework repository, validate that in-process speculative configuration is a colocated, native-MTP mapping with a positive token count. Forward `InferenceBinding.engine.kv_cache_dtype` as a vLLM constructor option for GRPO and distillation, along with distillation's speculative configuration. Keep normalized shared settings on `InferenceBinding`; use TRL-native fields only at the private adapter boundary.

Update `/home/hammad/projects/trl/CARBONTEQ_FORK.md` with the maintained delta and regression commands, creating the ledger because the fork currently lacks it. Update `docs/tooling/trl/README.md` with scope, configuration, evidence, and release gates. Do not change the immutable framework pin until the fork work is committed.

## Concrete Steps

From `/home/hammad/projects/trl`, run focused unit tests for `tests/test_vllm_generation.py`, `tests/test_grpo_trainer.py`, and `tests/experimental/test_distillation_trainer.py`, followed by format and static checks appropriate to touched files. From `/home/hammad/projects/rl`, run `uv run pytest packages/train/tests/test_api.py`, `uv run ruff check` on touched paths, `uv run pyright`, `uv run lint-imports`, and `git diff --check`.

A real qualification uses Qwen 3.5 0.8B, colocated sleep mode, one native MTP draft token, at least one complete environment rollout and backward pass, and a 32,768-token engine window. TurboQuant must be probed separately against normal KV using identical prompts and deterministic generation; it is not a release gate for MTP.

## Validation and Acceptance

The translation tests must prove that identical logical inference selections reach both TRL GRPO and distillation as `vllm_speculative_config` and `vllm_engine_kwargs`, and that unsupported speculative methods fail before trainer construction. TRL tests must prove constructor forwarding and verify exact per-generation counter deltas over two snapshots. A real MTP training run is accepted only when it completes rollout, loss, non-zero backward gradients, optimizer update, weight synchronization, and a post-update rollout while recording non-zero `rollout/spec_num_draft_tokens`, `rollout/spec_num_accepted_tokens`, and a bounded acceptance rate.

TurboQuant is configuration-supported when `kv_cache_dtype=turboquant_k8v4` reaches vLLM without bypassing TRL lifecycle control. It becomes quality-supported for Qwen 3.5 only after matched normal-KV and K8V4 tests pass at the required 32K context.

## Idempotence and Recovery

Unit tests and probes are repeatable. Preserve existing dirty files in the framework repository and restrict edits to named TRL adapter, test, plan, and tooling files. If a GPU run fails, retain its log and classify the failure before lowering capacity; do not convert the required full rollout/backward gate into a one-step constructor smoke. Do not update the framework's immutable TRL pin until the fork commit exists.

## Artifacts and Notes

The prior veRL qualification is comparison evidence for normalized metric names and Qwen MTP behavior, not proof that TRL's different lifecycle works. Existing TurboQuant artifacts establish memory allocation and the outstanding recall regression.

## Interfaces and Dependencies

`VLLMGeneration.last_generation_metrics` will be a mapping from normalized metric names to floats, empty when vLLM does not expose speculative counters. `GRPOConfig.vllm_speculative_config` remains the native constructor mapping. `DistillationConfig` gains the same field plus `vllm_engine_kwargs`. Framework adapters consume only `InferenceBinding.engine` and translate it privately; public request types do not gain backend-specific fields.

Revision note (2026-07-22): Created this plan after confirming the existing GRPO MTP path and identifying the missing TurboQuant, distillation, and observability surfaces.

Revision note (2026-07-22): Implemented fork and framework support, corrected effective-batch semantics, qualified two-step native-MTP AutomationBench GRPO on the 8 GiB device, and corrected speculative counters from per-turn means to step totals. Kept TurboQuant quality and real distillation as explicit release gates.
