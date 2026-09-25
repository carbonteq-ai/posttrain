# Configuration rules, settings calculator and Observatory configuration review

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md` and extends the advice intent of `docs/plan/model-settings-defaults-and-validation.md` ("warn when ... forced eager mode despite a qualified graph path"), which remains the canonical design for resolution and validation.

## Purpose / Big Picture

The agentic optimization work (`docs/plan/agentic-workload-inference-optimization.md`) measured what individual settings cost: eager execution made LFM2.5 rollouts 2.32x slower, draft-free speculative decoding made them 0.84x as fast, CPU KV offload can leak KV across policy versions, a too-short context failed 4% of episodes, and a LoRA learning rate 4x below the reference recipes left held-out results unchanged after 20 updates. Most bindings still used the slow settings because nothing flagged them.

After this plan:

- `posttrain work-package validate`, `job plan` and `job run` reject a binding that disables a measured optimization unless the binding acknowledges the finding with a written reason, and warn about LoRA RL settings far from the reference recipes.
- `posttrain settings suggest` calculates engine settings for a model, target and token budget (with reasons, a memory budget and step-sizing options), and with `--work-package` reviews a package: every rule finding, every calculator finding, and each binding's diff against the recommendation.
- Observatory's Run config page shows the same review for every run, including runs recorded before the rules existed: findings beside the values they are about, a findings summary with counts, a step-sizing table (prompts, oversampled groups, rows, waves, estimated step time calibrated from the run's measured times), and a Recommended settings panel. The run Overview shows a one-line configuration summary.

This plan does not change the frozen product baseline: bindings stay versioned selections, the rules are validation findings inside the existing report, acknowledgements are an optional binding field, and Observatory stays read-only (it computes an analysis over recorded selections, as it does for metrics).

## Progress

- [x] (2026-09-23 15:00Z) Performance rules with error codes `VLLM_EAGER_DISABLES_CUDA_GRAPHS`, `VLLM_PREFIX_CACHING_DISABLED`, `SPECULATIVE_DRAFT_FREE_WITH_SAMPLING`, `TRL_ROLLOUT_ENGINE_KEYS_IGNORED`, `ROLLOUT_CONTEXT_BELOW_EPISODE_BUDGET`, `VLLM_KV_OFFLOAD_IN_ROLLOUT`, `VLLM_FLOAT16_ON_BF16_CHECKPOINT`; `InferenceBinding.performance_acknowledgements` turns a finding into `info` with its reason.
- [x] (2026-09-23 15:00Z) `work-package validate` runs the configuration rules and prints every finding; `job plan` prints findings without `--explain`; both accept `--strict`.
- [x] (2026-09-23 15:10Z) Audit of 77 lab work packages: 38 bindings fail (eager in 36, TRL-ignored keys in 12, float16 on bf16 in 2), including the `posttrain init --template grpo` starter and the release-gate Qwen3.5-2B eval.
- [x] (2026-09-23 17:00Z) Training rules: `LORA_RL_LEARNING_RATE_BELOW_REFERENCE`, `LORA_RL_LEARNING_RATE_ABOVE_REFERENCE`, `LR_SCHEDULE_DECAYS_WITHIN_SHORT_RUN` (warnings), `LORA_PARTIAL_TARGET_MODULES` (recommendation). The settings snapshot now records `lr_scheduler_type`.
- [x] (2026-09-23 18:00Z) Settings calculator (`posttrain.advisor.calculator`) replacing the planned per-binding tuning sweep: architecture from `config.json` (hybrid, sliding-window, Qwen3-Next linear attention, Gemma 4 global heads, MoE active parameters), memory budget, engine settings with reasons, decode bandwidth bound, and step sizing (below).
- [x] (2026-09-23 20:00Z) New workspace package `packages/advisor` (`posttrain.advisor`, depends only on `posttrain-common` and `httpx`) holding the calculator, the rules and the review, all reading the recorded run snapshot. Work validation, the CLI and Observatory call the same code. `performance_guard.py`, `training_guard.py` and the TRL key set moved there; the run-start `configuration_validated` event prototype was removed.
- [x] (2026-09-23 21:00Z) Observatory: `ConfigurationReview` on every run view (metrics, evaluation, generic, serving), computed from the recorded selections with `HubModelReader` (`POSTTRAIN_OBSERVATORY_MODEL_CONFIG=hub|disabled`), step options calibrated from `train/rl/time/rollout_seconds` and `train/step_time_seconds`. Run config page redesigned; Overview summary strip.
- [x] (2026-09-23 23:30Z) Compatibility rules (`posttrain.advisor.compatibility`) for the remaining qualified optimizations, ported plan-time checks included so Observatory shows them for recorded runs: FA4 on SM120, TurboQuant with FA3/FA4, DSpark with TurboQuant, BF16/MTP/TurboQuant target support, SM120 FA4 on Gemma, truncated attention-backend priority, FlashAttention version per architecture, hybrid multi-turn prefix reuse before `vllm@0.29.1.dev3`, implicit prefix caching, TurboQuant recall on long Qwen3.5 contexts, TRL rollout (dtype not applied, no sleep, rollout-worker contract, Uno sync, Liger on GDPO/CAPO, QLoRA sync, LoRA weight-name prefix, relaxed parity limit, batch request mode), veRL overrides (eager default, forced prefix-caching off, max_num_seqs default), batch invariance (off for evaluations, untuned runtime, unvalidated architectures, SM120 FA4), and `SPECULATIVE_DECODING_AVAILABLE` (warning) from recorded MTP capability or the qualified-drafter registry.
- [x] (2026-09-23 23:30Z) `engine.batch_invariant` binding key: serving (server and offline) and TRL rollouts export `VLLM_BATCH_INVARIANT=1`; runs now record invariance. Runs recorded before it ran without invariance, since no binding could enable it.
- [x] (2026-09-23 23:30Z) Snapshot records the served model's facts on each inference seat (family, precision, base repo, parameters, MTP capability, native context) and `lr_scheduler_type`.
- [x] (2026-09-23 23:30Z) Calculator: speculative decoding (draft weights, drafter or native-MTP KV per token, k+1 verification tokens), co-tenant engines and trainer on the same target (the Gemma judge beside LFM rollouts now sizes to 0.46, against the hand-tuned 0.47), TRL-colocated suggestions limited to keys TRL reads; the unconditional draft-free speculation note is gone.
- [x] (2026-09-23 23:40Z) Re-audit of 78 lab work packages: 67 have errors (66 eager, 22 hybrid prefix reuse on vllm 0.25.1/0.26.1/dev2, 19 TRL-ignored keys, 3 float16 on bf16); warnings include 68 implicit prefix caching, 37 speculative decoding available, 34 LoRA learning rate, 23 evaluations without batch invariance.
- [x] (2026-09-25) veRL worker honours the binding's `enable_prefix_caching` (off when omitted; LoRA syncs with it are unqualified), `enforce_eager` and `batch_invariant` (`VLLM_BATCH_INVARIANT=1` for the veRL subprocess). `VERL_PREFIX_CACHING_FORCED_OFF` became `VERL_PREFIX_CACHING_OFF_BY_DEFAULT`, reported only when the binding omits the key.
- [x] (2026-09-25) Rebased onto `codex/release-0.4.5` as `codex/performance-guard`. New revisions from `posttrain settings suggest`: `inference/qwen3.5-0.8b-vllm-distill-rollout@4` (vllm 0.29.1.dev3, CUDA graphs, colocated 8 GB sizing; revision 1 forced eager after GDN CUDA graphs illegal-addressed on vllm 0.25.1) for the `posttrain init` GRPO starter and lab jobs, and `inference/qwen3.5-2b-vllm-eval@3` (CUDA graphs, prefix caching, 8,192 batched tokens) for the foundation screen and release-gate evaluation. The 37 bindings that recorded runs used acknowledge their error codes with the recorded packages and the successor. 85 of 86 lab work packages validate; `environment_library_automationbench_qualification` fails before and after this plan (its eval binding does not declare tool calling).
- [x] (2026-09-25) Serving defaults (`posttrain.serve.backends.vllm.bindings.engine_config`): an omitted `dtype` follows the checkpoint precision (bf16 -> bfloat16; float16 with a TurboQuant KV cache), and an omitted `enable_prefix_caching` is on.
- [ ] Qualify the colocated small-GPU starter (Qwen3.5-0.8B GRPO on 8 GB) with CUDA graphs through a real `job run` on the 8 GB instance.

## Surprises & Discoveries

- Observation: `work-package validate` never ran the model-configuration rules, and `job run` discarded non-error findings, so existing advice was invisible outside `job plan --json --explain`.
  Evidence: `validate_work_package` called only per-definition validators and preflight.
- Observation: TRL silently ignored engine keys that bindings set, and a hand-made key list missed keys read through a local `rollout` variable (`weight_name_prefix`), which is why the drift test scans the backend source.
  Evidence: `free_cache_engine` on every LFM TRL rollout binding.
- Observation: step size and concurrency are separate knobs. Memory alone says about 550 typical-length LFM2.5 sequences fit on the RTX PRO 6000, but memory-bound decode throughput flattens (80% of its limit) near 150, and calibrated against VORTEX v2's measured times (159 s rollout, 853 s step) the "recommended" 152-row step only gains about 1.07x rows per second because updates, scoring and refill rounds grow with rows.
  Evidence: `posttrain settings suggest --work-package apps/lab/.posttrain/work_packages/lfm26_automationbench_vortex_20_local_v2.yaml`; Observatory Run config for `lfm26-vortex-v2-agentic-20260923-r1`.
- Observation: the v2 run recorded no `performance_acknowledgements` (added after it started), so Observatory reports its ignored `free_cache_engine` as an error; that is correct for what the run recorded.

## Decision Log

- Decision: performance findings are errors unless acknowledged, not warnings.
  Rationale: the user asked for hard errors; warnings had been invisible for months. Acknowledgements keep deliberate trade-offs possible and visible in every plan.
  Date/Author: 2026-09-23 / user, recorded by Claude.
- Decision: superseded or recorded bindings are acknowledged, not edited under the same revision.
  Rationale: run snapshots record the binding's values; changing a revision in place would misstate history.
  Date/Author: 2026-09-23 / Claude.
- Decision: a calculator, not a per-binding tuning sweep.
  Rationale: the user: "we already know the numbers from the GEMM related work ... make a calculator which can take in hardware profile and model config and task settings". `scripts/qualification/binding_tuning/` is abandoned.
  Date/Author: 2026-09-23 / user.
- Decision: rules and calculator read the recorded run snapshot, and Observatory computes its own review; no launcher-recorded findings.
  Rationale: the user asked that Observatory "generate its own recommendations so older jobs show warnings as well". The snapshot is the one representation that plan-time validation and every recorded run share. A read-only product computing analysis from recorded evidence stays within the baseline.
  Date/Author: 2026-09-23 / user, implemented by Claude.
- Decision: a rollout's demand is the step's rows, capped at what fits; larger steps are collected in waves. Spare capacity is split: oversampling at most 20% of the step, the rest as more prompts per step. Options report waves and step time.
  Rationale: the user: "oversampling recommendation should be 20% of the step budget at max and then other recommendation should be related to increase prompts", and "provide step time and waves to them so they are able to decide". Oversampling currently runs as sequential active-sampling refill rounds; a parallel first-round oversampling setting does not exist yet, so the calculator reports the budget and the trainer change is a follow-up.
  Date/Author: 2026-09-23 / user.

## Outcomes & Retrospective

Rules, calculator and Observatory review are implemented and verified on real runs (VORTEX v1/v2, the September GRPO runs and held-out evals all show findings). Catalog bindings are migrated (new revisions for live paths, acknowledgements for recorded ones), the veRL worker honours the binding, and serving defaults follow the checkpoint. Remaining: the 8 GB starter qualification with CUDA graphs on vllm 0.29.1.dev3, which also settles whether the GDN colocated-LoRA illegal-address seen on vllm 0.25.1 is gone.

## Context and Orientation

A run's recorded selections (its resolved-input snapshot) map each role to `{selection_id, revision, resolved}`; `resolved` for an inference binding holds `engine`, `sampling`, `purpose`, `target_id`, `model_variant_id` and optional `performance_acknowledgements`; `execution_targets.targets[]` holds target memory and hardware; the model seat holds `artifact.repo_id/revision` and `weight_precision`. Plan-time validation sees the same snapshot as `resolved_inputs`, and trackers keep it with the run.

- `packages/advisor/src/posttrain/advisor/snapshot.py`: seat recognition by recorded fields.
- `rules.py`: performance and training rules; `TRL_ROLLOUT_ENGINE_KEYS`.
- `calculator.py`: `suggest(hardware, architecture, task)`; `StepCapacity`/`StepOption`.
- `review.py`: `review(snapshot, load_architecture)` → rule findings, calculator findings (`CALCULATOR_*`), recommendations.
- `hub.py`: `HubModelReader` (cached `config.json` plus safetensors parameter count over HTTP).
- `packages/work/src/posttrain/work/runner.py`: `_configuration_findings(seats, snapshot)` adds `rule_findings(snapshot)`.
- `apps/cli/src/posttrain_cli/commands/settings.py`: `posttrain settings suggest`.
- `apps/observatory/src/posttrain_observatory/configuration.py`: review projection and calibration; `service.py` attaches it to each view; `frontend/src/features/config/ConfigPage.tsx`: page, panel and Overview strip.

## Plan of Work

Remaining: add rules for the optimizations the inventory finds unenforced; generate new binding revisions from `posttrain settings suggest --work-package` output and move templates, packages and tests to them; change serving defaults; qualify the 8 GB starter.

## Concrete Steps

From the repository root:

    uv run --package posttrain posttrain settings suggest --work-package apps/lab/.posttrain/work_packages/<pkg>.yaml
    cd apps/lab && uv run --package posttrain posttrain work-package validate .posttrain/work_packages/<pkg>.yaml --strict
    uv run pytest packages/advisor packages/work apps/observatory apps/cli
    cd apps/observatory/frontend && npx vitest run && npm run build

## Validation and Acceptance

Every lab work package validates with `--strict`; starter and release-gate tests pass; Observatory's Run config for `lfm26-vortex-v2-agentic-20260923-r1` shows the TRL-ignored-key error on `free_cache_engine`, the learning-rate warning on `settings.learning_rate`, and step options with calibrated step times; an old eager run shows `VLLM_EAGER_DISABLES_CUDA_GRAPHS` on its Overview strip.

## Idempotence and Recovery

Reviews are pure functions of recorded selections and model configs; recomputing is safe. Binding changes are new revisions, so any work package can point back at a previous revision. `POSTTRAIN_OBSERVATORY_MODEL_CONFIG=disabled` turns off network reads, leaving rule findings.

## Artifacts and Notes

Observatory reads public model configs from the Hugging Face Hub (`HF_ENDPOINT`, `HF_TOKEN` honoured); the read is cached per process and failures are retried after five minutes.

## Interfaces and Dependencies

`posttrain.advisor.review(snapshot, load_architecture=None) -> Review(findings, recommendations)`; `posttrain.advisor.rule_findings(snapshot)`; `posttrain.advisor.HubModelReader()`; `InferenceBinding.performance_acknowledgements: Mapping[str, str]`; `posttrain.work.work_package_findings(context, package)`; Observatory `RunView.configuration: ConfigurationReview | None`.
