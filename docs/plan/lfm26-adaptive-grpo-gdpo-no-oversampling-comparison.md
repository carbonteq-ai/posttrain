# Run the matched LFM2.5 adaptive-GRPO and GDPO comparison

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

This experiment measures two LFM2.5-2.6B LoRA training policies on the same AutomationBench task mix after the Posttrain 0.4.4 release. One arm uses adaptive curriculum with ordinary GRPO and the other uses GDPO with a self-hosted Gemma judge. Each arm performs 50 optimizer updates, for 100 total planned updates. Every update collects eight prompts with four trajectories per prompt at inference concurrency 32. Neither arm may collect extra candidate batches. After both descendants exist, the same held-out evaluation is run against each adapter.

The first three optimizer updates of each arm are an operating gate. Logs and metrics must show finite optimization values, exactly 32 requested trajectories per update, correct LoRA policy-version advancement, and enough sustained GPU work to justify the selected concurrency. If the accelerator is materially under-filled, the run must be stopped rather than wasting the remaining budget, the inference policy adjusted, and the arm restarted from a new immutable run identity.

## Progress

- [x] (2026-09-20 12:00Z) Defined two matched 50-update training selections, c32 inference profiles, and one shared held-out evaluation package.
- [x] (2026-09-20 12:10Z) Corrected the no-oversampling arm from invalid OLMo3 to ordinary GRPO with adaptive initial-batch curriculum allocation.
- [x] (2026-09-20 12:12Z) Validated all three work-package compositions with the repository CLI.
- [x] (2026-09-20 12:35Z) Qualified the released runtime images and both job plans from a clean Posttrain 0.4.4 stable-index installation.
- [x] (2026-09-20 12:39Z) Passed 132 focused tests plus Ruff, committed the experiment as `480ece91`, and pushed its branch.
- [ ] Submit the first 50-update arm and monitor at least three completed optimizer updates with GPU and inference evidence (completed: attempt `lfm26-adaptive-grpo-50-c32-20260920a` proved a compiler/runtime attention-contract bug before GPU allocation; remaining: qualify the fix and resubmit).
- [ ] Submit or queue the second 50-update arm and monitor at least three completed optimizer updates with the same evidence (completed: attempt `lfm26-gdpo-50-c32-gemma-mtp2-20260920a` was cancelled when the shared defect was identified; remaining: qualify the fix and resubmit).
- [ ] Run the common held-out evaluation once against each materialized adapter.
- [ ] Reconcile the comparison and record final run identities and evidence.

## Surprises & Discoveries

- Observation: OLMo3 is not a valid no-oversampling algorithm in this framework.
  Evidence: work-package validation returned `OLMo 3 requires active group sampling`; `GRPOSettings` enforces this contract, while adaptive curriculum independently supports the `initial_batch` sampling mode with ordinary GRPO.

- Observation: The first live attempt failed before GPU allocation because the framework emitted a nonexistent vLLM `AttentionConfig.backend_priority` field.
  Evidence: vLLM raised `ValidationError: backend_priority Unexpected keyword argument`. The released fork exposes scalar `AttentionConfig.backend`; compilation now resolves the ordered framework policy to that scalar runtime field.

## Decision Log

- Decision: Interpret “100 steps total” as two matched 50-step arms rather than 100 steps per arm.
  Rationale: This preserves an equal comparison and matches the explicit total budget.
  Date/Author: 2026-09-20 / Codex

- Decision: Name the first arm adaptive GRPO, not VORTEX.
  Rationale: VORTEX combines the OLMo3 update recipe, active group refill, and adaptive curriculum. Removing oversampling removes active refill, and the typed framework contract correctly rejects OLMo3 without it. An honest ablation uses GRPO plus adaptive initial-batch allocation.
  Date/Author: 2026-09-20 / Codex

- Decision: Use native FA4 for LFM and the qualified Triton MTP-2 path for the Gemma judge, subject to pre-submission policy compilation.
  Rationale: Posttrain 0.4.4 qualifies native Blackwell FA4 for eligible target models but does not qualify the experimental SM120 FA4 path for Gemma. Gemma MTP-2 on Triton is the retained correct path.
  Date/Author: 2026-09-20 / Codex

## Outcomes & Retrospective

Configuration is complete and statically valid. Live backend qualification, submission, three-step monitoring, final evaluation, and comparison remain. No live run has yet been claimed successful.

## Context and Orientation

`apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` contains model, environment, training-setting, and inference selections. A selection is an immutable, named configuration value. The work packages under `apps/lab/.posttrain/work_packages/` bind those values into one train or evaluation job. `apps/lab/src/posttrain_lab/qualification/gates.toml` records the intended qualification and prevents an experimental result from being mistaken for released evidence.

The adaptive-GRPO arm chooses task classes before the initial batch is collected. It does not refill rejected or uniform-reward groups. The GDPO arm uses a Gemma 4 12B MTP-2 judge to produce component rewards. “Concurrency 32” means the inference scheduler may keep 32 sequences in flight; it does not mean 32 separate model processes. A “policy-version fence” prevents a rollout from using stale KV-cache state after a LoRA update.

The held-out evaluator is intentionally separate from training. It accepts a materialized training run through `--model-from-run`, so the exact trained adapter and lineage are preserved instead of evaluating a mutable path.

## Plan of Work

First, validate the catalog and compile both dstack job plans using Posttrain 0.4.4 installed from the stable package index. Confirm that the LFM rollout resolves to native FA4 with CUDA graphs and prefix caching, while the Gemma judge resolves to Triton attention with MTP-2. Reject any silent fallback that changes the intended operating point.

Second, run focused catalog, qualification-gate, training-policy, and work-package tests. Commit the configuration and plan so each submitted job has an immutable source revision. Submit the adaptive-GRPO and GDPO work packages with distinct run IDs. Because one RTX PRO 6000 may serialize jobs, queueing is acceptable; overlapping two memory-heavy training jobs on one GPU is not.

Third, observe each arm through at least three completed optimizer updates. Capture optimizer step, trajectory counts, truncation, rollout throughput, queue depth or scheduler occupancy, KV-cache utilization and hit rate where applicable, GPU utilization and memory use, speculative acceptance for the Gemma judge, LoRA/policy version, loss, KL, gradient norm, and reward statistics. A run is not healthy merely because it remains alive. If metrics indicate underutilization, preemption, cache thrash, invalid values, stale weights, or malformed rewards, stop the affected run, change only the evidence-supported bottleneck, recompile the job, and resubmit under a new run ID.

Finally, after both 50-step descendants materialize, submit the same held-out evaluation twice using each training run as `--model-from-run`. Compare task-level means, completion and truncation, tool-call correctness, reward components, throughput, and resource efficiency. Keep training evidence and held-out evidence distinct.

## Concrete Steps

Run from `/home/hammad/projects/rl/apps/lab` unless stated otherwise:

    uv run posttrain work-package validate lfm26_automationbench_adaptive_grpo_no_oversampling_50_local_v1.yaml
    uv run posttrain work-package validate lfm26_automationbench_gdpo_episode_50_local_8x4_v1.yaml
    uv run posttrain work-package validate lfm26_automationbench_heldout_eval_c32_v1.yaml

Create a temporary clean Python 3.13 environment and install `posttrain==0.4.4` and `posttrain-lab==0.4.4` from `https://pypi.lan/carbonteq/stable/+simple/`. Use that CLI for `posttrain doctor`, `posttrain job plan`, and `posttrain job run`. Do not bypass a failed doctor check.

Run focused tests from `/home/hammad/projects/rl`:

    uv run pytest apps/lab/tests/test_catalog.py apps/lab/tests/test_qualification_gates.py packages/train/tests/test_adaptive_curriculum.py packages/train/tests/test_api.py packages/work/tests/test_validation.py -q
    uv run ruff check apps/lab packages/train packages/work
    git diff --check

Submit training only after both job plans resolve the requested backend policies. Use stable, descriptive run IDs containing the algorithm, 50-step budget, and date. Evaluation submission waits for its source training run to materialize.

## Validation and Acceptance

Static acceptance requires all three work packages to validate and the focused test suite, Ruff, and `git diff --check` to pass. Compilation acceptance requires both dstack plans to select the released runtime and compatible inference backends before submission.

Live acceptance for each arm requires three completed optimizer updates with finite loss, reward, KL, and gradient metrics; 8 prompt groups and 32 trajectories requested per step; no hidden candidate refill; changing policy versions after updates; and no malformed reward or stale-cache warnings. GPU acceptance requires sustained utilization during model-active windows rather than an average diluted by environment waits. If the runtime exposes scheduler occupancy, queue depth, KV-cache use, and token throughput, those values are the primary explanation of saturation. Overall GPU percentage alone is insufficient.

Final acceptance requires 50 completed updates in each arm and two held-out evaluations derived from the immutable outputs. Any restarted attempt remains in lineage and is reported rather than overwritten.

## Idempotence and Recovery

Validation and job planning are read-only and safe to repeat. Every submitted attempt must use a new run ID. Never resume a run after changing configuration; create a descendant attempt so the changed inference policy is visible. If the only RTX PRO is occupied, leave the second arm queued instead of evicting unrelated work. Do not delete partial evidence: failed attempts are useful saturation and correctness records.

## Artifacts and Notes

The first validation exposed a useful contract boundary:

    error: OLMo 3 requires active group sampling

After changing the arm to GRPO with adaptive initial-batch allocation, all three compositions reported `Static composition validation: complete`.

## Interfaces and Dependencies

The experiment uses Posttrain 0.4.4, the released CarbonTeq vLLM `0.29.1.dev2` runtime, TRL LoRA training, the AutomationBench Verifiers environment, Trackio/Doris evidence storage, and dstack scheduling on the RTX PRO 6000. The LFM rollout inference selection is `inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c32-4k@2`. The Gemma judge selection is `inference/gemma4-12b-vllm-automationbench-judge-mtp2-local-32k@3`. The held-out evaluator uses `inference/lfm2.5-2.6b-vllm-automationbench-eval-local@2`.

Revision note (2026-09-20): Created this plan after static validation rejected an invalid OLMo3-without-active-sampling interpretation; the plan records the corrected adaptive-GRPO design and the live operating gates. Updated after the first live attempt found that Posttrain emitted a framework-only backend-priority list into vLLM's scalar backend contract.
