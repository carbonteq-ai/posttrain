# Qualify and run matched VORTEX v3/v4 held-out evaluations

This ExecPlan is a living document maintained according to `docs/templates/PLAN.md`. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current at every stopping point.

## Purpose / Big Picture

The earlier VORTEX v3 held-out evaluation treated provider context errors as scores: 39 of 60 episodes received HTTP 400 because the server reserved 12,288 output tokens inside a 16,384-token context. The user needs a trustworthy comparison of the VORTEX v3 and yield-first v4 step-20 descendants on the same 20 held-out AutomationBench tasks, three repetitions each. After this work, both runs must have complete, comparable traces, explicit error/truncation accounting, and exact model-checkpoint lineage. Neither a successful provider process nor a mean reward alone constitutes acceptance.

## Progress

- [x] (2026-09-24 07:50Z) Verified both training runs succeeded and expose step-20 model views. V4 reconciliation and cleanup succeeded; its 10 artifacts remain and the RTX PRO 6000 is idle.
- [x] (2026-09-24 08:05Z) Created isolated `codex/vortex-matched-evals` from the v4 source revision and cherry-picked the two commits of PR #118. The main checkout and dirty Verifiers-environments sibling remain untouched.
- [x] (2026-09-24 08:18Z) Added candidate held-out v2 and two-task canary selections with the same held-out task inventory, an 8K per-call and 16K episode output allowance, and a 48K context on the RTX PRO. Static work-package validation passed for both.
- [ ] Add focused catalog tests for identical v1/v2 task identities and distinct output budgets; run targeted and repository boundary checks.
- [ ] Prove actual-job packaging resolves the Trackio dev25 wheel over the older kind image, and inspect the exact image and source digest. If the kind-image contract forbids this, publish the minimum reviewed kind revision or stop and report that gate; never use the old dev24 runtime.
- [ ] Measure actual rendered prompt token counts with the selected server tokenizer, verify prompt plus reserved per-call output fits 48K with headroom, and run a two-task canary from the v3 step-20 model on the RTX PRO. Inspect GPU/KV admission, compiled inference, tool calls, per-call provider errors, episode truncation, and complete trace sync. The v2 full run cannot start if the canary fails.
- [ ] Check idempotency across local execution state, dstack, and Trackio. Submit one full v3 and one full v4 evaluation, both selecting exact step-20 model views, only after the same immutable image/profile is qualified. Monitor to terminal and reconcile before cleanup.
- [ ] Report 20/20 task and 60/60 episode coverage per arm, valid task-mean reward and success metrics, provider-error and truncation counts, trace-sync status, checkpoint lineage, GPU/KV usage, and the limitation that v3/v4 training changed both controller and rollout budget.

## Surprises & Discoveries

- The local main checkout lacks the v4 run's submission record; the submitting worktree at `/home/hammad/.codex/worktrees/vortex-yield-first-release/rl` owns its reconciliation and cleanup receipts. Running `posttrain run reconcile` from the main checkout reports `execution submission is missing`, although the provider run succeeded.
- The current eval job-kind image is based on Trackio dev24, whereas PR #118 pins dev25. The actual-job Dockerfile installs the hash-locked runtime closure before local framework source. This *may* upgrade Trackio without a public framework release, but no immutable image has yet proved it.
- `posttrain job plan --explain` reports one recommendation, `TURBOQUANT_AVAILABLE`, not an invalid binding. TurboQuant is deliberately not selected for this matched comparison.

## Decision Log

- Decision: evaluate the two step-20 descendants under one new manifest and inference profile; do not compare a new v4 score to the broken old v3 score. Rationale: output/context and transport differences would confound the result. Date/Author: 2026-09-24, Codex.
- Decision: use a candidate 48K context, 8K per-call output, and 16K episode output, with compiled FA2 and prefix caching. Rationale: the LFM catalog advertises 131K native context, the prior 16K profile left only 4K input headroom, and v4 trained with 8K per call/16K per episode. These are hypotheses until the live tokenizer and RTX PRO tests pass. Date/Author: 2026-09-24, Codex.
- Decision: retain a two-task canary distinct from the scored 20-task population. Rationale: runtime and transport failures should cost at most two episodes before spending 120 full-eval episodes. Date/Author: 2026-09-24, Codex.

## Outcomes & Retrospective

Preparation is incomplete. No eval has been submitted, no 48K GPU admission or tokenizer parity has been claimed, and PR #118 remains unmerged at the last check. The original run's errors remain historical evidence, not zero-valued model performance.

## Context and Orientation

The frozen product baseline in `docs/post-training/README.md`, `01-workflow.md`, `03-work-and-evidence.md`, and `06-observation-and-lineage.md` already separates execution errors from semantic task failure and keeps model lineage on artifact edges. This plan changes no product meaning. `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` defines the source environment, evaluation plan, and LFM inference binding. `apps/lab/.posttrain/work_packages/lfm26_automationbench_vortex_matched_eval_48k_v2.yaml` composes the full evaluation; `lfm26_automationbench_vortex_eval_context_canary_48k_v1.yaml` composes the bounded gate. `packages/eval` executes Verifiers evaluations, `packages/tracking-trackio` synchronizes facts, and the Trackio fork supplies the stored trace schema. An actual-job image is the content-addressed per-job layer on top of a job-kind image; its digest, not a mutable checkout path, is the runnable identity.

The relevant source branch is `/home/hammad/.codex/worktrees/vortex-matched-eval/rl` at `codex/vortex-matched-evals`. Its base was `57e3797c5382428a95e7c5e57056410e45ad200c`; commits `9189e738` and `bf96ef43` incorporate PR #118. The main `/home/hammad/projects/rl` and sibling `/home/hammad/projects/verifiers-environments` contain unrelated dirty work. Do not merge that work implicitly. The prior source run is `lfm26-vortex-v3-lr2e4-20260923-r1`; the new source run is `lfm26-vortex-yield-first-v4-8k-20260923-r1`. Both expose step 20. The previous invalid eval is `eval-lfm26-heldout20x3-vortex-v3-agentic-20260923-r1`.

## Plan of Work

First test that v1 and v2 task names and task-mix checksum are identical and that only the intended episode/sampling budgets differ. Keep the new selections versioned, because the earlier manifest must remain reproducible. Complete the static CLI composition and import checks. Then pack the canary from committed source with `--model-from-run` and `--model-checkpoint-step 20` through the supported `posttrain job run` path, which must produce and use one immutable image digest. Inspect the image runtime before spending full evaluation compute; the Trackio schema preflight must pass under dev25. The canary must establish actual server context, rendered prompt tokens and output reservation, GPU/KV admission, valid tool calls, and synchronized traces. If a request can exceed the 48K envelope, add bounded request sizing at the owner of the Verifiers inference call and version/pin that change instead of silently truncating evidence or increasing context blindly.

After a passing canary, submit each full run with its own stable ID and exact step-20 model view. Use one work package and selection digest for both. Query local execution state and the provider before each submit so retrying this plan never creates a second matching run. Observe every episode and task, then reconcile the terminal provider outcome with Trackio artifacts. Only after reconciliation collect each job to free GPU and workspace without deleting retained model/eval evidence.

## Concrete Steps

From `/home/hammad/.codex/worktrees/vortex-matched-eval/rl`, run `uv run pytest -q apps/lab/tests/test_catalog.py apps/lab/tests/test_work_packages.py packages/eval/tests/test_api.py packages/tracking-trackio/tests/test_adapter.py`, `uv run ruff check .`, `uv run lint-imports`, and `git diff --check`. From its `apps/lab` directory, run `uv run posttrain work-package validate lfm26_automationbench_vortex_eval_context_canary_48k_v1.yaml` and the corresponding matched file; both should report five resolved seats and one job. Use `uv run posttrain --json job plan <file> --explain` to inspect issue severity. The current only finding is the optional TurboQuant recommendation.

Before any submission, run `uv run posttrain job run --help` to verify that this checkout supports `--model-from-run` and `--model-checkpoint-step`, then verify no run with the chosen exact ID exists in both `posttrain run list` and dstack/Trackio. Use `--model-from-run lfm26-vortex-v3-lr2e4-20260923-r1 --model-checkpoint-step 20` for the canary and first full arm; use the v4 run ID for the second. Do not omit the checkpoint step. Record the immutable image digest and provider ID returned at submission. There is no safe shell command to paste for submission before the image and live capacity gates have been completed and the exact run IDs have been checked.

## Validation and Acceptance

The canary must produce two native episodes with no provider 400, no unreported truncation, valid tool-call parsing, a visible source checkpoint, and two synchronized trace records. The live serving log must confirm 48K context and a compiled path, and the GPU/KV series must show admission without OOM. Measure at least the longest rendered prompt actually observed and check prompt tokens plus reserved 8,192 output tokens plus a documented safety margin stay within 49,152. If this cannot be proved, stop; do not infer success from static YAML.

Each full run must retain 20 distinct task identities and three attempts per task, with 60/60 valid episodes and trace records. A provider error or truncation makes that attempt invalid/missing under strict measurement, not reward zero. Compare task means and success rates only when both runs have the same population and sampling, and separately report cost, latency, GPU/KV, output-token totals, and confidence/uncertainty. The v3/v4 controller comparison is not policy-only causal evidence because training output budget also changed.

## Idempotence and Recovery

The canary and each full eval use distinct immutable run IDs and model lineage. Check existing provider and tracking records before retrying ambiguous submissions. Do not restart a healthy running job or delete old evidence. If the image build fails due to Trackio/kind lock mismatch, preserve logs and stop at that publication gate; do not use dev24 or disable trace sync. If the canary fails because of context overflow, adjust a new versioned profile or request-budget owner, rerun static checks, publish a new image digest, and use a new canary ID. Reconciliation and cleanup are separately idempotent; cleanup only follows retained evidence.

## Artifacts and Notes

The v4 cleanup receipt is `/home/hammad/.codex/worktrees/vortex-yield-first-release/rl/apps/lab/.posttrain/state/executions/lfm26-vortex-yield-first-v4-8k-20260923-r1/cleanup.json`: `outcome=succeeded`, `evidence_state=reconciled`, `retained_artifact_count=10`, `workspace_disposition=removed`. `posttrain run checkpoint verify ... --step 20` reported `model_digest=true`, `recovery_digest=true`, `paired_views=true`. Provider deep blob verification is unsupported, so do not claim it passed.

## Interfaces and Dependencies

Use the existing `EnvironmentBinding`, `InferenceBinding`, and `EvaluationPlan` catalog schemas; do not introduce a new evaluator contract. The work-package `model` seat is replaced by the committed model view from `--model-from-run`, not by an untracked adapter directory. The evaluation source remains immutable Verifiers-environments commit `9bbd3116e6b5a444d6cf103dce18b1866ae32787` unless request-budget repair genuinely requires a new committed source pin. Trackio dev25 is the minimum trace-schema-compatible storage client. The runtime profile uses CarbonTeq vLLM `0.29.1.dev3`, LFM tool renderer `lfm2.5-tools-thinking@2`, FA2, compilation enabled, chunked prefill and prefix caching; the live canary must prove that those features actually activate.

Revision note (2026-09-24): Created after the training run was collected and the first static matched/canary profiles passed; the plan records the remaining immutable-image and live-evaluation gates rather than treating static validation as launch readiness.
