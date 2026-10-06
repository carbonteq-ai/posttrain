# Resubmit the 2.6B manifest-steps SAMPO run on the corrected reward and objective

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

The 2.6B run `manifest-steps-26-sampo-100-g16x8-20261006-r10` trained 100 updates (25 collections) with two defects found afterwards: manifest step credit almost never applied in live rollouts (Verifiers scores before setting `trace.ok`; fixed in environments `089b67d`), and SAMPO used one importance ratio per episode, so its clip never acted (clip fraction exactly 0 on all 110 logged updates). This plan resubmits the same experiment, not a resume, with the fixes, the 103-task mix `automationbench-manifest-luna-v2` (Simple 20), and SAMPO's per-turn objective. r10 was stopped and its checkpoint 100 is evaluated on the matched held-out suites so the new run has a direct comparison.

## Progress

- [x] (2026-10-06 07:30Z) r10 cancelled; checkpoint 100 (`training/sampo/checkpoints/step-00000100`) retained on the workstation.
- [x] (2026-10-06 07:35Z) Held-out evaluations of r10 checkpoint 100 queued, one at a time on the workstation: `heldout-matched-64k-v4-r10-c100-20261006` (temperature 0.1, 3 attempts) then `heldout-matched-64k-v4-t05-r10-c100-20261006` (temperature 0.5, 5 attempts), packages `lfm26_automationbench_heldout_matched_eval_64k_v4.yaml` and `..._v4_t05.yaml`, `--model-from-run manifest-steps-26-sampo-100-g16x8-20261006-r10 --model-checkpoint-step 100`.
- [x] (2026-10-06 07:40Z) Work package `apps/lab/.posttrain/work_packages/lfm26_automationbench_manifest_steps_sampo_100_g16x8_ws.yaml` retargeted: environment `automationbench-manifest-steps-luna2-4k16t-v1` (environments `089b67d`, 103 tasks), reward `reward/automationbench-manifest-steps@2` (scorer version 4, `f221a532...`), settings `lfm2.5-2.6b/automationbench-manifest-steps-sampo-100-g16x8-v1` revision 3 (objective `turn-rows`, 32-episode updates, 4 per collection, `mean`, clip 0.003/0.004, `retain: learning_signal`). `posttrain job plan --strict` resolves; only finding is the optional TurboQuant recommendation.
- [x] (2026-10-06 08:30Z) Settings revision 4 adds VORTEX's yield-first adaptive curriculum by domain (as validated locally in r5/r7) and the 103-task environment gains the domain facet. Pre-flight on the resolved catalog: objective `turn-rows`, 32-episode updates (400 in all), curriculum `yield_first`/`domain`, active sampling 5 rounds, oversample 4, refill 4, `retain: learning_signal`, token-truncated sampler correction at 2, truncation penalty 0.1, KL 0.01 to the base; environments `089b67d`, 103 tasks, manifest assessments on; reward projection digest equals the environment's turn-reward digest (`f221a532...`).
- [x] (2026-10-06 08:35Z) Submitted as `manifest-steps-26-sampo-100-luna2-20261006-r1` (user goal: submit and monitor two collections); queued behind r10's `v4` held-out evaluation on the workstation. The `v4_t05` evaluation is deferred until after this run.
- [ ] First-collection checks on the new run (below), then the held-out evaluations at checkpoint 100.

## Surprises & Discoveries

- Observation: the first submission (`manifest-steps-26-sampo-100-luna2-20261006-r1`) rejected every rollout group: `ContractError: the configured Trackio build cannot store environment metrics; install Trackio 0.31.5.post14.dev33 or newer`. The merged Observatory branch requires Trackio >= dev33 for traces that carry environment metrics, but every published job-kind image ships dev32. Three active rounds produced 60 groups, all `terminal: failed` with no traces; the run was cancelled before its fifth round. Settings did take effect (recorded config: trainer fp16, vLLM float16, task mix `automationbench-manifest-luna-v2` with 103 tasks, `turn-rows`, `retain: learning_signal`; curriculum metrics logged; device memory at the collection step).
- Observation: completing the dev35 pin in the job-kind locks (`fbe677bf`) is not enough on its own: `published.toml` binds each kind image to its lock digest, so packing refuses edited locks until the kind images are republished. `posttrain-release images plan` lists six kind images to rebuild (supervised, TRL, veRL, eval, serve, transform); the TRL kind is republished first to unblock this run.

- Observation: r10's checkpoint-100 held-out evaluation could not run on the suites' pinned environments revision `0bad6187`, which requires Verifiers `e6a3d9bb` while the current evaluation job pins `58df1306`. With the user's choice, both matched suites were re-pinned to `089b67d` on branch `wip/r10-heldout-eval-089b67d` (`ef3fd158`); 15 benchmark-core files changed in between, so scores are not strictly comparable with earlier systems.
- Observation: the first re-pinned evaluation (`heldout-matched-64k-v4-r10-c100-089b67d-20261006-b`) scored 0 on all 60 rollouts because every tool call returned `linked tool execution requires capture enabled`: since `427e802` AutomationBench executes tools only for tasks with `capture_actions: true`, which the held-out task config lacked. Fixed in `081e1288` (both re-pinned suites capture actions). The two evaluations rerun after this training run frees the workstation.

None yet for this run; the defects that motivate it are recorded in `docs/plan/automationbench-simple-manifests.md` and `docs/plan/vortex-resolved-engine.md`.

## Decision Log

- Decision: retarget the existing 2.6B work package instead of adding a parallel one.
  Rationale: the user asked to resubmit the experiment, not resume r10; r10's run record keeps its own resolved configuration.
  Date/Author: 2026-10-06, Claude.
- Decision: keep `advantage_normalization: mean` and the cont100 schedule.
  Rationale: the run changes only what was broken (step credit, objective) and the task mix, so its comparison with r10 and the older SAMPO system stays interpretable.
  Date/Author: 2026-10-06, Claude.

## Outcomes & Retrospective

None yet.

## Context and Orientation

Workstation: `carbonteq-ai-workstation.lan` (RTX PRO 6000, dstack), one job at a time. Launch from a clean worktree with its own CLI. Metrics are readable from Trackio directly (`posttrain_tracking_trackio.adapter.TrackioDataSource("posttrain-lab", server_url="https://trackio.lan")` with `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`).

## Concrete Steps

From `/home/hammad/projects/worktrees/rl-perf/apps/lab` on a clean tree:

    UV_HTTP_TIMEOUT=300 ../../.venv/bin/posttrain job run .posttrain/work_packages/lfm26_automationbench_manifest_steps_sampo_100_g16x8_ws.yaml --run-id manifest-steps-26-sampo-100-luna2-20261006-r1

## Validation and Acceptance

After the first collection (4 updates): `train/rl/policy_ratio_segments` counts turns (not episodes); `train/rl/policy_log_ratio_abs_mean` is 0 on the collection's first update and non-zero on updates 2-4; per-turn `manifest_goal_credit` is non-zero on most fully successful episodes (r7 local: 26 of 29); `train/rl/turn_credit_share` is reported; gradients are finite. At checkpoint 100, the two matched held-out suites are run with the same packages as r10's checkpoint 100.

## Idempotence and Recovery

A failed submission is resubmitted with a new run id. The evaluation chain is `/tmp/claude-1000/-home-hammad-projects-rl/06f4fe1b-c6e2-45ea-9bb2-d0b2a04f9815/scratchpad/runs/r10_evals.sh` (log `r10-evals.log`); rerun one suite with the `job run` line it prints.

## Interfaces and Dependencies

Environments revision `089b67d` (`carbonteq-ai/verifiers-environments`, branch `wip/automationbench-simple-manifests-2026-10-06`).
