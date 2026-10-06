# Resubmit the 2.6B manifest-steps SAMPO run on the corrected reward and objective

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

The 2.6B run `manifest-steps-26-sampo-100-g16x8-20261006-r10` trained 100 updates (25 collections) with two defects found afterwards: manifest step credit almost never applied in live rollouts (Verifiers scores before setting `trace.ok`; fixed in environments `089b67d`), and SAMPO used one importance ratio per episode, so its clip never acted (clip fraction exactly 0 on all 110 logged updates). This plan resubmits the same experiment, not a resume, with the fixes, the 103-task mix `automationbench-manifest-luna-v2` (Simple 20), and SAMPO's per-turn objective. r10 was stopped and its checkpoint 100 is evaluated on the matched held-out suites so the new run has a direct comparison.

## Progress

- [x] (2026-10-06 07:30Z) r10 cancelled; checkpoint 100 (`training/sampo/checkpoints/step-00000100`) retained on the workstation.
- [x] (2026-10-06 07:35Z) Held-out evaluations of r10 checkpoint 100 queued, one at a time on the workstation: `heldout-matched-64k-v4-r10-c100-20261006` (temperature 0.1, 3 attempts) then `heldout-matched-64k-v4-t05-r10-c100-20261006` (temperature 0.5, 5 attempts), packages `lfm26_automationbench_heldout_matched_eval_64k_v4.yaml` and `..._v4_t05.yaml`, `--model-from-run manifest-steps-26-sampo-100-g16x8-20261006-r10 --model-checkpoint-step 100`.
- [x] (2026-10-06 07:40Z) Work package `apps/lab/.posttrain/work_packages/lfm26_automationbench_manifest_steps_sampo_100_g16x8_ws.yaml` retargeted: environment `automationbench-manifest-steps-luna2-4k16t-v1` (environments `089b67d`, 103 tasks), reward `reward/automationbench-manifest-steps@2` (scorer version 4, `f221a532...`), settings `lfm2.5-2.6b/automationbench-manifest-steps-sampo-100-g16x8-v1` revision 3 (objective `turn-rows`, 32-episode updates, 4 per collection, `mean`, clip 0.003/0.004, `retain: learning_signal`). `posttrain job plan --strict` resolves; only finding is the optional TurboQuant recommendation.
- [ ] Submit after the evaluations finish and the user approves (not submitted).
- [ ] First-collection checks on the new run (below), then the held-out evaluations at checkpoint 100.

## Surprises & Discoveries

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
