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
- [x] (2026-10-06 11:50Z) Two-collection checks on r5 (see Outcomes).
- [x] (2026-10-06) Evaluations prepared for r5's checkpoints (not submitted while r5 holds the workstation): new suite `lfm26_automationbench_luna2_heldout_eval_6k16t_t05.yaml` (gate `lfm26-luna2-heldout-6k16t-t05`; luna-v2's 20 held-out tasks, environments `24e1fc9`, r5's budgets of 16 turns, 6144-token replies and 18432 per episode, temperature 0.5, 5 attempts, manifest findings recorded) and the older matched suites `..._64k_v4.yaml` / `..._64k_v4_t05.yaml` re-pinned to `24e1fc9` with action capture. All three resolve with `job plan`; `--strict` stops only on `BATCH_INVARIANCE_OFF_FOR_EVALUATION`, left off because every earlier evaluation of these suites ran without it.
- [x] (2026-10-06) Task-specific manifest coverage (environments `793b2c5`): ordered joins decide on projected receipts (receipt positions skip harness events), read goals accept re-reads, the quarterly hold notice accepts 'pause'. Replayed on r5 full successes: email_sf_log 6/6 credited (was 1/6), asana_dark_mode 4/6 (was 2/6), quarterly 6/6 (was 3/6).
- [x] (2026-10-06 17:57Z) r5 cancelled after update 207 to free the workstation; the cancel-time save left `trainer/.cancel-checkpoint` empty, so the resume point is checkpoint 200 (collection 50). Open: why the cancel save did not complete.
- [x] (2026-10-06) Deterministic simulated worlds (environments `c02546d`, vendored refresh `51ca09c`; AutomationBench fork `afb92ec` on `codex/deterministic-sim-runtime`). The 2.6B training environment `automationbench-manifest-steps-luna2-6k16t-v1` pins `51ca09c` with `deterministic_world: true`. Evaluation environments stay on `24e1fc9`.

## Surprises & Discoveries

- Observation: r3 failed at its first admission (`actual conditioning view exceeds qualified backend context capacity`): the TRL engine's capacity is `max_prompt_length + max_completion_length` (20480 + 6144) while vLLM allowed 28672. r4 (prompt 22528) ran collection 1 with goal credit on 46 of 47 full successes (r2: 27 of 80), harm debits in 29 episodes, 5630/872 decided/abstained findings, 12 of 160 episodes truncated, clipping 0.0003-0.008 on updates 2-4, no episode errors; it then ran out of memory in collection 2's optimizer update (87 GiB allocated by the trainer) because 28672-token turns made per-turn scoring too large. r5 keeps 6144-token replies within the 24576-token context r2/r10 trained in (settings revision 7, prompt 18432; inference `...-6k-t08-fp16@2`) and pins environments `24e1fc9` (quarterly_termination_queue guard reads `changed_fields` as a sequence).

- Observation: r2 (`manifest-steps-26-sampo-100-luna2-20261006-r2`) ran two collections cleanly: curriculum-chosen candidates, learning-signal retention (the 4 dropped groups had the lowest signal), per-turn ratios (`policy_ratio_segments` 170-253 turns), clipping active (0.0005-0.005 on updates 2-4), sampler correction 1.0 unclamped, harm debits applied, no assessment/credit/trace errors in 480 episodes. Two defects: truncation 17-21% (15 of 23 in collection 1 were single replies cut at 4096 tokens while still reasoning) and goal credit on only 27 of 80 fully successful episodes. The latter is the training toolset capturing calls as `{\"args\": [], \"kwargs\": {...}}` with None defaults, which manifest evidence cannot read; fixed in environments `d766982` (rescoring corrected r2 captures credits the previously uncredited tasks, except `simple.email_sf_log_task`, whose occurrence counts abstain for another reason).
- Decision: r3 uses 6144-token replies, 18432 output tokens per episode and a 28672-token vLLM context (settings revision 5, environment `automationbench-manifest-steps-luna2-6k16t-v1`, inference `...-c180-6k-t08-fp16@1`), at the user's request when truncation is high; KV cache stays 48 GiB.

- Observation: the first submission (`manifest-steps-26-sampo-100-luna2-20261006-r1`) rejected every rollout group: `ContractError: the configured Trackio build cannot store environment metrics; install Trackio 0.31.5.post14.dev33 or newer`. The merged Observatory branch requires Trackio >= dev33 for traces that carry environment metrics, but every published job-kind image ships dev32. Three active rounds produced 60 groups, all `terminal: failed` with no traces; the run was cancelled before its fifth round. Settings did take effect (recorded config: trainer fp16, vLLM float16, task mix `automationbench-manifest-luna-v2` with 103 tasks, `turn-rows`, `retain: learning_signal`; curriculum metrics logged; device memory at the collection step).
- Observation: completing the dev35 pin in the job-kind locks (`fbe677bf`) is not enough on its own: `published.toml` binds each kind image to its lock digest, so packing refuses edited locks until the kind images are republished. `posttrain-release images plan` lists six kind images to rebuild (supervised, TRL, veRL, eval, serve, transform); the TRL kind is republished first to unblock this run.

- Observation: r10's checkpoint-100 held-out evaluation could not run on the suites' pinned environments revision `0bad6187`, which requires Verifiers `e6a3d9bb` while the current evaluation job pins `58df1306`. With the user's choice, both matched suites were re-pinned to `089b67d` on branch `wip/r10-heldout-eval-089b67d` (`ef3fd158`); 15 benchmark-core files changed in between, so scores are not strictly comparable with earlier systems.
- Observation: the first re-pinned evaluation (`heldout-matched-64k-v4-r10-c100-089b67d-20261006-b`) scored 0 on all 60 rollouts because every tool call returned `linked tool execution requires capture enabled`: since `427e802` AutomationBench executes tools only for tasks with `capture_actions: true`, which the held-out task config lacked. Fixed in `081e1288` (both re-pinned suites capture actions). The two evaluations rerun after this training run frees the workstation.

The defects that motivated this run are recorded in `docs/plan/automationbench-simple-manifests.md` and `docs/plan/vortex-resolved-engine.md`.

## Decision Log

- Decision: make the simulator deterministic per call instead of masking volatile values in Posttrain's anchor key.
  Rationale: on six r5 groups (286 turns) 57.3% of turns were singleton anchors. Tool results carried wall-clock timestamps and OS-random ids, so equal calls on equal worlds looked like different states. Masking timestamps and long hex ids in the key gave 45.8%, but it also merged turns after genuinely different actions (different email bodies sent, distinguished only by the new message id). Seeding each call by the world before it and the call keeps the key exact: replaying the same groups' recorded calls gives 56.6% on the wall clock and 50.0% deterministic. The world clock starts at the task's `meta.current_time` (UTC midnight of the setup day when absent, 503 of 800 tasks) and advances one second per call that changed the world.
  Date/Author: 2026-10-06, Claude.

- Decision: retarget the existing 2.6B work package instead of adding a parallel one.
  Rationale: the user asked to resubmit the experiment, not resume r10; r10's run record keeps its own resolved configuration.
  Date/Author: 2026-10-06, Claude.
- Decision: keep `advantage_normalization: mean` and the cont100 schedule.
  Rationale: the run changes only what was broken (step credit, objective) and the task mix, so its comparison with r10 and the older SAMPO system stays interpretable.
  Date/Author: 2026-10-06, Claude.

## Outcomes & Retrospective

(2026-10-06 ~11:50Z) `manifest-steps-26-sampo-100-luna2-20261006-r5` completed two collections (8 updates) cleanly and continues. Verified from the run itself: recorded config fp16 trainer and vLLM, task mix luna-v2 (103 tasks), `turn-rows`, `retain: learning_signal`, yield-first curriculum; both collections chose candidates through the curriculum (20 unique tasks, one active round) and dropped only the lowest-signal groups; per-turn ratios over 182-296 turns per update with clipping 0.0008-0.0039 on non-first updates; sampler correction mean 1.0, never clamped; gradient norms 0.007-0.032; collection-grain and device-memory telemetry at the collection step; Trackio dev35 accepts traces with environment metrics. Rewards: goal credit on 257 of 403 episodes and 76 of 91 full successes (r2: 27 of 80), harm debits in 83; no assessment, credit or trace errors; 14,377 decided findings. Truncation 10-17% (r2 17-21%) with 6144-token replies in the 24576-token context. Remaining reward gaps are task-specific manifest coverage: `simple.email_sf_log_task` (1 of 8 successes credited), `simple.asana_dark_mode_from_email` (4 of 7), `hr.quarterly_termination_queue` (5 of 10); abstentions otherwise are `obligation_not_required`.

## Context and Orientation

Workstation: `carbonteq-ai-workstation.lan` (RTX PRO 6000, dstack), one job at a time. Launch from a clean worktree with its own CLI. Metrics are readable from Trackio directly (`posttrain_tracking_trackio.adapter.TrackioDataSource("posttrain-lab", server_url="https://trackio.lan")` with `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`).

## Concrete Steps

From `/home/hammad/projects/worktrees/rl-perf/apps/lab` on a clean tree:

    UV_HTTP_TIMEOUT=300 ../../.venv/bin/posttrain job run .posttrain/work_packages/lfm26_automationbench_manifest_steps_sampo_100_g16x8_ws.yaml --run-id manifest-steps-26-sampo-100-luna2-20261006-r1

Evaluating r5 after it finishes (one job at a time on the workstation, from a clean worktree; replace STEP with 100, 200, 300 or 400). Run the base model the same way without `--model-from-run` for the baseline:

    ../../.venv/bin/posttrain job run .posttrain/work_packages/lfm26_automationbench_luna2_heldout_eval_6k16t_t05.yaml --job evaluate --provider dstack --model-from-run manifest-steps-26-sampo-100-luna2-20261006-r5 --model-checkpoint-step STEP --run-id luna2-heldout-6k16t-t05-r5-cSTEP-20261007
    ../../.venv/bin/posttrain job run .posttrain/work_packages/lfm26_automationbench_heldout_matched_eval_64k_v4.yaml --job evaluate --provider dstack --model-from-run manifest-steps-26-sampo-100-luna2-20261006-r5 --model-checkpoint-step STEP --run-id heldout-matched-64k-v4-r5-cSTEP-20261007
    ../../.venv/bin/posttrain job run .posttrain/work_packages/lfm26_automationbench_heldout_matched_eval_64k_v4_t05.yaml --job evaluate --provider dstack --model-from-run manifest-steps-26-sampo-100-luna2-20261006-r5 --model-checkpoint-step STEP --run-id heldout-matched-64k-v4-t05-r5-cSTEP-20261007

r10's checkpoint 100 and the base model rerun on the re-pinned v4 suites so all three systems compare on environments `24e1fc9`.

## Validation and Acceptance

After the first collection (4 updates): `train/rl/policy_ratio_segments` counts turns (not episodes); `train/rl/policy_log_ratio_abs_mean` is 0 on the collection's first update and non-zero on updates 2-4; per-turn `manifest_goal_credit` is non-zero on most fully successful episodes (r7 local: 26 of 29); `train/rl/turn_credit_share` is reported; gradients are finite. At checkpoint 100, the two matched held-out suites are run with the same packages as r10's checkpoint 100.

## Idempotence and Recovery

A failed submission is resubmitted with a new run id. The evaluation chain is `/tmp/claude-1000/-home-hammad-projects-rl/06f4fe1b-c6e2-45ea-9bb2-d0b2a04f9815/scratchpad/runs/r10_evals.sh` (log `r10-evals.log`); rerun one suite with the `job run` line it prints.

## Interfaces and Dependencies

Environments revision `089b67d` (`carbonteq-ai/verifiers-environments`, branch `wip/automationbench-simple-manifests-2026-10-06`).
