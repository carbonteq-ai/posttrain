# AutomationBench-calibrated curriculum replay

This ExecPlan is a living document governed by `docs/templates/PLAN.md`.

## Purpose / Big Picture

Estimate whether a common, evidence-calibrated probability of a usable next rollout group can reduce candidate/refill waste without prescribing a fixed novelty quota. A reader can export compact, non-prompt trace facts from retained AutomationBench runs and run a reproducible, 100-step bootstrap comparison. The production objective is held-out learning per GPU-hour, not maximizing the selector's predicted variance; this replay screens a waste mechanism and cannot measure counterfactual learning.

## Progress

- [x] (2026-09-23) Located retained v2 and v3 Trackio trace metadata and confirmed task, class, step, batch, reward, and token fields.
- [x] (2026-09-23) Exported a compact, checked group-level snapshot from seven runs without prompts or tool transcripts: 8,128 traces, 2,035 groups, seven incomplete groups excluded from fitting.
- [x] (2026-09-23) Implement paired policy simulation, prequential checks, and cross-run calibration.
- [x] (2026-09-23) Ran five tests and lint, two 20-seed 100-step sensitivity replays, and recorded outcomes and limitations in `docs/research/proposals/simulations/automationbench-replay-analysis.md`.

## Surprises & Discoveries

- `include_payload=False` on the remote Trackio trace API still returned a large Verifiers payload. The exporter must project safe numeric fields in memory and must never write raw responses.
- The v3 run has 648 traces and v2 has 1,236 at inspection time; this is a live source, so the export records its own counts and time.
- The export later saw 680 v3 traces, seven incomplete four-rollout groups, and no missing required fields. Incomplete groups were excluded from fitting rather than labeled non-useful.
- Naively maximizing useful-probability per total token preferred short, unhelpful trajectories and collapsed 100-step completion on v2/v3. A completion-risk guard recovered most steps but did not meet its assumed 99% success target, so token-cost optimization is not production-qualified.

## Decision Log

- Decision: Keep this research-only under `docs/research/proposals/simulations/`; do not change the frozen product baseline or the live selector. Rationale: observational traces cannot establish policy effects on training. Date/Author: 2026-09-23, Codex.
- Decision: Compare policies on a paired, stationary empirical bootstrap and report observed support and sensitivity. Rationale: tasks a policy did not choose lack counterfactual outcomes. Date/Author: 2026-09-23, Codex.
- Decision: Keep seven retained runs separate and include source-to-target calibration checks. Rationale: the user requested evidence from multiple runs, but pooling different optimizer/model states would hide transfer failures. Date/Author: 2026-09-23, Codex.
- Decision: Include both posterior-mean and posterior-sampling selectors. Rationale: posterior sampling supplies uncertainty-driven rechecks without a fixed discovery budget. Date/Author: 2026-09-23, Codex.
- Decision: Keep cost-aware and risk-guarded selectors as ablations rather than the recommended production policy. Rationale: their measured token-work gains trade against step completion, and token count is not GPU seconds. Date/Author: 2026-09-23, Codex.
- Decision: Treat next-group variance as a waste-screening proxy, not a policy objective. Rationale: the user clarified that efficiency should come from eliminating unusable work while preserving learning; a high-variance or cheap group is not automatically a good training example. Date/Author: 2026-09-23, Codex.

## Outcomes & Retrospective

The simulator and compact seven-run evidence snapshot are complete. Chronological and cross-run Brier checks indicate task/class evidence predicts the chance of a usable group; stationary 100-step replay suggests a way to reduce candidate-group waste. The naive cost ratio fails, while a risk guard remains unqualified. Production selection and live jobs were not changed. The next gate is prospective shadow calibration followed by a same-checkpoint A/B measuring held-out learning per GPU-hour and GPU-seconds spent on non-admitted work.

## Context and Orientation

`packages/train/src/posttrain/train/adaptive_curriculum.py` is the production selector and is not edited here. `docs/research/proposals/simulations/controller_policy_experiment.py` is an earlier synthetic learner experiment. Trackio stores Verifiers rollout traces for `train.grpo-lfm26-vortex-v2-agentic-20260923-r1` and `train.grpo-lfm26-vortex-v3-lr2e4-20260923-r1` in project `posttrain-lab`. A useful group means its rollout rewards vary, so the group can contribute a nonconstant group-relative training signal. A candidate group is one task sampled several times before admission. A recheck is a repeat of a previously seen task.

## Plan of Work

Create `docs/research/proposals/simulations/automationbench_replay.py` with an explicit `export` command that reads seven named runs, projects only task identity, class, step, group identity, reward and token counts, checks group grain and missingness, and writes a compact JSON snapshot. Add a `simulate` command that reads this snapshot without network access. Compare uniform sampling with a hierarchical posterior predictor: unseen tasks inherit class evidence, seen tasks blend their own evidence with the class prior, and both compete on the same predicted useful-group scale. Add naive cost-aware and completion-guarded cost-aware ablations. Use paired random seeds and a 100-step horizon. Add tests under `docs/research/proposals/simulations/tests/` for grouping, no leakage in a temporal calibration split, and policy accounting.

## Concrete Steps

From `/home/hammad/projects/rl`, run `uv run python docs/research/proposals/simulations/automationbench_replay.py export --output docs/research/proposals/simulations/automationbench_groups.json`, then `uv run python docs/research/proposals/simulations/automationbench_replay.py simulate --input docs/research/proposals/simulations/automationbench_groups.json --output docs/research/proposals/simulations/automationbench_replay_results.json`. The first command is read-only against Trackio and writes only the projected local artifact. The second is fully offline and should print policy-level yield and discovery/reuse shares. Add `--exclude-truncated` and use a separate output file for the sensitivity.

## Validation and Acceptance

Run `uv run pytest docs/research/proposals/simulations/tests/test_automationbench_replay.py`, `uv run ruff check docs/research/proposals/simulations/automationbench_replay.py`, and `git diff --check`. The export must report zero raw prompts/tool fields and reconcile trace counts with its manifest. The simulator must report denominators, class/task support, calibration, and explicit noncausal limitations. A subsequent production change requires a separate plan and live qualification.

## Idempotence and Recovery

Export overwrites only the named generated snapshot; repeat it to refresh live data, preserving the prior file separately if comparison matters. Simulation is deterministic for a fixed snapshot and seeds. Failure never mutates a remote run.

## Artifacts and Notes

The compact snapshot contains no prompt, tool response, or model message. Its source run names and capture time are retained for provenance.

## Interfaces and Dependencies

The export uses the already-pinned `trackio.Api`; simulation uses Python standard library only. The only external service is the read-only Trackio endpoint `https://trackio.carbonteq.com`. No credentials are stored in outputs.

## Revision note

2026-09-23: Initial research-only plan after confirming trace metadata and avoiding live-policy mutation. Expanded to seven runs and cross-run calibration after the user requested multiple-run collection. Final revision records incomplete-group handling, the cost-ranking failure, guard ablation, and the prospective qualification gate. Reframed the objective around learning preserved per GPU-hour after user clarification; the replay remains a waste proxy only.
