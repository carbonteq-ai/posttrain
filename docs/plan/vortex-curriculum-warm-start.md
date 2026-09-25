# Warm-start a new VORTEX run from a finished run's curriculum evidence

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

VORTEX runs (OLMo 3 GRPO with active sampling and an adaptive curriculum) learn which AutomationBench tasks are informative: they record per-task reward history and yield statistics. `posttrain job run --resume-from-run` already restores that evidence, because the curriculum snapshot is stored inside every recovery checkpoint. A **new** run starts cold and spends its early steps rediscovering the same evidence. A new run is, for example, one started with `--model-from-run` from a checkpoint's adapter, or one with changed training settings.

After this change, `posttrain job run WORK_PACKAGE --curriculum-from-run RUN_ID` binds the source run's published `adaptive-curriculum-state` artifact. The new run's curriculum starts with that evidence while its own counters start fresh. The run emits `adaptive_curriculum_started` with `warm_started: true` and the source identity, so the effect is visible in Trackio.

## Progress

- [x] (2026-09-25) Traced the existing state: the snapshot (`adaptive-curriculum-state.json`) is written into each recovery checkpoint and restored only on resume. Each run also publishes a final `adaptive-curriculum-state` artifact (the state directory holding the snapshot and journal), which nothing reads.
- [x] (2026-09-25) Controller: `AdaptiveCurriculumController(..., warm_start_state=...)` keeps task evidence, resets per-run counters, re-bases yield-first ages and keeps least-recently-used order. Resume and warm start are mutually exclusive.
- [x] (2026-09-25) Runtime: `AdaptiveCurriculumRuntime(warm_start_state_dir=...)` reads `adaptive-curriculum-state.json` from the bound directory and reports `warm_started` on `adaptive_curriculum_started`.
- [x] (2026-09-25) Checkpoint curriculum view. Every TRL checkpoint save that carries a controller snapshot also publishes `training/<model>/<technique>/checkpoint-<step>/curriculum` (kind `adaptive-curriculum-state`, just the snapshot, with `checkpoint_step`). It uses the same `checkpoint_steps` cadence and the same `step-<n>/` workspace lifecycle as the model view. Interrupted runs publish it too, because the curriculum writer runs on both periodic and interrupted saves.
- [x] (2026-09-25) Request and job wiring: `GRPORequest.curriculum_from`, filled from the `curriculum_state` input artifact. Materialization is name-agnostic, and the launch environment forwards the resolved-input record.
- [x] (2026-09-25) CLI flags `--curriculum-from-run RUN_ID [--curriculum-checkpoint-step N]`. Without a step they select the run's final state; with a step, the curriculum view at that step, falling back to the recovery checkpoint at that step for runs recorded before views existed. They are rejected with `--resume-from-run`.
- [x] (2026-09-25) Canonical API amendment in `docs/post-training/05-apis.md`.
- [x] (2026-09-25) Tests: 51 targeted tests (curriculum, checkpoint view, CLI selection and binding). Train, jobs, execution, work and CLI suites: 950 passed. The remaining failures are pre-existing: the worker-client loopback test, which needs vLLM locally, and a GRPO plan test broken only by uncommitted advisor edits in the main checkout; it passes on committed code. Ruff, pyright and lint-imports are clean.
- [ ] Live qualification: warm-start a short run from `lfm26-vortex-v5-yield-first-64-20260925-r4` (for example at step 20, through the recovery-checkpoint fallback) and confirm `warm_started: true` and a non-cold first decision.

## Surprises & Discoveries

- `_last_selected` holds a candidate counter, not an optimizer step. `-1` means "never selected", and the smallest value wins least-recently-used tie-breaks. Re-basing must keep never-selected tasks as the oldest.
- History evidence records carry a `step`, but selection windows history by count, not by step.

## Decision Log

- **Decision:** The yield-first policy is the design target; the quota policy is supported but secondary.
  **Rationale:** User direction. Yield-first is the production VORTEX curriculum.
  **Date/Author:** 2026-09-25, user.
- **Decision:** Publish a small curriculum view with every checkpoint, governed by the standard checkpoint settings, with no separate knobs.
  **Rationale:** User direction. Warm start must also work from mid-run steps and from failed or cancelled runs, which never publish a final state.
  **Date/Author:** 2026-09-25, user and Claude.

- **Decision:** By default warm start reads the source run's final published curriculum state; `--curriculum-checkpoint-step` selects the state at a checkpoint.
  **Rationale:** The final state carries the full evidence, and checkpoint views cover interrupted runs and earlier steps.
  **Date/Author:** 2026-09-25, user and Claude.
- **Decision:** Require identical task inventory, curriculum settings and group size, as resume does.
  **Rationale:** Evidence keyed to other tasks, or aged with another half-life, would be silently misread. A mismatch fails at startup.
  **Date/Author:** 2026-09-25, Claude.
- **Decision:** Re-base time so that the source run's last active step becomes step 0 of the new run.
  **Rationale:** Yield statistics are first aged to the source's last step, then stamped at 0. At new step k they decay exactly as if k more steps had passed, and ages stay continuous.
  **Date/Author:** 2026-09-25, Claude.
- **Decision:** `--curriculum-from-run` cannot be combined with `--resume-from-run`, but can be combined with `--model-from-run`.
  **Rationale:** Resume already restores the checkpoint's own curriculum. Model-from-run is the main case that starts cold.
  **Date/Author:** 2026-09-25, Claude.

## Outcomes & Retrospective

Not started.

## Context and Orientation

- `packages/train/src/posttrain/train/adaptive_curriculum.py`: `AdaptiveCurriculumController`, including `state()`, `_restore()` and `_age_yield_moments()`.
- `packages/train/src/posttrain/train/backends/trl/policy_curriculum.py`: `AdaptiveCurriculumRuntime` loads the snapshot, writes checkpoint snapshots and `save_final_state()`.
- `packages/train/src/posttrain/train/backends/trl/policy_optimization.py`: builds the runtime and publishes the `adaptive-curriculum-state` artifact.
- `packages/train/src/posttrain/train/requests.py`: `GRPORequest`.
- `packages/jobs/src/posttrain/jobs/definitions.py`: maps input artifacts onto requests (`_recovery_checkpoint`).
- `apps/cli/src/posttrain_cli/commands/job.py` and `commands/work_package.py`: the job run options, with checkpoint selection in `_select_checkpoint_output`.
- `apps/cli/src/posttrain_cli/execution_planning.py`: `with_recovery_checkpoint` and `with_model_checkpoint`.
- `packages/execution/src/posttrain/execution/contracts.py`: the launch environment, including its resolved-input allow-list.

## Validation and Acceptance

- Unit tests prove a warm-started controller keeps history, first evidence, seen tasks and yield statistics. They also check that it resets decision and candidate counters, that ages continue (a moment aged k steps in the new run equals one aged S+k steps in the source), and that never-selected tasks stay least recently used.
- Unit tests prove an inventory, settings or group-size mismatch fails.
- CLI tests prove selection, the mutual-exclusion rule and the planned `curriculum_state` input.
- Run `uv run pytest` for the train, jobs, execution and CLI tests, plus `uv run ruff check .`, `uv run pyright` on changed files and `uv run lint-imports`.

## Idempotence and Recovery

The new input is opt-in. Runs without `--curriculum-from-run` are unchanged, and resume behaviour is unchanged.
