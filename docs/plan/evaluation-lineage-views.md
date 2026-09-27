# Show evaluations with the training runs and base models they evaluate

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

An evaluation run (job kinds `eval.*`) scores one model on one evaluation
suite. When that model is a checkpoint of a training run, the evaluation
answers the question the training run exists for: did training improve the
model on tasks it never saw? Today Observatory files each evaluation only under
its own evaluation work package, so nothing on screen connects an evaluation
to the training run and checkpoint step it scored, or to the base-model
evaluation it should be compared with.

After this change, in any project:

1. **Sidebar.** Under each training run, the evaluations of its checkpoints
   are listed ("eval · step 100 · 0.595"); each evaluation run shows which run
   and step it evaluated.
2. **Evals tab on a training run.** For each evaluation suite the run's
   checkpoints were scored on: a chart of score by checkpoint step with the
   base model's score as a reference line; a summary table of base and each
   step with the difference from base; and a per-task table (tasks × base and
   each step) with differences, sortable and filterable.
3. **Eval comparison page** (beside Compare in the primary navigation). Pick
   any two training runs that start from the same base model; the page lists
   the evaluation suites both runs were scored on, lets the reader choose a
   checkpoint of each (latest by default), and shows base, run A and run B
   side by side, overall and per task.

Observe it by opening a training run whose checkpoints were evaluated and
selecting Evals, or selecting Eval compare in the sidebar's primary
navigation.

## Progress

- [x] (2026-09-27) Confirmed the evidence exists: every evaluation run records
  `model_source.source_run_id` and `model_source.checkpoint_step` in its
  resolved settings (semantic dimensions `run.parent_run`, `run.parent_step`);
  base-model evaluations leave both empty. One SQL query over `runs` joined to
  `rollouts`, filtered by `run.job_kind=eval.*`, returns all 28 evaluation
  runs of `trackio-carbonteq` with suite, model, parent and score in about 2 s.
- [x] (2026-09-27) Service and HTTP: `ObservatoryService.evaluations()` and
  `GET /api/v1/evaluations` (0.4 s live for 28 runs, cached 60 s).
- [x] (2026-09-27) Scope agreed with the user: sidebar nesting, a per-run
  Evals tab (base against each step, overall and per task), and a comparison
  page for two runs of the same base model on their shared suites.
- [x] (2026-09-27) Service and HTTP: per-task scores, `GET /api/v1/evaluations/tasks`.
- [x] (2026-09-27) Frontend: sidebar nesting, Evals tab, Eval compare page
  (sidebar button "Evals").
- [x] (2026-09-27) Validation: 177 Observatory backend tests, 118 frontend
  tests, ruff, pyright, lint-imports and the production build pass. Live on
  `trackio-carbonteq`: the KL run's Evals tab shows base 0.591 (mean of 0.592
  and 0.589), step 100 +0.004, step 130 +0.009, step 150 −0.021, the failed
  step-100 attempt listed but not compared; the comparison page pairs it with
  the v1–v4 runs on `automationbench-heldout-matched-64k-v3`.
- [ ] Deploy to observatory.lan.

## Surprises & Discoveries

- The index is refreshed when a project is selected and whenever an Evals
  view opens, not on a timer: a second `setInterval` in `App.tsx` displaced the
  run poll that an existing test captures, and the list changes only when an
  evaluation finishes.
- `run.model` on a checkpoint evaluation is the base model's selection id
  (`models/lfm2.5-2.6b@bf16`); the checkpoint is identified only by
  `parent_run` and `parent_step`. A row's model identity is therefore
  `(model, parent_run, parent_step)`.

## Decision Log

- Decision: an evaluation suite is the evaluation's work package id.
  Rationale: the work package pins the evaluation settings, environment and
  task selection, and it is the grouping users already know. Comparing scores
  inside one suite is the only comparison the page makes. Date: 2026-09-27.
- Decision: the base-model reference for a training run on a suite is every
  evaluation in that suite with no `parent_run` whose `run.model` equals the
  model the checkpoint evaluations report. Several base runs are all shown
  (mean as the reference, each as a point), never silently reduced to one.
  Rationale: general-purpose; repeated base evaluations show run-to-run noise.
  Date: 2026-09-27.
- Decision: the score is the mean rollout reward over attempts that did not
  fail, counting truncated attempts at their recorded reward (usually 0), with
  failed and truncated counts shown beside it. Rationale: failures are
  execution errors, not model outcomes (06, trace facts); the same definition
  already heads the evaluation notes. Date: 2026-09-27.
- Decision: no change to the frozen baseline. 06 already calls for comparing a
  descendant with "parent and relevant baselines" in `qualify` and for views
  that mark missing evidence; this adds views over recorded run settings and
  creates no new lineage semantics. Date: 2026-09-27.
- Decision: failed evaluation runs stay visible but are marked and excluded
  from the reference mean and chart lines. Date: 2026-09-27.

## Outcomes & Retrospective

Implemented on branch `codex/policy-views`; not yet deployed. The per-run Evals
tab and the comparison page read the same index, so a new evaluation appears
in the sidebar, the tab and the page together once the index refreshes.

## Context and Orientation

- `apps/observatory/src/posttrain_observatory/semantic_layer/framework.py`
  defines the run dimensions used here (`run.parent_run`, `run.parent_step`,
  `run.model`, `run.work_package`, `run.job_kind`) and rollout facts.
- `ObservatoryService.query_semantics(SqlQuery(...))` in `service.py` runs
  read-only SQL inside the source (Trackio project SQL over Doris).
- `http.py` exposes the service under `/api/v1/...`; the frontend client types
  are generated from the OpenAPI schema (`npm run generate:api`).
- `frontend/src/App.tsx` renders the sidebar (work packages → runs), the run
  sections (`Section` type) and the Compare surface.

## Plan of Work

1. Models (`models.py`): `EvaluationRecord` (run key and locator, run id,
   job kind, suite, environment, model, parent run and key, parent step,
   status, started_at, attempts, score, truncated, failed) and
   `EvaluationIndex` (source id, records, score definition text).
2. Service: `evaluations(source_id)` issues one `SqlQuery` with
   `runs={"run.job_kind": "eval.*"}`, maps rows to records and run keys, and
   caches for 60 s like other project-wide reads.
3. Per-task scores: `evaluation_tasks(run keys)` issues one `SqlQuery` over
   `rollouts` for the given evaluation runs, grouped by run and task (attempts,
   score, truncated, failed).
4. HTTP: `GET /api/v1/evaluations?source_id=` and
   `GET /api/v1/evaluations/tasks?run_key=…` (repeatable).
5. Frontend: `lib/evaluations.ts` (grouping helpers: by parent run, by suite,
   base reference), `components/EvaluationViews.tsx` (the run tab and the
   comparison page), sidebar nesting in `App.tsx`, an `Evals` section shown
   for a run with evaluated checkpoints, and an `Eval compare` surface.

## Concrete Steps

From `apps/observatory`: `uv run pytest tests -q -k evaluation`, then the
frontend: `npm run generate:api && npx vitest run && npm run build` in
`apps/observatory/frontend`.

## Validation and Acceptance

- Backend test with a fake SQL source: records group correctly; base rows have
  no parent; failed runs are marked.
- Frontend tests: sidebar shows checkpoint evaluations under their training
  run; the run tab lists base and each step with its difference and compares
  tasks against base; the comparison page offers only same-model runs as run
  B and compares the chosen checkpoints overall and per task.
- Live: on `trackio-carbonteq`, the KL run shows steps 100, 130, 150 against
  base 0.589 and 0.592 on `automationbench-heldout-matched-64k-v3`.

## Idempotence and Recovery

Read-only feature; no stored state or migrations. Reverting the commit removes
it.

## Artifacts and Notes

None yet.

## Interfaces and Dependencies

No new dependencies. New public HTTP routes `GET /api/v1/evaluations` and
`GET /api/v1/evaluations/tasks`.
