# Materialize the distillation teacher's stored artifact

This ExecPlan is a living document maintained under `docs/templates/PLAN.md`.

## Purpose / Big Picture

A distillation run can name a teacher that is a stored adapter (an artifact produced by an earlier training run) instead of a Hugging Face model. Today the job materializes only the student's run-selected artifact, so a stored-adapter teacher reaches the trainer as an unresolved reference, and the work runner names every model input `model_adapter` or `model_weights`, so a teacher input would overwrite the student's. After this change the job downloads the teacher's artifact into the run the same way it does the student's, under teacher-specific input names, and binds the teacher inference to the materialized path. Observable result: the distillation operation's inputs list `teacher_adapter` beside `model_adapter`, and the request handed to the backend carries a teacher whose artifact is local.

## Progress

- [x] (2026-10-27) Reproduced on `main`: `distillation_definition` passes the raw teacher seats and `_artifact_inputs` writes only `model_*`.
- [x] Materialize the teacher with `role="teacher"` in `packages/jobs/src/posttrain/jobs/definitions.py`.
- [x] Name artifact inputs by seat role in `packages/work/src/posttrain/work/runner.py`.
- [x] Tests in `packages/jobs/tests/test_jobs.py` and `packages/work/tests/test_work.py`.
- [x] Package suites for jobs and work pass (127 passed).
- [ ] Review; the backend side (loading a materialized adapter teacher) is the separate plan `distillation-adapter-teacher.md`.

## Surprises & Discoveries

- Observation: `_materialize_selected_model` already accepts a `role` and reads `{role}_adapter` / `{role}_weights` input names, so the job side needed no new mechanism; only the distillation definition never called it for the teacher.
  Evidence: the helper's signature and its use for the student in the same definition.

## Decision Log

- Decision: the `model` and `student` seats keep the `model_` prefix; every other seat is named after its role.
  Rationale: existing single-policy jobs and recorded operations must keep their input names; only a second model seat needs a new name.
  Date/Author: 2026-10-27, owayys.
- Decision: a teacher without a stored artifact (for example a Hugging Face model) is left untouched.
  Rationale: nothing to materialize, and this keeps current distillation requests byte-identical.
  Date/Author: 2026-10-27, owayys.

## Outcomes & Retrospective

Jobs and work suites pass. Not verified: a real remote run with a stored-adapter teacher. Whether the backend accepts the materialized teacher is out of scope here (see the adapter plan).

## Context and Orientation

`distillation_definition` in `packages/jobs/src/posttrain/jobs/definitions.py` builds the on-policy distillation request from resolved seats (`model`/`student`, `teacher`, and their inference bindings). "Materializing" means resolving a stored or Trackio artifact reference to a local directory before the request is built. `_artifact_inputs` in `packages/work/src/posttrain/work/runner.py` lists the artifacts an operation consumes, keyed by input name.

## Plan of Work

In `distillation_definition`, after materializing the student, materialize the teacher with `_materialize_selected_model(context, _seat(seats, "teacher", ModelVariant), _seat(seats, "teacher_inference", InferenceBinding), role="teacher")` and pass the returned teacher and teacher inference to the request. In `_artifact_inputs`, iterate `seats.items()` and use `model` as the prefix for the `model` and `student` roles and the role name otherwise.

## Concrete Steps

From the repository root:

    UV_SYSTEM_CERTS=1 uv run --frozen --package posttrain-jobs pytest packages/jobs/tests/test_jobs.py -q
    UV_SYSTEM_CERTS=1 uv run --frozen --package posttrain-work pytest packages/work/tests/test_work.py -q

## Validation and Acceptance

`test_distillation_materializes_each_models_own_stored_adapter` materializes a student adapter and a teacher adapter and asserts the request receives each model's own path. `test_distillation_keeps_a_teacher_without_a_materialized_artifact` asserts a Hub teacher is passed through. `test_artifact_inputs_name_each_model_seat_so_a_teacher_cannot_replace_the_student` asserts `model_adapter` and `teacher_adapter` coexist; `test_artifact_inputs_keep_the_model_names_for_a_single_policy_seat` asserts existing names are unchanged. The first and third fail on `main`.

## Idempotence and Recovery

Tests are read-only and repeatable. Materialization of an artifact is keyed by its digest; a failed download is retried by rerunning the job.

## Artifacts and Notes

Stacked dependents: `distillation-adapter-teacher.md`.

## Interfaces and Dependencies

No public type changes. `_artifact_inputs` input names for non-policy seats change from `model_*` to `<role>_*`; any consumer that read a teacher's artifact under `model_*` would have been reading the student's.

Revision note (2026-10-27): Initial plan.
