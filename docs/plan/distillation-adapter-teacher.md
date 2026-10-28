# Score on-policy distillation with a materialized adapter teacher

This ExecPlan is a living document maintained under `docs/templates/PLAN.md`.

## Purpose / Big Picture

TRL distillation accepts only a Hugging Face teacher. After `distillation-teacher-materialization.md`, a stored-adapter teacher reaches the backend as a local PEFT adapter directory, and the backend rejects it. After this change the backend scores with the adapter's pinned base model plus the adapter, loaded in the training process, so a student can be distilled from an earlier fine-tune. Observable result: a distillation request whose teacher is a materialized adapter and whose teacher scoring uses the `transformers` binding builds a trainer whose teacher is the base with the adapter applied.

## Progress

- [x] (2026-10-27) Confirmed `main` raises "TRL distillation currently requires a Hugging Face teacher model" for any non-Hub teacher.
- [x] `_teacher_hub_ref` and `_colocated_teacher_model` in `packages/train/src/posttrain/train/backends/trl/distillation.py`.
- [x] Six tests in `packages/train/tests/test_api.py` using fakes for torch, transformers and peft.
- [ ] Run once on a GPU with a real adapter. Not done.

## Surprises & Discoveries

- Observation: a PEFT adapter directory does not contain its base weights, so the teacher cannot be given to TRL as a repo id; the base must be loaded and the adapter applied explicitly.
  Evidence: `request.teacher.base` carries the pinned base `HubModelRef`; the artifact is only the adapter.

## Decision Log

- Decision: support only materialized PEFT adapters with in-process `transformers` scoring. Full-weights teachers and the vLLM scoring server are rejected with clear errors.
  Rationale: a server cannot load a local adapter without further wiring, and a full-weights artifact needs its own loading path; neither is needed now.
  Date/Author: 2026-10-27, owayys (confirmed by the requester).
- Decision: pin the base by the request's revision.
  Rationale: an adapter is only valid against the base it was trained on.
  Date/Author: 2026-10-27, owayys.

## Outcomes & Retrospective

Behavior is covered through fakes. Not verified: a real model on GPU, memory with base plus adapter resident, or LoRA with `modules_to_save`.

## Context and Orientation

`run_distillation` builds an `IWOPDTrainer` (TRL experimental) with a teacher given as a repo id string, or none for server scoring. `LocalArtifactRef` (`posttrain.common`) points at a directory. `request.teacher.form` is `adapter` or `peft-adapter` for adapters. `_distillation_arguments` builds the trainer arguments.

## Plan of Work

`_teacher_hub_ref` returns the artifact when it is a Hub ref, `request.teacher.base` when the artifact is a local adapter, and otherwise raises. `_colocated_teacher_model` returns the repo id for a Hub teacher; for an adapter it loads the base with `AutoModelForCausalLM.from_pretrained` at the pinned revision and dtype, then `PeftModel.from_pretrained(model, path, is_trainable=False).eval()`, raising on a missing directory. `run_distillation` requires a `transformers` teacher-score binding for local artifacts and passes `teacher_model_init_kwargs` only for a Hub teacher.

## Concrete Steps

From the repository root:

    UV_SYSTEM_CERTS=1 uv run --frozen --package posttrain-train pytest packages/train/tests/test_api.py -q

Three tests in that file fail on `main` when torch and accelerate are not installed; they are unrelated.

## Validation and Acceptance

`test_distillation_scores_with_the_pinned_base_and_the_materialized_teacher_adapter` asserts base then adapter loading; `test_distillation_scores_with_a_hub_teacher_by_repo_id` asserts the Hub path is unchanged; `test_distillation_rejects_a_teacher_adapter_that_is_not_on_disk`, `test_distillation_rejects_materialized_teacher_weights` and `test_distillation_requires_in_process_scoring_for_a_teacher_adapter` assert the errors; `test_distillation_arguments_pin_the_base_revision_of_a_teacher_adapter` asserts the revision. The release gate is one real run on a GPU with a stored adapter, to be recorded here.

## Idempotence and Recovery

Loading is read-only. A failed load raises before training starts.

## Artifacts and Notes

Depends on `distillation-teacher-materialization.md`.

## Interfaces and Dependencies

No public type changes. Requires `peft` in the training image, as LoRA training already does.

Revision note (2026-10-27): Initial plan.
