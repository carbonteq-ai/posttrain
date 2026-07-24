# Make training execution contracts explicit

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Training selections currently carry shared runtime policy in an open-ended mapping, while the veRL adapter sends an unvalidated nested dictionary to a separate Python process. That makes misspelled settings silently survive catalog loading and moves correctness failures far away from the selection that caused them. After this change, shared runtime policy is a typed immutable value and the veRL launcher and worker exchange versioned, validated manifest and result values. A developer can see the boundary working by loading the catalog, building and round-tripping a veRL manifest, and observing immediate validation errors for unknown or malformed fields.

This is an internal contract hardening change. It does not amend the frozen product baseline: `docs/post-training/02-primitives.md` already distinguishes backend-neutral runtime settings from backend-native escape hatches, and `docs/post-training/05-apis.md` already assigns the binding and execution adapter contracts to `posttrain.train`.

## Progress

- [x] (2026-07-24 00:00Z) Read the repository instructions, plan template, canonical training-selection boundary, current catalog schema, launch adapter, worker, and affected tests.
- [x] (2026-07-24 00:31Z) Introduced and migrated the typed shared `TrainingRuntime` value.
- [x] (2026-07-24 00:48Z) Introduced and adopted versioned veRL launch and result contracts.
- [x] (2026-07-24 00:55Z) Added focused contract tests and migrated existing backend tests.
- [x] (2026-07-24 01:05Z) Ran package and repository validation and recorded the evidence.

## Surprises & Discoveries

- Observation: backend source provenance and TRL tuning knobs are stored under the shared `runtime` mapping in the base catalog even though consumers already treat `backend_options` as the backend-owned escape hatch.
  Evidence: `packages/catalog/src/posttrain/catalog/base/training.yaml` places `use_liger_kernel`, `logits_chunk_size`, `backend_source_revision`, and `dependency_lock_sha256` beside `global_batch_size`.

- Observation: the veRL worker already implements shared checkpoint retention and resume translation in the dirty worktree, so the typed contract must preserve those edits rather than recreate or revert them.
  Evidence: `packages/train/src/posttrain/train/backends/verl/worker.py` maps `checkpoint_limit` to both veRL retention settings and maps `resume_from` to Hydra resume settings.

- Observation: the catalog lock digest was already stale before this slice and caused the previous full-suite failure.
  Evidence: the old catalog value was `cc45cf91...`; `sha256sum uv.lock` reports `93bad681...`. Migrating source provenance to `backend_options` also refreshed the digest, and the full suite now passes.

- Observation: validating only the numeric result summary still left artifact paths as an implicit trust boundary.
  Evidence: the typed result review showed that only `metrics_file` had an output-directory containment check. The completed adapter applies the same containment rule to `model_dir` and `recovery_checkpoint`.

## Decision Log

- Decision: keep the frozen product documents unchanged.
  Rationale: the existing baseline already defines the desired distinction between normalized runtime policy and backend options; this work makes code match that contract.
  Date/Author: 2026-07-24 / Codex

- Decision: make `TrainingRuntime` closed rather than adding an `extra` mapping.
  Rationale: unknown shared settings must fail at construction or catalog decoding. Backend-specific extensibility already has an explicit home in `TrainingBinding.backend_options`.
  Date/Author: 2026-07-24 / Codex

- Decision: validate cross-process JSON with Pydantic models owned by the private veRL adapter.
  Rationale: Pydantic is already a direct `posttrain-train` dependency, supports strict versioned JSON parsing, and avoids duplicating hand-written shape checks in launcher and worker.
  Date/Author: 2026-07-24 / Codex

- Decision: remove legacy runtime aliases instead of accepting both old and normalized names.
  Rationale: accepting `nnodes`, `n_gpus_per_node`, source provenance, or trainer tuning keys in shared runtime would preserve the ambiguity this slice is intended to remove. Existing in-repository callers and catalog entries were migrated atomically.
  Date/Author: 2026-07-24 / Codex

- Decision: keep arbitrary JSON mappings only at backend-owned leaves.
  Rationale: vLLM engine settings and veRL Hydra overrides are intentionally provider-native extension points. The surrounding model, inference, environment, training, operation, and result structures are closed and versioned.
  Date/Author: 2026-07-24 / Codex

## Outcomes & Retrospective

The implementation is complete. `TrainingRuntime` is now a closed immutable public value, catalog runtime keys are schema checked, backend tuning and source provenance live under `backend_options`, and every in-repository caller uses the typed value. The veRL launcher and worker share frozen Pydantic launch and result models, reject unknown fields and mismatched operation roles, and constrain returned artifacts to the active run directory.

The change preserved the pre-existing checkpoint retention and resume work. The complete validation ladder passes with 296 tests passing and 15 expected skips. No GPU or external integration was invoked, so real veRL execution remains governed by its existing qualification procedure rather than claimed by this internal contract refactor.

## Context and Orientation

`packages/train/src/posttrain/train/bindings.py` owns `TrainingBinding`, the reusable selection that chooses a trainer backend, renderer, parameter update, execution target, parallelism, runtime policy, and backend options. `packages/train/src/posttrain/train/catalog_schema.py` validates YAML catalog entries before constructing that selection. TRL and veRL adapters consume it under `packages/train/src/posttrain/train/backends/`.

The veRL backend is process-isolated: `backends/verl/launcher.py` runs in the framework process, writes `posttrain-verl-launch.json`, and starts `backends/verl/worker.py` with the selected veRL interpreter. The worker translates the manifest to Hydra overrides, runs veRL, and writes `posttrain-result.json`. A cross-process contract here means the JSON values that both processes agree to read and write.

The existing dirty worktree contains correctness hardening and veRL checkpoint/resume changes. Those changes are in scope only where this migration must adapt their types; unrelated edits must remain intact.

## Plan of Work

First, add `TrainingRuntime` in `packages/train/src/posttrain/train/bindings.py` with explicit fields for global batch size, node and device topology, parameter and optimizer offload, and an optional process timeout. Validate positive counts and a finite positive timeout. Change `TrainingBinding.runtime` to this value, export it, and give the catalog a closed `TrainingRuntimeSchema`. Update request validation, result provenance, work snapshots, the Lab CLI, tools, catalog entries, backend adapters, and tests. Move source provenance and trainer-specific acceleration knobs to `backend_options`.

Second, create `packages/train/src/posttrain/train/backends/verl/contracts.py`. Define frozen Pydantic values for model references, inference selections, environment examples, training renderer/update/loop/parallelism/target/runtime, operation settings, the launch manifest, worker summary, and worker result. The launch manifest will carry an explicit schema version and enforce the role fields required by GRPO versus distillation. Both launcher and worker will use the same read/write methods; arbitrary backend-native engine and Hydra option leaves remain JSON mappings because they are intentionally owned by the veRL/vLLM adapter.

Third, rewrite the launcher builders to construct those values directly and rewrite the worker to use attributes rather than nested string indexing. Preserve the existing checkpoint retention and resume mappings, source-state verification, failure artifact recording, metric replay, and output-directory containment checks.

Finally, update focused tests to construct `TrainingRuntime`, assert typed manifest fields, round-trip JSON through the worker contract, reject unknown manifest fields and operation-role mismatches, and validate typed result parsing. Run focused train and catalog tests before the full validation ladder.

## Concrete Steps

All commands run from `/home/hammad/projects/rl`.

Implement and format the changes, then run:

    uv run ruff check packages/train packages/catalog apps/lab tools
    uv run pyright
    uv run lint-imports
    uv run pytest packages/train/tests apps/lab/tests/test_catalog.py apps/lab/tests/test_tracking_selection.py
    uv run pytest
    git diff --check

The focused train tests pass without requiring GPU, Docker, or network. The complete suite skips marked external integrations.

## Validation and Acceptance

Catalog decoding accepts documented runtime fields and rejects an unknown runtime key with a Pydantic `extra_forbidden` error. Direct Python construction rejects non-positive node/device/batch counts and invalid timeouts.

Building either veRL operation produces a schema-versioned `VerlLaunchManifest`. Writing and reading it preserves equality. Removing a required GRPO policy or adding an unknown nested field makes parsing fail before the worker starts. A worker result is likewise schema validated, and paths outside the current run output remain rejected.

Existing TRL behavior, veRL Hydra overrides, checkpoint retention, resume selection, failure artifact capture, and normalized GRPO metric replay remain covered by the package tests.

## Idempotence and Recovery

The edits are deterministic and safe to re-run. Tests create manifests, datasets, results, and checkpoints only under pytest temporary directories. No real training job or external service is started. If validation fails midway, inspect the focused failure and continue from this plan; do not reset the dirty worktree because it contains prior user work.

## Artifacts and Notes

The final repository-root validation produced:

    uv sync --all-packages --locked --python 3.12
    Resolved 278 packages
    Checked 87 packages

    uv run ruff check .
    All checks passed!

    uv run pyright
    0 errors, 0 warnings, 0 informations

    uv run lint-imports
    Contracts: 8 kept, 0 broken.

    uv run pytest
    296 passed, 15 skipped, 1 warning

    git diff --check
    (no output)

## Interfaces and Dependencies

`posttrain.train.TrainingRuntime` will be a frozen slotted dataclass with:

    global_batch_size: int | None
    nodes: int
    devices_per_node: int | None
    parameter_offload: bool
    optimizer_offload: bool
    timeout_seconds: float | None

`posttrain.train.backends.verl.contracts.VerlLaunchManifest` and `VerlWorkerResult` will be frozen Pydantic models with `extra="forbid"` and literal schema versions. `VerlLaunchManifest.write(path)` and `VerlLaunchManifest.read(path)` will be the only JSON serialization boundary used by the launcher and worker. `VerlWorkerResult` will provide the equivalent read/write methods for the result contract.

Pydantic remains the only added implementation mechanism and is already declared in `packages/train/pyproject.toml`; no dependency or lockfile addition is required.

Revision note (2026-07-24): Created this execution plan after inspecting the current runtime and veRL process boundary so the implementation can be resumed without chat history.

Revision note (2026-07-24): Marked implementation complete, recorded the removal of legacy aliases and artifact-path hardening, and added the final validation evidence.
