# Export evidence-backed AutomationBench task metadata

This living plan follows `docs/templates/PLAN.md`. Update Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective as implementation proceeds.

## Purpose / Big Picture

Allow Posttrain to select and report AutomationBench tasks by workflow, required capabilities and guard patterns, alongside domain. Keep measured model success and reference budgets tied to the exact task, scorer and run that produced them. Metadata is host-side information; it must not add solutions or reference traces to student prompts.

## Progress

- [x] (2026-10-04 UTC) Started a parallel workstream while six category workers continue manifest validation. Inspected native task data, evaluation inventory and training facet projection.
- [x] (2026-10-04 UTC) Preserve multi-label string facets through training dataset and rollout observations. Focused tests: 11 passed; existing native bridge module skipped because its runtime dependency is absent. Scoped Ruff passed and all nine import-boundary contracts passed.
- [x] (2026-10-05) Implement environment-owned versioned metadata export and opt-in native task loading. Preserve unused task serialization; attach export identity when metadata is selected.
- [x] (2026-10-05) Export the current selected 120 development references and validate attachment to all 120 corresponding frozen tasks. Retain measured outcomes and usage; leave all 360 semantic classification dimensions explicitly not reviewed. This is the selected cohort, not the complete benchmark inventory.
- [ ] Demonstrate environment loading, evaluation filtering and training observation round trips without changing prompts or category sampling.

## Surprises & Discoveries

Evaluation inventory already expands lists into independent facet values. Training `_task_facet_values` rejects non-scalar values and `_record_task_facets` silently drops them. Adaptive curriculum requires a single non-empty string class; retaining multiple labels in observations must not implicitly change that sampling rule.

The selected reference episodes contain SDK-reported output usage and native tool-server dispatch receipts, but none contains original sampled-token alignment. All 120 therefore expose observed reported output counts and tool dispatch counts while exact sampled output remains unavailable. Existing review-index `family` values are collection groupings (including broad values such as `hr.all-development`), not an authored workflow/capability/guard taxonomy. They remain source-bound reference-family observations.

## Decision Log

The environment owns AutomationBench label meaning and source bindings. Framework training only transports declared metadata. Keep workflow/capability/guard labels separate from measurements and qualification. Original Luna score 1 is an observed scorer outcome, not proof that redesigned guards pass. A single reference attempt is not a population success-rate estimate. Budget measurements are observations, not automatically enforced limits. No model calls, commits, publication or dependency pin changes are authorized for this workstream.

Optional loading requires both an artifact path and expected export digest. The digest binds source references and classification content, and loading rechecks referenced artifact bytes. Task attachment separately checks all base task data except positional `idx` and the reporting fields themselves. The export digest and classification statuses then enter native task data, changing its effective hash without changing its prompt. Freshly generated worlds with changed internal IDs are rejected rather than joined to old evidence by task name.

## Outcomes & Retrospective

Training transport now preserves deterministic string lists and unchanged scalar values, and a dataset test confirms labels do not enter the prompt. The environment candidate exports source-bound metadata for the selected 120 reference tasks; all 120 frozen-task attachments preserve prompts and change effective identity. The 10 metadata tests, six existing inventory/manifest tests and 25 existing environment tests pass (41 total); scoped Ruff and Pyright pass. Runtime selection, production adoption, authored semantic classifications and complete-benchmark coverage remain pending. Manifest workers retain exclusive ownership of their task draft/review files; this work did not modify them.

## Context and Orientation

The framework checkout is `/home/hammad/projects/rl`. The external environment candidate is `/home/hammad/projects/verifiers-environments-reward-candidate-20261003`, whose package is `environments/automationbench_v1`. Native assessment candidate is `/home/hammad/projects/verifiers-credit-candidate-20261003`. Resolve branch, commit and dirty paths before environment edits; candidates are not published dependencies.

`src/automationbench_v1/taskset.py` in the environment defines `AutomationBenchData`, currently containing domain, task name, initial state, assertions and tools. A facet is a declared task field exposed as a selectable/reportable dimension. Framework `packages/eval/src/posttrain/eval/backends/verifiers/inventory.py` reads declared fields and supports multiple values. `packages/train/src/posttrain/train/integrations/verifiers.py` projects task fields into dataset metadata and rollout observations. `packages/train/src/posttrain/train/adaptive_curriculum_runtime.py` continues to require one scalar sampling class.

Existing reference selection is `docs/research/verifiers-assessment-qualification/reward-candidate/luna-top20-per-nonsimple-category-selection.json`. It contains 120 development tasks selected using original Luna rewards, including partial-score fallbacks. Its source index and episode hashes are provenance inputs, not universal task difficulty labels. Canonical contracts are `docs/post-training/02-primitives.md`, `05-apis.md` and `06-observation-and-lineage.md`. Preserving multi-valued observations uses existing facet meaning; any new selection or allocation semantics require a canonical amendment first.

## Plan of Work

First extend training's declared facet transport to accept finite scalar values and sequences of non-empty strings, preserving scalar behavior. Normalize string lists deterministically and reject nested objects or mixed numeric lists. Preserve lists through rollout observation recovery. Add a focused standalone test module under `packages/train/tests`.

Then inspect calibration source indexes and episode token/tool counters before fixing the export schema. Add an environment-owned module and tests, separate from manifest workers' folders. The export must carry schema version, task identity/content digest, source revision and source artifact digests. Keep authored classifications separate from run-derived observations. Each model result needs model/run identity, scorer definition, attempt counts and measured outcomes. Each budget observation needs tokenizer/counting convention and truncation/completion status. Missing measurements remain absent with explicit reasons; never infer output tokens from character counts or claim an optimal tool budget from one successful trace.

Load the export through explicit environment configuration and verify task/content hashes before attaching declared fields. Keep heavy per-run evidence in the export artifact rather than flattening arbitrary nested data into facet strings. Use simple categorical fields for selection and retain evidence provenance separately. Classification rules must be inspectable and evidence-backed; avoid task-name substring guesses. Existing scalar domain curriculum remains unchanged.

## Concrete Steps

From `/home/hammad/projects/rl`, run:

    uv run pytest packages/train/tests/test_verifiers_task_facets.py
    uv run ruff check packages/train/src/posttrain/train/integrations/verifiers.py packages/train/tests/test_verifiers_task_facets.py
    uv run lint-imports
    git diff --check

Before external edits inspect `git status --short` and `git rev-parse HEAD` in the candidate checkout. For environment tests use package cwd and the existing runtime:

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src:tests .venv/bin/python -m pytest <new metadata tests>

Implemented environment checks (package cwd):

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src:tests .venv/bin/python -m pytest tests/test_task_metadata.py tests/test_calibration_inventory.py tests/test_calibration_manifest.py -q
    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src:tests .venv/bin/python -m pytest tests/test_environment.py -q
    .venv/bin/ruff check src/automationbench_v1/task_metadata.py src/automationbench_v1/taskset.py tests/test_task_metadata.py
    .venv/bin/ruff format --check src/automationbench_v1/task_metadata.py src/automationbench_v1/taskset.py tests/test_task_metadata.py
    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src:tests .venv/bin/pyright --pythonpath .venv/bin/python src/automationbench_v1/task_metadata.py src/automationbench_v1/taskset.py tests/test_task_metadata.py

Update this plan with exact export API, configuration field and integration test commands after inspecting source data. Do not sync or mutate workers' virtual environments.

## Validation and Acceptance

A task declaring capabilities `policy_retrieval` and `arithmetic` retains both labels in training dataset metadata and recovered rollout observations. Existing domain-only tasks behave identically. Invalid nested metadata is rejected at declared facet extraction. A curriculum configured on a list-valued class still fails explicitly. Evaluation filtering by either capability selects the same task once. Export regeneration from identical source bytes is deterministic; mismatched task/source hashes fail before adoption. Changing metadata cannot change the rendered prompt. Model outcomes from different scorers or sampling configurations are not silently pooled.

## Idempotence and Recovery

Keep metadata loading optional and additive. Write exports atomically and validate before replacement. Retry deterministic export without overwriting source evidence or manifests. Preserve all unrelated dirty changes. Publication and immutable pin adoption are later explicit gates; tests in a dirty candidate do not establish release qualification.

## Artifacts and Notes

The initial delegate launch was blocked by the seven-agent concurrency limit. After Operations completed its assigned work, the existing assessment API critic was reactivated for metadata implementation. Root owns the framework training bridge and its new test module. That worker owns environment `src/automationbench_v1/task_metadata.py`, `tests/test_task_metadata.py`, narrow optional attachment changes in `taskset.py`, and further updates to this plan. Shared scoring code and task manifest folders are outside its ownership.

## Interfaces and Dependencies

The environment slice was implemented on candidate branch `wip/automationbench-reward-redesign-2026-10-04`, inspected HEAD `f7790acf30089be7909f42bd779c5d10e9a4b289`. Existing task-manifest and evaluator edits were preserved.

Current export: `.posttrain/state/verifiers-assessment-qualification/task-metadata/luna-selected120-metadata.json`, digest `8e883896996e4fefcd03c93029ddb30f5049c864a62e09bcb68c438df99d9b6c`. Its compact validation record is `docs/research/verifiers-assessment-qualification/reward-candidate/task-metadata-export-validation.json`. Both record selected-cohort scope, original-score-only outcomes, unavailable exact sampled-token counts, unassessed redesigned guards, and observation-only budgets.

Training facet transport uses existing `JsonValue` and `RolloutExample.metadata`; it must not import evaluation or an environment-specific package. The environment export uses standard JSON and task-owned identities; native episodes remain replay authority. No new tracking backend or parallel trace store is introduced.

Environment public interfaces in `automationbench_v1.task_metadata`:

- `build_reference_metadata(index_path: Path) -> TaskMetadataExport` reads a retained development reference selection/index; verifies episode hashes, native/frozen task identity, source inventory and manifest digests, and run/attempt binding; exports each attempt separately without pooling source/scorer/settings.
- `save_task_metadata(export, path)` validates source bytes and writes atomically. `load_task_metadata(path, expected_digest=...)` validates the versioned export and retained source artifacts.
- `attach_task_metadata(data, export)` validates base content and attaches `workflow`, `capabilities`, `guard_patterns`, `task_metadata_status`, and `task_metadata_digest`. Classification objects distinguish reviewed-empty from unreviewed and require explicit evidence and definition revision for reviewed labels.
- `AutomationBenchConfig.task_metadata_path` and `task_metadata_digest` select the optional loader. Both are required together. Large observations stay in the export, while only the categorical fields, status and identity enter task data.

The schema intentionally does not infer semantic labels from task names, assertion identifiers, original full score or reference trace length. Authoring reviewed source-bound classifications is a remaining follow-up, not a silently completed taxonomy.
