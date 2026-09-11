# Add a persistent adaptive curriculum to GRPO training

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`; `.agents/PLAN.md` is not present in this checkout.

## Purpose / Big Picture

After this change, a GRPO training profile can optionally select rollout tasks from observed learning signal instead of relying on a fixed shuffled mixture. The first qualification combines this controller with the repository's OLMo 3 profile and AutomationBench task classes. A paired 20-update vanilla GRPO run provides the fixed-mixture baseline. The comparison will show which classes and tasks were proposed, what reward variation the current student produced, and how the next allocation changed.

The controller changes exposure before generation. OLMo 3 active sampling remains a separate algorithm feature that retains useful generated groups for the optimizer. This experiment therefore compares a vanilla GRPO baseline with a combined adaptive-curriculum plus OLMo 3 arm; it does not claim to isolate the curriculum's causal effect from the OLMo 3 update rule.

## Progress

- [x] (2026-09-12) Created branch `codex/adaptive-curriculum-automationbench` without reverting the existing proposal edits.
- [x] (2026-09-12) Confirmed AutomationBench exposes `domain` as a task-class facet and that the Verifiers bridge carries it into every TRL dataset row.
- [x] (2026-09-12) Confirmed TRL forms repeated prompt groups before rollout generation and OLMo 3 performs active group retention after rewards are computed.
- [x] (2026-09-12) Chose a queued file backend with checkpoint-aligned snapshots for controller persistence.
- [x] (2026-09-12) Added the narrow canonical contract amendment and typed profile configuration.
- [x] (2026-09-12) Implemented the controller, queued file backend, TRL integration, telemetry, and recovery snapshots.
- [x] (2026-09-12) Added unit and integration tests for selection, persistence, schema decoding, trainer composition, and checkpoint ordering.
- [x] (2026-09-12) Added the adaptive AutomationBench/OLMo 3 catalog selection and 20-step work package; both work packages pass static composition validation.
- [ ] Run targeted tests, static checks, and both 20-update training jobs.
- [ ] Compare run evidence and record the result here.

## Surprises & Discoveries

- Observation: OLMo 3 active sampling reserves up to ten candidate batches and then generates only enough candidates to fill the retained batch.
  Evidence: the installed TRL `GRPOTrainer._prepare_active_sampling_inputs` slices a bounded candidate pool, scores each generated group, and retains groups whose reward standard deviation is nonzero.

- Observation: a step count is not a sound evidence horizon because a class may receive no examples during a step.
  Evidence: the rollout dataset is sampled by prompt group; evidence freshness must therefore be counted per task from completed groups rather than inferred from optimizer steps.

- Observation: the machine config contains a retired `[providers.dstack.runtime_secrets]` table while this checkout reads runtime credentials from `[services.runtime_credentials]`.
  Evidence: `posttrain machine show` rejected `providers.dstack.runtime_secrets` as an unknown field. Qualification commands use an ephemeral config copy with only the retired duplicate table removed; the user's config remains unchanged.

- Observation: repository-wide Pyright currently reports 25 pre-existing errors outside the new controller modules.
  Evidence: focused Pyright reports zero errors for the changed train modules and controller test, while the full command reports existing errors in environment runtime, Observatory HTTP tests, and older rollout tests.

## Decision Log

- Decision: Put adaptive curriculum configuration on `GRPOSettings` as an optional profile capability, independent of `GRPOSettings.algorithm`.
  Rationale: task selection is a reusable training capability, while `grpo`, `dapo`, and `olmo3` name different update rules. The requested qualification composes the capability with OLMo 3 without making it an OLMo-only feature.
  Date/Author: 2026-09-12 / Codex

- Decision: Use the environment-provided `domain` field as the AutomationBench class and `example_id` as task identity.
  Rationale: the framework already preserves both fields. No AutomationBench-specific controller or new difficulty label is needed.
  Date/Author: 2026-09-12 / Codex

- Decision: Begin with equal class and within-class task probabilities, then mix a fixed exploration reserve with normalized recent reward-variance signal.
  Rationale: equal initialization covers unknown classes; recent variance directs practice toward tasks where the current student produces mixed outcomes; the exploration reserve revisits low-signal and previously saturated tasks so model changes can correct stale beliefs. If every score is zero or unavailable, the allocation returns to the equal base mixture.
  Date/Author: 2026-09-12 / Codex

- Decision: Measure recency by each task's last completed rollout groups, not optimizer steps.
  Rationale: optimizer steps do not guarantee class coverage. A bounded per-task evidence window changes only when that task is actually observed.
  Date/Author: 2026-09-12 / Codex

- Decision: Persist an ordered JSONL journal through one bounded writer queue and write an atomic controller snapshot when the model checkpoint callback runs.
  Rationale: ordinary evidence writes stay off the rollout critical path, writer failures remain visible, and resume pairs controller state with the exact model checkpoint instead of replaying newer evidence against older weights.
  Date/Author: 2026-09-12 / Codex

- Decision: Qualify the first implementation on one training process.
  Rationale: the requested local RTX PRO 6000 run uses one process. Cross-rank evidence aggregation and synchronized allocation require an explicit distributed design and must not be silently approximated.
  Date/Author: 2026-09-12 / Codex

## Outcomes & Retrospective

Implementation and qualification are in progress. This section will record the two run identifiers, terminal states, controller evidence, held conditions, and any limits revealed by the experiment.

## Context and Orientation

`packages/train/src/posttrain/train/profiles.py` owns typed algorithm/profile settings. `packages/train/src/posttrain/train/catalog_schema.py` decodes project YAML into those settings. `packages/train/src/posttrain/train/backends/trl/policy_optimization.py` materializes Verifiers rollout examples as a Hugging Face dataset and creates the TRL trainer. Each row includes `example_id` and environment observation facets such as AutomationBench's `domain`.

A task class is the category used by the sampler. For this experiment it is an AutomationBench domain such as `sales` or `support`; internally they are all classes. A task is one concrete rollout example. A prompt group is one task repeated for four fresh student attempts, which GRPO compares to compute relative advantages.

The adaptive controller will own an inventory of tasks, a bounded recent evidence window for each task, the current class and task probabilities, a deterministic decision counter, and a persistence backend. Its learning signal is within-group reward variance. For binary rewards, the normalized score is four times the population variance and lies between zero and one. For bounded continuous rewards the same formula is clipped to that interval. Zero is ambiguous: it can mean every attempt failed or every attempt succeeded. The controller therefore never removes a task solely because its variance is zero; a configured exploration reserve keeps it eligible and allows later model updates to change its state.

The existing `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` owns the model, environment, inference, and training selections for this comparison. The vanilla work package is `apps/lab/.posttrain/work_packages/lfm26_automationbench_grpo_20_local.yaml`. The adaptive arm will be a new work package rather than changing the existing OLMo 3 arm in place.

## Plan of Work

First amend the canonical training-settings contract in `docs/post-training/02-primitives.md` and `docs/post-training/05-apis.md` so an optional curriculum capability may select rollout examples before an algorithm consumes them. The amendment must keep environment task ownership and algorithm identity unchanged.

Add `AdaptiveCurriculum` to `profiles.py` and a matching strict Pydantic schema to `catalog_schema.py`. Its small public surface will name the class field, recent groups retained per task, exploration fraction, and seed. The controller derives equal base weights from the resolved inventory; users do not configure per-class weights for this first profile.

Create a train-owned adaptive curriculum module containing the pure selection/state logic and a persistence protocol. The file implementation will append versioned records to JSONL through one bounded queue. It will expose `flush`, `snapshot`, and `close`; errors raised by the writer thread must surface on the next public operation. Snapshots use write-then-rename so a model checkpoint either has a complete matching controller state or no controller state.

Compose a small TRL subclass around the existing telemetry subclass. Immediately before a generation boundary, it replaces the dataloader's scheduled prompt groups with controller-selected inventory rows while preserving the required number of repeats. When TRL has calculated raw rewards, it records one observation per complete task group and updates the next allocation. It emits allocation and evidence events with stable task/class identities. A trainer callback flushes and snapshots controller state on checkpoint saves. The backend closes the writer in `finally` on success or failure.

Add tests that use small synthetic task inventories and rewards. They must prove equal initialization, high-variance prioritization, exploration-based revisits, fallback when all scores are zero, per-task evidence windows, deterministic replay, ordered queued writes, snapshot restore, invalid class metadata rejection, and unchanged behavior when the profile field is absent. A trainer-wrapper test will prove that selection occurs before generation and observation after reward computation without requiring a GPU.

Add a new OLMo 3 training selection with the adaptive curriculum configured for `domain`, then bind it in a new 20-update work package. Keep the existing vanilla GRPO work package as the random shuffled baseline. Validate both packages before running them.

Run the baseline to completion, then run the adaptive OLMo 3 arm on the same machine and resolved task population. Record run identities before waiting. Compare completed updates, rollout cost, reward signal, class/task allocations, active-sampling retention, and terminal artifacts. Because the algorithm also changes, describe the result as a system comparison.

## Concrete Steps

All commands run from `/home/hammad/projects/rl` unless stated otherwise.

Implement and test incrementally:

    uv run pytest packages/train/tests/test_adaptive_curriculum.py packages/train/tests/test_catalog.py packages/train/tests/test_api.py
    uv run ruff check packages/train apps/lab
    uv run pyright
    uv run lint-imports
    git diff --check

Validate and launch each work package from `apps/lab`:

    uv run --package posttrain posttrain work-package validate .posttrain/work_packages/lfm26_automationbench_grpo_20_local.yaml
    uv run --package posttrain posttrain work-package validate .posttrain/work_packages/lfm26_automationbench_olmo3_adaptive_20_local.yaml
    uv run --package posttrain posttrain work-package run .posttrain/work_packages/lfm26_automationbench_grpo_20_local.yaml --job train
    uv run --package posttrain posttrain work-package run .posttrain/work_packages/lfm26_automationbench_olmo3_adaptive_20_local.yaml --job train

The exact run and log inspection commands will be added here after launch because the CLI returns the durable run identifiers.

## Validation and Acceptance

The capability is accepted when profile decoding rejects malformed settings, the pure controller tests pass, the queued backend produces an ordered replayable journal, and a checkpoint contains an atomic controller snapshot that restores the same evidence windows and next allocation.

Both work packages must validate against the same model, AutomationBench environment revision, task-mix digest, rollout sampling, optimizer budget, and local inference binding. The vanilla job must complete 20 optimizer updates with no adaptive-controller events. The combined job must complete 20 optimizer updates and produce controller journal records showing equal initial allocation, observed group evidence, and later allocation changes while OLMo active-sampling metrics remain separately visible.

The final report must distinguish observed facts from inference. It must not attribute any quality difference solely to adaptive selection because the second arm also uses the OLMo 3 update recipe.

## Idempotence and Recovery

Unit tests and package validation are safe to repeat. A repeated training command may resume only from a complete retained model checkpoint. At each model checkpoint, the controller callback flushes its journal and writes a matching snapshot inside that checkpoint directory. Resume loads that snapshot; it does not infer controller state from journal entries written after the checkpoint. If the writer thread fails, the job fails rather than continuing with unaudited state.

The existing proposal edits in `docs/research/README.md` and `docs/research/proposals/` predate this branch and must remain intact. Implementation commits will stage only their intended files.

## Artifacts and Notes

The AutomationBench environment is pinned to `carbonteq-ai/verifiers-environments` commit `12ff5e1abfab369b8dec4df3ce83c5984f55ad34`. Its resolved training population contains 160 selected tasks, each with four fresh attempts, and declares `domain` as an observation facet. The controller file names and run identifiers will be recorded after implementation and launch.

## Interfaces and Dependencies

In `packages/train/src/posttrain/train/profiles.py`, define an immutable `AdaptiveCurriculum` value and add `adaptive_curriculum: AdaptiveCurriculum | None` to `GRPOSettings`.

In a new train-owned module, define a controller with operations equivalent to:

    select(group_count: int, step: int) -> tuple[task rows, decision evidence]
    observe(groups: sequence of task identity, class identity, rewards, step) -> observation evidence
    snapshot(path: Path) -> None
    close() -> None

Define a persistence protocol whose initial implementation loads records, enqueues JSON-compatible records, flushes queued writes, writes an atomic snapshot, and closes. Do not introduce an AutomationBench import into `packages/train`; the controller consumes only row metadata named by configuration.

Change note, 2026-09-12: created the implementation plan after source inspection; recorded the separation between curriculum selection and OLMo 3 active sampling, per-task evidence windows, and queued checkpoint-aligned file persistence.

Change note, 2026-09-12: updated progress after implementing the controller and catalog profile; recorded the machine-config compatibility issue and current repository-wide Pyright baseline before remote qualification.
