# Run VORTEX on the resolved SAMPO engine

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

VORTEX ("Variance-Oriented Refill and Task Exploration", first released in Posttrain 0.4.2; components listed in `docs/plan/verl-vortex-port.md`) is the set of choices earlier AutomationBench runs used to keep every update informative: bounded active sampling that regenerates prompt groups whose rewards are all equal, an adaptive curriculum whose `yield_first` policy chooses tasks likely to produce reward spread, a truncation penalty, KL to the base model, per-token sampler correction, and, for GRPO, the OLMo 3 objective. The resolved policy-update engine (the columnar engine that applies one collected population as several optimizer updates) supported active sampling for SAMPO but refused the curriculum. After this change a SAMPO run on the resolved engine uses the curriculum: each active round's tasks are chosen by the curriculum after it has observed the earlier rounds of the same collection, and its state is saved with each checkpoint. Together with learning-signal retention (surplus groups with reward spread are kept by how much they teach, `docs/post-training/02-primitives.md`) this gives the 2.6B SAMPO run every VORTEX component that applies to SAMPO.

## Progress

- [x] (2026-10-06 09:30Z) Candidate sources for the resolved active collection (`packages/train/src/posttrain/train/backends/trl/policy_rollouts.py`): `ReservedCandidates` (the job's shuffled inventory slice, the previous behaviour) and `CurriculumCandidates` (asks `AdaptiveCurriculumRuntime.select_task_groups` per round, `initial_batch` then `active_sampling_refill`, and feeds each round's task rewards to the new `observe_groups` before the next round). Collection evidence records `candidate_source` and the candidates as each round takes them.
- [x] (2026-10-06 09:30Z) `ResolvedTRLJob.curriculum` (`policy_job.py`): the runtime built in `policy_optimization.py` is handed to the resolved job instead of wrapping the trainer's dataloader; the resolved guard admits a curriculum for SAMPO only. Checkpoints already write `adaptive-curriculum-state.json` through `CheckpointPublisher`.
- [x] (2026-10-06 09:45Z) Local VORTEX check: environment `automationbench-manifest-steps-mixv2-vortex-local-v1` (domain facet), settings `lfm2.5-1.2b/automationbench-manifest-steps-vortex-local-v1` (yield-first curriculum, `retain: learning_signal`, 1 prompt x 4 attempts, 2 updates per collection, 4 collections), binding `training/lfm2.5-1.2b-trl-lora-manifest-vortex-local@1`, work package `apps/lab/.posttrain/work_packages/lfm12_automationbench_manifest_steps_vortex_local.yaml`, gate `lfm12-automationbench-manifest-steps-vortex-local`. Tests: train 1094, lab 105 pass; pyright 0 errors.
- [ ] Run the local VORTEX check on the 8 GB card and record the curriculum decisions, refill observations, retention scores, gradients and the checkpointed curriculum state.
- [ ] Qualify the VORTEX speed bindings on the resolved engine (DSpark speculative decoding, compiled decoder layers) by measurement before enabling them for the 2.6B run.
- [ ] 2.6B run with the curriculum: the 4k16t environment needs the domain facet and a settings revision.

## Surprises & Discoveries

- Observation: the curriculum controller only accepts rewards in [0, 1] and treats any other group as invalid evidence. AutomationBench task rewards include a tool-mistake penalty (to -0.1) and the trainer's truncation penalty, so the legacy path, which observes the environment's weighted reward, has been discarding those groups silently.
  Evidence: `adaptive_curriculum.py` `observe` (`any(value < 0 or value > 1 ...)` counts the group invalid); `adaptive_curriculum_runtime.py` `observe_rewards` passes weighted environment rewards.
- Observation: the OLMo 3 objective is a GRPO objective; SAMPO keeps its own objective (episode-geometric ratio, hierarchical credit), so it is not part of VORTEX for this job. The resolved engine still refuses `algorithm: olmo3` for GRPO.

## Decision Log

- Decision: the resolved collection's curriculum observes each episode's task reward clamped to [0, 1].
  Rationale: the curriculum models task success and whether a task yields reward spread on the task's own scale; an attempt pushed below zero by penalties is a failed attempt, not invalid evidence.
  Date/Author: 2026-10-06, Claude.
- Decision: candidates are a pluggable source of the collection, not a second collection path.
  Rationale: rounds, retention, evidence and admission stay one implementation; only where a round's tasks come from changes.
  Date/Author: 2026-10-06, Claude.

## Outcomes & Retrospective

None yet.

## Context and Orientation

`collect_active_resolved_population` (`policy_rollouts.py`) runs bounded active rounds (`ActiveRoundPlan`, `update_active_rounds.py`), keeps complete groups with reward spread, retains up to `num_prompts_per_step` of them (by candidate order or learning signal) and admits that population. `AdaptiveCurriculumRuntime` (`adaptive_curriculum_runtime.py`) wraps `AdaptiveCurriculumController` (`adaptive_curriculum.py`), which proposes distinct tasks per optimizer step and checkpoints its state.

## Concrete Steps

From `/home/hammad/projects/worktrees/rl-perf` after committing:

    cd apps/lab && UV_HTTP_TIMEOUT=300 ../../.venv/bin/posttrain job run .posttrain/work_packages/lfm12_automationbench_manifest_steps_vortex_local.yaml --provider local-docker --run-id manifest-steps-12-vortex-local-20261006-r1

## Validation and Acceptance

The run applies 8 updates over 4 collections; each collection's evidence (`collection-evidence/`) shows `candidate_source: adaptive_curriculum`, rounds whose tasks came from curriculum decisions, and `learning_signal` scores; job logs show `adaptive_curriculum_allocation_selected` and `adaptive_curriculum_evidence_observed` events per round; the checkpoint holds `adaptive-curriculum-state.json`; gradient norms are finite and non-zero.

## Idempotence and Recovery

Settings without `adaptive_curriculum` take the reserved-pool path unchanged. A failed local run is resubmitted with a new run id.

## Interfaces and Dependencies

`CandidateSource.take(count, *, round_index) -> list[row]`, `CandidateSource.observe(groups)`; `AdaptiveCurriculumRuntime.observe_groups(groups, *, step)`; `ResolvedTRLJob(..., curriculum=runtime)`.
