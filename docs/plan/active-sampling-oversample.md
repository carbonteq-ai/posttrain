# Oversample active-sampling rounds so SAMPO and OLMo 3 updates rarely need a refill round

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Active sampling is the refill rule SAMPO and the OLMo 3 GRPO recipe use: an
update needs `num_prompts_per_step` prompt groups (one prompt with
`num_generations` rollouts each) whose rewards differ, because a group whose
rollouts all scored the same gives no learning signal. The trainer generates
the target groups, drops groups without reward spread, and then generates only
the missing groups in another round, and again, up to `max_candidate_batches`
rounds. Each round of multi-turn AutomationBench episodes takes minutes, and a
refill round runs only a handful of groups while vLLM and the environment
workers sit mostly idle.

On the LFM2.5-2.6B SAMPO run (24 prompts x 6 generations = 144 episodes, all in
flight at once) about 88% of groups keep their spread, so almost every update
needs a second round and some a third; update 2 of the latest run spent 225 s
in rollout over two rounds.

After this change a training settings entry can ask for extra prompt groups in
each round:

    active_sampling: {max_candidate_batches: 6, oversample: 4, oversample_refill: 2}

The first round then runs 28 groups (168 episodes) at once and usually fills the
update by itself. The update is assembled exactly as before, from the first 24
groups with spread in candidate order; the surplus is discarded. The rollout
topology must run the larger first round concurrently, and `posttrain job plan`
refuses the settings when it cannot.

This work does not change the frozen product baseline in
`docs/post-training/`: it adds an optional backend setting to existing training
settings and does not change what a job, run, or update means.

## Progress

- [x] (2026-09-28) TRL fork worktree `/home/hammad/projects/trl-oversample`,
  branch `codex/active-sampling-oversample` from post10 `4950b99d`.
- [x] (2026-09-28) TRL feature commit `0b428cd6`: `active_sampling_oversample`
  and `active_sampling_oversample_refill` in `GRPOConfig` and
  `GRPOTrainer._prepare_active_sampling_inputs`, docs, 11 new regression tests.
- [x] (2026-09-28) TRL release-preparation commit `c7321c4d`
  (`VERSION` 1.12.0.post11, `CARBONTEQ_FORK.md`); local wheel and sdist built.
- [x] (2026-09-28) Posttrain worktree `/home/hammad/projects/rl-oversample`,
  branch `codex/active-sampling-oversample` from `c3ae803d`: settings, schema,
  TRL argument translation, adaptive-curriculum sampler, job-plan and
  trainer-start concurrency guards, metrics, records, tests, docs.
- [ ] Maintainers: push the TRL branch, create the GitHub release, publish to
  `carbonteq/dev`, apply the pin change (see "Publication and pin change").
- [ ] Qualify on the workstation once it is free: one SAMPO update with
  `oversample: 4, oversample_refill: 2` and the concurrency in the worked
  example, comparing rollout seconds and `active_sampling_generation_rounds`
  with the post10 run.

## Surprises & Discoveries

- Observation: Posttrain's adaptive curriculum replaces TRL's active-sampling
  loop with its own (`_prepare_adaptive_active_sampling_inputs` in
  `packages/train/src/posttrain/train/backends/trl/policy_curriculum.py`), and
  the 2.6B SAMPO run uses a yield-first curriculum. A TRL-only change would not
  have reached that run, so the Posttrain sampler applies the same round sizes,
  reading the two values from the trainer.
  Evidence: `test_adaptive_active_sampling_oversamples_rounds_from_the_curriculum`.
- Observation: the SAMPO reward-contract digest hashes `asdict(settings)`. Adding
  fields to `ActiveGroupSampling` would have changed the digest of every
  existing SAMPO checkpoint and blocked their resumes.
  Evidence: `test_reward_contract_digest_ignores_oversampling`.
- Observation: accelerate's logger raises unless accelerate state exists, so a
  per-round log line broke the fake-trainer tests under pytest-xdist. Per-round
  counts are metrics instead.
- Observation: six `tests/test_grpo_trainer.py` cases failed in one parallel
  run with Hugging Face Hub HTTP 429 (rate limit); they pass when rerun.

## Decision Log

- Decision: fixed extra prompt-group counts, `oversample` for the first round
  and `oversample_refill` for each refill; no online retention estimate.
  Rationale: the extra groups must be sized to rollout concurrency, which is a
  fixed property of the bindings; an adaptive estimate adds state, resume
  semantics and tuning without a better bound. Date/Author: 2026-09-28, user.
- Decision: a refill round never exceeds the first round
  (`min(missing + oversample_refill, target + oversample, remaining pool)`).
  Rationale: the first-round size is then the only concurrency the topology
  must hold, so one guard covers every round. Date/Author: 2026-09-28, user.
- Decision: rounds are cut to the remaining candidate pool; only a round that
  cannot cover its missing groups fails. Rationale: the pool is the existing
  data-cost bound; an oversampling shortfall is not an error. Date: 2026-09-28.
- Decision: pass the TRL fields only when non-zero. Rationale: defaults keep
  working with the pinned post10 until the pin moves; a non-zero value on an
  older TRL fails before training with the required version. Date: 2026-09-28.
- Decision: exclude `oversample` and `oversample_refill` from the reward-contract
  digest. Rationale: they change how many groups a round generates, not
  rewards, credit, the loss, or the assembly rule (first target groups with
  spread in candidate order), so they do not change what is learned from a
  given update. Excluding them keeps existing checkpoints resumable and lets a
  resumed run turn oversampling on. With an adaptive curriculum the extra
  groups add evidence before a refill is chosen, which changes which tasks are
  sampled later; that is a sampling choice recorded in run attributes
  (`active_sampling_oversample`, `active_sampling_oversample_refill`), like the
  curriculum's own sampling. `max_candidate_batches` stays in the digest as
  before. Date: 2026-09-28.
- Decision: new metrics count prompt groups
  (`active_sampling/oversampled_groups`, `active_sampling/discarded_groups`,
  TRL's per-round `active_sampling/round_<n>_{requested,generated,retained}_groups`);
  the older `candidate_groups_*` counters count rows and are unchanged.
  Date: 2026-09-28.

## Outcomes & Retrospective

Implementation and local validation are complete in both repositories. What
remains is publication of TRL post11, the pin change, and a GPU qualification
update; none of it can run while the workstation is training.

## Context and Orientation

TRL fork (`/home/hammad/projects/trl-oversample`, sibling of the shared
`../trl` checkout): `trl/trainer/grpo_config.py` declares the fields and
validates them; `trl/trainer/grpo_trainer.py`
`GRPOTrainer._prepare_active_sampling_inputs` sizes each round. The candidate
pool is the dataloader's reservation of `active_sampling_max_batches` target
batches per update. `CARBONTEQ_FORK.md` records the delta.

Posttrain (this repository):

- `packages/train/src/posttrain/train/profiles.py` `ActiveGroupSampling`
  holds `max_candidate_batches`, `oversample`, `oversample_refill` and checks the
  reservation holds the first round; `catalog_schema.py`
  `ActiveGroupSamplingSchema` is the catalog form.
- `packages/train/src/posttrain/train/rollout_execution.py`
  `oversampled_round_capacity_error` is the shared capacity rule.
- `packages/jobs/src/posttrain/jobs/definitions.py`
  `_validate_oversampled_round_capacity` applies it at job compilation, which
  `posttrain job plan` runs (the first guard).
- `packages/train/src/posttrain/train/backends/trl/policy_config.py`
  passes the TRL arguments and `validate_oversampled_round_capacity` applies
  the rule before the trainer is built; `policy_optimization.py` re-checks
  with the engine's resolved `max_num_seqs` after construction and before the
  first rollout (the second guard), and rejects TRL releases without the fields.
- `policy_curriculum.py` sizes adaptive-curriculum rounds the same way.
- `reward_recovery.py` leaves the two fields out of the reward digest.
- `grpo_observations.py`, `apps/observatory/.../metric_catalog.py` and
  `telemetry.py` map and describe the new metrics; `packages/work/.../runner.py`
  and `api.py` record the settings.

## Plan of Work

The work is done; this section records it for a reader repeating or reviewing
it. In TRL, the round size is `missing` without oversampling; with it,
`requested = min(missing + (oversample_refill if refill else oversample), target
+ oversample)` rows, and the generated size is `max(min(requested, remaining
pool), missing)`. If the pool cannot supply the missing rows the method raises
`active sampling exhausted its bounded candidate pool: ...`. Retained rows are
concatenated in candidate order and cut to the target, so surplus rows are
dropped. In Posttrain the same arithmetic runs in the curriculum sampler, the
settings and guards described above were added, and tests cover each part.

## Concrete Steps

TRL, from `/home/hammad/projects/trl-oversample` (a `.venv` created with
`uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e ".[test]"`):

    .venv/bin/python -m pytest -q tests/test_dapo_dynamic_sampling.py tests/test_rollout_admission.py tests/test_olmo3_grpo_config.py
    47 passed
    .venv/bin/python -m pytest -q -n 4 -m "not slow and not low_priority" tests/test_vllm_generation.py tests/test_dapo_dynamic_sampling.py tests/test_sampo_precomputed_advantages.py tests/test_rollout_admission.py tests/test_olmo3_grpo_config.py tests/test_grpo_trim_padding.py tests/test_grpo_deferred_importance_sampling.py tests/test_grpo_trainer.py
    237 passed, 50 skipped, 6 failed (Hugging Face Hub 429); the 6 pass on rerun
    uvx ruff@0.13.3 check <changed files>; uvx ruff@0.13.3 format --check <changed files>

Build, from a `git archive` export of the release commit:

    SOURCE_DATE_EPOCH=1790592157 uv build --python 3.13 --out-dir /home/hammad/projects/trl-oversample/dist
    854b7f00356d23cd5b2e11e2e2c4b6836b940031d9aff5cb0d536135f042ee3f  trl-1.12.0.post11-py3-none-any.whl
    b2ef1e33115b691042703a4477a1f2b8b51fe55722c3f22bce7300346c1d7b33  trl-1.12.0.post11.tar.gz

A clean virtual environment with only that wheel imports `trl 1.12.0.post11`
and passes `tests/test_dapo_dynamic_sampling.py` (21 passed).

Posttrain, from `/home/hammad/projects/rl-oversample`:

    uv sync --all-packages --group dev --extra trl --python 3.13 --locked
    uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run lint-imports
    uv run pytest packages/train/tests packages/jobs/tests packages/work/tests apps/observatory/tests apps/lab/tests
    cd apps/lab && uv run posttrain catalog validate && uv run posttrain job plan .posttrain/work_packages/lfm26_automationbench_sampo_turns_continue40_fixed_tools_local_v1.yaml

With `oversample: 4` on that run's settings and its current bindings, the plan
fails with:

    active_sampling oversample 4 needs 168 concurrent episodes for the first round ((24 prompts + 4) x 6 generations), but rollout inference engine max_num_seqs is 144; environment max_concurrent is 144; training backend_options.rollout_execution env_workers x episodes_per_worker is 12 x 12 = 144. Raise each limit to at least 168 or lower active_sampling oversample.

## Validation and Acceptance

Defaults are unchanged: TRL's exact-refill test and the real two-update trainer
test pass unmodified, and existing catalog entries and work packages validate
and plan as before. The fake-trainer tests show the first round requesting
target plus `oversample`, refills requesting missing plus `oversample_refill`
capped at the first round, pool cuts, pool exhaustion, a round without admitted
rollouts, and the metrics. The job and trainer-start tests trip each concurrency
limit on its own and show `oversample_refill` alone never trips them. Live
acceptance is the GPU qualification in `Progress`.

## Idempotence and Recovery

Both worktrees are ordinary branches; nothing was pushed or published. To
discard, remove the worktrees with `git worktree remove` and delete the
branches. Setting both values to 0 (the default) disables oversampling.

## Worked example: LFM2.5-2.6B SAMPO, 24 x 6, 160 tasks

The candidate reservation is 24 x `max_candidate_batches` and must fit 160
tasks, so `max_candidate_batches` is at most 6 (144 prompts), which also holds
the 28-prompt first round. Treating each group as keeping spread independently
with probability 0.88:

    oversample / refill   rounds/update   one round   <=2 rounds   groups run   discarded
    0 / 0 (today)         2.29            5%          71%          27.3         0
    4 / 0                 1.29            76%         95%          28.5         1.1
    4 / 2                 1.24            76%         99.8%        28.9         1.4
    6 / 2                 1.06            94%         100%         30.2         2.6

`oversample: 4, oversample_refill: 2` removes about one serial refill round per
update for about 6% more episodes. The first round runs 168 episodes, so the
bindings need new revisions with vLLM `max_num_seqs: 168`, environment
`max_concurrent: 168`, and `rollout_execution` 12 workers x 14 episodes (or
14 x 12). KV cache: the c144 binding's 41 GiB holds about 1.6M tokens (27.5 KB
per token measured, including the DSpark drafter), 11K tokens per episode at
144 but 9.5K at 168. The measured average peak was about 8.4K per episode, so
168 fits with little headroom; keeping 11K per episode needs about 1.85M tokens,
roughly 47.5 GiB (`kv_cache_memory_bytes` about 51000000000), which must be
checked with `posttrain settings suggest` against the trainer's share of the
96 GB GPU. The saving per update is about one refill wave minus the slightly
longer 168-episode first round; with a refill wave of 60-90 s that is roughly
60-90 s of the 225 s measured, to be confirmed by the qualification update.

## Publication and pin change

The fork follows `docs/tooling/forks.md`. Maintainers, in order:

1. Push `codex/active-sampling-oversample` (head `c7321c4d`) to
   `carbonteq-ai/trl` and tag `carbonteq-v1.12.0.post11` at `c7321c4d`.
2. Create the GitHub release for that tag with the two files in
   `/home/hammad/projects/trl-oversample/dist/`, and confirm GitHub's asset
   digests equal the hashes above.
3. Dispatch Posttrain's `publish-trl-internal.yml` for the tag with those
   hashes (writes only to `carbonteq/dev`), and later
   `promote-retained-fork-candidate.yml` after qualification.
4. Apply the Posttrain pin change on branch
   `codex/active-sampling-oversample-post11-pin` (one commit on top of this
   branch): it moves `packages/train/pyproject.toml`, `release/forks.toml`,
   `release/github-constraints.txt`, `.github/workflows/quality.yml`, the base
   and lab training bindings, and the tests that name the version to post11.
   Its `uv.lock`, the generated catalog lock record and the lab bindings'
   `dependency_lock_sha256` are deliberately left at post10 because they can
   only be resolved against the published index. After step 3, from the
   repository root, regenerate them and amend that commit:

       uv lock --upgrade-package trl
       uv run posttrain-release lock-dependencies

   then copy the new `dependency_lock_sha256` from
   `packages/catalog/src/posttrain/catalog/base/locks.toml` into the lab
   training bindings that carry `dependency_lock: trl-fork@current`, move the
   runtime-image lock and profile pins as `docs/publishing.md` describes
   (`posttrain-release sync-runtime-profile-pins`, and
   `lock-runtime-dependencies` in the protected candidate), and run the
   validation ladder above. The post10 pin commit `6f9879e2` touched the same
   files.
