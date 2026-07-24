# Implement and harden DAPO across TRL and veRL

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Developers can select the DAPO algorithm for environment-driven online
reinforcement learning through either TRL or veRL. Both backends execute the
same DAPO semantics: token-level policy loss, asymmetric clipping, global
active-token normalization, retained-group dynamic sampling, truncation
handling, and optional soft-overlong reward shaping. The implementation improves
sampling efficiency, bounded failure behavior, memory use, and evidence without
quietly replacing the DAPO objective with another published algorithm.

Stable Agentic Multi-turn Policy Optimization, or SAMPO, is a separate future
algorithm for multi-turn tool-using agents. It is not implemented by adding
flags to DAPO because it changes agentic advantage construction and other
multi-turn training semantics.

This work changes the frozen product baseline because `GRPOSettings` previously
described one fixed GRPO objective. The canonical amendment is updated before
the public selection and adapters.

## Progress

- [x] (2026-07-24 01:15Z) Inspected the selected TRL and veRL sources and the
  official pinned DAPO recipe.
- [x] (2026-07-24 02:05Z) Prototyped retained-group dynamic sampling in the
  CarbonTeq TRL fork.
- [x] (2026-07-24 03:00Z) Decided to preserve DAPO algorithm identity and move
  multi-turn tool-agent improvements into a separate SAMPO implementation.
- [x] (2026-07-24 03:10Z) Finalized the canonical DAPO contract and separate
  SAMPO boundary.
- [x] (2026-07-24 03:12Z) Completed and cataloged the backend-neutral DAPO
  settings.
- [x] (2026-07-24 03:14Z) Completed retained-group DAPO sampling and
  documentation in the local TRL fork.
- [x] (2026-07-24 03:16Z) Translated DAPO to TRL and the pinned veRL DAPO
  recipe.
- [x] (2026-07-24 03:18Z) Added parity, bounded-failure, recipe-pin, catalog,
  and evidence tests.
- [x] (2026-07-24 03:22Z) Ran focused and repository-wide CPU validation.
- [ ] Publish the TRL fork commit, update the immutable consumer pin, and run
  short real GPU qualification on both backends.

## Surprises & Discoveries

- Observation: current TRL already implements DAPO loss aggregation,
  asymmetric clipping, and truncation masking, but not retained-group dynamic
  sampling.
  Evidence: the selected fork implements `loss_type="dapo"`, `epsilon`,
  `epsilon_high`, and `mask_truncated_completions`; its paper index previously
  marked dynamic sampling unsupported.

- Observation: an early upstream TRL dynamic-sampling implementation regenerated
  whole candidate batches, which discards already useful prompt groups.
  Evidence: review discussion on Hugging Face TRL pull request 3758 identified
  the incorrect multi-prompt behavior.

- Observation: the pinned official veRL DAPO recipe retains informative groups
  and generates only enough replacement groups to fill the optimizer batch.
  Evidence: `/home/hammad/projects/verl-recipe/dapo/dapo_ray_trainer.py` at
  revision `230ee612279d552a4f34ecbfab931c213abd514d`.

- Observation: ARPO and SAMPO are agentic algorithms rather than alternate DAPO
  loss kernels. They depend on multi-turn structure and finer-grained advantage
  information.
  Evidence: their published descriptions evaluate multi-step interactive tasks
  and define rollout or advantage behavior beyond DAPO's trajectory-level
  objective.

## Decision Log

- Decision: keep the public algorithm name `dapo`.
  Rationale: runtime and correctness improvements do not require a new
  algorithm name when the mathematical objective and sampling semantics remain
  DAPO.
  Date/Author: 2026-07-24 / User and Codex

- Decision: do not fold CISPO, GSPO, Dr. GRPO, ARPO, or SAMPO into an unnamed
  DAPO variant.
  Rationale: these works replace objective, normalization, rollout, or credit
  assignment semantics. Run evidence must preserve those distinctions.
  Date/Author: 2026-07-24 / User and Codex

- Decision: implement dynamic sampling by retaining reward-informative groups
  and refilling only missing groups.
  Rationale: regenerating the entire batch wastes valid rollouts and deviates
  from the official veRL DAPO behavior.
  Date/Author: 2026-07-24 / Codex

- Decision: candidate exhaustion raises an actionable error rather than
  training a partial optimizer batch.
  Rationale: a partial batch changes normalization and distributed update
  semantics and should not be silently accepted.
  Date/Author: 2026-07-24 / Codex

- Decision: implement SAMPO separately after the DAPO backend contract is
  complete.
  Rationale: multi-turn tool-agent training needs explicit turn structure and
  advantage semantics that should not leak into ordinary DAPO settings.
  Date/Author: 2026-07-24 / User and Codex

## Outcomes & Retrospective

The code and portable contracts are complete in the local worktrees. The TRL
retained-group sampler has four focused passing CPU tests; the framework has 83
focused passing adapter/catalog tests and 298 passing full-suite tests with 15
expected skips. Static typing, linting, import boundaries, and diff checks pass.

The feature is not yet reproducibly published. The TRL fork change remains
uncommitted on its local branch, so the main repository's immutable dependency
pin has intentionally not moved. Short real GPU qualification on TRL and veRL
also remains a release gate.

## Context and Orientation

`packages/train/src/posttrain/train/profiles.py` owns backend-neutral algorithm
settings. `catalog_schema.py` decodes catalog YAML into those values.
`backends/trl/grpo.py` creates TRL `GRPOConfig` arguments.
`backends/verl/launcher.py`, `contracts.py`, and `worker.py` create and execute
the isolated veRL manifest. The official veRL DAPO retained-group trainer is in
the separately pinned `/home/hammad/projects/verl-recipe` checkout.

Dynamic group sampling evaluates multiple generations for each prompt. If every
generation receives the same reward, the group has no relative learning signal
and is discarded. Informative groups are retained while new prompts are sampled
to fill only the missing group slots. A candidate-batch limit bounds the work
when an environment repeatedly returns constant rewards.

SAMPO is intentionally outside this plan. Its future plan must define the
multi-turn trajectory representation, tool-turn boundaries, advantage inputs,
backend capability gates, and qualification workloads before implementation.

## Plan of Work

First, amend the canonical documents to define DAPO precisely and state that
runtime optimizations do not change its algorithm identity. Record SAMPO as a
separate future multi-turn algorithm.

Second, finish `GRPOSettings` and its schema with the DAPO discriminator,
asymmetric clip values, bounded `DynamicGroupSampling`, truncation masking, and
optional soft-overlong shaping. Add a catalog smoke selection and preserve
every choice in run evidence.

Third, finish the TRL fork's retained-group sampler. It must keep informative
groups, refill missing groups, synchronize readiness across distributed
processes, recompute the global token normalizer over the final batch, and fail
when the configured candidate bound is exhausted.

Fourth, map DAPO to TRL's native DAPO loss and to the pinned official veRL DAPO
recipe. Validate the recipe checkout revision and cleanliness before launch.
Protect all algorithm-owned settings from backend-native override replacement.

Finally, test equivalent logical settings in both adapters, invalid settings,
candidate exhaustion, recipe pin enforcement, and stable run evidence.

## Concrete Steps

Run from `/home/hammad/projects/trl`:

    uv run pytest tests/test_dapo_dynamic_sampling.py -q

Run from `/home/hammad/projects/rl`:

    uv run pytest packages/train/tests/test_api.py packages/train/tests/test_verl_backend.py apps/lab/tests/test_catalog.py
    uv run ruff check packages/train apps/lab
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

No GPU is needed for contract tests. A short real training run on each backend
remains a release gate before DAPO is described as qualified.

## Validation and Acceptance

A catalog entry using `algorithm: dapo` resolves and emits evidence for DAPO
loss aggregation, low and high clip values, dynamic sampling, truncation
handling, and soft-overlong policy.

TRL receives `loss_type=dapo` and the retained-group sampler settings. veRL
receives token-mean loss aggregation, asymmetric clipping, and the pinned DAPO
recipe filter configuration.

If the sampler reaches its candidate limit without filling every optimizer
batch on every process, training raises an actionable error. Neither adapter
silently falls back to ordinary GRPO or a partial batch.

No DAPO run is labeled CISPO, GSPO, ARPO, or SAMPO. Future SAMPO run evidence
will use its own algorithm identity and multi-turn settings.

## Idempotence and Recovery

All contract tests use temporary output directories. Do not mutate the dirty
`../verl-upstream` checkout. Generic TRL changes remain isolated on its
`codex/dapo-dynamic-sampling` branch. Do not update the immutable consumer pin
until the TRL fork is committed and pushed. Existing unrelated dirty changes in
the main repository must be preserved.

## Artifacts and Notes

Focused TRL prototype evidence:

    uv run pytest tests/test_dapo_dynamic_sampling.py -q
    4 passed

Focused framework evidence:

    uv run pytest packages/train/tests/test_api.py packages/train/tests/test_verl_backend.py apps/lab/tests/test_catalog.py -q
    83 passed

Repository evidence:

    uv run pytest -q
    298 passed, 15 skipped, 1 warning

    uv run ruff check packages/train/src packages/train/tests apps/lab/tests/test_catalog.py
    All checks passed!

    uv run pyright packages/train/src packages/train/tests
    0 errors, 0 warnings, 0 informations

    uv run lint-imports
    Contracts: 8 kept, 0 broken

## Interfaces and Dependencies

`GRPOSettings.algorithm` accepts `grpo` or `dapo`.
`DynamicGroupSampling.max_candidate_batches` bounds replacement attempts.
Algorithm-owned clipping, truncation, and reward-shaping settings cannot be
replaced through `TrainingBinding.backend_options`.

The selected veRL DAPO recipe revision is
`230ee612279d552a4f34ecbfab931c213abd514d`. The selected veRL backend revision
is `a35908ca3c9632859c58d6a2855d858918ae21dc`. The TRL consumer pin must be
updated only after the fork change is published at an immutable commit.

Revision note (2026-07-24): Removed the proposed DAPO+++ composite. DAPO now
keeps its published identity with implementation hardening, while SAMPO is
reserved for a separate multi-turn tool-agent implementation.

Revision note (2026-07-24): Recorded completed local implementation and CPU
validation. Publication, immutable TRL pinning, and real GPU qualification
remain explicit release gates.
