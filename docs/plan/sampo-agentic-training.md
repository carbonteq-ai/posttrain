# Implement SAMPO for multi-turn tool agents

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Developers can select Stable Agentic Multi-turn Policy Optimization, or SAMPO,
for a Verifiers environment that performs multiple model/tool turns. The
framework preserves the exact turns, computes hierarchical episode and
anchor-state advantages, applies sequence-level importance-ratio clipping, and
filters reward-constant trajectory groups. A SAMPO run is independently
identified in catalog configuration and run evidence; it is not a DAPO flag.

The executable backends are the maintained TRL fork and a maintained veRL
fork. The veRL implementation uses the official ARL-Arena SAMPO extension as
its semantic reference: `adv_estimator=sampo` selects GiGPO-style hierarchical
advantages while the actor uses the existing GSPO loss.

This work changes the frozen product baseline by adding `train.sampo`,
`SAMPOSettings`, `SAMPORequest`, and an explicit agentic trajectory contract.
The canonical amendment precedes implementation.

## Progress

- [x] (2026-07-24 03:45Z) Read the ICML 2026 SAMPO paper and cloned its official
  source at immutable revision `a25a2a229c85431b421ac785fa5f375a99b2072a`.
- [x] (2026-07-24 04:00Z) Inspected the local Verifiers bridge, TRL fork, veRL
  checkout, and official ARL-Arena implementation.
- [x] (2026-07-24 05:00Z) Amended the canonical baseline with SAMPO's exact
  product and backend boundary.
- [x] (2026-07-24 05:20Z) Added agentic turn projection and pure SAMPO
  advantage computation.
- [x] (2026-07-24 05:35Z) Added `SAMPOSettings`, catalog decoding, request
  construction, `train.sampo`, and its standard job definition.
- [x] (2026-07-24 05:50Z) Added validated precomputed token advantages to the
  maintained TRL fork and translated SAMPO to sequence-level importance
  sampling.
- [x] (2026-07-24 06:00Z) Added focused mathematical, catalog, adapter,
  multi-turn projection, and fail-closed veRL tests.
- [x] (2026-07-24 06:20Z) Ran Ruff, Pyright, all eight import contracts, the
  full main-repository suite (306 passed, 15 skipped), `git diff --check`, and
  focused TRL validation (6 passed).
- [x] (2026-07-24) Published the TRL fork and moved the immutable consumer pin
  to `b43a0a3d622ab1547f4d2abbd1b25eab3c52a0b9`.
- [x] (2026-07-24 08:10Z) Re-inspected the official ARL-Arena veRL extension
  and identified its SAMPO-to-GiGPO-plus-GSPO mapping, anchor metadata, and
  dynamic-filtering seams.
- [x] (2026-07-24 09:00Z) Added and validated the hierarchical SAMPO advantage estimator in the clean
  `../verl` fork checkout.
- [x] (2026-07-24 09:20Z) Wired typed SAMPO launch manifests, agent-loop turn metadata, Hydra
  configuration, backend dispatch, and tests in this repository.
- [x] (2026-07-24 10:15Z) Published the veRL fork and its fork ledger, and
  selected immutable candidate revision
  `7ed83a140e0b2c1e794c062ecd288cb202e7592a`. Runnable project bindings provide
  this revision with their machine-local interpreter and checkout paths.
- [ ] Run a real multi-turn GPU qualification workload.

## Surprises & Discoveries

- Observation: published SAMPO is not an entropy-branching algorithm.
  Evidence: Equation 8 and the official trainer combine GSPO sequence-level
  clipping, GiGPO hierarchical advantages, and DAPO-style dynamic filtering.

- Observation: the official source maps `adv_estimator=sampo` to
  `loss_mode=gspo` plus the GiGPO advantage estimator.
  Evidence:
  `/home/hammad/projects/ARL-Arena/recipe/shop_agent/ppo/ray_trainer.py` at
  revision `a25a2a229c85431b421ac785fa5f375a99b2072a`.

- Observation: Verifiers already preserves each sampled assistant node, its
  exact tokens and log probabilities, and the preceding user/tool observation.
  Evidence: the pinned Verifiers `MessageNode` graph and
  `VerifiersEnvironmentRolloutBridge` retain sampled masks and the native trace.

- Observation: Verifiers rewards are trajectory-level unless an environment
  explicitly records step signals.
  Evidence: `Trace.rewards` is a trace-level mapping; `MessageNode` has no
  reward field.

- Observation: the current TRL loss already computes the GSPO sequence
  importance ratio when `importance_sampling_level="sequence"` is paired with
  the ordinary clipped surrogate. It does not accept externally computed
  per-token advantages.
  Evidence: `../trl/trl/trainer/grpo_trainer.py` reduces token log-ratios over
  the active sequence and conditionally supports `(B,T)` advantages.

- Observation: the selected veRL checkout implements `loss_mode=gspo` but
  contains no GiGPO or anchor-state advantage implementation.
  Evidence: source searches in the clean `../verl` fork checkout find GSPO and
  no GiGPO symbols at base revision
  `5da56132aebca765adb5ab23cf83b43fd5b5f1dc`.

- Observation: the official SAMPO veRL extension retains one optimizer row per
  active agent step, while this framework already retains one complete
  trajectory plus token-aligned turn spans.
  Evidence: ARL-Arena's `gather_rollout_data` flattens active step rows; the
  framework's `EnvironmentRollout` stores ordered `AgenticTurn` spans over one
  flattened trajectory.

## Decision Log

- Decision: add SAMPO as a separate operation and settings type.
  Rationale: its multi-turn trajectory, anchor grouping, discount, and
  step-advantage weight are not valid knobs on ordinary GRPO or DAPO.
  Date/Author: 2026-07-24 / User and Codex

- Decision: represent one Verifiers episode as one agentic trajectory with
  ordered sampled turns.
  Rationale: this retains environment ownership and exact token lineage while
  making the turn boundary explicit for credit assignment.
  Date/Author: 2026-07-24 / Codex

- Decision: use the latest user/tool observation before a sampled assistant
  turn as the portable anchor-state key.
  Rationale: GiGPO groups actions taken from equivalent observed states. Raw
  full-history hashes would prevent useful grouping after otherwise identical
  environment states.
  Date/Author: 2026-07-24 / Codex

- Decision: default missing intermediate step rewards to zero and assign the
  trajectory reward to the final sampled turn.
  Rationale: this is the standard sparse-outcome return and preserves the
  published discounted-return equation. Inventing intermediate rewards would
  change environment meaning. Environments may later expose explicit step
  rewards through a versioned trace contract.
  Date/Author: 2026-07-24 / Codex

- Decision: compute SAMPO advantages in the backend-neutral rollout layer and
  pass exact token-aligned advantages to the trainer.
  Rationale: anchor grouping and reward meaning belong above trainer-specific
  tensors, while the backend remains responsible for current-policy
  log-probabilities and the clipped loss.
  Date/Author: 2026-07-24 / Codex

- Decision: implement veRL SAMPO in the clean `../verl` fork using the official
  ARL-Arena extension as the semantic reference.
  Rationale: the reference confirms the required composition is hierarchical
  GiGPO advantages plus GSPO loss. The fork can implement that composition
  without approximating SAMPO as GSPO alone.
  Date/Author: 2026-07-24 / Codex

- Decision: retain one complete agent trajectory per veRL row and compute
  token-aligned hierarchical advantages from typed turn spans.
  Rationale: it preserves exact Verifiers lineage and lets GSPO form one
  sequence ratio over all sampled policy tokens, while implementing the same
  episode-plus-anchor advantage equations as the official row-per-step
  extension.
  Date/Author: 2026-07-24 / Codex

## Outcomes & Retrospective

The framework contract, backend-neutral calculator, Verifiers projection, TRL
adapter, standard job, and typed veRL adapter are implemented. The TRL fork is
immutably pinned. The veRL fork is published at
`7ed83a140e0b2c1e794c062ecd288cb202e7592a`; it contains the SAMPO
implementation commit `8a718e5be7a107587f63967336ece333a5c160e1` and its
published fork ledger. The framework maps that estimator to GSPO plus pinned
recipe-backed dynamic sampling. The main repository passes 309 tests with 15
skips; the focused veRL suite passes 35 tests. Real GPU qualification remains.

## Context and Orientation

`packages/train/src/posttrain/train/online_rl.py` owns the current flat
environment rollout. `integrations/verifiers.py` translates a native Verifiers
trace into that rollout. `profiles.py`, `requests.py`, and `api.py` own public
settings, requests, and operations. `backends/trl/grpo.py` supplies the custom
rollout to TRL. The maintained TRL fork lives at `../trl`.

SAMPO uses three components. Sequence-level clipping computes one geometric-mean
importance ratio for a trajectory and shares it across its trainable tokens.
Hierarchical advantage adds an episode-relative advantage to an anchor-state
step-relative advantage. Dynamic filtering keeps prompt groups whose final
rewards vary and draws replacement groups up to a configured bound.

For sparse rewards, a K-turn trajectory with final reward R has step rewards
`[0, ..., 0, R]`; discounted step return at turn k is
`gamma ** (K - 1 - k) * R`. Episode advantage is computed among trajectories
for the same prompt. Step advantage is computed among turns with the same
prompt and anchor-state key. The configured `step_advantage_weight` combines
them. Environment/tool tokens receive zero advantage and remain outside the
policy loss mask.

## Plan of Work

First, amend the canonical post-training documents with the new operation,
selection, agentic trajectory, and backend capability boundary.

Second, add `AgenticTurn` metadata to `EnvironmentRollout`. Project each sampled
Verifiers assistant node into its completion-token span and derive its anchor
key from the immediately preceding non-sampled observation. Add a pure
`compute_sampo_advantages` function that validates complete prompt groups,
computes sparse discounted returns, episode and anchor-state relative
advantages, and returns token-aligned values.

Third, add `SAMPOSettings`, `SAMPORequest`, catalog schema/decoder support,
Verifiers request construction, and `train.sampo`. Reuse the existing model,
environment, training, inference, quantization, checkpoint, and artifact seats;
do not duplicate environment ownership.

Fourth, extend the TRL fork with an opt-in
`use_precomputed_advantages` configuration. The rollout function returns
token-aligned advantages; the trainer pads and uses them after reward
calculation while retaining rewards for logging and dynamic filtering. Reject
the option without a rollout function or aligned values. Map SAMPO to
sequence-level lower/upper clipping, group reward scaling only inside the framework
advantage calculator, retained-group dynamic sampling, and sequence-mean loss.

Fifth, extend the maintained veRL fork with a registered SAMPO advantage
estimator. Consume typed turn spans, anchor-state keys, and optional step
rewards from the agent-loop metadata; compute episode and anchor-relative
advantages on the driver; and use veRL's existing GSPO policy loss.

Sixth, add a typed SAMPO launch plan and worker mapping in this repository.
The isolated worker selects `algorithm.adv_estimator=sampo`,
`policy_loss.loss_mode=gspo`, sequence-mean aggregation, the selected clipping
bounds, and bounded native group filtering. Add tests for the mathematical
calculation, metadata validation, launch manifest, Hydra mapping, and backend
dispatch.

## Concrete Steps

Run from `/home/hammad/projects/trl`:

    uv run pytest tests/test_sampo_precomputed_advantages.py tests/test_dapo_dynamic_sampling.py -q

Run from `/home/hammad/projects/verl`:

    .venv/bin/python -m pytest \
      tests/trainer/ppo/test_sampo_advantage_on_cpu.py \
      tests/trainer/ppo/test_core_algos_on_cpu.py \
      tests/special_sanity/test_config_docs.py -q
    PATH="$PWD/.venv/bin:$PATH" bash scripts/generate_trainer_config.sh

Run from `/home/hammad/projects/rl`:

    uv run pytest packages/train/tests/test_sampo.py packages/train/tests/test_verifiers_grpo_bridge.py packages/train/tests/test_api.py
    uv run ruff check packages/train/src packages/train/tests
    uv run pyright packages/train/src packages/train/tests
    uv run lint-imports
    uv run pytest -q
    git diff --check

## Validation and Acceptance

A catalog SAMPO selection resolves to `SAMPOSettings`. A native two-turn
Verifiers trace projects two ordered turn spans, with only assistant tokens
trainable and stable anchor keys derived from the user/tool observations.

For two trajectories of the same prompt, the pure calculator produces the
expected episode-relative and discounted anchor-state-relative advantages. It
rejects incomplete groups, misaligned spans, single-generation groups, and
non-finite rewards.

TRL receives sequence-level importance sampling, asymmetric clipping,
precomputed token advantages, and retained-group dynamic filtering. It refuses
missing or misaligned precomputed advantages. veRL receives the same logical
turn metadata, computes hierarchical advantages on the driver, and combines
them with its existing GSPO loss and bounded group filtering.

No CPU contract test claims training quality. Release qualification requires a
short real multi-turn tool-agent run and inspection of non-zero step-advantage,
sequence-ratio, dynamic-filter, KL, gradient, and success metrics.

## Idempotence and Recovery

All main-repository tests use temporary paths. ARL-Arena remains a clean,
read-only research checkout. Do not modify the dirty `../verl-upstream`
checkout. veRL changes belong on `../verl` branch
`codex/sampo-agentic-advantages`. TRL changes are published on
`codex/dapo-dynamic-sampling` and selected by the main repository at immutable
commit `b43a0a3d622ab1547f4d2abbd1b25eab3c52a0b9`.

## Artifacts and Notes

Primary paper: arXiv `2602.21534`, revision v3 dated 2026-07-04.

Official source research checkout:

    /home/hammad/projects/ARL-Arena
    a25a2a229c85431b421ac785fa5f375a99b2072a

## Interfaces and Dependencies

`SAMPOSettings` owns `discount_gamma`, `step_advantage_weight`, lower and upper
sequence clip epsilons, advantage normalization, bounded dynamic sampling, and
the existing loop/group/length/KL settings.

`AgenticTurn` owns the completion-token span, stable anchor-state key, and
optional explicit step reward. `EnvironmentRollout.turns` preserves ordered
turns without importing Verifiers types into the public capability package.

`compute_sampo_advantages(settings, example_ids, rollouts)` returns one tuple of
float advantages per rollout, aligned to `completion_ids`.

Revision note (2026-07-24): Created from the paper, official source, local
Verifiers bridge, and exact selected backend revisions.

Revision note (2026-07-24): Reopened the veRL milestone after confirming that
the official SAMPO repository includes a veRL extension. The revised work uses
that implementation as the semantic reference instead of treating veRL support
as unavailable.

Revision note (2026-07-24): Implemented the candidate veRL fork estimator and
framework adapter. Publication and GPU qualification remain explicit release
gates.
