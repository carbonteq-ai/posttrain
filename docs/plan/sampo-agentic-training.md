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

The first executable backend is the maintained TRL fork. The current veRL
checkout already has the GSPO loss kernel but not SAMPO's GiGPO hierarchical
advantage estimator, so its adapter must reject SAMPO until a maintained,
published veRL fork implements that missing contract.

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
  Evidence: source searches in `../verl-upstream/verl` find GSPO and no GiGPO
  symbols.

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

- Decision: support TRL first and make veRL fail closed.
  Rationale: adding a fake veRL mapping to GSPO alone would omit SAMPO's
  hierarchical advantage and mislabel the run. The selected veRL checkout is
  also dirty and unpublished, so it is not a safe implicit modification target.
  Date/Author: 2026-07-24 / Codex

## Outcomes & Retrospective

The framework contract, backend-neutral calculator, Verifiers projection, TRL
adapter, standard job, and fail-closed veRL boundary are implemented. Focused
main-repository validation passes and the full suite has 306 passes and 15
skips; focused TRL
validation passes with 6 tests. The implementation is contract-complete and
immutably pinned; real GPU qualification remains.

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

Fifth, reject SAMPO in the veRL dispatcher with an error naming missing GiGPO
support. Add tests for the mathematical calculation, multi-turn token spans,
catalog/request validation, TRL arguments, dynamic filtering, and veRL
rejection.

## Concrete Steps

Run from `/home/hammad/projects/trl`:

    uv run pytest tests/test_sampo_precomputed_advantages.py tests/test_dapo_dynamic_sampling.py -q

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

TRL receives sequence-level importance sampling, symmetric clipping,
precomputed token advantages, and retained-group dynamic filtering. It refuses
missing or misaligned precomputed advantages. veRL refuses SAMPO before process
launch and explicitly names missing hierarchical advantage support.

No CPU contract test claims training quality. Release qualification requires a
short real multi-turn tool-agent run and inspection of non-zero step-advantage,
sequence-ratio, dynamic-filter, KL, gradient, and success metrics.

## Idempotence and Recovery

All main-repository tests use temporary paths. ARL-Arena remains a clean,
read-only research checkout. Do not modify the dirty `../verl-upstream`
checkout. TRL changes are published on `codex/dapo-dynamic-sampling` and
selected by the main repository at immutable commit
`b43a0a3d622ab1547f4d2abbd1b25eab3c52a0b9`.

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
