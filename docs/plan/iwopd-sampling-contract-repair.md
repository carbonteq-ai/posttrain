# Repair IW-OPD sampling-contract compatibility

This ExecPlan is a living document. Maintain it in accordance with
`docs/templates/PLAN.md`.

## Purpose / Big Picture

An on-policy distillation run must apply the sampling settings selected by its
`InferenceBinding`, or reject a selection the trainer cannot represent. The
current Posttrain adapter passes `min_p` to TRL's `IWOPDConfig`, but the pinned
`trl==1.9.2.post2` configuration does not define that field. A remote run
therefore fails during trainer construction before it can produce a rollout.

After this change, the maintained CarbonTeq TRL fork accepts and forwards
`min_p`, `repetition_penalty`, and additional generation arguments such as
`presence_penalty` to both transformers and vLLM generation. Posttrain pins a
new immutable build and tests its adapter against the real configuration type.
The user has requested targeted qualification after publication: one
0.8B-student / 2B-teacher TRL IW-OPD canary, plus one 0.8B veRL canary with
non-default shared sampling controls. They remain blocked until the immutable
package reaches the internal index and the resolved job image contains it.

## Progress

- [x] (2026-08-12 00:00Z) Reproduced the construction failure against the
  installed `trl==1.9.2.post2` and confirmed that `IWOPDConfig` has `top_p`
  and `top_k` but not `min_p`, `repetition_penalty`, or `generation_kwargs`.
- [x] (2026-08-12 00:00Z) Located the maintained 1.9.2 fork checkout at
  `/home/hammad/projects/trl-1.9-upgrade`; `/home/hammad/projects/trl` is a
  separate 1.8 development checkout and must not be used for this release.
- [x] (2026-08-12 00:00Z) Extended the 1.9.2 fork configuration and IW-OPD
  trainer with the missing generic sampling controls and focused regression
  tests; source commit `61064605db84f84898692c2b3eefe1eb2b90a952` is tagged
  `carbonteq-v1.9.2.post4`.
- [x] (2026-08-12 00:00Z) Verified the veRL agent-loop path preserves the complete shared
  `PolicySampling` value through its vLLM server request, and added a focused
  no-GPU regression test.
- [x] (2026-08-13) Published the tagged immutable TRL package through
  Posttrain's manually dispatched retained-asset workflow `31634140384`.
  It verified release hashes, uploaded exact bytes to the stable internal index,
  constructed `IWOPDConfig` during its clean install, and retained the receipt.
  The mistakenly dispatched fork workflow `31618185508` was cancelled; forks
  do not receive release runners.
- [x] (2026-08-13) Updated the Posttrain pin and lockfile to post4; the real
  config construction and the veRL agent-loop sampling regression both pass.
- [x] (2026-08-13) Added a one-step 0.8B veRL sampling-contract canary with a
  distinct environment, training, and inference binding for top-p, top-k,
  min-p, repetition, and presence controls. Its resolved global batch is two,
  matching its 1×2 rollout budget; package planning, resolution, and registry
  tests pass.
- [ ] Build the resolved immutable job image and run the two explicitly scoped
  GPU canaries without disturbing the unrelated active dstack job.

## Surprises & Discoveries

- Observation: The existing Posttrain test asserts the dictionary produced by
  `_distillation_arguments` but never constructs `IWOPDConfig`.
  Evidence: `packages/train/tests/test_api.py` asserts `arguments["min_p"]`,
  while the remote failure is `IWOPDConfig.__init__() got an unexpected keyword
  argument 'min_p'`.
- Observation: The relevant source checkout is `../trl-1.9-upgrade`, not the
  sibling named `../trl`.
  Evidence: `../trl-1.9-upgrade` is on tag `carbonteq-v1.9.2.post2` and owns
  `trl.experimental.iw_opd`; `../trl` is a separate 1.8 checkout.
- Observation: A fork-local publication workflow can look valid while being
  architecturally ineligible for the repository-scoped Posttrain runner.
  Evidence: the fork workflow run `31618185508` queued indefinitely. The
  Posttrain repository already owns a manual retained-asset publisher that
  downloads a fork release, checks hashes, uploads exact bytes, and performs a
  clean install on its own runner.
- Observation: Existing veRL GRPO work packages reference the current GSM8K
  environment from a project catalog overlay, rather than the framework base
  catalog.
  Evidence: resolving a new package without the project overlay failed with a
  missing `environment/gsm8k-grpo-qualification`; the actual job CLI and the
  qualification registry load the overlay, and the canary regression now does
  the same.
- Observation: a small online-RL canary must preserve the joint environment,
  inference, trainer, and optimizer-batch sampling contract.
  Evidence: the first compact profile was rejected first because global batch
  eight did not equal its 1×2 rollout budget, then because the environment and
  inference sampling values differed. The final canary has a dedicated
  global-batch-two training binding and identical complete `PolicySampling`
  values at both generation boundaries.

## Decision Log

- Decision: Fix the maintained TRL fork instead of dropping unsupported fields
  in Posttrain.
  Rationale: These controls are framework-owned behavior-policy selections.
  Omitting a non-default selection would make a run succeed with different
  rollout semantics.
  Date/Author: 2026-08-12 / Codex
- Decision: Keep the behavior generic across transformers and vLLM paths.
  Rationale: `IWOPDConfig` is a reusable trainer contract, and its sampling
  values must not depend on which rollout engine a consumer selects.
  Date/Author: 2026-08-12 / Codex
- Decision: Treat veRL as a parity boundary, not as a second implementation of
  the sampling policy.
  Rationale: veRL receives the serialized `InferenceBinding`, then the
  Posttrain agent loop creates the actual per-turn vLLM request. Both mappings
  must preserve exactly the backend-neutral `PolicySampling` controls.
  Date/Author: 2026-08-12 / Codex
- Decision: Publish fork distributions manually from the Posttrain repository's
  retained-asset workflows; do not assign forks a release runner.
  Rationale: the runner and its private-index credential remain repository
  scoped. A manual workflow dispatch receives only a signed release tag and
  expected artifact hashes, never a fork checkout or fork-controlled workflow.
  Date/Author: 2026-08-13 / Codex

## Outcomes & Retrospective

Work in progress. The expected outcome is a published immutable TRL version
and a Posttrain pin that construct the real config successfully, followed by
the two requested targeted canaries. Publication follows the existing manual
Posttrain retained-asset workflow rather than fork-local release automation.

## Context and Orientation

Posttrain models one rollout policy as `PolicySampling` in
`packages/train/src/posttrain/train/online_rl.py`. Its TRL adapter builds the
keyword arguments in
`packages/train/src/posttrain/train/backends/trl/distillation.py` and constructs
`IWOPDConfig` immediately before `IWOPDTrainer` starts. The selected controls
include temperature, top-p, top-k, min-p, repetition penalty, and presence
penalty.

The resolved dependency is now `trl==1.9.2.post4` in `uv.lock`, sourced from
the stable internal index with the exact retained wheel and source hashes. Its
maintained source is `/home/hammad/projects/trl-1.9-upgrade`, branch
`v1.9.2.post4-release`.
`IWOPDConfig` is an experimental TRL dataclass whose declared fields are the
only permitted constructor keywords. `IWOPDTrainer` builds a transformers
`GenerationConfig` for local generation and a `VLLMGeneration` object for
vLLM-backed generation; both must receive the same selected sampling behavior.

## Plan of Work

First, make a small generic extension in the TRL 1.9.2 fork. Add the three
missing sampling controls to `trl/experimental/iw_opd/iw_opd_config.py`, with
the same defaults and validation semantics used by nearby trainer configs.
Update `trl/experimental/iw_opd/iw_opd_trainer.py` so its local
`GenerationConfig` and `VLLMGeneration` construction consume the fields.
Add focused tests in `tests/experimental/test_iw_opd_trainer.py` that inspect
the real generated configuration and the vLLM constructor arguments without
requiring a GPU.

Update `CARBONTEQ_FORK.md` with the new generic behavior, tests, and release
state. Bump the package version to a distinct post release, commit the fork,
push it, create the immutable release tag, and use the existing retained
package-publication workflow. Record only hashes and public commit identifiers,
never credentials.

Then update Posttrain's exact requirement, fork metadata, lockfile, and
`docs/tooling/trl/README.md`. Replace the dictionary-only regression with a
test that imports the pinned `IWOPDConfig` and constructs it from the adapter
arguments. This proves the integration boundary that failed remotely.

## Concrete Steps

From `/home/hammad/projects/trl-1.9-upgrade`:

    uv run pytest tests/experimental/test_iw_opd_trainer.py -q
    uv run ruff check trl/experimental/iw_opd tests/experimental/test_iw_opd_trainer.py
    git diff --check

Commit and push the fork only after those checks pass. Build and publish the
new package through Posttrain's manually dispatched retained-asset workflow,
then obtain the actual wheel and source hashes from the internal index or build
receipt. The workflow must download the release assets from the fork, verify
their supplied SHA-256 values, and never execute a fork-controlled workflow.

From `/home/hammad/projects/rl` after publication:

    uv lock --upgrade-package trl
    uv run --package posttrain-train --extra trl pytest packages/train/tests/test_api.py -q
    uv run --package posttrain-train --extra trl python -c 'from trl.experimental.iw_opd import IWOPDConfig; print(IWOPDConfig(min_p=0.0, repetition_penalty=1.1, generation_kwargs={"presence_penalty": 1.5}))'
    git diff --check

## Validation and Acceptance

The new TRL test must prove that a non-default `min_p`, repetition penalty, and
presence penalty reach the actual local generation configuration and the vLLM
generation object. The Posttrain test must call the actual pinned
`IWOPDConfig(**_distillation_arguments(...))` with a non-default selection and
must pass. A fresh package install must report the new exact version and
construct the configuration without `TypeError`.

Acceptance includes two remote GPU canaries requested by the user. The TRL
canary uses the smallest evidence-bearing 0.8B student / 2B teacher IW-OPD
profile and must pass trainer construction with non-default controls. The veRL
canary uses a 0.8B actor and must record the exact non-default sampling policy
in its retained evidence. Neither is a general 2B-by-0.8B matrix.

## Idempotence and Recovery

The code and test changes are additive and safe to rerun. If package
publication fails, do not update the Posttrain pin: retain the fork commit and
repair the release workflow or index receipt first. If the new package is
published but Posttrain validation fails, create a new post release rather than
overwriting the immutable package. The failed original remote run remains
retained evidence and is not retried by this work.

## Artifacts and Notes

The original run failed during trainer initialization with:

    TypeError: IWOPDConfig.__init__() got an unexpected keyword argument 'min_p'

The installed `IWOPDConfig` signature for `trl==1.9.2.post2` contains `top_p`
and `top_k` but no `min_p`, `repetition_penalty`, or `generation_kwargs`.

## Interfaces and Dependencies

`IWOPDConfig` must expose:

    min_p: float | None = None
    repetition_penalty: float = 1.0
    generation_kwargs: dict[str, Any] | None = None

`IWOPDTrainer` must use those values consistently when it creates both
`transformers.GenerationConfig` and `trl.generation.VLLMGeneration`. The
Posttrain adapter continues to pass all declared `PolicySampling` values;
version-specific filtering is deliberately not an adapter responsibility.

The veRL adapter serializes `InferenceBinding.sampling` into its launch
manifest and, for every native Verifiers turn, `VerlPolicyGenerator` sends the
resolved `PolicySampling` values to veRL's vLLM server manager. Its regression
test must capture that request and compare every sampling field with the shared
value.

Revision 2026-08-12: created after reproducing the failed remote canary and
identifying the correct 1.9.2 maintained-fork checkout.

Revision 2026-08-12: expanded after the user requested explicit veRL parity;
the plan now covers the serialized manifest and per-turn agent-loop boundary.

Revision 2026-08-13: corrected the publication route: forks have no release
runner. The repository-scoped Posttrain retained-asset workflow performs the
manual distribution publication and clean-install receipt before the requested
GPU canaries.

Revision 2026-08-13: publication, lock resolution, and local contracts passed.
The remaining gate is sequential remote qualification of the TRL IW-OPD and
veRL sampling-contract canaries from materialized immutable job images.
