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
- [x] (2026-08-13) Packed the first TRL canary image and retained the failed
  provider run `iwopd-post4-smoke-20260813`; its 2B vLLM teacher had no KV
  cache because the selected `0.15` memory fraction was smaller than the
  model's weights.
- [x] (2026-08-13) Added a 24 GiB capacity-bounded canary selection with an
  explicit teacher vLLM fraction of `0.35`, and a snapshot that names TRL
  post4. The resulting retry reached real remote trainer construction.
- [x] (2026-08-13) Published and retained runtime-image candidate
  `31638943255` as `0.3.16rc2`. Its generated manifest pins the rebuilt TRL
  kind `sha256:ea123930217d85620a50441027f9b4ca59135445d7b04ef48ab0e964ca016b27`
  and veRL kind
  `sha256:458e789ded3bc50a04f78b7182a7717608e2000b9c73f728344a94424185988c`,
  both from the post4 runtime lock
  `aa430e9e0d56790a7351ba4e1f7fcbc27b02c264b26eef096d2721e6d89730bd`.
- [x] (2026-08-13) Re-evaluated the actual-job image path. It already uses
  three stages (`packaged-context` → immutable runtime kind → smoke), and a
  verified receipt avoids every BuildKit call on a repeat pack. The meaningful
  cold-path cost is pulling the multi-gigabyte kind image onto a worker, not
  assembling the approximately 2.9 MiB job context or building workspace
  wheels (about one second with cached dependencies).
- [x] (2026-08-13) Added a per-publication lock around remote job-image
  publication. Concurrent callers now wait and reuse the producer's verified
  receipt instead of racing on a named context that publication cleanup removes.
- [x] (2026-08-13) Recorded the image/release performance contract in
  `docs/publishing.md`: multi-stage ownership, parent and receipt reuse,
  single-flight publication, BuildKit parallelism, compression defaults,
  cold-pull diagnosis, safe registry retention, and the separate future
  decision for kind-scoped locks.
- [x] (2026-08-13) Made required Quality install the selected TRL optional
  dependency closure from the frozen lock while injecting the hash-verified
  GitHub Release wheel. The real-config and compatibility tests now declare
  their CPU-only construction intent and read the selected TRL version from
  the consumer manifest instead of carrying a stale literal.
- [x] (2026-08-13) Added retained non-finite metric and gradient-parameter
  evidence to the TRL distillation path. The FP32-weight retry still reaches
  the first actor update and reports `on_policy_loss=-inf`, ruling out BF16
  model weights as the primary cause without hiding finite loss values.
- [x] (2026-08-13) Reproduced the veRL canary's real worker failure after
  preserving its diagnostic artifact, fixed the generic nested-tensor
  selection bug in the maintained veRL fork, and released immutable source
  `808923d487aa2c524fda02cf5289110541b4221f` as
  `carbonteq-v0.9.0.dev2`. The release assets are retained on GitHub; the
  Posttrain manual internal-publication workflow now still needs to publish
  those exact verified bytes before the runtime kind can select the revision.
- [x] (2026-08-13) Added the repository-owned manual veRL retained-asset
  publisher. It downloads release bytes by immutable tag, verifies caller
  supplied SHA-256 values, publishes only those bytes to the internal stable
  index, and proves a clean installation. No fork release runner is used.
- [x] (2026-08-13) Retrieved the failed TRL run's retained native traces and
  verified all 749 sampled-token behavior-policy log probabilities are finite.
  Released the generic loss-boundary diagnostics as
  `trl==1.9.2.post8` (fork tag `carbonteq-v1.9.2.post8`, commit
  `9a219ce5a593d85fe6058025de211ce42267e6b6`) and published exact release
  bytes through successful Posttrain retained-asset workflow `31644269156`.
- [x] (2026-08-13) Published runtime candidate `31644500455` as
  `0.3.16rc3` and materialized its generated parent manifest.  The selected
  TRL runtime kind now contains the immutable post8 dependency lock.
- [x] (2026-08-13) Corrected Lab's project-local TRL qualification binding
  from post4 to post8 (`@3`) and made the bounded work package select that
  revision.  The read-only work-package plan now resolves the exact post8
  source commit and lock digest before any GPU submission.
- [x] (2026-08-13) Ran the post8 bounded TRL canary
  `iwopd-post8-loss-boundary-20260813`. It retained two native traces, 749
  finite rollout log probabilities, complete teacher scoring, and ordinary
  IW-OPD advantages/weights, then failed at `on_policy_loss=-inf` during the
  student update. Released post9 with a generic token-loss overflow boundary
  that reports the finite source ranges. The retained-asset workflow
  `31646648237` then incorrectly published post9 directly to
  `carbonteq/stable`; that immutable artifact is retained only as a rejected
  audit record and must not be used by a candidate or qualification run.
- [x] (2026-08-13) Repaired the retained-fork path for TRL, veRL, and Trackio:
  candidate publishers write only to `carbonteq/dev`, runtime candidates
  materialize and retain a development-channel lock, and a separate
  allowlisted server-side workflow promotes hash-verified bytes to stable. The
  workflow changes merged in PRs 72 and 73.
- [x] (2026-08-13) Created TRL post10 and published its exact retained assets
  to `carbonteq/dev`. Run `31649996139` verified development storage, byte
  hashes, a clean install, and a retained publication receipt.
- [x] (2026-08-13) Published the retained veRL dev2 assets to
  `carbonteq/dev`. Run `31649999160` verified development storage, byte
  hashes, a clean install, and a retained publication receipt.
- [x] (2026-08-13) Ran the post10 TRL canary
  `iwopd-post10-trl-canary-20260813` from retained runtime candidate rc6. It
  reproduced `on_policy_loss=-inf` after 749 finite teacher-scored tokens and
  retained two native traces, ruling out image drift and exposing the actual
  denominator bug: the base Trainer counted the prompt-only raw batch before
  on-policy generation and passed `num_items_in_batch=0`.
- [x] (2026-08-13) Fixed IW-OPD generically by recomputing the global item
  count from buffered post-generation labels. The regression captures the
  zero Trainer-level count, a non-zero stamped count, and finite training loss;
  the full IW-OPD module passes 60 tests with one accelerator-only skip.
  Released immutable TRL post11 from
  `69cf80a7319079ec5523841553467e119ebc1cec` and retained exact development
  bytes through successful Posttrain workflow `31653581435`.
- [x] (2026-08-13) Selected post11 across Posttrain's dependency metadata,
  catalog lock, runtime profile, qualification binding, CI wheel receipt, and
  operator documentation. Exact-head Quality workflow `31653910418` passed.
  Protected runtime workflow `31653914990` then published and retained rc7;
  its TRL kind is
  `sha256:bc2c8a7cb184c55557aab353d9fe16513644411bce79d89d25746f9d2ae81b1d`
  and strict release validation passes with the retained generated lock and
  manifest installed.
- [x] (2026-08-13) Ran veRL dev2 canary
  `verl-dev2-sampling-canary-20260813` on the 96 GiB RTX PRO worker. Provider
  job `pt-c5ddca6668b22ee360c426fb` and Trackio run
  `e7778637e07649f8acd6a8210fe3ef77` succeeded. Reconciliation retained two
  native traces, the LoRA adapter, summary, retention manifest, and trace-sync
  receipt. Observatory resolves `min_p=0.01`, `repetition_penalty=1.1`, and
  `presence_penalty=1.5` for both environment and rollout inference. Both tiny
  canary trajectories truncated with zero reward variance, so this qualifies
  the backend/configuration contract and one completed optimizer step, not
  training-quality promotion.
- [x] (2026-08-13) Ran TRL post11 canary
  `iwopd-post11-trl-canary-20260813` on the 24 GiB RTX 4090. Provider job
  `pt-8151d73ebf6e805b2505754f` and Trackio run
  `f0c2996c4e0c4d5bb6a3f1d6846d6f45` succeeded. Reconciliation retained two
  native traces, 749 teacher-scored tokens, zero teacher failures, finite loss
  `-0.18772484362125397`, global optimizer step one, checkpoint/recovery,
  final LoRA adapter, and summary. The retained binding identifies
  `trl@1.9.2.post11`, source `69cf80a7319079ec5523841553467e119ebc1cec`,
  and dependency lock `c065acdefa105abe8870b62bca4d61ab70abe801f65ff56415cdef2ddf168183`.
- [x] (2026-08-13) Promoted the exact qualified TRL post11 and veRL dev2
  artifacts server-side from `carbonteq/dev` to `carbonteq/stable`. Workflows
  `31655218728` and `31655220576` re-downloaded the stable files, matched the
  retained wheel/source hashes, passed clean installs, and retained promotion
  receipts. The framework stable lock now records the same post11 hashes under
  stable URLs; its catalog digest is
  `195c5caff273ed4c5d34c6c20e92f5bbc9a4629d8e1196600beaef5f1cda13ee`.
- [x] (2026-08-13) Preflighted both requested work packages before GPU
  submission and caught stale project-local bindings: the TRL package still
  selected post9 and the veRL canary still selected dev1 even though the
  candidate dependency inputs were newer. Preserved the historical selections,
  added TRL `@5` and veRL canary `@2`, and moved the requested packages to the
  new immutable revisions.
- [x] (2026-08-13) Regenerated the veRL Python 3.13 runtime lock and exported
  constraints for `808923d487aa2c524fda02cf5289110541b4221f`. The new lock
  digest is `334831a7e6344bc8ea7cfb406b13a43da7e46da97f17faae4b8d416f30286494`;
  the exported constraints digest is
  `c12f960e7fb41901177e370e0f270e84e7f9662d5578c40b52b0d329240c7052`.
- [x] (2026-08-13) Materialized runtime candidate `0.3.16rc6` in workflow
  `31651704768`. Registry readback records TRL kind digest
  `sha256:90c24840335fb170f61cbcf761c950d8d47664d4cb7d64d494415ebc85f9fd9b`
  and veRL dev2 kind digest
  `sha256:bdd87bd97153a9f1fa64842aec2e29e918fbd40e2e5569f9c5648708eae64cf9`.
  The retained development-channel lock contains TRL post10, and the veRL
  manifest identity contains the exact dev2 source and lock digest.
- [ ] Run both explicitly scoped GPU canaries only from those candidate images
  and verify their retained evidence before any stable promotion.

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
- Observation: a job-layer repack does not replace third-party packages baked
  into a dependency-bearing runtime kind.
  Evidence: retry `iwopd-post4-capacity-smoke-20260813` used the new job image
  and Posttrain source but still raised `IWOPDConfig.__init__()` for `min_p`.
  Its parent `online-rl-trl-py312` kind was built from a profile pinned to
  `trl==1.9.2.post2`; the job layer installs first-party sources with
  `--no-deps`.
- Observation: the private OCI registry can exhaust its filesystem through
  superseded runtime manifests and failed resumable uploads even when source
  validation and image builds are valid.
  Evidence: candidates `31637453779` and `31638128689` failed while publishing
  the veRL kind, ultimately reporting `/var/lib/registry ... no space left on
  device`. The second failure left 4.1 GiB in two incomplete veRL upload
  sessions. After removing only those incomplete sessions, deleting 118 exact
  superseded runtime manifests with zero overlap against the seven published
  manifests, and running Registry v3 native garbage collection, the host rose
  from 0 to 81 GiB free. Candidate `31638943255` then built, read back, and
  retained every runtime image successfully.
- Observation: the Qwen 3.5 0.8B student is a hybrid architecture: 20 of its
  24 layers use linear attention and only four use full attention. The broad
  `all-linear` LoRA choice can therefore cover a much different update surface
  than on a conventional attention-only model.
  Evidence: the pinned model `config.json` enumerates the 24 layer types, and
  the run reports the linear-attention fast path is unavailable.
- Observation: actual-job Docker publication was multi-stage and cache-aware,
  but concurrent callers for one publication key could race after both
  materialized the same context. The winner's cleanup made the loser fail as
  its smoke build began.
  Evidence: the reproduced error was `failed to get build context ... no such
  file or directory`; the publisher now serializes each key and rechecks the
  verified receipt inside the lock. A focused lock regression proves the
  follower cannot enter until the producer releases it.
- Observation: veRL's `DataProto.reorder` assumed a nested tensor's recorded
  ragged axis was also its storage ragged axis. That is false for repaired
  three-dimensional `position_ids`: the desired per-sample sequence axis is
  two, while `torch.nested.as_nested_tensor` stores the channel axis as ragged.
  Evidence: the failed remote worker raised `split_with_sizes ... input tensor's
  size at dim 1 ... got [4, 4]`; the CPU regression now exercises the repaired
  position-id path and 39 focused upstream tests pass.
- Observation: a framework-level pin does not supersede a project overlay.
  Evidence: the Lab `iwopd-sampling-qualification.yaml` still selected
  `trl@1.9.2.post4` after the root lock moved to post8; read-only
  work-package planning exposed the stale binding before it could enter a GPU
  job.
- Observation: finite sampled behavior-policy and teacher log probabilities do
  not prove the final student objective is representable in float32.
  Evidence: post8 retained 749 finite rollout log probabilities in
  [-7.6603, 0], no teacher failures, mean absolute advantage 0.1650, and
  weights in [1.00005, 1.5], yet the aggregate student objective was `-inf`.
- Observation: the retained-fork publishers and runtime-lock materializer
  contradicted the documented candidate channel by hard-coding `stable`.
  Evidence: `docs/publishing.md` requires candidate files in `carbonteq/dev`,
  while all three retained publishers and the runtime-lock selector named only
  `carbonteq/stable`; workflow `31646648237` therefore sent TRL post9 directly
  to stable.

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
- Decision: Rebuild the parent runtime kind through Posttrain's manual
  runtime-image candidate workflow before retrying the GPU canaries.
  Rationale: installing a new application layer cannot change the immutable
  TRL distribution in the parent kind. The candidate workflow materializes
  the exact internal wheel receipt, reads OCI digests back from the registry,
  and emits the matching image manifest.
  Date/Author: 2026-08-13 / Codex
- Decision: recover OCI capacity by exact manifest retention analysis plus
  native Registry v3 collection, not by broad repository or blob deletion.
  Rationale: runtime parents are shared dependencies and actual job images are
  evidence-bearing. The recovery retained every manifest named by the current
  generated manifest and excluded all actual-job repositories; the delete
  plan and receipts are stored under the machine-local operations record.
  Date/Author: 2026-08-13 / Codex
- Decision: retain the base → kind → job image hierarchy and improve its
  publication and worker-cache behavior rather than collapsing it into one
  image per job.
  Rationale: kind images isolate expensive third-party dependency layers.
  Warm job packs take tens of seconds and workspace wheel assembly takes about
  one second; rebuilding the ML stack for each source/configuration change
  would make the usual path slower. The lock removes the observed duplicate
  work; worker prewarming is the next separately measurable cold-start change.
  Date/Author: 2026-08-13 / Codex
- Decision: instrument the existing TRL canary before narrowing the hybrid
  Qwen LoRA surface.
  Rationale: broad `all-linear` adaptation may include numerically fragile
  linear-attention projections, but changing it before retaining the affected
  gradient evidence would conflate diagnosis and mitigation.
  Date/Author: 2026-08-13 / Codex
- Decision: use Posttrain's repository-owned retained-asset workflow for veRL
  just as for TRL and Trackio.
  Rationale: it makes the manual internal publication independently verifiable
  while keeping the private-index capability outside the fork. The runtime
  kind may move to the new source revision only after the release receipt and
  stable-index install succeed.
  Date/Author: 2026-08-13 / Codex
- Decision: treat the protected Posttrain runner as a private-publication and
  registry-qualification boundary, not as a generic build farm.
  Rationale: maintainers build, test, hash, tag, and release fork artifacts
  locally.  The runner receives only immutable release identities and hashes
  when private-index publication or registry-backed qualification is required.
  Date/Author: 2026-08-13 / Codex
- Decision: diagnose non-finite IW-OPD objectives at the token-loss boundary
  before attempting a model or LoRA policy change.
  Rationale: retained evidence rules out rollout sampling and teacher-scoring
  corruption. The new generic check identifies whether an otherwise finite
  student logprob, teacher value, behavior-policy value, advantage, or weight
  causes the reduction overflow without silently clipping the objective.
  Date/Author: 2026-08-13 / Codex
- Decision: treat direct-to-stable retained-fork publications as rejected
  audit artifacts and restart from a new immutable development candidate.
  Rationale: stable is immutable and cannot be repaired in place. Candidate
  qualification must resolve the development-channel bytes, and only a
  hash-verified server-side promotion may make those bytes stable.
  Date/Author: 2026-08-13 / Codex

## Outcomes & Retrospective

Work in progress. The expected outcome is a development-channel immutable TRL
candidate and a Posttrain runtime lock that construct the real config
successfully, followed by the two requested targeted canaries. Only unchanged,
hash-verified candidate bytes may then be promoted to stable. Fork release
creation remains manual; Posttrain owns the narrow private-index workflows.

## Context and Orientation

Posttrain models one rollout policy as `PolicySampling` in
`packages/train/src/posttrain/train/online_rl.py`. Its TRL adapter builds the
keyword arguments in
`packages/train/src/posttrain/train/backends/trl/distillation.py` and constructs
`IWOPDConfig` immediately before `IWOPDTrainer` starts. The selected controls
include temperature, top-p, top-k, min-p, repetition penalty, and presence
penalty.

The branch currently names `trl==1.9.2.post9` in `uv.lock`, but that version
was direct-published to stable by the flawed retained-asset workflow and is not
an accepted candidate input. The next unique post release must be published to
`carbonteq/dev`; its generated candidate lock, not the current stable lock,
will select the runtime bytes. Its maintained source is
`/home/hammad/projects/trl-1.9-upgrade`, branch `v1.9.2.post5-release`.
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

Revision 2026-08-13: the generated runtime manifest is now materialized from
successful candidate `31638943255` after a retention-safe private-registry
recovery. The remaining work is packaging and executing the two intended
remote canaries against those digests.

Revision 2026-08-13: image-path review retained the multi-stage topology,
fixed concurrent same-key publication, and identified worker kind-image pull
as the cold-path optimization target. The canaries now also retain precise
non-finite-gradient evidence if training fails.

Revision 2026-08-13: the veRL retry exposed a generic nested-tensor axis
selection defect rather than a sampling-policy violation. Its immutable dev2
release is ready for repository-owned internal publication and a subsequent
runtime-kind rebuild; the TRL retry now exposes `on_policy_loss=-inf`, which
remains under ingress/loss-path investigation before another GPU retry.

Revision 2026-08-13: corrected the retained-fork publication contract. The
previous direct-to-stable TRL post9 publication is rejected as an immutable
audit artifact; new candidates must flow GitHub Release → `carbonteq/dev` →
development-backed runtime lock and qualification → byte-identical server-side
promotion to `carbonteq/stable`.
