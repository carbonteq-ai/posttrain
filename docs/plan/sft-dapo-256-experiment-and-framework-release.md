# Publish the RL framework and qualify SFT-backed OLMo 3 GRPO

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current after every milestone.
The repository has no `.agents/PLAN.md`; this file follows the contract in
`docs/templates/PLAN.md` and the frozen baseline in `docs/post-training/README.md`
and documents the missing repository-local plan instructions explicitly.

## Purpose / Big Picture

This work answers two related but separate questions in a deliberate order.
First, it turns the reviewed framework, tracking and Observatory changes into a
reproducible release that an independent project can install and use without
this checkout. After that exact stable release is installed into Ambient Agent,
it establishes whether the named OLMo 3 GRPO recipe can learn from the existing
SFT LoRA adapter under the high-concurrency RTX PRO target, using a
small run that is cheap to retry and rich enough to detect a broken learning
signal.

The current 256-concurrency run is not the final algorithm experiment: its
selected project training settings enable DAPO and truncation handling but do
not explicitly set `advantage_scaling: none`, so it still uses the compatibility
default group scaling. It must not be presented as evidence for the corrected
DAPO recipe. The release is also not allowed to start from the current dirty
checkout: release checks must pass on one intentional commit, then the
candidate and final protected workflows must build, qualify, and publish exact
bytes.

This work does not change the frozen post-training product meaning. It makes
the existing artifact, observation, execution and read-product responsibilities
enforceable at release time and changes implementation only after the proposed
ADR is accepted.

## Progress

- [x] (2026-08-09) Read the canonical post-training baseline and plan template;
  confirmed the product boundary is frozen and `train.grpo` selects DAPO through
  `GRPOSettings.algorithm`.
- [x] (2026-08-09) Audited the active experiment inputs. The SFT adapter,
  environment revision, framework source, image digest, and remote run IDs are
  recorded below; the active settings omit `advantage_scaling: none`.
- [x] (2026-08-09) Audited the release state. `release/manifest.toml` remains
  `0.3.2`, tag `v0.3.2` exists, `main` is at `116b1fd1`, and the working tree
  contains 69 dirty paths.
- [x] (2026-08-09) Ran `uv run --no-sync posttrain-release check`; it fails on a
  stale `packages/catalog/src/posttrain/catalog/base/locks.toml` digest.
- [ ] Add or select a dedicated project settings revision with
  `advantage_scaling: none`, while retaining scalar DAPO rewards, asymmetric
  clipping, truncation masking, no dynamic sampling, and the SFT LoRA input.
- [ ] Stop or mark the current invalid 256-concurrency probe before its first
  optimizer update; retain its partial evidence as a rejected diagnostic.
- [ ] Run the corrected two-step probe and verify actor/sampler parity, reward
  spread, advantage statistics, truncation exclusion, optimizer updates,
  checkpoint/LoRA artifact output, and tracking finalization.
- [ ] Inventory the dirty framework tree and create one intentional release
  commit containing only reviewed framework changes and regenerated release
  inputs; leave unrelated edits untouched.
- [ ] Choose the next unused release version, update `release/manifest.toml`,
  regenerate dependency locks, and pass the complete local quality ladder.
- [x] (2026-08-09) Prepared the `0.3.3` release branch and draft PR, passed
  the local quality ladder and exact-SHA quality CI, and resolved the Trackio
  post10 wheelhouse asset required by the protected workflow.
- [x] (2026-08-09) Published Trackio
  `carbonteq-v0.31.5.post11` from exact source `7a2b885b...`, including the
  missing-completed-upload recovery fix and complete package/frontend tests.
- [x] (2026-08-09) Stopped release retries after candidate `31281418857`
  reached the real dstack canary and exposed a deployed Trackio artifact
  compatibility failure. Recorded the cross-system cause and proposed
  [ADR 0014](../decisions/0014-attested-release-promotion-graph.md).
- [x] (2026-08-09) Manually published and deployed Trackio
  `0.31.5.post11`, then qualified scalar history and a cache-independent S3
  artifact upload/download against the live service. Producer run
  `50f9a44eda0e4588b8b5ac4d88c8d8c8` and consumer run
  `829d2776062a4825bbf3c228dfceac89` read back artifact SHA-256
  `bf6e71abcbd1631441dbc2b45610f9d4c59648c6dc1039dc0106d930d7fc143c`.
- [x] (2026-08-09) Manually published TRL `1.9.2.post1` from source
  `a82ecebc0fa081efd58302a34a553445fc73271d`, retained wheel and sdist
  hashes, and changed Posttrain to consume the stable package instead of a Git
  checkout.
- [x] (2026-08-09) Added the two-phase runtime-lock materialization boundary.
  Pull-request CI validates authored dependency receipts while keeping the last
  published OCI manifest strict; the protected candidate materializes the
  internal wheel URLs, rebuilds affected images, then applies strict validation
  and retains both generated lock and manifest.
- [x] (2026-08-09) Passed the complete source ladder: release metadata and
  repository audits, Ruff lint/format, Pyright, eight import contracts, 1,081
  tests with 21 expected skips, and `git diff --check`. Installed the exact
  published TRL `1.9.2.post1` extra and passed the focused TRL compatibility
  suite (8 passed, 1 environment-dependent skip).
- [ ] Re-run the protected release candidate using the accepted manual Trackio
  and TRL receipts. Require dev-index/OCI readback, clean install, packed dstack
  job, Trackio artifact round trip, and Observatory readback before merge.
- [ ] Merge the release PR and dispatch the final workflow for the exact merged
  SHA and accepted candidate materialization. Create no tag before stable
  readback succeeds.
- [ ] Update the local Ambient Agent project to the exact stable Posttrain
  release, replace its direct legacy TRL/Trackio pins with the release-resolved
  dependency graph, and revalidate its SFT-backed online-RL work packages.
- [ ] Add a new versioned Ambient Agent `algorithm: olmo3` setting and work
  package after that dependency update. Preserve the historical DAPO settings
  and runs; do not relabel them as OLMo 3 evidence.
- [ ] Pack the resolved OLMo 3 job and run a bounded two-step SFT-LoRA canary
  before scheduling the longer campaign.
- [ ] Record final experiment and release receipts, update this plan’s outcome,
  and report the tag, release URL, package/image digests, and remaining gates.

## Surprises & Discoveries

- Observation: The active 256-concurrency package is built from the current
  framework source and the SFT adapter, but its selected settings do not enable
  the intended advantage fix.
  Evidence: the packaged project catalog contains `algorithm: dapo`,
  `mask_truncated_completions: true`, and DAPO clip bounds, but no
  `advantage_scaling`; the corrected `advfix` settings entry does contain
  `advantage_scaling: none`.
- Observation: The runtime image is content-addressed and contains the current
  framework source, so changing the project catalog or settings requires a new
  actual-job package digest even when the framework release image is reused.
  Evidence: current run image
  `registry.lan/carbonteq/posttrain-job@sha256:5a158389834c089edc4eb5743b559324a9b238e5eca71a13c258c33c39f676b4`;
  embedded `grpo.py` matches the working-tree source SHA
  `8277ce5cc5853db4048fbbaa64b1ccb6f35bf96c797658253b553e9a8efefae7`.
- Observation: The framework release checker currently fails before build or
  publication because one generated catalog lock digest is stale.
  Evidence: `posttrain-release check` reports `locks.toml` digest
  `0f122dfc...` but expects `2c482c10...`.
- Observation: The release source is not clean enough to tag safely.
  Evidence: `git status --porcelain` reports 69 modified or untracked paths,
  including train, Observatory, tracking, docs, packaging, and release tests.
- Observation: Prior SFT-backed DAPO runs have both positive and negative reward
  trajectories, so a negative short run is not by itself proof that the SFT
  adapter is unusable.
  Evidence: completed two-step Trackio runs recorded reward means
  `+0.647602 → +0.667955` and `−0.118092 → −0.212115` from the same SFT source
  tree. The corrected probe must therefore verify loading, parity, and learning
  telemetry before interpreting reward direction.
- Observation: The first protected `0.3.3rc1` build reached distribution
  receipt verification but stopped while assembling the wheelhouse because the
  repository did not contain the developer-facing `docs/release-and-consumption.md`
  file that the release script embeds as its README.
  Evidence: candidate workflow `31279099084` failed at the `cp` step in
  `scripts/release/build-python-distributions`; the guide is now present and
  release tests/checks pass locally.
- Observation: After the guide fix, the candidate exposed a second release
  pipeline defect: OCI publication happened after wheelhouse staging. The
  consumer virtualenv therefore installed a wheelhouse containing the old
  runtime-image manifest even though the registry had newly built images.
  Evidence: candidate `31280341495` refused the transform canary because the
  installed manifest expected lock `d26cadd4…` while its pinned registry image
  still carried `a9a2f0c8…` / framework `0.3.2`.
- Observation: Moving OCI publication before wheelhouse construction was not
  sufficient because the distribution builder deliberately stages from
  `git archive HEAD`, which excludes the generated `published.toml` mutation.
  Evidence: candidate `31280782204` published fresh transform digest
  `c26e6451…`, but the clean consumer still packed old digest `c772968f…` and
  the dstack canary reported framework `0.3.2` / lock `a9a2f0c8…`.
- Observation: The correct ownership boundary is a narrow overlay of the
  generated runtime manifest after immutable source staging, not a broad dirty
  checkout copy. The overlay is covered by a release regression test and will
  be requalified in the next protected candidate.
- Observation: The narrow checkout overlay is a useful regression repair but
  not a sufficient release architecture. It does not attest who generated the
  manifest, which dependency deployment the canary exercised, or whether the
  runtime-image dependency lock matches the Python wheelhouse.
  Evidence: candidate `31281418857` successfully consumed the refreshed image
  manifest and launched the packed job, then failed while Trackio committed an
  artifact manifest whose referenced blob was absent from the deployed server.
- Observation: a dedicated Trackio-owned release workflow is not required to
  unblock this release because the operator completed the publication and
  deployment manually and retained live readback evidence. Automation remains
  a follow-up; Posttrain consumes and verifies Trackio rather than publishing it.
  Evidence: the live post11 producer/consumer qualification succeeded only
  after deleting the producer cache, proving the artifact was served by the
  deployed S3-backed path rather than local residue.
- Observation: Trackio dependency versions currently have more than one source
  of truth across package metadata, `uv.lock` and runtime-image profiles/locks.
  A framework release can therefore build a client wheelhouse and an execution
  image against different Trackio versions unless CI rejects the drift.
- Observation: Observatory is named in the supported release workflow but is
  not yet qualified as a deployed read product with an immutable image/config
  receipt. A local import or package test cannot prove production readback.
- Observation: Ambient Agent cannot select `algorithm: olmo3` before the
  Posttrain release is installed. Its current project metadata still resolves
  `posttrain==0.3.2`, Trackio post10 and the legacy TRL Git commit `91b0ce...`,
  none of which contains the named Posttrain OLMo 3 selection contract.
  Evidence: `/home/hammad/projects/ambient-agent/pyproject.toml` and `uv.lock`
  retain those exact constraints. Changing only the project YAML would fail
  catalog decoding or execute the wrong trainer implementation.

## Decision Log

- Decision: Treat the current 256-concurrency run as a rejected diagnostic, not
  as corrected-DAPO evidence, until `advantage_scaling: none` is resolved in the
  selected settings.
  Rationale: otherwise the comparison confounds the algorithm fix with the
  previous group-normalized behavior.
  Date/Author: 2026-08-09 / Codex.
- Decision: Keep the DAPO reward scalar and retain component rewards only as
  diagnostic telemetry.
  Rationale: this is DAPO, not GDPO; splitting objectives would change the
  experiment rather than repair the scalar learning signal.
  Date/Author: 2026-08-09 / Codex.
- Decision: Use the existing SFT LoRA adapter as the policy input and require a
  LoRA adapter/checkpoint output, not a full model export.
  Rationale: the question is whether DAPO can improve the trained policy, and
  the recovery contract must preserve adapter, optimizer, scheduler, trainer,
  and RNG state.
  Date/Author: 2026-08-09 / Codex.
- Decision: Use a two-step, 32 prompt groups × 4 generations probe before any
  longer run. Keep environment and serving concurrency at 256 for the capacity
  test, but retain global training batch 128 and the existing safe microbatch /
  accumulation contract.
  Rationale: this isolates setup and learning-signal correctness while keeping
  the requested rollout capacity; a two-step run is not a quality claim.
  Date/Author: 2026-08-09 / Codex.
- Decision: Release only from one clean, reviewed commit and use the existing
  protected candidate/final workflows. Do not hand-build a tag from a dirty
  tree or publish an unqualified wheelhouse.
  Rationale: the release workflow is designed to preserve exact source,
  dependency, image, and distribution receipts and to create the final tag last.
  Date/Author: 2026-08-09 / Codex.
- Decision: Make the release-consumption guide a first-class source document
  and wheelhouse README rather than teaching consumers from a workflow-only
  implementation detail.
  Rationale: the release script intentionally embeds this guide in every
  wheelhouse; omitting it makes a technically built release unusable to an
  independent consumer and caused the candidate build failure.
  Date/Author: 2026-08-09 / Codex.
- Decision: Publish changed OCI inputs before building the Python wheelhouse in
  the candidate workflow.
  Rationale: `published.toml` is packaged inside `posttrain-runtime-images`; a
  clean consumer can only select the newly qualified image digests when the
  wheelhouse is built after image publication. The canary must fail closed on
  drift rather than bypassing the check with `--build-missing`.
  Date/Author: 2026-08-09 / Codex.
- Decision: Keep `git archive` as the release source boundary, but explicitly
  overlay the generated runtime-image manifest into the staged tree before
  building distributions.
  Rationale: this preserves protection against arbitrary dirty state while
  ensuring the wheelhouse and OCI registry carry identical image digests.
  Date/Author: 2026-08-09 / Codex.
- Decision: Supersede the checkout-copy mechanism as the production boundary
  with an explicit release materialization receipt. `git archive` remains the
  committed-source input; only hash-declared generated inputs are projected
  into the staged tree.
  Rationale: a generated file is valid release input, but it is not source and
  must not enter the build through ambient working-tree mutation.
  Date/Author: 2026-08-09 / Codex.
- Decision: Require maintained dependencies to publish, deploy and qualify
  themselves before the Posttrain candidate begins. Posttrain verifies their
  receipts and never publishes Trackio as an incidental workflow step.
  Rationale: package availability, server deployment and live compatibility are
  distinct states with different owners and credentials.
  Date/Author: 2026-08-09 / Codex.
- Decision: Treat Trackio artifact round trip and Observatory readback as
  first-class candidate gates against the same packed run.
  Rationale: a remote job that reaches terminal provider state is not a
  supported Posttrain result if its artifacts cannot be committed or its
  evidence cannot be read through the product surface.
  Date/Author: 2026-08-09 / Codex.
- Decision: Finish the framework release before updating Ambient Agent or
  submitting further DAPO runs.
  Rationale: experiments must consume the exact stable framework and dependency
  graph being qualified; running from the release branch would recreate the
  provenance ambiguity this release is intended to remove.
  Date/Author: 2026-08-09 / Codex.
- Decision: Use the named `Olmo3GRPOConfig` recipe for the next post-release
  Ambient Agent learning canary rather than approximating it with another DAPO
  settings combination.
  Rationale: active refill and token-level TIS are trainer/runtime behavior, not
  aliases for `loss_type`; selecting `algorithm: olmo3` binds the complete
  recipe and its invariant checks while retaining the common `GRPOTrainer`.
  Historical DAPO runs remain valid comparisons under their original labels.
  Date/Author: 2026-08-09 / Codex.

## Outcomes & Retrospective

The post-release OLMo 3 canary remains a separate, explicitly unfinished
qualification item. The framework release is prepared as `0.3.3` on branch
`codex/release-0.3.3` with draft PR `carbonteq-ai/posttrain#34`. Four candidate
passes exposed a missing consumer guide, incorrect OCI/build ordering, an
implicit generated-manifest input, and an unqualified deployed Trackio artifact
path. Trackio post11 and TRL 1.9.2.post1 are now manually published with exact
receipts, the live Trackio artifact path passes, and the runtime dependency lock
has a deterministic candidate materialization step. The next action is local
validation and a fresh protected candidate. No final tag has been created,
Ambient Agent has not been updated, and no post-release OLMo 3 run has been
submitted.

## Context and Orientation

The framework is a Python 3.12/3.13 `uv` workspace. `packages/train` owns the
backend-neutral training settings and TRL adapter, `packages/jobs` owns the
standard job definitions, `packages/catalog` owns framework selections and
dependency-lock metadata, `packages/runtime-images` owns framework image
definitions and the generated published-image manifest, and `apps/observatory`
reads provider-neutral evidence. The `posttrain-release` package in
`apps/release` is framework-owner tooling and is intentionally not installed by
consumer projects.

The experiment lives in the sibling project `/home/hammad/projects/ambient-agent`.
Its work package binds the SFT adapter, a Verifiers environment, a training
selection, an inference binding, and an RTX PRO execution target. A run is one
execution of that package; Trackio and dstack IDs are evidence references, not
the work-package identity.

The active diagnostic run is:

    framework run: ambient-k1-dapo-rtxpro-sft-g4-seq256-2step-20260809
    dstack provider: pt-7cfc0eebe7f3a2ff28b2bd85
    Trackio: 1823843611f24ad4a71efdeb66240e5d
    image: registry.lan/carbonteq/posttrain-job@sha256:5a158389834c089edc4eb5743b559324a9b238e5eca71a13c258c33c39f676b4

The corrected run must preserve the SFT model variant and environment revision,
record the framework/project/configuration digests in its snapshot, and emit
the DAPO telemetry required by `docs/post-training/06-observation-and-lineage.md`.

## Plan of Work

First, finish the current release from the accepted manual dependency receipts.
Trackio post11 and TRL 1.9.2.post1 are immutable inputs; the protected candidate
must materialize their exact wheel URLs into the runtime constraint lock before
publishing images. Retain the generated lock and `published.toml`, apply both to
the release branch, rerun strict CI, and only then merge.

The corrected candidate must pass exact-SHA CI, registry and development-index
readback, clean index-only installation, a bounded packed dstack job, Trackio
artifact finalization and Observatory readback. The final workflow uses the
exact merged SHA and accepted materialization, verifies identical stable bytes,
and creates `v<version>` only after all gates pass. Update Ambient Agent to that
stable version, remove its old direct Posttrain/TRL/Trackio resolution, and
revalidate the SFT model, environment, inference and training bindings before
scheduling work.

Then add an additive, versioned Ambient Agent setting with `algorithm: olmo3`.
It must resolve the released `Olmo3GRPOConfig`: zero-gradient filtering with
bounded active refill, global token-level loss normalization, beta `0`, clipping
`0.20/0.272`, token-level TIS with no lower cap and upper cap `2.0`, and
mean-only group advantages. Keep the selected batch, generation, length,
learning schedule, truncation policy and active-refill budget explicit in the
project catalog. Do not mutate or rename an existing DAPO selection.

Bind the new setting to the retained 1,938-update SFT LoRA model variant and its
matching vLLM inference binding. Run exactly two logical steps before any longer
campaign. Retain reward components, group spread, active-sampling rounds,
generated/retained rows, advantage statistics, truncation semantics, TIS ratios,
clip fractions, entropy, rollout/runtime telemetry, actor/sampler parity,
optimizer completion, recovery checkpoint and LoRA artifact. Independently
recompute advantages and require a floating-point tolerance match before a
longer run is considered.

## Concrete Steps

Work from `/home/hammad/projects/rl` unless a command names the ambient-agent
checkout.

1. Refresh release state without mutating authoritative systems:

       git status --short
       uv run --no-sync posttrain-release check
       git show --stat --oneline HEAD

   Confirm the exact release branch, source SHA, candidate receipts and current
   deployed dependency identities. Do not infer deployment from a GitHub tag.

2. In the Trackio and `ai-infra` repositories, implement the protected,
   repository-scoped internal publication and deployment transaction. Produce a
   receipt binding post11 source, wheel/sdist hashes, internal-index readback,
   service image digest, deployed configuration identity and the dedicated S3
   artifact compatibility canary.

3. In Posttrain, add the materialization model and explicit `stage` input,
   migrate the candidate/final workflows to it, enforce the single dependency
   lock, and remove the temporary Posttrain-owned Trackio publisher. Add focused
   receipt, tamper, stale-input and dependency-drift tests before changing the
   workflow.

4. Push the reviewed release branch and wait for exact-SHA quality CI. Dispatch
   **Prepare release candidate** only after the dependency receipt is accepted.
   Inspect every gate result independently; a candidate is rejected if Trackio
   writes or Observatory readback fail even when dstack reaches terminal state.

5. Merge the release PR only after the candidate passes. Dispatch **Publish
   release** with the exact merged SHA and accepted materialization, verify
   stable readback, and create the tag/GitHub Release last. Never rebuild or
   repair a dependency inside final qualification.

6. Update `/home/hammad/projects/ambient-agent` to the exact stable framework
   release. Update `pyproject.toml` and `uv.lock` so Posttrain, Trackio and TRL
   resolve through the released framework graph rather than the current
   `posttrain==0.3.2`, Trackio post10 and direct legacy TRL Git pin. Validate the
   catalog and save the dependency receipt before editing the job selection.

7. Add a new Ambient `algorithm: olmo3` catalog entry and new work package. Bind
   it to `models/qwen3.5-2b-sft-10k-json@lora-v0`, the matching SFT-policy vLLM
   inference binding, the selected verifier environment and an effective batch
   equal to prompt groups times generations. Verify the resolved trainer config
   is `Olmo3GRPOConfig` and that the immutable recipe fields match the release.

8. Pack and run the two-step OLMo 3 canary. Save the package JSON, resolved
   inputs, dstack and Trackio identities, telemetry and final evidence query.
   Recompute mean-only advantages independently, verify actor/sampler parity,
   active-refill behavior and TIS bounds, and confirm both the recovery
   checkpoint and LoRA artifact before considering a longer run.

## Validation and Acceptance

The experiment is accepted only when the run snapshot resolves the SFT adapter,
the intended environment and target, and `algorithm: olmo3`; the resolved
configuration must prove active sampling, mean-only advantages, token-level
global normalization, zero KL, `0.20/0.272` clipping and token-level TIS capped
at `2.0`. Both logical steps complete rollout and actor-update phases; independent advantage
recomputation matches telemetry within the documented tolerance; truncated
completions contribute neither group statistics nor loss; actor and sampler
weights are equal at the parity gate; a recovery checkpoint and LoRA adapter
artifact are linked to the run; and Trackio finalization succeeds.

The release is accepted only when the maintained-dependency receipt proves the
exact internally published and deployed Trackio version; the materialization
binds source, locks and generated image evidence; the full quality ladder
passes; a clean environment installs exact index bytes without workspace/Git
sources; OCI digests match registry readback; the dstack, Trackio artifact and
Observatory readback gates succeed against one run; stable contains the exact
final files; and tag `v<version>` points to the verified merged commit. A
two-step probe remains diagnostic evidence, not a production-learning claim.

## Idempotence and Recovery

All catalog and plan edits are additive and versioned. Repacking after a config
change creates a new content-addressed actual-job image. Re-running the two-step
probe creates a new run identity and never overwrites prior evidence. If the
probe fails before its first checkpoint, record that no safe resume point exists;
otherwise resume only from its immutable training checkpoint and retain the
original run as the parent attempt.

Release staging is isolated and can be repeated against any manifest version.
If candidate qualification fails, publish no final tag and allocate the next RC
after the fix. If final promotion succeeds but GitHub finalization fails, retry
only tag/Release creation from the retained receipt. If the release-scope audit
cannot separate user edits safely, stop before commit and report the exact paths
instead of using `git reset`, `git clean`, or a broad destructive cleanup.

## Artifacts and Notes

Retain the corrected run’s package JSON, resolved catalog snapshot, dstack and
Trackio IDs, independent advantage-recompute transcript, checkpoint and LoRA
artifact references, and a concise metrics receipt. Retain the release branch
commit, `release/manifest.toml`, generated lock diff, wheelhouse receipt,
published image manifest, candidate workflow URL, final workflow URL, stable
readback hashes, and GitHub Release URL. Redact tokens, registry credentials,
and private certificate material from every report.

Current evidence anchors:

    current framework source grpo.py sha256:
    8277ce5cc5853db4048fbbaa64b1ccb6f35bf96c797658253b553e9a8efefae7
    SFT adapter source tree sha256:
    79b17299c7808b373eaa67f3c34153ab513e27d2f28105f0ef633c58cccaa7b7
    current release branch HEAD:
    c8adbc37 (codex/release-0.3.3)
    current target release:
    0.3.3 (not tagged or published)

## Interfaces and Dependencies

The experiment uses `posttrain.train.GRPOSettings.algorithm = "olmo3"`, the
released TRL `Olmo3GRPOConfig`, and the shared adapter in
`packages/train/src/posttrain/train/backends/trl/grpo.py`; the project catalog
and work-package files remain in `/home/hammad/projects/ambient-agent/.posttrain`.
The run lifecycle uses dstack through `packages/execution-dstack` and evidence
through the provider-neutral Trackio adapter. The release uses
`apps/release/src/posttrain_release`, `release/manifest.toml`,
`packages/catalog/src/posttrain/catalog/base/locks.toml`,
`packages/runtime-images/src/posttrain/runtime_images/published.toml`, and the
protected workflows `.github/workflows/release-candidate.yml` and
`.github/workflows/release.yml`.

The framework release must preserve the canonical package boundaries: common
contracts remain backend-neutral; train, eval, and serve remain independent;
Observatory reads providers only through tracking adapters; and framework
runtime images are distinct from private actual-job images. Dependency pins are
immutable and the lockfile is regenerated, not hand-edited.

Revision note (2026-08-09): created this combined living plan after discovering
that the active 256-concurrency run did not select the explicit advantage fix
and that the framework release checker failed on a stale generated lock digest.
Revised it after the OLMo 3 recipe was implemented so the next experiment first
updates Ambient Agent to the stable framework, then adds an additive OLMo 3
selection and runs a two-step SFT-LoRA canary before any longer campaign.
