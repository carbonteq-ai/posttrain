# Qualify corrected SFT-backed DAPO and publish the RL framework release

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current after every milestone.
The repository has no `.agents/PLAN.md`; this file follows the contract in
`docs/templates/PLAN.md` and the frozen baseline in `docs/post-training/README.md`
and documents the missing repository-local plan instructions explicitly.

## Purpose / Big Picture

This work answers two related but separate questions. First, it establishes
whether the corrected DAPO implementation can learn from the existing SFT LoRA
adapter under the high-concurrency RTX PRO target, using a small run that is
cheap to retry and rich enough to detect a broken learning signal. Second, it
turns the reviewed framework changes into a reproducible RL-framework release
that an independent project can install and use without this checkout.

The current 256-concurrency run is not the final algorithm experiment: its
selected project training settings enable DAPO and truncation handling but do
not explicitly set `advantage_scaling: none`, so it still uses the compatibility
default group scaling. It must not be presented as evidence for the corrected
DAPO recipe. The release is also not allowed to start from the current dirty
checkout: release checks must pass on one intentional commit, then the
candidate and final protected workflows must build, qualify, and publish exact
bytes.

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
- [ ] Dispatch the protected release-candidate workflow from the exact branch,
  verify the dev-index wheelhouse and OCI receipts, then dispatch the final
  workflow for the exact merged SHA. Do not create a tag before final stable
  readback succeeds. Candidate `31280782204` reached OCI publication and clean
  consumer installation but stopped at the canary because the wheelhouse still
  contained the pre-publication image manifest; the builder overlay fix is
  queued for the next candidate.
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

## Outcomes & Retrospective

The corrected DAPO probe remains a separate, explicitly unfinished
qualification item. The framework release is now prepared as `0.3.3` on
branch `codex/release-0.3.3` (current fix commit follows the release-guide fix)
with draft PR `carbonteq-ai/posttrain#34`. Local validation and exact-SHA
quality CI pass. Protected candidates caught and fixed two release defects:
the missing wheelhouse README and the generated OCI manifest being omitted by
`git archive` staging. Candidate `31280782204` therefore remains rejected at
the canary; the next candidate must prove that the clean consumer carries the
fresh image digests before merge or final promotion. No final tag has been
created.

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

First, inspect the current run and cancel it before an optimizer update if it is
still active. Preserve its Trackio and dstack IDs as a rejected diagnostic.
In the ambient-agent overlay, select the existing `advfix` settings or add a
new versioned settings entry whose full contract is explicit:
`algorithm: dapo`, scalar reward bridge, `advantage_scaling: none`, no dynamic
sampling, clip low/high `0.20/0.28`, zero KL, one on-policy iteration,
truncation masking with soft overlength shaping, and the existing LR and batch
contract. Update the 256-concurrency work package to that settings ID without
changing the SFT adapter, environment revision, or target. Validate and pack a
new actual-job image; never reuse the old package digest after changing config.

Run exactly two logical steps. For every step, retain rollout-level scalar
rewards and component diagnostics, group reward spread and zero-variance rate,
advantage mean/std/absolute mean and positive/negative/zero fractions,
corable/truncated fractions, importance-ratio range and clamp fraction, clip
fractions, entropy, rollout throughput, KV-cache peak, VRAM capacity, GPU
compute, actor/sampler parity, and optimizer-update completion. Recompute the
advantages independently from retained rewards and require a floating-point
tolerance match. Confirm that truncated completions have zero advantage and do
not change the group statistics. Confirm that the produced recovery checkpoint
and LoRA artifact are present and that a restart can identify the checkpoint;
do not resume from a promoted model artifact in place of a training checkpoint.

Only after the two-step probe is valid, isolate the framework release scope.
Review each dirty path against the current release objective, keep unrelated
user edits out of the release commit, regenerate the catalog lock digest, and
run the checker and full validation ladder. Select the next unused version after
checking the existing dev/stable indexes; prepare the manifest in a release
branch, not directly on `main`.

The candidate workflow must pass exact-SHA CI, build and receipt-check the
wheelhouse, publish only to the development index, verify a clean index-only
consumer install, qualify changed OCI images and the bounded dstack canary, and
retain the generated image manifest. The final workflow must use that exact
merged SHA and candidate run, verify stable readback of identical bytes, and
create `v<version>` only after all evidence passes.

## Concrete Steps

Work from `/home/hammad/projects/rl` unless a command names the ambient-agent
checkout.

1. Refresh state without mutating anything:

       git status --short
       uv run --no-sync posttrain-release check
       git show --stat --oneline HEAD

   The checker is expected to fail until the generated lock digest is repaired;
   do not bypass it.

2. Inspect and, if necessary, stop the invalid diagnostic through the framework
   lifecycle command. Verify the provider status first and retain the IDs above.
   A cancelled diagnostic is evidence of configuration invalidity, not a failed
   model result.

3. In `/home/hammad/projects/ambient-agent`, update only the project catalog and
   work-package references needed to select the corrected DAPO settings. Run
   `posttrain catalog validate`, `posttrain work-package validate`, and the
   project’s existing package plan command. Ensure the resolved selection prints
   `advantage_scaling=none` before submission.

4. Pack and run the two-step probe with the current source CLI and private
   registry configuration. Save the package JSON, resolved inputs, dstack run
   output, Trackio run ID, and the final evidence query under the ambient
   project’s ignored state or a retained release-evidence directory. Do not
   delete the rejected diagnostic until the corrected run has been inspected.

5. Recompute the DAPO advantage arrays from the retained scalar group rewards
   in a small independent test/helper. Compare the result with the telemetry
   emitted by the trainer and record the maximum absolute difference. Run the
   focused train tests, then the two-step remote probe; do not infer learning
   from reward movement alone.

6. Create a release branch from the reviewed base, stage only the intended RL
   framework changes, and update `release/manifest.toml` to the next unused
   release line. Run `posttrain-release lock-dependencies`, then
   `posttrain-release check`, `uv sync --all-packages --locked --python 3.13`,
   Ruff, Pyright, import-linter, the full pytest suite, and `git diff --check`.
   The source templates remain `0.0.0`; the manifest is the sole authored
   version and staging renders release metadata into a temporary tree.

7. Push the release branch and wait for exact-SHA quality CI. Dispatch “Prepare
   release candidate” with that branch, inspect its wheel, image, dstack, and
   Observatory receipts, and merge only the reviewed generated manifest.
   Dispatch “Publish release” with the exact merged SHA and candidate run ID.
   If a final workflow fails after retaining receipts, resume from that run ID;
   never rebuild different bytes under the same version.

## Validation and Acceptance

The experiment is accepted only when the run snapshot resolves the SFT adapter,
the intended environment and target, and `advantage_scaling: none`; both logical
steps complete rollout and actor-update phases; independent advantage
recomputation matches telemetry within the documented tolerance; truncated
completions contribute neither group statistics nor loss; actor and sampler
weights are equal at the parity gate; a recovery checkpoint and LoRA adapter
artifact are linked to the run; and Trackio finalization succeeds.

The release is accepted only when `posttrain-release check` reports the manifest,
all staged package metadata, dependency locks, and published image shape as OK;
the full quality ladder passes; a clean environment installs the exact staged
wheelhouse without workspace sources; candidate artifacts and OCI manifests
match receipts by hash; the dstack canary and Observatory readback succeed; the
stable index contains the exact final bytes; and tag `v<version>` points to the
verified merged commit. A two-step probe is diagnostic evidence, not a claim of
production learning quality; a longer qualification run remains a separate
decision.

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
    current framework HEAD:
    116b1fd1 (main, origin/main)
    current authored release:
    0.3.2 / tag v0.3.2

## Interfaces and Dependencies

The experiment uses `posttrain.train.GRPOSettings` and the TRL DAPO adapter in
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
