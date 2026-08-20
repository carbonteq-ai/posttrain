# Restore Gemma 4 multimodal SFT and qualify the PolicyPrism workflow

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`. It is self-contained so that a
contributor with only this repository and this file can resume the work without relying on chat history.

## Purpose / Big Picture

Posttrain can currently load Gemma 4 through Transformers' multimodal model class, but its TRL SFT adapter converts
every supervised example into text token IDs before TRL sees it. That conversion discards images and prevents a
user from training Gemma 4 on a complete document represented as ordered page images plus an instruction. The
actual-job packer also copies only `data.jsonl` and `manifest.json`, so local image paths cannot reach a remote GPU.

After this change, a project can package an ordered set of page images as immutable dataset assets, submit those
images with a conversational prompt and canonical JSON completion to Gemma 4 E4B, run a bounded TRL LoRA SFT job,
and retain a reloadable adapter with normal Posttrain evidence. A two-step Lab qualification will demonstrate that
the processor receives the ordered images, the model receives vision tensors, optimization produces finite metrics,
and the resulting adapter can be reloaded. A subsequent project work package can use the same capability for
PolicyPrism training. A managed `eval.domain` job will use the project's existing Verifiers taskset to score the
foundation and trained models under the same visual input and metric contract.

This work does not change the frozen product meaning of `train.sft`, datasets, actual-job images, or domain
evaluation. The canonical product documents already define supervised data as an immutable public training input,
allow declared data contracts, require every materialized dataset to be included in the actual-job image, and define
`train.sft` independently of one trainer's tokenization strategy. The implementation extends the supervised dataset
representation and package format without changing those meanings. If implementation reveals that multimodal
examples require a new product-level selection or job-kind meaning, stop and add a narrow amendment to
`docs/post-training/` before proceeding.

## Progress

- [x] (2026-08-20 08:43Z) Read the repository instructions, canonical product baseline, plan template, current SFT
  adapter, data contracts, dataset materializer, actual-job packer, runtime verifier, Gemma qualification catalog,
  pinned TRL metadata, and existing Verifiers evaluation definitions.
- [x] (2026-08-20 08:43Z) Located the current regression boundary: Posttrain pre-tokenizes SFT records into
  `input_ids` and `labels`, passes a tokenizer instead of a processor, and omits image assets from actual-job
  packages even though the pinned TRL fork contains a vision-language collator path.
- [x] (2026-08-20 08:43Z) Created this implementation-ready plan on the separate
  `feature/policyprism-gemma-vlm-sft` branch.
- [x] (2026-08-20 14:00Z) Added an exact-version multimodal prompt-completion collator test and an opt-in probe for
  the pinned Gemma E4B processor. Source inspection and remote Linux execution confirm that the pinned fork provides
  the required ordered-image, processor, completion-mask, and vision-tensor behavior. The base macOS environment
  still reports the CUDA-dependent probe skipped rather than substituting another TRL version.
- [x] (2026-08-20 09:04Z) Added frozen `SupervisedMedia` references, ordered media on `SupervisedExample`, strict
  path/digest/type validation, Hugging Face and NeMo round trips, and text-only row-shape compatibility. Validation
  passed: all 54 data tests, all 178 training tests, scoped Ruff, formatting, Pyright, and all import-boundary
  contracts. Eight existing or opt-in training tests skipped as designed.
- [x] (2026-08-20 15:41Z) Materialized, digest-locked, packaged, and runtime-verified optional dataset asset bundles.
  The manifest and job lock enumerate every regular file, runtime verification retains the exact-file-set invariant,
  and text-only lock payloads omit the new fields. Focused tests cover ordered assets, mutation, deletion, additions,
  and symlink rejection.
- [x] (2026-08-20 15:41Z) Added a dataset-modality-selected visual branch to the private TRL SFT adapter while
  preserving the renderer-pretokenized text path. Visual rows use the pinned prompt-completion shape, `AutoProcessor`,
  completion-only loss, no packing or padding-free mode, and no truncation by default; the saved adapter also retains
  its processor.
- [x] (2026-08-20 11:47Z) Added a license-safe two-page PPM fixture, visual dataset selection, E4B two-step SFT
  settings, language-only rank-8 LoRA binding, Lab work package, and candidate qualification inventory entry. Static
  catalog tests prove that the target expression includes language attention and MLP projections while excluding
  vision, audio, and embedding modules. Runtime parameter evidence rejects any trainable non-LoRA parameter or LoRA
  parameter outside the selected expression. Processor retention and multimodal adapter reload routing have focused
  tests. Work-package validation and semantic job planning both pass.
- [ ] (2026-08-20 11:47Z) The local repository validation ladder is complete except for the user-owned
  `git diff --check`: locked sync, Ruff, formatting, Pyright, and all eight import contracts pass. Focused
  qualification tests report 92 passes and one expected skip. The full suite reports 1,301 passes and 25 skips; the
  same three pre-existing macOS host-assumption failures remain (`/var` resolving to `/private/var` and a Linux CA
  bundle path absent on macOS). Linux execution of the pinned processor probe and the two optimizer steps is still
  required before qualification can be called complete.
- [x] (2026-08-20 14:00Z) Built and published immutable supervised and actual-job images, then completed real Gemma 4
  E4B visual SFT run `18a279b9-9d82-4f4e-9b13-539ee8e29bd1` on the assigned RTX worker. Both optimizer steps had
  finite losses and gradient norms, and Trackio reconciliation retained the final adapter as immutable version `v0`
  with content digest `76bc42ece7cda3050bb9f30bad3cd2d0647cd112f6de3d4b3d455e0b340ccc39`.
- [ ] (2026-08-20 14:32Z) Added and locally packaged the composed adapter-reload qualification. Its catalog model pins
  the exact Trackio artifact and Gemma base revision, and the job materializes it as `model_adapter` before one more
  visual SFT update. Focused catalog, work-package, and TRL tests report 47 passes and one expected local CUDA skip.
  Immutable package `1f15ce6ec3be6205e6a5ddf1bfa312332d7fa45bfa6466dca013d84ddae84343` and actual-job image
  `registry.lan/carbonteq/posttrain-lab/posttrain-job@sha256:7469f7439639dca6ea2d0428f1cb52e8a28bc97479caa5dd118a70fbff607d34`
  are ready for the remote reload run.
- [x] (2026-08-20 12:12Z) Closed two asset-verification regressions exposed by the first real pack attempt. The
  provider-neutral pack service and actual-job Dockerfile smoke verifier now include every locked dataset asset in
  their closed file sets and verify its size and digest. Focused validation invocations report 44 and 38 passes
  across execution-pack, runtime, and runtime-image tests. A local linux/amd64 OCI package for the two-step Gemma visual smoke passed
  actual-job qualification with package key
  `402b766c5531884d7b74925b82fc849e3b6895c1e066e98d7e4f96fe511047f0`.
- [ ] (2026-08-20 13:43Z) Submitted the first RTX Pro 6000 visual smoke as run
  `bca64b49-6488-40b0-9e53-241aecc28a64`. Placement succeeded, but the runtime stopped before model loading because
  macOS canonicalized the configured cross-host trust path from `/etc/posttrain/trust/internal-ca.pem` to
  `/private/etc/posttrain/trust/internal-ca.pem`. The narrow repair preserves configured trust-path spelling while
  still requiring an existing certificate file. Focused configuration and dstack tests pass, and `posttrain doctor`
  now reports the Linux worker path. Repackaging and remote retry remain pending.
- [ ] Package the PolicyPrism project training work package and managed Verifiers domain-evaluation work package.
- [ ] Run the frozen visual baseline and post-SFT evaluation on the sealed set, then record comparable metric deltas.

## Surprises & Discoveries

- Observation: The exact CarbonTeq TRL version selected by Posttrain already contains a vision-language SFT path.
  Evidence: `packages/train/pyproject.toml` pins `trl==1.9.2.post11` at source revision
  `69cf80a7319079ec5523841553467e119ebc1cec`; inspection of that revision shows processor loading, an image-aware
  collator, `image` and `images` dataset columns, and model inputs such as `pixel_values`.

- Observation: The effective regression is at the Posttrain-to-TRL boundary even if it originated during the TRL
  backend fork redesign.
  Evidence: `packages/train/src/posttrain/train/backends/trl/sft.py::run_sft` renders each record ahead of TRL,
  constructs rows containing only `example_id`, `input_ids`, and `labels`, sets `skip_prepare_dataset`, and passes
  `processing_class=tokenizer`. The image-aware TRL collator therefore cannot be selected.

- Observation: A trainer-only fix would pass local tests but fail in a real remote job.
  Evidence: `packages/execution-pack/src/posttrain/execution_pack/datasets.py::DatasetPackager.package` copies only
  `data.jsonl` and `manifest.json`. `apps/runtime/src/posttrain_runtime/execute.py::_verify_datasets` requires the
  observed dataset file set to equal those two locked paths. Source image paths on the developer's Mac would not
  exist inside the immutable actual-job image.

- Observation: Existing Gemma qualification intentionally proves language-model-only SFT rather than multimodal
  SFT.
  Evidence: `apps/lab/.posttrain/catalog/gemma4-qualification.yaml` contains E2B and E4B model selections but its
  current inference bindings set `text_only: true`, while its SFT qualification selections cover text-only 12B and
  31B runs.

- Observation: Gemma 4 E2B and E4B guidance supports a conservative first visual SFT smoke.
  Evidence: The manager-provided Unsloth Gemma 4 guidance recommends placing multimodal content before text,
  initially freezing vision layers, and fine-tuning language, attention, and MLP layers. That matches the repository's
  existing Gemma LoRA target pattern, subject to exact E4B module discovery.

- Observation: The repository's locked supervised dependency stack cannot be installed on the developer's macOS
  arm64 host.
  Evidence: the lock selects the CUDA build `torch==2.11.0+cu130`, for which no macOS arm64 distribution exists. The
  exact TRL compatibility test therefore skips without the optional dependency locally and must run in the Linux
  supervised runtime before Milestone 1 is closed.

- Observation: The complete test suite contains three host-specific failures on macOS that are unrelated to this
  branch.
  Evidence: two execution-configuration assertions assume Linux `/var` and `/etc` path behavior, and one remote
  builder test requires `/etc/ssl/certs/ca-certificates.crt`. Ruff, formatting, Pyright, import contracts, and all
  focused data, execution, pack, runtime, and train tests pass.

- Observation: The Lab treats every retained work package as an explicitly classified qualification gate or
  experiment.
  Evidence: adding the visual SFT work package without an entry in
  `apps/lab/src/posttrain_lab/qualification/gates.toml` caused the qualification inventory tests to fail as designed.
  The E4B visual smoke is now registered as an experimental candidate rather than silently expanding the active
  release gate set.

- Observation: Unit coverage of the dataset packager and runtime verifier did not cover the two other closed-world
  file-set checks used by `JobPackService` and the actual-job Dockerfile.
  Evidence: the first real visual pack failed first in `_validate_dataset_packages` and then in the Dockerfile smoke,
  because both expected only `data.jsonl` and `manifest.json` even though the manifest carried valid asset locks.
  Both paths now have asset-aware checks and the real local OCI pack succeeds.

- Observation: A configured certificate path can be a cross-host dstack instance contract, not merely a local file
  identity.
  Evidence: on macOS, `Path.resolve()` rewrote `/etc/posttrain/trust/internal-ca.pem` to the host-only path
  `/private/etc/posttrain/trust/internal-ca.pem`. dstack mounted that spelling on the Linux worker, where the runtime
  found no parseable certificate and failed before model loading. Preserving the validated absolute spelling keeps
  the shared `/etc/posttrain/trust/internal-ca.pem` worker contract intact without disabling TLS verification.

## Decision Log

- Decision: Implement and qualify the capability on the existing branch
  `feature/policyprism-gemma-vlm-sft` rather than modifying `main`.
  Rationale: The manager explicitly requested a separate Posttrain branch, and the user has already created that
  branch from current `origin/main`.
  Date/Author: 2026-08-20 / Codex and user.

- Decision: Treat a TRL fork modification as a gated outcome, not an assumed first edit.
  Rationale: Repository ownership rules put reusable trainer defects in the TRL fork, but the exact pinned fork
  already exposes the required VLM collator and processor path. The first integration test will determine whether a
  generic fork defect remains. If it does, this plan must be amended into a two-repository plan before edits are made
  to TRL.
  Date/Author: 2026-08-20 / Codex.

- Decision: Keep text-only and multimodal SFT as two translations behind the same public `train.sft` operation.
  Rationale: Modality is a property of the selected model and dataset, not a new training technique. Existing
  text-only callers must retain their renderer-built token masks and behavior.
  Date/Author: 2026-08-20 / Codex.

- Decision: Select the multimodal branch from declared dataset media and validate the selected model's image
  capability.
  Rationale: Gemma 4 also supports text-only training, so model family alone cannot select visual behavior. A visual
  dataset paired with a model that does not declare image support must fail before model loading.
  Date/Author: 2026-08-20 / Codex.

- Decision: Represent media in canonical data with immutable, ordered, project-independent references rather than
  PIL images, Torch tensors, Transformers objects, base64 payloads, or host-absolute paths.
  Rationale: `posttrain.data` is backend-neutral, and actual-job packages must remain portable and content-addressed.
  Tuple order will express PDF page order. Binary page data will live in a separate package asset tree to avoid JSONL
  bloat and duplicated memory.
  Date/Author: 2026-08-20 / Codex.

- Decision: Name the public reference `SupervisedMedia`, restrict its first schema to image assets below `assets/`,
  and use tuple position as the authoritative page order.
  Rationale: One normalized package-relative path plus MIME type and SHA-256 identity is sufficient for portable
  packaging. Tuple order avoids redundant page-order fields that could conflict, while optional JSON metadata can
  retain source page numbers for audit without controlling training order. Text-only exports omit the media field so
  their existing row shape remains unchanged.
  Date/Author: 2026-08-20 / Codex.

- Decision: Use conversational prompt-completion records for the visual TRL translation and train only on the
  completion.
  Rationale: The PolicyPrism instruction and images are conditioning input; the deterministic canonical JSON is the
  desired assistant output. Completion-only loss prevents the model from being trained to reproduce the input prompt.
  Date/Author: 2026-08-20 / Codex.

- Decision: Start Gemma 4 E4B qualification with BF16 LoRA on language-model attention and MLP projections while
  leaving vision layers frozen.
  Rationale: This matches the manager-provided multimodal fine-tuning guidance and minimizes risk for the first smoke.
  The exact regular expression will be accepted only after empty-weight or metadata-based E4B module discovery proves
  that it selects non-empty intended modules and excludes the multimodal towers.
  Date/Author: 2026-08-20 / Codex.

- Decision: Do not make `TrainingLoop.max_length` optional globally.
  Rationale: DPO, GRPO, SAMPO, and distillation validate and consume an integer loop length. The visual SFT path needs
  an SFT-specific no-truncation choice because truncation can remove image tokens. The implementation will add the
  smallest SFT-owned setting or backend translation necessary without weakening other techniques.
  Date/Author: 2026-08-20 / Codex.

- Decision: Keep PolicyPrism-specific prompts, labels, sealed-set policy, and Verifiers tasksets outside reusable
  framework packages.
  Rationale: Posttrain owns generic visual SFT and managed domain-evaluation capabilities. The PolicyPrism project
  owns its data and evaluation semantics.
  Date/Author: 2026-08-20 / Codex.

- Decision: Use tiny ASCII portable-pixmap images for the Lab qualification fixture and support their registered MIME
  type in the generic image-media contract.
  Rationale: the fixture remains license-safe, human-reviewable, deterministic, and patchable as source while PIL
  still exercises the same image loading and RGB conversion boundary used for production PNG, JPEG, and WebP pages.
  Portable pixmap is a qualification convenience, not a PolicyPrism production rendering recommendation.
  Date/Author: 2026-08-20 / Codex.

- Decision: Lock visual assets individually and also lock a canonical bundle digest, while omitting both fields for
  text-only datasets.
  Rationale: Individual locks identify the exact portable files used by training; the bundle digest makes the set and
  its deterministic path order part of package identity. Conditional serialization preserves the established
  two-file text dataset package shape.
  Date/Author: 2026-08-20 / Codex.

- Decision: Restrict the first visual translation to homogeneous datasets with one final textual assistant target.
  Rationale: TRL's pinned prompt-completion collator can then mask the complete conditioning prompt unambiguously.
  Mixed text/visual populations, tool records, or multiple trainable messages fail before model loading instead of
  silently changing loss semantics.
  Date/Author: 2026-08-20 / Codex.

## Outcomes & Retrospective

The exact pinned-TRL proof now exists as an executable collator test plus an opt-in exact Gemma E4B processor probe,
and source inspection of the pinned fork confirms the required ordered-image, processor, vision-tensor,
completion-mask, and no-truncation path. Their real execution remains a Linux-runtime gate rather than being
represented as a Mac result. The canonical data layer now has a backend-neutral ordered media contract with strict
portable identity and unchanged text-only exports. Dataset materialization, actual-job locks, copied build contexts,
and runtime verification now carry and verify the complete asset bundle. The TRL adapter selects its visual path from
dataset media, validates model capability before loading weights, re-verifies every asset, emits visual data-profile
evidence, and retains the processor with the trained artifact. The Lab now resolves a two-page E4B visual fixture,
two-step qualification settings, language-only LoRA binding, and classified candidate work package. Static checks,
processor retention, adapter reload routing, and intended-module enforcement pass. The corrected supervised runtime
and the first immutable actual job have now completed the two-step remote GPU qualification with a retained Trackio
adapter. A second immutable job is packaged to materialize that exact artifact and continue visual SFT for one step;
its remote success is the final generic reload gate before binding PolicyPrism's real training data. The only
full-suite failures are three unrelated Linux-path assumptions exercised on macOS.

## Context and Orientation

Posttrain expresses work through catalog selections, work packages, reusable job definitions, typed operation
requests, and immutable actual-job OCI images. An actual-job image is the complete content-addressed unit submitted to
Docker or dstack. It contains the project code, dependency closure, configuration, dataset snapshots, and the
framework job-kind image. A dataset snapshot is currently a canonical `data.jsonl` plus `manifest.json`.
Visual snapshots additionally own a digest-locked `assets/` tree enumerated by both the materialization manifest and
the actual-job package lock.

`packages/data/src/posttrain/data/models.py` defines framework-neutral records. `SupervisedExample` contains messages,
the ordered indices of messages that receive loss, optional tools, JSON metadata, and an ordered tuple of immutable
`SupervisedMedia` references. `packages/data/src/posttrain/data/adapters/huggingface.py` imports and exports supported
supervised JSONL shapes. `packages/data/src/posttrain/data/catalog.py::DatasetMaterialization` records the prepared
JSONL, manifest, optional asset locks, bundle digest, example count, and provenance.

`packages/execution-pack/src/posttrain/execution_pack/datasets.py` materializes selected datasets and copies them into
the actual-job build context. `packages/execution/src/posttrain/execution/job_package.py::DatasetPackageLock` records
the immutable dataset paths and digest in the package manifest. `apps/runtime/src/posttrain_runtime/execute.py` verifies
that the packaged bytes, paths, record count, and materialization manifest match the lock before invoking a job.

`packages/train/src/posttrain/train/backends/trl/sft.py::run_sft` is the private adapter from a backend-neutral
`SFTRequest` to TRL's `SFTTrainer`. The current text path asks `packages/train/src/posttrain/train/rendering.py` to
build token IDs and labels before constructing TRL. `packages/train/src/posttrain/train/backends/trl/common.py` owns
lazy imports, model and tokenizer loading, LoRA application, shared trainer arguments, callbacks, checkpoint recovery,
artifact retention, and runtime evidence. Framework imports remain lazy so the base package can be imported without
Torch or Transformers installed.

A processor is the Transformers object that combines a tokenizer with image preprocessing. A collator is the function
that takes several dataset rows and turns them into one padded model batch. For a vision-language model, the collator
loads or receives images, applies the processor, aligns image placeholders with the conversation, and produces tensors
such as `pixel_values`. The current Posttrain text path bypasses that behavior by passing already-tokenized rows.

`packages/jobs/src/posttrain/jobs/definitions.py::sft_definition` is already model-neutral and should not gain a Gemma
or PolicyPrism conditional. `apps/lab/.posttrain/catalog/gemma4-qualification.yaml` and the corresponding Lab work
packages are the appropriate place for exact Gemma E4B settings, target, fixture selection, and qualification policy.

`packages/jobs/src/posttrain/jobs/definitions.py::managed_evaluation_definition` already composes a managed model
endpoint with a Verifiers evaluation plan. The PolicyPrism project should use that standard `eval.domain` path. A
Verifiers taskset is the fixed collection of evaluation tasks and scoring behavior consumed by the Verifiers runtime.
It must send the same ordered images, prompt, and generation settings to the foundation and trained model so their
metric delta is comparable.

The exact Posttrain TRL dependency is declared in `packages/train/pyproject.toml` and locked in `uv.lock`. Repository
rules say generic trainer/runtime fixes belong in the sibling CarbonTeq TRL fork, while Posttrain-specific selections,
packaging, and evidence belong here. If the dependency probe proves a fork fix is required, use a sibling `../trl`
checkout, update its `CARBONTEQ_FORK.md`, add fork regression tests, publish the fork commit first, then update
`docs/tooling/trl/README.md`, `packages/train/pyproject.toml`, and `uv.lock` in Posttrain. Do not mix uncommitted
changes between repositories.

## Plan of Work

### Milestone 1: Prove the exact pinned dependency contract

Add a focused integration test under `packages/train/tests/` that exercises the exact installed TRL
`1.9.2.post11` behavior with a tiny in-memory RGB image and a conversational prompt-completion record. Avoid full
Gemma weights in the ordinary test suite. Use a small fake processor or the smallest stable processor fixture for the
unit boundary, and add a separately marked network/GPU probe for the exact E4B processor and model contract.

The test must prove that ordered images survive in the dataset row, the processor rather than a bare tokenizer is
passed to the trainer, the selected collator produces a vision tensor, prompt tokens receive the ignore label `-100`,
and completion tokens remain trainable. It must also prove that image content precedes the instruction in the
conversation, because this is required by the manager-provided Gemma guidance.

If the exact TRL fork passes, record that result here and make no TRL repository changes. If it fails due to generic
TRL behavior rather than the Posttrain adapter, stop production edits. Amend this plan to name the TRL checkout,
failing test, fork files, commit order, wheel publication, immutable source revision, hash updates, and validation for
both repositories. Only then implement the fork fix.

Milestone acceptance is a focused test that fails against the current Posttrain translation for the expected reason
but proves the underlying pinned TRL vision path works. The plan's `Surprises & Discoveries` and `Decision Log` must
record the exact result.

### Milestone 2: Add ordered media to canonical supervised data

In `packages/data/src/posttrain/data/models.py`, add a frozen, slotted, backend-neutral media reference and an ordered
tuple of those references on `SupervisedExample`. The reference must at minimum identify media kind `image`, a safe
materialization-relative POSIX path, an expected SHA-256 digest, and a MIME type. It must reject absolute paths,
parent traversal, empty path segments, malformed digests, unsupported media kinds, and unsupported MIME types. Tuple
order is page order, so do not sort it during validation.

Update `packages/data/src/posttrain/data/adapters/huggingface.py` and the canonical dataset validation/materialization
path to preserve media during JSONL round trips. Increment the supervised schema representation only if existing
schema-version rules require it; retain backward compatibility for text-only records with no media. Do not put PIL
images, binary blobs, or framework-specific message objects in public contracts.

Add tests beside `packages/data` for text compatibility, media round trips, page ordering, duplicate or conflicting
paths, malformed digests, path escape attempts, and stable deterministic serialization. Acceptance is that loading,
writing, and reloading a visual example produces the same ordered references and that every existing text dataset test
continues to pass unchanged in behavior.

### Milestone 3: Package and verify immutable dataset assets

Extend dataset materialization so a visual dataset owns an `assets/` directory beside `data.jsonl` and
`manifest.json`. The materialization manifest must enumerate every asset in deterministic POSIX-path order with its
digest and size, while supervised examples retain their semantic page order independently. Compute one deterministic
asset-bundle digest from the ordered file records. The materialization must reject symlinks, special files, path
escapes, missing files, duplicate paths, and digest mismatches before publishing state.

Extend `DatasetMaterialization` in `packages/data/src/posttrain/data/catalog.py` with the smallest optional asset-bundle
metadata needed by execution packaging. Extend `DatasetPackageLock` in
`packages/execution/src/posttrain/execution/job_package.py` with optional asset-root identity and bundle digest while
preserving payload compatibility for old text-only locks. Update
`packages/execution-pack/src/posttrain/execution_pack/datasets.py::DatasetPackager.package` to copy asset files into
the build context without following symlinks and to verify every copied byte. Update
`apps/runtime/src/posttrain_runtime/execute.py::_verify_datasets` to derive its expected file set from the locked data,
manifest, and enumerated assets, then verify the complete bundle before exposing the materialized dataset to a job.

Do not relax the existing exact-file-set check. The stronger contract must still reject rogue files. Do not copy raw
project input directories wholesale and do not rely on the generic project source snapshot, which intentionally
excludes dataset inputs and has independent size limits.

Add focused tests in `packages/data/tests`, `packages/execution-pack/tests`, `packages/execution/tests`, and
`apps/runtime/tests`. Acceptance is a packaged visual fixture whose assets are available at portable paths inside the
actual-job root, plus negative tests showing that mutation, deletion, extra files, and symlinks fail before operation
execution. Existing two-file text packages must remain byte-for-byte compatible unless a recorded contract reason
requires otherwise.

### Milestone 4: Restore the visual TRL SFT translation

Refactor `packages/train/src/posttrain/train/backends/trl/sft.py` into explicit text and visual dataset preparation
helpers. Preserve the current renderer-pretokenized text helper and its observation metrics. The visual helper must
resolve packaged media paths, construct a Hugging Face dataset with ordered `images`, conversational `prompt`, and
conversational `completion`, and leave image preprocessing to TRL's vision-language collator.

In `packages/train/src/posttrain/train/backends/trl/common.py`, lazily import `AutoProcessor` and add a processor loader
that uses the exact model repository and immutable revision with `trust_remote_code=False`. Pass the processor as
`processing_class` for visual datasets. Configure completion-only loss, disable packing and padding-free modes, and do
not set `skip_prepare_dataset=True` where doing so would bypass the image collator. Ensure the finalized canonical JSON
assistant completion is serialized deterministically before training.

Add an SFT-owned no-truncation control in `SFTSettings` and `SFTSettingsSchema`, or an equally narrow typed setting
proved by the dependency spike. Translate the visual default to TRL `max_length=None`. Preserve
`TrainingLoop.max_length` as a required positive integer for all other techniques and for text SFT. Fail before model
loading when visual examples are paired with a model that does not declare image capability, when a visual dataset
mixes incompatible row formats, or when packaged media cannot be resolved.

Add visual dataset observation metrics that are meaningful before optimization: example count, image count, image
bytes, page-count percentiles, prompt and completion token counts where safely measurable, and whether truncation is
disabled. Do not label image bytes as text tokens. Continue emitting existing model, runtime, optimization,
checkpoint, and artifact evidence.

Add tests in `packages/train/tests` using fake imports and a recording trainer to prove the exact processor, dataset
shape, completion-only loss, image ordering, capability validation, no-truncation translation, callbacks, and
validation-dataset behavior. Existing text SFT tests must prove that token IDs, labels, renderer masks, and arguments
remain unchanged.

### Milestone 5: Add Gemma 4 E4B visual qualification

Create a tiny, license-safe fixture containing at least one example with two ordered images, a short extraction
instruction, and a deterministic JSON assistant completion. Register it as a Lab supervised dataset selection using
the normal materialization path. Add Gemma 4 E4B visual SFT settings and a training binding to
`apps/lab/.posttrain/catalog/gemma4-qualification.yaml`, or a focused adjacent overlay listed by the Lab layer. Add a
new work package under `apps/lab/.posttrain/work_packages/` that resolves the standard `train/trl-sft@1` job with the
E4B BF16 model, visual dataset, two-step settings, LoRA binding, and RTX Pro 6000 target.

Before finalizing LoRA targets, inspect the exact E4B module names without allocating full weights where possible.
The selected expression must include intended language-model attention and MLP projections, select at least one
module, and exclude vision and audio towers. Keep vision layers frozen for the first smoke. Use rank 8 or another
small rank consistent with existing qualification settings, per-device batch size 1, conservative gradient
accumulation, gradient checkpointing, BF16, logging each step, and one retained final checkpoint.

Extend Lab catalog, work-package, and qualification inventory tests. Static acceptance is that catalog resolution,
work-package validation, and job planning succeed without a GPU. Actual-job packaging and supervised-runtime
identification are the first gates of Milestone 6 because they require the configured builder and registry. No
PolicyPrism data or prompt belongs in this Lab fixture.

### Milestone 6: Validate, package, publish, and run the GPU smoke

Run focused tests after each package edit, then the complete validation ladder from the repository root. Build the
supervised job-kind or actual-job image through the repository's existing release and job-pack paths rather than a
custom Dockerfile. Publish an immutable actual-job digest to the approved registry and submit the bounded E4B visual
job through Posttrain's dstack provider with an explicit timeout.

GPU acceptance requires two completed optimizer steps, finite loss and gradient norm, positive trainable parameters
smaller than total parameters, evidence that the batch contains non-empty vision tensors, one retained adapter,
successful materialization of that adapter, and a clean-process reload that produces finite logits or a non-empty
generation for the fixture. Record the Posttrain run id, dstack run id, image digest, model revision, dataset digest,
adapter digest, key metrics, and Observatory or Trackio evidence links in `Artifacts and Notes` without recording
credentials.

If the GPU is unavailable, preserve all static evidence and leave the real integration checkbox incomplete. Do not
claim multimodal SFT support from unit tests or image publication alone.

### Milestone 7: Package PolicyPrism training and domain evaluation

After generic qualification passes, update the separate PolicyPrism project rather than adding project data here.
Materialize each complete PDF as ordered rendered page images followed by the finalized parsing instruction; use its
canonical PolicyPrism JSON as the deterministic assistant completion. Keep the sealed 100-document set and all of its
aliases or near-duplicate revisions out of training, validation, prompt tuning, checkpoint selection, and
hyperparameter selection.

Create PolicyPrism Gemma E4B training and validation dataset selections and a project work package using the standard
`train/trl-sft@1` definition. First run a bounded project smoke, then the agreed full run. Retain the exact prompt,
rendering parameters, source-document split manifest, model revision, settings, dataset digests, and output adapter
lineage.

Package the existing PolicyPrism Verifiers environment and taskset through an `EvaluationPlan` and the standard
`eval/verifiers-managed@1` definition. Add an image-capable vLLM inference binding rather than reusing current
`text_only: true` Gemma bindings. The taskset must send ordered page images before the text instruction and compute the
same canonical JSON metrics for both model versions. Run the sealed 100-document set once on the immutable foundation
checkpoint and once on the selected trained checkpoint only after training decisions are frozen. Store native
Verifiers traces and compute per-metric and aggregate post-SFT minus baseline deltas.

Acceptance is a pair of comparable managed evaluation runs with identical sealed taskset revision, prompt, page
rendering, generation settings, environment revision, and metric code, differing only in the selected model artifact.

## Concrete Steps

All commands in this section run from `/Users/ct/Desktop/policy_prism/posttrain` unless a milestone explicitly names a
sibling repository. Update the commands and expected transcripts as implementation reveals exact test names.

Confirm the user-created branch before editing or committing:

    cd /Users/ct/Desktop/policy_prism/posttrain
    git branch --show-current
    git status --short

Expected branch output:

    feature/policyprism-gemma-vlm-sft

Run focused tests during implementation:

    uv run pytest packages/data/tests -q
    uv run pytest packages/execution/tests packages/execution-pack/tests apps/runtime/tests -q
    uv run pytest packages/train/tests -q
    uv run pytest apps/lab/tests/test_catalog.py apps/lab/tests/test_work_packages.py -q

Run static work-package checks using the exact new visual qualification file names recorded during Milestone 5:

    cd apps/lab
    uv run posttrain work-package validate .posttrain/work_packages/gemma4_e4b_visual_sft_qualification.yaml
    uv run posttrain job plan .posttrain/work_packages/gemma4_e4b_visual_sft_qualification.yaml --job train

Run the full repository validation ladder:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

The user owns Git execution. At each implementation milestone, inspect only the intended paths, stage those exact
paths with `git add -- <paths>`, and create one professional logical commit. Never use `git add .`, `git add -A`, or
`git add --all`. Commit, push, and pull-request creation remain separately authorized operations.

The exact job-pack, registry, and dstack submission commands will be added after Milestone 5 selects the work package
and registry namespace. They must use Posttrain's normal `job pack` and `job run` commands, an explicit provider
timeout, and the immutable image digest produced by the branch.

## Validation and Acceptance

Contract acceptance requires a text-only `SupervisedExample` to retain its current behavior and a visual
`SupervisedExample` to round-trip through canonical JSONL with ordered immutable media references. Unsafe paths,
invalid digests, unsupported media, and asset mutations must fail with contract errors before training.

Packaging acceptance requires the actual-job manifest to lock all dataset data, manifests, and image assets. A clean
runtime verification must expose every expected image and reject a missing, altered, extra, symlinked, or escaped
file. The visual fixture must remain portable when the build context is moved to a different absolute directory.

Trainer acceptance requires the visual path to pass ordered images and conversational prompt-completion records to
TRL with a processor and image-aware collator. The resulting batch must contain a non-empty vision tensor, prompt
tokens must be ignored for loss, completion tokens must be trainable, and visual truncation must be disabled unless a
later explicit qualification proves a safe bound. Existing text SFT golden tests must remain unchanged.

Static Gemma acceptance requires the E4B model, settings, dataset, LoRA binding, target, standard SFT definition, and
supervised runtime image to resolve and package. The LoRA target test must prove language projections are selected and
multimodal towers are excluded.

Real integration acceptance requires a remote E4B visual SFT job to finish two optimizer steps with finite metrics,
retain a digest-addressed adapter, and reload that adapter successfully. The exact processor, model, TRL, dataset,
actual-job image, and adapter revisions must be recoverable from evidence.

Project acceptance requires a PolicyPrism training work package to consume ordered complete-document page images and
canonical JSON completions without chunking, and requires managed Verifiers baseline and post-SFT runs over the same
sealed 100-document taskset. Reported deltas are valid only when all non-model evaluation inputs are identical.

## Idempotence and Recovery

Unit and static validation commands are safe to rerun. Dataset materialization and actual-job packaging must remain
content-addressed: rerunning with identical bytes and selections must produce identical digests rather than duplicate
mutable identities. A failed temporary build may be retried after removing only the explicitly reported temporary
build output through the repository's supported cleanup path; do not delete tracked state or broad workspace roots.

Do not overwrite an existing immutable registry tag. If a package record conflicts after content changes, use the
normal Posttrain materialization invalidation or a new content-derived package identity rather than manually editing
the record. Preserve failed GPU run identifiers and logs as diagnostic evidence.

Keep the existing text SFT path available throughout the additive implementation. If the visual path fails, revert or
repair only the uncommitted visual milestone rather than weakening text behavior. The user performs all Git recovery
commands and must inspect the exact paths first.

If the pinned TRL probe fails, stop before changing Posttrain dependency pins. A fork change must be committed,
published, and verified independently before Posttrain consumes its immutable revision. Credentials, Hugging Face
tokens, Trackio tokens, registry credentials, and dstack environment values must remain in approved external
credential files and never appear in source, tests, fixtures, plans, logs committed to Git, or command history shown
in this document.

## Artifacts and Notes

Manager requirement being implemented:

    Restore the multimodal SFT path on a separate Posttrain branch, use the Gemma 4 E2B/E4B multimodal fine-tuning
    guidance, run Gemma smoke training through TRL, understand and prove actual-job packaging and publication, and
    package a domain-specific evaluation job using the existing Verifiers tasksets.

The intended PolicyPrism logical example is:

    ordered PDF page images + finalized parsing instruction -> one deterministic canonical JSON completion

There is no page-level generation or merge in this baseline. Multiple ordered page images are one model example.

External technical reference reviewed during planning:

    https://unsloth.ai/docs/models/gemma-4/train#multimodal-fine-tuning-e2b-e4b

The relevant guidance is embedded in this plan: use E2B or E4B for multimodal fine-tuning, place images before text,
start with vision layers frozen, fine-tune language attention and MLP layers first, and enable vision-layer training
later only if evidence shows the task needs it.

Add concise evidence here as milestones complete. Do not paste secrets or unbounded logs.

## Interfaces and Dependencies

The final public data contract must expose a frozen, slotted media reference from `posttrain.data` and an ordered media
field on `SupervisedExample`. The exact final name must be chosen once in Milestone 2 and recorded throughout this
plan. Its logical shape is:

    @dataclass(frozen=True, slots=True)
    class SupervisedMedia:
        kind: Literal["image"]
        path: str
        sha256: str
        mime_type: str

    @dataclass(frozen=True, slots=True)
    class SupervisedExample:
        ...
        media: tuple[SupervisedMedia, ...] = ()

The execution contract must extend `DatasetMaterialization` and `DatasetPackageLock` with optional asset-bundle
identity while retaining existing text-only payloads. The dataset manifest is the canonical enumeration of relative
asset files, sizes, and SHA-256 digests. Runtime verification must validate both the bundle lock and every enumerated
file.

The private train contract must add a processor loader beside `load_tokenizer` and split SFT preparation into text and
visual helpers. The visual helper consumes only verified packaged media paths and produces the dataset shape expected
by the exact pinned TRL version:

    {
        "images": [<ordered image values>],
        "prompt": [{"role": "user", "content": <finalized instruction>}],
        "completion": [{"role": "assistant", "content": <canonical JSON string>}],
    }

The final representation may use typed multimodal content blocks if the exact pinned processor requires them, but it
must preserve the same semantics: all page images precede the instruction, page order is stable, and only the
assistant completion receives loss.

The implementation uses existing compatible ranges and exact resolutions from `packages/train/pyproject.toml` and
`uv.lock`: Python 3.13, Torch, Torchvision, Transformers, TRL, PEFT, Datasets, Accelerate, BitsAndBytes, Renderers,
Hugging Face Hub, and the repository's standard runtime image. Do not add Unsloth as a runtime dependency. Its Gemma
guidance is a reference for data ordering and conservative layer selection; Posttrain continues to execute through
its pinned TRL stack.

The domain evaluation uses existing `posttrain.eval` Verifiers support and the standard managed evaluation job. It
must not introduce a second metric implementation. Native Verifiers traces remain the replay authority and the
PolicyPrism project owns its taskset, canonical JSON metrics, sealed-set manifest, and prompt.

Revision note (2026-08-20): Created the initial implementation-ready plan after tracing the current Posttrain SFT,
dataset, package, runtime, Gemma qualification, and evaluation paths and inspecting the exact pinned TRL fork. The
plan records the manager's separate-branch, Gemma multimodal smoke, packaging, publication, and Verifiers evaluation
requirements; gates any TRL fork edit on a focused exact-version failure; and keeps PolicyPrism policy out of reusable
framework packages.

Revision note (2026-08-20): Added the pinned-TRL multimodal compatibility proof and completed the backend-neutral
`SupervisedMedia` contract with ordered adapter round trips and backward-compatible text exports. Recorded the macOS
arm64 versus locked CUDA dependency limitation explicitly, leaving real TRL test execution as a Linux supervised
runtime acceptance gate rather than weakening or silently substituting the pinned dependency.

Revision note (2026-08-20): Completed deterministic visual asset materialization, package locks, actual-job copying,
runtime exact-set verification, and the modality-selected TRL visual translation. Added focused positive and
fail-closed tests, processor retention, completion-only loss, explicit no-truncation behavior, and visual input
profile metrics. Recorded the three unrelated Linux-path assumptions that prevent a fully green suite on macOS.

Revision note (2026-08-20): The second remote visual qualification reached the assigned RTX GPU, installed internal
trust, and initialized Trackio, then failed while importing `Gemma4Processor` because the supervised runtime image
omitted Torchvision. Added the workspace-pinned Torchvision package to the supervised runtime profile and strengthened
static validation of the authored profile and generated narrow lock. This makes the processor's visual runtime
dependency part of the immutable supervised image contract rather than a project-level workaround. The real Gemma
processor import remains an acceptance condition of the remote two-step visual qualification.

Revision note (2026-08-20): Published the corrected supervised candidate image at
`registry.lan/carbonteq/posttrain-kind-supervised@sha256:e4e6e3ad6cd7234927c864eff3cae731975166fc52d4b90162095787b5a97d1c`
and completed remote visual SFT qualification run `18a279b9-9d82-4f4e-9b13-539ee8e29bd1` on the assigned CarbonTeq RTX
worker. Gemma 4 E4B completed two optimizer steps with finite losses `0.9947` and `0.6037`, finite gradient norms
`5.856` and `3.174`, and final mean token accuracy `0.8462`. The run finished with status `succeeded` and published
recovery, checkpoint-model, model, and summary artifact roles. This proves remote visual processor loading, visual TRL
training, finite optimization, and adapter retention through the actual-job path. Adapter consumption/reload remains
the next composed-job gate before full PolicyPrism training.

Revision note (2026-08-20): Reconciled the successful run and pinned its retained Trackio adapter as catalog model
`models/gemma4-e4b-it@bf16/sft-visual-qualification-v0`. Added a one-step visual continuation work package that
materializes the immutable adapter through Posttrain's standard `model_adapter` input and reloads it on the pinned
multimodal Gemma base. Expanded validation reports 84 passes and one expected local CUDA skip. Release checks, local
packaging, and OCI qualification succeeded with package
`1f15ce6ec3be6205e6a5ddf1bfa312332d7fa45bfa6466dca013d84ddae84343` and actual-job image digest
`7469f7439639dca6ea2d0428f1cb52e8a28bc97479caa5dd118a70fbff607d34`.
