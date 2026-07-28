# Package every framework job as an immutable OCI capsule

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with
`docs/templates/PLAN.md`.

## Purpose / Big Picture

After this work, a developer can statically plan a selected work-package job,
pack its exact code, environments, configuration, and dataset snapshot into an
OCI image, and run that image locally or through dstack. The framework owns the
complete image hierarchy and packaging lifecycle. dstack receives only a
digest-pinned image, execution resources, non-secret environment-variable
names, and typed volumes; it no longer receives a second uploaded code/data
bundle.

The image hierarchy has three levels. One universal framework base supplies
Python, CUDA, PyTorch, trusted certificates, and common system dependencies.
Job-kind images add stable dependencies for supervised training, online RL,
evaluation, serving, or model transformation. An actual packed-job image adds
the selected project code, framework worker, exact environment repositories,
resolved configuration, and materialized datasets.

The same packed-job image may be executed more than once. Its digest identifies
job meaning and immutable inputs; canonical run ID, attempt, provider, worker,
tracking credentials, timeout, and mounts belong to a separate launch envelope
and do not force an image rebuild.

## Current Status and Release Handoff

This is the authoritative living plan for the OCI job-capsule work. It has been
updated while implementation and qualification proceeded; it is not merely the
original proposal. As of 2026-07-27 07:03Z, its append-only evidence stream is
`docs/plan/framework-oci-job-capsules-execution-log.jsonl`, with 75 ordered,
unique events before this handoff update. More focused plans retain the detail
for the dstack lifecycle, job-kind qualification, Trackio/Doris storage, and
model artifact lifecycle:
`docs/plan/dstack-execution-provider.md`,
`docs/plan/job-kind-capsule-qualification.md`,
`docs/plan/trackio-apache-doris-engine.md`, and
`docs/plan/trackio-model-artifact-lifecycle.md`. The infrastructure repository
has its own deployment plan at
`/home/hammad/projects/ai-infra/docs/plan/reproducible-unraid-ai-infrastructure.md`.
Those documents support this plan; they do not replace its framework boundary
or release order. A point-in-time task handoff for the next agent is retained
at `docs/plan/framework-oci-job-capsules-agent-handoff.md`; any progress made
from that handoff must be written back into this living plan and its execution
log.

The framework implementation now has the intended three-level image model:
one universal framework base, exact job-kind runtime variants, and one
immutable actual-job image containing the selected code, configuration,
environment repositories, and materialized dataset snapshot. A launch envelope
adds run-specific identity, provider policy, non-secret environment names, and
typed mounts without rebuilding the image. `posttrain job plan`, `job pack`,
and `job run` drive that lifecycle. Local Docker and dstack providers consume
the same digest-pinned job image, and the stable worker entrypoint is
`posttrain-runtime execute`. dstack's uploaded-file transport is no longer the
normal distribution mechanism.

Environment inclusion is reproducible rather than copied ad hoc. Each selected
environment records a canonical HTTPS Git URL, full commit, package
subdirectory, source-tree digest, built-wheel digest, and serializable
activation. Multiple environments from one repository and commit share one
checkout while retaining distinct package identities. Dataset materialization
belongs to packing, so planning and catalog composition remain detached from
network and dataset side effects. Base-model weights and mutable checkpoints
remain on worker caches or volumes; they are not duplicated into every job
image.

The execution lifecycle is implemented across planning, packing, submission,
status, logs, cancellation, reconciliation, and cleanup. Admission and
idempotency state are durable. Each attempt retains a provider-neutral evidence
locator and writes an atomic terminal marker only after tracking and artifact
finalization unwind. Cleanup rejects incomplete or mismatched markers and
retains results while removing eligible scratch state. Training code emits its
observations directly to Trackio; jobs receive the tracking endpoint and secret
names from execution-target configuration and do not know that Doris is the
Trackio storage implementation.

The framework source gate most recently passed 665 tests with 16 intentional
skips, Ruff, Pyright, all eight import contracts, the actual-job static
validator, execution-log ordering, and `git diff --check`. Live capsule
qualification has completed for local SFT; dstack-hosted SFT, DPO, SAMPO, and
fifteen-update GRPO; managed general and multi-environment evaluation; dataset
preparation; serving smoke and a bounded capacity sweep; and model
transformation. Those runs used realistic GPU execution and retained evidence
through remote Trackio backed by Doris. The corrected Trackio post3 source and
the newest Observatory projections have not yet been deployed, so source
validation must not be described as production promotion.

One algorithm/provider gate remains deliberately open. Distillation is now
closed: run `339100a5-a4c2-4ae6-aa5a-1b080513b50e` completed ten real optimizer
updates on the RTX PRO workstation with finite loss and gradient norms on every
step, reconciled consistently without recovery, and resolved through deployed
Observatory. The blocker was not numerics. The base `Trainer` counts
`num_items_in_batch` from the raw pre-generation dataloader batch, whose labels
are prompt-only for on-policy rows, so a fully on-policy window counted zero and
the divergence loss divided by it; `logging_nan_inf_filter` then hid the
non-finite loss and exposed only `grad_norm=nan`, which misdirected four prior
attempts. See execution-log sequence 91.
Graceful cancellation before submission is qualified; graceful cancellation
of a running job is not. The bounded dstack `stop_duration` propagation patch
passed its Python and Go source gates, but the server, runner, and shim must be
published and deployed atomically before a run can prove that a finalizer
lasting more than ten seconds completes within the configured grace period.

The backend release candidates are bounded and validated but unpublished:

* Trackio contains the first-class Apache Doris storage provider, migration and
  schema negotiation, SQLite/Turso parity, model-artifact work, and the
  upgraded dashboard dependency/security surface. Its focused post3 suite
  passed 93 tests; the broader non-hardware suite and wheel checks also passed.
* TRL contains the distillation LoRA/vLLM synchronization mode and regression
  coverage. Its final combined release gate passed without retained
  distributed or CUDA state.
* dstack contains only the graceful-stop-duration propagation and regression
  tests intended for the public fork.
* veRL was reconstructed from published CarbonTeq `origin/main` in
  `/home/hammad/projects/verl-release-candidate`, excluding the failed
  TurboQuant/runtime-environment research surface. It passed 42 focused CPU
  tests, the 34-test SAMPO/core regression, and five focused RTX 4090 source
  tests. Its Python 3.13 dependency lock, kind image, cross-interpreter capsule
  smoke, and live workload qualification remain open.

Publication has moved past repository creation. The four maintained fork
candidates are now committed and pushed on their CarbonTeq remotes:
`trackio:codex/doris-storage-post3` at
`9b0c4af0414cbd5264fa1012d81c494e46c1150d`,
`trl:codex/distillation-lora-sync` at
`6828a84716e0b9e29c3aedb40df3d28b81770e5b`,
`dstack:codex/graceful-cancellation-stop-duration` at
`371ff53b1d67f254bc6cc4259aae8653c3916b7d`, and
`verl:codex/runtime-release-qwen35` at
`1dcdf67e9473db5297c98c9c88cf4dae6c4a8932`. The framework worktree has been
re-pinned to those immutable commits and the full validation ladder now passes
against the updated lockfile. `ai-infra` still has no initial commit, and no
deployment has been performed during this publication attempt.

Resume in this order. First, re-run the bounded secret and diff audits and
publish Trackio, TRL, dstack, and veRL as reviewed fork commits without adding
unrelated dirty changes. Second, update the exact immutable fork references and
locks in this framework, split the framework release into the scoped series
already recorded in the publication audit, and re-run the complete framework
gate. Third, make and push the reviewed initial private ai-infra commit.
Fourth, build and publish the immutable runtime images and deploy the corrected
services and dstack components. Finally, run the open ten-backward-pass
distillation, veRL cross-interpreter/GPU, and running-cancellation
qualifications, reconcile retained Trackio/Observatory evidence, and perform
policy-driven cleanup. The user asked to stop before deployment in the current
release task, so only the first three steps are authorized until that stopping
point is reached.

## Progress

- [x] (2026-07-27 07:03Z) Reconcile the living plan, append-only evidence log,
  local repositories, and GitHub remotes into the current-status handoff above.
  The plan was current through source qualification but previously lacked one
  concise resume point.
- [x] (2026-07-27 07:03Z) Create the public
  `carbonteq-ai/dstack` fork and private `carbonteq-ai/ai-infra` repository.
  No candidate branch, framework series, ai-infra initial commit, or deployment
  was produced before publication was paused.
- [x] (2026-07-27 07:17Z) Publish the Trackio Doris post3 candidate on
  `carbonteq-ai/trackio:codex/doris-storage-post3` at
  `9b0c4af0414cbd5264fa1012d81c494e46c1150d`, the TRL distillation LoRA sync
  candidate on `carbonteq-ai/trl:codex/distillation-lora-sync` at
  `6828a84716e0b9e29c3aedb40df3d28b81770e5b`, the dstack graceful stop-duration
  candidate on
  `carbonteq-ai/dstack:codex/graceful-cancellation-stop-duration` at
  `371ff53b1d67f254bc6cc4259aae8653c3916b7d`, and the clean veRL runtime delta
  on `carbonteq-ai/verl:codex/runtime-release-qwen35` at
  `1dcdf67e9473db5297c98c9c88cf4dae6c4a8932`. Each remote head was verified with
  `git ls-remote` after push.
- [x] (2026-07-27 07:18Z) Re-pin the framework to the published Trackio and TRL
  commits, publish the veRL runtime candidate revision into the blocked profile,
  refresh the lockfile, and rerun the complete repository ladder. The first
  reruns exposed missing `pymysql`, a stale Trackio/TRL scaffold expectation,
  and stale training lock digests; after correcting those pin-driven
  expectations, locked sync, Ruff, Pyright, all eight import contracts, 664
  passing tests with 16 skips, and `git diff --check` all passed.
- [x] (2026-07-27 07:24Z) Push the scoped framework series on
  `carbonteq-ai/posttrain:codex/serving-capacity-observatory` and publish the
  private `carbonteq-ai/ai-infra` initial commit on `main` at
  `46c6f76e5823c7889b834aaadf133e7d1b86a8b5`. The framework branch now carries
  the ordered release series and `ai-infra` remains source-only; no images were
  built and no deployment or live qualification command was run.
- [x] (2026-07-26 18:20Z) Selected the three-level framework-owned image
  hierarchy and rejected dstack file upload as the normal job-distribution
  mechanism.
- [x] (2026-07-26 18:20Z) Defined the multi-environment Git identity as
  canonical repository URL, full commit SHA, package subdirectory, tree digest,
  built-wheel digest, and serializable activation digest.
- [x] (2026-07-26 19:05Z) Add generic immutable Git source planning, content-addressed checkout,
  multi-subdirectory deduplication, and a deterministic environment source
  lock.
- [x] (2026-07-26 19:30Z) Exercise the source packer against the real pinned
  Prime Intellect Verifiers commit: GSM8K and Reverse Text produced one
  checkout, two subdirectory locks, and deterministic lock digest
  `646310aa7d8b4d90799e2afdd8bdea3bbddcdbab74645b277303a2ed0e3c117d`;
  the temporary checkout was moved to Trash after inspection.
- [x] (2026-07-26 19:50Z) Build real GSM8K and Reverse Text wheels from that
  one checkout twice. The ordered wheel lock was stable across reversed input
  order with digest
  `e8a024d0c7b26fc1eef3128b68a9c6ad31930771a3ec7c98d0b085a741e5531a`;
  the temporary source and wheel cache were moved to Trash.
- [x] (2026-07-26 19:15Z) Amend the frozen primitive, framework, and API
  baseline so environment bindings retain serializable source/activation identity
  and the framework owns all three image levels.
- [x] (2026-07-26 20:00Z) Replace eager environment callables with
  serializable declarative Verifiers or Python-factory activations. Catalog and
  work-package validation remain detached; eval/train activate only in the
  execution runtime. Canonical Git sources are now secret-free HTTPS plus full
  commit and normalized subdirectory.
- [x] (2026-07-26 20:05Z) Revalidate the live composed catalog (52 entries)
  and the 15-step GSM8K GRPO work package after activation migration; static
  composition completed with all five required seats and no environment
  activation on the developer host.
- [x] (2026-07-27 03:20Z) Move universal and job-kind Dockerfiles, BuildKit
  definitions, smoke checks, and immutable image publication into the
  framework repository. The universal base and supervised kind image have
  real registry digests; the latter passed Torch/Transformers and hash-locked
  source-build-backend qualification.
- [x] (2026-07-27 03:05Z) Split the run-bearing contract into image-owned
  `JobPackageManifest` and launch-time `ExecutionLaunchEnvelope`, migrate the
  stable runtime entrypoint, and make local/dstack providers receive the final
  image digest without a copied code bundle.
- [x] (2026-07-27 03:22Z) Add `JobPackService` plus first-class
  `posttrain job plan`, `job pack`, and `job run`. The first real SFT package
  materialized its dataset and exact framework/project sources, passed the
  non-publishing image smoke target, emitted provenance/SBOM, wrote a
  mode-`0600` receipt, and published
  `registry.lan/carbonteq/posttrain-job@sha256:4d8cc2ac81d8446c1b5f7436597d12e110cdfa1805a3fd3bffb2b68ca862dc5f`.
- [x] (2026-07-26 21:10Z) Add the framework-owned third image definition under
  `containers/posttrain-job`: digest-pinned kind parent, dependency/code/data
  cache ordering, stable worker entrypoint, zstd OCI media types, provenance,
  SBOM, staged-context validation, and no launch identity or model weights.
- [x] (2026-07-26 21:25Z) Separate selection-time seats from activated runtime
  seats for standard SFT/DPO definitions. `job plan` and work-package
  validation now retain `DatasetLoadPlan` values without fetching or writing;
  packing owns dataset materialization.
- [x] (2026-07-26 21:40Z) Invert the package boundary so
  `execution-pack` owns Git/wheel request contracts and the BuildKit package
  implements them. Rename the provisional package identity to `plan_key`;
  final `package_key` remains the digest of the materialized
  `JobPackageManifest`.
- [x] (2026-07-26 22:10Z) Add explicit project source snapshot configuration
  through `[tool.posttrain.pack]`, content-addressed working-tree snapshots,
  bounded copy limits, and rejection of Git metadata, secrets, model weights,
  overlapping paths, and undeclared repository contents. Normal projects
  default to their installable root; monorepos declare package roots.
- [x] (2026-07-26 22:05Z) Run the first cold universal-image build and use its
  failure to close an index-confusion gap: the CUDA index's incompatible
  `triton==3.6.0` artifact can no longer replace the exact x86_64 PyPI wheel
  selected by `uv.lock`. The corrected Python 3.12/CUDA 13 image passed its
  Torch smoke and was published as
  `registry.lan/carbonteq/posttrain-base@sha256:fa1a63063b7cb63815fee0c72876a0afb9f92cb27d3c1d53b5aaaf0710d1e835`.
- [x] (2026-07-27 01:35Z) Refine kind-image selection from one image per
  logical job kind to exact runtime variants. The publishable TRL target is
  `online-rl-trl-py312`; the generic `online-rl` target is gone. A fail-closed
  `online-rl-verl-py313` definition records the Python 3.12 control/Python 3.13
  worker split and exact qualified dependency versions, but is deliberately
  absent from the Bake publication graph because the CarbonTeq veRL changes
  are still dirty and unpublished.
- [x] (2026-07-27 04:45Z) Define the dormant actual-job side of the Python
  3.12 control/Python 3.13 veRL split, including the intended worker
  projection and blocked release metadata. This was definition-level evidence:
  it did not install either closure, build an image, or prove a two-interpreter
  runtime. Sequence 51 in the append-only execution log supersedes the
  over-broad installation wording recorded at sequence 50.
- [x] (2026-07-27 05:17Z) Correct the veRL capsule boundary. Selected
  environment wheels now resolve into separate hash-locked Python 3.12 control
  and Python 3.13.12 backend closures, each with its interpreter, requirements
  digest, and resolution digest in `JobPackageManifest` and the package key.
  Packing rejects host interpreter/worktree paths, dirty or unpinned fork
  facts, and content-digests the exact `common`, `data`, and `train`
  projection. Build and execution verification check the capsule interpreter,
  base dependency lock, clean fork revision, projection content, and module
  origins. The launcher discards inherited Python path/user-site state. The
  dormant dedicated Docker/Bake smoke graph is structurally valid, while its
  real build remains a mandatory ready-state gate. Final focused validation
  passed 111 tests plus Ruff, Pyright, all eight import contracts, Bake graph
  parsing, append-only log validation, and diff check.
- [x] (2026-07-27 05:18Z) Close the scheduled worker-retention handoff. The
  stable runtime now writes one mode-`0600`, atomic, directory-fsynced
  `.posttrain-terminal.json` only after successful, failed, or cancelled
  execution has unwound through tracking and artifact finalization. The
  provider-neutral marker binds run ID, attempt, provider, immutable job-image
  digest, status, and timezone-aware completion time. The ai-infra collector
  rejects mismatched, symlinked, legacy, incomplete, or non-terminal markers
  before age-based deletion. Runtime tests passed 15 cases and the
  cross-repository collector contract test passed.
- [x] (2026-07-27 05:21Z) Re-run the complete framework gate after the
  interpreter-specific veRL contract and terminal-marker handoff. Locked
  Python 3.12 sync, Ruff, Pyright, all eight import contracts, 664 passing
  tests with 16 intentional skips, execution-log ordering, and repository diff
  checks pass.
- [x] (2026-07-27 05:23Z) Remove compatibility noise from the normal job DX
  without breaking migration callers. `posttrain job plan`, `pack`, and `run`
  now advertise the project entry and intentional packaging overrides, while
  hiding the deprecated `--host` alias and temporary `--in-process` path.
  Existing compatibility behavior remains callable during the bounded
  migration window. The CLI suite passed 30 tests, with Ruff and Pyright clean.
- [x] (2026-07-27 05:32Z) Re-run the complete source gate after the CLI and
  terminal-marker hardening: 665 tests passed with 16 intentional skips,
  Ruff and Pyright are clean, all eight import contracts remain intact, and
  the repository diff check passes.
- [x] (2026-07-27 06:17Z) Audit the dormant veRL dual-environment release path
  without publishing it. The Python 3.12 control/Python 3.13.12 backend
  contract passed 89 focused framework tests, and the current Python 3.13
  research environment passed 43 focused veRL CPU tests. The release gate
  correctly rejects the missing dependency-only lock, empty fork revision and
  lock digest, blocked profile, omitted publication target, dirty detached
  checkout, and absent remote verification. The local checkout is based on
  older upstream `a35908ca3c9632859c58d6a2855d858918ae21dc`, while published
  CarbonTeq `origin/main` is
  `553280b88afe4e7fbc4aefeff27bbf0a22e7c048`; the release must be
  reconstructed from the latter and must exclude the failed TurboQuant
  research surface. The actual-job static validator was also corrected to
  recognize the dual-lock Docker boundary.
- [x] (2026-07-27 06:18Z) Audit the publication boundary across Trackio, TRL,
  dstack, veRL, this framework, and ai-infra. Trackio post3, the TRL
  distillation synchronization surface, and dstack graceful cancellation each
  form one bounded fork commit after their remaining gates. veRL requires a
  selectively reconstructed series from published CarbonTeq `origin/main`.
  Framework publication must be split into execution, packaging/images,
  providers/DX, capabilities/jobs, Observatory/tracking, and qualification
  history rather than committing the large dirty tree wholesale. Ambient
  artifacts, stale deployment documents, the legacy upstream-veRL tool, and
  the large editor artifact are excluded from this release. The safe order is
  fork publication, exact framework pins and controlled commits, first
  ai-infra source publication, immutable builds/deployment, and finally live
  GPU/cancellation/evidence qualification.
- [x] (2026-07-27 06:28Z) Reconstruct the supported veRL runtime delta in a
  separate detached worktree at published CarbonTeq
  `553280b88afe4e7fbc4aefeff27bbf0a22e7c048`. The reconstruction preserved
  newer SAMPO/replay-buffer and QLoRA/FSDP behavior while layering only
  dependency compatibility, LoRA pre-wake staging, dense-FSDP entropy
  chunking, Qwen 3.5 attention dispatch, response-token totals, and step-local
  MTP telemetry. It passed 42 focused CPU tests and the existing 34-test
  SAMPO/core regression. The explicit exclusion audit found no runtime
  environment, `sitecustomize`, TurboQuant bootstrap/compatibility/test,
  package hook, or TurboQuant-only vLLM server change. No branch, commit, push,
  or release metadata was created.
- [x] (2026-07-27 06:30Z) Run the safe single-GPU source gates against the
  reconstructed veRL candidate. Python 3.13.12 imports resolved exclusively
  from the candidate worktree; the dense-FSDP chunked-entropy node passed, and
  the complete single-GPU FSDP regression file plus Qwen 3.5 wrapper test
  passed five cases. Both dstack workers were healthy, idle, and unsliced
  before and after, with no retained compute process. These tiny correctness
  fixtures are source evidence, not workload or capacity qualification.
- [x] (2026-07-27 06:57Z) Close the TRL prepublication combined-order gate.
  The exact vLLM-generation, GRPO, and distillation order that previously
  reported one NCCL-contamination failure now reports 153 passed and 60
  skipped. A fresh post-run interpreter confirms distributed and CUDA state
  are uninitialized, and no process or GPU client remains.
- [ ] Publish `online-rl-verl-py313` only after committing and pushing the
  CarbonTeq fork, generating a dependency-only Python 3.13 lock, adding the
  dormant kind stage to the publication graph, passing the
  clean-checkout/remote release gate, and running the cross-interpreter
  actual-job smoke and bounded GPU qualification.
- [x] (2026-07-27 03:23Z) Make `job run` reuse a matching mode-`0600`
  packed-image receipt on cache hit and submit only the final image digest.
  The first detached local SFT run reused the package, pulled the cold image,
  and reached real GPU execution.
- [x] (2026-07-27 02:05Z) Make each new submission retain a provider-neutral,
  secret-free evidence locator containing the tracking provider, source ID,
  project/scope, and credential-free endpoint. Reconcile, cleanup,
  cancellation recovery, and `run show` now use that immutable destination
  while loading only credentials and CA material from the protected current
  environment. Legacy v1-v4 receipts retain current-config fallback with a
  visible warning; explicit tracking-disabled v5 runs remain disabled.
- [x] (2026-07-27 02:12Z) Split immutable capsule planning from launch
  planning. `PackageOverrides` admits only target and runtime profile;
  `PlannedJobPackage -> PackedJobPackage` owns publication inputs, while
  `PlannedJobExecution -> PackedJobExecution` adds run ID, provider policy,
  environment names, tracking destination, and worker mounts. `job pack`
  exposes only package-affecting options and succeeds without dstack worker
  storage; `job plan` and `job run` continue to fail closed when launch
  bindings are incomplete.
- [x] (2026-07-27 02:20Z) Complete the repository validation ladder:
  locked Python 3.12 sync, Ruff, Pyright, all eight import contracts, 621
  passing tests with 16 intentional skips, execution-log invariants, and
  `git diff --check`. Refresh the exact training catalog lock digest after the
  execution packages changed `uv.lock`, and make the lab entry regression test
  enforce side-effect-free configuration rather than obsolete scratch
  creation.
- [x] (2026-07-27 03:58Z) Re-run the complete validation ladder after durable
  admission, expanded Observatory job views, distillation preparation, and
  cancellation characterization: locked Python 3.12 sync, Ruff, Pyright, all
  eight import contracts, 648 passing Python tests with 16 intentional skips,
  19 passing frontend tests, a successful TypeScript check and production
  build, ordered unique execution-log sequences, and `git diff --check`.
- [x] (2026-07-27 04:20Z) Harden durable admission after an independent
  lifecycle audit. Provider submission exceptions now retain explicit
  `submission_failed` state and require `run retry-submit`; reconciliation is
  idempotent; read-only status never pumps the queue; no-op tracking has an
  honest provider-terminal barrier; successful untracked outputs are protected
  from cleanup; waiting positions are worker-local; exact dstack host
  placement, provider-binding fingerprints, snapshot invariants, directory
  fsync, and bounded terminal-entry retention fail closed. The obsolete
  process-local queue and characterization script were removed. Focused
  admission, reconciliation, cleanup, execution-configuration, and CLI tests
  passed 71 tests; Ruff, Pyright, all eight import contracts, and both
  repository diff checks passed.
- [x] (2026-07-27 04:32Z) Correct two P0 submission-race defects found by the
  follow-up audit before another live job. A nonblocking per-run kernel claim
  now permits exactly one provider caller without serializing different
  workers and is automatically released on process death. Ambiguous provider
  exceptions keep the physical-worker reservation quarantined until explicit
  idempotent retry resolves the deterministic run. Tests reproduce concurrent
  CLI callers and abrupt process death after provider acceptance. Local target
  aliases now share one host key; run history is newest-first within lifecycle
  priority; next-admission failure is visible from reconciliation; and pruned
  terminal entries are retained as compact mode-`0600` receipts rather than
  discarded. The expanded focused framework, CLI, and deployed-Observatory
  qualification suite passed 86 tests with Ruff and Pyright clean.
- [x] (2026-07-27 04:34Z) Re-run the complete repository gate after admission
  concurrency recovery, canonical cross-provider worker identity, compact
  terminal archival, and the deployed-Observatory public API qualifier:
  locked Python 3.12 sync, Ruff, Pyright, all eight import contracts, 659
  passing Python tests with 16 intentional skips, 19 frontend tests, three
  Playwright journeys, TypeScript, the production frontend build, ordered
  unique execution events, and diff checks across framework, ai-infra, dstack,
  and TRL worktrees.
- [x] (2026-07-27) Make `--target` override the selected job's exact primary
  execution target rather than only a top-level `target` seat. Training jobs
  replace the nested `TrainingBinding.target`; inference-only jobs replace
  their nested inference target; eval/serve jobs keep their explicit target
  and colocated inference target synchronized. Ambiguous, unsupported, and
  explicit unchanged CLI overrides fail closed, and the worker reapplies the same pure
  override from the verified launch target before checking package identity.
- [x] (2026-07-27 04:05Z) Remove linked `COPY` from mutable actual-job named
  context inputs after a target-only repack reused a stale cached
  `package.json` layer. The manifest/build-argument identity check rejected the
  image before publication; an uncached reproduction passed. Normal ordered
  `COPY` retains the expensive dependency-layer cache while making every
  source, config, dataset, and manifest checksum part of the ordinary parent
  chain.
- [x] (2026-07-27 03:05Z) Remove dstack `files` from normal submissions. The
  provider contract rejects a legacy bundle on the digest-only path.
- [x] (2026-07-27 00:41Z) Qualify one packed SFT job and the fifteen-update
  packed GSM8K GRPO job, including remote Trackio/Doris/Observatory evidence
  and bounded cleanup. The first terminally successful GRPO capsule established
  fused-path runtime evidence but was not research-ready because it lacked
  entropy. The corrected non-fused capsule completed all fifteen updates,
  emitted entropy on every update, retained four outputs, reached five of five
  required plus one of one active conditional evidence groups, and passed
  independent remote readback and exact cleanup.
- [ ] Qualify managed evaluation end to end (completed: publish the corrected
  eval kind image; execute two real GSM8K rollouts; retain the native
  evaluation and serving-log artifacts; add provider-neutral rollout
  population counters; rebuild and rerun the actual-job capsule; reconcile
  provider and Trackio success; verify all counters, two traces, and two
  artifact links directly in Doris; project mean reward 1.0, success rate 1.0,
  and compact slice labels through corrected Observatory code; verify the
  deployed Observatory sees complete counters; perform exact cleanup with both
  artifacts retained; package GSM8K and Reverse Text as two independently
  hashed wheels from one pinned checkout; add and validate the self-contained
  `eval/verifiers-managed-general@1` definition after the external-endpoint
  characterization; execute two reward-bearing Reverse Text traces; reconcile
  Trackio/Doris and clean the exact workspace; remaining: deploy the corrected
  WireTrace projector from an intentional clean framework commit).
- [x] (2026-07-27 01:48Z) Qualify the packed serving capsule through dstack on
  `carbonteq-ai-workstation.lan`. The bounded Qwen 3.5 2B workload measured
  concurrency 1, 2, 4, and 8 with two 1024-input/128-output batches per point.
  All 30 measured requests completed. Aggregate output throughput increased
  from 114.52 to 825.92 tokens/s; at concurrency eight p95 TTFT was 105.96 ms,
  p95 TPOT was 9.22 ms/token, and peak VRAM was 77,944,455,168 bytes.
  Trackio retained the serving result and 30 request traces; Doris independently
  contained one config row, 94 metric rows, 175 system-metric rows, 30 trace
  rows, and one artifact link. The local current Observatory selected the
  concurrency-eight point and classified the bounded sweep as passing but
  unsaturated. The deployed Observatory discovered the run, artifact, and
  traces but fell back to its generic view because its preceding clean commit
  lacks the `serve.benchmark` job-view registration. Exact-worker cleanup
  removed only the reconciled run workspace and retained the result artifact.
- [x] (2026-07-27 01:58Z) Qualify the packed transform capsule through dstack
  on `carbonteq-ai-workstation.lan`. The transform kind image selected its
  independently locked LLM Compressor interpreter, passed its dependency and
  environment smoke, and was published with zstd, provenance, and SBOM.
  A calibration-free RTN job transformed Qwen 3.5 0.8B into a W4A16,
  group-128, symmetric language-model variant. The worker completed in 9.50
  seconds, reported 2.44 GiB peak allocated GPU memory, validated the
  serialized quantization groups, reloaded the derived model, and generated
  one token. Trackio retained the 842,697,786-byte model artifact; direct Doris
  readback found one config row, ten metric rows, fourteen system-metric rows,
  and one artifact link. The deployed Observatory exposed the completed run,
  transform metric namespace, immutable inputs, and model artifact through its
  provider-neutral generic view. Exact-worker cleanup removed only the
  reconciled workspace and retained the model.
- [x] (2026-07-27 03:30Z) Qualify the packed SFT image through local Docker:
  two real Qwen 3.5 2B LoRA optimizer updates completed, remote Trackio
  retained model/recovery/summary artifacts, provider and evidence
  reconciliation was consistent, the job-aware Observatory projection was
  complete, and cleanup removed the stopped container only after evidence
  retention. The plan-specific append-only execution log records both the
  initial CA-trust failure and the successful identical-image rerun.
- [x] (2026-07-27 03:45Z) Replace dstack's placeholder
  `provider-managed` workspace result with an exact-worker native cleanup
  task. The task re-verifies the terminal source run and observed hostname,
  mounts only the exact run directory, removes and verifies its contents,
  retains its own dstack history, and returns reclaimed-byte evidence. The
  reconciled SFT run workspace on `pop-os.lan` reclaimed 6 logical bytes; the
  separately authorized stale test workspace
  `sft-dstack-qual-20260726-173000` reclaimed 1,034 logical bytes. No cache or
  sibling run was mounted.
- [x] (2026-07-26 22:57Z) Close the dstack pre-assignment cleanup case.
  Terminal failed/cancelled runs with no hostname are now classified from
  dstack's retained native job-submission history. Cleanup records the
  workspace as `not-created` only when every native submission lacks
  provisioning, runtime, and connection evidence; assigned or incomplete
  histories still fail closed. Run
  `6abdc7f9-df28-40e8-a475-9dd7a1574f78` retained an empty bounded diagnostic
  and provider reconciliation, created no cleanup task, and produced a
  mode-`0600` receipt with zero reclaimed bytes.
- [x] (2026-07-27) Add a fail-fast remote Trackio readiness gate to the
  immutable worker entrypoint. Every detached actual-job package whose project
  selects Trackio now verifies its server URL, TLS/network path, write token,
  Trackio API compatibility, and storage-backed write capability before
  constructing the job runtime. The probe is non-mutating and sanitized;
  explicit `tracking = "none"` and ordinary in-process/local Trackio runtimes
  remain available.
- [x] (2026-07-26 23:11Z) Convert provider SIGTERM into graceful worker
  cancellation at the stable runtime boundary. The first SIGTERM now unwinds
  through canonical tracked execution as `cancelled`, exits with code 143, and
  ignores repeated SIGTERM while the bounded tracking finalizer completes.
  Cooperative `OperationCancelled` exits use the same outcome. Focused
  runtime/work tests passed without mutating the already-stranded live run.
- [x] (2026-07-26 23:25Z) Add
  `posttrain run recover-cancelled-tracking RUN_ID` as a separate audited writer
  path. It requires the persisted framework submission, a terminal cancelled
  provider record, exact project/canonical/provider-run/start identity, exactly
  one matching Trackio run, and tracking state `running` or `cancelled`.
  Successful and already-cancelled recovery writes a mode-`0600` receipt;
  mismatch, ambiguity, or another lifecycle state fails before mutation.
- [x] (2026-07-26 23:23Z) Exercise the guarded command only for pre-fix run
  `1d3a4c57-68b3-480c-9136-c2188595b33e`. It finalized exact Trackio run
  `f6bc4696fd6f45fbba52c777e44ec52c` as `cancelled`, wrote mode-`0600` audit
  state, and ordinary read-only reconciliation became `consistent` /
  `cancelled`. No cleanup or other run mutation was performed.
- [x] (2026-07-27 03:54Z) Replace the process-local serial queue with durable
  provider-neutral admission keyed by physical worker placement. Admission
  writes a mode-`0600` atomic snapshot containing the immutable execution plan
  and evidence locator, restores waiting and active work after process restart,
  persists submission intent before provider contact, and releases a worker
  only after provider termination and retained evidence reconcile. The normal
  CLI now exposes waiting position, status, cancellation, and consistent
  reconciliation through this service. Focused execution, local-provider, and
  CLI validation passed 123 tests.
- [x] (2026-07-27 03:54Z) Qualify cancellation before provider submission.
  Framework run `ca60d4dc-8202-4a0e-a3c2-d5298a9bf081` waited behind an active
  capsule on `carbonteq-ai-workstation.lan`; `posttrain run cancel` changed it
  to `cancelled-before-submission` without creating or contacting a dstack run.
- [x] (2026-07-27 04:42Z) Add direct retained-run lookup to Observatory and
  make the deployment gate use it instead of listing a thousand historical
  runs for each qualification identity. The public
  `GET /api/v1/runs/locate?run_id=...` endpoint resolves one canonical run
  across configured sources; the credential-safe deployment qualifier then
  checks the job-aware views for retained data preparation, SAMPO, serving
  smoke, and failed distillation evidence.
- [x] (2026-07-27 04:42Z) Correct the human `posttrain run list` contract.
  New admission-managed and legacy submitted runs now show an explicit
  `state=...` field and a separate `submitted=...` field rather than placing a
  lifecycle state or timestamp in the same unlabeled column. Global `--json`
  output remains stable.
- [x] (2026-07-27 04:42Z) Re-run the complete repository validation after the
  direct Observatory lookup and operator-DX correction: locked sync, Ruff,
  Pyright, eight import contracts, 659 passing Python tests with 16 intentional
  dependency/credential/GPU skips, 19 frontend tests, TypeScript, production
  build, three Playwright journeys, generated-client stability, and all diff
  checks passed. The repository pins Node 24.18.0; this host currently has Node
  22.19.0, which passed the frontend gates but remains a release-environment
  alignment item.
- [x] (2026-07-27 12:42Z) Close the workstation-only ten-update on-policy
  distillation gate. Run `339100a5-a4c2-4ae6-aa5a-1b080513b50e` performed ten
  real LoRA optimizer updates in 136 seconds under the required LoRA student,
  vLLM `weight_sync_mode=lora`, and colocated bf16 teacher; loss stayed in
  0.0639–0.0970 and grad norm in 0.962–1.732 on every step. Reconciliation was
  `consistent` with `recovery_used=false`, four artifacts were retained
  (adapter, recovery checkpoint, summary, Verifiers traces), deployed
  Observatory returned HTTP 200 with `job_kind=train.distill` and zero alerts,
  and exact-worker cleanup reclaimed only the 4,411-byte run workspace.
  The root cause was `num_items_in_batch=0`, fixed in the TRL fork at
  `6e7739b8ec741d21ecd79c0c212694cd15ff20d8` with a regression test.
  Two supply-chain notes: TRL is installed by the job-kind image and is absent
  from the actual-job runtime requirements, so re-pinning the framework alone
  could not have changed the executed TRL — `posttrain-kind-online-rl-trl-py312`
  had to be rebuilt and republished (`sha256:3c793f8c…`) and its in-image
  `trl` commit verified directly. `posttrain-base` was deliberately not rebuilt
  because it copies `workspace.lock.txt` but installs only the locked CUDA
  PyTorch, which a TRL git revision cannot affect.
- [ ] Qualify graceful running cancellation after the dstack task stop path
  propagates the selected `stop_duration` through runner and shim termination
  instead of using a fixed ten-second runner delay followed by a zero-second
  shim timeout. Zero-grace baseline run `6f999e81-e048-4182-81e5-d9c6883bd65c`
  on the RTX PRO workstation required audited recovery. After dstack release
  `371ff53b1d67f254bc6cc4259aae8653c3916b7d` deployed, pop-os gate attempts on
  `targets/pop-os-rtx4090-24gb` showed provider grace is live (~304–316s) but
  Trackio still does not finalize as `cancelled` without recovery: SAMPO run
  `ed9147ca-9efe-47c5-a5ff-c5181968fed1` completed during grace (Trackio
  `succeeded`, provider `cancelled`); GRPO run
  `37d2f98d-9d77-4b37-b78e-06d58a0a0cfa` retained eight traces at cancel,
  continued rollouts during grace, and left Trackio `running` after hard kill.
  Workstation cancel-gate attempts and any distill-owned runs on
  `carbonteq-ai-workstation.lan` are out of scope for this item.
- [x] (2026-07-26 23:32Z) Qualify the target-specific packed SFT capsule on
  `carbonteq-ai-workstation.lan`. dstack placed digest
  `sha256:e55522efb9d6aab6d642e2359ba226489afc52f52e8fcb05b55de2e7729b0799`
  on the RTX PRO 6000 worker; two real Qwen 3.5 2B LoRA optimizer updates
  completed in 4.014 seconds. Trackio/Doris retained 28 metric rows, three
  system-metric rows, and three artifact links. The authenticated deployed
  Observatory returned HTTP 200, resolved the run as a complete job view, and
  exposed all three artifacts. Exact-worker cleanup then removed only the
  reconciled run workspace and retained the shared model cache and artifacts.
- [x] (2026-07-27 00:25Z) Complete the first packed fifteen-update GSM8K GRPO
  execution on the RTX PRO worker. Run
  `ca196308-82f4-4629-a9e8-0ad51c544754` performed fifteen real LoRA optimizer
  updates in 559.6 seconds, retained 120 native Verifiers traces plus the
  adapter, step-15 recovery checkpoint, and summary, reconciled successfully,
  and appeared through authenticated remote Observatory. Direct storage
  verification found one config row, 352 metric rows, 775 system-metric rows,
  and four artifact links in Apache Doris. Exact-worker cleanup removed only
  the 4,096-byte run workspace.
- [x] (2026-07-27 00:41Z) Repeat the fifteen-update GRPO qualification with the
  non-fused TRL loss path. Run
  `fe6d67e9-6f68-4d6f-a217-15693571b434` completed in 553.3 seconds and emitted
  finite entropy from 0.512 to 0.859 on all fifteen updates. Authenticated
  deployed Observatory returned `research_ready=true`; direct Doris inspection
  found one config row, 352 metric rows, 765 system-metric rows, and four
  artifact links. Exact-worker cleanup reclaimed the 4,096-byte run workspace
  and retained the traces, adapter, step-15 checkpoint, and summary.
- [ ] Remove compatibility bundle transport and superseded operator scripts
  after parity is proven.

## Surprises & Discoveries

- Observation: a non-finite training metric can be reported for the gradient
  norm while the logged loss looks healthy, which sends diagnosis in the wrong
  direction.
  Evidence: the distillation gate failed four times as `grad_norm=nan` with a
  finite logged loss, so three of those attempts changed loss numerics
  (`use_liger_kernel`, `bf16`, an fp32 local-teacher log-softmax patch, padding
  neutralization) and none helped. The loss was in fact `inf`/`nan`, but
  `transformers` sets `logging_nan_inf_filter=True` by default and substitutes
  the running average for a non-finite step loss before the callback observes
  it. Only `grad_norm`, which bypasses that filter, revealed the failure.
  Implication: when a framework guard rejects one non-finite metric, treat the
  other logged metrics as unreliable rather than as evidence of locality.

- Observation: re-pinning a dependency in the framework does not change what a
  job actually executes when that dependency lives in the job-kind image.
  Evidence: TRL is installed by `containers/posttrain-job-kinds/profiles/`
  `supervised.txt` under the `workspace.lock.txt` constraint, and it is absent
  from the actual-job `runtime.requirements.txt`, which only carries hash-locked
  external dependencies plus framework source. Updating
  `packages/train/pyproject.toml` and `uv.lock` therefore left the published
  `posttrain-kind-online-rl-trl-py312` image running the pre-fix TRL commit. The
  drift was detected only by manually inspecting the published image's
  `org.carbonteq.posttrain.lock-digest` label and then reading `direct_url.json`
  inside the image.
  Implication: nothing in the framework reconciles a published kind image
  against current dependency pins, so a stale kind image can silently invalidate
  any GPU qualification. A drift check belongs in `posttrain doctor`.

- Observation: recording two virtual environments in profile metadata did not
  make the actual-job capsule a two-environment runtime.
  Evidence: before the 2026-07-27 correction,
  `containers/posttrain-job/Dockerfile` installed selected environment wheels
  and all framework source only into inherited `VIRTUAL_ENV` on Python 3.12.
  The veRL Python 3.13 process therefore had neither the packaged environment
  wheel nor the framework agent loop needed to reconstruct the portable
  Verifiers bridge. The first dormant correction still reused a Python 3.12
  resolution under Python 3.13.12 and inherited arbitrary `PYTHONPATH`. The
  corrected design resolves separate closures, binds both into package
  identity, projects only `posttrain.common`, `posttrain.data`, and
  `posttrain.train`, rejects host runtime paths during packing, and verifies
  module origins at build and execution.

- Observation: the existing successful SFT path distributes one reusable job
  image and a second dstack `files` bundle.
  Evidence: dstack configuration contains a `files` mapping for
  `/opt/posttrain/bundle`, while `ExecutionRequest.image` separately names the
  framework runtime image. This is reproducible but creates two packaging and
  transfer identities.

- Observation: embedding run identity in the image manifest destroys useful
  OCI reuse.
  Evidence: the current `ExecutionJobManifest` contains `run_id`, provider, and
  runtime image. If copied into the image, an otherwise identical retry would
  require a new image and the image could not contain its own final digest
  without a circular identity.

- Observation: one Verifiers repository may supply several independently
  installable environments.
  Evidence: the pinned Prime Intellect repository contains GSM8K, Reverse Text,
  Code Golf, and Alphabet Sort under different subdirectories at the same
  commit. A packer should fetch that commit once and build only the selected
  subdirectories.

- Observation: developer-side environment activation is incompatible with
  detached packing.
  Evidence: GSM8K validation previously failed on a development environment
  without Verifiers installed. Static composition now succeeds; native
  environment activation remains an image qualification and worker-execution
  concern.

- Observation: the first CLI target override silently ignored standard SFT and
  GRPO packages because their scheduler-facing target is nested in the
  `TrainingBinding`, not bound to a recipe seat named `target`.
  Evidence: `plan_job_execution` previously rewrote only
  `package.bindings["target"]`, while `_execution_target` explicitly preferred
  the prepared training binding. Applying the change only during local planning
  would also make the worker reject the image because it reconstructs the
  packaged work package and checks the resolved-input digest.

- Observation: source provenance alone is insufficient to activate an
  environment after installation.
  Evidence: the current catalog schema converts `factory: str` into a live
  callable while decoding. The pinned GSM8K package itself exports Taskset
  classes, not a `load_environment` factory. A packed job therefore needs a
  serializable activation: declarative Verifiers configuration normally, with
  `module:callable` reserved for packages that genuinely export a custom
  factory. Activation happens only inside the actual-job image.

- Observation: the multi-environment checkout strategy works against the real
  upstream layout, not only a fake Git gateway.
  Evidence: packing `environments/gsm8k_v1` and
  `environments/reverse_text_v1` at commit
  `284a868d6a9022109b749710672a0460e8a996d4` performed one checkout and emitted
  two ordered subdirectory locks.

- Observation: `uv build --wheel` writes a one-byte `.gitignore` beside the
  wheel in a fresh output directory.
  Evidence: the first real wheel qualification initially rejected the extra
  output. The builder now permits exactly that known `*` marker while still
  rejecting any other extra file, and then built two real wheels reproducibly.

- Observation: a generic online-RL image built during characterization is
  useful implementation evidence but has the wrong repository ownership.
  Evidence:
  `registry.lan/carbonteq/posttrain-online-rl-runtime@sha256:5b1f1ac094f5bfe2acb1b32745737f288484bcce6879d11dcc804c024cbf2595`
  was produced by an `ai-infra` script. The Dockerfile and build contract must
  move here before it becomes a selected framework release.

- Observation: one kind dependency layer can serve many actual jobs without
  absorbing their environment code.
  Evidence: the new `supervised`, `online-rl`, `eval`, `serve`, and `transform`
  BuildKit targets resolve entirely from framework locks; static validation
  rejects GSM8K, AutomationBench, and other concrete environment names from
  their Docker inputs.

- Observation: the previous “static” SFT/DPO plan used the activated
  `JobRuntime` and therefore materialized dataset bytes during planning.
  Evidence: the CLI constructed a runtime with its dataset seat resolver before
  calling `prepare_work_package_job`. Standard definitions now declare
  selection-time `DatasetLoadPlan` seats separately from runtime
  `SupervisedDataSource`/`PreferenceDataSource` seats; static validation has a
  focused no-materialization test.

- Observation: a pre-I/O identity cannot be the final package identity.
  Evidence: Git tree digests, built wheel digests, the combined dependency
  lock, and normalized dataset bytes are unknown until packing. The framework
  now distinguishes `plan_key`, final manifest `package_key`, and publication
  key.

- Observation: installing framework code in both kind and actual-job layers
  creates two competing framework revisions.
  Evidence: the first kind Dockerfile copied the workspace and installed the
  worker even though the actual-job manifest separately records
  `framework_source_digest`. Kind images are now dependency-only; exact
  framework/project source is installed only in the actual-job image.

- Observation: logical job kind is too coarse to select a Python dependency
  environment.
  Evidence: the TRL/Verifiers online-RL profile is compatible with the
  framework's Python 3.12 venv, but the currently qualified veRL/TurboQuant
  runtime uses Python 3.13, Torch 2.11/CUDA 13, vLLM 0.25.1, and a
  process-isolated veRL worker. A single `online-rl` dependency layer silently
  selected the TRL stack and would have misrepresented veRL compatibility.

- Observation: selecting Python 3.13 as the actual-job `VIRTUAL_ENV` is not a
  viable veRL design.
  Evidence: every framework package currently requires Python
  `>=3.12,<3.13`, while the isolated veRL agent loop imports framework
  train/common code and unpickles the selected Verifiers environment inside
  the Python 3.13 process. The actual-job image must retain Python 3.12 as its
  control environment and install a deliberate worker/environment projection
  into `/opt/posttrain-verl`; simply inheriting one selected venv cannot work.

- Observation: linked copies from a changing named BuildKit context are not a
  safe integrity boundary for actual-job capsules on the installed builder.
  Evidence: after changing only the packed target overlay, the cached build
  reused the prior linked `package.json` layer and the final manifest-key check
  failed. Rebuilding the same staged bytes with `--no-cache` passed. Replacing
  the small mutable-context `COPY --link` operations with ordered `COPY`
  preserves the dependency cache and prevents linked-layer rebasing from
  bypassing the new context identity.

- Observation: a lifecycle command can be machine-readable while still being
  ambiguous for a human operator.
  Evidence: `posttrain --json run list` already returned separate
  `admission_state` and `submitted_at` fields, but the default terminal output
  placed either value in one unlabeled column. The corrected output labels
  both values and distinguishes pre-admission history as `legacy-submitted`.

- Observation: omitting a kind-provided package from the emitted runtime lock
  is insufficient if installation re-resolves wheel metadata.
  Evidence: Alphabet Sort correctly omitted the Verifiers dependency already
  installed by the online-RL kind image, but `uv pip install
  --require-hashes` still inspected the wheel metadata and rejected the
  unpinned `verifiers` edge. The runtime lock is already a complete expanded
  closure, so actual-job installation now uses `--require-hashes --no-deps`:
  every explicit line is installed and hashed, while metadata cannot add an
  unrecorded dependency.

- Observation: dstack's `instances` field is a structured resource selector,
  not a list of host-name strings.
  Evidence: the first RTX PRO submission encoded
  `instances: [carbonteq-ai-workstation.lan]`; dstack interpreted the string as
  an instance type and returned no offers. Encoding
  `instances: [{hostname: carbonteq-ai-workstation.lan}]` selected the intended
  worker, whose observed OS hostname was
  `carbonteq-ai-workstation-X870-EAGLE-WIFI7`, and the bounded SFT
  qualification succeeded.

- Observation: the existing veRL research lock is qualification evidence, not
  a kind-image release lock.
  Evidence: `../verl-upstream/runtime/turboquant-cu130/uv.lock` selects veRL
  from an editable local path and directly includes AutomationBench and GSM8K.
  The checkout is detached at upstream commit
  `a35908ca3c9632859c58d6a2855d858918ae21dc` with uncommitted CarbonTeq
  patches. The release gate rejects editable/path sources, concrete
  environments, dirty checkouts, and commits not reachable on the configured
  CarbonTeq remote.

- Observation: a successful native evaluation can still be evidence-incomplete
  when its adapter discards irreducible rollout outcomes.
  Evidence: run `ef06cd8c-f522-41db-8bc9-9b44b9268b5a` completed two of two
  GSM8K rollouts with correct rewards and synchronized both Verifiers traces,
  but emitted none of `eval/run/rollouts_attempted`,
  `eval/run/rollouts_complete`, `eval/run/rollouts_failed`, or
  `eval/run/rollouts_truncated`. The deployed Observatory therefore reported
  zero of one required evidence groups despite consistent provider/Trackio
  success.

- Observation: retained Verifiers v1 traces use the WireTrace schema rather
  than Observatory's older flattened reward/error/truncation fields.
  Evidence: the retained records store reward components under `rewards`,
  task identity under `info`, failures under `errors`, completion under
  `is_completed`, and truncation through `stop_condition` or model-call finish
  reason. Before correction Observatory scanned both expected traces but
  reported null mean reward and success rate.

- Observation: producer and read-product source identities can diverge even
  when they share the same tracking backend.
  Evidence: eval capsule
  `registry.lan/carbonteq/posttrain-job@sha256:6042327cc35599e4792725c95f4f82b6e54e2f12f42e0b4b2c92403917bb0b4a`
  contained framework source digest
  `e4c8ed18d12fcdecdb2208f0d1cd26e9861c44863729f8ec6a2a1f8c5ff64def`,
  while `https://observatory.lan` still ran clean commit
  `5b81cdabdbb77297c483955f0853439bf197c279`. The deployed service read all
  new counters but could not project WireTrace reward/success until the
  uncommitted projector is intentionally committed and redeployed.

- Observation: vLLM kind images require a C++ compiler even when their Python
  dependency installation and static import smoke pass.
  Evidence: the first eval capsule reached FlashInfer JIT on the GPU worker but
  failed because GCC could not execute `cc1plus`. A shared
  `vllm-kind-common` stage now installs `g++`, and TRL online RL, evaluation,
  and serving smoke checks require `c++`.

- Observation: `eval/verifiers-general@1` is an external-endpoint operation,
  not a self-contained CLI job.
  Evidence: the first two-environment capsule installed both environment
  wheels and activated Reverse Text, but both traces ended with
  `ProviderError: All connection attempts failed` because no component owned
  the endpoint lifecycle. The new
  `eval/verifiers-managed-general@1` definition launched vLLM, reran the same
  selected cell, and retained rewards `0.2823529411764706` and `0.0` with no
  rollout errors.

- Observation: multi-environment source deduplication survives actual image
  assembly and GPU execution.
  Evidence: package
  `c69819c51b39d1dd94920f3122b3c99a9a9a8bd2c2b5f02da4bab2788cc1c02e`
  retained GSM8K and Reverse Text tree/wheel digests from one repository at
  commit `284a868d6a9022109b749710672a0460e8a996d4`, installed both activations,
  and executed the second environment on the RTX PRO worker.

- Observation: the Trackio Git package tries to build its dashboard during a
  client-only worker image build.
  Evidence: the first kind-image build failed because npm was intentionally
  absent. Worker images now build the pinned Python client with
  `SKIP_FRONTEND_BUILD=1`; the separately deployed Trackio server owns the
  dashboard frontend.

- Observation: the resolved training snapshot named a backend but omitted its
  backend options.
  Evidence: source revision, dependency-lock digest, kernel choices, chunk
  sizes, and veRL interpreter/worktree settings could affect execution without
  changing `resolved_inputs_digest`. Training snapshots now retain the complete
  JSON-valued `backend_options`, so image/package lineage and runtime
  reconstruction cannot silently ignore those choices.

- Observation: independently copied workspace packages are not independently
  buildable merely because their external dependency closure is locked.
  Evidence: the first actual-job build rejected `workspace = true` references
  when each package was installed from its staged subdirectory. The image now
  derives one minimal uv workspace from the hash-bound `code.requirements.txt`
  and projects named indexes from the framework root `pyproject.toml`, itself
  covered by `framework_source_digest`.

- Observation: disabling build isolation requires the shared kind image to
  contain the build backend, not only runtime dependencies.
  Evidence: the next real build reached Hatchling and failed before source
  installation because it was absent. Every kind image now installs the
  separate `build-tools.lock.txt` closure with `--require-hashes`; actual-job
  source builds do not fetch an implicit backend.

- Observation: host TLS trust does not automatically cross an OCI boundary.
  Evidence: the first detached GPU SFT run reached `https://trackio.lan` but
  Trackio reported `CERTIFICATE_VERIFY_FAILED` and fell back locally, even
  though the developer host trusts the internal CA. Remote evidence is a
  qualification gate, so execution providers need an explicit read-only trust
  bundle contract; verification will not be disabled.

- Observation: supplying a remote Trackio URL does not make
  `trackio.init()` a remote-readiness barrier.
  Evidence: Trackio deliberately tolerates remote-client construction and
  synchronization failures by buffering locally. The first detached SFT run
  therefore consumed two optimizer updates before its required artifact export
  exposed the missing container CA trust. A detached qualification run needs a
  separate fail-closed probe before model loading, while interactive/local
  Trackio may retain its offline-friendly behavior.

- Observation: dstack's SSH GPU fleet does not offer a CPU-only task on a
  hostname-selected GPU worker.
  Evidence: a read-only plan for `instances: [{hostname: pop-os.lan}]` and
  `gpu.count=0` returned zero offers, while the otherwise equivalent
  `gpu.count=1` plan returned the exact idle worker. Cleanup therefore first
  plans CPU-only, requests one short-lived scheduler GPU reservation only when
  the CPU plan has no offer, and never initializes CUDA. The initial CPU-only
  applied characterization task failed with `no offers`; the corrected
  fail-before-apply fallback completed successfully and its history remains in
  dstack.

- Observation: a terminal dstack run can fail before it ever receives a worker,
  so requiring a hostname for every workspace cleanup is both misleading and
  non-idempotent.
  Evidence: run `6abdc7f9-df28-40e8-a475-9dd7a1574f78` was terminal `failed`
  with no hostname or provider logs. The native run retained complete job
  submissions with no `job_provisioning_data`, `job_runtime_data`, or
  `job_connection_info`; no worker-scoped cleanup task was needed.

- Observation: provider cancellation and tracking cancellation were not one
  atomic lifecycle before the worker gained signal handling.
  Evidence: dstack run `1d3a4c57-68b3-480c-9136-c2188595b33e` received SIGTERM
  during model download and became terminal `cancelled`, but its Trackio run
  `f6bc4696fd6f45fbba52c777e44ec52c` remained `running`. The tracked executor
  already finalized `KeyboardInterrupt` and `SystemExit` as cancelled, but the
  default SIGTERM action terminated Python without raising either exception.

- Observation: dstack's documented graceful-stop duration does not currently
  govern the complete dockerized task stop path.
  Evidence: both installed dstack 0.20.29 and upstream `master` at
  `1a2b5d5299c387a6b31c5140d56ba0d493fd4ace` request runner stop, wait a fixed
  ten seconds, and then call
  `terminate_task(..., timeout=0)` in
  `jobs_terminating.py`, while the public task schema defaults
  `stop_duration` to five minutes. Two current-capsule cancellations, including
  one after retained rollout traces existed, left the exact Trackio run
  `running` and required the already-audited recovery command. Framework
  SIGTERM handling remains correct but cannot execute after an immediate
  provider kill.

- Observation: singular-experiment policy must be scoped by physical worker,
  not by framework process or project.
  Evidence: one global active slot would unnecessarily serialize the RTX 4090
  and RTX PRO 6000. Admission schema v2 keys the active run by resolved target
  hostname, allows one run on each worker, and still queues a second run for
  the same placement. Restart, independent-worker, queued-cancel, and
  evidence-barrier tests cover this distinction.

- Observation: required output roles are success obligations, not cancellation
  obligations.
  Evidence: after exact tracking recovery, provider and Trackio both reported
  `cancelled`, but reconciliation initially remained inconsistent because it
  demanded `model` and `summary` artifacts that the interrupted run never
  produced. Reconciliation now retains the declared roles for lineage while
  reporting no missing roles for matched `failed` or `cancelled` outcomes.

- Observation: a mixed PyPI/CUDA dependency projection needs the same uv index
  strategy as the workspace lock that selected it.
  Evidence: the first online-RL environment-wheel install searched the CUDA
  index exclusively for `filelock`, where the pinned version was absent. The
  dependency compiler now passes `--index-strategy unsafe-best-match`; hashes
  and exact versions remain mandatory, so this changes index search rather
  than resolution identity.

- Observation: online-RL batch-seat inconsistency is statically knowable.
  Evidence: the first GRPO capsule inherited a global batch of two while the
  selected two prompt groups and four generations require eight samples per
  update. A qualification-specific training binding now declares eight, and
  job validation rejects the mismatch before packing or GPU admission.

- Observation: replaying retained Verifiers evidence after training cannot
  reuse its source logical steps on a writer that already observed all trainer
  updates.
  Evidence: the next capsule completed all fifteen backward passes and then
  failed finalization because replay began at source step zero after Trackio
  had observed step fifteen. Replay now appends with no logical step and
  retains the trace's original step as the `source_step` attribute.

- Observation: ordinary `COPY` alone did not invalidate the installed
  BuildKit builder's first mutable named-context layer on every first attempt.
  Evidence: a changed package key still reused a stale first context layer
  until a package-key-dependent `RUN` separated the immutable kind parent from
  the mutable job context. Static Dockerfile validation now requires that
  barrier before the first actual-job context copy.

- Observation: TRL's Liger GRPO loss path does not expose the same evidence as
  its non-fused loss path.
  Evidence: the fifteen-update run logged reward, clipping, gradients,
  learning rate, rollout population, and throughput, but no entropy. The
  pinned TRL source appends entropy only in `_compute_loss`; `compute_liger_loss`
  emits clipping and optional KL. Observatory therefore reported four of five
  required GRPO groups and `research_ready=false`.

- Observation: the first 32K serving start on the RTX PRO 6000 is dominated by
  KV-cache reservation and kernel warmup rather than model weight loading.
  Evidence: Qwen 3.5 2B weights loaded in 1.29 seconds and occupied 3.63 GiB,
  while engine initialization took 68.17 seconds and reserved 67.01 GiB for
  5,510,144 KV-cache tokens. The subsequent bounded sweep completed without a
  request failure and remained unsaturated at concurrency eight.

- Observation: deployed Observatory compatibility must be qualified per job
  kind, not inferred from run discovery.
  Evidence: `observatory.lan` listed the successful `serve.benchmark` run,
  retained its artifact and 30 traces, but returned no registered job view and
  fell back to generic mode. The current local code projected the complete
  four-point serving-capacity view from the same remote Trackio evidence.

- Observation: publishing a small quantized model can dominate the bounded
  transform's end-to-end time and retained storage.
  Evidence: the LLM Compressor subprocess finished in 9.50 seconds and the
  inner transform measured 5.99 seconds, but the tracked operation completed
  88.03 seconds after it started while hashing and retaining an 842,697,786-byte
  model artifact. The execution workspace was only removed after Trackio and
  Doris independently retained the artifact link.

## Decision Log

- Decision: bind one dependency closure to each interpreter rather than
  installing a Python 3.12 resolution into both environments.
  Rationale: wheel markers, ABI tags, transitive versions, and hashes may
  differ between Python 3.12 and Python 3.13.12 even when the selected
  environment wheels are identical. Each closure therefore records its role,
  Python target, capsule interpreter, requirements digest, and resolution
  digest in `JobPackageManifest`; changing either closure changes the package
  key. A ready veRL release must execute the dedicated Docker/Bake import smoke,
  not merely match strings in a Dockerfile.
  Date/Author: 2026-07-27 / Codex.

- Decision: project a minimal content-addressed framework namespace into the
  veRL interpreter instead of installing the Python 3.12 framework
  distributions a second time.
  Rationale: framework package metadata intentionally requires Python
  `>=3.12,<3.13`, but the veRL worker imports the framework contracts, portable
  Verifiers bridge, agent loop, and retention finalizer under Python 3.13.
  Copying the exact `common`, `data`, and `train` namespace sources from the
  already digested actual-job context preserves one framework source identity
  without weakening package metadata or projecting unrelated project code.
  The isolated launcher explicitly prepends this path, and both the release
  gate and actual-job smoke bind the path and worker module.
  Date/Author: 2026-07-27 / Codex.

- Decision: the framework owns universal, job-kind, and actual-job images.
  Rationale: image contents encode framework dependency, worker, job,
  environment, and dataset semantics. Infrastructure should provide a registry,
  BuildKit reachability, worker compatibility, credentials, and lifecycle
  policy without owning those semantics.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: use the final OCI image as the sole normal distribution unit.
  Rationale: one immutable digest should cover everything needed to reconstruct
  job meaning. This removes ordering and reconciliation problems between an
  image digest and a separately uploaded directory.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: actual-job named-context inputs use ordinary ordered `COPY`, not
  `COPY --link`.
  Rationale: code, configuration, datasets, and the package manifest are small
  relative to the shared CUDA/backend layers, and correctness is more valuable
  than linked-layer rebasing. Dependency locks and environment wheels remain
  before source/config/data in the Dockerfile, so normal cache reuse still
  avoids reinstalling the expensive runtime closure.
  Date/Author: 2026-07-27 / Codex.

- Decision: keep packed-job identity separate from execution identity.
  Rationale: code, environment, dataset, and resolved selections determine the
  reusable capsule. Run ID, attempt, target, provider, credentials, and mounts
  describe one launch and must not invalidate cached image layers.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: persist a secret-free evidence locator in the execution submission,
  not the tracking client or its credentials.
  Rationale: provider lifecycle and evidence reconciliation happen at different
  times. A later change to `.posttrain/project.toml` or the local service
  binding must not redirect reconciliation, cleanup, cancellation recovery, or
  Observatory inspection to another Trackio/W&B namespace. The receipt records
  destination identity; the current protected environment supplies only
  credentials and trusted CA material. Legacy receipts cannot reconstruct the
  original locator, so their fallback remains explicit and warned rather than
  being silently rewritten.
  Date/Author: 2026-07-27 / Codex.

- Decision: represent package planning and launch planning with different
  types and different override sets.
  Rationale: provider choice, retries, priority, named environment variables,
  run ID, and host mount paths cannot change OCI capsule bytes. Requiring those
  values during `job pack` couples reproducible publication to whichever
  worker happens to be attached to the developer machine. Target and runtime
  profile remain package inputs because they change resolved job meaning or
  select a different job-kind runtime. The composed execution facade preserves
  the existing `job plan` and `job run` payload while preventing a packed
  capsule from accepting a post-pack target/runtime change.
  Date/Author: 2026-07-27 / Codex.

- Decision: internal HTTPS trust is an execution-provider binding, not image or
  package content. A configured host/instance CA bundle is exposed at
  `/opt/posttrain/trust/ca-certificates.crt`, and providers set
  `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` in the launch environment. Local
  Docker enforces a read-only bind and rejects a missing source. dstack uses a
  mandatory instance mapping and a file preflight because dstack 0.20.29 does
  not expose a read-only flag for instance mounts; this limitation must remain
  visible until dstack adds that capability or the worker contract adopts a
  different verified trust projection. The CA bytes and source path never
  enter `JobPackageManifest`.
  Rationale: jobs must verify internal TLS without disabling verification or
  baking one network's CA into reusable job images.
  Date/Author: 2026-07-27 / Codex.

- Decision: remote Trackio readiness is a worker-entrypoint gate, not a
  provider health check or a training-backend concern.
  Rationale: local Docker, dstack, custom project entries, and every job kind
  must receive the same guarantee after launch-time URL, token, and trust
  injection but before expensive operation code. The Trackio adapter checks an
  empty artifact-digest set through the authenticated write API, exercising
  TLS, routing, server compatibility, authorization, and storage without
  creating a run. Failures are replaced with a secret-free contract error.
  Direct in-process runtimes do not opt into this detached evidence gate, and
  projects selecting `tracking = "none"` remain valid offline executions.
  Date/Author: 2026-07-27 / Codex.

- Decision: the evidence qualification uses TRL's non-fused GRPO loss until
  the maintained TRL fork makes the Liger path evidence-equivalent.
  Rationale: policy entropy is a required collapse/exploration signal. It must
  be measured from policy logits during the actor update and cannot be
  reconstructed faithfully from sampled-token rollout log-probabilities.
  Keeping the prior fused run provides an honest runtime comparison without
  weakening the research-ready contract.
  Date/Author: 2026-07-27 / Codex.

- Decision: translate SIGTERM at the stable worker entrypoint and retain
  read-only reconciliation.
  Rationale: local Docker and dstack both express ordinary cancellation as
  SIGTERM. Raising `SystemExit(143)` lets existing tracked execution durably
  write the canonical `cancelled` outcome and then preserves the provider's
  conventional process status. Duplicate SIGTERM is ignored only while that
  unwind is active so it cannot interrupt tracking finalization; the provider's
  bounded grace period may still end with SIGKILL. Reconciliation must continue
  to report disagreement rather than altering provider evidence.
  Date/Author: 2026-07-26 / Codex.

- Decision: make framework admission durable and singular per physical worker
  while leaving resource scheduling to dstack.
  Rationale: the framework owns logical idempotency, one-experiment-per-worker
  research policy, cancellation intent, and the retained-evidence barrier.
  dstack still owns GPU availability, placement, native task lifecycle, and
  infrastructure retries. A mode-`0600` atomic admission snapshot and
  interprocess lock preserve waiting/active state across CLI processes;
  submission intent is persisted before provider contact, and an accepted run
  does not release its placement until reconciliation is consistent.
  Date/Author: 2026-07-27 / Codex.

- Decision: expose ambiguous submission as a recoverable lifecycle state
  rather than silently resubmitting from status or cancellation.
  Rationale: a provider call can succeed remotely and fail locally before its
  handle is persisted. Retaining the immutable intent as
  `submission_failed`, quarantining the local placement, and requiring an
  explicit idempotent `run retry-submit` makes that ambiguity visible without
  admitting another run onto a GPU that may already be occupied. A nonblocking
  per-run kernel claim lets only one CLI process contact the provider and is
  automatically released after process death. Restored entries also compare a
  secret-free provider-binding fingerprint before contacting the provider so
  configuration drift cannot redirect a historical run.
  Date/Author: 2026-07-27 / Codex.

- Decision: let a deliberately untracked run reconcile at the provider
  terminal barrier without claiming retained evidence.
  Rationale: the no-op observer is a supported explicit selection. Its
  reconciliation must record that no durable evidence was asserted, and
  cleanup must preserve successful workspaces that declare required outputs.
  This is safer and more honest than either blocking admission forever or
  fabricating a Trackio result.
  Date/Author: 2026-07-27 / Codex.

- Decision: treat dstack running-cancel grace as an infrastructure release
  gate and retain explicit Trackio recovery only as a fallback.
  Rationale: the stable worker already handles SIGTERM and bounded finalization,
  but both the installed and current upstream dstack server allow only a fixed
  ten-second runner interval before sending a zero-second task termination
  timeout. Changing framework reconciliation to mutate
  tracking evidence would hide the provider defect. The dstack deployment must
  be upgraded or patched to propagate a non-zero grace duration, then the same
  live cancellation scenario must prove terminal Trackio state without running
  `recover-cancelled-tracking`.
  Date/Author: 2026-07-27 / Codex.

- Decision: recovery is an explicit audited writer command, never a side
  effect of reconciliation.
  Rationale: repairing a pre-fix or SIGKILL-stranded tracking run necessarily
  mutates retained evidence. The command re-collects the immutable framework
  submission, requires provider-terminal `cancelled`, reads the normalized
  Trackio identity, and makes the Trackio adapter re-enumerate exactly one
  matching provider run before and after writing `cancelled`. A mode-`0600`
  receipt records both provider identities and the original start time.
  Ordinary `run reconcile` remains read-only and must be run afterward.
  Date/Author: 2026-07-26 / Codex.

- Decision: let the stable worker publish a minimal terminal-workspace marker
  after the evidence finalizer, while infrastructure owns retention timing.
  Rationale: the framework is the only layer that knows tracking and required
  artifact finalization have unwound; the worker is therefore responsible for
  the provider-neutral terminal fact. The worker must not choose a retention
  period or recursively collect other runs. Conversely, ai-infra may apply an
  age and free-space policy only after validating the exact marker and run
  directory. This closes automatic stale-workspace cleanup without moving
  research evidence or experiment policy into infrastructure.
  Date/Author: 2026-07-27 / Codex.

- Decision: dstack workspace cleanup is a post-reconciliation provider action,
  not SSH administration.
  Rationale: the execution provider already owns the immutable source-run
  handle, terminal observation, actual-job image, and exact run workspace.
  Cleanup re-resolves that source run through the dstack SDK, requires the same
  observed hostname, submits a deterministic native task pinned with
  `instances: [{hostname: ...}]`, mounts only the exact run directory, and
  fails closed unless the task reports successful empty verification and a
  non-negative reclaimed-byte count. Provider and cleanup task logs/history
  remain available; caches and sibling runs are outside the mount.
  Date/Author: 2026-07-27 / Codex.

- Decision: represent a provider-proven pre-assignment workspace as
  `not-created`, distinct from `already-absent` and `removed`.
  Rationale: an unassigned cached target string is insufficient evidence.
  The dstack bridge must re-read the terminal native run and find at least one
  retained submission for every job, with no provisioning, runtime, or
  connection evidence anywhere. Only terminal failed/cancelled runs may take
  this path. Missing jobs/submissions, any assignment evidence, or a successful
  run without a hostname remains ambiguous and fails closed without scheduling
  a cleanup task.
  Date/Author: 2026-07-27 / Codex.

- Decision: materialize Git environments during `pack`, never during worker
  startup.
  Rationale: packing can verify an exact commit, build wheels, resolve all
  selected environment dependencies together, record source and wheel digests,
  and fail before consuming GPU capacity. Private credentials remain in the
  developer credential helper or BuildKit SSH session and never enter the
  image or receipt.
  Date/Author: 2026-07-26 / Codex.

- Decision: deduplicate environments by canonical repository and full commit.
  Rationale: evaluation and research jobs may use several packages from one
  monorepo revision. One content-addressed checkout can supply the union of
  normalized subdirectories while retaining a separate package/tree/wheel
  identity for each environment.
  Date/Author: 2026-07-26 / Codex.

- Decision: evaluation adapters return a provider-neutral
  `EvaluationPopulation`, and Observatory computes views from retained native
  traces.
  Rationale: attempted, complete, failed, truncated, and coverage-missing
  counts are irreducible run-level evidence and must survive the backend
  boundary. Mean reward and success rate remain rebuildable views over
  WireTrace records; creating a second score store would split authority from
  the Verifiers traces.
  Date/Author: 2026-07-27 / Codex.

- Decision: distinguish external-endpoint and managed-endpoint evaluation in
  the job-definition identity.
  Rationale: endpoint ownership changes executable behavior and failure
  recovery. `eval/verifiers-general@1` remains valid when an orchestrator
  supplies a declared endpoint; ordinary standalone CLI execution selects
  `eval/verifiers-managed-general@1`, which owns launch, health, evaluation,
  and shutdown in one tracked run.
  Date/Author: 2026-07-27 / Codex.

- Decision: retain environment package locks separately from environment
  activation locks.
  Rationale: one wheel lock owns package name, Git source/subdirectory,
  source-tree digest, and wheel digest. One or more binding activation locks
  may reference that package and retain environment id, activation kind/digest,
  and optional real `module:callable`. This installs a wheel once even when a
  job uses separate train and test configurations, while every activation
  remains immutable job meaning.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: `packages/execution-pack` owns semantic package planning and
  materialization; `packages/execution-buildkit` is its BuildKit adapter.
  Rationale: the CLI should expose packing without owning the reusable
  application logic, and infrastructure should provide services without
  deciding image content.
  Date/Author: 2026-07-26 / Codex.

- Decision: use three identities for one packing lifecycle.
  Rationale: `plan_key` covers static requests known without I/O;
  `package_key` covers the final materialized manifest; `publication_key`
  additionally covers repository, platforms, compression, and attestations.
  Receipts key only the final publication identity, never timestamps or local
  paths.
  Date/Author: 2026-07-26 / Codex.

- Decision: hash selected working-tree bytes rather than requiring a clean Git
  commit for framework/project code.
  Rationale: developers must be able to test uncommitted research changes, but
  copying an entire repository would embed unrelated data and machine state.
  `[tool.posttrain.pack]` declares install roots and explicit includes; the
  framework separately closes over required project configuration.
  Date/Author: 2026-07-26 / Codex.

- Decision: keep base-model weights and mutable outputs outside normal job
  images.
  Rationale: foundation weights would dominate registry churn and invalidate
  otherwise reusable layers. Workers use persistent model/compile caches;
  Trackio or another artifact backend retains final models, summaries, and
  traces. A deliberately offline capsule may opt into weight embedding later.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: select dependency/runtime profiles below the logical job-kind
  level.
  Rationale: `online-rl` describes framework behavior, not a compatible Python
  environment. The TRL implementation uses `online-rl-trl-py312`; the veRL
  implementation uses `online-rl-verl-py313`. The selected profile owns its
  Python targets, dependency constraints, backend interpreter contract, and
  exact kind-image digest. The actual-job `VIRTUAL_ENV` remains the Python
  3.12 control environment for both; the veRL variant additionally owns
  `/opt/posttrain-verl`. Keeping that distinction in planning prevents a
  backend from accidentally running in another backend's environment.
  Date/Author: 2026-07-27 / user and Codex.

- Decision: keep veRL dependencies outside both the developer environment and
  the job control environment.
  Rationale: the project `.venv` and `/opt/posttrain/venv` remain Python 3.12
  framework environments. A veRL job capsule also carries the independently
  locked Python 3.13 environment at `/opt/posttrain-verl`; only the
  process-isolated veRL backend worker uses that interpreter. The dstack client
  environment under `.posttrain/state/` is a fourth, machine-local control
  dependency and is not packed into jobs. “One actual-job image” therefore
  means one distributable OCI artifact, not one Python dependency environment.
  Date/Author: 2026-07-27 / user and Codex.

- Decision: fail closed instead of publishing the dirty veRL qualification
  checkout.
  Rationale: a local tree digest can reproduce a research experiment, but a
  shared kind image is a release artifact. Its veRL source must be a full
  CarbonTeq fork commit present on the configured remote, its checkout must be
  clean, and its dependency-only Python 3.13 lock must contain neither local
  sources nor concrete environments. Until then the profile remains visible
  as blocked metadata and has no BuildKit publication target.
  Date/Author: 2026-07-27 / Codex.

- Decision: reconstruct the veRL release from the published CarbonTeq branch
  instead of committing the detached qualification tree.
  Rationale: `origin/main` already contains the published SAMPO history and
  later upstream work, while the detached tree starts from an older upstream
  revision and mixes supported patches with runtime environments and failed
  TurboQuant research. Selective replay plus full requalification preserves
  released history and keeps experimental failures out of the production
  dependency closure.
  Date/Author: 2026-07-27 / Codex.

- Decision: treat repository creation, immutable source publication, image
  publication, deployment, and live qualification as separate release states.
  Rationale: the public dstack fork and private ai-infra repository now exist,
  but none of the validated candidate branches has been pushed and no service
  was deployed. Recording these boundaries prevents a created repository or a
  green source test from being mistaken for a reproducible or operating
  release.
  Date/Author: 2026-07-27 / Codex.

## Outcomes & Retrospective

The framework-owned OCI path is now qualified for local SFT, dstack-hosted TRL
GRPO, managed evaluation, a bounded serving-capacity sweep, and model
transformation, including
fifteen optimizer updates, cancellation/recovery, exact worker cleanup, remote
Trackio backed by Doris, and authenticated Observatory readback. Evaluation
producer evidence and multi-environment execution are complete. Serving
produced a complete four-point operating curve and retained all request-level
evidence; it passed the product envelope but remained unsaturated at the
configured concurrency-eight ceiling. The separately deployed Observatory is
still pinned to the preceding clean framework commit, so it lacks both the
corrected Verifiers WireTrace projection and the registered serving-capacity
job view that have been validated locally against the same remote evidence.
The transform job is fully visible through the deployed generic Observatory
view, including its metrics and retained derived-model artifact. The
submission lifecycle now also survives tracking-configuration drift through a
secret-free v5 evidence locator. Package publication no longer requires worker
attachment or launch-only configuration. The deployed semantic gate now
resolves retained runs directly, and the complete local repository gate passes
after that API addition. Production promotion still requires an intentional
immutable framework release; the remaining CLI help/navigation audit also
remains open. The veRL two-interpreter path is now a strict, definition-level
candidate with interpreter-specific closures and capsule-owned runtime facts;
it remains intentionally unpublished and unqualified until the clean fork,
dependency-only Python 3.13 lock, real Docker/Bake smoke, and bounded GPU run
exist.

The predeployment release handoff is now explicit. The implementation and
qualification work is substantial and recorded, while publication remains
limited to creating the public dstack fork and private empty ai-infra
repository. No candidate commit or deployment was silently inferred from that
setup. The remaining work has a safe, resumable order: fork commits, immutable
framework pins and scoped commits, ai-infra initial publication, then—under
separate authorization—builds, deployment, and the remaining live gates.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`.
`packages/execution-buildkit` currently wraps Docker Buildx Bake and writes
immutable image build receipts. `packages/execution` owns provider-neutral
execution contracts, the image-owned `JobPackageManifest`, the launch-time
envelope, provider lifecycle, reconciliation, and cleanup. The old directory
bundle remains only as migration input; `packages/execution-dstack` submits the
final OCI image digest and does not use dstack `files` for normal jobs.
`packages/execution-pack` owns deterministic source, environment, dataset, and
manifest assembly. `apps/runtime` verifies and executes a registered job from
inside the image. `apps/cli` owns the normal project commands.

`EnvironmentSource` in
`packages/eval/src/posttrain/eval/requests.py` already records package,
repository, full revision, and optional subdirectory. `EnvironmentBinding`
uses one source. `EvaluationPlan` may contain several bindings. The packing
application must traverse both direct environment seats and environments nested
inside plans.

For example, two environments from one repository are authored independently:

    environments:
      - package: gsm8k-v1
        repository: https://github.com/PrimeIntellect-ai/verifiers
        revision: 284a868d6a9022109b749710672a0460e8a996d4
        subdirectory: environments/gsm8k_v1
        activation:
          kind: verifiers-v1-config
          config: {taskset: {id: gsm8k-v1, split: train}}
      - package: reverse-text-v1
        repository: https://github.com/PrimeIntellect-ai/verifiers
        revision: 284a868d6a9022109b749710672a0460e8a996d4
        subdirectory: environments/reverse_text_v1
        activation:
          kind: verifiers-v1-config
          config: {taskset: {id: reverse-text-v1, dataset_split: train}}

The packer groups these entries by canonical repository and revision, performs
one fetch, verifies both package roots, builds two wheels, and installs the
wheels together so dependency conflicts fail during packing. It retains two
package locks because their package, source subtree, and wheel identities
differ, plus one activation lock per selected binding. If train and test
bindings use the same package, the package lock remains singular while the two
activation locks differ. Public GitHub sources use canonical HTTPS. Private
HTTPS sources use an external Git credential helper during the framework
source-fetch phase; credentials never enter the package manifest, OCI layers,
receipt, or logs.

The infrastructure repository remains responsible for the operational registry
and BuildKit service, registry TLS/authentication/retention, dstack server and
workers, NVIDIA host setup, DNS, and persistent cache/workspace directories.
It does not own a post-training Dockerfile or decide which dependencies an SFT,
GRPO, evaluation, serving, or transform job requires.

## Plan of Work

First introduce provider-neutral source locking beneath
`packages/execution-buildkit`. A `GitSourceRequest` names a canonical repository,
one full commit, and normalized subdirectories. A materializer uses non-shell
Git commands, a content-addressed cache, and exact `HEAD` verification. It
rejects path escape, symlinks, dirty drift, floating revisions, unpinned
submodules, and conflicting requests for the same repository at different
commits. It returns a deterministic lock containing repository, revision,
selected subdirectories, and tree digests without credentials or local paths.

Next create framework-owned universal and job-kind BuildKit targets under
`containers/`. The universal target supplies a pinned base OS, uv, trusted
certificate roots, and common native libraries, but does not force every
backend into one Python/Torch environment. Runtime-profile targets derive from
that universal digest. The Python 3.12 family creates
`/opt/posttrain/venv` and supplies the compatible Torch/CUDA stack;
supervised training adds TRL/PEFT/datasets, TRL online RL adds
vLLM/Verifiers core, evaluation adds Verifiers and inference clients, serving
adds vLLM, and transform adds quantization/compression tools. The veRL family
creates `/opt/posttrain-verl` with Python 3.13 and its independently locked
Torch/vLLM/veRL stack while retaining `/opt/posttrain/venv` as the Python 3.12
control environment. Actual-job assembly installs exact framework/project
control code into Python 3.12 and a minimal framework worker plus every
selected environment wheel required by the portable bridge into Python 3.13.
The environment dependencies are not one lock installed twice: the same
selected wheels are resolved once for Python 3.12 and once for Python 3.13.12,
producing two interpreter-specific hash-locked closures and two resolution
identities.
The runtime launches the backend through the variant's stable interpreter and
working-directory paths. No kind image contains GSM8K or another concrete
environment.

Then replace the current run-bearing `ExecutionJobManifest` with two contracts.
`JobPackageManifest` lives inside the actual job image and records schema,
project/work-package/job-definition identity, resolved-selection digest,
framework and project source digests, universal- and kind-image digests,
environment lock,
dataset lock,
expected output roles, and worker-contract version. It contains no run ID,
provider, credential, mount, or final image digest. `ExecutionRequest` remains
the launch envelope and supplies canonical `RunSpec`, final image digest,
target, environment-variable names, mounts, timeout, priority, and attempt
policy.

Add a `JobPackService` application layer. Static planning derives the required
kind profile, Git sources, dataset selections, and a provisional pack key
without fetching or writing. Packing materializes source checkouts and dataset
snapshots, builds deterministic environment wheels, resolves every selected
environment together, writes locks and the package manifest, and creates a
BuildKit context. Environment package locks contain package name, repository,
full commit, subdirectory, tree digest, and wheel digest. Separate activation
locks contain binding id, package name, activation kind/digest, and optional
importable `module:callable`. Dataset paths in the manifest are relative to the package root rather
than host or container absolute paths. The actual-job target starts from the selected kind-image
digest. Before BuildKit, the packer resolves all environment wheels together
against every interpreter owned by the selected runtime variant. Ordinary
variants emit one Python 3.12 closure; `online-rl-verl-py313` emits separate
Python 3.12 control and Python 3.13.12 backend closures. The image installs
every explicit line from the applicable closure with
`--no-deps`, preventing wheel metadata from adding an unrecorded dependency;
using `--no-deps` before closure expansion remains invalid. It then copies
framework/project code and datasets in
cache-friendly layers, and sets the stable worker entrypoint. BuildKit pushes
zstd-compressed OCI layers with provenance and SBOM attestations. The service
activates every selected environment and writes a protected receipt
containing the pack key and final image digest.

Add `posttrain job pack`. Human and JSON output show the base/kind/job image
chain, source and dataset locks, cache hit or build, final digest, qualification
status, and receipt path without credentials. `posttrain job run` performs the
same static plan, reuses a matching qualified receipt or calls the pack service,
then submits the image. `job plan` remains read-only and may report that exact
tree and wheel digests are pending materialization.

Finally remove `files` from normal local/dstack provider configuration. The
worker reads `JobPackageManifest` from its image and receives launch values
through explicit command arguments or non-secret environment variables.
Qualify cache reuse, source drift rejection, multiple environments from one
commit, dependency conflict rejection, local/dstack equivalence, both dstack
workers, queued/running cancellation, remote evidence, and cleanup.

## Concrete Steps

Run from `/home/hammad/projects/rl`.

During source-lock and image-hierarchy implementation:

    uv run pytest packages/execution-buildkit/tests -q
    uv run ruff check packages/execution-buildkit containers
    uv run pyright packages/execution-buildkit
    docker buildx bake --file <bake-file> --print
    git diff --check

During pack/CLI implementation:

    uv run pytest \
      packages/execution/tests \
      packages/execution-buildkit/tests \
      packages/execution-local/tests \
      packages/execution-dstack/tests \
      apps/runtime/tests \
      apps/cli/tests -q
    uv run ruff check \
      packages/execution packages/execution-buildkit \
      packages/execution-local packages/execution-dstack \
      apps/runtime apps/cli
    uv run pyright \
      packages/execution packages/execution-buildkit \
      packages/execution-local packages/execution-dstack \
      apps/runtime apps/cli
    uv run lint-imports

The intended user flow is:

    posttrain job plan train.yaml --job grpo --provider dstack
    posttrain job pack train.yaml --job grpo
    posttrain job run train.yaml --job grpo --provider dstack
    posttrain run status <run-id>
    posttrain run reconcile <run-id>
    posttrain run cleanup <run-id>
    posttrain run show <run-id>

Before final qualification:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

Planning is accepted when it identifies the exact job kind, base/kind image
requirements, environment repository/commit/subdirectories, dataset
selections, and expected outputs without creating a checkout, build context,
receipt, image, or provider run.

Git packing is accepted when two environments from one repository/commit use
one checkout, produce distinct tree/wheel entries, and yield the same lock
regardless of input ordering. A wrong commit, missing subdirectory, symlink,
dirty cache, conflicting revision, unpinned submodule, or dependency conflict
must fail before BuildKit or dstack submission.

Image packing is accepted when the actual job image can execute with no
project-code or dataset file upload, contains a secret-free package manifest,
starts from the recorded kind-image digest, and passes worker plus environment
activation smoke checks. Repacking identical inputs must reuse the receipt and
digest. Changing code, an environment tree, dataset bytes, resolved selection,
kind-image digest, or worker contract must change the pack key.

The veRL runtime variant has an additional release gate. The CarbonTeq source
checkout must be clean at the recorded full fork commit, that commit must be
reachable from its configured remote and descend from the recorded upstream
base, and the Python 3.13 `uv.lock` digest must match the profile. The lock must
pin veRL to that remote commit, contain no editable/path source, and contain no
concrete environment package. An actual-job smoke must then prove that the
Python 3.12 control worker can launch the Python 3.13 backend and that the
backend can import the exact packaged framework agent loop and selected
environment. Candidate metadata alone is not qualification.

Execution is accepted when local Docker and dstack receive the same actual-job
image digest and logical launch envelope. dstack configuration must contain no
normal `files` mapping. The worker records actual image, worker/GPU context,
source/environment/dataset locks, and run identity through Trackio.
Cancelling an active worker with SIGTERM must exit with code 143 only after the
tracking writer has retained canonical `cancelled`; repeated SIGTERM during
that unwind must not interrupt finalization. SIGKILL remains an unrecoverable
provider hard stop and requires the explicit audited recovery path described
below rather than write-capable reconciliation.
For a detached Trackio-backed run, missing destination, missing or rejected
write credentials, invalid TLS trust, unavailable API, or unhealthy storage
must fail before runtime construction and before model loading. The readiness
probe must not create a provider run or expose a token in its error.

The final GRPO qualification must complete at least fifteen real
backward/optimizer updates with realistic bounded rollouts, retain native
Verifiers traces and required model/summary artifacts, appear through remote
Trackio backed by Doris, remain queryable after cleanup, and leave only
policy-retained worker state.

## Idempotence and Recovery

Git caches and pack contexts are content addressed. A cache is never trusted
solely by path: commit, clean state, selected tree digests, and lock are
revalidated. A partial checkout, wheel, context, image build, or receipt is
written under a temporary path and atomically promoted only after validation.

Build receipts are mode `0600`. A matching qualified receipt makes packing
read-only except for remote image existence verification. If an image was
garbage-collected, the same inputs rebuild it. Failed images are never selected
by `run`.

Cancellation and cleanup continue to target exact persisted provider handles
and run-scoped workspaces. Registry garbage collection may remove unreferenced
actual-job images only after retained execution records and Trackio lineage no
longer require replay.

A failed detached Trackio preflight is safe to retry after correcting URL,
credential injection, CA trust, routing, or server storage health: no operation
has started and no Trackio run has been created. Do not bypass the gate by
disabling TLS verification or changing a qualification project to local
tracking.

The pre-fix stranded run cannot be repaired by read-only reconciliation.
`posttrain run recover-cancelled-tracking RUN_ID` resumes the exact Trackio run
through the writer API and finalizes it as `cancelled`, with a durable recovery
receipt. It runs only when the persisted provider handle is terminal
`cancelled`, the retained tracking run is still `running` or already
`cancelled`, canonical run identity and start time match, and no conflicting
terminal tracking outcome exists. For
`1d3a4c57-68b3-480c-9136-c2188595b33e`, recovery must target only Trackio run
`f6bc4696fd6f45fbba52c777e44ec52c`; no direct Doris mutation is permitted.
Run ordinary reconciliation immediately afterward; cleanup continues to fail
closed until reconciliation observes consistent cancellation.

## Artifacts and Notes

The durable packing artifacts are the Git source lock, environment wheel lock,
dataset lock, `JobPackageManifest`, BuildKit provenance/SBOM, actual-image
receipt, submission record, execution journal, reconciliation, and cleanup
receipt. Secrets, mutable caches, and raw provider environment dumps are not
durable packing artifacts.

The existing execution-provider lifecycle remains documented in
`docs/plan/dstack-execution-provider.md`. This plan supersedes its directory
bundle as the normal distribution path while preserving provider lifecycle,
queue, reconciliation, and cleanup work.

## Interfaces and Dependencies

The generic Git source layer must expose values equivalent to:

    @dataclass(frozen=True, slots=True)
    class GitSourceSpec:
        repository: str
        revision: str
        subdirectories: tuple[str, ...]

    @dataclass(frozen=True, slots=True)
    class GitSourceLock:
        repository: str
        revision: str
        checkout_digest: str
        subdirectories: tuple[GitSubdirectoryLock, ...]

The image package layer must expose values equivalent to:

    @dataclass(frozen=True, slots=True)
    class JobPackageManifest:
        project_id: str
        work_package_id: str
        job_id: str
        job_definition_id: str
        resolved_inputs_digest: str
        framework_source_digest: str
        kind_image: RuntimeImageRef
        runtime_dependency_locks: tuple[RuntimeDependencyLock, ...]
        backend_runtime: BackendRuntimeLock | None
        environments: tuple[EnvironmentPackageLock, ...]
        datasets: tuple[DatasetPackageLock, ...]
        expected_artifact_roles: tuple[str, ...]
        worker_contract_version: str

    class JobPackService:
        def plan(self, prepared: PreparedWorkPackageJob) -> JobPackPlan: ...
        def pack(self, plan: JobPackPlan) -> PackedJobImage: ...

`ExecutionRequest.image` becomes the final `PackedJobImage.image`. Provider
adapters never import environment, dataset, Git, wheel, or BuildKit logic.

Revision note (2026-07-26): created after selecting one universal framework
base, job-kind layers, and actual-job OCI capsules, with framework ownership of
all image contents and multi-environment Git packing.

Revision note (2026-07-27): recorded the first managed-eval GPU
characterizations, made rollout population evidence and Verifiers v1 trace
projection explicit release gates, and kept evaluation incomplete pending a
fresh remote rerun and multi-environment capsule proof.

Revision note (2026-07-27): recorded the evidence-complete managed-eval rerun,
direct Doris row/metric verification, exact cleanup, and the remaining clean
commit/deployment boundary for the corrected Observatory projector.

Revision note (2026-07-27): closed the one-checkout/two-wheel runtime proof,
recorded the external-endpoint failure characterization, introduced the
managed-general evaluation definition, and qualified its reward-bearing
Reverse Text rerun.

Revision note (2026-07-27): made the submitted evidence destination immutable
without persisting credentials, retained warned fallback for v1-v4 receipts,
and validated configuration-drift behavior across the CLI lifecycle.

Revision note (2026-07-27): separated package-only and launch-bearing CLI
plans, removed launch-only flags from `job pack`, and validated that capsule
publication is independent of dstack worker storage.

Revision note (2026-07-27): completed the dormant actual-job side of the veRL
two-environment contract by installing selected environment wheels and a
minimal content-addressed framework source projection into the isolated Python
3.13 runtime, then bound that assembly to the fail-closed release gate. Fork
publication, the dependency-only lock, the kind target, and live GPU
qualification remain open release gates.

Revision note (2026-07-27): corrected the preceding definition-level veRL
claim after review. Environment dependencies now resolve independently for
Python 3.12 and Python 3.13.12; the manifest binds both resolutions plus the
capsule fork, base lock, worktree, interpreter, and worker projection; inherited
Python import state is removed; and a ready profile must execute the dedicated
Docker/Bake smoke. No image installation or runtime qualification is claimed
while the profile remains blocked.

Revision note (2026-07-27): completed the source-level Trackio/Doris P0
correction after release audit, revalidated the complete framework and
Observatory frontend gates, and removed superseded framework/veRL image
assembly from ai-infra. Real Doris migration, clean fork publication, immutable
consumer pins, and live GPU/provider qualification remain open.

Revision note (2026-07-27): added the consolidated current-status and release
handoff after reconciling local repositories with GitHub. The plan and evidence
log had been updated throughout implementation, but the new section makes the
completed architecture, live qualifications, unpublished candidates, two
created repositories, open release gates, and exact predeployment resume order
readable from one place.
