# Deliver a reproducible local and dstack post-training artifact lifecycle

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with
`docs/templates/PLAN.md`. It implements the product meanings frozen in
`docs/post-training/01-workflow.md` through
`docs/post-training/06-observation-and-lineage.md`; it does not replace them.

## Purpose / Big Picture

After this work, a developer can run the same small post-training experiment on
the local runner or either dstack GPU worker without manually copying source,
datasets, base-model weights, or result models. A bounded dataset and a
Verifiers environment travel as versioned Python code. A base model is selected
by a full Hugging Face commit and materialized through a bounded worker cache.
Training code records observations directly in Trackio and publishes only
selected result models or reusable capability checkpoints as versioned Trackio
artifacts. A later job can consume the exact published version on another
machine and prove producer-to-consumer lineage.

The first visible end-to-end proof is deliberately small. A synthetic job
publishes a multi-file model artifact on one runner, another runner downloads
that exact version and validates its digest, and Trackio shows both runs and the
producer/consumer edge. The next proof performs a limited Qwen 3.5 0.8B LoRA
run, reloads the resulting adapter from Trackio, and evaluates it. The same path
then qualifies a limited 2B LoRA run. Full-weight training is not admitted until
the transport, disk, and cleanup gates have passed at the corresponding
multi-gigabyte size.

The existing dstack control plane, two attached GPU workers, private OCI
registry, Trackio deployment, DNS, and credentials are prerequisites, not
deliverables of this plan. This plan reads and qualifies that infrastructure. It
does not reprovision it.

## Progress

- [x] (2026-07-26) Read the canonical post-training product baseline, current
  framework artifact and work-package seams, the CarbonTeq Trackio fork, and
  the deployed dstack/Trackio infrastructure contracts.
- [x] (2026-07-26) Confirm that the requested behavior aligns with the frozen
  product baseline; no baseline amendment is required before implementation.
- [x] (2026-07-26) Record the important current gaps: large Trackio uploads are
  buffered in server memory, a work-package result does not expose the durable
  artifact version, cleanup lacks a remote reference-audited acceptance test,
  and the deployed Trackio build can drift from the framework pin.
- [ ] (2026-07-26 07:46Z) Milestone 0 partially complete: implemented
  provider-neutral storage admission and a redacted live preflight; focused
  tests pass and existing Trackio write/read, remote Observatory, and both
  worker preflights pass. Trackio and Observatory now run the aligned immutable
  revisions. Remaining: clear the measured local storage reserve shortfall.
- [x] (2026-07-26 07:46Z) Milestone 1 complete: implemented and published
  Trackio `0.31.5.post2` at
  `e2784c1536b20832f3937d7589c10bce76df4b43`, wired resumable transport
  behind `log_artifact`, passed focused, separate-process, retry,
  process-restart, full non-GPU, and 512 MiB memory-bounded gates, deployed the
  exact image, and verified 64 MiB and 512 MiB upload/readback against the
  remote service.
- [x] (2026-07-26 08:58Z) Milestone 2 complete: incomplete-session expiry has
  a dry-run-by-default administrator CLI and tests prove completed blobs
  survive. The deployed dry-run found zero abandoned sessions. The exact
  disposable `artifact-transport-qualification` project was then deleted and a
  canonical 1 MiB proof recreated, reducing the complete remote Trackio store
  from 650 MiB to 74 MiB while retaining all Ambient model artifacts.
- [x] (2026-07-26 07:18Z) Milestone 3 complete: provider-neutral finalization
  now resolves exact versions and digests before workspace cleanup, exposes
  them on `WorkPackageJobResult`, and a real Trackio producer-to-consumer test
  proves exact-version materialization and lineage. Trackio and W&B adapter
  tests, work/jobs tests, Ruff, Pyright, and all eight import contracts pass.
- [ ] (2026-07-26 08:50Z) Milestone 4 partially complete: generated the standalone
  `/home/hammad/projects/ambient-agent` project from the public GRPO template
  and initialized its independent Git history. A package-owned 12-row pure SFT
  dataset now builds as a clean wheel. The separate
  `/home/hammad/projects/ambient-agent-environments` Verifiers package builds,
  contains eight executable tasks, and passes its three focused tests.
  Remaining: publish immutable CarbonTeq revisions and the environment through
  the actual Prime team namespace.
- [x] (2026-07-26 08:50Z) Milestone 5 complete for the bounded qualification
  tier: both exact Qwen3.5 revisions materialized through persistent,
  worker-owned Hugging Face caches. The 0.8B and 2B one-backward-pass
  preflights succeeded on the PRO worker at 4,381,700,096 and 10,329,959,936
  peak allocated bytes respectively. dstack cache mounts are expressed as
  target placement, not job-bundle content.
- [x] (2026-07-26 08:50Z) Milestone 6 complete: added a local Docker provider
  implementing the same plan, submit, status, bounded-log, cancel, collect,
  cleanup, immutable-image, and verified-bundle contracts as dstack. A live
  CUDA smoke succeeded on the RTX 4090 with PyTorch 2.11.0+cu130 and the
  terminal test container was removed after its receipt was journaled.
- [x] (2026-07-26 08:50Z) Milestone 7 complete for the bounded lifecycle:
  added neutral
  execution contracts, an append-only mode-0600 receipt journal, a direct
  dstack-Python-SDK adapter isolated across the Pydantic 1/2 process boundary,
  bounded logs, cancellation, terminal collection, and a singular queue with
  an explicit Trackio-evidence barrier. Live vector-add jobs succeeded on both
  `pop-os.lan` and `carbonteq-ai-workstation.lan`; remote cancellation reached
  `terminated`. The immutable runtime image
  `registry.lan/carbonteq/posttrain-runtime@sha256:431f52cd4eac584ccf73c0329c832679e286945d3f880fa892515a340cf00274`
  verified deterministic bundles on both machines. The PRO-produced 0.8B LoRA
  adapter was then materialized as exact Trackio input `reinforce-adapter:v0`,
  loaded over the pinned base revision, and exercised on the RTX 4090.
  Observatory shows the producer output and consumer input edge. Remaining
  resilience work is provider retry/reconnect qualification. Atomic mode-0600
  queue restoration, plan-identity drift rejection, the terminal-evidence
  barrier, and live cancellation after two controller reconstructions now
  pass.
- [x] (2026-07-26 08:50Z) Milestone 8 complete for limited LoRA gradient-path
  qualification: both 0.8B and 2B completed exactly 15 sampled REINFORCE
  optimizer updates and published separate adapter and summary artifacts.
  These runs prove bounded policy-gradient execution and artifact finalization,
  not GRPO quality; both declare `grpo_qualified: false`, and Observatory
  correctly marks their GRPO evidence incomplete. The rebuilt runtime
  containing separate file/tree content-digest metadata is
  `registry.lan/carbonteq/posttrain-runtime@sha256:047334f8246533071a77b12542f9e7df4a8b413bf4452918f71a625c3e234b9f`;
  an exact-digest dstack CUDA smoke passed.
- [ ] (2026-07-26 07:46Z) Milestone 9 partially complete: published the
  Trackio fork change, updated framework and standalone-project pins, updated
  the fork ledgers, and deployed and qualified the exact build through the
  existing infrastructure scripts. Observatory now requires the complete
  unique healthy source set. Remote cleanup and replacement canonical
  transport evidence, limited workload gates, durable queue recovery, and the
  full framework release ladder now pass. Remaining: authenticated
  Prime/CarbonTeq environment publication and the separately recorded local
  storage reserve decision.

## Surprises & Discoveries

- Observation: the framework already has the correct provider-neutral input
  shape. `packages/work/src/posttrain/work/runner.py` deliberately converts a
  `TrackioArtifactRef` into a `StoredArtifactRef(provider="trackio")` before
  calling the tracking backend.
  Evidence: `_artifact_inputs` performs this normalization, while
  `packages/jobs/src/posttrain/jobs/definitions.py` can materialize both
  reference forms. The implementation must not widen `ArtifactInput` merely to
  expose Trackio in a shared contract.

- Observation: current training operations upload artifacts while their
  temporary workspace still exists, but `WorkPackageJobResult` returns only the
  operation value and run ID. A returned `LocalArtifactRef` can therefore point
  to a path that has already been deleted.
  Evidence: `packages/work/src/posttrain/work/contracts.py` defines
  `WorkPackageJobResult` without produced artifacts, and training backends
  return `TrainingResult` values containing local paths.

- Observation: the pinned Trackio fork already provides content-addressed
  artifact versions, aliases, manifests, download, and producer/consumer
  lineage. The missing large-model property is transport safety rather than a
  new registry.
  Evidence: `../trackio/trackio/run.py` commits artifact versions and
  `../trackio/trackio/asgi_app.py` serves artifact blobs, but the current upload
  handler calls `await upload.read()`, buffering an entire uploaded file.

- Observation: Trackio project deletion removes the project database and the
  project artifact directory, but there is no current remote, measured
  acceptance test proving that interrupted upload staging and retained
  content-addressed blobs obey the intended policy.
  Evidence: `../trackio/trackio/__init__.py::delete_project` removes
  `utils.project_artifacts_dir(project)` recursively. The plan therefore starts
  with tests around existing deletion and adds only the missing cleanup
  behavior.

- Observation: the current infrastructure is usable but not identical to the
  framework dependency state. The framework resolves Trackio commit
  `c5072198b3b1556d31ed96ffc246a03f65418ab8`, while the deployed image label
  observed during planning was based on `c47bcc0e`.
  Evidence: `packages/tracking-trackio/pyproject.toml` and `uv.lock` own the
  consumer pin; `/home/hammad/projects/ai-infra` owns the deployed image. The
  exact live state must be refreshed in Milestone 0 rather than assumed.

- Observation: the root filesystem was approximately 88 percent occupied
  during planning.
  Evidence: about 114 GiB was free. Large artifact tests and 2B runs require a
  preflight reservation and cleanup budget rather than optimistic execution.

- Observation: pruning 24.5 GiB of uv cache entries reclaimed only about 1 GiB
  of filesystem blocks because most pruned archive entries were hard-linked
  into environments.
  Evidence: `uv cache prune` reported 24.5 GiB removed, while the follow-up
  preflight still measured a 26,307,495,731-byte admission shortfall. Do not
  equate cache logical size with reclaimable filesystem blocks.

- Observation: the deployed infrastructure is healthy independently of the
  Trackio revision drift.
  Evidence: the live Trackio write/read qualification persisted two records;
  Observatory passed authenticated ingress, healthy Trackio source, and image
  digest checks; both dstack accounts passed SSH, passwordless-sudo, registry,
  Docker, and GPU checks. The workers reported an RTX 4090 with 24,564 MiB and
  an RTX PRO 6000 Blackwell Workstation Edition with 97,887 MiB.

- Observation: a same-process Trackio server/client test can falsely avoid
  transport because both sides share the same content-addressed directory.
  Evidence: the qualification was changed to launch the server in a distinct
  process with a distinct `TRACKIO_DIR`. This is now the required E2E shape for
  artifact transport tests.

- Observation: the first 512 MiB resumable qualification completed 64 chunks
  while server RSS rose by 15,486,976 bytes.
  Evidence: the separate-process gate reported baseline RSS 79,048,704 bytes
  and peak RSS 94,535,680 bytes, well below the 128 MiB bound and far below the
  artifact size.

- Observation: dstack 0.20.x cannot share the framework's Python process.
  Evidence: its distribution requires Pydantic 1 while framework contracts
  require Pydantic 2. The adapter now invokes the public dstack Python SDK
  through a small JSON bridge running in the existing ai-infra virtual
  environment; it does not use the dstack CLI.

- Observation: fixed SSH-fleet capacity did not queue a second constrained
  task under contention.
  Evidence: while the remote 90 GiB GPU slot was occupied, the second task
  ended `failed_to_start_due_to_no_capacity` with `no offers`. The framework
  singular queue must therefore hold work before submission and release the
  next item only after provider terminal state and Trackio evidence reconcile.

- Observation: dstack's `terminating` state is a nonterminal teardown phase
  after both successful execution and cancellation.
  Evidence: the first live SDK job transitioned through `terminating` to
  `done` and logged `Test PASSED`; mapping `terminating` to cancellation caused
  premature collection and was corrected.

- Observation: a healthy Observatory Trackio source does not prove that any
  run is readable as canonical framework evidence.
  Evidence: the first 64 MiB and 512 MiB transport probes used ad-hoc Trackio
  configuration, so the provider connection was healthy but Observatory
  correctly skipped both runs. The qualification producer now emits schema-v4
  framework identity and terminal fields; a 1 MiB proof appeared remotely as
  a succeeded `model.transform` run with upload and download throughput,
  duration, and byte metrics. Historical transport records were not rewritten.

- Observation: Trackio's artifact `digest` and the framework's local
  directory-tree SHA-256 are different digest domains.
  Evidence: `reinforce-adapter:v0` has provider digest
  `55208642cb297bea2cfd1977beeef58083dce4316a0d13944c3e45d3df98b64d`,
  while its downloaded directory tree hashes to
  `3ccb34724dc4a840fbaa6c4f493fff4d74321eaa57868d0d0f86cf71572d10c5`.
  The adapter now preserves Trackio's digest as immutable provider identity and
  records `posttrain_content_digest` separately on new publications. It never
  compares unlike digest domains.

- Observation: dstack's resource interval is a better portable selector than
  a hard-coded local hostname for this two-worker fleet.
  Evidence: requesting 20–30 GB of GPU memory excluded the 96 GB PRO card and
  selected `pop-os.lan`; requesting at least 90 GB selected
  `carbonteq-ai-workstation.lan`. The adapter now validates an optional
  `gpu_memory_max_gb` placement constraint.

- Observation: an `eval.model` consumer is readable in Observatory but has no
  specialized job view.
  Evidence: the consumer run
  `851a883a-8be4-4e39-8426-116f4654f2b6` resolved to the generic view, while
  still exposing its succeeded status, four metrics, and exact input artifact
  edge. Add a specialized model-evaluation schema only when the product has a
  stable evaluation completeness contract.

- Observation: sorting content-digest receipt filenames does not select the
  newest runtime image.
  Evidence: the new source receipt began `0144...` and sorted before the older
  `a7a5...` receipt, causing the first post-rebuild smoke to reuse the older
  image. `latest_runtime_image` now selects by nanosecond modification time,
  validates the immutable reference, and has a regression test where the newer
  receipt sorts lexically first. A repeated smoke used and passed the intended
  `sha256:6994aef6...` image.

- Observation: infrastructure qualification scripts must declare their own
  runtime dependencies.
  Evidence: after remote cleanup, `uv run
  scripts/qualify_trackio_artifact.py` initially failed because `httpx` was not
  part of the ai-infra project environment. The script now has PEP 723 metadata
  pinning the Trackio fork, HTTP client, Python, and YAML dependency and
  recreated the canonical proof from a clean script environment.

- Observation: dstack capacity and framework research admission are separate
  concerns.
  Evidence: live queue qualification submitted `complete-first`, reconstructed
  the controller from an atomic snapshot, and held `cancel-second` until the
  first provider terminal state was explicitly evidence-acknowledged. After a
  second reconstruction, only then was the second job submitted and cancelled.
  The provider IDs were `pt-b55d50d9308e078b4375ed70` and
  `pt-d9c1bef08258433206cde2cb`; dstack reported succeeded and cancelled
  respectively. Snapshot restore rejects any changed run, bundle, image, job
  definition, or idempotency identity.

- Observation: the legacy Docker builder amplified small source changes into
  large unique local image layers.
  Evidence: three runtime builds left approximately 90.1 GB of images on the
  PRO worker, including about 53 GB of builder stages and 8.9 GB per
  superseded runtime virtualenv. Exact removal of the two superseded local
  tags and post-training builder IDs reduced image storage to approximately
  20.0 GB with zero containers. The current runtime and 6.1 GB exact model
  cache remain. Future builder stages carry a source label; the build script
  prunes that exact label after a successful push and retains only the newest
  local runtime snapshot. Old runtime digests remain recoverable from the
  internal registry.

## Decision Log

- Decision: use Trackio as the experiment evidence service and internal result
  artifact registry; do not introduce a second model registry for the first
  implementation.
  Rationale: Trackio already owns exact artifact versions, aliases, manifests,
  lineage, and the run UI. Adding another registry would duplicate identity and
  authorization.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: package bounded SFT/preference data as project code and executable
  task data with its Verifiers environment; do not claim that all SFT data must
  originate from retained Verifiers traces.
  Rationale: pure curated SFT datasets are valid inputs. Trace-derived SFT is a
  specialized data-production path, not the general contract.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: keep foundation-model weights outside Git, job bundles, and runtime
  images. Represent them as `HubModelRef` values with full commits and
  materialize them through bounded worker caches.
  Rationale: this preserves reproducibility without repeatedly transferring
  multi-gigabyte weights with code.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: publish final models, adapters, and explicitly retained reusable
  capability checkpoints through Trackio; expire ordinary recovery checkpoints.
  Rationale: generalized-capability gains need reusable immutable identities,
  while every trainer save does not deserve durable retention.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: observations and experiment outcomes are emitted by job code to
  Trackio. There is no sidecar log collector and no terminal artifact manifest.
  Rationale: the training/evaluation code knows the semantic meaning of its
  metrics, traces, selections, and results. A second terminal document would
  duplicate Trackio and create reconciliation ambiguity.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: retain only a compact provider execution receipt outside Trackio
  for submissions that may fail before a Trackio run is created.
  Rationale: scheduler IDs, attempts, state transitions, and bounded log cursors
  are required to reconcile dstack with the framework. The receipt points to a
  Trackio run when one exists but never lists result files or becomes an
  experiment evidence store.
  Date/Author: 2026-07-26 / Codex.

- Decision: use the dstack Python SDK directly and the existing dstack control
  plane. Do not add DBOS, Hatchet, Prefect, Temporal, Airflow, Kestra, Dagster,
  Celery, or another workflow orchestrator.
  Rationale: dstack supplies placement, task retry, status, logs, and
  cancellation. A small framework-owned serial admission queue is still
  required because a fixed SSH fleet can fail a contending submission with no
  offers instead of holding it. This remains a thin execution adapter and
  queue, not a second workflow scheduler.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: run one research experiment at a time, while allowing each
  qualification scenario to choose either available GPU worker.
  Rationale: serial research reduces contention and disk amplification during
  the first artifact-lifecycle qualification. Parallel research can be added
  after measured resource and cleanup behavior exists.
  Date/Author: 2026-07-26 / Codex and user.

- Decision: preserve `ArtifactInput.reference: StoredArtifactRef` and normalize
  backend-specific references at the composition boundary.
  Rationale: shared work, train, eval, and serve contracts must remain
  independent of Trackio and retain equivalent W&B behavior.
  Date/Author: 2026-07-26 / Codex.

- Decision: implement large-file transport in the CarbonTeq Trackio fork before
  any full-model result is trusted.
  Rationale: the current whole-file server buffer is unsafe for model-sized
  artifacts. Direct worker access to Trackio's backing storage would leak
  storage concerns and credentials into every job.
  Date/Author: 2026-07-26 / Codex.

## Outcomes & Retrospective

Current outcome: the bounded vertical slice is implemented and deployed.
Trackio `0.31.5.post2` streams and resumes artifact blobs through 8 MiB chunks;
a separate-process 512 MiB round trip stayed approximately 15 MiB above
baseline RSS, and the exact fork/image is deployed. Observatory is remotely
authenticated and reads four exact healthy sources, canonical runs, metrics,
and lineage. The same immutable runtime and verified bundle ran through dstack
on both workers and through the local Docker provider on the RTX 4090. Exact
0.8B and 2B preflights and 15-update LoRA REINFORCE runs succeeded, with model
artifacts retained in Trackio; a cross-worker consumer proved exact-version
reload. The measured remote cleanup gate and fresh runtime rebuild now pass.
Queue/restart reconciliation and the full repository ladder now pass: 389 tests,
16 intentional skips, Ruff, Pyright, all eight import contracts, and diff
checks are green. The remaining external release gate is Prime/CarbonTeq
environment publication, blocked on user authentication and the
actual writable team slug; the official CLI reports no API key configured.
The local filesystem reserve shortfall remains recorded rather than hidden.

Update this section at each milestone with the behavior delivered, the exact
evidence location, any remaining release gate, and what was deleted or retained.

## Context and Orientation

The primary repository is `/home/hammad/projects/rl`, a Python 3.12 `uv`
workspace. The sibling `/home/hammad/projects/trackio` checkout is the
CarbonTeq Trackio fork. The sibling `/home/hammad/projects/ai-infra` repository
owns the already-deployed dstack server, GPU fleet, OCI registry, Trackio
service, DNS, secrets, and backups. Implementation must not mix uncommitted work
across these repositories.

A **work package** is a reproducible research unit containing selected jobs and
their shared configuration. A **job** is one typed action, such as SFT,
evaluation, or model serving. A **run** is one actual attempt at a job. A
**model variant** is an exact loadable model selection. These meanings come from
`docs/post-training/02-primitives.md` and
`docs/post-training/03-work-and-evidence.md`.

An **artifact version** is an immutable Trackio record, such as
`ambient-agent-08b-sft:v3`, whose manifest names every file and SHA-256 digest.
An **alias**, such as `candidate` or `qualified`, is a mutable navigation aid.
Planning may resolve an alias, but the run must snapshot and consume the
immutable version.

A **recovery checkpoint** is trainer state retained briefly so an interrupted
run can continue. It is workspace state until explicitly nominated. A
**capability checkpoint** is a model result deliberately retained because an
evaluation demonstrated a reusable generalized capability gain. Capability
checkpoints are durable versioned artifacts; ordinary recovery checkpoints
expire.

A **producer/consumer edge** is Trackio lineage showing that one run published
an artifact version and another run used that exact version. A **scheduler
receipt** is a compact framework record containing the provider task ID,
attempt, timestamps, target snapshot, lifecycle state, bounded diagnostic
reason, and optional Trackio run ID. It is not a model manifest, observation
store, or result registry.

The relevant framework files are:

- `packages/common/src/posttrain/common/artifacts.py`, which owns
  `HubModelRef`, `StoredArtifactRef`, `TrackioArtifactRef`, and
  `ProducedArtifact`;
- `packages/common/src/posttrain/common/selections.py`, which owns
  `ModelVariant` and the reproducible selection values;
- `packages/tracking/src/posttrain/tracking/contracts.py`, which defines the
  provider-neutral tracking writer and reader contracts;
- `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`, which
  maps those contracts to Trackio;
- `packages/work/src/posttrain/work/runner.py` and
  `packages/work/src/posttrain/work/execution.py`, which create tracked runs and
  execute work-package jobs;
- `packages/work/src/posttrain/work/contracts.py`, which defines
  `WorkPackageJobResult`;
- `packages/jobs/src/posttrain/jobs/definitions.py` and
  `packages/jobs/src/posttrain/jobs/runtime.py`, which materialize model inputs
  and invoke typed operations; and
- `packages/train/src/posttrain/train`, whose backend operations produce model
  and checkpoint outputs.

The current Trackio upload path begins in `../trackio/trackio/run.py`, queues
blobs through `../trackio/trackio/pending_uploads.py`, and accepts them in
`../trackio/trackio/asgi_app.py` and `../trackio/trackio/server.py`. Trackio's
SQLite artifact tables and content-addressed storage live in
`../trackio/trackio/sqlite_storage.py` and `../trackio/trackio/artifact.py`.

The initial project repository is `/home/hammad/projects/ambient-agent`. If it
does not yet contain Git history, initialize it only when Milestone 4 begins.
The Verifiers environment is a separately publishable package. Prefer a
dedicated `/home/hammad/projects/ambient-agent-environments` repository so its
version, CarbonTeq publication, and compatibility can evolve independently of
one experiment project. Never store credentials in either repository.

The existing catalog already selects:

- `Qwen/Qwen3.5-0.8B` at full revision
  `2fc06364715b967f1860aea9cf38778875588b17`; and
- `Qwen/Qwen3.5-2B` at full revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.

Those revisions are the initial base-model inputs. A different model must be a
new `ModelVariant`, not an in-place edit to a completed run.

## Scope and Product-Baseline Assessment

This plan changes implementation and integration behavior, not the frozen
product meaning. The baseline already says that exact selections identify
datasets, environments, models, and execution targets; training produces
materialized descendants; only nominated materializations become durable
artifacts; Trackio is the default evidence backend; and reusable packages must
remain backend-neutral.

The plan must pause for a narrow baseline amendment if implementation would:

- make a mutable alias a run input;
- redefine all SFT data as Verifiers traces;
- make recovery checkpoints durable by default;
- expose Trackio types from common train, eval, serve, work, or execution
  contracts;
- treat dstack completion as experiment success without Trackio finalization; or
- require bucket credentials in workload code.

Correct `ops/dstack-trackio/object-storage.md` during Milestone 3 so it no
longer describes normalized `StoredArtifactRef(provider="trackio")` as a
contract defect.

## Plan of Work

### Milestone 0: Freeze live evidence and enforce admission

Create `scripts/qualification/capture_artifact_preflight.py` in the framework
repository. It must make read-only checks and write a redacted JSON record under
`artifacts/qualification/artifact-lifecycle/<timestamp>/preflight.json`.
Record the framework revision and dirty-file names, Trackio consumer pin,
Trackio server-reported version or health identity, dstack client version,
fleet and worker identities, GPU model and VRAM, root and cache free space,
runtime image digest, exact base-model selections, and whether required
environment variables are present. Record only boolean secret presence, never a
secret value.

Add a small admission module under
`packages/work/src/posttrain/work/admission.py`. It should accept declared
download bytes, peak workspace bytes, retained output bytes, and a safety
margin. It must reject a job before GPU allocation when the target cannot
satisfy the sum. For initial qualification, use a 15 percent filesystem safety
margin and preserve at least 30 GiB after a limited run. Make both values named
policy inputs so evidence can explain them.

The deployed Trackio revision must equal the revision qualified by the
framework integration tests before Milestone 6. A mismatch is not an automatic
infrastructure rewrite: report it as `blocked_version_drift`, qualify the
intended Trackio build locally, and deploy it through the existing ai-infra
script in Milestone 9.

The end-to-end gate is:

1. `trackio.lan` answers its health endpoint.
2. write authorization can create and finish one uniquely named disposable
   smoke run without exposing the token.
3. `dstack fleet list` reports both configured workers and no unexplained
   active research task.
4. each worker reports its GPU, driver, free disk, and cache path through a
   read-only task.
5. admission accepts a 64 MiB synthetic job and rejects a deliberately
   impossible request with a message showing required and available bytes.
6. deleting the disposable Trackio project leaves the retained research
   projects untouched.

Stop here if a worker identity is ambiguous, a route reaches the wrong machine,
Trackio cannot persist a run, or the reservation cannot fit. Do not delete
existing research artifacts to force admission.

### Milestone 1: Stream and resume model-sized Trackio artifacts

Work in `/home/hammad/projects/trackio` from a new branch based on the immutable
commit currently pinned by the framework, initially
`c5072198b3b1556d31ed96ffc246a03f65418ab8`. Do not build the change on the
older local `codex/trackio-read-api` tip.

Add a capability endpoint and resumable artifact upload protocol to Trackio.
The protocol is private to Trackio; framework jobs continue to call
`log_artifact`. It must provide these operations:

1. Query content-addressed blob presence by project and full SHA-256 digest.
2. Create or resume an upload session using project, digest, total size, and a
   server-selected bounded chunk size.
3. Upload one raw chunk by opaque session ID and zero-based index. Stream the
   request body into a temporary file while calculating its SHA-256. Reject a
   wrong range, wrong chunk digest, oversized chunk, or unauthorized request.
4. Return acknowledged and missing chunk indices so a client restart does not
   resend successful parts.
5. Complete the session only after streaming all chunks in order into the
   content-addressed destination and verifying the declared total size and
   whole-file digest.
6. Commit the artifact version only after every manifest blob is durable.
7. Abort or expire a session without creating an artifact version.

Use opaque random upload IDs; never place filesystem paths in the API. Store
session metadata and acknowledged part metadata durably beside Trackio's
project artifact state so a Trackio process restart can resume. Write each
chunk to a same-filesystem temporary path, `fsync` it, verify it, and use atomic
rename when promoting the completed blob. Concurrent sessions for an already
completed digest must converge on the same blob without corruption.

The client should discover the server capability. Preserve the old small-file
endpoint for compatibility during one release window, with a named maximum
size. If a model-sized artifact reaches an old server, fail before upload with
a message saying that resumable artifact transport is required. Do not
silently fall back to the whole-file buffer.

Add focused tests under `../trackio/tests/unit/` for:

- exact behavior below, at, and above the compatibility threshold;
- empty files and a multi-file artifact;
- duplicate and out-of-order chunks;
- wrong chunk and whole-file digests;
- lost client response followed by idempotent retry;
- server restart followed by resume;
- two clients racing to publish the same digest;
- authorization on every session operation;
- completion refusal while a chunk is missing;
- no artifact version when transport fails; and
- bounded server resident memory.

For the memory test, upload deterministic generated bytes at least eight times
larger than the negotiated chunk size. Sample server RSS before and during the
request. Accept a fixed test-process allowance plus at most three chunk buffers;
never assert that RSS must be exactly constant.

The standalone end-to-end gate starts a real Trackio ASGI server in a temporary
directory, uploads 1 MiB, 64 MiB, and 512 MiB deterministic artifacts, restarts
the server during the 512 MiB upload, resumes it, downloads the exact version,
and compares every SHA-256. Only the 512 MiB case may be skipped in ordinary CI;
it is mandatory before deployment.

### Milestone 2: Make cleanup measurable and safe

First characterize the existing deletion behavior before adding code. Create a
disposable Trackio project, upload one de-duplicated blob through two artifact
versions, record database, staging, and artifact-directory byte counts, delete
the project, and prove those project-owned bytes are gone. Create another
project containing a retained artifact and prove it is unchanged.

Add abandoned-upload expiry if Milestone 1 does not already supply it. Expiry
must use a configurable age, ignore active sessions, report session count and
reclaimable bytes, default to dry-run for an administrator command, and never
delete committed content-addressed blobs. The command should support a
project-scoped real deletion only after an explicit confirmation flag.

Do not build artifact-version deletion or global garbage collection unless the
characterization proves project deletion leaks committed blobs. If committed
blobs are shared across projects in the qualified implementation, add a
reference audit that scans every artifact manifest before deletion. Dry-run is
mandatory and a blob with any live manifest reference must survive.

Add framework retention tests distinguishing:

- temporary workspace files, removed after finalization;
- retry staging, retained for a bounded retry window after failure;
- recovery checkpoints, retained according to the job's checkpoint policy;
- nominated capability checkpoints, committed as artifacts; and
- final model artifacts, retained until explicit project-level policy changes.

The end-to-end gate interrupts a 512 MiB upload, verifies no artifact version is
visible, expires the abandoned session, and observes reclaimed staging bytes.
It then completes a new upload, deletes its disposable project, and observes
reclaimed committed bytes while a separate retained project and its exact
download remain valid.

### Milestone 3: Finalize framework results into exact durable references

Keep `ArtifactInput.reference` provider-neutral. Add a durable result value to
`packages/common/src/posttrain/common/artifacts.py`, tentatively:

    @dataclass(frozen=True)
    class PublishedArtifact:
        logical_name: str
        kind: str
        reference: StoredArtifactRef
        size_bytes: int | None

Reuse the existing asynchronous
`RunDataSource.artifacts(run_id) -> ArtifactSet` reader. Its output links already
carry direction, logical name, kind, provider, namespace, exact version, and
optional digest. Add a pure conversion helper that selects output links and
normalizes each one into `PublishedArtifact`. The Trackio adapter already reads
run artifacts; ensure it returns
`StoredArtifactRef(provider="trackio", namespace=project, name=name,
version="vN", digest=digest)` through that conversion. W&B and fake adapters
must return logically equivalent values without importing Trackio.

Extend `WorkPackageJobResult` with
`published_artifacts: tuple[PublishedArtifact, ...]`. Define required output
roles on the job definition or execution request, for example `model`,
`checkpoint`, and `trace`. After the operation has synchronously called the
tracking writer, perform a read-after-write query by run ID while the temporary
workspace still exists. Require exactly one committed artifact for each
required singular role, verify its immutable version and manifest digest, and
only then return success and remove the workspace.

If the tracking backend cannot resolve a required output, mark the run
`partial` or `failed` according to the canonical outcome taxonomy, retain
bounded retry staging, and return a diagnostic that names the missing role. Do
not return a successful `TrainingResult` containing a dead local path.

Add a helper that converts a resolved model artifact into a descendant
`ModelVariant` using its durable `StoredArtifactRef`. The next work package
should consume that model selection without manually typing a Trackio version.
Aliases may be updated after qualification, but the descendant always stores
the exact version and digest.

Emit experiment outcomes from the owning code into Trackio using canonical
summary fields: objective/reward metrics, evaluation result, capability
checkpoint decision, cleanup outcome, source revision, backend revision,
execution target revision, and produced artifact versions. The execution
provider may retain a compact receipt for pre-run reconciliation, but it must
not create a terminal manifest or duplicate semantic observations.

Add focused fake-provider and real-Trackio integration tests proving:

- output resolution happens before workspace deletion;
- a producer returns an exact durable reference;
- a consumer materializes that reference and records input lineage;
- an alias move after planning does not change the consumer's version;
- two artifacts with the same role fail unambiguous finalization;
- upload failure cannot produce a successful work-package result; and
- the W&B adapter can satisfy the same logical contract.

Update `ops/dstack-trackio/object-storage.md` to describe normalization into
`StoredArtifactRef(provider="trackio")` as intentional. Update
`docs/architecture/proposed-dstack-execution-provider.md` only where necessary
to reflect the finalizer and compact-receipt interfaces. Do not change the
frozen product baseline unless the implementation crosses a boundary listed in
the baseline assessment.

### Milestone 4: Publish bounded datasets and the Verifiers environment as code

Initialize `/home/hammad/projects/ambient-agent` as a normal Git repository if
it is still empty. Create a Python 3.12 project with bounded SFT and preference
rows under package resources. Each dataset revision must include a manifest
with schema version, row count, split counts, source/provenance note, license,
creation code revision, and SHA-256 for every data file. Loading uses
`importlib.resources`, not the developer checkout path.

Create `/home/hammad/projects/ambient-agent-environments` as a separately
versioned Python project for the Verifiers environment. Package executable task
fixtures with verifier/rubric code. The environment must expose the standard
Verifiers entry point used by the framework, produce native Verifiers traces,
and be publishable under the CarbonTeq environment namespace. Do not require an
external dataset download for the initial bounded task.

Pure SFT data remains a valid independent `DatasetSelection`. Trace-derived SFT
may be generated later through an explicit transformation with parent trace
and transformation revisions. Evaluation-only rows must have disjoint stable
IDs and a test that rejects any training overlap.

Use package data declarations so wheels contain the resources. Build each wheel
in a clean temporary directory, install it into a clean Python 3.12
environment, disable network access for the dataset load test, and verify the
manifest hashes and row counts. For the environment, run at least one CPU-only
episode and verify a native trace plus deterministic verifier score for a fixed
response.

The end-to-end gate installs only the built wheels and locked dependencies into
a clean environment, loads every declared split, runs one environment episode,
and produces the same manifest digest as the source checkout. Publish the
environment only after this gate, then pin its immutable Git commit and package
version in the framework project overlay. Record the repository and commit for
each packaged dataset in the Trackio run.

### Milestone 5: Materialize immutable base models through bounded caches

Add a provider-neutral model materializer in the appropriate data/common
integration package; do not put Hugging Face imports in `posttrain.common`.
Its input is a `HubModelRef` containing the repository ID and full 40-character
revision. Its cache key includes repository ID, revision, file allow patterns,
and materializer schema version.

Before GPU admission, resolve and download the snapshot into a worker-local
cache under an infrastructure-configured bounded path. Use a per-key lock so
two attempts cannot corrupt one cache entry. Write a completed-entry manifest
only after every required file exists, with file sizes and digests. An
incomplete entry is ignored or resumed safely. Never log a Hugging Face token.

Add configurable high-water and low-water limits. Eviction may remove only
unlocked, unpinned least-recently-used base snapshots. It must never evict the
active job's snapshot, a result artifact, or Trackio retry staging. Admission
must include both the incoming snapshot and training workspace estimate.

The local and dstack gates use the pinned 0.8B base:

1. start with a clean qualification-only cache namespace;
2. materialize the exact revision and record a cache miss;
3. load tokenizer and model and run a finite-logit forward pass;
4. repeat and record a cache hit with zero downloaded model bytes;
5. corrupt one copied cache file and prove validation repairs or rejects the
   entry; and
6. demonstrate an impossible capacity request is rejected before GPU use.

Repeat the materialization and forward smoke for the 2B revision before the 2B
training gate. Do not include model weights in the OCI image or job bundle.

### Milestone 6: Prove the complete lifecycle on the local runner

Create a framework-owned qualification job that writes a synthetic loadable
model directory containing deterministic binary shards, model configuration,
tokenizer fixture, generation configuration, and a manifest. It must use the
normal training artifact writer and finalizer, not a special Trackio upload
script.

Run a producer with a uniquely named disposable Trackio project. Require it to
return `published_artifacts[0].reference` with an exact version and digest. Start
a second local-runner job using the returned descendant `ModelVariant`. It must
download the artifact, validate all files, perform a deterministic load check,
and record the artifact as an input.

The gate passes only when:

- both runs are queryable from Trackio;
- the producer has one required model output;
- the consumer uses the exact immutable version, not `latest`;
- Trackio reports the producer/consumer lineage edge;
- moving an alias between planning and execution has no effect;
- code-originated metrics and the outcome summary are present;
- local scheduler receipt data links to both run IDs but contains no artifact
  file manifest;
- the result remains downloadable after a Trackio restart; and
- workspace and retry staging return to the measured baseline while the
  committed Trackio artifact remains.

Run this at 64 MiB first, then at 512 MiB. If the larger case fails, fix the
transport or capacity policy before using a GPU.

### Milestone 7: Prove cross-machine lifecycle through dstack

Use the existing `/home/hammad/projects/ai-infra` dstack control plane and fleet.
The framework submission adapter should call the dstack Python SDK directly and
use the framework-owned stable runtime entry point. Until the first-class
adapter from `docs/architecture/proposed-dstack-execution-provider.md` exists,
the same entry point may be wrapped by the existing ai-infra `package-job`
script for qualification; the bundle digest, image digest, target snapshot, and
run specification must still be immutable.

The job environment contains only service connectivity and scoped
credentials: Trackio URL, Trackio project, Trackio write token, and an optional
scoped base-model read token. Trackio backing-store configuration and
credentials remain on the server. The bounded dataset and environment are in
the exact code distribution; the base model is a Hub reference; result bytes
flow through Trackio.

Submit the synthetic producer to one named worker and the consumer to the other
worker. The provider receipt must record dstack task ID, attempt, target
revision, task state, bounded log cursor, and Trackio run ID. Reconciliation
maps provider states into the canonical outcome but never interprets dstack
`done` as success until Trackio required-output finalization passes.

Exercise these failure cases:

- cancel before the runtime creates a Trackio run, then reconcile from the
  scheduler receipt;
- cancel after a Trackio run starts but before upload completion, then verify
  the run is interrupted and no false artifact version appears;
- lose the client connection during upload, retry the dstack attempt, and
  resume acknowledged chunks;
- restart Trackio during upload, resume, and verify the exact digest;
- retry after the provider reports a transient failure and prove the logical
  run/attempt relationship is preserved; and
- submit while the singular experiment slot is occupied and prove the next job
  queues rather than running concurrently.

The cross-machine gate passes when an artifact produced on worker A is loaded
on worker B, Trackio shows both runs and their lineage, dstack and framework
terminal states reconcile, no storage credential appears in the bundle or
logs, and both worker scratch directories return to baseline.

### Milestone 8: Qualify limited 0.8B and 2B GPU experiments

Run experiments serially. Start with Qwen 3.5 0.8B LoRA because it exercises
the real tokenizer, model, trainer, checkpoint, artifact, and evaluation path
at lower cost. Use a deliberately bounded dataset slice, one epoch or a small
fixed step count, deterministic seed, BF16 where the target supports it,
gradient checkpointing, and a checkpoint policy that retains at most the
latest recovery checkpoint plus the nominated final adapter.

Before training, record the exact model, dataset, environment, recipe, backend,
runtime image, execution target, and source revisions. During training, owning
code emits canonical loss, learning rate, throughput, memory, phase timing, and
task-relevant capability metrics to Trackio. After training, publish the
complete adapter directory with tokenizer/rendering facts and base-model
revision. Resolve the exact artifact version before deleting local output.

Create a separate qualification run that consumes that exact adapter version,
loads it over the pinned base model, evaluates both task-specific and selected
generalized capability checks, and records a pass/fail decision. Promote the
`candidate` alias only after finalization. Promote `qualified` only when the
recorded evaluation gate passes. The immutable version, not the alias, becomes
the parent for later training.

Repeat with Qwen 3.5 2B LoRA using the same lifecycle after updating the memory
and disk estimates from the 0.8B evidence. The two variants are independent
deliverables; neither replaces the other.

Before any full-weight or merged-model run, publish and consume a deterministic
artifact whose byte count is at least the predicted real result size. Require
successful interruption/resume, server restart, download, digest validation,
and cleanup at that size. Full fine-tuning remains a later algorithm variant,
not a hidden escalation of the LoRA qualification.

Each GPU gate must prove:

- the run used the intended GPU and immutable selections;
- the model parameters or adapter changed;
- the result reloads from Trackio in a clean process and produces finite logits;
- evaluation consumes the exact published version;
- metrics, trace evidence, source revisions, and produced/consumed lineage are
  queryable;
- the capability result is recorded even when it is negative;
- only nominated result/capability artifacts remain;
- recovery checkpoints and worker scratch obey retention; and
- disk and Trackio growth match the recorded artifact sizes within a documented
  storage-overhead allowance.

### Milestone 9: Publish, deploy, and close the evidence loop

Trackio and framework changes form a multi-repository release. Use this order:

1. In `/home/hammad/projects/trackio`, update implementation, tests,
   `CARBONTEQ_FORK.md`, compatibility notes, and release version. Commit and
   push the fork change. Merge or otherwise publish the exact immutable commit.
2. In `/home/hammad/projects/rl`, update
   `docs/tooling/trackio/README.md`, the exact Git revision in
   `packages/tracking-trackio/pyproject.toml`, and `uv.lock`.
3. Commit framework contracts, adapters, finalization, tests, project templates,
   documentation corrections, and this updated plan.
4. In `/home/hammad/projects/ai-infra`, change only the Trackio deployment input
   needed to consume the published qualified build. Use its existing deployment
   and rollback scripts; do not copy framework code into the infra repository.

Deploy the qualified Trackio build only after local fork tests pass. Back up the
Trackio data directory through the existing infra mechanism, deploy, verify
health and version, run the 64 MiB producer/consumer smoke, then run the 512 MiB
resume gate. Roll back the image digest if health, schema compatibility, or
artifact readback fails. A rollback must leave artifacts created by the prior
version readable.

Record each qualification as a Trackio run with code-originated metrics,
outcomes, selections, and lineage. Store concise command transcripts and
redacted environment snapshots under
`artifacts/qualification/artifact-lifecycle/<gate-id>/`. The filesystem
evidence supports audit and recovery; Trackio remains the queryable execution
and experiment history. Do not create a second result database or terminal
manifest.

The release is complete only after a fresh clone/clean-wheel consumer can submit
the limited job locally and through dstack, retrieve its exact result from
Trackio, and reproduce the qualification query without importing `apps/lab`.

## Concrete Steps

All commands that may print configuration must be reviewed for secret safety.
Do not use shell tracing. Replace generated gate IDs below with a timestamped,
non-secret value.

From `/home/hammad/projects/rl`, establish the starting state:

    git status --short
    git rev-parse HEAD
    git diff --check
    uv run python scripts/qualification/capture_artifact_preflight.py \
      --output artifacts/qualification/artifact-lifecycle/preflight.json

Expected behavior is a redacted report ending with either:

    admission: pass
    dstack_workers: 2
    trackio_health: ready

or a specific non-zero refusal such as:

    admission: blocked_version_drift
    expected_trackio_revision: <commit>
    observed_trackio_revision: <commit>

Inspect dstack through the existing infra wrapper:

    cd /home/hammad/projects/ai-infra
    ./scripts/dstack --version
    ./scripts/dstack fleet list

Do not proceed to remote mutation merely because both workers appear. Run the
read-only worker qualification supplied by that repository and retain its
redacted output in the framework gate directory.

Develop Trackio from the pinned base:

    cd /home/hammad/projects/trackio
    git fetch origin
    git switch -c codex/resumable-artifact-transport \
      c5072198b3b1556d31ed96ffc246a03f65418ab8
    uv sync --all-extras
    uv run pytest tests/unit/test_artifact.py \
      tests/unit/test_artifact_server.py \
      tests/unit/test_artifact_persistence.py \
      tests/unit/test_artifact_e2e.py -q

Add the new resumable and cleanup tests to this focused command as their files
are created. Run the larger real-server test explicitly:

    TRACKIO_LARGE_ARTIFACT_TEST=1 uv run pytest \
      tests/unit/test_artifact_resumable_e2e.py -q

Expect the test to report successful interruption/resume and readback. Capture
peak RSS, uploaded bytes, downloaded bytes, reclaimed staging bytes, and all
digests as test output or a JSON evidence file.

Develop the provider-neutral framework integration:

    cd /home/hammad/projects/rl
    uv run pytest packages/tracking/tests \
      packages/tracking-trackio/tests \
      packages/work/tests \
      packages/jobs/tests -q
    uv run lint-imports

Run the clean packaged-data gate from the new project repositories:

    cd /home/hammad/projects/ambient-agent
    uv build
    uv run pytest -q

    cd /home/hammad/projects/ambient-agent-environments
    uv build
    uv run pytest -q

The tests must install the wheel into a temporary clean environment and verify
resource loading there. A passing test that reads files from the checkout is
not sufficient.

Run the synthetic local gate through a framework command introduced during
Milestone 6. The exact CLI name may be adjusted to the existing CLI structure,
but the stable intended form is:

    cd /home/hammad/projects/rl
    uv run posttrain qualify artifact-lifecycle \
      --provider local \
      --size 64MiB \
      --project artifact-lifecycle-local-<gate-id>
    uv run posttrain qualify artifact-lifecycle \
      --provider local \
      --size 512MiB \
      --interrupt-at 40% \
      --project artifact-lifecycle-local-<gate-id>

Expected output includes the producer run ID, consumer run ID, exact artifact
version, digest, lineage result, cleanup byte counts, and `PASS`. It must not
print credentials, backing-store configuration, or model file contents.

Run the cross-worker gate:

    uv run posttrain qualify artifact-lifecycle \
      --provider dstack \
      --producer-target <worker-a-target-id> \
      --consumer-target <worker-b-target-id> \
      --size 512MiB \
      --interrupt-at 40% \
      --project artifact-lifecycle-dstack-<gate-id>

Expected output includes two dstack task IDs, two Trackio run IDs, one exact
artifact version, matching SHA-256 values, a producer/consumer edge, reconciled
terminal states, reclaimed scratch bytes, and `PASS`.

Run limited GPU gates only after synthetic gates pass:

    uv run posttrain run .posttrain/work_packages/ambient-agent-08b-lora.toml \
      --provider dstack
    uv run posttrain run .posttrain/work_packages/ambient-agent-08b-qualify.toml \
      --provider local
    uv run posttrain run .posttrain/work_packages/ambient-agent-2b-lora.toml \
      --provider dstack
    uv run posttrain run .posttrain/work_packages/ambient-agent-2b-qualify.toml \
      --provider local

The work-package filenames are deliverables of Milestone 8. Update this plan if
the public CLI or generated template selects a different stable name.

Before release, run the framework validation ladder from
`/home/hammad/projects/rl`:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Also run the full Trackio suite from `/home/hammad/projects/trackio` using the
commands documented by that repository at implementation time. Record exact
counts in this plan; do not write a guessed count in advance.

## Validation and Acceptance

Every milestone is a stop/go gate. A later milestone may not replace a failed
earlier proof with a larger run.

Milestone 0 passes when live services, versions, workers, credentials, disk, and
model selections are redacted and reproducible, and impossible storage is
rejected before GPU allocation.

Milestone 1 passes when a real Trackio process streams and resumes a 512 MiB
artifact with bounded memory, survives restart, rejects corruption, and exposes
no artifact version before complete durability. Unit-only evidence is
insufficient.

Milestone 2 passes when interrupted staging and disposable project bytes are
measurably reclaimed, while a retained artifact in another project remains
downloadable by exact digest.

Milestone 3 passes when a normal work-package producer returns a durable exact
reference before its workspace disappears and a normal consumer records input
lineage. Manual entry of `vN` is not acceptable as the primary flow.

Milestone 4 passes when clean wheels load all bounded data without network or
checkout access, the Verifiers environment produces a native trace, and
training/evaluation IDs are disjoint.

Milestone 5 passes when the exact 0.8B and 2B revisions load from verified
caches, a second materialization downloads zero model bytes, corruption is
detected, and capacity rejection occurs before GPU admission.

Milestone 6 passes when the local 512 MiB producer/consumer scenario survives
interruption and Trackio restart, records code-originated evidence and lineage,
and cleans workspace without deleting the result.

Milestone 7 passes when two different dstack workers exchange the exact result
only through Trackio, provider and Trackio states reconcile, retry does not
create a false version, and no backing-store secret reaches a job.

Milestone 8 passes separately for 0.8B and 2B when each result reloads in a clean
process, evaluation consumes the exact result version, generalized-capability
evidence is recorded, and retention keeps only nominated artifacts. A negative
capability result is valid evidence; missing evidence is not.

Milestone 9 passes when the published Trackio commit equals the framework pin
and deployed revision, a clean consumer completes both local and dstack
lifecycle gates, and the full validation ladders pass.

The final human-visible acceptance story is:

1. Open the Trackio project for a qualified limited run.
2. Observe immutable inputs, metrics, traces, outcome, and one selected model
   artifact.
3. Open the model artifact and observe its exact version and producer run.
4. Open the qualification run and observe that exact version as an input.
5. Download it in a clean process and reproduce the recorded model-load smoke.
6. Inspect worker storage and observe no abandoned job workspace or unselected
   recovery checkpoint.

## Idempotence and Recovery

Use a unique project and gate ID for every qualification attempt. Tests and
commands may be rerun without overwriting earlier research runs. Delete only a
project whose generated name and ownership marker identify it as disposable.
Never run project deletion against a research project selected by a mutable
shell variable or wildcard.

Trackio upload creation, chunk acknowledgement, completion, and artifact-version
commit must be idempotent. Retrying a completed chunk returns its existing
acknowledgement. Retrying completion returns the same completed digest. A new
artifact version is not created merely because an HTTP response was lost.

Preserve failed-run retry staging for the configured retry window. The cleanup
command defaults to dry-run and prints project, session IDs, age, and bytes.
Require an explicit project and confirmation flag for deletion. After cleanup,
rerun the read-after-write download of every retained qualification artifact.

The model cache uses completed-entry manifests and locks. Removing an incomplete
qualification cache entry is safe after no process owns its lock. Never clear a
shared Hugging Face cache wholesale. Tests should use a dedicated cache
namespace that can be removed independently.

Before deploying Trackio, record the current image digest and create the normal
data backup. On failure, redeploy the prior image digest, verify health, and
download a pre-deployment artifact. Do not roll back the database by deleting
new files unless the Trackio fork's migration explicitly documents that path.

The dstack retry path reuses immutable bundle and image digests but creates a
new provider attempt. Reconciliation preserves the prior attempt. Cancellation
is safe to repeat. If Trackio is unavailable, stop new GPU admission, retain
bounded staging, and retry finalization after the service returns.

The repositories may be dirty. Before each edit, inspect `git status --short`,
change only files named by the current milestone, and never reset or discard
unrelated user work. Do not commit until the user asks or the implementation
workflow reaches the explicit publication stage.

## Artifacts and Notes

Store non-secret qualification evidence under:

    artifacts/qualification/artifact-lifecycle/<gate-id>/
        preflight.json
        transport.json
        cleanup.json
        local-e2e.json
        dstack-e2e.json
        qwen35-08b-lora.json
        qwen35-2b-lora.json

These are bounded evidence summaries, not copies of datasets, model weights,
full logs, or terminal manifests. Each JSON file should contain schema version,
gate ID, timestamps, source revisions, invoked command, status, relevant run and
task IDs, byte counts, digests, and redacted diagnostics. Full semantic metrics,
traces, outcomes, artifact versions, and lineage remain in Trackio.

The compact provider receipt has this conceptual shape:

    {
      "schema_version": 1,
      "provider": "dstack",
      "provider_task_id": "...",
      "attempt": 1,
      "job_id": "...",
      "target_revision": "...",
      "bundle_digest": "sha256:...",
      "image_digest": "sha256:...",
      "state": "succeeded",
      "trackio_run_id": "...",
      "diagnostic_code": null
    }

It intentionally has no metrics, trace payload, artifact file list, storage
endpoint, credential, or “terminal manifest” field.

When implementation finishes a milestone, add a concise evidence excerpt here,
for example:

    Gate: dstack-cross-worker-20260726T...
    Producer task/run: ... / ...
    Consumer task/run: ... / ...
    Artifact: ambient-agent-synthetic:v3
    SHA-256: ...
    Upload resumed after: 7 of 16 chunks
    Worker scratch reclaimed: ... bytes
    Retained Trackio bytes: ... bytes
    Result: PASS

## Interfaces and Dependencies

The Trackio fork must expose a resumable transport behind its existing
high-level artifact API. Exact HTTP route names may follow Trackio conventions,
but the implementation must have typed internal request/response values
equivalent to:

    class ArtifactUploadCapabilities:
        resumable: bool
        compatibility_max_bytes: int
        chunk_size_bytes: int

    class ArtifactUploadSession:
        upload_id: str
        digest: str
        size_bytes: int
        chunk_size_bytes: int
        acknowledged_chunks: tuple[int, ...]
        expires_at: datetime

    class ArtifactUploadCompletion:
        digest: str
        size_bytes: int
        already_present: bool

The high-level `trackio.log_artifact` and `trackio.use_artifact` APIs remain the
job-facing interface. Do not require the framework to call chunk routes.

In `packages/common/src/posttrain/common/artifacts.py`, define a
provider-neutral durable output value equivalent to:

    @dataclass(frozen=True)
    class PublishedArtifact:
        logical_name: str
        kind: str
        reference: StoredArtifactRef
        size_bytes: int | None = None

Keep the existing
`RunDataSource.artifacts(run_id) -> ArtifactSet` method in
`packages/tracking/src/posttrain/tracking/contracts.py`. Add a conversion helper
beside the tracking models that turns each output `ArtifactLink` into a
`PublishedArtifact`. Preserve compatibility for existing readers and do not add
a duplicate artifact-list operation.

In `packages/work/src/posttrain/work/contracts.py`, extend the result
equivalently to:

    @dataclass(frozen=True)
    class WorkPackageJobResult:
        job_id: str
        kind: str
        definition: JobDefinition
        status: str
        run_id: str | None
        value: object
        published_artifacts: tuple[PublishedArtifact, ...] = ()

Use the repository's existing generic types rather than `object` if they are
available at implementation time.

In the work execution lifecycle, add a finalization callback or phase
equivalent to:

    async def finalize_required_artifacts(
        *,
        run_id: str,
        required_roles: Sequence[str],
        data_source: RunDataSource,
    ) -> tuple[PublishedArtifact, ...]: ...

It runs after the operation has logged outputs and before temporary workspace
cleanup. It performs read-after-write validation and fails closed.

The model materializer accepts a `HubModelRef` and returns a verified local
snapshot plus evidence:

    @dataclass(frozen=True)
    class MaterializedHubModel:
        reference: HubModelRef
        path: Path
        manifest_digest: str
        cache_state: Literal["hit", "miss", "repaired"]
        downloaded_bytes: int

No public value contains a Hugging Face token.

The execution provider uses the direct dstack SDK and the existing provider
proposal. Its compact receipt must be serializable, bounded, and independent of
Trackio. The common execution package must not import dstack; the dstack adapter
may import only the neutral execution contract and dstack SDK.

Required dependency policy:

- Python 3.12 and the repository's `uv` workspace remain authoritative.
- Trackio stays an immutable Git dependency until a qualified CarbonTeq release
  is published.
- dstack remains an adapter-only dependency in its owning package.
- Verifiers and each environment use immutable commits or published exact
  environment versions.
- runtime images are selected by OCI digest for qualification.
- no workflow orchestrator, direct bucket SDK, model registry, or sidecar log
  collector is added by this plan.

Update note (2026-07-26): Initial plan created to turn the selected packaged
dataset, immutable base-model reference, Trackio result registry, local runner,
and dstack runner path into staged implementation work. The plan adds explicit
large-file transport, durable result finalization, cleanup, version-drift,
cross-worker, and limited-GPU gates because small artifact tests alone would
not qualify the intended workflow.

Update note (2026-07-26 06:54Z): Recorded the first implementation evidence.
Milestone 0 now has executable storage/preflight checks and live infrastructure
qualification. Milestones 1 and 2 record the resumable Trackio candidate,
separate-process transport requirement, interrupted-client retry, bounded-memory
512 MiB gate, and incomplete-session cleanup, while leaving deployment and
remaining remote gates explicitly incomplete.
