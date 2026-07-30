# Deliver local and dstack execution through the normal framework CLI

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with
`docs/templates/PLAN.md`.

Packaging authority has moved to
`docs/plan/framework-oci-job-capsules.md`. The lifecycle, provider, queue,
reconciliation, and cleanup work in this document remains active. References
below to uploading a directory bundle with dstack `files` describe the
qualified characterization path and must not be used as the final distribution
contract. Normal execution will submit one framework-packed actual-job OCI
image and a launch envelope.

## Purpose / Big Picture

After this work, a developer can develop a normal post-training project on the
current machine, ask the `posttrain` CLI to plan and run one selected
work-package job locally or through dstack, leave the CLI, and later inspect
status, bounded logs, cancellation, and reconciled evidence using the canonical
run ID. The CLI and Python SDK use one application service. Both providers run
the same digest-pinned actual-job OCI image and launch-envelope contract, while
dstack remains responsible for remote placement and task lifecycle. The
framework's singular admission queue decides when a research job may be
submitted and does not release the next entry until terminal evidence is
reconciled.

The framework owns job meaning, deterministic job-image construction, logical
idempotency, local execution state, and the barrier between provider
termination and durable Trackio evidence. The infrastructure repository owns
the dstack service and worker enrollment, registry, Trackio, Doris, object
storage, credentials, and host storage.

More precisely, the framework repository owns work packages, registered job
definitions, resolved selections, reusable qualification scenarios,
`ExecutionRequest`, the execution lifecycle, deterministic code/configuration/
dataset bundles, local and dstack providers, job-kind runtime requirements,
final immutable image selection, the stable worker entrypoint, local run state,
terminal reconciliation, the CLI and Python SDK, and training-time observation
emitted by capability code.

The `ai-infra` repository owns dstack server deployment and worker attachment,
container-registry and BuildKit availability, Trackio, Doris, Observatory and
object-storage deployments, DNS, TLS, backups, credentials and service health,
worker Docker/NVIDIA configuration, host cache/checkpoint/scratch/cleanup
directories, and infrastructure qualification receipts. The framework
repository owns universal, job-kind, and actual-job Dockerfiles, dependency
selection, BuildKit invocation, image qualification, and image receipts.

This work makes one narrow amendment to the frozen baseline: detached
composition is static and runtime dependency activation belongs inside the
selected execution image. It otherwise implements the existing project,
work-package, job, run, execution-target, artifact, and observation contracts.

## Progress

- [x] (2026-07-26 16:24Z) Removed the optional payload-distribution research
  branch and selected the existing deterministic directory bundle plus
  digest-pinned OCI/BuildKit runtime as the implementation baseline.
- [x] (2026-07-26 16:24Z) Verified the existing provider-neutral, local Docker,
  and dstack package tests: 18 tests passed.
- [x] (2026-07-26 16:43Z) Added the provider-neutral
  `JobExecutionService` and durable run-to-provider submission store. The
  focused execution suite now passes 17 tests; Ruff and Pyright are clean.
- [x] (2026-07-26 17:33Z) Added project execution-target configuration and
  provider factories without
  storing service tokens or machine credentials in project files. Completed:
  validated secret-free project defaults for provider, logical target, runtime
  profile, timeout, attempts, priority, and environment-variable names; added a
  mode-protected local binding, deterministic precedence with provenance, and
  local/dstack provider factories. CLI overrides and resolved provenance are
  visible in `work-package plan`; virtualenv interpreter symlinks are preserved.
- [x] (2026-07-26 16:16Z) Added and qualified the framework BuildKit pack/build
  service. It combined the infrastructure-published Python 3.12 base with the
  framework job layer, pushed an OCI image with provenance and SBOM
  attestations to `registry.lan`, cold-pulled it by digest, and ran
  `posttrain-runtime --help`. Retained image:
  `registry.lan/carbonteq/posttrain-job-runtime@sha256:dd5bb8f0dd0b9e4ab55f248e8498e8392bf59b2d02f8a27dd80e7734cc15f315`.
- [x] (2026-07-26 17:33Z) Added `work-package plan` and provider-backed
  `work-package run`; the current in-process path remains only as a temporary
  compatibility mode.
- [x] (2026-07-26 17:33Z) Added `run status`, `run logs`, `run cancel`,
  `run reconcile`, and evidence-gated `run cleanup` over persisted provider
  handles.
- [x] (2026-07-26 17:33Z) Replaced arbitrary copied-script commands with a versioned execution
  manifest and stable `posttrain-runtime execute` worker entrypoint. Completed:
  immutable manifest contract, bundle coverage, prepare-before-execute API,
  installed worker command, an end-to-end registered-job integration test, and
  exclusive use by new CLI requests.
- [ ] Migrate one non-Ambient qualification scenario from the characterization
  harness to the CLI and prove equivalent local and dstack behavior. Local
  Qwen 2B SFT run `sft-cli-smoke-004` completed two real backward passes,
  reconciled three retained artifacts, appeared in remote Observatory, and
  passed bounded cleanup. Remote run
  `sft-dstack-qual-20260726-173000` completed the same two-update job on
  `pop-os.lan`, reconciled three retained artifacts through Trackio, remained
  queryable through Doris-backed Observatory after provider-managed cleanup,
  and left only a 1,071-byte run-scoped worker bundle. Remaining: remove the
  superseded characterization path after the substantial GRPO check.
- [x] (2026-07-26 17:41Z) Qualified placement and lifecycle behavior across
  both dstack workers. Two simultaneous digest-pinned jobs ran on
  `pop-os.lan` (RTX 4090) and `carbonteq-ai-workstation.lan` (RTX PRO 6000);
  running cancellation released its block, native no-capacity waiting entered
  `pending` only with an explicit event/duration policy, queued cancellation
  became terminal, and both workers ended healthy and idle.
- [x] (2026-07-29 16:32Z) Requalified concurrent placement through the normal
  immutable actual-job path. While DAPO MTP run
  `verl-dapo-mtp-fixed-20260729` occupied the healthy RTX PRO worker, capacity-
  only run `capacity-concurrency-4090-20260729` requested one CUDA GPU with at
  least 8 GiB and no hostname. dstack assigned it to the idle healthy RTX 4090.
  Both fleet instances reported `1/1` busy concurrently, both provider runs
  completed successfully, and neither submission was held behind the other by
  framework admission.
- [x] (2026-07-26 18:00Z) Removed developer-side Verifiers materialization
  from work-package validation. The GSM8K GRPO package now validates on the
  development machine without Verifiers installed; native environment
  construction remains in the worker's GRPO request builder.
- [ ] Replace the qualified directory-bundle transport with the framework OCI
  capsule flow in `framework-oci-job-capsules.md`. The existing provider and
  cleanup behavior remains the parity reference, not the final pack contract.
- [x] (2026-07-27) Replace the process-local serial queue with durable
  per-worker admission in the normal CLI. Admission persists immutable plans,
  submission intent, waiting position, failure state, and the evidence barrier;
  supports explicit idempotent submission retry; and permits the two physical
  workers to run independently. The obsolete queue implementation and its
  characterization script were removed.
- [x] (2026-07-27 04:20Z) Close the admission audit gaps around provider
  ambiguity and restored state. Submission failure is retained and explicitly
  retryable, status and cancellation are side-effect free, reconciliation is
  idempotent, queue position and admission are scoped to one exact physical
  worker, restored provider bindings fail closed on drift, no-op observation
  has a truthful terminal barrier, successful untracked outputs cannot be
  deleted, and snapshot replacement is directory-synced, validated, and
  bounded. The focused lifecycle/CLI suite passed 71 tests.
- [x] (2026-07-27 04:32Z) Close the follow-up admission race before live
  testing. Per-run kernel claims prevent concurrent provider calls and recover
  automatically after process death; ambiguous failures quarantine rather
  than release the physical-worker key. Abrupt post-accept death and concurrent
  caller tests pass, local target aliases share the same host admission key,
  next-admission errors are surfaced by reconciliation, and bounded snapshot
  pruning first writes compact mode-protected terminal receipts.
- [x] (2026-07-27 05:47Z) Qualified the maintained dstack
  `stop_duration` source candidate at upstream tag `0.20.29`. The bounded
  duration is admitted by the task schema, forwarded from server to runner,
  applied as Go `Cmd.WaitDelay`, and reused for the server removal deadline;
  legacy omission retains the 300-second fallback. SQLite-focused tests passed
  83 cases with 21 PostgreSQL variants skipped, the PostgreSQL gate passed all
  104 cases, Go executor/schema tests passed 35 top-level cases, and Ruff,
  Python/Go formatting, and diff checks are clean. Publication and the
  same-revision server/runner/shim build remain open because the CarbonTeq
  repository does not yet exist.
- [ ] Run the required realistic algorithm qualification through the CLI,
  including at least fifteen backward passes, remote Trackio evidence, Doris-backed
  Observatory visibility, retained results, and bounded worker cleanup.
- [ ] Remove the superseded operator scripts after CLI parity and update the
  related qualification plan with final evidence.

## Surprises & Discoveries

- Observation: GPU memory capacity is an offer constraint, not a fractional
  sharing unit for one physical GPU.
  Evidence: the live SSH fleet exposes one scheduling block on each worker.
  The 96 GiB and 24 GiB GPUs therefore admitted two simultaneous one-GPU jobs
  on different workers; remaining VRAM on a busy single-GPU worker did not
  create a second safe placement. Capacity-only selection still allowed dstack
  to choose the idle 24 GiB worker without a hostname pin.

- Observation: the three execution packages already implement a coherent
  provider protocol and both lifecycle adapters.
  Evidence: the focused baseline suite passed 18 tests before this plan changed
  code.

- Observation: provider handles are not yet durably associated with canonical
  run IDs.
  Evidence: `ExecutionJournal` appends provider observations, but no neutral
  store can recover an `ExecutionHandle` after the submitting process exits.

- Observation: the current worker runtime verifies a bundle and then executes
  an arbitrary command supplied by the caller.
  Evidence: `packages/execution/src/posttrain/execution/runtime.py` accepts
  `argparse.REMAINDER` and calls `os.execvp`. This proves bundle verification,
  but it does not reconstruct a registered versioned job definition.

- Observation: the qualification runner currently owns bundle selection,
  runtime receipt resolution, target translation, environment names,
  submission, waiting, evidence validation, and final receipt writing.
  Evidence:
  `scripts/qualification/run_algorithm_scenario.py` imports both execution
  providers and assembles their requests directly. It remains a useful
  characterization harness, but it is not the product interface.

- Observation: the work-package API previously coupled resolution and
  execution.
  Evidence: remote planning could not obtain the canonical `RunSpec` without
  copying private runner logic. `prepare_work_package_job` now provides one
  preflighted meaning for local execution, manifest creation, and worker
  verification.

- Observation: the current generic framework runtime and the veRL
  characterization runtime use different Python contracts.
  Evidence: `ai-infra/infra/posttrain-runtime/Dockerfile` uses Python 3.12,
  matching every framework package's `>=3.12,<3.13` requirement. The current
  veRL runtime uses Python 3.13 and injects framework source through
  `PYTHONPATH`; it cannot be promoted as a generic framework base until either
  the base returns to Python 3.12 or the complete framework is qualified and
  published for Python 3.13.

- Observation: building individual framework packages outside their uv
  workspace initially failed even with `--no-deps`.
  Evidence: package metadata contains `tool.uv.sources` entries with
  `workspace = true`; uv correctly rejected those references when the image
  contained only the selected runtime packages. The job image now uses
  `uv pip install --no-sources --no-deps` so local package inputs remain
  explicit while workspace-only source overrides are ignored.

- Observation: the corrected cold build produced an OCI image index rather
  than a Docker-only manifest.
  Evidence: registry inspection reports one Linux/AMD64 image manifest and one
  attestation manifest. The mode-`0600` receipt records the exact source,
  lock, base-image, build, and output-image digests.

- Observation: a successful remote-observed training run can leave root-owned
  files in a local bind-mounted workspace even when its retained artifact
  policy is correct.
  Evidence: cleanup of `sft-cli-smoke-004` removed its exact terminal
  container, then initially failed with `Permission denied` on the run-scoped
  dataset directory. The local provider now uses the already-pulled immutable
  job image as a fixed-command cleanup helper and requires no sudo.

- Observation: resolving an executable path is not equivalent to making it
  absolute.
  Evidence: `Path.resolve()` collapsed the configured dstack virtualenv
  `bin/python` symlink to the base uv interpreter, which could not import
  dstack. Executable parsing now preserves the virtualenv symlink and has a
  regression test.

- Observation: dstack validates uploaded `files` before calculating offers.
  Evidence: a side-effect-free CLI plan failed because its future deterministic
  bundle directory did not yet exist. The dstack adapter now omits `files`
  only from offer planning, records that upload validation is deferred, and
  submits the complete configuration only after the CLI materializes and
  verifies the bundle.

- Observation: startup failures may never create a Trackio run but still need
  safe cleanup.
  Evidence: failed local runs `sft-cli-smoke-001` through `003` retained
  mode-`0600` diagnostics of 25, 63, and 198 lines, then removed their exact
  containers and reclaimed 905 logical workspace bytes each.

- Observation: dstack retry policy is event-and-duration based, not an attempt
  counter.
  Evidence: a third fixed-fleet job with `retry: false` failed immediately
  with `FAILED_TO_START_DUE_TO_NO_CAPACITY`; a job configured with
  `retry.on_events: [no-capacity]` and a two-minute duration remained
  `pending` until cancelled. Bare `retry: true` cannot faithfully represent
  `ExecutionPolicy.max_attempts`.

- Observation: terminal dstack responses lose placement information that was
  visible while a run was active.
  Evidence: both workers were observed during the successful two-job placement
  test, but terminal `run get` responses no longer included the top-level
  hostname. Framework state must retain the last observed placement.

- Observation: dependency preflight had leaked from worker execution into
  developer composition.
  Evidence: validating
  `.posttrain/work_packages/gsm8k_qwen08b_grpo_qualification.yaml` previously
  failed with `install the Verifiers integration dependencies`. Removing the
  environment materialization call from the standard runtime seat resolver
  made the same command pass while
  `create_verifiers_training_bridge` still performs native preflight during
  job execution.

- Observation: the deployed evidence path is genuinely remote and
  Doris-backed, but research evidence is less complete than execution
  evidence.
  Evidence: `posttrain run show sft-cli-smoke-004` and deployed Observatory
  resolved the same Trackio run through `https://trackio.lan`; live control
  inspection reported `DorisStorage`. The run retained complete required
  artifacts, yet actual worker/GPU identity, bundle/image digests, canonical
  GPU series, and source revision were absent from remote run evidence.

## Decision Log

- Decision: supersede the directory bundle plus runtime image with one
  framework-packed actual-job OCI image.
  Rationale: the previous path proved provider lifecycle and reproducibility
  but left two transport identities. The actual-job image now covers exact
  code, environment repositories, datasets, resolved configuration, and worker
  contract; the launch envelope covers run/attempt/provider values.
  Date/Author: 2026-07-26 / user and Codex, superseding the earlier bundle
  decision.

- Decision: introduce an application service before adding CLI lifecycle
  commands.
  Rationale: the CLI and Python callers must share idempotency, state
  persistence, journaling, and provider-result handling instead of each
  rebuilding lifecycle logic.
  Date/Author: 2026-07-26 / Codex.

- Decision: initiate execution from `work-package plan` and
  provider-backed `work-package run`, then manage it with `run status`,
  `run logs`, `run cancel`, and `run reconcile`.
  Rationale: a work package and selected job define what to start; the
  canonical run ID defines the detached execution afterward. The existing
  `run show` remains the read-only Observatory evidence view.
  Date/Author: 2026-07-26 / Codex.

- Decision: separate singular framework admission, execution attempts, and
  dstack resource placement.
  Rationale: durable framework admission owns one-experiment-per-physical-worker
  policy, logical idempotency, submission recovery, and the terminal-evidence
  barrier. An admitted dstack task is fail-fast and dstack owns worker
  selection, startup, termination, and live resource accounting.
  An explicit provider capacity-wait duration maps only to dstack's
  `no-capacity` retry event; execution attempt count remains a separate
  framework concept with explicit lineage. `posttrain run queue` reports
  framework and provider queue scopes plus requested and assigned host
  identity.
  Date/Author: 2026-07-29 / user and Codex, revised after a busy pinned worker
  failed immediately instead of remaining pending.

- Decision: provider submission ambiguity requires an explicit retry command.
  Rationale: neither `status` nor `cancel` may repeat a write merely because
  the original provider call raised before the local handle was saved.
  `submission_failed` retains the exact plan, idempotency key, and
  physical-worker admission key; `run retry-submit` is the only operation
  allowed to resolve the ambiguity. A per-run kernel claim excludes concurrent
  provider calls and disappears automatically if its process dies. A persisted
  secret-free provider-binding fingerprint prevents the retry from using a
  materially different target or storage binding after restart.
  Date/Author: 2026-07-27 / Codex.

- Decision: the framework owns the complete three-level image hierarchy.
  Rationale: universal, job-kind, and actual-job image contents are framework
  product semantics. Infrastructure supplies a registry, BuildKit reachability,
  worker compatibility, and operational policy without choosing dependencies.
  Date/Author: 2026-07-26 / user and Codex, revised after the image hierarchy
  was finalized.

- Decision: require a worker-reachable OCI registry for dstack execution.
  Rationale: dstack schedules and starts a selected image but does not build or
  publish the framework's final job image. The framework BuildKit service
  pushes the verified image and records its digest; infrastructure makes the
  registry available and configures worker pull credentials. Local-only
  execution may use a local image, but remote planning must reject an image
  that workers cannot pull by immutable digest.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: put selected datasets, project code, and pinned Verifiers
  environment packages in the actual-job image.
  Rationale: the final OCI digest becomes the single distributable job identity.
  Model weights, secrets, mutable checkpoints/caches, and final model artifacts
  remain outside the image.
  Date/Author: 2026-07-26 / user and Codex, revised after selecting OCI
  capsules.

- Decision: use layered execution configuration with visible provenance.
  Rationale: committed defaults remove repetitive flags, while explicit CLI
  overrides keep one-off runs easy. `.posttrain/project.toml` remains the
  canonical project manifest; `pyproject.toml` declares package/build runtime
  requirements rather than duplicating project execution settings. Protected
  user-local configuration owns machine paths and service bindings, environment
  variables own secret values, and CLI flags own one-run overrides.
  Precedence is CLI, user-local configuration, project default, then registered
  job default. `plan` reports the resolved value and source for every
  operational setting.
  Date/Author: 2026-07-26 / user and Codex.

- Decision: keep the qualification scripts temporarily as characterization
  tests, not supported operator interfaces.
  Rationale: they contain proven request and acceptance behavior that can
  validate CLI parity. Removing them before the CLI path is qualified would
  discard useful evidence.
  Date/Author: 2026-07-26 / Codex.

- Decision: make cleanup an explicit evidence-gated lifecycle operation.
  Rationale: successful jobs may release provider and workspace state only
  after Trackio reconciliation proves their required artifact roles. A failed
  or cancelled startup with no Trackio record may instead retain a bounded
  provider diagnostic before exact-handle cleanup. Inconsistent tracking and
  provider outcomes never bypass the barrier.
  Date/Author: 2026-07-26 / Codex.

- Decision: keep provider planning side-effect-free and defer dstack bundle
  upload validation to submission.
  Rationale: code files do not affect resource offers. Planning can calculate
  offers from the same image, resource, placement, environment-name, volume,
  timeout, retry, and priority contract without requiring the future bundle
  path. Submission still validates and uploads the exact materialized digest.
  Date/Author: 2026-07-26 / Codex.

- Decision: environment installation and native preflight occur only in the
  execution runtime.
  Rationale: a developer must be able to inspect, validate, and plan a remote
  job without installing its CUDA, vLLM, Verifiers, or environment-package
  dependencies. Composition validates immutable source metadata and selection
  types; the packaged worker constructs the native environment immediately
  before the operation starts and records a typed failure if it is unusable.
  Date/Author: 2026-07-26 / Codex.

## Outcomes & Retrospective

The optional packaging/distribution research branch is removed. The selected
implementation baseline is now the deterministic bundle, BuildKit/OCI runtime,
local Docker adapter, and dstack adapter. The provider-neutral application
service durably maps canonical run IDs to provider handles and supports
restart-safe status, bounded logs, cancellation, collection, reconciliation,
and cleanup. The framework
job image is now published and smoke-qualified by immutable digest with
provenance and SBOM attestations. The first CLI-only local SFT run completed,
retained its remote evidence after cleanup, and converted three failed startup
attempts into bounded diagnostics rather than abandoned containers. Remote
dstack SFT parity, both-worker placement, running and queued cancellation,
remote Trackio/Doris reads, and retained cleanup evidence are now proven. The
substantial fifteen-update GRPO qualification, CLI admission integration, and
the remote execution-context/telemetry gaps remain open.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`. A project contains tracked
configuration under `.posttrain/` and ignored machine-local state under
`.posttrain/state/`. A work package groups runs at one stage. A run is one
observed execution of a versioned job definition.

`packages/execution` owns scheduler-neutral request, plan, handle, status, log,
result, bundle, journal, and waiting contracts.
`packages/execution-local` translates them to detached Docker containers.
`packages/execution-dstack` translates them to dstack through its Python SDK
bridge. `apps/cli` owns the stable `posttrain` command.

The existing `ExecutionJournal` is an append-only stream of observed provider
states. A submission store is different: it is one compact durable record
mapping a canonical run ID to its provider handle and immutable execution
identities. It enables a later CLI process to recover the remote or local
execution without searching providers heuristically.

Trackio receives observations from training and evaluation code. It stores
model and evidence artifacts and exposes them to Observatory. Jobs know the
Trackio service URL and write token through injected environment-variable
names. Jobs do not know that Trackio uses Apache Doris internally.

The shared image boundary has two layers. `ai-infra` publishes and qualifies
generic immutable CUDA/PyTorch or veRL base images. Framework job definitions
declare their runtime layer and exact dependencies. A framework-owned BuildKit
pack/build service creates or resolves the final immutable job image. The
selected final image digest becomes part of `ExecutionRequest`, the worker
manifest, the submission record, and run lineage.

## Plan of Work

First add `packages/execution/src/posttrain/execution/service.py`. Define an
immutable `ExecutionSubmission` record, a filesystem-backed
`ExecutionSubmissionStore`, and `JobExecutionService`. The store writes one
mode-`0600` JSON record beneath
`.posttrain/state/executions/<run_id>/submission.json` using atomic create and
replace. Rewriting an identical submission is idempotent; conflicting provider
or immutable execution identities fail. The service plans and submits requests,
persists the handle before returning, recovers handles by run ID, appends every
status observation to that run's execution journal, pages logs, cancels, and
collects terminal provider results. It receives a concrete provider through
dependency injection and does not import Docker, dstack, Trackio, or the CLI.

Next define project-side execution configuration. Provider bindings may name a
provider, logical target selectors, runtime receipt location, and required
environment-variable names. Service URLs and tokens come from the process or
protected provider configuration. The project file must not contain credential
values. A local provider factory creates `LocalDockerExecutionProvider`; the
dstack optional dependency creates `DstackExecutionProvider` and its SDK
bridge.

Keep execution configuration layered and deterministic. Committed defaults
such as provider, logical target, runtime profile, timeout, retry policy, and
non-secret environment names live in `.posttrain/project.toml`.
`pyproject.toml` may declare the package's framework runtime profile, build
target, and dependency group because those values describe the installable job
package. A protected user-local config supplies dstack SDK paths, provider
project bindings, receipt locations, and machine-specific mount roots.
Environment variables supply secret values. CLI flags override non-secret
values for one invocation. Resolution order is CLI, user-local, project, then
registered job default, and the planned execution records the winning source
for every value.

Add a framework pack/build application boundary after provider configuration.
Registered job definitions declare a runtime profile and source inputs. The
builder combines the selected infrastructure base-image digest with the
framework runtime layer, dependency lock, registered job handlers, and stable
worker entrypoint. It uses BuildKit, records cache inputs, runs image smoke
checks, pushes remote-job images to the infrastructure-owned OCI registry, and
returns only a verified immutable final-image reference to the execution
planner. Provider adapters consume the resulting image; they do not build or
publish it. Worker registry authentication is infrastructure configuration and
never enters a bundle or execution manifest.

Then extend `apps/cli/src/posttrain_cli/commands/work_package.py`. Planning
resolves the selected work-package job and all selections, builds the bounded
bundle, including selected project code, configuration, Verifiers repository
content, and selected dataset when the registered definition requires them. It
then resolves or builds the final job image, creates a canonical `RunSpec` and
`ExecutionRequest`, calls
`JobExecutionService.plan`, and prints the run ID, provider, target, bundle
digest, image digest, and redacted provider offer summary. Running repeats the
same deterministic planning calculation, submits idempotently, and returns
after provider acceptance unless the user explicitly requests bounded waiting.
The CLI, rather than a repository script, selects files and submits the job.

Extend `apps/cli/src/posttrain_cli/commands/run_cmd.py` with detached lifecycle
commands. Each command resolves the persisted submission by canonical run ID,
constructs the configured provider, and calls the application service. Status
and logs are read-only. Cancel records cancellation intent before asking the
provider to stop. Reconcile requires a provider-terminal result and durable
Trackio terminal evidence before reporting framework completion.

Add explicit cleanup after reconciliation. `posttrain run cleanup <run-id>`
must write a cleanup plan before destructive work, be retryable after partial
cleanup, target only the persisted provider handle and exact run-ID workspace,
and retain compact reconciliation and cleanup receipts. Successful outcomes
require consistent Trackio evidence. Failed or cancelled startups that have no
tracking record retain a bounded mode-`0600` diagnostic first. Local Docker
uses a fixed command in the digest-pinned runtime image to empty root-owned
bind-mount contents; dstack retains provider history and owns task-resource
release.

After lifecycle DX works, replace the arbitrary command with a versioned
execution manifest. The bundle contains the project manifest, work-package
configuration, resolved selections, registered job-definition ID, runtime
image digest, expected inputs and outputs, and non-secret environment names.
It also records retention policy and typed workspace/cache mount purposes. It
contains no provider submission or other control-plane script. The stable
worker command is:

    posttrain-runtime execute \
      --manifest /opt/posttrain/bundle/.posttrain/job.json

That entrypoint verifies the bundle and manifest, rebuilds the standard job
runtime, rejects version mismatches, and executes exactly the selected job. It
must not import or call a qualification script.

The selected registered job handler emits observations through `RunContext`
from inside training, evaluation, serving, or data code. Trackio's URL may come
from project or execution-target configuration. Its write token is injected by
the execution environment and is never serialized. Doris remains an
implementation detail behind Trackio.

Use the existing singular-experiment coordination in
`packages/execution/src/posttrain/execution/queue.py` only as a framework
admission policy around dstack submissions. It must not calculate GPU
placement. The framework admits one research job, submits it to dstack
fail-fast, and does not release the next entry until terminal evidence is
reconciled. dstack owns worker selection, live GPU availability, startup, and
termination. A later concurrent policy may explicitly request provider
capacity waiting, but it must not derive that behavior from an execution
attempt count.

Finally migrate one bounded GSM8K or another non-Ambient scenario to the new
CLI path. Compare its `ExecutionRequest`, provider state transitions, Trackio
run, model artifact, native traces, and terminal receipt with the existing
characterization harness. After parity, remove the corresponding script-based
operator path and run the substantial algorithm qualification only through the
CLI.

## Concrete Steps

Run commands from `/home/hammad/projects/rl`.

For the first application-service milestone:

    uv run pytest packages/execution/tests -q
    uv run ruff check packages/execution
    uv run pyright packages/execution
    git diff --check

For provider and CLI milestones:

    uv run pytest \
      apps/cli/tests \
      packages/execution/tests \
      packages/execution-local/tests \
      packages/execution-dstack/tests -q
    uv run ruff check apps/cli packages/execution packages/execution-local packages/execution-dstack
    uv run pyright apps/cli packages/execution packages/execution-local packages/execution-dstack
    uv run lint-imports

The intended user flow is:

    posttrain work-package plan train.yaml --job train --provider dstack
    posttrain work-package run train.yaml --job train --provider dstack
    posttrain run status <run-id>
    posttrain run logs <run-id>
    posttrain run cancel <run-id>
    posttrain run reconcile <run-id>
    posttrain run cleanup <run-id>
    posttrain run show <run-id>

Before the final GPU qualification, run the complete repository ladder:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

The application service is accepted when tests prove that submission persists
the canonical run-to-provider mapping, the mapping survives a new service
instance, identical resubmission is idempotent, conflicting resubmission fails,
status observations are append-only, logs page correctly, cancellation reaches
the original handle, and collection rejects non-terminal runs.

The CLI lifecycle is accepted when a developer can submit, exit, and use a new
CLI process to inspect and control the same local or dstack execution by run
ID. JSON output must contain no token value, dstack credential, Trackio write
credential, or full environment dump.

Local and dstack equivalence is logical rather than storage-identical. Given
the same manifest, the two providers must receive the same run, job, bundle,
image, target, input, output, and retention identities. Native provider plans,
handles, state names, and receipts may differ.

Framework completion requires both a provider-terminal result and retained
Trackio evidence. Provider success without a terminal Trackio run and required
artifact references remains unreconciled. Trackio success while the provider
is still active also remains unreconciled.

Cleanup is accepted when a reconciled successful run keeps its Trackio
artifacts and Observatory view after the exact container and disposable local
workspace are gone. Failed/cancelled pre-tracking runs must retain a bounded
diagnostic and compact provider receipt. Repeating cleanup must return the same
receipt without contacting or deleting another run.

The final algorithm check must use realistic rollout settings and complete at
least fifteen backward passes. It must publish queryable observations from inside
the training code, retain selected model and trace artifacts through Trackio,
appear in remote Observatory through Trackio's Doris backend, and leave only
policy-retained worker data.

## Idempotence and Recovery

Planning is read-only and repeatable. Submission uses a stable idempotency key
and provider-native deterministic name. If a process exits after provider
acceptance but before the local submission record is written, recovery first
queries the deterministic provider identity and writes the matching record; it
does not blindly submit another run.

Submission records and journals are compact retained state and are never
deleted by normal worker cleanup. Bundles are immutable. A failed partial
bundle build is removed by the bundle builder. Provider cancellation and
cleanup target an exact persisted handle and never use global Docker or
filesystem pruning.

Cleanup writes `cleanup-plan.json` before releasing provider state. If the
process exits after the container is removed, the next invocation resumes the
same plan; local Docker treats an absent exact container idempotently and still
empties the run workspace. `cleanup.json` makes subsequent calls read-only.

If Trackio is temporarily unavailable after provider termination, reconciliation
remains retryable. The run is not marked complete until the evidence reader can
verify the terminal run and required artifact roles.

## Artifacts and Notes

The durable control-plane artifacts are the immutable execution manifest,
bundle digest, runtime-image digest, submission record, append-only execution
journal, provider result, and reconciliation result. Full model bytes, native
traces, and training summaries remain Trackio artifacts. Provider logs are
bounded diagnostic streams and are not copied wholesale into the submission
record.

The supporting architecture is
`docs/architecture/proposed-dstack-execution-provider.md`. Algorithm
qualification continues in
`docs/plan/multi-environment-algorithm-qualification.md` after this CLI
vertical slice replaces its operator script.

## Interfaces and Dependencies

In `packages/execution/src/posttrain/execution/service.py`, define:

    @dataclass(frozen=True, slots=True)
    class ExecutionSubmission:
        run_id: str
        provider: str
        provider_id: str
        idempotency_key: str
        bundle_digest: str
        runtime_image: str
        submitted_at: datetime

    class ExecutionSubmissionStore:
        def save(
            self,
            submission: ExecutionSubmission,
        ) -> ExecutionSubmission: ...

        def load(self, run_id: str) -> ExecutionSubmission: ...

    class JobExecutionService:
        def plan(self, request: ExecutionRequest) -> ExecutionPlan: ...
        def submit(self, plan: ExecutionPlan) -> ExecutionSubmission: ...
        def status(self, run_id: str) -> ExecutionRecord: ...
        def logs(
            self,
            run_id: str,
            cursor: LogCursor | None = None,
            *,
            limit: int = 200,
        ) -> LogPage: ...
        def cancel(self, run_id: str) -> None: ...
        def collect(self, run_id: str) -> ExecutionResult: ...
        def cleanup(self, run_id: str) -> ProviderCleanupResult: ...

`JobExecutionService` depends only on `ExecutionProvider`,
`ExecutionSubmissionStore`, and `ExecutionJournal`. Provider factories live
outside this neutral package. The initial implementation adds no scheduler,
workflow orchestrator, new artifact registry, or direct Doris client.

Revision note (2026-07-26): updated the living plan after the CLI lifecycle,
stable worker, immutable job image, real local SFT, terminal reconciliation,
and evidence-gated cleanup were implemented. Recorded the virtualenv symlink,
dstack planning, and root-owned workspace defects because each changed the
production boundary and retry behavior.

Revision note (2026-07-26): packaging moved to
`framework-oci-job-capsules.md`. Corrected ownership so the framework builds
all image levels and infrastructure provides only operational registry,
BuildKit, worker, and service capabilities.

Revision note (2026-07-26): revised scheduling after live qualification on both
GPU workers. Singular framework admission, provider capacity waiting, and
execution attempts are now separate concepts; dstack submissions are
fail-fast. Recorded remote SFT cleanup, Doris-backed Observatory evidence, and
the developer/worker Verifiers preflight boundary.

Revision note (2026-07-26): Created this implementation plan after the user
selected the existing deterministic bundle and OCI path and asked to begin the
main framework implementation. It replaces optional packaging research with a
CLI-first provider lifecycle.

Revision note (2026-07-26): Expanded the plan boundary to match the selected
architecture exactly: framework-owned scenarios, payloads, final job images,
providers, runtime entrypoint, lifecycle and evidence reconciliation;
infrastructure-owned registry/BuildKit services, worker setup and
operational health.
