# Qualify every maintained job kind through immutable OCI capsules

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. Maintain this document according to
`docs/templates/PLAN.md`.

## Purpose / Big Picture

After this work, a framework developer can point to retained provider,
Trackio, Apache Doris, Observatory, artifact, and cleanup evidence for every
maintained public job kind rather than inferring support from unit tests or
from another job that happens to use the same base image. Training
qualifications are deliberately substantial enough to exercise repeated
backward passes but bounded enough to remain infrastructure tests rather than
full research runs. The normal `posttrain job run` and `posttrain run`
lifecycle is the only qualification path.

## Progress

- [x] (2026-07-27 02:25Z) Audit retained capsule evidence and identify missing
  live coverage for `data.prepare`, `train.dpo`, `train.sampo`,
  `train.distill`, and `serve.smoke`; identify that final-capsule SFT has only
  two optimizer steps.
- [x] (2026-07-27 02:25Z) Reconfirm both dstack workers are active, healthy,
  unsliced, and idle: `pop-os.lan` exposes an RTX 4090 with 24 GiB and
  `carbonteq-ai-workstation.lan` exposes an RTX PRO 6000 with 96 GiB.
- [x] (2026-07-27 02:35Z) Add exact qualification selections and work packages
  for ten-step SFT, ten-step DPO, ten-step SAMPO, ten-step on-policy
  distillation, serving smoke, plus a deterministic project-owned preference
  dataset.
- [x] (2026-07-27 02:37Z) Validate the five new work packages statically and
  prove their package plans select only the published `supervised`,
  `online-rl-trl-py312`, and `serve` runtime profiles.
- [x] (2026-07-27 02:49Z) Implement `data.prepare` as a provider-neutral
  capability and two standard canonicalization jobs, publish all six
  qualification capsules from one framework source digest, and retain a real
  local prepared-dataset artifact through Trackio and Apache Doris.
- [x] (2026-07-27 02:54Z) Fix post-cleanup lifecycle reconciliation so a
  deliberately removed provider object resolves through its append-only
  terminal evidence rather than being reclassified as lost.
- [x] (2026-07-27 03:32Z) Qualify ten-step SFT and DPO plus serving smoke on
  `pop-os.lan`, and ten-step SAMPO on `carbonteq-ai-workstation.lan`, never
  placing two simultaneous experiments on one worker.
- [x] (2026-07-27 03:32Z) Reconcile and clean every completed SFT, DPO, SAMPO,
  and serving-smoke workspace after current-source Observatory and direct
  Doris readback.
- [x] (2026-07-27 12:42Z) Complete ten-step on-policy distillation. Run
  `339100a5-a4c2-4ae6-aa5a-1b080513b50e` performed ten real LoRA optimizer
  updates on `carbonteq-ai-workstation.lan` in 136 seconds with finite loss and
  gradient norms on every step, reconciled `consistent` without recovery,
  retained four artifacts, resolved through deployed Observatory with zero
  alerts, and cleaned up to 4,411 reclaimed bytes. Earlier characterization
  runs had rejected an invalid generation-batch relation, an undeclared teacher
  sidecar, and TRL distillation's missing LoRA-to-vLLM synchronization
  selection; the final blocker was a zero `num_items_in_batch` on fully
  on-policy steps, fixed in the TRL fork at
  `6e7739b8ec741d21ecd79c0c212694cd15ff20d8`. See
  `framework-oci-job-capsules-execution-log.jsonl` sequence 91.
- [x] (2026-07-27 06:23Z) Add the generic TRL distillation
  `vllm_weight_sync_mode` selection and close its order-dependent source-test
  failure. The failure was not a trainer defect: one vLLM-generation test
  leaked distributed-launch environment variables, a GRPO test consequently
  initialized NCCL, and the later CPU distillation test inherited that process
  group. Restoring the five variables after each leaking-module test makes the
  exact failure chain and the broader 15-case selection pass without changing
  production behavior. Fork publication, framework pinning, image rebuild,
  and the live ten-backward-pass gate remain open.
- [x] (2026-07-27 06:57Z) Run the complete formerly failing TRL order in one
  interpreter after the isolation fix. vLLM generation, GRPO, and experimental
  distillation reported 153 passed and 60 skipped in 23 minutes 52 seconds,
  with no failure or warning summary. The process left no initialized process
  group, CUDA context, distributed environment variable, Ray/test process, or
  GPU client. Repository-pinned Ruff formatting/checks, compile/import smoke,
  and diff checks pass. This closes the source release gate but not fork
  publication or the real ten-backward-pass qualification.
- [ ] For every remaining run, wait for terminal provider state, reconcile retained
  Trackio evidence, inspect the job-aware Observatory view, verify independent
  Doris counts, and perform exact evidence-gated workspace cleanup.
- [x] (2026-07-27 03:54Z) Exercise queued cancellation through durable
  per-worker admission. The waiting run was cancelled before submission and no
  dstack provider object was created.
- [x] (2026-07-27 03:54Z) Characterize running cancellation after two native
  Verifiers traces proved active rollout execution. Provider cancellation was
  immediate, Trackio remained `running`, and the explicit audited recovery
  restored consistent cancellation before exact workspace cleanup.
- [ ] Re-run running graceful cancellation after dstack propagates a non-zero
  selected stop duration through runner and shim termination; acceptance
  forbids tracking recovery. Pop-os attempts after release
  `371ff53b1d67f254bc6cc4259aae8653c3916b7d` confirm dstack grace (~5m) but not
  Trackio finalization: `ed9147ca-9efe-47c5-a5ff-c5181968fed1` (SAMPO,
  inconsistent succeeded/cancelled) and
  `37d2f98d-9d77-4b37-b78e-06d58a0a0cfa` (GRPO, Trackio stuck `running`).
- [ ] Remove the legacy directory-bundle and superseded operator-script path
  only after all qualification commands use the normal CLI.
- [x] (2026-07-27 03:58Z) Run the repository validation ladder: locked sync,
  Ruff, Pyright, eight import contracts, 648 passing Python tests with 16
  intentional skips, 19 passing frontend tests, TypeScript check, production
  frontend build, execution-log invariants, and `git diff --check`.

## Surprises & Discoveries

- Observation: shared kind-image qualification is not equivalent to public
  job-kind qualification.
  Evidence: `serve.benchmark` proves the serving image and vLLM runtime, but no
  retained run exercises the separate `serve.smoke` operation. The same gap
  exists for DPO, SAMPO, distillation, and data preparation.

- Observation: the final OCI path has strong SFT mechanics evidence but not a
  substantial SFT loop.
  Evidence: retained final-capsule SFT runs completed two optimizer updates;
  older fifteen-step SFT runs predate the final package/launch contract.

- Observation: framework-level SIGTERM handling is necessary but insufficient
  when the scheduler's container stop path supplies zero grace.
  Evidence: installed dstack 0.20.29 and current upstream `master` both use a
  fixed ten-second runner-stop interval before passing `timeout=0` to
  `terminate_task`; the selected five-minute stop duration does not reach that
  path. The current SAMPO capsule therefore left Trackio running even when
  cancellation occurred after two retained rollouts. Recovery and cleanup
  passed, but graceful running cancellation remains an infrastructure
  qualification gap.

- Observation: the CLI could drop required tracking bindings when a developer
  added `--env`.
  Evidence: before the fix, `--env HF_TOKEN` replaced
  `POSTTRAIN_TRACKIO_SERVER_URL` and `TRACKIO_WRITE_TOKEN`. Environment names
  are now additive and deduplicated.

- Observation: veRL cannot share either the developer environment or the
  framework control environment.
  Evidence: the release-blocked `online-rl-verl-py313` profile declares
  `/opt/posttrain/venv` as its Python 3.12 control environment and
  `/opt/posttrain-verl` as its separately locked Python 3.13.12 worker
  environment. Current qualification plans select the published TRL profile,
  never the blocked veRL profile.

- Observation: a pinned environment wheel can depend on a package already
  installed by its immutable kind image.
  Evidence: Alphabet Sort declares `verifiers`; the selected online-RL kind
  image already installs the exact Verifiers commit, but the environment
  dependency compiler attempted to emit the VCS requirement into a
  hash-required portable lock and rejected it. The fix must explicitly declare
  kind-provided packages and omit only those packages from emitted
  requirements while keeping all emitted dependencies hashed.

- Observation: `uv --require-hashes` still inspects dependency metadata when
  installing a fully expanded portable lock.
  Evidence: omitting the kind-provided `verifiers` line during compilation was
  insufficient because installing the Alphabet Sort wheel caused uv to inspect
  that wheel's VCS dependency. Actual-job installation now uses `--no-deps
  --require-hashes`; the framework compiler, not the installer, is responsible
  for expanding every transitive dependency and hashing every emitted line.

- Observation: infinite Verifiers tasksets do not implement finite sequence
  indexing.
  Evidence: Alphabet Sort exposes an infinite taskset, so the bridge now uses
  the environment's deterministic `select()` operation and verifies exact,
  unique task identities instead of calling `len()` and indexing.

- Observation: current-source Observatory needed first-class semantics for the
  remaining qualification matrix.
  Evidence: telemetry schemas, completeness rules, metric help, chart
  selection, and frontend labels now cover SAMPO, distillation, serving smoke,
  and data preparation without adding a second run-view API.

- Observation: cleanup made a successful local run appear lost on later
  reconciliation.
  Evidence: the provider object is intentionally absent after cleanup, while
  the append-only reconciliation journal retains the exact terminal provider
  record that authorized removal. Status and collection now reuse that record
  only when a matching cleanup receipt exists; a genuinely missing uncleaned
  execution still fails as lost.

## Decision Log

- Decision: interpret “each kind” literally for maintained public job kinds,
  not as one run per Docker kind-image family.
  Rationale: DPO and SFT share an image but exercise different data contracts,
  trainers, metrics, and artifacts. SAMPO, GRPO, and distillation similarly
  share online-RL dependencies while using different algorithms and evidence.
  Date/Author: 2026-07-27 / user and Codex.

- Decision: require ten backward passes for newly qualified training
  algorithms and retain the existing fifteen-step GRPO qualification.
  Rationale: one or two steps catch import and launch failures but do not
  exercise checkpoint cadence, repeated observation, memory stability, or
  optimizer behavior. Ten is the explicit minimum algorithm check; fifteen
  remains the proven GRPO reference.
  Date/Author: 2026-07-27 / user and Codex.

- Decision: parallelize across physical workers, never within one worker.
  Rationale: two independent machines can shorten qualification without
  introducing GPU-memory contention or making evidence attribution ambiguous.
  Each run still has one canonical run ID, provider handle, target, tracking
  run, and exact workspace.
  Date/Author: 2026-07-27 / user and Codex.

- Decision: do not claim a job kind qualified until provider termination,
  retained evidence, Observatory readback, direct Doris verification, and
  exact cleanup all agree.
  Rationale: a zero exit code alone does not prove observability, artifact
  retention, or safe worker cleanup.
  Date/Author: 2026-07-27 / Codex.

- Decision: qualify current SAMPO and distillation through
  `online-rl-trl-py312`, while keeping veRL publication as a separate release
  gate.
  Rationale: these checks qualify public algorithm behavior and the published
  runtime. They cannot establish reproducibility for the dirty, unpublished
  veRL candidate or its distinct Python 3.13 dependency environment.
  Date/Author: 2026-07-27 / user and Codex.

- Decision: treat the environment compiler's expanded hash lock as the complete
  dependency graph and install it with `--no-deps --require-hashes`.
  Rationale: dependency metadata may name packages intentionally supplied by
  the immutable kind image. Re-resolving that metadata during image build
  defeats the identity-bearing provided-package contract; bypassing hash
  verification would weaken it.
  Date/Author: 2026-07-27 / Codex.

- Decision: a completed cleanup receipt plus a matching pre-cleanup
  append-only reconciliation is the durable provider terminal authority.
  Rationale: provider objects and run workspaces are disposable by design.
  Cleanup must not destroy the ability to reconcile retained evidence, but a
  receipt must never mask an uncleaned lost execution.

- Observation: successful provider execution does not imply a research-ready
  view for every job kind.
  Evidence: ten-step SFT and DPO retained complete required systems telemetry
  and model artifacts, while their current schemas correctly leave
  `research_ready` false because these bounded mechanics checks do not include
  a held-out quality evaluation. Ten-step SAMPO retained all five required
  groups and both conditional groups and is research-ready for its declared
  systems question.

- Observation: one native trainer metric and one replayed trace metric can
  describe the same logical step with different statistical conventions.
  Evidence: SAMPO persisted live TRL sample-standard-deviation points and
  Verifiers replay population-standard-deviation points. The reader now
  preserves metric attributes, rebases replay to source step, selects trace
  replay as the reward-population authority, and collapses only numerically
  equivalent same-step duplicates.

- Observation: TRL's experimental distillation trainer constructs
  `VLLMGeneration` with full-weight synchronization even when the selected
  training update and inference binding both require LoRA synchronization.
  Evidence: the colocated teacher run reached student generation, then vLLM
  rejected the merged PEFT parameter set before the first backward pass. The
  generic fix belongs in the maintained TRL fork because the framework cannot
  select the already-supported `VLLMGeneration(weight_sync_mode="lora")`
  constructor through `DistillationConfig`.
  Date/Author: 2026-07-27 / Codex.

## Outcomes & Retrospective

The qualification matrix is now complete for GRPO, managed domain/general
evaluation, serving benchmark and smoke, model transformation, data
preparation, ten-step SFT, ten-step DPO, and ten-step SAMPO. Distillation is
not qualified: three failed runs have progressively removed two framework
configuration defects and isolated one missing generic TRL configuration
surface. That source surface and its order-dependent test isolation are now
corrected but remain unpublished and have not run the live ten-step gate.
Durable queued cancellation is qualified. Running cancellation remains
unqualified: the pre-release zero-grace shim defect is replaced by a
post-release split where dstack honors `stop_duration` (~304–316s on pop-os)
but Trackio either completes as `succeeded` during grace or stays `running`
after hard kill unless recovery is used. The final repository/deployment gates
also remain.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`. Framework-owned catalog
entries live under `packages/catalog/src/posttrain/catalog/base/`; project
qualification overlays live under `.posttrain/catalog/`; work packages live
under `.posttrain/work_packages/`. `apps/cli` composes a work package into a
`PlannedJobPackage`, publishes a `PackedJobPackage`, then composes one
`PlannedJobLaunch` and submits it through local Docker or dstack.

An immutable actual-job capsule is the final OCI image containing framework
code, project code, selected datasets, selected environment wheels, resolved
configuration, and a stable worker entrypoint. Launch identity such as run ID,
provider, retry policy, credentials, and host mounts is not part of capsule
identity. A bounded qualification is a real job with enough repeated work to
exercise its runtime without spending the budget of a full research run.

Two dstack workers are declared in the `local-gpu-workers` fleet. The 24 GiB
RTX 4090 worker is suitable for smaller LoRA/DPO/SFT and serving smoke jobs.
The 96 GiB RTX PRO worker is required for colocated online-RL and teacher/student
distillation jobs. Remote Trackio persists through Apache Doris. Observatory is
the read-only evidence product over Trackio.

## Plan of Work

First add qualification-only settings rather than changing reusable smoke
defaults. Use short sequence lengths, small prompt populations, LoRA, and
ten optimizer updates. Add a deterministic preference fixture for DPO so the
dataset is packaged inside the actual-job image. Add standalone work packages
whose metadata states the exact infrastructure question and expected evidence.

Second use `posttrain work-package validate` and the package-only planner to
prove all selections before any GPU is reserved. Pack each job through
BuildKit. The generated image must retain zstd compression, provenance, an
SBOM, exact source digests, and no credentials.

Third submit independent jobs to explicit target hostnames. Run SFT or DPO on
`pop-os.lan` while SAMPO or distillation uses
`carbonteq-ai-workstation.lan`. Do not schedule two jobs to the same worker.
Use `posttrain run list`, `posttrain run wait`, `posttrain run logs`,
`posttrain run reconcile`, `posttrain run show`, and `posttrain run cleanup`
instead of qualification scripts.

Finally verify direct Doris row counts using the protected operations path,
record non-secret results in
`docs/plan/framework-oci-job-capsules-execution-log.jsonl`, and remove the
superseded bundle/script route once all current-capsule paths have parity.

## Concrete Steps

Run all commands from `/home/hammad/projects/rl`.

    uv run posttrain work-package validate .posttrain/work_packages/<package>.yaml
    uv run posttrain --json job plan .posttrain/work_packages/<package>.yaml --job <job> --provider dstack --target <target>
    uv run posttrain --json job run .posttrain/work_packages/<package>.yaml --job <job> --provider dstack --target <target>
    uv run posttrain --json run list
    uv run posttrain --json run wait <run-id> --timeout-seconds 7200
    uv run posttrain --json run reconcile <run-id>
    uv run posttrain --json run show <run-id>
    uv run posttrain --json run cleanup <run-id>

After implementation, run:

    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Validation and Acceptance

A training job passes only when at least ten distinct optimizer-step
observations and a terminal summary are retained, required model/summary
artifacts reconcile, metrics are finite, and peak memory does not show
unbounded step-over-step growth. Online jobs must additionally retain their
native Verifiers traces and algorithm-specific metrics.

An evaluation, serving, transformation, or data job passes only when its
operation-specific result and traces or artifacts are retained and its
job-aware Observatory view is complete. Every remote run must have independent
Doris counts for configuration, metrics, system metrics, traces when
applicable, and artifact links.

Cleanup passes only after reconciliation and only when the provider removes or
proves absent the exact run workspace. Model caches, compile caches, sibling
runs, and retained Trackio artifacts must remain untouched.

The plan is complete when every maintained public job kind has one current
capsule qualification, the queue/cancel scenarios use current capsules, the
deployed Observatory renders the corrected eval and serving projections, and
the full repository validation ladder passes.

## Idempotence and Recovery

Package planning and packing are content addressed and may be repeated. Reusing
the same package plan should reuse the OCI publication receipt. Every live
retry must use a new run ID unless it is recovering the exact same idempotent
provider submission.

If a job fails before tracking starts, retain bounded provider diagnostics,
reconcile the failure, and run exact cleanup. If provider cancellation strands
a tracking run, use the audited `recover-cancelled-tracking` command only after
its exact identity checks pass. Never edit Doris directly. Never remove shared
cache roots to recover one failed qualification.

## Artifacts and Notes

The authoritative retained artifacts are submission receipts, provider
journals, reconciliation receipts, cleanup receipts, Trackio artifacts,
Trackio traces, direct Doris counts, Observatory views, OCI publication
receipts, provenance, and SBOM attestations. Credentials, raw environment
dumps, mutable caches, and temporary workspaces are not retained evidence.

## Interfaces and Dependencies

`apps/cli/src/posttrain_cli/execution_planning.py` owns project composition and
must expose `PlannedJobPackage`, `PackedJobPackage`, `PlannedJobLaunch`, and the
temporary `PlannedJobExecution` compatibility facade. `packages/execution`
owns provider-neutral submission storage, waiting, reconciliation, and
cleanup. `packages/execution-dstack` owns dstack SDK translation only.

Qualification settings remain catalog selections. Work packages bind those
selections to immutable standard definitions from
`packages/jobs/src/posttrain/jobs/definitions.py`. Training operations remain
in `packages/train`; the plan must not add training logic to the CLI or
infrastructure repository.

Revision note (2026-07-27): created after the strict completion audit showed
that image-family coverage was being mistaken for literal public-job-kind
coverage. The plan makes the remaining live evidence and cleanup gates
explicit.
