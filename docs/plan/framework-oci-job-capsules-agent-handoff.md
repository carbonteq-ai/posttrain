# Handoff: publish the validated OCI job-capsule release

This is a point-in-time handoff for the next agent. The authoritative living
plan is `docs/plan/framework-oci-job-capsules.md`, and its append-only evidence
stream is
`docs/plan/framework-oci-job-capsules-execution-log.jsonl`. Update those two
files as work proceeds. Do not let this summary become a competing plan.

This handoff was reconciled against the local repositories, GitHub remotes, and
live dstack fleet on 2026-07-27 at 07:08Z.

## Objective and stopping point

Publish the already validated, scoped source release across the maintained
forks, the post-training framework, and ai-infra:

1. Publish the Trackio, TRL, dstack, and veRL candidate commits.
2. Replace framework candidate references with exact immutable fork commits and
   locks.
3. Commit and push the framework as a reviewed, ordered release series.
4. Commit and push the first private ai-infra source release.
5. Stop before building, deploying, restarting services, replacing dstack
   components, or submitting a new GPU job.

The current authorization explicitly stops before deployment. Do not interpret
“the repositories exist” or “the source tests pass” as authority to promote
services. If the user later supersedes that boundary, deployment and live
qualification resume from the final section of this handoff.

## Product and repository boundary

The framework repository owns work-package/job selections, the
`JobPackageManifest`, the launch envelope, deterministic code/environment/data
packing, universal and job-kind image definitions, actual-job images, local
and dstack execution providers, the runtime worker, CLI/SDK lifecycle,
reconciliation, cleanup, and training-time observations.

The ai-infra repository owns the dstack control service and worker attachment,
registry and BuildKit availability, Trackio/Doris/Observatory/object-storage
deployment, DNS/TLS/backups/secrets, NVIDIA/Docker worker setup, persistent
cache/checkpoint/run volumes, generic service health, and deployment receipts.
It must not acquire framework job definitions, Verifiers environments,
datasets, or algorithm policy.

The worker contract is image based. One universal base feeds exact job-kind
runtime variants; an actual-job image adds selected code, configuration,
environment Git packages, and materialized dataset bytes. dstack receives the
final immutable image digest and launch-time values. It does not receive a
second normal code/data bundle. Trackio configuration is injected by the
execution target; Doris is private to the Trackio server and is not known by
training jobs.

## Current progress

- The universal base, job-kind variants, deterministic source/environment/data
  packer, actual-job image, runtime entrypoint, local and dstack providers, CLI
  lifecycle, durable admission, terminal marker, reconciliation, and cleanup
  paths are implemented.
- The latest complete framework source gate passed 665 tests with 16
  intentional skips, Ruff, Pyright, all eight import contracts, the static
  actual-job validator, ordered/unique execution-log validation, and
  `git diff --check`.
- Real capsule qualification is complete for local SFT and dstack-hosted SFT,
  DPO, SAMPO, fifteen-update GRPO, managed general and multi-environment
  evaluation, data preparation, serving smoke/capacity, and transformation.
- The RTX PRO worker completed a two-update Qwen 3.5 2B LoRA SFT and two
  separate fifteen-update GSM8K GRPO runs. Trackio backed by Doris retained the
  metrics, native Verifiers traces, adapters, checkpoints, summaries, and
  artifact links; exact-worker cleanup removed run workspaces while retaining
  shared caches and durable evidence.
- Distillation remains unqualified live. The TRL source fix passed its release
  gate, but a realistic ten-backward-pass run is still required after
  publication, image promotion, and deployment.
- Queued cancellation is qualified. Running cancellation remains unqualified
  because the deployed dstack component set still uses the faulty grace path.
  The candidate source propagates `stop_duration`, but it must eventually be
  deployed as one matching server/runner/shim set before the greater-than-ten
  second finalizer test.
- The corrected Trackio post3/Doris source and latest Observatory projections
  are not deployed. Existing retained runs prove the current remote evidence
  path, not promotion of the new candidate.

## Live infrastructure snapshot

The dstack fleet `local-gpu-workers` is active. Both nodes were healthy,
reachable, and idle at handoff:

- `pop-os.lan`: 24 CPUs, 64,036 MiB host memory, RTX 4090 with 24,576 MiB
  VRAM.
- `carbonteq-ai-workstation.lan`: 32 CPUs, 60,856 MiB host memory, RTX PRO
  6000 Blackwell Workstation Edition with 98,304 MiB VRAM, and 1,761,833 MiB
  worker disk.

The RTX PRO host is already generic dstack worker infrastructure. Do not copy
the framework checkout or developer virtual environment onto it. A future
submission pulls the actual-job image by digest and mounts
`/var/lib/posttrain` cache/run storage. The new dstack graceful-cancellation
release has not been installed on the control service or worker.

Read-only fleet verification, without printing the protected environment file:

    cd /home/hammad/projects/ai-infra
    set -a
    . .state/dstack/client.env
    set +a
    .venv/bin/dstack fleet --project main get local-gpu-workers --json

## Repository snapshot

### Trackio

Path: `/home/hammad/projects/trackio`

Current branch/base:
`codex/resumable-artifact-transport` at
`e2784c1536b20832f3937d7589c10bce76df4b43`.

Target branch: `codex/doris-storage-post3`.

The dirty candidate contains the first-class Apache Doris storage provider,
schema negotiation and migration, SQLite/Turso parity, model-artifact
transport, fork ledger updates, and the remediated Vega/Vite/Svelte dashboard
surface. The focused post3 suite passed 93 tests; the non-hardware unit suite
passed 376 with 3 skips; the real-Doris suite passed four cases. Frontend lint,
48 tests, production build, clean wheel build, and production/full npm audits
with zero vulnerabilities passed. Seven candidate files were formatted; nine
pre-existing unrelated format-only files remain out of scope.

No `codex/doris-storage-post3` branch exists on the remote.

### TRL

Path: `/home/hammad/projects/trl`

Current branch/base:
`codex/dapo-dynamic-sampling` at
`b43a0a3d622ab1547f4d2abbd1b25eab3c52a0b9`.

Target branch: `codex/distillation-lora-sync`.

The dirty candidate is five files: `CARBONTEQ_FORK.md`, the distillation
config/trainer, and the distillation/vLLM regression tests. It adds the
explicit vLLM weight-synchronization mode required by LoRA distillation. The
exact combined test order passed 153 tests with 60 skips in 1,432.67 seconds;
Ruff format/check and import smoke passed; no distributed process group, CUDA
client, or worker process remained.

No `codex/distillation-lora-sync` branch exists on the remote.

### dstack

Path: `/home/hammad/projects/dstack`

Current state: detached at upstream
`2f9618f4d521140350efd1b344412d122c1e0322`.

Target branch: `codex/graceful-cancellation-stop-duration`.

The dirty candidate is limited to server-to-runner-to-shim `stop_duration`
propagation, its Python/Go schemas and tests, and `CARBONTEQ_FORK.md`. The
SQLite server gate passed 83 cases with 21 PostgreSQL variants skipped, the
PostgreSQL gate passed 104 cases, and the Go executor/schema gate passed 35
top-level cases.

`carbonteq-ai/dstack` now exists as a public GitHub fork, but this checkout
still names upstream dstack as `origin`. Before pushing, rename that remote to
`upstream` and add `git@github.com:carbonteq-ai/dstack.git` as `origin`. No
target branch exists on the CarbonTeq remote.

### veRL

Path: `/home/hammad/projects/verl-release-candidate`

Current state: detached at published CarbonTeq `origin/main`
`553280b88afe4e7fbc4aefeff27bbf0a22e7c048`.

Target branch: `codex/runtime-release-qwen35`.

This is the cleanly reconstructed candidate, not the older dirty
`/home/hammad/projects/verl-upstream` research checkout. It layers dependency
compatibility, LoRA pre-wake staging, dense-FSDP entropy chunking, Qwen 3.5
attention dispatch, response-token totals, and step-local MTP telemetry over
the published CarbonTeq base. The exclusion audit found no runtime environment,
`sitecustomize`, TurboQuant bootstrap/compatibility/test, package hook, or
TurboQuant-only vLLM server change. It passed 42 focused CPU tests, the
34-test SAMPO/core regression, and five focused RTX 4090 source tests.

No `codex/runtime-release-qwen35` branch exists on the remote. Publication does
not qualify the runtime: the Python 3.13 dependency-only lock, kind image,
cross-interpreter smoke, and live workload remain later gates.

### Post-training framework

Path: `/home/hammad/projects/rl`

Current branch/base:
`codex/serving-capacity-observatory` at
`5b81cdabdbb77297c483955f0853439bf197c279`.

The worktree is intentionally large and dirty. Do not stage it wholesale. The
release audit requires an ordered series grouped by:

1. framework-neutral execution contracts, durable lifecycle, and runtime;
2. deterministic packing, BuildKit integration, image definitions, and locks;
3. local/dstack providers, project configuration, CLI, and developer DX;
4. data/eval/serve/train/job capabilities and qualification definitions;
5. Trackio adapters, Observatory projections/UI, generated schemas, and tests;
6. tooling ledgers, plans, execution evidence, and exact dependency pins.

Update Trackio, TRL, dstack, and veRL references only after their commits are
reachable on the CarbonTeq remotes. Regenerate `uv.lock` from those immutable
commits and run the full validation ladder after the series is assembled.

Do not include ambient-agent research artifacts, failed TurboQuant research,
the old upstream-veRL qualification tool, large editor artifacts, secrets,
machine-local `.posttrain/state`, or stale deployment notes merely because
they are present in the dirty tree. Preserve every unrelated user change.

### ai-infra

Path: `/home/hammad/projects/ai-infra`

Current state: unborn `main`; no commit and no Git remote configured.

`carbonteq-ai/ai-infra` exists as a private, empty GitHub repository. The
prepublication audit found 161 intended source files, zero ignored-sensitive
path leaks, and zero high-confidence secret signatures. Runtime state,
credentials, OpenTofu state/backups, virtual environments, caches, and real
cluster auto-tfvars remain ignored.

After reviewing the ignored boundary again, add
`git@github.com:carbonteq-ai/ai-infra.git` as `origin`, create the initial
commit, and push `main`. Do not add `.state`, credential files, generated
backups, or host-specific secrets.

## Required publication sequence

### Milestone 1: refresh evidence without changing scope

For every repository, record branch, base commit, status, staged paths, remotes,
and `git diff --check`. Re-run the already documented bounded release gate when
the candidate bytes differ from the validated snapshot. Do not mix changes
from sibling repositories or “clean up” unrelated work.

### Milestone 2: publish maintained forks

Create the four target branches from their current reviewed bases, update each
root `CARBONTEQ_FORK.md`, commit only the candidate paths, push, and verify the
remote head with `git ls-remote`. Capture each immutable commit in the main
plan and append one event per repository to the execution log.

For dstack, fix remote naming before branch creation. The public fork's default
branch is currently `master`; do not force-push or rewrite it. Push only the
new topic branch.

### Milestone 3: pin and publish the framework series

Replace candidate fork references with the four published commits, regenerate
exact locks, and ensure the veRL profile remains blocked until its dependency
lock/image/runtime gates actually pass. Assemble the scoped framework commits
in dependency order. At minimum run:

    cd /home/hammad/projects/rl
    uv sync --all-packages --locked --python 3.12
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Also run the Observatory frontend lint, tests, TypeScript check, production
build, and Playwright journeys documented in the main plan when its files are
part of the series. Validate that every line in the execution log parses and
that sequences and event IDs are ordered and unique.

### Milestone 4: publish ai-infra source and stop

Repeat the secret/path audit, review the full staged list, make the initial
private commit, push `main`, and verify repository visibility remains private.
Update the living plan and execution log with exact commit heads and validation
results.

Stop here. Report the fork heads, ordered framework commits, ai-infra head,
remote verification, and any deliberately excluded paths. Do not build images
or deploy.

## Deployment work that remains after the stop

This section is context, not current authorization.

After explicit deployment approval, publish immutable Trackio, Observatory,
framework runtime/kind, veRL, and dstack server/runner/shim artifacts. Promote
the dstack server, runner, and shim as one versioned component set. Recheck both
worker nodes, then run:

- realistic distillation with at least ten backward passes;
- the veRL Python 3.12 control/Python 3.13 worker cross-interpreter image smoke
  and bounded GPU qualification;
- running cancellation whose tracking/artifact finalizer lasts more than ten
  seconds but completes inside the configured `stop_duration`;
- authenticated Trackio/Doris and Observatory readback;
- exact policy-driven worker cleanup while retaining results.

Do not claim the overall runtime release complete until those retained results
are appended to the execution log and the living plans reflect the observed
outcomes.

## Progress

- [x] Reconciled the plan, evidence log, six local repositories, four target
  remote branches, two GitHub repository creations, and live dstack fleet.
- [x] Recorded the validated source and prior live-GPU evidence.
- [ ] Publish the four maintained-fork candidates.
- [ ] Pin and publish the scoped framework series.
- [ ] Publish the private ai-infra initial commit.
- [ ] Stop before deployment and provide immutable heads.

## Surprises & Discoveries

- Observation: the RTX PRO workstation is already healthy worker capacity, but
  that does not mean the new dstack component set is deployed.
  Evidence: the live fleet reports the node healthy and idle while the dstack
  candidate remains detached, uncommitted, and absent from the remote.
- Observation: the source candidates passed release gates on top of older local
  topic branches or detached commits.
  Evidence: none of the four intended target branches exists remotely.
- Observation: repository creation and source publication are independent.
  Evidence: the public dstack fork and private ai-infra repository exist, while
  ai-infra remains unborn and the dstack candidate has no CarbonTeq remote in
  its checkout.

## Decision Log

- Decision: publish forks before pinning the framework.
  Rationale: framework locks must refer to commits already reachable from their
  configured remotes.
- Decision: split the framework into dependency-ordered commits rather than
  committing the dirty worktree wholesale.
  Rationale: the worktree contains multiple logical surfaces and unrelated or
  deliberately excluded artifacts.
- Decision: stop after source publication.
  Rationale: deployment mutates shared services and workers and was explicitly
  excluded from the current authorization.

## Outcomes & Retrospective

The design, implementation, source validation, and broad live capsule
qualification are substantially complete. The immediate work is release
engineering: turn the validated dirty candidates into immutable, reviewable
remote commits without losing scope discipline. The system is not yet a
complete promoted release because distillation, veRL runtime, corrected
Trackio/Observatory, and running graceful cancellation still depend on later
artifact publication, deployment, and retained live evidence.
