# ADR 0017 — dstack owns run-scoped storage and durable lifecycle hooks

## Status

Proposed.

Date: 2026-08-30
Deciders: Posttrain framework, dstack fork, and AI infrastructure maintainers
Related Plan: `docs/plan/dstack-r2-registry-routing.md`
Supersedes: None
Superseded By: None

## Context

Posttrain submits one immutable actual-job OCI image to dstack. The local site
currently runs that image on retained SSH-fleet workers prepared by the
`ai-infra` Ansible `worker` role. The first cloud backend is RunPod. RunPod is a
container-based dstack backend: its API creates the requested job container
directly, so dstack and Ansible do not receive a provider host on which they can
install Docker, drivers, certificates, systemd units, or host cache policy.

Cloud training still needs customization around allocation, persistent
checkpoint storage, service connectivity, terminal reporting, and cleanup.
RunPod spot pods are interruptible and can disappear without allowing the job
process to execute a final command. RunPod network volumes survive pod
termination, are bound to one Secure Cloud data center, and must be attached
when a pod is created. They therefore preserve checkpoints across a spot retry,
but also continue billing until explicitly deleted.

The maintained Trackio fork independently supports content-addressed,
resumable multipart artifact uploads to an S3-compatible store. Workers can
upload directly through short-lived presigned URLs, and Trackio commits an
artifact only after size and digest verification. This supplies durable and
cross-provider checkpoint publication, but it does not replace the low-latency
mutable workspace required to recover a spot attempt without first downloading
a multi-gigabyte checkpoint. The Trackio artifact store is separate from the
OCI registry and its R2-backed image publication path.

dstack already has useful pieces: a PostgreSQL-backed run/job/instance state
machine, retry classification for `INTERRUPTED_BY_NO_CAPACITY`, graceful stop
for reachable runners, provider volume create/delete methods, volume attachment
records, timed volume cleanup, an event table, plugins, and an SSH tunnel that
can be established before a job command starts. These pieces do not yet form a
run-scoped storage transaction or a stable runtime-hook contract. Existing
events are human-readable messages rather than typed delivery payloads, RunPod
loss is inferred from SSH unreachability rather than authoritative provider
state, and a run can only reference a separately created named volume.

Infrastructure-specific Ansible playbook names, Trackio URLs, certificates,
and shell fragments must not become dstack or Posttrain product fields. At the
same time, an operator needs a safe way to run site automation at selected
lifecycle transitions. User-supplied arbitrary shell inside the dstack server
would create an unacceptable remote-code-execution boundary.

## Decision

1. dstack will own a first-class **run-scoped network volume** lifecycle. A run
   may request persistent storage by size, mount path, and retention policy
   without naming a pre-created provider volume. A provider-backend policy may
   require this storage for interruptible/spot placement while leaving ordinary
   on-demand behavior unchanged. After dstack selects a backend offer and
   region, it creates the provider volume, records its provider ID and owning
   logical run in PostgreSQL, attaches it to every attempt of that run, and
   deletes it only after the logical run is terminal and its cleanup barrier is
   satisfied.

2. A spot interruption ends an **attempt**, not the logical run. dstack retains
   the run-scoped volume, constrains replacement capacity to the volume's RunPod
   data center, emits a typed `attempt.interrupted` event, and applies the run's
   retry policy. Success, failure after retry exhaustion, or user cancellation
   makes the logical run terminal.

3. dstack will add an optional provider-lifecycle capability. The RunPod
   implementation polls the provider record for each active pod and records
   `desiredStatus`, presence/absence, `interruptible`, provider timestamps, and
   safe lifecycle detail. Provider evidence takes precedence over generic SSH
   loss when classifying a disappeared spot pod. SSH loss remains a fallback
   when the provider API is unavailable.

4. dstack will persist typed lifecycle events and hook deliveries
   transactionally with the state transition they describe. Delivery is
   at-least-once. Every delivery carries a stable event ID and idempotency key,
   and every executor must be idempotent. A background dispatcher retries with
   bounded backoff and retains safe failure state; it never relies on an
   in-memory callback to establish terminal truth.

5. Hooks have two execution locations. A **control hook** runs outside the job
   and observes or prepares infrastructure. A **workload hook** is a bounded
   command executed through the existing dstack runner after the pod and any
   private tunnels are ready but before the user command starts. Workload hooks
   can initialize mount permissions or local configuration; they cannot change
   the provider host or rescue the OCI pull that created the pod.

6. dstack core defines hook events and an executor protocol, not ai-infra
   playbooks. The built-in production executor is a signed HTTP webhook. An
   optional server-admin command executor accepts a fixed argument vector from
   protected server configuration, never a project-supplied command and never
   `shell=True`. ai-infra owns a small hook runner that maps allow-listed hook
   IDs to either an `ansible-playbook` invocation or a fixed command. Secrets,
   playbook paths, inventories, and credentials stay in ai-infra.

7. Blocking hooks are limited to `before_provision` and `before_run`, have a
   declared timeout and fail-open/fail-closed policy, and are used sparingly.
   `attempt.interrupted`, `attempt.terminal`, `run.terminal`, and storage cleanup
   hooks are asynchronous notifications. No hook may rewrite image identity,
   provider identity, run identity, terminal status, or volume ownership.

8. Storage cleanup is a durable barrier, not a terminal shell hook. For a
   terminal logical run, dstack first detaches/terminates provider compute,
   dispatches required finalization hooks, and waits for their terminal delivery
   state or configured deadline. It then marks the volume for deletion, invokes
   the provider delete API idempotently, confirms absence, and records a cleanup
   receipt. Spot-interrupted attempts never cross this barrier. User cancellation
   does cross it after the bounded finalization window.

9. dstack will expose a consolidated inventory read model without conflating
   owned capacity with provider offers. The inventory contains configured
   backends, retained fleets and instances, active allocations, provider offers
   with freshness/provenance, run-scoped volumes and cleanup deadlines, and
   failed hook deliveries. Unknown or stale availability is shown as unknown or
   stale, never as zero or available.

10. Posttrain remains provider-neutral. It may request a provider-neutral
    persistent workspace and retry/retention policy and must record attempts in
    evidence, but it does not name RunPod, Ansible, shell hooks, ai-infra
    services, or provider volume IDs. Trackio remains an observer. A generic
    hook receiver may reconcile dstack attempt/run status into Trackio, but
    dstack does not import or call Trackio-specific code.

11. Checkpoint recovery has two tiers. The training runtime atomically writes
    frequent recovery checkpoints to the mounted workspace and publishes
    selected completed checkpoints through the existing Trackio artifact API.
    The run-scoped volume is the first recovery path for another attempt in the
    same data center. A Trackio artifact becomes the durable recovery pointer
    only after Trackio verifies and commits it; it is the fallback when the
    original volume or its data-center capacity cannot be reused. dstack owns
    volume lifecycle and attempt placement, while Posttrain and Trackio own
    checkpoint meaning, publication, verification, and lineage.

12. Exactly one live attempt may write a run-scoped mutable volume. dstack must
    provider-confirm the previous pod absent or fence it before starting a
    replacement writer. A cross-data-center retry is explicit: dstack creates a
    new volume, the runtime restores the latest Trackio-verified recovery
    artifact, and the run records that work newer than that durable pointer may
    have been lost.

13. Retry budgets are event-specific and compact. Capacity admission lasts 24
    hours from initial submission. Provider-confirmed interruption recovery
    lasts two hours from the first interruption, never resets after a
    replacement provisions, and permits at most five recovery actions.
    Runtime errors remain fail-fast. dstack persists per-event counters and
    first-event timestamps on the logical run so pruning old submission rows
    cannot reset the budget. Pending retry backoff retains the exponential
    15s/30s/1m/2m/5m/10m base schedule with stable plus-or-minus-20-percent
    jitter. Pre-provision storage-region failures cool down for ten minutes.

## Consequences

Checkpoints survive an interrupted RunPod spot attempt and can be reused by a
replacement pod without making shell scripts responsible for storage. Trackio
provides a separately verified durable copy for ordinary workloads and for
restoration when the original RunPod data center cannot supply replacement
capacity. Cancelled and completed runs have a bounded, inspectable cleanup
transaction, reducing the risk of indefinitely billed network volumes.
Operators can integrate Ansible and custom commands while dstack retains a
generic, testable contract.

The dstack fork gains database migrations, new lifecycle workers, RunPod API
polling, an inventory API/CLI surface, and more state-machine tests. Placement
becomes region-aware after volume creation: once a run-scoped RunPod volume
exists, retries trade cross-region availability for checkpoint continuity. The
normal path reuses that volume; an explicit portability fallback may allocate a
new volume elsewhere and resume from the older Trackio-verified durable
recovery pointer.

A sudden spot eviction cannot guarantee an in-process final checkpoint or final
Trackio call. The authoritative report is the dstack server's provider-observed
`attempt.interrupted` event. Training code must checkpoint periodically to the
network volume and publish selected completed checkpoints independently. The
hook receiver can reconcile the attempt as interrupted and keep the logical run
open while dstack retries. Recovery-point objectives are therefore bounded by
local save cadence for same-volume retry and verified Trackio publication
cadence for cross-data-center retry.

At-least-once delivery means hook receivers must deduplicate. A failed required
finalizer can delay cleanup only until its configured deadline; afterward the
system records a dead-letter/cleanup warning and follows the explicit retention
policy rather than leaking storage forever.

## Alternatives Considered

### Run Ansible against every RunPod pod as if it were a VM

Rejected because RunPod does not expose a stable provider host lifecycle. The
job image is pulled before dstack can connect, and changes inside the container
disappear with the pod. Workload hooks cover bounded container initialization;
host preparation remains impossible on this backend.

### Let hook scripts create and delete provider volumes

Rejected because a script cannot atomically coordinate provider resources with
dstack's retry, cancellation, and terminal state. Failures between script
execution and state persistence would leak or prematurely destroy storage.

### Pre-create one permanent shared RunPod volume

Rejected as the only model because concurrent writers can corrupt shared state,
cleanup and cost attribution become ambiguous, and every job becomes pinned to
one data center. Named long-lived volumes remain supported for deliberately
shared datasets or caches; mutable job checkpoints default to run-scoped
volumes.

### Report spot death only from the job process

Rejected because an interruptible pod may be stopped at any time and may not run
a shutdown handler. The provider-observed server transition is authoritative;
the job's own final signal is useful only when available.

### Call Trackio directly from dstack

Rejected because it would couple a generic scheduler fork to one observation
backend. Typed hooks and stable correlation fields allow ai-infra/Posttrain to
perform reconciliation without changing dstack's ownership.

### Use only the RunPod network volume

Rejected because a network volume is tied to one data center and is not an
off-provider durable backup. Same-data-center capacity loss would strand the
run even though a checkpoint exists, and deleting the volume after terminal
cleanup would also delete the only recovery copy.

### Use only Trackio artifact publication for spot recovery

Rejected as the default spot path because serialization and multi-gigabyte
upload completion widen the interruption loss window and make every retry pay a
restore delay. Trackio remains sufficient for ordinary durable checkpoint
publication; the network volume adds fast mutable continuity for spot attempts.

### Treat catalog offers as live fleet inventory

Rejected because the current RunPod adapter marks catalog offers available
without a provider availability check. Inventory must distinguish retained
capacity, active allocations, catalog candidates, and live provider evidence.

## Implementation Notes

The initial dstack interfaces should be introduced in the maintained fork:

- `Compute.observe_instance()` returns a typed provider lifecycle observation;
  RunPod implements it from the pod API's `desiredStatus`, `interruptible`, and
  provider timestamps.
- `RunScopedVolumeSpec` adds `size`, `path`, and `retention` to run configuration
  without exposing a provider volume ID. A new database relation owns the
  generated `VolumeModel` from logical `RunModel` and mount key.
- `LifecycleEventModel` and `HookDeliveryModel` form a transactional outbox.
  Event kinds are stable enums; payloads are versioned, redacted JSON.
- `LifecycleHookExecutor` receives a delivery and returns a typed outcome.
  Webhook delivery uses a signature, timestamp, event ID, timeout, and bounded
  retry. The command executor is server-admin-only and passes event JSON through
  stdin or a mode-protected temporary file.
- `JobServerConnection` is the existing seam for optional reverse-forwarded
  private services. It is established before `_submit_job_to_runner`; workload
  hooks execute only after that connection succeeds.
- `GET /api/project/{project_name}/inventory` and `dstack inventory` return
  separate `capacity`, `allocations`, `storage`, and `automation` sections with
  `observed_at`, `source`, and `freshness` on every provider-derived record.

The first ai-infra hook runner must have an allow-list such as
`runpod-before-run`, `attempt-terminal-report`, and `run-terminal-finalize`.
Configuration maps those identifiers to repository-owned Ansible playbooks or
fixed commands. Event payload values are data, never executable command text.

Real qualification must force one bounded RunPod spot interruption, observe the
typed provider event, confirm retry with the same volume and region, verify a
checkpoint marker survives, and verify a selected checkpoint is committed by
Trackio before terminal volume deletion. A portability test restores that
verified artifact into a new volume without reusing the first volume. Finish or
cancel the logical run and prove both pod and run-scoped volume are absent
afterward. A second test kills the dstack server between the terminal transition
and hook delivery and proves delivery and cleanup resume after restart.

## Revision History

- 2026-08-30: Initial proposed decision. Reason: define provider-neutral hooks,
  RunPod spot recovery, run-scoped persistent storage, cleanup, and truthful
  inventory before extending the maintained dstack fork.
- 2026-08-30: Added two-tier checkpoint recovery. RunPod network volumes own
  fast same-data-center spot continuity; Trackio owns verified durable and
  cross-data-center recovery artifacts, independently of the OCI R2 registry.
