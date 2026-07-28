# Operate the internal dstack and Trackio cluster

Status: architecture and qualification specification. O0 read-only discovery
is conditionally complete. No service has been installed, no worker has been
enrolled, and no UniFi, Unraid, VM, container, or DNS setting has been changed
by this work.

## Goal

Use the always-on Unraid server as the physical services host. Create dedicated
`ai-control` and `ai-doris` Ubuntu VMs rather than relying on the existing
Dokploy guest. Run dstack, Trackio, a private OCI registry, ingress, and
operational monitoring on `ai-control`; isolate Apache Doris on `ai-doris`;
use Trackio as the internal artifact registry with a private server-managed
Hugging Face Storage Bucket for bytes. Add a local `ai-storage` VM only when
capacity, offline operation, or data-residency requirements justify it. Attach
both upgraded GPU workstations as dstack workers. Develop source and construct
jobs on the developer workstation; send only immutable workload packages to
execution targets.

This topology supports one measured experiment at a time today without closing
the path to concurrent screening or additional workers later.

## Selected stack

| Concern | Selection | Boundary |
| --- | --- | --- |
| Infrastructure scheduling | dstack server and SSH fleets | Places, starts, monitors, retries, cancels, and drains accepted workloads |
| Experiment policy | project goal runner | Selects the next experiment and holds the global serial-experiment lease |
| Experiment evidence and artifact registry | self-hosted Trackio, Apache Doris, plus Observatory | Trackio owns versioned artifacts and control metadata, Doris holds analytical projections, and Observatory presents read-only research views |
| Trackio artifact blobs | Private Trackio-managed Hugging Face Storage Bucket initially | Stores content-addressed model, checkpoint, trace, and result bytes; workers never receive its credentials |
| Workload transport | private OCI Distribution registry | Stores immutable images addressed by digest |
| Scheduler state | PostgreSQL | Durable dstack state; SQLite is allowed only for a disposable bootstrap |
| Service ingress | UniFi `.lan` DNS plus Caddy or Traefik on `ai-control` | Exposes authenticated service endpoints only to approved internal networks |
| Service deployment | Ansible; pinned Docker Compose for control services and native Doris services | Reproducible, reviewable deployment without a platform UI dependency |
| Hardware operations | Beszel initially; Coroot or SigNoz when distributed tracing is justified | Retains host/container health independently of experiment metrics |
| Recovery executor | SSH, user systemd, and the same OCI image | Break-glass path, not the normal scheduler |

Begin with CNCF Distribution because it is the smallest standards-based private
registry. Promote to Harbor only when team-scoped RBAC, robot-account lifecycle,
replication, or integrated scanning justifies the additional service surface.

dstack is the infrastructure scheduler, not the research controller. It may run
different independent jobs concurrently in the future, but the Ambient Agent
goal runner must currently enforce exactly one active measured experiment
across both workers. Trackio is evidence storage, not a queue or scheduler.

## Topology

```text
local development machine
  ├─ builds and tests source
  ├─ resolves the framework job and execution-target selection
  ├─ builds and pushes an image by digest ───────────────┐
  └─ submits a generated dstack task ────────────────┐  │
                                                     │  │
Unraid services host                                │  │
  ├─ ai-control VM                                   │  │
  │   ├─ dstack server/API/UI <──────────────────────┘  │
  │   ├─ PostgreSQL                                     │
  │   ├─ Trackio control API/UI                         │
  │   ├─ OCI registry <─────────────────────────────────┘
  │   ├─ Observatory
  │   ├─ Beszel
  │   └─ Caddy or Traefik TLS ingress
  │       │
  │       ├─ SSH fleet worker: local RTX 4090
  │       └─ SSH fleet worker: RTX PRO 6000
  └─ ai-doris VM
      └─ native Doris FE/BE <──── metrics/traces/events

private Hugging Face services
  ├─ Storage Bucket <──────────── Trackio server only
  └─ model repositories ───────── pinned foundation-model inputs
```

Use separate internal names, even when they initially resolve to one machine:

- `dstack.lan`
- `trackio.lan`
- `registry.lan`
- `observatory.lan`
- `beszel.lan`
- `doris.lan`
- `s3.lan` only if a local object-store adapter is later deployed
- `storage.lan` only if a local object-store console is later deployed

The exact reservation and DNS proposal is recorded in
[`unraid-o0-inventory.md`](unraid-o0-inventory.md). Do not create records until
both initial VM addresses are selected, confirmed unused, and fixed in UniFi.

Co-location is acceptable for the initial non-HA target, but each service
gets an independent volume, credential, health check, retention policy, and
backup. This permits later movement without changing framework run identity or
job semantics.

## Service-host contract

Before deployment, record the services host's hostname, stable address, OS,
CPU, RAM, available durable disk, filesystem, backup destination, Docker
version, internal DNS support, and TLS issuer. Confirm bidirectional network
reachability between the services host and both workers on only the required
ports.

The initial service deployment must provide:

- dstack behind TLS, with PostgreSQL state and AES encryption configured before
  any worker, registry, model-provider, or Trackio secret is stored;
- Trackio behind TLS, with its write token injected into jobs and `TRACKIO_DIR`
  or its successor control-store configuration on durable storage;
- Apache Doris in its own VM and NVMe-backed data volume, receiving only an
  idempotent analytical projection from Trackio;
- Trackio artifact versioning and lineage backed by a private Hugging Face
  Storage Bucket whose credentials remain on the Trackio server, as specified
  in [`object-storage.md`](object-storage.md);
- a TLS-protected OCI Distribution registry with authenticated push and
  pull-only worker identities;
- Observatory configured as a read-only consumer of Trackio;
- Beszel for the initial host/container dashboard and alerting; add Coroot,
  SigNoz, or an NVIDIA-specific collector only when the measured diagnostic
  requirement justifies it;
- host-level backup jobs and a tested restore procedure for PostgreSQL,
  Trackio control data and retained artifact bytes, Doris data, registry
  manifests/blobs, configuration, and encryption keys.

Do not expose PostgreSQL, Doris internode/database ports, raw Trackio storage,
object-storage administration ports, registry storage, or Docker sockets to
the LAN. Do not store real values in compose files. A future implementation
should provide `.env.example` files containing names only and load real values
from host-protected files or a secret manager.

## Attach a GPU machine

Enrollment is an explicit, reversible operation per worker:

1. Run a read-only preflight for stable hostname/address, supported Linux,
   available CPU/RAM/disk, GPU model and VRAM, NVIDIA driver, Docker, CUDA
   compatibility, NVIDIA Container Toolkit, time synchronization, DNS, and
   service-host reachability.
2. Create a dedicated `dstack-worker` service account. Generate a dedicated
   fleet SSH key on the services host and install only its public key on the
   worker. A password may bootstrap the account but is not retained or used by
   normal scheduling.
3. Meet the current dstack SSH-fleet prerequisites: Docker, CUDA 12.1
   compatibility for NVIDIA workers, NVIDIA Container Toolkit, passwordless
   sudo for the dedicated account, and SSH TCP forwarding.
4. Restrict SSH ingress to the services host where the network permits it.
   Do not reuse a personal or repository-held administrative key.
5. Add the host to a reviewed dstack fleet definition, apply it, and verify the
   fleet and offer inventory before accepting work.
6. Run a CPU no-op, an OCI pull by digest, and a bounded CUDA tensor job. Record
   the observed GPU model, VRAM, driver, image digest, exit status, logs, and
   cleanup result.
7. Mark the worker qualified only after cancellation, disconnect/reconnect,
   duplicate submission, failure collection, Trackio reporting, and run-scoped
   cleanup pass.

Draining a worker prevents new placement, waits for or explicitly cancels its
active task, verifies terminal evidence and cleanup, then removes the fleet
entry. Uninstalling Docker or deleting caches is a separate destructive action
and is never implied by detaching a worker.

## Package and schedule a job

The transport unit is a small content-addressed job capsule plus an immutable
OCI image. The capsule contains identifiers and references, not model weights,
dataset copies, caches, or secrets. It records:

- schema version, submission ID, project, work package, job, and run attempt;
- exact source and dependency revisions;
- image repository and digest;
- resolved model, dataset, environment, training, evaluation, and
  execution-target selections;
- command, resource request, timeout, retry class, retention policy, and
  required produced-artifact roles;
- references and digests for large immutable inputs.

The local packager validates the capsule, builds the image from a clean locked
source revision, scans it for obvious secret material, pushes it, and verifies
the registry digest. The scheduler adapter then generates a dstack task with:

- the immutable image digest and `registry_auth` from encrypted dstack secrets;
- GPU count/model/memory plus CPU, RAM, disk, fleet, and instance constraints;
- explicit maximum duration and priority;
- retries limited to classified infrastructure failures;
- Trackio endpoint/project variables and an injected write token;
- no object-store variables or bucket credentials; Trackio owns those
  server-side; and
- framework submission metadata.

Bounded task data ships with the versioned environment/project code. Base
models remain immutable Hub references and may use a bounded worker cache.
Result models are uploaded through Trackio and returned as pinned
`TrackioArtifactRef` values. Run-scoped scratch and non-retained recovery state
are finalized only after required Trackio artifacts pass read-after-write
verification.

The project controller submits with a stable idempotency key. It maps:

```text
submission_id
  -> dstack project and run/job ID
  -> framework run_id and run_attempt
  -> Trackio provider run ID
  -> versioned Trackio artifact references
```

Normalize provider states to `accepted`, `queued`, `starting`, `running`,
`succeeded`, `failed`, `cancelled`, or `lost`. Keep the native dstack state and
diagnostic alongside the normalized value. A retry receives a new attempt
identity while retaining the original submission lineage.

## Observe and learn

Use each surface for one question:

- dstack UI/API: Where is the job, why is it queued, and is the worker healthy?
- Trackio and Observatory: What did the experiment learn, and can its evidence
  and lineage be compared with earlier runs?
- Beszel: Is the services host or GPU fleet degrading over hours or weeks?
- the project execution log: What happened before a framework run existed,
  including packaging, registry transfer, queueing, infrastructure failure,
  cancellation, collection, and cleanup?

dstack's recent CPU/RAM/GPU view is useful for job operations but is not the
long-term metrics store. Do not duplicate training metrics into dstack. Every
attempt reconciles the identities above so later autoresearch can learn from
successful runs and failed submissions alike.

## Security, retention, and recovery

- Keep service and worker credentials out of Git, images, capsules, commands,
  logs, traces, and Trackio metadata.
- Use a dedicated SSH key, registry pull-only credentials on workers, scoped
  push credentials for builders, and a scoped Trackio write token for jobs.
- Back up dstack state, Trackio control data, Doris, registry data, service
  configuration, and encryption keys separately. Encrypt backups and test
  restoration.
- Give registry images, scheduler logs, Trackio evidence, shared model caches,
  and run scratch independent retention policies.
- Never recursively delete an unresolved path, home directory, repository
  root, shared cache, or another run. Cleanup is run-scoped, dry-run capable,
  idempotent, and recorded as a Trackio event plus provider receipt status.
- Keep SSH/systemd/OCI documented and qualified as the recovery path for a
  dstack outage; it consumes the same capsule and image and produces the same
  normalized result contract.

## Delivery sequence

### O0 — Inventory and freeze inputs

Observe the services host and both workers. Resolve DNS/TLS, ports, storage,
backup, service accounts, registry choice, and version pins. Produce a redacted
inventory and a go/no-go report. Make no service changes. The current Unraid
result is a conditional go in
[`unraid-o0-inventory.md`](unraid-o0-inventory.md); new VM sizing, fixed
addresses, and exact version pins remain open.

### O1 — Bring up persistent services

Deploy PostgreSQL, dstack, Trackio, Doris, Trackio's artifact-backing-store
configuration, the registry, Observatory, and monitoring with pinned versions,
TLS, separate
volumes, health checks, and backups. Follow the staged shadow-projection and
cutover gates in [`trackio-doris.md`](trackio-doris.md) and the provider gates
in [`object-storage.md`](object-storage.md).
Acceptance requires authenticated health checks and a successful backup/restore
drill using non-production test data.

### O2 — Attach and qualify one worker

Enroll one GPU machine and pass CPU, image-pull, CUDA, cancellation,
reconnection, evidence, and cleanup tests. Do not add the second worker until
the first enrollment is repeatable from documented inputs.

### O3 — Attach and qualify the second worker

Repeat the same procedure, then prove resource constraints select the intended
GPU and that draining either worker prevents new placement without corrupting
an active run.

### O4 — Integrate framework job submission

Implement the provider-neutral capsule and execution result contract plus a
dstack adapter. Prove idempotent submit/status/log/cancel/collect, Trackio
round-trip, identity reconciliation, and the global one-experiment lease.

### O5 — Exercise failure and recovery

Test registry unavailability, scheduler restart, worker disconnect, task
failure, cancellation, duplicate submission, low disk, Trackio outage,
break-glass execution, cleanup replay, and service restore. Retain small test
evidence and remove disposable images, scratch, and recovery state.

## Planned implementation layout

Implementation should extend this directory rather than a research runbook:

```text
ops/dstack-trackio/
  README.md
  compose.yaml
  .env.example
  dstack/
    server-config.example.yml
    fleet.example.dstack.yml
  reverse-proxy/
  doris/
  object-storage/
  beszel/
  ansible/
    inventory.example.yml
    playbook.yml
    roles/
  scripts/
    preflight-services-host.sh
    preflight-worker.sh
    enroll-worker.sh
    qualify-cluster.sh
    backup.sh
    restore-test.sh
```

Scripts must default to read-only or dry-run behavior, print no secrets, require
explicit targets, and refuse broad or unresolved destructive paths. Exact files
are created only after O0 resolves the real service host and pinned versions.

## Alternatives and promotion signals

dstack is selected for the current two-workstation on-prem fleet because it
already provides GPU-aware placement, task queues, priorities, retries,
status/log/cancel APIs, events, and a focused operations UI.

- Use SkyPilot when one control surface across several clouds, Kubernetes,
  managed jobs, or an existing Slurm estate becomes the dominant requirement.
- Use direct Slurm with Apptainer when CarbonTeq operates a real multi-user HPC
  cluster requiring partitions, quotas, accounting, and fair-share policy.
- Use Kubernetes with the NVIDIA device plugin and Kueue when organization-wide
  multi-tenant platform controls justify Kubernetes operations.
- Use Nomad when a broader mixed service/batch fleet, rather than ML-specific
  workflow support, becomes the primary need.
- Use HTCondor when many loosely administered workstations and opportunistic
  file-transfer scheduling become the actual fleet.
- Treat Ray Jobs, Submitit, Prefect, and Flux as layers for their specific
  runtime, Slurm adapter, workflow, or nested-scheduling problems; none replaces
  the operational contract above by itself.

Any replacement must run the same capsule and pass the same
submit/status/log/cancel/collect, identity, evidence, security, and cleanup
qualification before becoming primary.

## Decision log

- 2026-07-26: Move cluster operations out of the Ambient Agent research
  runbook and into root `ops/`.
- 2026-07-26: Select the always-on non-GPU services machine, now identified as
  the Unraid server, as the persistent physical host; GPU workstations remain
  disposable execution workers.
- 2026-07-26: Select dstack as the primary infrastructure scheduler, Trackio
  plus Observatory as experiment evidence, OCI Distribution as the initial
  registry, PostgreSQL for durable scheduler state, and Beszel for initial
  long-horizon operations.
- 2026-07-26: Keep the goal runner responsible for serial research policy and
  retain SSH/systemd/OCI as a provider-independent recovery path.
- 2026-07-26: Use Doris as Trackio's idempotent analytical projection rather
  than replacing Trackio's transactional control store.
- 2026-07-26: Reject the existing Dokploy VM as a dependency. Provision
  dedicated `ai-control` and `ai-doris` Ubuntu guests on Unraid and manage them
  with versioned Ansible, Compose control services, and native Doris services.
- 2026-07-26: Use Trackio as the internal research artifact registry. Package
  bounded task data with environment code, keep base models as immutable Hub
  references, and publish result models through Trackio. Its private backing
  bucket remains server-side; a local `ai-storage` VM is deferred.

## Primary operational sources

- [dstack server deployment](https://dstack.ai/docs/guides/server-deployment/)
- [dstack fleets and SSH prerequisites](https://dstack.ai/docs/concepts/fleets/)
- [dstack tasks and scheduling](https://dstack.ai/docs/concepts/tasks/)
- [dstack CLI and API](https://dstack.ai/docs/guides/cli-api/)
- [dstack secrets and registry authentication](https://dstack.ai/docs/concepts/secrets/)
- [dstack metrics](https://dstack.ai/docs/concepts/metrics/)
- [dstack events](https://dstack.ai/docs/concepts/events/)
- [Trackio self-hosting](https://huggingface.co/docs/trackio/self_hosted_server)
- [Trackio environment variables](https://huggingface.co/docs/trackio/environment_variables)
- [Trackio artifacts](https://huggingface.co/docs/trackio/en/artifacts)
- [Doris cluster planning](https://doris.apache.org/docs/dev/install/preparation/cluster-planning)
- [Doris deployment-mode selection](https://doris.apache.org/docs/4.x/install/choosing-deployment-mode/)
- [Doris manual deployment](https://doris.apache.org/docs/4.x/install/deploy-manually/intro/)
- [Doris Unique Key model](https://doris.apache.org/docs/4.x/table-design/data-model/unique/)
- [Doris Stream Load](https://doris.apache.org/docs/dev/data-operate/import/import-way/stream-load-manual/)
- [Unraid VM setup](https://docs.unraid.net/unraid-os/using-unraid-to/create-virtual-machines/vm-setup/)
- [Ansible Docker Compose v2 module](https://docs.ansible.com/projects/ansible/latest/collections/community/docker/docker_compose_v2_module.html)
- [RustFS feature status](https://github.com/rustfs/rustfs)
- [SeaweedFS S3 API](https://github.com/seaweedfs/seaweedfs/wiki/Amazon-S3-API)
- [OCI Distribution deployment](https://distribution.github.io/distribution/about/deploying/)
- [Harbor robot accounts](https://goharbor.io/docs/2.12.0/administration/robot-accounts/)
- [SkyPilot existing-machine pools](https://docs.skypilot.ai/en/stable/reservations/existing-machines.html)
- [SkyPilot Slurm integration](https://docs.skypilot.ai/en/latest/reference/slurm/)
- [Slurm architecture](https://slurm.schedmd.com/quickstart_admin.html)
- [Nomad device scheduling](https://developer.hashicorp.com/nomad/docs/job-specification/device)
- [Kubernetes Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Kueue overview](https://kueue.sigs.k8s.io/docs/overview/)
