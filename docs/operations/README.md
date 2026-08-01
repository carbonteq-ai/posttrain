# Operations

This directory preserves the framework-facing operational requirements for
services and execution targets. The deployable infrastructure has moved to the
independent repository at `/home/hammad/projects/ai-infra`, which owns VM
lifecycle, service configuration, worker enrollment, DNS operations, backup,
restore, and teardown. A runbook still decides what experiment to run; this
directory records the contract that an infrastructure implementation must
satisfy.

The original deployment specification and migration evidence remain in
[`dstack-trackio/README.md`](dstack-trackio/README.md). The Unraid server hosts
the persistent service VMs, and both upgraded GPU workstations remain execution
workers. The read-only host/network evidence is in
[`dstack-trackio/unraid-o0-inventory.md`](dstack-trackio/unraid-o0-inventory.md);
the Trackio analytical-store boundary is in
[`dstack-trackio/trackio-doris.md`](dstack-trackio/trackio-doris.md), and the
provider-neutral artifact storage boundary is in
[`dstack-trackio/object-storage.md`](dstack-trackio/object-storage.md).

Do not add executable infrastructure or secret-bearing configuration here.
Make deployment changes in `/home/hammad/projects/ai-infra` and keep only the
framework-facing contract synchronized here.

## Ownership boundary

`ops/` owns:

- service topology, networking, TLS, secrets, persistence, backup, and restore;
- worker enrollment, qualification, draining, and removal;
- scheduler-specific job translation and normalized execution status;
- container image transport and retention;
- infrastructure monitoring and operational recovery.

The reusable framework packages remain provider-neutral. They own logical jobs,
runs, selections, artifacts, observations, and lineage. Research projects own
experiment policy and consume the operational service through a small
submission/status/result contract; they do not deploy or configure the cluster.

No secret, private key, password, service token, or raw environment dump belongs
in this directory. Checked-in examples must use placeholders and document how
the real value is injected.
