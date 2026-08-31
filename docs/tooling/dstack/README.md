# dstack

## Consumer path

Developers submitting remote GPU jobs should follow
[getting-started.md](../../getting-started.md) § “Run on dstack”: install
`posttrain[dstack]`, bind the client interpreter and a named protected
credential source in `~/.config/posttrain/config.toml`, and use
`posttrain job run --provider dstack --target …`. This page is the fork/ops
ledger, not that walkthrough.

The client config never owns worker storage. The execution-dstack adapter and
ai-infra Ansible deployment define the worker paths beneath
`/var/lib/posttrain`; this repository's machine config only locates the dstack
client and selects its project.

## Selection

The local AI infrastructure uses dstack `0.20.29` for SSH-fleet GPU placement
and provider lifecycle. The framework does not recreate its scheduler: durable
framework admission owns research concurrency and retained-evidence release,
while dstack owns offers, placement, startup, cancellation, and worker state.
Optional catalog `placement.instances: [{hostname: …}]` is an exact instance
constraint passed through to dstack; it is not a posttrain admission lock.
Prefer capacity-only
targets (`device_class` / `memory_gb`) unless you need a specific worker — see
[getting-started.md](../../getting-started.md) § “Run on dstack”.

The protected dstack binding sets `capacity_wait_seconds` to 86,400 for the
24-hour capacity-admission window. Training jobs also retry provider-confirmed
interruptions for at most two hours from the first interruption and at most
five recoveries. Neither clock resets after successful provisioning, and
arbitrary runtime errors remain fail-fast. Use `posttrain run queue` to
inspect provider-native waiting independently of `posttrain workers`, which
reports framework admission placements. `posttrain run status RUN_ID` reports
the requested logical target and hostname constraints separately from the
assigned hostname and provider run identity.

A target's GPU count and minimum per-GPU memory are scheduling constraints.
They do not authorize fractional sharing of one physical GPU. Each current SSH
worker advertises one scheduling block, so two one-GPU jobs run concurrently
when both physical workers match; otherwise the unmatched job remains in
dstack's capacity wait. Omit `placement.instances` when any matching worker is
acceptable, and use an exact hostname only when qualification requires that
specific machine.

Production runs the matching CarbonTeq server, runner, and shim release from
commit `c1fda1a8e1d7bb6978d086073d467636ac15b4f1` on branch
`codex/registry-default-auth`. This selected release includes regional
failover, bounded retry budgets, and failed-region cooldowns and passed the
immutable release, component, idle-worker rolling, and scheduler-cancellation
gates.
The branch includes exact-host server credential injection and live RunPod GPU
spot discovery, exact-digest image-readiness admission, a bounded RunPod
provisioning-timeout override for large cold image pulls, and immediate
provider-absence reporting while a Pod is provisioning or running. It also
supports opt-in, per-logical-run RunPod network storage for single-node spot
tasks, including retry reuse, pre-start regional failover, and terminal
cleanup. Runner diagnostics emit environment names only and never values.
Live spot offers are also cross-checked against RunPod's authenticated
per-data-center GPU stock before managed storage is created.

## Candidate fork

The candidate fixes graceful task cancellation. Upstream resolves
`stop_duration` but the installed and current upstream runner path gives the
application about ten seconds before forced termination. This can kill the
stable post-training runtime while it is finalizing Trackio state or a bounded
checkpoint.

The fork propagates finite and zero values through the server payload, Go
runner deadline, and server container-removal deadline. It rejects `off` for
new task submissions and preserves a 300-second compatibility fallback for
legacy stored jobs. The detailed maintained delta and validation commands are
in `/home/hammad/projects/dstack/CARBONTEQ_FORK.md`.

The same fork candidate also applies server-owned default registry credentials
to an already-qualified image hostname only when that hostname exactly equals
the configured default registry and run-level auth is absent. This supports one
digest-pinned canonical image across LAN and cloud placement without teaching
dstack about R2, mirrors, or split DNS. Prefix, suffix, and port mismatches do
not receive credentials.

Runner diagnostic logs contain environment variable names only, never their
values. This is required for cloud operation because provider, registry, and
tracking credentials can all be present in the job environment.

RunPod pulls and unpacks the job image before its dstack runner becomes
reachable. ai-infra configures the successor's optional
`provisioning_timeout_seconds` for the actual-job canary and dstack persists the
resolved value with the provider attempt. Omitting the setting preserves the
upstream 20-minute RunPod default; it is not a task runtime limit and does not
change `max_duration`.

If RunPod removes an interruptible Pod before runner connection, the adapter
now treats provider absence as authoritative and fails that provisioning
attempt immediately. A Pod that still exists without runtime port metadata
continues waiting. Live qualification on the published candidate waited for
R2 verification with zero Pods, then completed the actual CUDA image on an
A100 in `EUR-IS-1`; the Pod and temporary registry objects were absent after
cleanup.

For single-node spot tasks, ai-infra may configure RunPod `run_storage` with a
Secure Cloud region pool, size, and absolute mount path. Dstack cross-checks
each live offer against RunPod's authenticated per-data-center GPU stock,
rejects regions without reported stock, and ranks `High`, `Medium`, then `Low`
before price and configured order. Only then does it create one
network volume owned by the logical run, injects that generated mount into the
persisted run and job specifications, reuses it across interruption retries,
and schedules it for deletion only when the logical run becomes terminal. A
unique run owner is the fencing boundary. Explicit volumes, services,
multinode tasks, on-demand placement, and other providers keep their existing
behavior. Provider delete failures remain retryable and do not create a false
deleted event or timestamp.

The stock signal is admission evidence, not a capacity reservation. If provider
allocation or volume creation still fails before any Pod has provisioned,
the candidate deletes the still-empty volume and reuses its logical row in the
next eligible region. Any provisioning record or attachment closes
that failover path permanently, so checkpoint-bearing storage never moves
between regions. A failed region cools down for ten minutes; when every region
is cooling down, the run waits instead of immediately cycling through them.
The managed mount remains recognizable after dstack persists it into the run
specification, while arbitrary user volumes remain untouched. Rotation counts
only no-capacity failures recorded after the current regional volume became
active, preventing an earlier region's failure from evicting a newly created
replacement.

The infrastructure pool must contain only regions that both report stock for
the requested GPU and support RunPod network volumes. The current
North-America-first A100 pool is `US-KS-2`, `US-CA-2`, `US-WA-1`, and
`CA-MTL-3`. `US-MD-1` currently reports stronger A100 stock but is deliberately
excluded because RunPod's live volume API rejects network volumes there.

The maintained retry successor stores compact per-event attempt counters and
the first event timestamp on the logical run, independent of pruned submission
rows. Pending retries use the existing 15s, 30s, 1m, 2m, 5m, 10m exponential
base schedule with stable per-run, per-attempt jitter of plus or minus 20
percent. The ten-minute value is the base cap, so the jittered delay is bounded
between eight and twelve minutes once the cap is reached.

The immutable `663974b8` candidate passed the automatic-storage interruption
canary on 2026-08-30. Logical run
`dfed0521-0f75-4773-b86e-38358a19fd98` created one 10 GB volume in
`CA-MTL-3`, lost its first A100 Pod through a direct provider API deletion,
observed a distinct retry after 51.059 seconds, recovered the exact marker from
the same volume, and finished `done`. Dstack then removed both Pods and the
owned volume. The cost estimate was $0.0302 and final provider inventory was
empty.

## Operational configuration

The ai-infra release must:

1. build the server, runner, and shim from one full CarbonTeq fork commit;
2. pin the server image by registry digest;
3. expose versioned runner and shim binaries over trusted internal HTTPS;
4. set `DSTACK_RUNNER_VERSION`, `DSTACK_SHIM_VERSION`, and their download URL
   templates to the same component release;
5. allow dstack component reconciliation to update both retained workers; and
6. retain the existing locked dstack Python client because the patch does not
   change the client protocol; and
7. keep the canonical pull username and password only in protected server
   environment variables alongside the exact canonical registry hostname.

Framework jobs use a bounded stop duration and execute
`posttrain-runtime` without an intermediate shell that swallows signals.
`stop_duration: off` is not part of the supported contract.

## Qualification evidence

Source validation currently proves 83 SQLite-focused Python cases with 21
PostgreSQL variants skipped, all 104 PostgreSQL-enabled cases, and 35
top-level Go executor/schema cases including three schema subtests. Ruff,
Python and Go formatting, and diff checks pass. It does not prove a deployed
release.

The immutable actual-job placement path was requalified on 2026-07-29. DAPO
MTP run `verl-dapo-mtp-fixed-20260729` used
`carbonteq-ai-workstation.lan` while capacity-only serve run
`capacity-concurrency-4090-20260729` was independently assigned to
`pop-os.lan`. dstack reported both healthy instances `1/1` busy concurrently,
and both provider runs finished `done`.

The release gate remains:

- immutable CarbonTeq fork commit and remote;
- server image and runner/shim receipts from that commit;
- a greater-than-ten-second finalizer completing before the configured
  deadline;
- Trackio reporting `cancelled` without recovery mutation;
- exact container removal;
- both GPU workers returning healthy and idle; and
- a normal framework `run reconcile` plus evidence-gated cleanup.

Selected production commit:
`6494f15c7a36a2cdb92cec2f9b33696adb143fef`. Regional-failover candidate:
`e9d74b0cfd330500879946141469313e46de2e7d`.
