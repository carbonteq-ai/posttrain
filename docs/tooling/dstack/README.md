# dstack

## Consumer path

Developers submitting remote GPU jobs should follow
[consumer-setup.md](../../consumer-setup.md) § “Run on dstack”: install
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
[consumer-setup.md](../../consumer-setup.md) § “Run on dstack”.

The protected dstack binding may set `capacity_wait_seconds`. Posttrain maps
that only to dstack's persistent `no-capacity` retry event, leaving
interruption and runtime-error retries disabled. Use `posttrain run queue` to
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

Production still runs the upstream image
`dstackai/dstack:0.20.29@sha256:6d57647be04cad42dff2343f4f50d41a3b8bb438ebc67165bc56aa92858e69ce`.
The published CarbonTeq candidate is commit
`371ff53b1d67f254bc6cc4259aae8653c3916b7d` on branch
`codex/graceful-cancellation-stop-duration`. Production is still on the
upstream image until one matching server/runner/shim release is built and
deployed.

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

## Operational configuration

The ai-infra release must:

1. build the server, runner, and shim from one full CarbonTeq fork commit;
2. pin the server image by registry digest;
3. expose versioned runner and shim binaries over trusted internal HTTPS;
4. set `DSTACK_RUNNER_VERSION`, `DSTACK_SHIM_VERSION`, and their download URL
   templates to the same component release;
5. allow dstack component reconciliation to update both retained workers; and
6. retain the existing locked dstack Python client because the patch does not
   change the client protocol.

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

Selected fork commit: `371ff53b1d67f254bc6cc4259aae8653c3916b7d`.
