# Trackio inbox throughput and prioritized evidence ingestion

This execution plan is living. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as the implementation proceeds. The repository has no `.agents/PLAN.md`; this plan follows the checked-in `docs/templates/PLAN.md` requirements.

## Purpose / Big Picture

Training continues to produce rollout, reward, and step evidence faster than the deployed Trackio server persists it. Developers therefore see stale Observatory runs even though the trainer and native Verifiers traces are current. This change makes the Trackio inbox importer drain durable fragments in batches, uses a single scanner with bounded workers, prioritizes scalar progress metrics over large rollout traces, and keeps the server available while backlog recovery runs. The permanent acceptance test is a real backlog containing mixed metric, system-metric, and trace fragments: the server remains HTTP-healthy, scalar reward steps catch up first, and the backlog drain rate exceeds the producer rate.

## Progress

- [x] (2026-08-07 07:30 UTC) Verified the deployed Trackio server uses the Doris backend with async inbox writes, eight importer workers, and a five-second poll interval.
- [x] (2026-08-07 07:32 UTC) Applied the temporary operational change to 16 workers and recovered startup by preserving the existing inbox as `/srv/ai-control/trackio/hf/trackio/inbox.backlog-20260807T073202Z`; no fragments were deleted.
- [x] (2026-08-07 07:38 UTC) Implemented bounded batch claims, grouped storage calls, one scanner, and non-blocking startup in the Trackio working tree.
- [x] (2026-08-07 07:38 UTC) Added separate scalar and trace queues with a dedicated scalar worker lane; trace fragments cannot consume scalar queue capacity.
- [x] (2026-08-07 07:45 UTC) Built a wheel from the working tree and updated the Trackio fork/consumer notes to state that the repair is unreleased.
- [ ] Add deployment configuration and a safe replay command for the preserved backlog.
- [x] (2026-08-07 07:42 UTC) Trackio unit suite passes: 389 passed, 4 skipped; focused fragment/server tests pass: 23 passed. Real-Doris integration tests are present but skipped because their credentials are not exported in the test environment.
- [ ] Validate the deployed image with real-Doris backlog tests; deploy the Trackio fork and replay the preserved backlog safely.

## Surprises & Discoveries

- The deployed "async Doris" path is not asyncio-based. `TRACKIO_ASYNC_DORIS_WRITES=true` means HTTP requests write JSONL fragments to a durable inbox; importer threads call synchronous Doris storage methods.
- `start_inbox_poller()` performs an unbounded `import_inbox_dir()` synchronously at startup. Restarting with a 1.1 GiB backlog caused Trackio to return HTTP 502 until the backlog was moved aside.
- Each importer loop calls `import_inbox_once(max_files=1)` and sleeps at least five seconds. The implementation sorts the complete recursive inbox on every poll. The current eight-worker deployment therefore underuses the 12-core Trackio host while repeatedly scanning tens of thousands of files.
- Doris itself has capacity headroom: the FE/BE host has 24 cores and approximately 110 GiB available memory, with observed CPU below 25%. The bottleneck is the Trackio importer and transaction granularity, not Doris host saturation.

## Decision Log

- Decision: Apply a reversible temporary worker increase to 16, not 64.
  Rationale: The host has 12 cores, but the current one-file-per-poll and full-directory-sort design would make 64 threads contend on directory scans and synchronous writes. The permanent design must batch before worker count is tuned.
  Date/Author: 2026-08-07 / Codex.
- Decision: Preserve the existing backlog instead of deleting or replaying it during recovery.
  Rationale: Native traces and metrics are evidence; loss would make the two live runs unreconcilable. The holding directory is recoverable and must be replayed after the importer is fixed.
  Date/Author: 2026-08-07 / Codex.
- Decision: Treat scalar metrics/events as high priority and traces as low priority.
  Rationale: `train/rl/reward_mean`, `train/global_step`, and tracking health are small and time-sensitive; full Verifiers traces are high-volume and remain available in the native trace artifact.
  Date/Author: 2026-08-07 / Codex.
- Decision: Steps 1–4 are the required permanent scope; separate worker-count tuning after implementation is not a required milestone.
  Rationale: Batching, single scanning, prioritization, and trace separation address the structural bottleneck. Additional tuning can be derived from measured backlog rate if needed.
  Date/Author: 2026-08-07 / Codex.

## Outcomes & Retrospective

At the temporary milestone, Trackio is healthy again on an empty inbox and the old queue is preserved. The permanent importer repair is implemented and unit-tested in the Trackio working tree, but is not deployed. Record the measured producer rate, importer rate, scalar lag, trace lag, and replay receipt here after deployment.

## Context and Orientation

The Trackio fork is `/home/hammad/projects/trackio`, currently on `codex/async-doris-writes`. The server entry points are `/home/hammad/projects/trackio/trackio/server.py`; fragment writing, claiming, parsing, and importing are in `/home/hammad/projects/trackio/trackio/fragments.py`; Doris persistence is in `/home/hammad/projects/trackio/trackio/doris_storage.py`. The deployed production container is `ai-control-trackio-1` on `ai-control` (`192.168.110.53`) and writes to Doris on `ai-doris` (`192.168.110.69`). The framework's observation contract requires step-level train metrics and rollout-level traces; see `docs/post-training/06-observation-and-lineage.md`.

An inbox fragment is an immutable JSONL file containing one or more metric, system-metric, or trace records. Claiming means atomically renaming a `.jsonl` file to `.jsonl.processing` before reading it, so two workers cannot import the same file. A bounded worker pool means a fixed number of importer workers with a queue limit, so backlog growth cannot exhaust memory. A priority lane means metrics/events can be drained ahead of trace payloads without deleting traces.

## Plan of Work

First add an importer batch-size setting and a single scanner. The scanner should enumerate candidate files once, atomically claim a bounded batch, and place claims into separate metric/event and trace queues. Workers should group records across claimed fragments by `(project, run, run_id)` and call `DorisStorage.bulk_log`, `bulk_log_system`, or trace persistence once per group, deleting a fragment only after its group writes succeed. Startup must not perform an unbounded import; it should start the HTTP server immediately and let the background importer recover the queue.

Next implement priority and trace isolation. Metric and event fragments must be serviced before trace fragments, with explicit queue depth and oldest-pending timestamps. Full native Verifiers payloads remain in the job artifact; Trackio may retain a bounded dashboard trace population or a separate trace queue, but it must never block scalar reward and step metrics.

Finally add regression tests for mixed fragments, atomic claim/retry, startup health with a large backlog, grouped Doris calls, ordering guarantees for scalar metrics, and preservation of failed trace fragments. Add a real-Doris integration path where credentials are present; otherwise skip with a clear reason. Deploy the fork, replay the preserved holding directory through the importer, and compare Doris logical reward steps against trainer/native trace steps.

## Concrete Steps

Work in `/home/hammad/projects/trackio` for Trackio code and tests. Run focused tests with `uv run pytest tests/unit/...` and the real-Doris integration marker when configured. Build and deploy the immutable Trackio image through the existing ai-infra compose workflow; do not modify the framework repository's unrelated dirty files.

The operational rollback is to restore the backed-up `/srv/ai-control/compose.yml.before-trackio-workers-<timestamp>` and restart only `trackio`. The preserved queue is under `/srv/ai-control/trackio/hf/trackio/inbox.backlog-20260807T073202Z` and must not be removed until replay verification is complete.

## Validation and Acceptance

The server must return Trackio version HTTP 200 and Observatory readiness must report the Trackio source healthy while a backlog is present. A seeded backlog test must show scalar reward and global-step fragments imported before trace fragments, no duplicate event IDs, and no deleted fragment after a simulated Doris failure. During real replay, the count and byte size of the holding queue must decrease, the newest Doris scalar logical step must advance monotonically, and the server must remain responsive throughout.

Run the normal Trackio checks from `/home/hammad/projects/trackio` and then the relevant framework checks from `/home/hammad/projects/rl`; finish with `git diff --check`. Do not claim completion until the deployed server, Doris records, and Observatory agree on the same latest scalar step for a canary run.

## Idempotence and Recovery

Fragment IDs and Doris inserts remain idempotent. A worker crash leaves a `.processing` file that recovery returns to `.jsonl`; a failed Doris group must be retried without deleting the source fragment. The preserved production backlog is recoverable by moving it back to `inbox` only after the new importer is deployed and tested on a small copied subset. Never delete the holding directory as part of this plan.

## Artifacts and Notes

Temporary deployment evidence:

    Trackio workers: 16
    Trackio inbox before recovery: approximately 56,000 files / 1.1 GiB
    Preserved backlog: /srv/ai-control/trackio/hf/trackio/inbox.backlog-20260807T073202Z

## Interfaces and Dependencies

The importer must keep `FragmentWriter`, `parse_fragment_bytes`, and the provider-neutral Trackio HTTP API compatible. It may add internal functions such as `claim_inbox_batch`, `import_fragment_batch`, and `start_inbox_poller`, but must not make `posttrain` depend on Trackio implementation details. Doris writes must continue through `trackio.storage.Storage`/`DorisStorage`; Trackio's HTTP request handlers must not wait on the full backlog.

Revision note (2026-08-07): Added after observing the 1.1 GiB startup-blocking backlog. The temporary worker change was applied, then the backlog was moved to a recoverable holding directory so the service could return to HTTP 200. The permanent scope is explicitly limited to importer batching, single scanning, metric priority, and trace isolation.
