# Trackio and Apache Doris integration

Status: selected integration architecture; implementation and qualification
remain pending.

## Decision

Apache Doris is the analytical store behind CarbonTeq Trackio and Observatory.
It is not a drop-in replacement for Trackio's transactional storage.

The current CarbonTeq Trackio fork accepts only `sqlite` and `turso` in
`../trackio/trackio/database.py`. Its `SQLiteStorage` implementation in
`../trackio/trackio/sqlite_storage.py` owns SQLite connections, PRAGMAs,
schema inspection, mutation, Parquet export, and artifact bookkeeping. Pointing
that implementation at Doris would preserve neither its SQL behavior nor its
transactional assumptions.

Use a split store:

| Responsibility | Store |
| --- | --- |
| projects, run lifecycle, configs, artifact links, idempotency, projection cursor | Existing Trackio SQLite/Turso control store until a separate transactional adapter is qualified |
| metrics, system metrics, execution events, alerts, queryable trace fields | Doris analytical projection |
| large trace payloads and artifact bytes | Trackio content-addressed artifact backing storage |
| native Verifiers traces | retained immutable Trackio artifacts and optional Doris projection |

This preserves the framework contract: Trackio is an adapter behind
provider-neutral observation APIs, and Observatory computes views without
becoming a second tracking authority.

## Write path

```text
framework run
  -> Trackio ingest
     -> transactional control write
     -> append durable projection event
        -> bounded micro-batch projector
           -> Doris Stream Load with Group Commit
              -> projection cursor and health evidence
```

The transactional store and durable event must commit atomically, normally
through an outbox row in the same store. A Trackio request must not report
success after only an in-memory enqueue. Each event carries:

- a deterministic `event_id`;
- schema version;
- project, work package, logical run, attempt, and provider run identities;
- event kind and event time;
- source revision and payload digest;
- the lowest trustworthy observation grain;
- artifact references rather than large artifact bodies.

Use deterministic event IDs and Doris Unique Key tables with merge-on-write for
idempotent control-like projections. Use append-oriented tables for immutable
time-series points when their keys cannot be revised. Partition by event day
and choose buckets using measured project/run cardinality, not a guessed large
cluster layout.

Start with Stream Load micro-batches. Kafka plus Doris Routine Load is a later
promotion when measured ingest rate, independent consumers, or replay
operations justify operating Kafka.

## Read path

```text
Trackio API / Observatory
  ├─ control reader -> Trackio control store
  ├─ analytics reader -> Doris
  └─ artifact reader -> Trackio artifact versions backed by content-addressed storage
```

Expose one provider-neutral Trackio reader contract. A future PostgreSQL
control-store adapter is a separate Trackio change and qualification, not a
precondition for proving the Doris projection. Do not leak Doris SQL,
table names, or connection options into `posttrain.common`, train, eval, or
serve packages. The CarbonTeq Trackio fork owns generic analytical-store
support; this repository owns post-training-specific Observatory views.

Queries must tolerate projection lag. Every response that combines stores
includes the last projected event/time or a freshness state so the UI cannot
present a partial projection as complete evidence.

## Initial Doris topology

The Unraid deployment begins with one Doris FE and one Doris BE in a dedicated
`ai-doris` Ubuntu VM, with separate OS and NVMe-backed data disks. Trackio and
the projector run in the separate `ai-control` VM. This is a non-HA internal
deployment.

Use the official manual integrated-storage deployment, automated and pinned by
Ansible. Docker remains acceptable for D1's disposable integration fixture, but
the Doris quick-start documentation explicitly excludes Docker from production
deployment.

Do not claim production availability from this topology. Doris recommends
multiple FE nodes for high availability, while its single-machine Docker
examples are development-oriented. Promotion to three FEs and replicated BEs
requires additional failure domains, not merely more containers on the same
Unraid host.

Keep these interfaces private:

| Interface | Default port | Exposure |
| --- | ---: | --- |
| FE HTTP/API | 8030 | reverse proxy only when an authenticated UI/API needs it |
| FE edit log | 9010 | application network only |
| FE RPC | 9020 | application network only |
| FE MySQL protocol | 9030 | application network only |
| BE web service | 8040 | application network only |
| BE heartbeat | 9050 | application network only |
| BE Thrift RPC | 9060 | application network only |
| BE BRPC | 8060 | application network only |

Pin exact container digests after a compatibility test. Never use `latest`.

## Schema boundary

The first qualification schema has five bounded families:

1. run metrics at `run_id`, metric name, step/time, and stable tag-set grain;
2. system metrics at run, host, device, name, and time grain;
3. execution events at submission, attempt, event kind, and sequence grain;
4. alerts at run and alert identity grain;
5. trace indexes containing searchable fields and a pointer to the retained
   native payload.

Do not duplicate rebuildable p95/mean/rate/Pareto aggregates as source facts.
Observatory computes them from the lowest retained trustworthy grain. Material
aggregates are caches with an explicit calculator version and population.

## Failure behavior

- If Doris is unavailable, Trackio continues accepting control writes only
  while the durable outbox remains under its configured byte and age limits.
- At the limit, ingest fails closed with an actionable error rather than
  silently dropping evidence.
- The projector retries with bounded exponential backoff and preserves order
  within a logical run where ordering is meaningful.
- Replaying an event is safe because its identity is deterministic.
- Cleanup never deletes an event that is not durably projected or intentionally
  quarantined with retained diagnostics.
- A Doris restore is followed by outbox replay and a count/digest
  reconciliation before Observatory is marked ready.

## Implementation sequence

### D0 — Contract test

In the Trackio fork, define analytical writer/reader protocols and test
equivalent logical results against the existing storage path. Keep Doris behind
the new boundary.

### D1 — Local integration

Run a disposable one-FE/one-BE Doris environment. Project a bounded fixture
containing metrics, system metrics, events, and traces. Prove duplicate replay,
ordering, lag reporting, restart recovery, and deletion/retention behavior.

### D2 — Unraid staging

Deploy pinned Doris services on the dedicated `ai-doris` VM. Load only
synthetic test data. Measure ingest latency, query latency, disk growth,
compaction, restart behavior, backup, and restore.

### D3 — Trackio shadow projection

Keep the current Trackio read path authoritative while writing to Doris.
Compare counts, identities, and query results for limited GPU runs. Record
projection lag and mismatches in the execution log.

### D4 — Analytical read cutover

Move only qualified analytical queries to Doris. Keep a feature flag and
documented rollback to the existing reader until the retention and restore
drills pass.

## Qualification gates

- no acknowledged observation is lost during Doris restart;
- replaying a complete event population produces no duplicate logical facts;
- control-store and Doris run identities reconcile;
- the UI exposes stale/partial projection state;
- retention removes expired hot rows but preserves explicitly retained
  artifacts and native Verifiers traces;
- a backup restores into an empty deployment and passes count/digest checks;
- a limited GPU run creates less evidence than its declared byte budget;
- cleanup removes disposable test data while retaining terminal result and
  execution-log summaries.

## Primary sources

- [Doris database connectivity and FE MySQL protocol](https://doris.apache.org/docs/4.x/db-connect/database-connect/)
- [Doris cluster planning and ports](https://doris.apache.org/docs/dev/install/preparation/cluster-planning)
- [Doris deployment-mode selection](https://doris.apache.org/docs/4.x/install/choosing-deployment-mode/)
- [Doris manual deployment](https://doris.apache.org/docs/4.x/install/deploy-manually/intro/)
- [Doris Unique Key model](https://doris.apache.org/docs/4.x/table-design/data-model/unique/)
- [Doris merge-on-write](https://doris.apache.org/docs/dev/table-design/data-model/merge-on-write/)
- [Doris Stream Load](https://doris.apache.org/docs/dev/data-operate/import/import-way/stream-load-manual/)
- [Doris Routine Load](https://doris.apache.org/docs/dev/data-operate/import/import-way/routine-load-manual/)
- [Doris Docker quick start](https://doris.apache.org/docs/3.x/gettingStarted/quick-start/)
