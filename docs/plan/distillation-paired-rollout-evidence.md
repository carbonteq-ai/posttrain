# Present paired distillation evidence without inventing teacher rollouts

This ExecPlan is a living document and follows `docs/templates/PLAN.md`. The
sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes &
Retrospective` must be updated as work proceeds.

## Purpose / Big Picture

An Observatory user inspecting a `train.distill` run should be able to see the
student trajectories as one teacher-scored distillation population. The table
must show each student's request latency, completion tokens, and tool calls
without fetching complete transcripts, and the surrounding presentation must
name the student, the teacher, and the exact-token scoring relationship. It
must not describe teacher scoring as a separately generated teacher rollout.

The deployed Policy Prism OPD run
`opdq-ceil03-iwopd-e2b12b-c12-lb12-pb3-rseq12` is the live acceptance case. It
contains 12 student trajectories from `models/gemma4-e2b-it@bf16`, scored by
`models/gemma4-12b-it@bf16`, with 11,803 scored tokens. Before this work, the
selected detail shows values such as 48.6 seconds and 825 completion tokens,
while the corresponding paged table row shows dashes.

## Progress

- [x] (2026-08-12) Reproduced the historical OPD mismatch through the deployed
  summary and detail endpoints and identified the two owning-layer data losses.
- [x] (2026-08-12) Resolved the product meaning: student trajectories are
  grouped with teacher-scoring evidence; there is no independent teacher
  rollout.
- [x] (2026-08-12) Teach Trackio's payload-bounded trace read to derive safe scalar timing,
  usage, model-call, and tool-call summaries from native Verifiers calls.
- [x] (2026-08-12) Preserve enriched distillation batch attributes by marking the live trace
  as observed before terminal evidence replay.
- [x] (2026-08-12) Add a provider-neutral paired-distillation presentation to Observatory's
  paged trace response and table.
- [x] (2026-08-12) Validate focused Trackio, tracking adapter, Observatory service, schema,
  and frontend tests plus production frontend build.
- [x] (2026-08-12) Publish and pin immutable Trackio `0.31.5.post13`.
- [x] (2026-08-12) Deploy Trackio and the exact RL revision, then verify the
  historical OPD run in the live browser.

## Surprises & Discoveries

- Observation: Trackio's bounded reader already retains top-level `latency_ms`,
  token, and tool fields, but native Verifiers records keep this evidence under
  `calls[*].time`, `calls[*].usage`, and message tool calls.
  Evidence: the deployed summary for trace `aeabefef9f3a45d68beadc9a94399d52`
  has null latency/tokens/tools while its full detail derives 48,617 ms, 825
  completion tokens, and zero tools from the same stored record.
- Observation: distillation adds `distillation_batch_id`, student identity, and
  teacher identity before its live trace write, but final bridge replay writes
  the same external trace again without those additions. The historical detail
  therefore retains only the bridge attributes even though the run's scored
  token metric retains the batch ID.
  Evidence: GRPO uses the bridge's live-observed marker; the distillation
  rollout hook currently calls `context.trace(...)` without that marker.
- Observation: Trackio's repository-scoped internal-publication workflow has no
  `lan-release` runner registered and remains queued. The retained GitHub build
  artifacts were therefore SHA-256 checked and published from the isolated
  release VM with its protected credential environment; a clean install from
  `carbonteq/stable` reports `0.31.5.post13`.
- Observation: The generic retained-run qualifier completed successfully but
  did not map this project's catalog IDs to its requested job-kind samples.
  Direct acceptance against the exact Policy Prism OPD run was therefore the
  evidence for this feature: all 12 bounded rows exposed tools, latency, and
  tokens, and the paired student/teacher presentation matched the run.

## Decision Log

- Decision: Model the UI as student rollouts plus attached teacher scoring, not
  as paired student and teacher completions.
  Rationale: On-policy distillation generates with the student and asks the
  teacher for token scores over those exact student token IDs. Calling the
  teacher operation a rollout would misstate the training contract.
  Date/Author: 2026-08-12 / Codex
- Decision: Derive bounded summary scalars inside the generic Trackio fork and
  keep transcript bodies excluded.
  Rationale: This fixes existing records and every consumer of Trackio's
  `include_payload=false` contract without making Observatory fetch multi-megabyte
  native records for a table.
  Date/Author: 2026-08-12 / Codex
- Decision: Keep the API provider-neutral and make distillation context an
  optional frontend projection of the existing provider-neutral run view.
  Generic, evaluation, GRPO, and SAMPO trace tables remain unchanged when the
  context is absent.
  Rationale: The relationship is job semantics owned by Observatory, while
  scalar trace compaction is generic Trackio behavior.
  Date/Author: 2026-08-12 / Codex

## Outcomes & Retrospective

Implementation, deployment, and live acceptance are complete. Trackio
`0.31.5.post13` passed its live producer/consumer qualification. Observatory
source revision `62c53f01f8aa1bfa4ba2097ca67ea6b9634885db` was deployed as
`localhost:5000/carbonteq/posttrain-observatory@sha256:1dd041a54906643dc33dec2adb5a9af4f56db8f53350f8a747749170cfd81018`.
The local Observatory view of the retained OPD run rendered 12 student
trajectories, the E2B-to-12B exact-token scoring relationship, 11,803 teacher
scored tokens, 1.5 seconds of teacher latency, zero failures, and one recorded
batch. Trace `aeabefef9f3a45d68beadc9a94399d52` rendered 48.6 seconds, 825
tokens, and 0 tools in its paged row.

## Context and Orientation

`../trackio/trackio/sqlite_storage.py` owns the shared payload-bounded trace
projection used by SQLite and Doris reads. `include_payload=false` is the
listing contract: it may return safe scalar summaries but must omit native
nodes, transcript bodies, tool arguments, and response content.

`packages/train/src/posttrain/train/backends/trl/distillation.py` turns fresh
Verifiers trajectories into exact-token TRL distillation batches. Each emitted
trace can carry `distillation_batch_id`, student and teacher model IDs, policy
revision, optimizer step, and the number of scored student tokens.
`packages/train/src/posttrain/train/api.py` later replays the bridge's terminal
evidence for durability. A bridge live-observed marker prevents a less specific
replay from replacing a trace already submitted with richer attributes.

`apps/observatory/src/posttrain_observatory/traces.py` projects provider records
into `TraceSummary`. `apps/observatory/src/posttrain_observatory/service.py`
combines trace pages with run-owned resolved selections and metric evidence.
`apps/observatory/frontend/src/components/TraceTable.tsx` renders the bounded
rows, while `apps/observatory/frontend/src/App.tsx` owns the run page and its
selected trace detail.

## Plan of Work

First, create a Trackio fork branch from the published post12 source. Extend the
bounded payload projection to calculate end-to-end call latency, prompt and
completion usage, reasoning usage, model-call count, and tool-call count while
discarding the calls and messages themselves. Test both populated and explicit
zero values, because zero tools is evidence and must not become missing.

Second, update the distillation rollout hook so a successful enriched live
trace is marked on the bridge before final replay. Add the optimizer step and
per-trace scored token count to its attributes. Test that replay no longer
re-emits the same trace and that the retained observation carries the batch,
student, teacher, step, and scored-token fields.

Third, derive optional distillation population context in the frontend from the
typed run view's resolved student/teacher selections and distillation metrics.
Render a compact relationship band inside the trace
table: student rollout population on the left, exact-token scoring relationship
in the center, and teacher scoring on the right. Rows remain student
trajectories and expose latency, completion tokens, and explicit zero tool
calls. When a batch ID is retained, expose it as secondary grouping evidence;
historical records without a trace-to-batch key stay grouped at run-population
grain rather than being guessed into a batch.

Finally, publish Trackio before changing its immutable consumer pin. Commit and
push the Trackio fork, build and publish its wheel, update
`packages/tracking-trackio/pyproject.toml`, `uv.lock`, the Trackio consumer page,
and fork ledger, then commit RL. Update ai-infra's Trackio release manifest,
deploy and qualify Trackio, deploy the exact RL commit as Observatory, and use
the historical OPD run for live API and browser acceptance.

## Concrete Steps

From `/home/hammad/projects/trackio`, run the focused storage tests and the fork
test suite required by `CARBONTEQ_FORK.md`. From `/home/hammad/projects/rl`, run:

    uv run pytest packages/train/tests/test_api.py packages/tracking-trackio/tests/test_adapter.py apps/observatory/tests
    npm --prefix apps/observatory/frontend test -- --run
    npm --prefix apps/observatory/frontend run build
    uv run ruff check <changed Python files>
    uv run pyright
    uv run lint-imports
    git diff --check

After immutable publication, deploy Trackio and then Observatory from
`/home/hammad/projects/ai-infra` using the repository release scripts and their
saved deployment state. Do not place credentials in command output or plan
evidence.

## Validation and Acceptance

Automated acceptance requires a Trackio test proving that a bounded Verifiers
trace derived from nested calls retains numeric latency, input tokens,
completion tokens, model calls, and zero tool calls while omitting `calls` and
`nodes`. A training test must prove the enriched distillation trace is not
replayed with weaker attributes. Observatory schema/service tests must prove
that distillation context is present only for `train.distill`. Frontend tests
must prove that a distillation page renders the student, teacher, exact-token
relationship, scored tokens, and populated row scalars without changing GRPO
or evaluation tables.

Live acceptance requires the deployed health, source discovery, Trackio
qualification, image digest, and source revision gates to pass. On the selected
historical OPD run, trace `aeabefef9f3a45d68beadc9a94399d52`
must show approximately 48.6 seconds, 825 tokens, and 0 tools in its paged row,
and the table must explain that Gemma 4 E2B student trajectories were
exact-token scored by the Gemma 4 12B teacher. Values absent from retained
evidence must remain explicitly unavailable rather than inferred.

## Idempotence and Recovery

Source tests and package builds are repeatable. Trackio publication must use a
new immutable version and tag; never move the post12 tag. ai-infra deployment
must retain its existing rollback behavior and qualify Trackio before deploying
an Observatory image that pins the new wheel. If Trackio qualification fails,
leave the current post12 service and v0.3.15-derived Observatory active. If
Observatory qualification fails, its deployer restores the last qualified
image and state.

## Artifacts and Notes

The live baseline captured on 2026-08-12 is:

    trace summary: latency=null, completion_tokens=null, tool_calls=null
    trace detail:  latency_ms=48617.406, completion_tokens=825, tool_calls=0
    run relation:  student=models/gemma4-e2b-it@bf16
                   teacher=models/gemma4-12b-it@bf16
                   scored_tokens=11803

## Interfaces and Dependencies

Trackio's public `Run.traces(..., include_payload=False)` shape remains
backward-compatible and gains only derived scalar keys in `payload`.
Observatory's existing provider-neutral `RunView` and `TraceSummaryPage`
contracts remain unchanged. No reusable train, eval, or serve package may import
Observatory, and `posttrain.common` remains independent of Trackio.

Revision note (2026-08-12): Created this plan after reproducing the deployed
OPD row/detail mismatch and correcting the proposed teacher-rollout framing to
the actual exact-token scoring contract.

Revision note (2026-08-12): Recorded completed implementation and validation,
the immutable post13 publication/pin, and the release-runner registration gap;
kept deployment and live acceptance open.

Revision note (2026-08-12): Recorded the qualified Trackio and Observatory
deployments and the direct local-browser acceptance of the retained OPD run.
