# Native MCP parent-call evidence checkpoint

2026-10-04. Local unpublished Verifiers candidate; no model call, training,
publication or dependency-pin change. This is an execution-provenance increment,
not a newly qualified AutomationBench manifest or token mask.

## Implemented relation

The host issues an opaque dispatch ticket only for an admitted sampled MCP call.
The harness carries it through reserved request metadata, outside model tool
arguments. Server receipts retain their independent invocation ID, host parent
execution ID, dispatch ticket, server namespace and transport-attempt index.
The host validates the relation before acknowledging dispatch or running the
business handler. A retry is a separate physical invocation under the same
sampled parent; its effects remain visible even if a response is lost.

`verifiers.v1.resolve_execution_parent(source, ref)` resolves a linked server
receipt to the exact retained harness dispatch prefix. It validates source
membership, ticket authorization, route, original sampled assistant call and
call ordinal. Missing calls, wrong roles, ambiguous provider IDs, changed
arguments and duplicate physical attempt ownership are rejected. Optional
generated-attempt membership is checked separately. Historical unlinked
receipts return `None`; there is no content-based or ordinal-based backfill.

This establishes a call relation. Exact generated-token projection still needs
retained generation spans and consumer qualification; the execution projector
continues to report unsupported rather than inventing token coordinates.

Subsequent execution alignment is recorded in
[execution token alignment checkpoint](execution-token-alignment-checkpoint.md).
That increment enables exact projection only when separately retained generated
spans qualify; historical unlinked or spanless sources remain unsupported.

## Evidence

- Client: 13 focused gates passed, including four real Null/subprocess/MCP
  variants and a live HTTP mutation/retry gate. The retry gate discarded the
  first committed response, then observed a second mutation with distinct
  attempt indices 0 and 1. A no-ticket legacy call also worked.
- Resolver extension: all four real Null/subprocess variants passed in 15.595
  seconds. Reused provider IDs resolve to distinct parents; save/reload retains
  the exact two-event dispatch prefix.
- Native server/trace/scoring regression: 129 cases passed. Independent raw
  source probes reject missing calls, wrong message roles and duplicate call
  IDs; a valid source resolves its exact parent.
- Final server suite: 42 cases passed, including nine additional real-ticket
  adversarial regressions. Copied boolean attempt indices are rejected before
  retained history changes. Existing AutomationBench external evidence tests
  also pass: 59 cases in 3.43 seconds.
  Seven further cases cover strict raw generated coordinates and legitimate
  un-emitted attempts alongside a linked call.
- Scoped Ruff and diff checks passed. Existing Pyright diagnostics remain in
  source capture, session/client helpers and server generic configuration;
  this checkpoint does not claim whole-library typing is clean.
  The new `_execution_links.py` helper itself passes Pyright with zero errors.

Frozen helper SHA-256:
`83bd5b61a51acfb55d5afd4af52dc59db6b9c9b11459de8d8a30cbb6e03e8a08`.
Source resolver SHA-256:
`04ef7e3efcea64d8ce9a31679c32da13a158d67507d95c09bc87e9641dba8807`.
Final server test SHA-256:
`29b8fac7ec06d3609eebc31f230ccc5444172608c4c692eb896e61f83e33ac89`.
Final real client test SHA-256:
`80c3d28055b90043309f2c1fc9bd9e1086f5fd5a163af710ee5c061c366309e9`.

Run native checks from `/home/hammad/projects/verifiers-credit-candidate-20261003`
with `uv run --no-sync pytest tests/v1/test_env_server.py tests/v1/test_trace.py
tests/v1/test_scoring.py -q --tb=short`. The existing `tests/v1/test_e2e.py`
contains the real client and retry gates. No paid credentials are required.

## Next gate

The adversarial parent-link regressions pass locally. Next connect qualified
generated spans through explicit alignment. Continue manifest qualification in
batches of ten. Public policy review, supported component installation and whole
task qualification retain separate acceptance counts. Current AutomationBench
catalog counts remain thirteen development components and zero whole tasks.

Independent review additionally found that raw generated-attempt metadata could
compare a boolean coordinate equal to an integer. The helper now requires exact
integer attempt and emitted-call coordinates before membership comparison;
the 35-case server suite was rerun after this repair.
Un-emitted attempts may retain an absent emitted ordinal; that does not provide
call attribution and does not invalidate a separate correctly linked attempt.
