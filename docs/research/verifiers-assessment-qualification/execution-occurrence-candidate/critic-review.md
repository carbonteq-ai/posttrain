# Execution-occurrence candidate: critic review

2026-10-03. Design review of the isolated candidate at
`/home/hammad/projects/verifiers-credit-candidate-20261003`, detached from
`84ab782391bbfe1ac4f4ca32fa612e56d01b5b81` with the current candidate source
copied in. The collection runtime remains frozen. This review initially examines
the proposed interface and the existing assessment/source/credit/projection
seams; implementation and regression qualification remain pending. No production
files, runtime sources, or model calls were changed.

The small native extension is justified: a source-bound execution occurrence is
different from a sampled generated call. It can be assessed and receive semantic
credit without inventing assistant nodes or token coordinates. Keep the existing
assessment → assignment → projection transport.

## Required design corrections sent to the implementation worker

1. **Keep producer-facing identity payload-free.** `SourceIdentity` deliberately
   contains coordinates and digests, not hidden source content. A canonical
   lifecycle-prefix JSON field on an `ExecutionRef` must not be copied into that
   identity: it would bypass the declared observation view and expose arguments,
   results or private state. Store lifecycle bytes in `SourceSnapshot.source_json`;
   identity manifests retain origin, trace/invocation, prefix digest and minimal
   coordinates. Native sources/views must resolve those bytes explicitly.
2. **Verify source membership, not only self-consistent hashing.** Resolve the
   exact occurrence against exactly one retained trace child and its native
   ledger. Require the declared execution origin and invocation to match; derive
   or compare the lifecycle-prefix digest against actual selected events. A
   forged reference carrying its own UUID and digest is not source proof. Reject
   duplicate/cross-trace members, conflicting event payloads and unresolved IDs.
3. **Separate occurrence continuity from evidence revision.** The stable key is
   origin + trace + invocation; the source-bound evidence key includes the
   snapshot and exact prefix digest. A whole-ledger digest changes after unrelated
   appends and cannot be the stable occurrence identity. Expanded evidence needs
   a new reference, with an explicitly checked unchanged old prefix before
   declaring continuity. Never silently strengthen an existing finding.
4. **Separate the assessed subject from the observed view.** A completed action
   may be judged from dispatch-only evidence. Rejecting every completed subject
   under `scope="prefix"` would conflate these concepts. Restrict the observed
   evidence independently: a mechanical execution-view builder should select
   only permitted phases/prefix bytes and bind that selection to source/revision.
   An arbitrary manual `input_json` with a prefix label is not proof that future
   content is absent. Keep broader views trusted and avoid claiming a Python
   callback sandbox or actual SDK conditioning reconstruction.
5. **Observe failures without calling them success.** Returned `success:false`,
   raised, rejected and interrupted receipts are meaningful outcomes. A
   through-action-results view should not require a domain-successful return;
   missing terminal evidence remains explicit. State effects require qualified
   acknowledgements/snapshots and conflict handling, not receipt order alone.
6. **Handle semantic overlap before token projection.** Distinct prefixes of one
   occurrence may have distinct subject IDs. If both become recipients in one
   channel/branch, rejecting token overlap alone cannot prevent duplicated
   semantic credit when token projection is unsupported. Reject duplicate stable
   occurrence targets or require an explicit declared aggregation policy. An
   occurrence appearing both individually and inside a group also needs an
   explicit overlap disposition. Distinct UUID calls with identical arguments
   must remain separate occurrences.
7. **Preserve compatibility and unsupported alignment.** Existing empty-manifest
   snapshots keep version-1 identities. New execution manifests must participate
   in snapshot hashing and declared trace membership. Execution subjects carry no
   generated ordinal or token span by inference. Projection reports unsupported
   until a separately qualified execution→generated-attempt→original-token link
   exists; semantic findings and assignments remain usable meanwhile.

These requirements concern consistency relative to the retained source and
declared visibility. They cannot independently prove an external system emitted
the original bytes: source capture and its authenticated native transport remain
the provenance authority.

## Smallest useful acceptance set

- Two identical calls with distinct invocation IDs remain distinct; identical
  receipt retransmission is idempotent, while a conflicting duplicate fails.
- Same invocation string in two traces or origins does not alias; a substituted
  source member or digest is rejected on request/batch/assignment validation.
- Producer request identity contains no distinctive argument, result or private
  state marker; only the requested view exposes allowed evidence.
- A dispatch-only observed view excludes its later result/ack and unrelated
  future events, even when assessing the completed occurrence. Reload preserves
  this scope; no live-ledger resolver can silently enrich the old view.
- Missing return, domain failure and transport failure remain distinct observed
  outcomes. Concurrent conflicting writes do not receive unsupported exact
  transition attribution.
- Old snapshots with no execution manifest retain exact IDs. New snapshots seal
  execution membership, event-prefix bytes and origin metadata; source mutation
  is detected.
- Per-channel duplicate semantic recipients and group/member overlap are either
  rejected or explicitly resolved under the published policy; no implicit sum.
- Semantic assignment persists independently of unsupported token projection;
  no SDK item ID becomes a generated-call ordinal or assistant-turn index.

The candidate need not implement a workflow DSL, universal policy interpreter,
new reward transport, token mapper or replay executor to satisfy this slice.
Publication, integration with the frozen runtime, and final whole-bank reward
acceptance remain separate gates.

## Implementation review and independent checks

The worker's initial implementation now exists. Independently ran
`uv run --frozen pytest tests/v1/test_scoring.py -q` from the isolated candidate:
10 tests passed. Read `assessments.py`, `assessment_source.py`, `credit.py`,
`assessment_projection.py`, `episode_assessment.py` and the new scoring cases.
The worker reports a broader focused suite; that count is not this critic's
independent verification.

Implemented strengths confirmed by source and passing cases:

- Execution coordinates are payload-free; raw lifecycle bytes remain in the
  full snapshot. Empty execution manifests preserve legacy snapshot IDs.
- A completed subject can use a separately selected dispatch-only observed
  prefix. The reserved builder verifies its canonical shape and digest, and
  full-source batch validation compares the selected bytes to the sealed ledger.
- Distinct invocation IDs remain distinct despite identical arguments; assignment
  checks group/member semantic overlap. Execution projection is explicitly
  unsupported rather than fabricating native token coordinates.

Prefix scope here governs **selected evidence payload**, not all information
available to the producer. A completed subject's phase metadata can deliberately
reveal that completion exists while its observed view contains dispatch only.
That follows the API's independent subject/context contract. It does not claim
zero future metadata, actual model-conditioning reconstruction, or isolation from
ambient Python reads. Tests and documentation must use this bounded claim.

Two concrete standalone-indexing defects were independently reproduced and sent
to the worker; they remain open until fixes are reviewed:

1. `ToolExecutionEvent(source="interceptor")` is accepted by the native record,
   but `execution_refs` reports its origin as `harness`. Preserve the actual
   origin or reject unsupported origin explicitly; silently aliasing identities
   is incorrect.
2. A typed harness `before` receipt with decision `{"type":"stop"}`, followed
   by `dispatch`, is accepted by `execution_refs`. This contradicts the native
   terminal-before-decision lifecycle contract. Reuse minimal retained-decision
   validation, and check stable call/node/emitted/generated-link metadata across
   phases. A helper claiming native lifecycle validity must not accept this
   internally contradictory prefix.

These diagnostics constructed typed records in a temporary Python process; they
made no edits or model calls. Normal `capture_trace_source` already invokes the
stronger Trace validator, but standalone snapshot/index constructors still need
the invariants they advertise. If standalone indexing deliberately offers only
receipt-byte membership, it must state that weaker scope and never turn a claimed
applied result into acknowledged transition attribution.

Episode capture currently does not merge child execution manifests, and the
resolver initially understands trace-shaped sources only. This is a scope gap,
not evidence of cross-source acceptance: keep trace-only support explicit, or add
unique-child resolution and aggregate manifests before claiming episode-level
execution subjects. Require a two-trace case with colliding invocation strings
before accepting that extension.

## Root follow-up verification

Root independently ran `uv run --frozen pytest tests/v1/test_scoring.py -q -k execution`
in the isolated native candidate: five tests passed. The current tests preserve
the interceptor origin and reject dispatch after a stopped pre-dispatch decision.
The broader worker log records 178 passed and 83 skipped; those skipped paths are
not qualified by that result. Episode aggregation was still under development at
this check, so its later tests require a separate result.

One remaining boundary was sent to the worker: standalone occurrence indexing
groups by `(origin, invocation_id)`, whereas the native Trace lifecycle validator
currently groups harness/interceptor receipts by `execution_id` alone. A
cross-origin UUID collision must not be described as qualified native capture
solely because the standalone helper accepts it. Either qualify a consistent
native contract or explicitly reject/document that unsupported boundary. The
standalone helper's receipt validation also does not by itself prove correspondence
to a sampled node; the stronger native Trace validation owns that relation.

Original-source isolation was independently checked against all 230 entries in
`source-copy.json`: none changed. Environment work now proceeds in a separate
528-file copied candidate; no collection source was modified.

## Efficient parallel HR rule slice

An isolated external-environment worktree can proceed while collection remains
frozen. It should copy the selected dirty sources/tests explicitly, use its own
virtual environment and bind the isolated native API source. Do not change the
loaded environment, pins or official scorer.

Separate public-policy facts and pure deterministic predicates from native
evidence/view integration. The former can test offboarding's future-notification
threshold, referral finance To/CC/BCC exclusions, and NDA per-entity send/status
obligations without waiting for the execution resolver. Integration then supplies
qualified receipts, acknowledgements, entity/template identity and coverage.
Publish opt-in native findings with explicit unavailable states; add no reward
weights, efficiency penalties, optimizer writes or token-alignment claims.

Independent counterexamples and replay of the reviewed development examples are
the acceptance evidence. Do not derive obligations solely from hidden assertions
or inspect held-out tasks to fit these rules. Whole-bank task-family acceptance,
deployment to the campaign runtime and Qwen eligibility remain later gates.
