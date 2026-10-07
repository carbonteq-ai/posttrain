# Execution occurrence API: isolated candidate qualification

2026-10-03. Implemented only in `/home/hammad/projects/verifiers-credit-candidate-20261003`.
The live collection runtime at `/home/hammad/projects/verifiers` and environment
packages were not edited, installed into, or switched. Both parent and worker
independently checked the 230 original files from `source-copy.json`: zero changes.
No new model calls, dependency changes, commits, or publication occurred.

## Contract

`ExecutionRef` identifies an occurrence by episode, trace, actual reporter origin
(`harness`, `interceptor`, or `tool_server`), and invocation identity. Its
`occurrence_id` stays stable when a lifecycle gains a return. Its `prefix_digest`,
`event_count`, and `phase` bind the exact observed lifecycle prefix separately.
References contain coordinates only; arguments, results, errors, and state
evidence remain in canonical `SourceSnapshot.source_json` or declared views.

Source snapshots with execution references use schema 2. Empty execution manifests
retain schema 1 and the original snapshot hash formula. Empty additive fields are
omitted from serialization, preserving legacy subject and source representations.
Trace and episode source capture retain the execution manifest. Episode resolution
requires exactly one retained child for a trace identity; equal invocation strings
across children remain distinct occurrences.

`SubjectRef(kind="execution", ..., execution=ref)` can be an assessment target or
domain credit recipient without sampled nodes, generated calls, or token IDs.
Assignment checks require exact manifest membership. Trace/episode/group overlap
checks include these recipients. Identical arguments on different invocations do
not make the occurrences equal.

`capture_execution_view(source, subjects, scope=..., observed=...)` can assess a
completed occurrence while showing an earlier dispatch-only prefix. Observed
references must resolve to the same occurrence and exact retained prefix bytes.
The reserved builder revision validates its narrow input shape on reload; full
assessment execution validates its bytes against the sealed source before the
assessor runs. Prefix context rejects terminal receipts. Result context requires
a terminal observation, including failures: a returned `success=false` is valid
observed evidence, not a successful task outcome.

`resolve_execution` returns working copies. Argument equality and provider call
IDs never establish joins. Every execution subject currently projects to explicit
`unsupported` with reason `execution_generated_call_token_relation_unqualified`.
No generated turn, span, token mask, or causal link is fabricated from SDK usage.

## Validation

Run from the candidate directory with the original dependency environment read
only; `--no-sync` prevents dependency installation:

```bash
UV_PROJECT_ENVIRONMENT=/home/hammad/projects/verifiers/.venv \
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003 \
uv run --frozen --no-sync pytest tests/v1 -q -o addopts=''
```

Final result: **179 passed, 83 skipped**, saved in `pytest-v1.log`. Skips retain
existing credential-dependent integrations and seven opt-in SDK accounting cases;
they are not new integration qualification. Six new regression tests extend the
existing scoring test file; no native test file was added. They cover mutation,
stale references, lifecycle extension, original serialization, immutable working
copies, origin collisions, decision/coordinate drift, incomplete lifecycle,
distinct equal-argument invocations, semantic credit/overlap, actual assessor
visibility, and unique child resolution. Existing scalar APIs continue to pass.

Touched-file Ruff passes (`ruff.log`), and `git diff --check` passes
(`diff-check.log`). Explicit `ty` checks find the same seven pre-existing
diagnostics in original and candidate (`ty-original.log`, `ty-candidate.log`);
the repository excludes v1 from its normal typing gate. This is not a claim that
the explicit type check passes.

`actual-hr-replay.json` records a read-only replay of the retained completed HR
I-9 episode: **24 real tool execution occurrences** resolve through the episode
source and reload correctly; every token projection remains unsupported.
`implementation.json` records candidate source hashes and isolation checks.

## Limits and integration boundary

- Standalone `execution_refs` validates reported lifecycle, payload identities,
  receipt order, immutable invocation fields, and harness decisions/coordinates.
  It does not establish authenticated state transitions or native graph validity.
  `capture_trace_source` additionally invokes the existing native trace validator,
  including state acknowledgements and generated-call consistency when present.
- The current native validator keys harness/interceptor lifecycle by invocation
  without reporter origin. Standalone indexing therefore explicitly rejects an
  equal invocation reused across those two origins. It does preserve each actual
  origin, and supports tool-server versus harness/interceptor disambiguation.
- The receipt-only view guarantees selected observed prefix bytes. It does not
  prove actual SDK model conditioning, complete assistant history, absence of
  future information in arbitrary caller-authored views, or retrieval causality.
- Reported return, acknowledged simulator persistence, domain success, and policy
  compliance remain separate claims owned by the appropriate assessor.
- Source identity metadata can identify the completed target phase even when the
  declared observation is an earlier prefix. It contains no hidden raw payload.
- Source or lifecycle extension requires a fresh snapshot/assessment. Historical
  views and findings are immutable; later results cannot strengthen them in place.
- This candidate is not integrated into the live collection runtime, released,
  pinned, or qualified for training. Deployment and token relation qualification
  remain separate gates after collection and environment reward review.
