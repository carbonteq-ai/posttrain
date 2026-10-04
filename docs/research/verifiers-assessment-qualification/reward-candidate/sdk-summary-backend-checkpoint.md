# Signed-in SDK summary assessment backend

Latest qualification, 2026-10-04: protocol 3 now passes one real mixed Contact
attempt. Earlier failed attempts below remain immutable historical evidence.

Local unpublished candidate, 2026-10-04. This backend is optional environment
composition. It does not replace deterministic checks, change official scalar
scores, prescribe a solver path or establish semantic model accuracy.

## Implemented behavior

`automationbench_v1/summary_backends/codex_sdk.py` implements the existing
`SummaryBackend` protocol through the pinned native Codex SDK worker in an
isolated subprocess. The protected authentication-file path is private; the
serialized assessor selection pins Luna, SDK, worker/catalog/rubric hashes,
disabled tools, time and size limits, and the 16,384 observed output threshold.

The complete public prompt and initial state accompany exact outputs, relations
and returned invocation facts. Hidden assertions and raw before/after backend
worlds are not sent. The request is source-bound and journaled before dispatch;
original stdout and stderr survive failure and cancellation. An owned process
group contains the SDK child tree and receives bounded shutdown/escalation.

Admission checks authenticated ChatGPT access, accepted Luna thread selection,
disabled execution, clean SDK lifecycle, exact answer inventory and reconciled
reported usage. Observed errors/retries, missing or regressing usage, overshoot,
interruption and malformed capture stay unavailable. Current notifications do
not expose a physical provider response model or complete provider finish
metadata. These are SDK-selection/completion observations, not proof of hidden
provider execution. No automatic outer retry or provider fallback is added.

The parser supplies trusted context/output/assessor identities, rejects duplicate
JSON keys and preserves independently valid decisions when another certificate
is malformed. The existing reducer verifies exact citations and relationships.
Incomplete coverage cannot establish compliance, while known harm survives.

## Qualification so far

Owner gate: **38 passed, one explicitly opt-in live case skipped, 5.12 seconds**.
Root combined SDK/summary/Slack/native-coverage gate: **88 passed, one skipped,
15.69 seconds**. That combined invocation read pytest configuration before the
new network marker was registered and emitted one marker warning; the marker is
now declared in the environment package metadata. Lint, typing and diff checks
are clean in the owner's released files. Independent critic repeats **38 cases
in 5.05 seconds**, deselecting the live case, and accepts one transport probe.
The live signed-in attempt remains a separate gate.

The fake-process cases exercise public-source binding, request-before-dispatch,
model/account controls, tools, usage, lifecycle errors, timeout/cancellation,
partial raw retention, process-group cleanup, strict parsing and native retained
Contact scoring/reload. They do not emulate model reasoning accuracy.

Frozen source hashes:

- `summary_backends/codex_sdk.py`: `fa437e6b55789c3445ad8cfe0c51c24b64d7cacbd980d24882762aeb75aa07c3`
- `summary_backends/__init__.py`: `6fbd658ae9f35e7fbf48fbf865aee2894e34138293b6375dc2b48979ca0dab5e`
- `tests/test_codex_sdk_summary_backend.py`: `471b6a008c4cf36f1daaf16b344453961330e8db927350915bbc271c04cf2227`

## Real integration gate

From the environment package, enable only the single opt-in test after independent
review. Supply an absolute protected authentication-file path and a fresh
absolute evidence directory through `AUTOMATIONBENCH_SUMMARY_AUTH_FILE` and
`AUTOMATIONBENCH_SUMMARY_OUTPUT_DIR`; set
`AUTOMATIONBENCH_RUN_SIGNED_IN_SUMMARY=1`. Do not store credential contents.

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_codex_sdk_summary_backend.py -k signed_in_summary_backend_live_one_attempt -q --tb=short --strict-markers

This performs one bounded signed-in attempt and retains the source, draft
contract, native trace and qualification scope even on failure. It requires an
actual complete exchange, but records the model verdict without requiring
compliance. Never overwrite its output directory or silently repeat the attempt.
Recorded references remain unchanged. The user permits existing subscription
allowance or service-selected credits; purchase, quota reset and API fallback
are outside this route. The latest account check reports included allowance
consumed and existing credits available; it is not proof of model entitlement.

Whole Contact qualification still needs semantic acceptance and its complete
manifest. Summary action/token credit, compressed archive publication, wider
environment migration and student benchmarking remain open.

One real attempt was dispatched as terminal session **83825**, retaining
evidence at `.posttrain/state/verifiers-assessment-qualification/summary-sdk-live-01`.
It terminates with **one passed, 38 deselected, 23.17 seconds**. The fresh directory
must never be reused for another attempt. Candidate sources remained frozen
through this terminal result and are now released for the next increment.

## Observed live result and parser gap

Transport and clean SDK completion pass. Reported usage is **19,304 input,
533 output, zero reasoning output**, with one reconciled response. The original
source before/after is exactly equal. Request, original stdout/stderr, exchange
and parsed result are retained; no automatic retry occurred.

The semantic aggregate is **abstained**, with no numeric value. The assistant
summary is judged compliant and the name field inapplicable. The model also
quotes the correct email value, but reports the interval `[0, 37)` for a text of
33 Unicode code points. The strict reducer rejects that out-of-bounds citation;
the test passing establishes transport, not semantic guard acceptance.

The next version changes only the backend parser: resolve exact citations from
a valid supplied interval or an unambiguous exact quote, preserving the original
response and strict reducer. Count overlapping occurrences when checking
uniqueness. Reject absent/ambiguous quotes rather than using fuzzy matching or
normalizing text. Bump the parser revision and test against the retained failure;
do not describe an offline derived correction as a new model execution. A fresh
bounded live attempt follows qualification of the repaired parser.

Retained SHA-256 values:

- `trace.json`: `7fe9cc4a95bb83810027a4a09fa408a9b71ebe40378f3268ee7f44ed300e19a0`
- `contract.json`: `d1ac1f2197f6f73498cae366f9b54d637786a308a639505de0ed1a8f101351c8`
- `source-before.json`: `6af3be2192b609ff9fc332c21a39dc2b427c9fa3b7ca393e395bf1a5bbbaa279`
- `qualification-scope.json`: `234b553c817d3004714aed803718dab3ffd8c4ca5f7bcc738f6a2e64f29fbb56`

## Parser revision 2 and the mixed Contact attempt

Parser revision 2 is implemented and independently checked: 55 offline tests
pass, with the live gate deselected. It resolves an invalid interval only when
the supplied exact quote occurs once in the same output, counting overlapping
matches. No identity, text normalization or fuzzy matching is repaired. The
historical parser-1 exchange remains unchanged; reparsing it offline is not a
fresh model qualification.

The next live gate preserved the entire installed Contact contract, including
its original introduction-read and update checks and both action-credit rules.
Only a fresh development revision and the optional summary check were added.
Terminal session **25672** reports **one failed, 55 deselected, 19.27 seconds**.
Artifacts are immutable under
`.posttrain/state/verifiers-assessment-qualification/summary-sdk-live-02`.

The SDK completed cleanly with **19,296 input and 532 output tokens**, zero
reported reasoning output and one reconciled response. Original episode bytes,
source material and scalar rewards were preserved. Both original findings were
one, and both original action contributions remained one. This establishes that
an unavailable optional summary assessment does not erase deterministic credit.

Summary qualification failed. The model's final answer contains a JSON object
followed by `</final>` and mistypes one long output identifier. Strict parsing
rejects the trailing content, so the result is abstained with no numeric guard
value. The test first reports an exchange-count assertion because the evidence
journal contains both the actual exchange and a separate parse-error envelope;
those are two records, not two model calls. The live gate must distinguish them.

Before another explicitly bounded live confirmation, review SDK structured-output
support and a request-local short-reference interface mapped by the trusted
driver to exact original identities. Do not fuzzy-match an invented identifier,
strip arbitrary trailing prose or overwrite this failed attempt. Semantic
contrast cases, summary negative action credit and whole-task qualification
remain open.

Retained SHA-256 values for this second attempt:

- `trace.json`: `1566e327d33941311171c5f93733ee0fd341bba6ad3a6c2e854c9f8dc0ef94f1`
- `contract.json`: `7df2a9b53288823c1b1094e064cbcd6f6fab8ba2c2fb806798b6f639acbc4b4a`
- `source-before.json`: `6af3be2192b609ff9fc332c21a39dc2b427c9fa3b7ca393e395bf1a5bbbaa279`
- `qualification-scope.json`: `bc18523e0ff7608c051d9eac71d39fb215f5c6ba48fed54e024218b16fb82559`

The independent root Gmail/Zendesk and Slack output-capture gate passes **74
tests in 3.25 seconds**. The critic passes **44 tests in 2.90 seconds** and two
additional adversarial probes. This qualifies submitted-text capture and known
evidence preservation, not the meaning of a message or complete task rewards.

## Protocol 3: structured response and exact local references

The optional SDK backend now selects parser revision 3 and rubric revision 2.
It sends an actual SDK final-output JSON Schema, using the audited native worker
forwarding facility. Request-local `o`, `r` and `i` references map deterministically
to the original output, relation and invocation identities. Unknown aliases,
duplicate or missing decisions, malformed JSON and ambiguous quotes remain
errors. Original IDs are never fuzzy-matched and trailing tags are not stripped.
Nullable citation offsets use an exact unique quote in the same original output.

Owner passes **69 tests with one live case skipped in 6.05 seconds**. Root passes
the same 69 with the live case deselected in 6.04 seconds; critic independently
passes 69 in 6.02 seconds. Scoped lint and backend typing are clean. Actual SDK
schema forwarding passes fifteen independent tests; this does not imply that
all native worker typing is clean (existing notification-type errors remain).

Fresh terminal session **65802** passes **one live case, 69 deselected, 16.48
seconds**. Evidence is retained under
`.posttrain/state/verifiers-assessment-qualification/summary-sdk-live-03`.
All **743 source files** in the native and environment source trees have exactly
equal hashes before and after the attempt. The original archive, safe source,
scalar rewards and two original Contact credits remain unchanged.

The actual summary aggregate is **compliant, value 1**, with zero decision
errors. The assistant summary is compliant; the requested name and email values
are inapplicable to exclusion narration. Reported usage is **18,852 input and
376 output tokens**, with **124 reasoning tokens reported as an output detail**;
do not add that detail to the reported output count. One response reconciles.
This qualifies this positive trace's real response and integration path, not
the reviewed 26-case semantic corpus or complete task reward coverage.

Frozen SDK backend SHA: `83ecdf9c763c4cca14ef29d9e934f825dd299178d1c0cf33877324938234fde3`.
Frozen backend test SHA: `f6ab02e0c00e8db998350b6eb6c727f07312820489c9ea45efc44167859fa015`.
Retained attempt SHA-256 values:

- `trace.json`: `907dfa746a459097794290737416d4f8650943510a94d5673aa5863cd5f8bd5a`
- `contract.json`: `943067572894b049da8c387fe2d3a69a5141030af57558638301556f60754a83`
- `source-before.json`: `6af3be2192b609ff9fc332c21a39dc2b427c9fa3b7ca393e395bf1a5bbbaa279`
- `qualification-scope.json`: `b624404bc07c459c1bb767d46a5023c300b50c38d1df9637b82591e45d4d7fa8`
- `source-fingerprints.json`: `f5acf50c021b22ddd77ef376806c5d3f44cec18750f272bc0140f0dbfa354035`

Next qualify semantic contrasts, including explicit violations and ambiguity,
without substituting another clean trace. Shared action-harm publication and
once-only negative credit are being implemented separately; SDK assistant text
still has no invented original token or action mapping.
