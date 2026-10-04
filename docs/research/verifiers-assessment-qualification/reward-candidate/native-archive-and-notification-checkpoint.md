# Native archive and Gmail capability checkpoint

2026-10-04. Local candidate evidence; no published dependency or complete
task reward qualification is claimed.

## Archive retention

The native candidate now pools complete source snapshots and inline input views
within each owning trace or episode. Lifecycle records retain exact identity
references. Reload resolves these references into fully validated native objects
before existing source membership and prefix checks run. Standalone assessment
records remain inline; older inline archives remain readable. Artifact-backed
views remain inline. Arbitrary assessor context is preserved without reducing
the retained input to a handpicked subset.

Implementation lives in `verifiers/v1/assessment_archive.py`, with JSON-mode
serialization and admission hooks in `trace.py` and `episode.py`. The pool does
not cross child-trace ownership. Conflicting, dangling, duplicate and unused
pool entries reject; reference metadata must match with type-sensitive checks.
Roundtrip, legacy admission, prefix visibility, copied-model tampering, artifact
views and linear archive growth are covered by the native regression suite.
The focused scoring/trace/judge suite passed 114 cases at the pooling checkpoint;
after strict source schema-version validation, it passes 123 cases. Its archive
subset has 27 cases. Scoped Ruff passes. Nine added cases reject boolean,
floating-point and string versions at raw admission, reject tampered copied
models, and preserve valid versions 1/2 and omitted legacy defaults. Raw field
validation alone would miss the copied-model path; the source's after-validator
checks exact type too.

The same full-fidelity `operations.access_request_validation` Luna replay
retains six queue rows, fourteen effects and 85 independent assessment runs.
The original representation reached approximately 7.5 GB resident memory.
An intermediate source-only pool took 78.02 seconds in pytest and peaked at
5,150,944 KiB. Source-and-view pooling takes 71.44 seconds in pytest
(72.27 seconds process wall time) and peaks at 468,672 KiB, about 458 MiB.
These are individual local measurements, not repeated performance estimates.
Original trace bytes and official scalar reward remain unchanged.

Storage duplication is substantially reduced. Repeated content validation still
makes dense replay slow. Any future validation reuse must trust only the exact
unchanged admitted object, preserve fresh wire validation, reject copied-model
tampering and leave semantic equality unchanged.

## Shared Gmail send evidence

`contracts/notification_effects.py` implements `gmail.messages@1` using the
existing effect-fact contract. It checks native acknowledgement, revision,
returned identity and persisted sent state, and retains recipient and content
facts. Draft creation or label changes do not establish a send. Unsupported
raw/API send forms remain unavailable. The adapter contains no task policy.

The source type is registered in manifest validation and the existing native
guard dispatcher. Manifest predicates can match recipients against declared
table populations. A witnessed forbidden send publishes factual harm separately
from its declared negative action contribution. Distinct send executions remain
distinct; reload/rescore cannot consume them twice. Ambiguous recipient matching
and overlapping penalties require explicit handling rather than silent summation.

Twenty-one adapter tests and ten native notification tests pass. The combined
environment manifest/record regression batch passes 415 cases with the dense
recorded replay excluded; that replay passes separately as described above.
The positive recorded adapter example is
`simple.email_airtable_customer_welcome`. The recorded invoice example lacks a
qualified send and remains unavailable; no positive behavior is invented.

## Tracking and coverage boundaries

Training Trackio export removes raw source/view pools and retains compact
identity, digest and provenance results. Native archives retain full inputs.
The training bridge carries the native pools to the existing observation
boundary so export can apply that policy. The combined Trackio adapter and
bridge/artifact batch passes 76 cases with eight native-runtime-extra skips. Those
skips leave complete consumer qualification open.

The installed task catalog still contains ten Simple Salesforce manifests.
The 105-task sample has passed raw transport only. Gmail send support does not
establish required-action credit, full guard inventory or complete task reward
coverage. Next implement shared positive-effect obligations and reviewed task
declarations, then qualify meaningful findings and credit for at least fifteen
tasks in each category. Reserved tasks remain outside reward fitting. Qwen
eligibility, wider environment migration and publication remain open.
