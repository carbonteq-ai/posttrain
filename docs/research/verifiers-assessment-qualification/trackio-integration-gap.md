# Native assessments through Trackio and Observatory

Inspection on 2026-10-03 found an integration gap outside the new Verifiers API. Posttrain selects `carbonteq-trackio==0.31.5.post14.dev32`; its consumer ledger identifies fork commit `d71cf2a5cbcc561bd72e3932d8032200a791d79c`. The sibling Trackio checkout is clean on `codex/artifact-finalization` at `a3d96e5913eaac54b67bccb242ed623f3ba2e7ae`, so edits must start from the selected maintained revision rather than silently treating that checkout as the consumer version.

## Observed behavior

Trackio's `VerifiersTrace` retains the complete JSON record in its payload. A local fixture against the installed dev32 confirmed that trace-level `assessment_batches` survive `_to_dict` unchanged. This is serialization evidence, not a real remote database qualification.

The same installed dev32's `SQLiteStorage._trace_payload_for_read(..., include_payload=False)` omitted both `assessment_batches` and `episode_id`. Summary reads currently have no native assessment coverage/value projection. Full assessment input snapshots should not be added wholesale to summary pages; they contain large transcript/state inputs and may be verifier-private.

Posttrain's evaluation adapter expands an episode into child trace observations in `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py::_emit_batch`. It carries episode identity/status in attributes but does not emit top-level episode assessment batches. Training's native bridge also emits child trace observations while retaining the episode artifact separately. Native artifacts preserve replay authority; tracking projections cannot yet expose cross-trace assessments.

The Trackio Verifiers display and Observatory's reward-component projection do not present the new native assessment records. Existing scalar reward cards cannot substitute for assessment visibility. The local Posttrain environment lacks the Verifiers extra, so this inspection does not claim a producer-to-pinned-validator runtime test. Adoption must ensure the validator understands the new fields before parsing and serializing them, rather than silently discarding unknown assessment fields through an older schema.

## Required integration

Preserve one episode envelope or separately identified episode evidence record, linked to its children. Do not copy every episode batch into each child: cross-trace assessments and repeated progress snapshots would be counted several times. Preserve episodes with no child trace and their failure/coverage evidence.

Define a bounded assessment summary independent of the scalar reward: subject kind/identity, producer and signal revision, invocation/attempt, terminal status, requested/returned coverage, and explicitly selected valid values. Missing, failed, abstained, and inapplicable are distinct. Running and partial journal snapshots are history, not additional outcomes. Attempt selection must be explicit; never choose the largest value or sum retries automatically.

User decision, 2026-10-03: during training, Trackio retains assessment results only. Complete inputs, sources, judge execution traces, rationales, derivations, and masks stay in native filesystem records, compressed and published as artifacts. Trackio links to the archive for download; it does not index or query those raw records. This supersedes the earlier proposal to expose full assessment details through tracking queries.

The result projection uses an explicit allowlist: subject and signal identities/revisions, numeric values or unavailable statuses, producer revision, invocation/attempt, terminal status, coverage, source digest, and archive references. Keep valid zero distinct from missing evidence. Do not copy progress journal snapshots into result rows or implicitly reduce retries. Cross-trace results belong to one episode result envelope, including failed episodes without children. Ordinary solver rollout tracking is unchanged by this assessment-specific decision.

The training bridge currently passes the complete trace record as `TraceObservation.payload`; Trackio's raw-payload preservation would therefore store embedded assessment inputs. Sanitize at the observation/export boundary without mutating the native record or dropping evidence needed by training consumers. Its existing `VerifiersRolloutCollector.finalize` compresses native episodes into a replay artifact; qualify that path with the new fields before adding another archive mechanism. A locally retained archive is not yet a published artifact, so pending publication must remain visible and retryable.

Display result values, terminal status, coverage, and archive availability separately from legacy scalar scores. Verify results-only payloads with distinctive input/transcript markers, SQLite and Doris logical parity for result queries, idempotent retries, selected-attempt accounting, and artifact download/decompression/digest/native-reload reconciliation. Raw assessment trace queries are intentionally outside the supported tracking surface. No tracker assessment value is automatically added to policy rewards.

Generic storage/query transport belongs in the Trackio fork. Native episode expansion and backend-neutral projections belong in Posttrain; the read-only presentation belongs in Observatory. New schema/index work, if needed, requires an additive migration, retained-asset publication, immutable consumer adoption, and qualification before claiming server support. No Trackio code, release, or deployment was changed by this inspection.

## Local implementation after inspection

`TrackioTrackedRun.trace` now checks authoritative `RunSpec.stage` and applies `training_verifiers_results` to a copy of training payloads and observation attributes. The new Posttrain adapter module projects native journals and the known legacy judge paths. It retains signal meaning, zero/unavailable distinctions, source/subject digests, view identities/input digests, configuration digests, derivation parent/rule identities, and separate attempts without exporting their bodies. Ordinary solver records, rewards, metrics, and usage remain intact. Evaluation writes retain their existing behavior.

Sixty adapter tests pass, including a local SQLite payload round trip, copying checks, known-path sentinel exclusion, attempt/configuration distinction, and unknown-schema rejection. Ruff, focused Pyright, and import-boundary checks pass. This does not yet qualify results in summary queries or Observatory, episode-level result envelopes, explicit assessor-only child roles, published archive links, external artifact resolution, or remote Doris parity. No Trackio fork release or deployment is implied by the local adapter change.
