# Project shared Verifiers trace facts into Trackio

This ExecPlan is a living document. Maintain it according to
`docs/templates/PLAN.md`. The repository does not contain `.agents/PLAN.md`;
that template is therefore the applicable local execution-plan authority.

## Purpose / Big Picture

Every GRPO, on-policy distillation (OPD), and environment evaluation run keeps
native Verifiers traces. Today, a consumer can obtain a rollout chart only by
loading those large traces and repeatedly interpreting their JSON payloads.
After this work, Posttrain's shared Verifiers evidence adapter will calculate a
small, common set of facts once per native trace. The train and eval bridges
will emit those typed facts with the untouched trace, Trackio will persist and
aggregate them generically, and Observatory will query them without loading
transcripts, token arrays, or tool arguments.

Reward components will also retain lightweight semantic provenance: whether a
signal came from an LLM judge, deterministic code, a human label, the
environment, a group-relative scorer, a teacher, a composite calculation, or
an unknown historical source. This provenance describes how the score was
produced; it does not materialize judge calls, prompts, responses, models,
tokens, latency, or cost.

The raw Verifiers record and its native `traces.jsonl` artifact remain replay
authority. A trace fact is a rebuildable, versioned read projection: it makes
common evidence fast to query but never replaces or mutates the native record.

This does not change the frozen post-training product baseline. It implements
the existing rule that Verifiers traces are the authority for evaluation,
online RL, and OPD, while tracking backends provide queryable normalized
evidence. It adds no new job kind, selection, or project decision.

## Progress

- [x] (2026-08-14 00:00Z) Confirmed that `train.grpo`, `train.distill`, and
  `eval.*` all use `VerifiersTrace` as their rollout evidence source.
- [x] (2026-08-14 00:00Z) Confirmed that Trackio already persists the full
  Verifiers payload and supports payload-free trace listings, but has no
  persisted scalar fact columns, reward-component relation, or grouped
  aggregate API.
- [x] (2026-08-14 00:00Z) Reproduced the historical failure on
  `ambient-k1-olmo3-sft10k-rtxpro-full7749-100step-20260810-r1`: 21,572
  retained traces encode training attribution as `payload.run.step`, so the
  existing Observatory raw-payload projection initially found no attributed
  rollout behavior.
- [x] (2026-08-14 00:00Z) Audited the pinned Verifiers v1 commit
  `284a868d6a9022109b749710672a0460e8a996d4` against upstream `main` at
  `b6bb8feae9b1dbcfc5b5df7265dea012bb5e9ca7`. Both preserve the native fields
  needed for this projection: per-call usage (with optional reasoning tokens),
  structured assistant tool calls, normalized finish reasons, and generated
  token IDs with sampled masks. Upstream is newer but no pin advance has been
  approved or made by this plan.
- [x] (2026-08-14 18:00Z) Added the provider-neutral `TraceFactSet` envelope,
  shared pure-Python Verifiers projector, immutable model/template identity,
  extensible `ThinkingTokenRule` protocol, and producer wiring for the training
  and evaluation bridges. Focused common, environment, eval, and train tests
  cover provider usage, structured tools, complete and truncated Qwen 3.5
  reasoning, ambiguous boundaries, evaluation step isolation, future-model
  rule injection, and portable bridge snapshots.
- [x] (2026-08-14 19:00Z) Extended the envelope with immutable long-form
  measure groups and retained every named Verifiers reward contribution.
  Structured historical records additionally retain raw score and weight;
  scalar reward remains the sum of valid weighted contributions rather than a
  replacement for them.
- [x] (2026-08-13 20:54Z) Revised this plan around a lightweight signal-source
  abstraction and explicitly rejected judge-execution materialization.
- [x] (2026-08-13 20:54Z) Revised Trackio persistence to extend the existing
  trace row for one-to-one scalar facts and retain only dynamic reward
  components in a normalized child table; removed the proposed scalar
  `trace_facts` table and database materialized-view interpretation.
- [x] (2026-08-14 21:30Z) Closed the persistence-readiness architecture gaps:
  native trace facts and trainer-owned algorithm reward now use two explicit
  write phases; deterministic projection identity is separate from calculator
  semantics; token names and accounting invariants are fixed; the Trackio SDK
  and HTTP contracts have exact names; and Doris performs a deliberate global
  schema version 1 to version 2 migration rather than capability negotiation.
- [x] (2026-08-14 22:15Z) Added typed per-component signal-source metadata to
  the common envelope, environment declarations, projector, and fixtures.
  Added scalar-only trainer enrichment observations and the token-subset
  invariant: incompatible thinking/output accounting is retained as missing,
  never as an impossible pair.
- [x] (2026-08-14 22:35Z) Added Trackio's typed trace-fact SDK contract,
  initial Verifiers-trace projection serialization, SQLite current-projection
  persistence, idempotent enrichment write, and restricted SQLite aggregation.
  Unit coverage proves initial source writes and repeat enrichment writes do
  not require native payload reads.
- [x] (2026-08-14 23:50Z) Implemented the matching Doris v1-to-v2 schema
  migration, scalar/current-projection writes, component replacement, remote
  HTTP operations, and restricted aggregation. The fork version is now the
  unpublished manual candidate `0.31.5.post14.dev0`; fork-owned publication
  workflows were removed so the retained-asset process remains Posttrain-owned.
- [x] (2026-08-14 06:00Z) Qualified the repaired Trackio candidate against the
  real isolated Doris database. A backup/restore receipt preceded the explicit
  v1-to-v2 migration; a synthetic native Verifiers source projection, later
  algorithm-reward enrichment, payload-free grouped aggregate, and exact
  synthetic-project cleanup all passed. The first two manual dev candidates
  exposed Doris DDL incompatibilities and were superseded by the retained,
  readback-qualified `0.31.5.post14.dev4` release.
- [x] (2026-08-14 23:10Z) Added the provider-neutral aggregate reader models
  and capability state. The Trackio reader maps the restricted aggregate API;
  W&B returns an explicit unsupported result. Observatory now prefers that
  aggregate for rollout behavior and has a regression test proving it does
  not open retained trace payloads when facts are available.
- [x] (2026-08-14 23:35Z) Added an isolated-worker append-only
  trace-keyed reward journal and trusted parent-process replay for veRL
  algorithm reward. Unit fixtures verify exact trace-ID pairing, optimizer
  step attribution, finite journal validation, and the scalar-only update
  contract.
- [x] (2026-08-14 06:20Z) Consolidated the durable architecture decision: the
  deployed Trackio estate moves directly to one required v2 fact schema, while
  future model support is added only through versioned Posttrain projector
  rules and fixtures. There is no per-capability database version, no
  model-specific Trackio schema, and no materialized judge-execution surface.
- [x] (2026-08-14 07:05Z) Added a bounded `posttrain trace-facts backfill`
  preview/apply command. It reads one physical Trackio run page at a time,
  calls the shared projector, and writes only through the authenticated generic
  fact endpoint. A real isolated-Doris qualification proved synthetic native
  trace ingestion, preview (one trace/no writes), apply (one trace),
  payload-free aggregate read, and exact synthetic-project deletion.
- [x] (2026-08-14 23:55Z) Published Trackio dev4 manually from fork commit
  `9dd9fa24920054a0ada636dbaf8f861971dfb81a`, retained hash-verified wheel
  and source assets in `carbonteq/dev`, and updated Posttrain's exact
  development pin and lockfile. The isolated candidate and shared Trackio
  servers both report dev4; a real native trace/component aggregate
  qualification passed and removed its synthetic project.
- [x] (2026-08-15 11:38Z) Materialized Posttrain-owned development runtime
  candidates through `0.3.16rc17` and completed two bounded local jobs. The
  six-update DAPO run retains 48 traces with complete source and algorithm
  reward coverage; the evaluation run retains 16 traces with complete shared
  source-fact coverage and correctly absent algorithm reward.
- [x] (2026-08-15 00:35Z) Repaired the manual runtime-candidate materializer:
  it now synchronizes internal runtime-profile pins from the candidate's
  already-resolved dev lock in its disposable checkout. This prevents a dev
  Trackio lock from being combined with the old stable profile pin. The
  release regression suite passes 45 tests; a fresh candidate dispatch is the
  remaining image-publication step.
- [x] (2026-08-15 00:45Z) Corrected SQLite-to-Doris reconciliation for an
  existing target. Trackio dev5 introduced a bounded `source-inclusion` mode
  that proves every source record is present while allowing retained target
  history. The candidate exposed a package-version consistency release gate;
  dev6 corrects that candidate and awaits its retained-index receipt.
- [x] (2026-08-15 08:00Z) Ran the bounded local Verifiers evaluation job
  `c7b42715-e4ce-4752-ac3d-d79539b003de`. It completed and retained 159
  native traces in Trackio provider run `d1c6cdb879154e088fab7166ee5b9b44`.
  The exact-run backfill applies the shared source projection through the
  generic Trackio endpoint; its aggregate read now reports 159 projections,
  full model-call/tool-call coverage, and correctly absent output-token
  coverage for the error traces that did not report usage.
- [x] (2026-08-15 08:05Z) Built candidate `0.3.16rc12` from the pushed
  framework branch using the dev-channel Trackio dev7 receipt after recovering
  52 GiB of OCI registry capacity from superseded candidate manifests. Its
  exact base, evaluation, and TRL images are now bound in Ambient Agent's local
  executor state.
- [x] (2026-08-15 08:28Z) Completed the bounded local
  `qualify/k1-extract-trace-facts-eval-smoke-0.8b` evaluation run
  `2015f127-e0aa-42ff-86e1-1de41cfa9cf3`. Its Trackio provider run
  `dbe516ea2eb44a5eaf3e41322439e367` retained all 16 native Verifiers traces.
  After the calculator v3 exact-run backfill, the payload-free aggregate has
  full coverage for model calls, input/output tokens, tool calls, latency, and
  task reward; no trace reports model reasoning content, so thinking-token
  coverage correctly remains zero.
- [x] (2026-08-15 08:35Z) Published and activated `0.3.16rc13`, which carries
  the calculator-v3 output-token fallback and Trackio dev7. The bounded local
  GRPO smoke reached live rollouts and retained five native traces, but failed
  before enrichment: dev7's timing retry exhausted because the queued source
  trace batch had not yet been sent to the service.
- [x] (2026-08-15 08:48Z) Released `carbonteq-trackio==0.31.5.post14.dev8`
  from immutable fork commit `21d3be91aaf7a4e07ee879e5225fe9be9f3d7ba3`.
  The manual GitHub prerelease retained the verified wheel and sdist; Posttrain
  retained-asset publisher run `31785289443` read those exact bytes, published
  them to `carbonteq/dev`, proved index storage, and completed a clean install.
  Dev8 sends queued native logs before a dependent fact upsert under the client
  lock; the focused Trackio regression passes.
- [x] Published and activated `0.3.16rc14` with Trackio dev8. Its bounded
  local GRPO smoke reached live rollout generation but reproduced missing-parent
  enrichment. This exposed the Doris service's asynchronous trace-write
  acknowledgement, not another Posttrain projection defect.
- [x] (2026-08-15 09:05Z) Released Trackio dev9 from
  `bf18b6dc02315404370094241516a20bf1ed7e30`, published the retained
  development asset through Posttrain publisher run `31787049987`, and
  deployed it to the shared Trackio service. Dev9 synchronously persists
  native trace batches while retaining asynchronous scalar logging.
- [x] (2026-08-15 09:20Z) Reproduced and repaired the remaining causal-order
  defect in Trackio dev10 (`04b58848`). Remote `Run.flush()` had written native
  trace logs only to the local retry buffer, allowing dependent enrichment to
  reach the service first. Dev10 sends the queued parent batch before draining
  scalar retries; the focused regression, live source/flush/enrichment probe,
  manual prerelease, and Posttrain publisher run `31789084642` passed.
- [x] (2026-08-15 09:35Z) Activated candidate `0.3.16rc16` with the exact
  Trackio dev10 wheel and completed a live local DAPO source-fact qualification.
  Framework run `8dafeb63-d9aa-4bc4-8c6b-2185b82af4c7` retained a rollout
  group whose four traces have full coverage for calls, input/output and
  thinking tokens, tool calls, latency, and task reward through Trackio's
  payload-free aggregate API.
- [x] (2026-08-15 11:35Z) Candidate `0.3.16rc17` was built by successful
  Posttrain workflow `31791363589` from framework commit `a394de76`. Three
  bounded Qwen groups measured `0.062074`, `0.064933`, and `0.077103`; the
  selected local DAPO policy records and enforces `0.10`, not an opt-out.
  The completed DAPO run has six optimizer updates and full source plus
  algorithm-reward aggregate coverage on each rollout step.
- [x] (2026-08-15 11:38Z) Completed the second real local job,
  `qualify/k1-extract-trace-facts-eval-smoke-0.8b` execution
  `e9ff1a9b-521e-4a56-9e2c-259d23651b8c`. Trackio provider run
  `b75986584fc54081ada613501ac38e90` retained all 16 evaluation traces with
  full calls, input/output-token, tool-call, latency, and task-reward coverage.
  Thinking-token coverage is zero because these traces provide no supported
  reasoning-token evidence, which is the expected null-preserving behavior.
- [x] (2026-08-15 10:45Z) The rc17 local DAPO run
  `65a9177b-5e0e-4794-b733-64e6435aa85b` reached its first optimizer update
  with an observed policy-parity delta of `0.07617`. Its Trackio provider run
  `6b72f56b015d47a3b6a963151dc35ab6` has an eight-trace, payload-free aggregate
  with complete coverage for calls, input/output/thinking tokens, tool calls,
  latency, task reward, and post-rollout algorithm reward.
- [x] (2026-08-15 21:55Z) Diagnosed the failed remote base-model run
  `b5c79e2d-3642-44a6-84a5-94c6644b5949`: its Trackio adapter synchronously
  flushed queued native traces before every trainer reward enrichment and the
  `RemoteClient` `/bulk_log` request exceeded its 60-second timeout. Trackio,
  Doris, and object-storage capacity were healthy; this is a causal-delivery
  design fault, not storage exhaustion.
- [ ] Replace synchronous trace/fact delivery with one durable asynchronous
  Trackio inbox protocol, add regression coverage for delayed Doris writes,
  release the fork manually, update the Posttrain pin/runtime image, and resume
  the run from its verified latest checkpoint as a fresh attempt.
- [x] Extend Trackio's existing SQLite/Doris `traces` schema with nullable
  scalar fact columns, add normalized reward-component rows, and implement
  grouped reads.
- [x] Release the Trackio fork manually and consume its immutable development
  candidate from Posttrain. (Dev4 is consumed; stable promotion remains.)
- [x] Add the provider-neutral reader contract and use it from Observatory.
- [ ] Backfill retained production Verifiers traces, qualify complete
  GRPO/OPD/evaluation reads, promote, and pin the stable release. Completed:
  Trackio dev15 is released and deployed to both Doris services; the retained
  GRPO, OPD, and evaluation gates pass; all 21,572 historical Ambient Agent
  traces project into all 100 optimizer-step buckets. The identical dev15
  assets have been promoted unchanged to the stable index, and Observatory now
  returns an explicit unavailable state rather than scanning raw payloads when
  trace facts are unavailable. Remaining: materialize the next Posttrain
  candidate image graph and run its final release validation.
- [x] (2026-08-14 15:25Z) Replaced the five-page historical accumulator with a
  checkpointed pipeline: one source and writer session processes an at-most
  5,000-trace orchestration window, but reads, projects, writes, and flushes a
  sanitized resume receipt after every at-most-1,000-trace physical page. Unit
  coverage proves one-session reuse, ordered checkpoints, and interruption
  recovery after the last successful write; all 146 CLI tests pass.
- [x] (2026-08-14 15:25Z) Finished the historical Ambient Agent projection from
  cursor 14,100 through EOF at 21,572. The payload-free aggregate now reports
  21,572 traces in exactly 100 ordered `rollout_step` buckets, from step 1
  through step 100.
- [x] (2026-08-14 15:25Z) Previewed, applied, and EOF-checked the terminal
  succeeded OPD run `opdq-ceil03-iwopd-e2b12b-c12-lb12-pb3-rseq12` (provider
  run `e3aa5ad1a76c4dfd80dd9e4132c4da95`). Its payload-free aggregate reports
  all 12 retained traces in rollout step 0 with full model-call, output-token,
  tool-call, and latency coverage; the historical records correctly have no
  task-reward coverage.
- [x] (2026-08-14 16:15Z) Published the selected CarbonTeq vLLM source overlay
  manually as GitHub prerelease `carbonteq-v0.25.2.dev2`, bound to runtime
  revision `7817d845727af570352622dc8d58f2d43c76d89d`. The retained source
  archive and its portable checksum were independently downloaded and verified
  as `8d4736461fbc3bf72075b4d84417208b3c5fc9ffc6f48bf26cbe9ef955cf307b`.
  This does not substitute for the existing upstream ABI wheel or GPU
  qualification; it closes the missing immutable source-overlay receipt.
- [x] (2026-08-14 16:20Z) Published the selected AutomationBench compatibility
  source manually as GitHub prerelease `carbonteq-v1.0.5.post1`, bound to the
  vendored environment revision `908db2abd4a868acc37ab0850474bff653bea25c`.
  Independent release download verified wheel SHA-256
  `bd80b4947fbdd60706d9545e79635b79931d89dfc294ed45b01df6886c1f1509` and
  source SHA-256 `04ccef85e2a83bd26777a10a08702b4fb6a47169352777ab8564fa1bbba9acf6`.
  The inherited tag-triggered fork publisher was removed from the maintained
  branch; subsequent releases remain manual.
- [x] (2026-08-14 16:30Z) Replaced the former Trackio/TRL-only readiness
  snapshot with `posttrain-release fork-ledger`. It cross-checks direct package
  metadata, the veRL kind profile, the AutomationBench environment source, and
  the deployed dstack identity before embedding the six-entry result in a
  version-2 readiness receipt. The ledger marks the deployed upstream dstack
  digest as non-required; its unrelated unpublished cancellation candidate can
  no longer silently block or falsely satisfy a Posttrain candidate.
- [x] (2026-08-14) Re-ran the complete local source gate after the candidate
  manifest and trace-fact protocol changes: `1245 passed, 23 skipped`; `ruff`,
  `pyright`, `lint-imports`, and `git diff --check` pass. Runtime/CLI unit
  tests now model a pending candidate structurally while release tests retain
  strict shipped-manifest validation, so a source candidate cannot make the
  production consumer loader permissive merely to satisfy unit tests.
- [ ] (2026-08-14) Materialize the next Posttrain candidate base and six
  dependent kind images from a committed, pushed candidate ref. The live
  desired-versus-observed registry plan is intentionally `build` for every
  node because no compatible candidate base exists. The materializer verifies
  `HEAD` against the requested remote ref, so it cannot safely or reproducibly
  run from this dirty working tree.

## Surprises & Discoveries

- Observation: the same native Verifiers format is used by training and
  evaluation, but only training traces have an optimizer-step meaning.
  Evidence: current trainers write `optimizer_step` in trace attributes;
  historical training traces instead carry `payload.run = {type: "train",
  step: N}`. Evaluation traces have no valid optimizer-step dimension.

- Observation: a complete raw-payload scan is unsuitable for an ordinary run
  page.
  Evidence: reading the first 5,000 payloads of the 21,572-trace historical
  run took about 22 seconds. The cached response was under 1 ms, proving that
  the expensive work belongs in an explicit stored projection rather than
  normal page loading.

- Observation: completion-token and tool-call facts can be obtained from
  Verifiers without model-specific policy, but thinking-token recovery cannot.
  Evidence: usage fields and structured `message.tool_calls` are native
  record fields. The current Qwen 3.5 recovery requires a known generated
  `</think>` token boundary plus the sampled-token mask.

- Observation: `Trace.is_truncated` is a derived Verifiers property, not a
  serialized field. The raw record carries the inputs needed to recompute it:
  a framework stop condition and the final successful model call's normalized
  `finish_reason`. The shared Verifiers evidence projector must derive this
  value itself;
  it must not treat an absent serialized `is_truncated` field as false.

- Observation: a length-capped response still has observed model output tokens,
  prior structured tool calls, and possibly provider-reported reasoning tokens.
  These are evidence of what was emitted before the cap, not failed or zero
  values. Only a model-specific reasoning fallback needs an additional proof
  that the unfinished sampled suffix is a thinking segment.

- Observation: exact whole-project evidence equality is invalid when a
  cutover copies a point-in-time SQLite snapshot into a Doris project that
  already contains later or independently retained history.
  Evidence: the migration's target `ambient-agent` count was larger than the
  stopped SQLite snapshot (for example 117,075 versus 490 metrics), while the
  import receipt and ordinary Doris reads succeeded. A full target scan was
  also too expensive for routine operational verification, so the verifier
  needs a bounded, source-inclusion proof rather than a whole-project equality
  comparison.

- Observation: a candidate runtime image has two independent dependency
  inputs: the hash-addressed constraint lock and the version constraints in
  the job-kind profiles.
  Evidence: the first rc9 candidate resolved and installed Trackio dev4, then
  failed before image construction because `profiles/common.txt` still named
  post13. Candidate publication must synchronize both inputs in its disposable
  checkout; changing the committed stable profile would make ordinary
  consumers select an unpublished candidate.

- Observation: constraining a fresh `uv pip compile` with an older exported
  requirements file is not a release-closure proof. Evidence: Trackio dev15
  requires `boto3`, `botocore`, and `s3transfer`, which were present in the
  root `uv.lock` but absent from the stale runtime export; a fresh compile also
  enumerated later artifacts for already-selected versions. The runtime
  materializer now exports its workspace constraint file directly from
  `uv.lock`, replaces internal requirements with their immutable wheel
  receipts, and projects each non-transform image closure through the exact
  lock graph without a second solve.

- Observation: a fork source overlay can have a release boundary without
  manufacturing a CUDA wheel. Evidence: the veRL profile verifies a separately
  retained upstream ABI3 vLLM wheel and overlays only the CarbonTeq Python
  source revision. A manual source archive plus a commit-bound GitHub release
  makes that overlay independently auditable without reintroducing a local
  native build into the release path.

- Observation: release readiness cannot infer the full fork closure solely
  from the root Python lock. veRL and vLLM are selected by an isolated kind
  profile, AutomationBench is vendored by a dynamically resolved environment,
  and dstack is a service identity. The generated ledger cross-checks each
  owning selection surface and records an upstream deployed service separately
  from an unpublished CarbonTeq service candidate.

- Observation: an algorithm enrichment must not request reward-component
  replacement. Trackio accepts replacement only for the original
  `verifiers.trace` source projection; later trainer updates are deliberately
  scalar-only and may carry only `algorithm_reward`.
  Evidence: the first real local GRPO smoke reached `run_rollouts` and the
  Trackio write was rejected with `only a Verifiers source projection may
  replace reward components`. The Trackio bridge now makes initial writes
  replace components and later enrichment writes preserve them, covered by a
  focused adapter test.

- Observation: a bounded retry does not establish a dependency ordering when a
  producer's native trace is still waiting in the SDK log queue. The real GRPO
  smoke retained five source traces and then exhausted the former 0.8-second
  missing-parent retry window. The fix belongs in Trackio's SDK: send queued
  native logs and the dependent fact update under one client lock, retaining
  retry only for a true service-side visibility delay.

- Observation: a payload-free trace-list response is not evidence that the
  stored fact columns are null. The Trackio trace list deliberately returns
  native trace fields only; scalar fact coverage is proven through the
  restricted aggregate endpoint. The local evaluation read initially appeared
  empty because it inspected the list projection rather than that fact API.
  Evidence: after an exact-run bounded backfill, `get_trace_facts` reported
  159 projected traces with 159 model-call and tool-call values while
  `get_traces` continued to omit fact columns.

- Observation: a managed Verifiers evaluation can retain sampled assistant
  messages with empty token arrays while its provider call usage contains
  nonzero `completion_tokens`. Treating those empty arrays as sampled-token
  evidence incorrectly projected zero output tokens. The projector must treat
  an empty token array as unavailable and fall back to complete provider usage.
  Evidence: the completed 16-trace local evaluation initially aggregated zero
  model-output tokens despite native call usage such as 333 and 468 completion
  tokens. Calculator v3 backfill changed the aggregate mean to 256.94 tokens
  with full output-token coverage.

- Observation: in pinned Verifiers, a judge is a scoring mechanism rather than
  a separate reward namespace. A configured judge can return one scalar or a
  mapping of named scalars; `Task.score()` records those results into the same
  `trace.rewards` collection used by deterministic task and group rewards.
  The native `trace.info["judge"]` entries do not reliably link a judge call to
  its resulting reward component, especially when a rubric makes multiple
  calls or a task reward invokes a judge internally.
  Evidence: pinned `verifiers/v1/task.py`, `judge.py`, and `trace.py` show the
  judge response capture and reward recording as separate operations.

- Observation: both Trackio storage engines already model one physical row per
  trace and retain the native Verifiers record in a nullable `payload` column.
  SQLite keys that row by `id`; Doris uses `(project_id, trace_id)`. The row
  already carries `trace_type`, external ID, schema version, messages, metadata,
  and indexes used by run reads.
  Evidence: `trackio/sqlite_storage.py` creates `traces` and its run indexes;
  `trackio/doris_schema.py` defines the equivalent unique-key table. Therefore
  a second one-to-one scalar table would duplicate identity and lifecycle
  without providing a different measurement grain.

- Observation: trainer-owned algorithm reward does not exist when the native
  trace is first observed. In the TRL GRPO path,
  `run_observed_rollouts(..., observe_trace)` emits every trace before
  `shape_online_reward(...)` calculates the algorithm reward. The isolated
  veRL path similarly tails native trace rows independently of the later
  agent-loop reward fields.
  Evidence: `packages/train/src/posttrain/train/backends/trl/grpo.py` observes
  traces around lines 385-405 and shapes rewards around lines 440-454;
  `packages/train/src/posttrain/train/backends/verl/launcher.py` tails native
  traces while `agent_loop.py` later emits `algorithm_reward` keyed by
  `rollout_trace_id`.

- Observation: one calculator version is not a safe physical generation key
  for a replaceable component set. A retry using the same semantics can omit a
  component that an earlier partial write inserted, leaving that stale row
  visible when child rows are keyed only by calculator version.
  Evidence: Doris component replacement is multi-row and cannot share the
  single SQLite transaction used by the local backend. A deterministic digest
  of the complete projected payload gives both engines one immutable
  generation identity while `calculator_version` continues to describe
  semantics.

- Observation: an `output_tokens` label is ambiguous unless the system states
  whether reasoning is included. The rollout chart must never compare total
  thinking tokens with answer-only output tokens and present the former as a
  larger subset of the latter.
  Evidence: Verifiers may retain native generated token IDs, provider usage,
  and reasoning usage through different paths. The projector must normalize
  them into `model_output_tokens`, defined as all generated assistant tokens,
  before persisting a thinking-token subset.

- Observation: an explicit `Run.flush()` must have stronger semantics than a
  background retry drain when a following write has a foreign-key-like parent
  dependency. The Trackio remote client formerly put native trace logs into a
  local SQLite retry buffer, then sent trace-fact enrichment directly to Doris.
  Evidence: dev9 made server native trace batches synchronous but the rc15
  DAPO job still failed `trace '<id>' does not exist`; Trackio dev10's live
  source/flush/enrichment probe passed once flush synchronously delivered the
  in-memory trace batch before draining scalar retries.

- Observation: the dev9/dev10 causal-order repair makes one overloaded Doris
  request fatal to training. The remote client defaults `/bulk_log` to a
  60-second HTTP timeout; `TrackioTrackedRun.trace_fact_update()` calls
  `Run.flush()` immediately before `Run.upsert_trace_facts()`, and both
  `Run.flush()` and `Run.upsert_trace_facts()` synchronously submit every queued
  trace batch.
  Evidence: the failed job's preserved Python stack terminates at
  `RemoteClient.predict(api_name="/bulk_log")` with `httpx.ReadTimeout`; live
  inspection found 236 GB free on Trackio control, 65 GB free on Doris, 93 GB
  free on object storage, and only 861.8 MB in the Trackio Doris database.

- Observation: the default TRL vLLM policy-parity threshold is a framework
  safety default, not a universal calibration for every model/template/runtime
  combination. Two Qwen 3.5 0.8B live rollout groups measured 0.064933 and
  0.062074 mean per-token log-probability deltas, including a no-MTP control.
  Evidence: the MTP and no-MTP local DAPO smoke jobs both failed before their
  first optimizer update at the default 0.050000 threshold. The qualification
  needs an explicit selection-level bound, recorded alongside its runtime
  evidence, rather than silently disabling the guard.

- Observation: local actual-job packaging currently materializes a full
  approximately 11 GiB `posttrain-local` image for each package key, and the
  Docker `type=docker` import needs substantial temporary free space. A host
  with 14 GiB free failed during final image import even though the immutable
  kind image already existed. Evidence: the rc16 DAPO launch failed with
  `no space left on device`; deleting only its failed container/image restored
  24 GiB and allowed the exact same candidate to launch. Follow-up runtime
  image work must make framework-owned package-image reuse and capacity
  preflight explicit before a build, rather than relying on manual cache
  cleanup.

- Observation: increasing the historical maintenance command from one 1,000
  trace page to a 5,000-trace in-memory page reduced command invocations but
  weakened progress visibility and recovery. The replacement pipeline retained
  the 5,000-trace orchestration window while checkpointing each 1,000-trace
  physical page. The live completion used five checkpoints followed by three
  checkpoints, exposed every safe cursor, and reached EOF without retaining
  more than one raw page.
  Evidence: the payload-free aggregate now reports 21,572 traces in 100 ordered
  `rollout_step` buckets; an exact read at cursor 21,572 returned zero records
  and no next cursor.

## Decision Log

- Decision: Posttrain's shared Verifiers evidence adapter owns semantic fact
  projection and historical recomputation; Trackio owns generic fact
  persistence and grouped reads; Observatory only requests and renders them.
  Rationale: GRPO, OPD, and evaluations share Verifiers semantics, while
  Trackio must remain reusable for other trace types and model families.
  Keeping a per-chart raw-payload scanner in Observatory duplicates parsing,
  but teaching Trackio Qwen or Verifiers rules puts domain semantics in the
  storage backend.
  Date/Author: 2026-08-14 / Codex

- Decision: the Verifiers bridges own lossless source capture and call one
  shared semantic projector at the producer boundary; Trackio owns generic fact
  persistence at the tracking boundary. `VerifiersEnvironmentRolloutBridge`
  (GRPO, OPD, and SAMPO) and the evaluation `VerifiersTraceSynchronizer` must
  both emit the same typed fact envelope for model, token usage, tool calls,
  truncation, error state, trace schema, and any valid training attribution.
  Rationale: the bridges are the only place to guarantee every live Verifiers
  record reaches tracking with its provenance. Calculating durable facts there
  would duplicate logic between train and eval, miss historical records and
  external sync paths. A historical backfill must call the same shared
  projector used by live bridges, then upsert the resulting generic facts.
  Date/Author: 2026-08-14 / Codex

- Decision: extend Trackio's existing physical `traces` row with the current
  one-to-one scalar projection rather than creating a separate `trace_facts`
  table or a database materialized view. Add nullable `fact_*` columns for the
  calculator version/state, normalized dimensions, numeric measures, and
  calculation time. Keep the native Verifiers `payload` unchanged. Store only
  unbounded one-to-many reward components in a normalized
  `trace_reward_components` child table.
  Rationale: every scalar fact has exactly the same identity and lifecycle as
  its retained trace. Keeping those values on `traces` avoids an unnecessary
  join, reuses the existing trace key and deletion/export lifecycle, supports
  efficient run/step indexes, and permits an atomic SQLite write. Dynamic
  reward names still require child rows because adding one column per reward
  function would not remain generic. Neither structure is a materialized view:
  Posttrain supplies the projection at ingest or explicit backfill time.
  Date/Author: 2026-08-14 / Codex

- Decision: expose a restricted, typed grouping API over approved dimensions
  and numeric facts; do not provide arbitrary JSON-path or SQL aggregation.
  Rationale: callers need reusable grouping and calculations, but arbitrary
  payload expressions would be unsafe, unindexable, impossible to keep
  equivalent across SQLite and Doris, and would reintroduce raw JSON parsing
  at read time.
  Date/Author: 2026-08-14 / Codex

- Decision: `posttrain.environment` owns the pure Verifiers-record fact
  projector because it already owns portable Verifiers activation/binding
  contracts shared by train and eval. `posttrain.common` adds only a generic
  typed fact envelope to `TraceObservation`; it does not learn Verifiers or
  model-template semantics. Trackio receives that envelope and never parses the
  native record to invent facts.
  Rationale: this keeps one calculation for GRPO, OPD, SAMPO, and eval without
  making train import eval, eval import train, common import Verifiers, or the
  tracking backend own model semantics. Unknown thinking conventions remain
  missing, never estimated from text or represented as zero.
  Date/Author: 2026-08-14 / Codex

- Decision: support existing historical model families explicitly in the
  shared Posttrain projector, beginning with Qwen 3.5, after inventorying the
  retained model identifiers. Add a new
  rule only with fixtures that prove its tokenizer/template boundary.
  Rationale: this solves the present corpus without prematurely creating a
  broad template-plugin system. `calculator_version` and `thinking_method`
  retain how each fact was obtained.
  Date/Author: 2026-08-14 / Codex

- Decision: provider-reported reasoning usage is the default compatibility
  path for a new model and needs no model-specific rule. A fallback
  `ThinkingTokenRule` is registered only when retained native traces lack that
  fact, and it matches an immutable semantic identity: model family, tokenizer
  revision, renderer revision, template revision, and trace schema version.
  Historical Qwen records may retain a narrowly tested legacy model-name match
  because those older traces predate the full identity envelope.
  Rationale: adding a model must be an explicit compatibility addition rather
  than a growing chain of UI or storage heuristics. Every fallback rule ships
  complete, truncated, ambiguous-boundary, and structured-tool fixtures;
  changing its output requires a calculator-version bump and scoped backfill.
  Date/Author: 2026-08-14 / Codex

- Decision: retain multi-reward evidence at two distinct grains. The core
  scalar `task_reward` is the Verifiers algorithm input (the sum of valid weighted
  contributions), while `reward.component` is a long-form measure group keyed
  by the exact component name. When a historical/native record exposes raw
  score and weight separately, retain those in `reward.score` and
  `reward.weight`; never reconstruct them from a weighted contribution.
  Downstream shaping emits a separate `algorithm_reward` fact after it is
  calculated. Advantages, returns, and normalization statistics retain their
  actual token/turn/group grain and are not mislabeled as trace rewards.
  Rationale: multi-reward GRPO, OPD, SAMPO, and evaluation need component-level
  coverage and trends, while one sum cannot explain reward hacking, component
  conflict, sparse signals, or a change in configured weights.
  Date/Author: 2026-08-14 / Codex

- Decision: persist native facts and trainer-owned reward shaping in two
  explicit phases. The shared Verifiers projector emits the initial
  `verifiers.trace` projection with task reward and components. After TRL or
  veRL calculates its per-trajectory shaped value, the training backend emits
  a scalar-only `posttrain.train.reward` enrichment keyed by the same external
  trace ID. Trackio merges that approved scalar without rewriting the native
  payload or replacing the source projection. OPD and evaluation leave
  `algorithm_reward` null unless their selected algorithm genuinely produces
  a trace-grain value. Historical backfill only reconstructs this enrichment
  when retained trainer output has an exact trace-ID join.
  Rationale: the environment projector cannot own a value that is calculated
  later from training settings. Delaying or rewriting native trace capture
  would weaken evidence retention, while falsely equating task and algorithm
  reward would erase DAPO shaping. A trace-keyed second phase preserves both
  meanings and their true owners.
  Date/Author: 2026-08-14 / Codex

- Decision: separate semantic calculator version from physical projection
  identity. `calculator_version` states which calculation rules were used;
  `projection_id` is the lowercase SHA-256 digest of the namespace,
  calculator version, and canonical JSON fact payload, excluding calculation
  time. Posttrain computes the digest and Trackio verifies it. The parent trace
  points at its current source `projection_id`; component rows use that digest
  rather than calculator version in their logical key. The trainer enrichment
  has its own namespace, calculator version, projection ID, and calculation
  time.
  Rationale: identical retries become idempotent, a changed payload receives a
  new immutable generation even under the same semantic version, and staged
  Doris child rows cannot leak omitted components from a prior attempt.
  Date/Author: 2026-08-14 / Codex

- Decision: normalize token facts with unambiguous names and a subset
  invariant. `model_input_tokens` is the sum of provider-observed model input
  tokens across recorded calls. `model_output_tokens` is every generated
  assistant token, including reasoning. `thinking_tokens` is a subset of that
  output; `answer_tokens` is computed only when both facts share a compatible
  call and tokenizer scope. Prefer retained generated token IDs for total
  output, otherwise use provider usage only when its contract includes
  reasoning; when a provider reports visible and reasoning tokens separately,
  sum them for total output. If the projector cannot prove compatible
  accounting or obtains `thinking_tokens > model_output_tokens`, it keeps the
  thinking value null, marks the projection partial, and records the method in
  provenance. `tool_calls` counts structured assistant tool-call objects,
  `model_calls` counts recorded provider calls, and `trace_latency_ms` is
  end-to-end trace duration rather than summed model latency.
  Rationale: the persisted schema and UI must not mix answer-only and total
  generated-token populations or manufacture a mathematically impossible
  relationship.
  Date/Author: 2026-08-14 / Codex

- Decision: Trackio exposes exactly three trace-fact operations:
  `Run.upsert_trace_facts`, `Run.aggregate_trace_facts`, and the optional fact
  envelope on an initial `VerifiersTrace` write. The remote server exposes
  `/upsert_trace_facts` and `/get_trace_facts`. Requests and responses use the
  exact types `TraceFactUpdate`, `TraceFactWriteReceipt`, `TraceFactsQuery`,
  `TraceAggregate`, `TraceAggregateBucket`, and `TraceAggregateResult`; these
  are contracts, not illustrative names. The initial source projection may
  replace its complete component set, while trainer enrichment is scalar-only.
  Rationale: one generic idempotent upsert serves live ingest, post-shaping
  enrichment, and historical backfill without adding Posttrain semantics to
  Trackio or creating provider-specific endpoints.
  Date/Author: 2026-08-14 / Codex

- Decision: retain the synchronous `Run.upsert_trace_facts()` operation for
  explicit read-after-write callers, but add a separate asynchronous
  `Run.enqueue_trace_facts()` operation for normal live observation. In Doris
  asynchronous mode, all native trace metric batches and queued fact updates
  become immutable server-inbox fragments. The importer writes native traces
  before dependent fact fragments, retries a temporarily missing parent without
  deleting its source fragment, and applies the fact through the existing
  idempotent storage method. The Posttrain Trackio adapter uses this asynchronous
  operation for trainer enrichment and never calls `flush()` on the training
  critical path.
  Rationale: source traces and algorithm reward are separate in time, but
  neither requires immediate query visibility during an optimizer update. A
  durable server acknowledgement gives the producer a bounded request while the
  importer preserves parent-before-enrichment ordering. Keeping the synchronous
  API avoids breaking explicit maintenance/read-after-write callers.
  Date/Author: 2026-08-16 / Codex

- Decision: machine configuration owns service credentials, provider bindings,
  and scheduling defaults; project-local protected state may override only the
  digest-pinned `[registry]` runtime selection. Local execution validates the
  selected configured binding rather than assuming the stable shipped manifest.
  Rationale: a retained candidate must be trialed by one project without
  copying credentials into project state or changing every job on the machine.
  This preserves reproducibility and allows an explicit candidate rollback by
  removing one project-local selection.
  Date/Author: 2026-08-15 / Codex

- Decision: make trace facts a deliberate Trackio database schema version 2
  change. Bump the global Doris `SCHEMA_VERSION` from 1 to 2, add the extended
  trace columns and `trace_reward_components` to the version-2 bootstrap
  schema, and implement an explicit version 1 to version 2 migration. Do not
  add independent capability-version negotiation. The candidate deployment
  runs the migration and server upgrade as one coordinated operation; an old
  server must not be restarted against the migrated database.
  Rationale: this is an owned database change with one deployed Trackio
  service. We intentionally choose the simpler global contract over a
  conservative per-capability compatibility layer: every deployed Trackio
  service must understand the version-2 schema. Explicit migration, backup,
  verification, and repair-forward recovery still provide operational safety
  without keeping two database semantics alive.
  Date/Author: 2026-08-14 / Codex

- Decision: future extensibility has two deliberately separate seams. A new
  model, template, or provider accounting convention adds a tested
  `ThinkingTokenRule` (or relies on the provider-usage path) in Posttrain and
  bumps the semantic calculator version when its output changes. A new reward
  signal adds a `TraceRewardComponent` with its declared `SignalSource`; it
  does not add a Trackio column. A new scalar fact follows the same typed
  schema-version workflow as any other Trackio storage change.
  Rationale: model/template semantics and reward names evolve independently
  from storage topology. This keeps the database generic, makes a model
  addition explicit and testable, preserves multi-reward detail without a
  widening table, and avoids treating optional feature discovery as a second
  schema system. All deployed Trackio servers use schema v2; no caller may
  negotiate an older trace-fact capability after the release cutover.
  Date/Author: 2026-08-14 / Codex

- Decision: model judge and reward are represented through one lightweight
  scored-signal abstraction, not a judge-execution entity. Each named reward
  component carries a `SignalSource` with `kind` equal to `llm_judge`,
  `deterministic`, `human`, `environment`, `group`, `teacher`, `composite`, or
  `unknown`, plus an optional stable semantic source ID. “Human” means an
  actual manually supplied label; ordinary verifier code is `deterministic`.
  Reward remains a use of the signal: score, optional weight, and weighted
  contribution. The source classification does not change reward arithmetic.
  Rationale: this answers whether a reward was model-generated, human-provided,
  or programmatic without coupling storage and UI to the implementation details
  or operational cost of a judge call.
  Date/Author: 2026-08-14 / Codex

- Decision: do not project, persist, aggregate, or display judge execution.
  The fact path must not read `trace.info["judge"]` or `trace.extra_usage` to
  create judge calls, and it must not store judge prompts, responses, parsed
  verdict payloads, model identity, call count, tokens, latency, or cost. The
  untouched native Verifiers trace remains replay authority and may already
  contain some of this source data, but Trackio facts and Observatory do not
  duplicate it. Historical components without an explicit immutable source
  declaration use `unknown`; never join judge entries to rewards by position or
  infer LLM provenance from a reward name.
  Rationale: component provenance is needed for interpretation, while detailed
  judge execution would add high-cardinality data, ambiguous joins, storage
  cost, and product surface that this work does not need.
  Date/Author: 2026-08-14 / Codex

- Decision: aggregate results always return full-population and per-measure
  coverage. Null facts are excluded from a measure's mean and reported as
  missing; they are never zero-filled.
  Rationale: evidence completeness is required for training and evaluation
  interpretation, and missing thinking support must remain visible.
  Date/Author: 2026-08-14 / Codex

- Decision: materialize facts for truncated traces exactly as for completed
  traces, retaining `is_truncated` as a dimension rather than filtering the
  trace out. Model output/input tokens, model calls, latency, task reward, and every
  structured tool call observed before the terminal cap remain part of the
  extended trace row and its aggregate denominator.
  Rationale: a cap changes the interpretation of an observation; it does not
  erase the response, its token use, or tool activity. Consumers can group or
  filter on `is_truncated` and must receive its count and coverage alongside
  an all-trace aggregate.
  Date/Author: 2026-08-14 / Codex

- Decision: provider-reported `reasoning_tokens` is authoritative even for a
  truncated response. A Qwen 3.5 fallback may count an unfinished sampled
  suffix only when the final sampled assistant node is proven to be the
  terminal length-capped model call and a fixture proves the tokenizer/template
  semantics. In every other no-end-boundary case, persist null rather than
  estimating from rendered text.
  Rationale: this recovers accurate partial reasoning where native token
  evidence proves it, while preventing an unterminated response, tool JSON, or
  ordinary output from being misclassified as thought.
  Date/Author: 2026-08-14 / Codex

- Decision: treat 5,000 traces as a bounded orchestration window, not as one
  physical read, write, or checkpoint. Pipeline at most 1,000 native payloads
  at a time, write that page through Trackio's generic batch endpoint, emit its
  sanitized receipt and next cursor immediately, then continue until the
  window limit or end of run. Reusing one source and writer session avoids
  process churn without retaining five pages or hiding partial progress.
  Rationale: Trackio's Doris fast path and the provider trace reader are already
  bounded at 1,000. Preserving that boundary gives reliable retries and
  observable cursors while one command can still process 5,000 traces.
  Date/Author: 2026-08-14 / Codex

## Outcomes & Retrospective

Milestones 1 through 3 are implemented and qualified. Training and evaluation
attach identical versioned Verifiers facts; signal-source metadata and reward
components are retained without judge execution telemetry; Trackio SQLite and
Doris persist and aggregate the projection; Observatory prefers the generic
aggregate; and real local DAPO and evaluation jobs prove the producer path.
Trackio dev15 is manually released, hash-retained, published to the development
channel, and deployed to both Doris services.

The end-to-end outcome remains incomplete only at the final release boundary.
The historical 100-step, retained OPD, and evaluation reads are qualified; the
manually released Trackio dev15 and its exact Posttrain pin are selected, and
the runtime lock/profiles now derive all image dependency bytes from that exact
workspace resolution. The remaining work is to materialize the next candidate
base/kind graph, prove first-install and partial-registry reuse against the
live registry, then run the bounded actual-job image proof before stable
promotion. The rollout view has no compatibility payload scan: it uses the
trace-facts aggregate or reports unavailable. No new GPU training job is
required unless an image or live integration gate exposes a defect.

## Context and Orientation

The work spans two repositories that must remain separately commit-able:

- `/home/hammad/projects/trackio` is the CarbonTeq Trackio fork. Its
  `trackio/trace.py` preserves a full native Verifiers record,
  `trackio/sqlite_storage.py` and `trackio/doris_storage.py` persist/read it,
  `trackio/api.py` exposes `Run.traces`, and `trackio/server.py` exposes remote
  read functions. `CARBONTEQ_FORK.md` is the required fork ledger.
- `/home/hammad/projects/rl` is the consuming Posttrain workspace. Its
  `packages/tracking` owns provider-neutral evidence-reader models/contracts;
  `packages/tracking-trackio` translates Trackio reads; and
  `apps/observatory` owns read-only job-aware views and the frontend.

`VerifiersTrace` means Trackio's queryable copy of a native Verifiers rollout.
The complete original record remains in `payload`; final-branch messages are a
display projection. A *trace fact* is a scalar such as total generated-token count,
tool-call count, or training optimizer step, calculated from that stored record
and persisted in nullable columns on the same Trackio `traces` row. Named reward
components are the only normalized child records because one trace may contain
an arbitrary number of them. A *trace-facts aggregation* groups trace rows and,
when requested, their current-projection reward-component rows to calculate a
small approved set of operations without reading the raw JSON payload.

A *source projection* is the complete set of native facts that can be calculated
when the Verifiers trace finishes. A *trainer enrichment* is a later scalar-only
fact update keyed by the same external trace ID after a training backend has
calculated a per-trajectory value such as shaped algorithm reward. Both carry a
semantic `calculator_version` and deterministic `projection_id`; neither changes
the native payload. A projection ID is the lowercase SHA-256 digest of canonical
JSON containing the namespace, calculator version, dimensions, measures,
components, provenance, and projection state. The canonical payload sorts object
keys and reward components by name, uses compact UTF-8 JSON, rejects non-finite
numbers, and excludes `calculated_at`.

There are two producer paths today. Training uses
`packages/train/.../integrations/verifiers.py:VerifiersEnvironmentRolloutBridge`,
which preserves the native record before trainability validation and emits a
live `TraceObservation`; it already adds model, truncation, error, and training
attributes. Evaluation uses
`packages/eval/.../backends/verifiers/synchronization.py:VerifiersTraceSynchronizer`,
which streams the same native JSONL records after an eval starts but currently
does not add the equivalent trace attributes. Both ultimately converge in the
Lab Trackio observer as `trackio.VerifiersTrace`. This plan makes them call one
projector in `posttrain.environment` and carry one generic fact envelope before
Trackio stores the durable facts.

The initial supported facts are shared across all Verifiers consumers:

- identity/dimensions: physical Trackio trace step, normalized
  `rollout_step` when the trace is a training rollout, model identifier, task
  type, completion/truncation/error state;
- numeric values: model input tokens, total model output tokens including
  thinking, thinking tokens, tool calls, model calls, end-to-end trace latency,
  and scalar task reward;
- long-form reward components: exact name, nullable score, nullable weight,
  nullable weighted contribution, bounded source kind, and optional stable
  semantic source ID;
- provenance: `calculator_version`, `thinking_method`,
  `tool_call_method`, and `projection_state` (`complete`, `partial`, or
  `unsupported`).

`rollout_step` is null for evaluation traces by design. It is normalized from,
in order, a current explicit `optimizer_step` field, an explicit payload
equivalent, or the historical `payload.run.step` only when
`payload.run.type == "train"`. The Trackio logging row's ordinary `step` must
not be silently treated as the optimizer step.

A *signal source* is small semantic metadata attached to one numeric reward
component. Its `kind` answers whether the value came from an LLM judge,
deterministic code, a human, the environment, a group scorer, a teacher, a
composite calculation, or an unknown historical source. Its optional `id`
names the scoring rule, such as `reference_judge` or `format_check`; it never
identifies one execution. Signal source is declared by the immutable
environment or job configuration and carried into trace facts. It is not
recovered from judge response text, array order, or naming heuristics.

The existing `EvaluationSignalRef` in
`packages/eval/src/posttrain/eval/requests.py` remains the small reference used
by evaluation success predicates. Signal-source metadata complements that
reference; it does not create another success or reward definition. A signal
can remain a metric, become a reward component, or be selected by a success
predicate without changing its producer classification.

Thinking-token calculation in the shared Verifiers projector is deliberately
small and explicit:

1. use summed provider `usage.reasoning_tokens` when present;
2. otherwise apply a registered existing-model rule to retained generated
   token IDs and sampled masks; the first rule is Qwen 3.5's verified native
   `</think>` boundary. For the final node of a length-capped response, the
   rule may count the sampled suffix through the retained terminal token only
   after an explicit fixture proves that the unfinished segment is thought;
3. otherwise persist null plus `thinking_method = "unsupported"`.

Tool-call calculation is also projector-owned and schema-based: prefer an explicit
native count when present, otherwise count structured
`nodes[].message.tool_calls`. Text that merely looks like tool JSON is not a
tool call. Neither token accounting nor tool-call counting excludes a trace
because it is truncated.

## Plan of Work

### Milestone 1 — Define the shared Verifiers projector and observation contract

In `/home/hammad/projects/rl/packages/environment`, use the implemented
`posttrain.environment.verifiers_evidence` module to parse pure Python mappings
and produce a generic `TraceFactSet`. It does not import the Verifiers runtime,
Trackio, a tokenizer library, a trainer, or an ORM. Its
`ThinkingTokenRule` protocol is the sole fallback extension seam for a model
whose provider usage omits reasoning tokens. `Qwen35ThinkingTokenRule` is the
first registered rule; unsupported or ambiguous values remain `None`. The
calculator version changes whenever projector semantics change.

In `posttrain.common`, extend `TraceObservation` with a provider-neutral fact
envelope containing a namespace, calculator version, scalar dimensions,
numeric measures, long-form named measure groups, per-field provenance, and
projection state. This is the only cross-cutting addition. Common does not
define Verifiers field names or perform calculation.

Before Trackio persistence, refine the current in-tree long-form envelope with
two provider-neutral values in
`packages/common/src/posttrain/common/execution.py`:

    SignalSourceKind = Literal[
        "llm_judge", "deterministic", "human", "environment",
        "group", "teacher", "composite", "unknown",
    ]

    @dataclass(frozen=True, slots=True)
    class SignalSource:
        kind: SignalSourceKind
        id: str | None = None

    @dataclass(frozen=True, slots=True)
    class TraceRewardComponent:
        name: str
        contribution: float | None
        score: float | None = None
        weight: float | None = None
        source: SignalSource = SignalSource("unknown")

Change the current `TraceFactSet.measure_groups` nested numeric mapping to a
`reward_components: tuple[TraceRewardComponent, ...]` field while it remains
unpublished. Keep scalar `dimensions` and `measures` provider-neutral; reward
components are typed because score, weight, contribution, and source provenance
must remain associated.
The component name is exact bounded text; source ID is a stable semantic rule
identifier, not an execution ID. Validate finite numeric fields, immutable
collections, known source kinds, and the invariant that contribution equals
`score * weight` within a documented floating-point tolerance when all three
are present. Do not require score or weight for
pinned historical traces that retain only weighted contributions.

In `packages/environment/src/posttrain/environment/requests.py`, add an
optional immutable mapping from declared `reward_components` to
`SignalSource`, and expose the same field through the environment catalog
schema. Existing declarations without source metadata remain valid and project
as `unknown`. Reject source declarations for names absent from
`reward_components`. Pass this declaration explicitly into the shared
projector from both training and evaluation; do not hide it in a UI mapping.

The training `VerifiersEnvironmentRolloutBridge` and evaluation adapter now
call the same projector. `ModelVariant.trace_identity()` supplies model family,
model revision, tokenizer fingerprint, renderer ID, and template identity at
the producer boundary. Training adds its optimizer step before emission;
evaluation leaves training-only dimensions absent. The untouched native record
remains `TraceObservation.payload`. veRL and persisted host-side replay still
need parity coverage before Milestone 1 is considered completely qualified.

Add `TraceFactUpdateObservation` to
`packages/common/src/posttrain/common/execution.py`, add
`Observer.trace_fact_update(observation)` to the smallest observation protocol,
and add `RunContext.trace_fact_update(observation)` as its cancellation-aware
forwarder. The value carries `trace_type`, `external_id`, one scalar-only fact
update, and ordinary attributes. It never carries or mutates the native payload.
Every observer implementation must either persist the update or report the
capability unsupported; silently dropping it is invalid.

In the TRL GRPO/SAMPO backend, emit one `TraceFactUpdateObservation` for each
rollout immediately after `shaped_rewards` is calculated, pairing by the
existing rollout order and external trace ID. In the isolated veRL backend,
keep `rollout_trace_id` and `algorithm_reward` in the worker result, then add a
parent-process `_replay_trace_fact_updates` beside `_replay_grpo_metrics` so the
trusted parent emits the same observation after reading the completed result.
Do not make the Verifiers bridge import or accept training settings. The
tracking observer preserves trace-before-enrichment ordering; a remote adapter
flushes any buffered initial trace before submitting its enrichment and retries
a missing-trace receipt with the existing bounded tracking retry policy.

Before Trackio work begins, rename the current projected measures to the exact
public vocabulary: `model_input_tokens`, `model_output_tokens`,
`thinking_tokens`, `tool_calls`, `model_calls`, `trace_latency_ms`, and
`task_reward`. `model_output_tokens` includes reasoning and every other sampled
assistant token. Prefer retained generated token IDs for that total. When only
provider usage is available, accept a completion total only when the provider
contract includes reasoning; if it reports visible and reasoning tokens as
disjoint values, add them. Persist `thinking_tokens` only when it is proven to
share the same call/tokenizer population and does not exceed total output.
`answer_tokens` is a computed consumer value and is null unless subtraction is
valid on the same population; it is not a stored v1 fact.

Add parity fixtures proving the same native record produces the same shared
facts through GRPO/OPD, SAMPO, eval, veRL replay, and host-side replay. Add exact
unit fixtures for a non-thinking record, structured tool calls,
provider-reported reasoning usage, Qwen 3.5 token-boundary thinking, an
unknown-model record, a current `optimizer_step`, a historical training
`run.step`, and an evaluation `run.step` that must remain null. Add terminal
truncation fixtures for: a length-finished final call with usage, a prior tool
call followed by a length-finished final call, a provider-reported partial
reasoning count, a Qwen 3.5 unfinished thought with a verified sampled-token
boundary, and an ambiguous no-end-boundary record that must keep thinking
tokens null. Assert that no truncated trace is silently removed or zeroed.
Add reward fixtures for deterministic, LLM-judge, human, group, teacher,
composite, and unknown sources; multi-component mappings; pinned
contribution-only records; structured score/weight records; and a source
declaration whose name is absent from the observed trace. The last case remains
missing rather than creating a zero component. Add a negative fixture proving
that `trace.info["judge"]` and `trace.extra_usage` populate no scalar fact
columns, reward-component rows, or measures.

### Milestone 2 — Persist and aggregate generic facts in Trackio

In `/home/hammad/projects/trackio`, define the exact public values
`TraceFactUpdate`, `TraceFactWriteReceipt`, `TraceFactsQuery`,
`TraceAggregate`, `TraceAggregateBucket`, and `TraceAggregateResult` in
`trackio/trace.py`, and export them through `trackio/__init__.py`.
`TraceFactUpdate` contains `trace_type`, `external_id`, `namespace`,
`calculator_version`, `projection_id`, `calculated_at`, optional scalar
dimensions, optional numeric measures, an optional complete tuple of
`TraceRewardComponent` values, provenance, and projection state. A boolean
`replace_reward_components` is legal only for the `verifiers.trace` source
projection. A `posttrain.train.reward` update is scalar-only and may supply
only `algorithm_reward`. Trackio validates names, scalar types, finite numbers,
the SHA-256 projection identity, and component replacement semantics but does
not import Posttrain, Verifiers, tokenizers, trainers, or model-template rules.

`TraceFactsQuery` accepts only these fields:

    trace_type: str = "verifiers"
    filters: mapping of approved equality filters
    group_by: tuple of approved dimensions
    measures: tuple of approved aggregate requests

The approved dimensions are `rollout_step`, `trace_step`, `model`,
`task_type`, `is_truncated`, and `has_error`. The first approved numeric facts
are `model_input_tokens`, `model_output_tokens`, `thinking_tokens`, `tool_calls`,
`model_calls`, `trace_latency_ms`, `task_reward`, and `algorithm_reward`. Named
`reward.component` values are queried by exact component name or as an
explicit all-components expansion; component coverage is returned separately.
The first operations are `count`, `sum`, `mean`, `min`, and `max`. Reject
unknown fields or operations at the API boundary. Do not implement a JSON-path
language.

Each aggregate bucket returns its grouped dimensions, `trace_count`, and for
each requested numeric measure: value, observed-count, and missing-count. The
top-level result returns matching trace count, projected-trace count,
unavailable-trace count, calculator versions observed, and
`complete | partial | unavailable`.
This makes an evaluation aggregation valid without a rollout step and lets a
GRPO/OPD view group by it.

Extend the existing `traces` table in `trackio/sqlite_storage.py` and
`trackio/doris_schema.py`. Keep its current logical identity—SQLite `id`, and
Doris `(project_id, trace_id)`—and add these nullable columns:

    fact_projection_id
    fact_calculator_version
    fact_projection_state
    fact_calculated_at
    fact_rollout_step
    fact_model
    fact_model_family
    fact_task_type
    fact_is_truncated
    fact_has_error
    fact_model_input_tokens
    fact_model_output_tokens
    fact_thinking_tokens
    fact_tool_calls
    fact_model_calls
    fact_trace_latency_ms
    fact_task_reward
    fact_algorithm_reward
    fact_algorithm_projection_id
    fact_algorithm_calculator_version
    fact_algorithm_calculated_at

Add the remaining approved provenance fields, such as thinking and tool-call
method, with the same prefix. Do not add a second JSON copy of the fact
envelope: the typed columns and reward child rows are query authority, while
`payload` remains native replay authority. Existing traces keep every new
column null and remain readable.

Use `TEXT` in SQLite and bounded `VARCHAR` in Doris for identifiers and
provenance: 64 characters for projection IDs, 128 for calculator versions, 16
for projection state, 64 for ISO-8601 UTC calculation time, 1,024 for model
identity, 128 for model family and task type, and 64 for method fields. Use
SQLite `INTEGER` and Doris `BIGINT` for steps and token/call counts; SQLite
`INTEGER` constrained to zero or one and Doris `BOOLEAN` for flags; and SQLite
`REAL` and Doris `DOUBLE` for latency and rewards. Reject negative counts,
negative latency, non-finite numeric values, invalid state values, and a
thinking-token value larger than total model output before storage.

Add a normalized `trace_reward_components` table in both storage engines. Its
logical key is `(project_id, trace_id, projection_id, component_name)` in
Doris and the equivalent trace/projection/name key in SQLite. Each row also
carries `run_id` as deliberate query denormalization, plus nullable score,
weight, and contribution, `source_kind`, and optional `source_id`. Use 255
characters for project and run IDs, 768 for trace ID, 64 for projection ID, 256
for component name and source ID, 32 for source kind, and `DOUBLE`/`REAL` for
the numeric values. This avoids adding one database column per reward function
and preserves exact component names without lossy identifier normalization.
Aggregate queries join only rows whose projection ID equals the parent trace's
`fact_projection_id`, return observed and missing population counts for every
requested component, and never treat an absent component as zero. Source
filters and groupings use the bounded source kind; source ID is returned for
display but is not an unrestricted grouping field in the first API.

For SQLite, update the trace row and replace the current-projection component rows
inside one transaction. Add a foreign key to the trace identity where existing
SQLite migration constraints allow it, plus indexes on
`(run_id, trace_type, fact_rollout_step)` and current component lookups. For
Doris, write all component rows for a new projection ID first and upsert
the parent trace's scalar columns and `fact_projection_id` last. Readers
ignore an incomplete staged generation because it is not the parent's current
projection. Retain old generations until a bounded cleanup confirms no parent
references them; these rows are recovery data, not user-visible history.

Make this a global Doris schema version 2 migration. In
`trackio/doris_schema.py`, set `SCHEMA_VERSION = 2`, add
`trace_reward_components` to `MANAGED_TABLES`, and make the bootstrap DDL
create the complete version-2 shape. Add
`trackio/doris_migrations.py:migration_statements(from_version, to_version)`
with one supported path, `1 -> 2`: add every nullable trace column, create the
component table, add the run/trace lookup structures validated for the deployed
Doris version, verify the resulting columns and table, and update the
`schema_versions` row last. Add `trackio storage migrate-doris --to 2
--preview|--apply` in the Trackio CLI. `--preview` prints the current and target
versions plus ordered DDL without changing storage; `--apply` requires a
pre-migration backup receipt, records each completed statement, and is safely
retryable through `IF NOT EXISTS` plus schema inspection. Startup continues to
refuse a lower schema version and tells the operator to run this command.

Treat migration and candidate deployment as one coordinated change. Pause
Trackio writers, take and verify a Doris backup, run preview, apply version 2,
start the candidate server, verify ordinary run/trace reads plus trace-fact
capability, and then resume writers. The old server is not supported after the
database records version 2. Recovery before writer resumption restores the
version-1 backup and old server; after writer resumption, repair forward with a
new migration rather than attempting destructive down-migration.

Update every existing trace lifecycle path, not only live logging:
`trackio/sqlite_storage.py` local export/import and run/project deletion,
`trackio/doris_storage.py` insert/import/read/delete paths, and
`trackio/doris_migration.py` SQLite-to-Doris migration. Scalar columns travel
with the parent trace; component rows travel and delete with that trace. Older
Parquet exports lacking these columns import with null facts. A new export must
carry both the extended trace rows and a separate reward-component Parquet
relation so round trips do not silently discard provenance.

Do not add a judge table or judge-specific endpoint. Trackio does not inspect
`trace.info["judge"]` or `trace.extra_usage`, and the fact schema contains no
judge prompt, response, parsed payload, model, execution ID, calls, tokens,
latency, or cost. Generic source metadata arrives only on the supplied
`TraceRewardComponent`.

When `SQLiteStorage` or `DorisStorage` writes any trace carrying an initial
fact envelope, upsert scalar values onto that trace row and persist its complete
reward-component set in the same logical operation. Implement
`Run.upsert_trace_facts(updates: Sequence[TraceFactUpdate]) ->
TraceFactWriteReceipt` for later trainer enrichment and historical backfill.
Resolve each update by `(run_id, trace_type, external_id)`, reject zero or
multiple matches, verify the projection ID, and merge only the supplied
namespace's approved fields. The receipt returns attempted, inserted, updated,
unchanged, missing-trace, and rejected counts plus a resumable cursor when the
server accepted a bounded page. Existing traces without facts remain valid.
Trackio does not derive or alter measures from the raw payload; a partial
producer fact set persists its known values and explicit missing/projection
state.

Implement the aggregate query in SQLite and Doris over indexed `traces`
columns and current-projection `trace_reward_components`, returning the identical
shape and coverage semantics. Add
`Run.aggregate_trace_facts(query: TraceFactsQuery) -> TraceAggregateResult`
and remote `/get_trace_facts` and `/upsert_trace_facts` endpoints through
`trackio/api.py` and `trackio/server.py`. Do not add raw SQL access. The read
must not select
`traces.payload`, `nodes`, `calls`, messages, or token arrays.

Add a generic, idempotent fact-upsert API. The Posttrain maintenance command
owns historical backfill: it reads bounded pages of retained native Verifiers
records through the provider-neutral reader, calls the same projector from
Milestone 1, and writes `TraceFactUpdate` values through the Trackio adapter. It
must have preview/apply modes, explicit project/run/trace-type scope, a bounded
page size, a resumable cursor, and counts by projection state and thinking
method. Repeating a completed page safely upserts the same facts and never
changes the raw payload, messages, metadata, or original trace timestamp.

Add SQLite/Doris parity tests covering additive schema migration from a database
with the old `traces` shape, write-time projection onto the trace row, aggregate
coverage, current/historical training step normalization, evaluation without a
rollout step, rejected unapproved aggregate fields, resumable backfill, and
truncated-versus-complete buckets. The latter must prove that all observed
facts survive a terminal cap and that each optional measure reports its own
known/missing coverage. Assert that existing trace IDs, payloads, messages,
metadata, timestamps, exports, imports, and deletion behavior remain unchanged.
Add component parity cases covering every source kind, exact semantic source
IDs, contribution-only historical rows, structured score/weight rows, missing
components, and rejection of any attempted judge-execution payload.

### Milestone 3 — Consume the generic capability through Posttrain

In `/home/hammad/projects/rl/packages/tracking/src/posttrain/tracking`, add
provider-neutral aggregate models to the raw reader contract. The contract
represents the logical grouping query and result, not Trackio table names.
Extend reader capabilities so a backend can report trace-fact aggregation as
available, unsupported, or unavailable. The W&B reader must explicitly report
unsupported rather than pretending an aggregate is zero.

In `packages/tracking-trackio`, translate the provider-neutral request to
`Run.aggregate_trace_facts(...)`, preserving exact coverage/status. Tests must prove the
adapter does not fall back to downloading raw trace payloads when the Trackio
API is available. During the compatibility window only, detect a pre-facts
Trackio server and report unsupported cleanly.

In `apps/observatory`, replace the `rollout_behavior_view` full-payload scan
with one trace-facts aggregate query grouped by `rollout_step`. Use it for the
existing GRPO Policy optimization internal panel and make the same projection
available to OPD runs when their job telemetry selects rollout behavior.
Evaluation views can use the same aggregate API without `rollout_step`, for
example grouping by task type or completion state; no evaluation pass/fail
meaning is added. Once the minimum Trackio version is released and deployed,
remove the raw-payload fallback. The implemented view now does so: it reports
an explicit unavailable state when facts are unavailable. Make partial coverage
visible in the chart footer and never render missing thinking tokens as zero.

In reward views, add a compact source label for each component: LLM judge,
deterministic, human, environment, group, teacher, composite, or unknown. Show
score, weight, and contribution only when each value is available. Do not add a
judge panel, judge execution drill-down, or judge calls/tokens/latency/cost
chart. The raw trace viewer remains unchanged and is the only path that may
show native Verifiers content already retained in the source record.

Update Observatory fixtures and HTTP/frontend tests to prove three consumers:
GRPO groups by steps, OPD groups by steps, and evaluation groups by a
non-training dimension. The UI should make no Trackio-specific assumptions.
Add a mixed-source reward fixture proving the component table displays source
labels and distinguishes missing score/weight from numeric zero. Assert that
the Observatory response schema contains no judge-execution collection.

### Milestone 4 — Inventory, backfill, release, and qualify

Before applying a historical backfill, query retained Verifiers traces in each
target Trackio project to inventory model identifiers, trace schema versions,
and existing provider reasoning usage. Record the exact recognized model strings
and any unsupported population in the fork ledger and the Posttrain tooling
page. Do not infer a template rule from model name alone without a fixture that
proves the stored token boundary.

Start with the target projects that retain durable GRPO, OPD, or evaluation
evidence (including `ambient-agent` and `posttrain-lab` when their data remains
in scope). Run preview first, retain its sanitized counts, then apply the
bounded backfill page-by-page until completion. Verify that the historical
Ambient Agent 100-step run returns 100 `rollout_step` aggregate buckets without
returning full raw payloads. Verify at least one OPD run and one evaluation run
with the same API.

Publish the fork manually, following `docs/tooling/forks.md`: commit and push
Trackio; build and validate release assets locally; create an immutable GitHub
release with retained hashes; then dispatch Posttrain's retained-asset publisher
to `carbonteq/dev`. No fork release runner or fork-controlled publisher is
added. Update Posttrain to the exact development candidate, lock it, and run
the integration qualification. After deployment/backfill evidence succeeds,
use the repository-owned promotion workflow to transfer identical bytes to
`carbonteq/stable`, then update the exact stable version, hashes, `uv.lock`, and
both fork documentation records.

The candidate deployment owns the Doris schema transition. Before switching
production traffic, pause writers, verify the retained backup receipt, run the
version-2 migration preview and apply commands, start the candidate against the
migrated database, and execute ordinary trace reads plus one source projection,
one algorithm-reward enrichment, and one grouped aggregate. Resume writers
only after those probes pass. Record the backup identity, migration output,
candidate image/release identity, probe receipt, and writer-resumption time as
release evidence.

## Concrete Steps

Run the Trackio implementation and focused tests from
`/home/hammad/projects/trackio`:

    uv sync --extra dev --extra spaces
    uv run pytest tests/unit/test_trace_facts.py tests/unit/test_api_reads.py tests/unit/test_storage_provider.py tests/unit/test_doris_schema.py tests/unit/test_doris_migration.py -q
    uv run ruff check trackio tests/unit

Run the storage-engine parity gate only where Doris is configured:

    TRACKIO_DATABASE_ENGINE=doris uv run pytest tests/integration/test_doris_storage.py -q
    trackio storage migrate-doris --to 2 --preview

Before applying the migration to a retained deployment, record and verify its
backup receipt, then run:

    trackio storage migrate-doris --to 2 --apply --backup-receipt <receipt>

Expect the command to report source version 1, target version 2, every applied
or already-present DDL step, successful schema verification, and the version-row
update as the final operation.

Run the Posttrain contract and Observatory tests from
`/home/hammad/projects/rl`:

    uv sync --all-packages --locked --python 3.13
    uv run pytest packages/environment/tests packages/train/tests/test_verifiers_grpo_bridge.py packages/eval/tests/test_trace_sync.py packages/tracking/tests packages/tracking-trackio/tests apps/observatory/tests/test_product_service.py apps/observatory/tests/test_http.py -q
    uv run ruff check packages/common packages/environment packages/train packages/eval packages/tracking packages/tracking-trackio apps/observatory
    uv run lint-imports
    git diff --check

Run the frontend checks from `/home/hammad/projects/rl/apps/observatory/frontend`:

    npm run generate:api
    npm test -- --run src/App.test.tsx src/components/EvidenceChart.test.tsx
    npm run build

The targeted live acceptance probe uses a concrete run key but does not expose
trace content:

    curl -fsS "http://127.0.0.1:7861/api/v1/runs/<encoded-run-key>/rollout-behavior" | jq '{state, expected, included, points: (.points | length)}'

Expected after the backfill and Trackio aggregate rollout: `state` is
`complete`, `points` is `100`, and the response contains only aggregate values
and coverage—not messages, tool arguments, calls, nodes, or token IDs.

## Validation and Acceptance

Acceptance requires all of the following observable behavior:

1. A newly ingested Verifiers trace writes one ordinary `traces` row whose
   nullable `fact_*` columns contain the supplied scalar projection, without
   altering its native payload, messages, metadata, identity, or timestamp.
   Qwen 3.5 thinking, structured tool calls, and ordinary provider usage are
   calculated once by the shared projector before write. The training bridge
   and evaluation synchronizer emit equivalent shared facts for the same native
   record, and Trackio persists them without reinterpretation.
2. A bounded aggregate read returns equivalent SQLite and Doris buckets for
   GRPO/OPD grouping by `rollout_step`, and evaluation grouping without that
   dimension. Each numeric result carries its own observed/missing denominator.
3. An old Qwen 3.5 training trace with only `payload.run.step` becomes
   attributable after backfill; an evaluation trace with the same nested shape
   does not gain a false rollout step.
4. The historical 100-step Ambient Agent run returns all 100 behavior points
   without Observatory requesting full trace payloads.
5. An unknown-model thinking boundary remains missing and is visibly reported
   as such, while completion, tool-call, and other supported facts remain
   queryable.
6. A pre-release or unsupported provider returns an explicit unsupported state,
   not a zero-valued chart or a hidden failed request.
7. The Trackio release has a committed/pushed fork ledger, immutable release
   assets, candidate readback, Posttrain integration evidence, promotion proof,
   stable pin, lockfile, and updated consumer tooling page.
8. A length-capped trace remains in the all-trace aggregate and in the
   `is_truncated=true` bucket. Its emitted model output/input tokens, model
   calls, and structured tool calls match the raw record; provider reasoning
   usage is retained, a verified Qwen partial-thought fixture is counted, and
   all ambiguous partial-thought cases remain missing rather than guessed.
9. A mixed reward population returns component score, weight, contribution,
   source kind, optional semantic source ID, and per-component coverage. An LLM
   judge component is distinguishable from deterministic and human components;
   an undeclared historical source is `unknown`, not guessed.
10. Trackio and Observatory expose no materialized judge executions. Their
    schemas and responses contain no judge prompts, responses, parsed payloads,
    model identity, execution IDs, calls, tokens, latency, or cost, even when
    those fields exist inside the untouched native trace payload.
11. Opening an existing SQLite database or upgrading Doris adds nullable scalar
    columns without rewriting native trace contents. A trace with no projection
    remains readable and reports unavailable facts. Re-export/import and run or
    project deletion preserve the extended trace and component lifecycle.
12. A TRL or veRL GRPO/DAPO trace first persists native task reward and reward
    components, then receives one trainer-owned `algorithm_reward` enrichment
    under the same external trace ID. The second write does not alter payload,
    messages, metadata, timestamp, source projection ID, or components. OPD and
    evaluation traces do not gain a fabricated algorithm reward.
13. Whenever both token values are observed, `thinking_tokens <=
    model_output_tokens`. A provider or fallback mismatch produces a partial
    projection with missing thinking tokens rather than an impossible chart.
    `answer_tokens` is shown only when compatible accounting permits exact
    subtraction.
14. Replaying an identical source projection produces the same deterministic
    projection ID and an unchanged write receipt. Changing or removing one
    reward component produces a different projection ID; after the parent
    pointer switches, no component from the old projection appears in current
    reads.
15. The retained Doris deployment migrates globally from schema version 1 to
    version 2 using the preview/apply command, records the version only after
    all DDL verifies, starts the candidate Trackio server, and passes ordinary
    trace reads plus SQLite/Doris trace-fact parity before writers resume.

## Idempotence and Recovery

Fact ingestion updates the retained trace row by its existing identity.
Backfill is explicitly scoped, paged, and resumable from its returned cursor;
re-running a page recalculates the same nullable scalar columns and deterministic
projection ID without deleting raw trace data. Identical canonical facts are an
unchanged write; changed facts stage a new projection and switch the parent only
after all component rows exist. Start with preview and a single
terminal run before applying a project-wide backfill.
If a calculator release is found incorrect, ship a new `calculator_version`,
re-run the same bounded backfill, and keep the original raw records intact for
verification. Do not purge any traces as part of this work.

Changing a declared signal source is also a versioned projection change. Update
the immutable environment/job declaration, bump the calculator version, and
re-run the scoped backfill. Never rewrite the native trace or silently relabel
old component rows in place.

SQLite applies scalar and component changes transactionally. Doris stages the
new projection's component rows before switching the parent trace's
`fact_projection_id`; a failed attempt before that switch is invisible to
readers and safe to retry. A bounded cleanup may remove unreferenced old
component generations only after verification. It must never delete or rewrite
the parent trace or native payload.

The two repositories remain independent until publication. If Trackio tests
pass but release publication is not complete, Posttrain must stay pinned to its
current stable dependency and can test the fork only as an explicit local
candidate. If candidate qualification fails, leave the stable pin unchanged and
use the backfill preview receipt to diagnose the fork.

## Artifacts and Notes

The motivating run is:

    ambient-agent / ambient-k1-olmo3-sft10k-rtxpro-full7749-100step-20260810-r1

It has 21,572 retained Verifiers traces and 100 optimizer steps. Before the
historical compatibility correction, its per-rollout behavior projection was
unavailable because all training attribution lived under `payload.run.step`.
The deployed Trackio aggregate confirms all 21,572 projected records in 100
ordered optimizer-step buckets. An EOF probe at cursor 21,572 returned no
records and no next cursor, so this retained population is complete and no
longer blocks removal of the Observatory payload fallback.

The current stable framework dependency remains
`carbonteq-trackio==0.31.5.post13`. The qualified candidate is
`carbonteq-trackio==0.31.5.post14.dev16` from fork commit
`d57c31d5d6d597f7739dc3f6cf89816a39c59a48`. Its retained wheel and sdist
hashes are recorded in `docs/tooling/trackio/README.md`, and those exact bytes
were promoted unchanged to the stable index. Production deployment and the
stale resumable-upload canary remain explicit qualification gates.

## Interfaces and Dependencies

Trackio exposes these exact SDK operations:

    Run.upsert_trace_facts(
        update: TraceFactUpdate,
    ) -> TraceFactWriteReceipt

    Run.aggregate_trace_facts(
        query: TraceFactsQuery,
    ) -> TraceAggregateResult

The remote server maps those operations to `/upsert_trace_facts` and
`/get_trace_facts`. Historical maintenance additionally uses the generic
`/bulk_upsert_trace_facts` endpoint, which accepts a bounded sequence and
returns one ordinary receipt per trace. This batch endpoint is transport and
storage optimization only; it does not add Verifiers semantics to Trackio.

`TraceFactsQuery` contains the approved trace type, equality filters, group
dimensions, and aggregate specifications. `TraceAggregateResult` contains
coverage, calculation provenance, and ordered aggregate buckets. Trackio's
storage abstraction exposes an equivalent method, implemented by both
`SQLiteStorage` and `DorisStorage`; no caller needs direct database access.

`TraceFactUpdate` is the wire value for both full source replacement and
scalar-only enrichment. `TraceAggregate` is one requested `(measure,
operation, component_name?)` calculation. `TraceAggregateBucket` contains the
grouped dimension values, matching trace count, and a value plus observed
coverage for every requested aggregate. `TraceFactWriteReceipt` contains the
physical trace ID, deterministic projection ID, and whether that projection
changed storage. The Posttrain maintenance receipt—not the Trackio SDK
receipt—owns page counts and the next cursor.

The Posttrain-facing observation contract remains provider-neutral:
`TraceObservation.facts` carries the initial source projection and
`RunContext.trace_fact_update(TraceFactUpdateObservation(...))` carries the
later trainer enrichment. The Lab Trackio observer translates both into the
Trackio SDK. Neither train nor eval imports Trackio, and Trackio never imports
Posttrain.

This query API is backed by nullable `fact_*` columns on Trackio's existing
`traces` table, not by a `trace_facts` table or SQL materialized view. The
logical `VerifiersTrace` write accepts the optional typed fact envelope;
Trackio decomposes it into those columns and the normalized
`trace_reward_components` rows while serializing the native Verifiers record
unchanged into `payload`.

The provider-neutral component result exposes `name`, nullable `score`,
nullable `weight`, nullable `contribution`, `source_kind`, optional `source_id`,
observed count, and missing count. The source fields carry semantic provenance
only. No public or storage interface in this plan defines `JudgeCall`,
`JudgeExecution`, or an equivalent execution record.

At the end of Milestone 3, the provider-neutral reader surface adds equivalent
logical values in `posttrain.tracking`. `TrackioDataSource` implements them;
other providers explicitly report whether they support them. Observatory calls
that reader surface only. It must not import Trackio or depend on a Trackio
database schema.

The only model-specific initial dependency is the shared Posttrain projector's
documented Qwen 3.5 fact rule. It is pure Python and uses stored native fields,
not a Transformers or vLLM dependency. A future model first uses native
provider reasoning usage. If that is unavailable, add a `ThinkingTokenRule`
matched by immutable model, tokenizer, renderer, template, and trace-schema
identity; add complete, truncated, ambiguous, and tool-call fixtures; bump the
calculator version; run a scoped backfill; and document the compatibility
entry. The Trackio fork ledger records only the generic fact storage/query
capability and its compatibility contract.

Verifiers v1 itself is not the grouped-metrics API. Its trace contract exposes
the ingredients—`calls[].usage`, `calls[].finish_reason`, assistant
`message.tool_calls`, generated `token_ids`/`mask`, task/run metadata, and
derived truncation semantics—but does not publish a durable cross-trace
aggregation or materialized fact view. Trackio remains the owning storage and
grouped-query layer, while Posttrain owns Verifiers semantic projection. Before
any future Verifiers pin advance, run the
same trace-fact fixtures against the candidate commit and record the accepted
immutable SHA in `docs/tooling/verifiers/README.md` and `uv.lock`.

Pinned Verifiers does not provide reliable per-reward judge lineage: judge
responses and weighted rewards are retained separately. Therefore Posttrain's
source classification comes from the immutable environment/job signal
declaration when present and is `unknown` otherwise. The projector never joins
native judge entries to reward components by list order or reward-name
heuristics. A future Verifiers pin may supply stronger semantic source metadata,
but adopting it must preserve this lightweight contract and must not expand the
Trackio or Observatory surface into judge execution telemetry.

Plan revision (2026-08-14): folded the agreed extensibility direction into the
execution contract. Trackio storage is a direct, global v2 schema transition;
Posttrain owns model/template compatibility rules and scored-signal semantics;
and reward components remain long-form rows with lightweight provenance rather
than a model-specific column set or a separate judge-execution store.

Revision note (2026-08-14): created after confirming that GRPO, OPD, and
evaluation must share one Verifiers trace-facts path rather than receiving
separate Observatory-specific rollup logic.

Revision note (2026-08-14): moved Verifiers and Qwen semantic calculation out
of Trackio into the shared `posttrain.environment` projector. Trackio now owns
only generic fact persistence and aggregation; live bridges and historical
backfill invoke the same projector.

Revision note (2026-08-14): implemented the shared observation/projector seam
and documented the immutable identity plus rule-registration path for future
models. Generic Trackio persistence, replay parity, and retained-data backfill
remain subsequent milestones.

Revision note (2026-08-14): added multi-reward component preservation and
separated native weighted contributions, raw score/weight when available,
scalar task reward, downstream algorithm reward, and non-trace advantage
evidence by their correct measurement grains.

Revision note (2026-08-14): replaced the proposed judge-execution direction
with a lightweight `SignalSource` on each reward component. The plan now
distinguishes LLM, deterministic, human, environment, group, teacher,
composite, and unknown sources while explicitly forbidding materialized judge
calls, prompts, responses, model details, tokens, latency, and cost.

Revision note (2026-08-14): replaced the proposed one-to-one `trace_facts`
table with nullable scalar columns on Trackio's existing `traces` row. Only
unbounded reward components remain normalized in
`trace_reward_components`. This avoids the common scalar join, preserves the
native Verifiers payload, and uses deterministic projection-ID staging for retry-safe
Doris component updates without introducing a materialized view.

Revision note (2026-08-14): closed the implementation-readiness gaps after an
architecture audit. The plan now uses two-phase trace facts for native and
trainer-owned reward values, deterministic projection IDs for replacement and
Doris staging, typed reward components, unambiguous total-output/thinking-token
semantics, exact SDK and HTTP contracts, and a coordinated global Doris schema
version 1 to version 2 migration. The earlier capability-version alternative
was deliberately not retained; this deployment owns and upgrades the complete
Trackio database schema.

Revision note (2026-08-14): reconciled the plan with the implemented and live
state after the local-job and historical-backfill work. Marked DAPO/evaluation
qualification complete, recorded 14,100/21,572 historical coverage, corrected
the actual Trackio SDK and receipt contracts, and replaced the five-page
accumulator direction with a checkpointed 1,000-page pipeline inside a
5,000-trace orchestration window. The remaining path reuses dev15 and existing
job evidence rather than rebuilding or rerunning them.

Revision note (2026-08-14): reconciled the documented rollout fallback with
the implementation. `rollout_behavior_view` has already been aggregate-only:
tests make `traces()` fail if that view attempts a native payload read. The
service wording and release status now state the actual contract: facts are
returned when supported and an explicit unavailable view is returned otherwise.
