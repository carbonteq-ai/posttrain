# Preserve and stream terminal Verifiers rollout evidence across TRL and veRL

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This plan follows `docs/templates/PLAN.md`; the planning skill's usual `.agents/PLAN.md` entrypoint is not present in this repository.

## Purpose / Big Picture

After this change, a developer can inspect every Verifiers rollout that reached a terminal state during online training, including rollouts that ended because inference, a harness, or an environment failed. A terminal trace becomes durable and queryable before the framework decides whether the rollout is valid training input. Failed rollouts never receive an invented reward, advantage, response mask, or optimizer loss.

The behavior is the same for the public `train.grpo`, `train.sampo`, and on-policy `train.distill` operations whether the selected training backend is TRL or veRL. TRL submits terminal traces through its in-process observer as they are produced. veRL keeps Trackio credentials outside its isolated Ray runtime: workers append complete native Verifiers JSONL records to the mounted run workspace, while the Posttrain parent process tails that journal and submits records through the selected observer. The native JSONL remains the replay authority if Trackio is unavailable.

A human can see the change working in three ways. A focused test can force a model-call failure and show that the run raises the same training error while a failed Verifiers trace already exists. A live run can show trace count increasing before the complete rollout population or training process finishes. Observatory can display the failed trace and error class while reward and advantage charts exclude it.

This work repairs implementation conformance with `docs/post-training/06-observation-and-lineage.md`. That canonical document already requires error traces to be retained by default, native Verifiers JSONL to remain replay authority, completed rows to be streamed as idempotent Verifiers traces, and missing evidence never to be converted into zero-valued scores. No frozen product-baseline amendment is required.

## Progress

- [x] (2026-08-11 14:23Z) Read the repository agent guide, canonical observation and run-evidence documents, `docs/templates/PLAN.md`, and the planning workflow instructions.
- [x] (2026-08-11 14:23Z) Audited the shared Verifiers bridge, TRL rollout callback, veRL agent loop and launcher, evaluation trace synchronizer, Trackio adapter, Trackio SQLite and Doris trace keys, and Observatory error projection.
- [x] (2026-08-11 14:23Z) Diagnosed qualification run `ambient-math-olmo3-sft-coldstart-g8-1step-4090-c64-20260811-r3`: a CUDA out-of-memory response became a Verifiers `HarnessError`, `_project()` rejected its zero branches, and no native JSONL or Trackio trace was retained.
- [x] (2026-08-11 14:23Z) Decided that the implementation must preserve terminal evidence before trainability projection and keep infrastructure failures fail-fast by default.
- [x] (2026-08-11 22:06Z) Milestone 1: freeze the defect with deterministic terminal-error, observer-outage, missing-population, and reward-aggregate regression tests.
- [x] (2026-08-11 22:06Z) Milestone 2: implement evidence-first terminal trace recording and bounded fatal-error handling in the shared Verifiers bridge.
- [x] (2026-08-11 22:06Z) Milestone 3: adapt TRL to the terminal-trace callback and preserve full population replay when the backend fails.
- [x] (2026-08-11 22:06Z) Milestone 4: add a host-owned veRL JSONL tailer, final drain, compact sync receipt, and cancellation-aware process lifecycle.
- [x] (2026-08-11 22:06Z) Milestone 5: normalize failed, truncated, unscorable, and missing-population evidence without fabricating reward or advantage values.
- [x] (2026-08-11 22:06Z) Milestone 6a: add Observatory requested-versus-terminal population projection and chart scale support; unit presentation is covered by the product service and frontend suites.
- [ ] Milestone 6b: qualify real Trackio SQLite/Doris storage with successful and failed traces (requires configured external storage).
- [x] (2026-08-11 23:20Z) Milestone 7a: publish the immutable CarbonTeq veRL `0.9.0.dev1` candidate from clean source commit `a6fe39c22719ec981ed8544ad8feffd59995cc13` (tag `carbonteq-v0.9.0.dev1`), build/read back the wheel and source distribution from `carbonteq/dev`, update the exact runtime source lock, and publish the matching kind image after its real Docker/Bake import smoke. The registry index digest is `sha256:5684281f0f85ab741a156f1a92e11062a0e910f58aad03a488e8a2188db2421a`.
- [x] (2026-08-11 23:59Z) Prepared Posttrain `0.3.7` release inputs and developer-facing notes; the complete local implementation ladder passes with 383 backend tests, 46 frontend tests, Ruff, Pyright, import contracts, and diff checks.
- [x] (2026-08-12 00:18Z) Rebased the candidate on current `origin/main`, published the official Posttrain `0.3.7` veRL kind image at `sha256:555f8c59f006cc5c19df34678bb532a147cec2409468f6317d13f132df010986`, regenerated `published.toml` from registry readback, repaired the public-consumer TRL `post2` wheel mirror, and passed the 1,193-test repository suite plus the two clean-wheel consumer journeys.
- [ ] Milestone 7b: build the candidate veRL image and run real TRL and veRL canaries, reconcile evidence, and promote only after every live gate passes.

## Surprises & Discoveries

- Observation: The earlier per-rollout streaming repair is present, but it runs after `VerifiersEnvironmentRolloutBridge._project()` creates a trainable `EnvironmentRollout`.
  Evidence: `packages/train/src/posttrain/train/integrations/verifiers.py` currently calls `_project()` before `_preserve()` and `on_completed()`. A terminal Verifiers trace with zero trainable branches therefore disappears before either native or provider evidence is written.

- Observation: TRL's callback is typed around `EnvironmentRollout`, so its observation contract cannot represent a failed terminal trace without first pretending that failure is training data.
  Evidence: `packages/train/src/posttrain/train/online_rl.py` defines `AsyncRolloutCompletionObserver` over `EnvironmentRollout`; `packages/train/src/posttrain/train/backends/trl/grpo.py` derives `TraceObservation` only from that projected rollout.

- Observation: veRL successful traces are written by isolated Ray workers, but live provider ingestion does not occur while the subprocess runs.
  Evidence: `PosttrainVerifiersAgentLoop.run()` calls `bridge.run()` without an observer. The parent `_launch()` blocks in `process.wait()`. Host finalization replays the JSONL only after success or failure.

- Observation: The shared bridge currently conflates arbitrary execution errors with truncation.
  Evidence: `_trace_is_truncated()` returns true whenever the record contains any `errors`, so the CUDA OOM would be counted as both failed and truncated even though no length boundary caused it.

- Observation: Failed TRL finalization can suppress the only available population counters.
  Evidence: `_run_environment_backend()` applies `_rollout_replay_exclusions()` in both success and exception paths. TRL excludes `train/rl/rollouts_*` from replay even when the live callback failed before emitting those batch counters.

- Observation: The evaluation adapter already contains most of the required safe journal-tailing mechanics, but train cannot import eval.
  Evidence: `packages/eval/src/posttrain/eval/backends/verifiers/synchronization.py` waits for complete lines, validates records, deduplicates trace ids, batches submissions, retries at finalization, and reports sync statistics. Package boundaries prohibit importing this module from `posttrain.train`.

- Observation: Trackio already derives a deterministic physical trace id from `(run_id, trace_type, external_id)`.
  Evidence: the Trackio SQLite path constructs that key and uses `INSERT OR IGNORE`; the Doris traces table is `UNIQUE KEY(project_id, trace_id)`. Replaying one Verifiers external id is therefore an upsert-safe operation rather than a second logical trace.

- Observation: the local veRL checkout is not a reproducible release input.
  Evidence: `/home/hammad/projects/verl-upstream` is based on `a35908ca3c9632859c58d6a2855d858918ae21dc` and contains uncommitted maintained-fork changes. A local canary can inform development, but cannot close a release gate until the fork is committed, published, and pinned.

- Observation: a native error record can still contain a reward field, but that value is not learning evidence.
  Evidence: the repaired reducer now excludes error-bearing records before finite reward aggregation; the regression `test_terminal_error_reward_is_never_folded_into_learning_aggregates` proves it remains unscorable and does not fabricate reward spread.

- Observation: the dirty local veRL checkout was not the release source; a clean descendant already contained the maintained runtime, SAMPO, and dense-distillation history.
  Evidence: `codex/distill-dense-teacher-logprobs` at `c3f49b9117b882fa888e25e4a771461e13167848` was clean and passed the focused CPU suite. The release candidate `a6fe39c22719ec981ed8544ad8feffd59995cc13` changes only version and ledger state, while the dirty checkout includes unqualified TurboQuant/K8V4 research that is deliberately excluded.

- Observation: the framework selected TRL `1.9.2.post2`, but public CI still downloaded and staged `1.9.2.post1` for external consumers.
  Evidence: the clean-wheel SFT starter failed dependency resolution for `trl==1.9.2.post2`; `.github/workflows/quality.yml` named the older tag, filename, and hash. The workflow now derives its regression expectations from `[tool.posttrain.trl]`, and both consumer journeys pass with the hash-verified `post2` wheel.

## Decision Log

- Decision: Treat a Verifiers trace returned by an environment as terminal evidence even when `is_completed` is false or it contains an error.
  Rationale: “Terminal” means the environment produced a final trace record for that attempt. It does not mean successful, scorable, or trainable. This vocabulary preserves failed evidence without mislabeling it as a completed training sample.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Persist and submit terminal evidence before projecting a trainable `EnvironmentRollout`.
  Rationale: evidence existence must not depend on branches, token masks, finite reward, or algorithm eligibility. Projection can then reject the trace without erasing the explanation for that rejection.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Keep infrastructure and harness failures fail-fast in the first implementation.
  Rationale: replacing an OOM, HTTP 502, lost server, or harness crash through active sampling would hide a broken runtime and could repeatedly execute expensive or unsafe work. This plan changes evidence retention, not training-failure policy.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Never coerce a failed or unscorable trace into reward zero, a zero advantage, an all-zero response mask, or a synthetic `AgentLoopOutput`.
  Rationale: zero is a meaningful verifier result. Substituting it for missing execution evidence corrupts group baselines, reward variance, active sampling, and optimizer inputs.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Use one backend-neutral terminal-trace contract with backend-specific delivery mechanisms.
  Rationale: TRL can call the host observer directly, while veRL runs inside isolated Ray processes that intentionally do not receive tracking credentials. Both backends must preserve the same logical trace and training eligibility semantics.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Keep veRL provider writes in the Posttrain parent and tail the mounted native JSONL.
  Rationale: this preserves runtime isolation, avoids distributing Trackio credentials to Ray workers, and lets a parent reconcile evidence even if the isolated process exits nonzero.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Generalize complete-line JSONL synchronization in `posttrain.common`, while keeping Verifiers validation and projection in capability packages.
  Rationale: train and eval need the same bounded journal mechanics but may not import one another. A provider-neutral tailer can live in common without importing Trackio, Verifiers, TRL, or veRL; eval and train supply their own validators and emitters.
  Date/Author: 2026-08-11 / Codex.

- Decision: Do not modify the TRL or veRL forks unless a focused integration test proves that an upstream lifecycle prevents the Posttrain-owned behavior.
  Rationale: terminal trace construction, native persistence, the TRL observer callback, and veRL parent tailing are Posttrain adapter responsibilities. Avoiding unnecessary fork changes shortens release work and preserves backend independence.
  Date/Author: 2026-08-11 / Codex.

- Decision: Release veRL as an immutable pre-release candidate from the clean maintained branch, then pin its full commit in the kind image rather than selecting a dirty sibling checkout.
  Rationale: the evidence-streaming mechanics are Posttrain-owned, but the underlying veRL runtime, SAMPO, and distillation fixes must still be reproducibly consumable. Version `0.9.0.dev1` and tag `carbonteq-v0.9.0.dev1` identify those exact bytes without incorrectly asserting that the new image has passed live GPU qualification.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Preserve the current successful-run metric de-duplication, but disable success-path replay exclusions during failure finalization.
  Rationale: successful TRL batches already emit population metrics live. A batch that fails before that emission needs bridge-derived failure counters. Outcome-aware finalization fixes the gap without double-writing successful counters.
  Date/Author: 2026-08-11 / Codex.

- Decision: Report `rollouts_requested`, `rollouts_attempted` (terminal evidence), and `rollouts_missing` independently.
  Rationale: a terminal trace is auditable observed work, while a request with no terminal record has an unknown cause. Collapsing either into failure would corrupt operational diagnosis and reward population accounting.
  Date/Author: 2026-08-11 / Codex and user.

- Decision: Treat maintained-fork wheel mirrors as metadata-derived release inputs, not independently edited CI constants.
  Rationale: a clean consumer must receive the exact package selected by the framework. The new release regression binds TRL's GitHub tag, filename, and SHA-256 to the package selection metadata so a future pin change fails before consumer qualification.
  Date/Author: 2026-08-12 / Codex.

## Outcomes & Retrospective

Implementation now makes native terminal evidence durable before trainability checks. A deterministic zero-branch `HarnessError` is appended, delivered to the terminal observer, and only then raises `VerifiersRolloutFailure`; it is never returned as an `EnvironmentRollout`. Tracking submission exceptions retain the native record for final replay. The bridge stops dequeuing new work after its first fatal terminal-projection error, while preserving any concurrently completed terminal records.

TRL now receives `TraceObservation` rather than a trainable rollout in the streaming callback. On a failed backend, API finalization replays all bridge-derived population metrics instead of applying the successful-path exclusions. veRL keeps its worker credential-isolated: the parent polls the mounted append-only JSONL every 200 ms, submits complete records through `RunContext`, performs a final drain on every terminal path, and writes a compact `posttrain-verl-trace-sync.json` receipt. The receipt contains counts and an acknowledged byte offset only; no trace content, endpoint, token, environment dump, or secret is copied.

Population evidence now distinguishes requested work, terminal evidence, failed, truncated, unscorable, and missing terminal records. Error records are failed but not automatically truncated. A reward field on an error record is excluded from reward aggregates; absent valid rewards leave reward-spread metrics missing rather than emitting zero. Observatory exposes requested and missing population summaries and treats them as one compatible rollout-count scale.

Local verification on 2026-08-11: `uv run pytest packages/common/tests packages/eval/tests packages/train/tests packages/tracking-trackio/tests apps/observatory/tests -q` passed **382** tests with **4** dependency/environment skips; focused suites subsequently passed after the reward edge-case regression. `uv run ruff check packages/common packages/eval packages/train packages/tracking-trackio apps/observatory`, `uv run pyright`, `uv run lint-imports`, and `git diff --check` passed. Observatory frontend `npm run check` and `npm test` passed (**46** tests). The maintained veRL source release candidate `0.9.0.dev1` at `a6fe39c22719ec981ed8544ad8feffd59995cc13` passed its 114-test focused Python 3.13 CPU suite, wheel/source build, installed-wheel import, index upload, and index readback. The corresponding kind image passed its real Docker/Bake import smoke and registry label readback. The remaining gates are deliberately external: real Trackio SQLite/Doris qualification and TRL/veRL canaries. The dirty local veRL checkout remains diagnostic-only and cannot satisfy any release gate.

Post-rebase verification on 2026-08-12 passed the complete repository suite (**1,193 passed, 13 skipped**), the two isolated wheel-consumer journeys, all **46** frontend tests, Ruff lint and formatting, Pyright, all eight import contracts, the release consistency check, and diff validation. The official Posttrain `0.3.7` veRL runtime manifest was generated from registry digest `sha256:555f8c59f006cc5c19df34678bb532a147cec2409468f6317d13f132df010986`; its OCI labels bind Posttrain revision `d1070e184722dff502c74ef5c6cd02940bcf458b`, veRL revision `a6fe39c22719ec981ed8544ad8feffd59995cc13`, and veRL dependency lock `df7769f306ef8606fed37fd89c3fa2589ac72f39a45e4743816368a1acffe69f`. This prepares release inputs but does not replace the still-open live Trackio and GPU canary gates.

## Context and Orientation

The repository root is `/home/hammad/projects/rl`. At plan creation it is on branch `codex/qwen35-thinking-rollout-qualification` at commit `7e55da80579df6b94deb8dfcd1944ca2d8d122a4`. The worktree was clean before this plan file was added. The exact starting revision is historical context, not a permanent pin; future implementers must record their current `git rev-parse HEAD` and preserve unrelated worktree changes before editing.

The public training API lives in `packages/train/src/posttrain/train/api.py`. `train.grpo`, `train.sampo`, and on-policy `train.distill` accept a backend-neutral request and a Verifiers environment bridge. `packages/train/src/posttrain/train/online_rl.py` owns the backend-neutral policy-generation and rollout values. An `EnvironmentRollout` is specifically a trainable projection: it has prompt token ids, completion token ids, rollout log probabilities, an environment-selected response mask, a scalar reward, a truncation flag, and the source trace.

The shared Verifiers implementation is `packages/train/src/posttrain/train/integrations/verifiers.py`. A Verifiers environment produces a native trace containing messages or graph nodes, token information, rewards, stop conditions, and errors. `VerifiersEnvironmentRolloutBridge._run()` schedules environment episodes. `_project()` currently turns each trace into `EnvironmentRollout`; `_preserve()` appends JSONL under an operating-system file lock; `evidence()` replays records not already observed live; and `finalize()` publishes the native JSONL as a rollout artifact.

“Terminal trace” in this plan means a native Verifiers trace object returned by an environment episode. A terminal trace can be successful, semantically wrong, truncated, unscorable, or failed. “Trainable rollout” means a terminal trace that passes projection: it has a valid sampled branch, aligned tokens and log probabilities, an algorithm-acceptable reward, and the required masks. Evidence recording happens for every terminal trace. Training projection happens only for eligible traces.

The TRL adapter is `packages/train/src/posttrain/train/backends/trl/grpo.py`. Its `rollout_func` invokes `run_observed_rollouts()`, currently receives projected `EnvironmentRollout` objects, and calls `RunContext.trace()`. The selected Trackio adapter turns this call into a bounded local submission; network persistence is asynchronous inside the Trackio client. TRL must not wait on remote persistence, but it must know whether local submission accepted the record.

The veRL adapter is under `packages/train/src/posttrain/train/backends/verl/`. `agent_loop.py` runs one Verifiers episode for a veRL dataset row and returns an `AgentLoopOutput`. `worker.py` starts the native veRL trainer inside an isolated Python environment. `launcher.py` runs that worker as a subprocess from the Posttrain host. The isolated environment deliberately strips `TRACKIO_*` and `WANDB_*` variables. The bridge snapshot contains an absolute trace path inside the mounted run workspace; multiple Ray processes append records with `fcntl` locking.

The existing evaluation tailer is `packages/eval/src/posttrain/eval/backends/verifiers/synchronization.py`. Its complete-line, deduplication, batching, retry, and sync-stat behavior is useful, but train and eval are independent capability packages and cannot import one another. The generic byte/journal mechanics should move to a framework-neutral module such as `packages/common/src/posttrain/common/jsonl_sync.py`. Verifiers schema validation remains in eval and train wrappers because `posttrain.common` must not import Verifiers.

The Trackio adapter is `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`. It maps `TraceObservation(trace_type="verifiers")` to `trackio.VerifiersTrace`. The consumed distribution is `carbonteq-trackio==0.31.5.post12`; the sibling fork is `/home/hammad/projects/trackio` at `4c73e8b6e71c3da65cac41fc1371830e4435ecea` when this plan was created. Trackio already uses deterministic physical trace ids for Verifiers external ids in both SQLite and Doris. No storage-schema change is expected.

Observatory is under `apps/observatory`. `apps/observatory/src/posttrain_observatory/traces.py` already projects native `errors` into an error outcome, and telemetry definitions already expose failed rollout counts. This plan requires correctness tests and possibly small presentation adjustments; it does not create a second failed-trace schema.

The active TRL package pin is `trl==1.9.2.post2`. The sibling TRL fork was at `c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1` when inspected, while the generated framework lock records the exact published source revision separately. Do not infer package bytes from the sibling checkout. The local veRL checkout is dirty and unpublished; no release claim may depend only on that checkout.

The incident motivating the plan used a 24 GiB RTX 4090, 64 environment and inference sequences, and a 32-prompt by 8-generation OLMo3 candidate update. The first inference wave exhausted GPU memory. Verifiers returned a `HarnessError` with zero branches, `_project()` raised, and the run failed without a native trace file. The implementation must test that failure deterministically without deliberately reproducing GPU OOM.

## Scope and Non-Goals

This plan changes evidence lifecycle, failure accounting, and asynchronous delivery. It covers native Verifiers rollouts used by TRL and veRL for GRPO-family training, SAMPO, and on-policy distillation. It covers successful, failed, truncated, and unscorable terminal traces. It covers local durability, Trackio submission, replay de-duplication, finalization, and Observatory interpretation.

This plan does not change reward functions, DAPO or OLMo3 advantage calculation, active-sampling criteria, KL policy, clipping, sequence concurrency, GPU memory allocation, or retry policy. It does not turn infrastructure errors into replaceable candidate samples. It does not add Trackio credentials to isolated workers. It does not make veRL generally release-ready or resolve the existing dirty-fork publication work. It does not delete failed runs or native evidence.

## Plan of Work

### Milestone 1: freeze the failure semantics with tests

Start with tests that fail on the current implementation. In `packages/train/tests/test_verifiers_grpo_bridge.py`, construct a native Verifiers trace with a stable id, an error, and zero branches. Exercise the bridge through its observed path. Assert that the native JSONL contains exactly that trace and the observer receives exactly one `TraceObservation` before the bridge raises a typed rollout execution error. Assert that the error trace never appears in returned `EnvironmentRollout` values.

Add adjacent cases for a trace with one partial branch plus an execution error, a length-truncated trace without an execution error, a non-finite or absent reward, and a contract-invalid trace with zero branches but no error. The first and last cases are fatal after evidence retention. The truncation case keeps the selected existing mask behavior. The unscorable case is retained and must not produce a fabricated reward metric.

In `packages/train/tests/test_api.py`, make a TRL-like runner fail before its live population metrics are emitted. Assert that failure finalization replays `train/rl/rollouts_failed` and the terminal trace instead of applying successful-run exclusions. Add the complementary success test that confirms counters remain single-written.

In `packages/train/tests/test_verl_backend.py`, create an isolated worker fixture that appends one valid failed trace line and exits nonzero. The parent launcher must eventually submit that trace and retain a sync receipt and diagnostic artifacts before raising the worker error. This test is the local proof that veRL evidence does not depend on a successful result contract.

Do not make production code changes until these tests demonstrate the missing evidence, misleading truncation classification, and failure-replay exclusion.

### Milestone 2: separate terminal evidence from trainability in the shared bridge

Refactor `packages/train/src/posttrain/train/online_rl.py` so the optional observed bridge accepts a terminal-trace observer rather than a trainable-rollout observer. Use `TraceObservation` as the stable provider-neutral value rather than defining a second trace payload. Introduce internal callable aliases equivalent to:

    type TerminalTraceObserver = Callable[[TraceObservation], None]
    type AsyncTerminalTraceObserver = Callable[[TraceObservation], Awaitable[None]]

    class ObservedEnvironmentRolloutBridge(Protocol):
        async def run_observed(
            self,
            batch: RolloutBatch,
            generator: PolicyGenerator,
            *,
            on_terminal: AsyncTerminalTraceObserver,
        ) -> Sequence[EnvironmentRollout]: ...

Keep `run_observed_rollouts()` as the compatibility entrypoint. If a bridge implements `run_observed`, pass the terminal observer. If a legacy bridge only returns trainable rollouts, submit each returned `rollout.trace` after the batch, preserving its previous behavior. The helper must serialize local provider submissions with an `asyncio.Lock` and move synchronous observer work off the event loop with `asyncio.to_thread`.

Refactor `VerifiersEnvironmentRolloutBridge` so one helper stamps run identity, environment id, task index, example id, task facets, terminal outcome flags, and the existing stable trace id. Convert the native trace to a record, append that record to JSONL under the existing process-safe file lock, construct `TraceObservation`, and invoke `on_terminal` when present. Only then call a separate projection helper that validates branches, token alignment, masks, log probabilities, and rewards.

The bridge must record callback acceptance separately from native persistence. Mark an external id as live-observed only after local provider submission succeeds. Catch ordinary callback exceptions, retain the id for final replay, increment bounded sync-failure state, and continue only when native JSONL is safe. Do not catch cancellation or process-exit exceptions. A native JSONL append failure remains fatal because neither live submission nor later reconciliation is sufficient proof of replay authority by itself.

Replace the pre-created `asyncio.gather()` population with a bounded worker scheduler owned by the bridge. At most `max_concurrent` environment episodes may be in flight. When one terminal trace has an execution error or cannot satisfy the trainable contract, record it, set a fatal flag, stop dequeuing new occurrences, let already-terminal siblings finish their evidence callbacks, cancel remaining pending work, and raise one typed batch error that cites counts rather than concatenating every payload. A successful batch must preserve deterministic ordinal ordering exactly as before.

Do not send failed traces into `EnvironmentRollout`, TRL's result dict, veRL's `AgentLoopOutput`, DAPO active sampling, SAMPO advantages, or distillation teacher scoring. A semantically incorrect answer with a valid finite reward remains a normal trainable rollout; “failure” in this plan means execution or contract failure, not verifier reward zero.

### Milestone 3: make TRL stream terminal traces correctly

In `packages/train/src/posttrain/train/backends/trl/grpo.py`, change the observer closure to accept `TraceObservation` directly. Add TRL-owned attributes such as algorithm, policy variant, settings id, optimizer step, and rollout-batch ordinal without changing the native payload or external id. Call `RunContext.trace()` through the shared async helper.

Preserve the existing batch-level time, token-throughput, truncation, and selected-token metrics after a successful rollout batch. On failure, let bridge evidence finalization supply terminal population counters. Update `packages/train/src/posttrain/train/api.py` so `_publish_bridge_artifacts()` knows whether it is finalizing a successful or failed backend execution. Apply `_rollout_replay_exclusions()` only after a successful TRL batch emitted its live population metrics; do not exclude the bridge-derived failure counters in the exception path.

The Trackio adapter remains synchronous at the `Observer` protocol boundary but must only enqueue bounded local work. Verify with `packages/tracking-trackio/tests/test_adapter.py` that successful and failed `TraceObservation` values create `trackio.VerifiersTrace`, preserve the external id, and round-trip error outcome fields. Verify that calling the adapter twice with the same run, trace type, and external id produces one logical trace in SQLite and Doris-compatible row construction.

No TRL-fork change is expected. If the pinned TRL trainer prevents the callback from running or swallows the typed bridge error, first add a minimal failing compatibility test in `packages/train/tests/test_trl_vllm_compat.py`. Only then edit `/home/hammad/projects/trl`, update its `CARBONTEQ_FORK.md`, publish an immutable fork build, and update the exact framework pin and `uv.lock` in a separate commit.

### Milestone 4: tail veRL native traces from the Posttrain parent

Extract the provider-neutral complete-line journal mechanics from `packages/eval/src/posttrain/eval/backends/verifiers/synchronization.py` into a module such as `packages/common/src/posttrain/common/jsonl_sync.py`. The common module must not import Verifiers or any tracking backend. It accepts a path, a record validator, a stable-key function, a bounded emitter, and a positive batch size. It reads only newline-terminated records, never holds the unconsumed journal in memory, advances its acknowledged offset only after successful submission or an explicitly recorded invalid line, and retains bounded error summaries. A failed submission stops at the current batch and retries from native storage after a bounded delay; later records remain only on disk rather than accumulating in memory.

Keep `packages/eval/src/posttrain/eval/backends/verifiers/synchronization.py` as a thin compatibility wrapper that supplies `WireTrace` validation and the trace-id key. Its existing tests must continue to pass, with expectations updated only where the new bounded retry behavior is more precise.

Add a train-owned Verifiers validator and emitter wrapper. It converts validated native records to `TraceObservation` with the same attributes used by host finalization. Do not create a lossy generic inference trace. The external id remains the native Verifiers id.

In `packages/train/src/posttrain/train/backends/verl/launcher.py`, replace the blocking `process.wait(timeout=...)` section with a bounded poll loop. While the child is running, check `RunContext.cancellation`, drain newly completed native JSONL lines, and sleep briefly without busy-waiting. Use a default poll interval near 100 milliseconds and a trace submission batch small enough that a completed trace becomes queryable within one second under a healthy local Trackio client. Enforce the existing runtime deadline and process-group termination behavior.

When the child succeeds, fails, times out, or is cancelled, perform a final journal drain before interpreting the result contract or raising. Write a compact JSON sync receipt inside the run output directory containing native records observed, records submitted, duplicates, invalid records, retry failures, unsynchronized records, first/last offsets, and completion state. Do not store tokens, endpoints with credentials, exception stacks containing environment values, or trace payloads in the receipt. Publish the receipt as an optional diagnostic artifact on both success and failure.

The isolated veRL environment continues to receive no Trackio or W&B variables. `PosttrainVerifiersAgentLoop` calls the shared bridge without a provider observer; the bridge writes terminal native records before projecting `AgentLoopOutput`. If projection fails, the agent loop raises after the record is flushed. The host tailer discovers that record independently of the child exit code.

Do not modify the veRL fork merely to add provider delivery. If a real integration proves that veRL catches agent-loop errors and continues scheduling new waves after a fatal runtime error, document the exact native behavior and add a narrow maintained-fork lifecycle hook with upstream-quality tests. Such a fork change must be committed and pushed before its immutable revision is consumed by Posttrain.

### Milestone 5: make population and error semantics truthful

In `packages/train/src/posttrain/train/integrations/verifiers.py`, split helpers for error detection, truncation detection, reward extraction, and trainability. An arbitrary `errors` list must no longer imply truncation. Explicit length or configured rollout boundaries, such as `max_output_tokens` or a final model call with `finish_reason="length"`, remain truncation evidence. A harness timeout may be both failed and boundary-terminated when the native stop condition explicitly says so; CUDA OOM or HTTP failure without a length boundary is failed but not truncated.

Derive `train/rl/rollouts_attempted`, completed, failed, truncated, and unscorable counters from the retained native population. Preserve the current external metric names for compatibility. Add a requested-population value or attribute derived from the trainer input so Observatory can distinguish “256 requested, 64 terminal, 64 failed” from “64 requested, 64 failed.” Do not call absent terminal records failed; report their coverage as missing or cancelled according to actual scheduler evidence.

Reward aggregates must use only finite rewards from valid terminal traces. If no valid reward exists, omit reward mean, standard deviation, zero-variance fraction, and advantage fields instead of emitting zero. A single valid reward may legitimately have standard deviation zero; that is different from no reward population. Failed traces must never enter group reward spread, zero-variance groups, active-sampling retention, TIS, clipping, entropy, or advantage calculations.

Extend normalized observation definitions and tests in `packages/train/src/posttrain/train/grpo_observations.py` and `apps/observatory/src/posttrain_observatory/telemetry.py` only as needed to expose the requested-versus-terminal gap and trace-sync completeness. Keep trace-level error detail in trace storage rather than scalar metric fields. Observatory should show a failed trace's type and safe message, show population coverage separately, and leave reward/advantage cards missing when there was no valid learning signal.

The native record remains the replay authority. Product serialization continues through `apps/observatory/src/posttrain_observatory/redaction.py`; this plan must not add credentials, request authorization headers, or process environment values to trace attributes. Add a regression containing sentinel secret-like values in runtime environment variables and assert that sync receipts, run errors, and exposed trace attributes do not copy them.

### Milestone 6: integration and product qualification

Run a local Trackio integration using the real `posttrain-tracking-trackio` adapter and its SQLite storage. Start one observed bridge batch with one successful trace and one controlled model-call failure. While the operation is still blocked on another controlled rollout, query Trackio and observe the successful trace. Release the failure, observe its failed trace, and then observe the run fail. Query by external id and prove one logical record after final reconciliation.

Add a Doris-marked integration test when the configured service is available. It must submit the same external id twice and query one logical trace using the Doris unique key. The test skips with a clear reason when credentials or network are absent; it is a required release gate before deployment, not an optional claim.

Exercise Observatory against the integrated run. The run view must show terminal trace population, the failed trace error outcome, and incomplete coverage when work was cancelled. Reward and advantage charts must include only the valid trace. The trace-detail route must load the failed native record without requiring the training run to have succeeded.

Run an isolated veRL integration fixture through the actual parent poll loop, not only a mocked synchronizer. The child appends complete and partial lines over time, exits both zero and nonzero in separate cases, and verifies that incomplete lines are ignored until terminated, final drain is idempotent, and the sync receipt matches provider records.

### Milestone 7: real canaries, release, and consumer adoption

Build a candidate framework wheelhouse and immutable job image from the exact Posttrain commit. Do not call it a release until wheel hashes, image digest, source revision, dependency locks, and validation receipts agree. Use the current release tooling rather than installing from a dirty checkout inside a job.

Run a safe TRL success canary on the RTX 4090 using the retained SFT LoRA, lower memory pressure than the failed 64-sequence run, and enough total rollouts to produce more than one inference wave. A reasonable starting shape is 32 environment/vLLM sequences rather than 64 and a lower vLLM memory reservation selected by capacity validation. The evidence gate is not model quality: at least the first wave's traces must become queryable before the complete rollout population or actor update finishes, the run must retain native JSONL, and final reconciliation must produce no duplicates.

Test failure behavior with a deterministic controlled environment or generator error, never by intentionally causing another GPU OOM. The failed trace must be visible before the run reaches terminal failure, must contain the stable task and run identity, and must not appear in rewards or advantages.

Run a veRL canary only from an immutable maintained-fork revision and reproducible isolated environment. If the currently dirty checkout is used during development, label its evidence diagnostic and insufficient for release. The veRL canary must show host-side trace count increasing while the child process is still active and final sync completeness after process exit.

If no Trackio change was necessary, retain the existing exact `carbonteq-trackio==0.31.5.post12` bytes and record that fact. If a generic Trackio defect is discovered, make and validate that change in `/home/hammad/projects/trackio`, update both `CARBONTEQ_FORK.md` and `docs/tooling/trackio/README.md`, commit and push the fork, publish and verify the new package, then update the exact Posttrain pin and `uv.lock`. Follow the same fork-ledger sequence for TRL or veRL only when their code actually changes.

After all local and live evidence passes, run the framework release audit, publish the framework and immutable runtime images, update the exact Ambient Agent dependency and lockfile in a clean scoped commit, and run one consumer canary. Do not publish a release merely because unit tests pass; validation, publication, deployment, and live qualification are separate states.

## Concrete Steps

All Posttrain commands run from `/home/hammad/projects/rl`. Before editing, record and inspect state:

    git status --short
    git rev-parse HEAD
    git branch --show-current
    git -C ../trackio status --short
    git -C ../trl status --short
    git -C ../verl-upstream status --short

Use `apply_patch` for hand-authored changes. Preserve unrelated worktree changes. Run the smallest tests after each slice:

    uv run pytest packages/train/tests/test_verifiers_grpo_bridge.py -q
    uv run pytest packages/train/tests/test_api.py -q
    uv run pytest packages/train/tests/test_verl_backend.py -q
    uv run pytest packages/eval/tests/test_trace_sync.py -q
    uv run pytest packages/tracking-trackio/tests/test_adapter.py -q
    uv run pytest apps/observatory/tests/test_product_service.py -q

After the focused tests pass, run package-level validation:

    uv run pytest packages/common/tests packages/eval/tests packages/train/tests packages/tracking-trackio/tests apps/observatory/tests -q
    uv run ruff check packages/common packages/eval packages/train packages/tracking-trackio apps/observatory
    uv run pyright
    uv run lint-imports
    git diff --check

Before release, run the normal repository ladder from the agent guide:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

When Trackio and Doris integration configuration is available, run the existing real-storage marker selected by the Trackio tests and record the exact command and result in this plan. Do not print credentials. When building release candidates, record wheel names and SHA-256 digests, image references by immutable digest, and the Posttrain commit in `Artifacts and Notes`.

For each live canary, record the framework run id, provider id, tracking run id, image digest, source commit, selected backend, target, rollout population, environment concurrency, inference sequence cap, vLLM memory reservation, trace count observed before batch completion, native JSONL count, final provider outcome, and trace-sync receipt. Do not store tokens or unredacted environment dumps.

## Validation and Acceptance

The implementation is accepted only when all of the following behaviors are proven.

A native error trace is appended before trainability projection. A test that supplies a zero-branch `HarnessError` observes one complete JSONL record and one terminal `TraceObservation`, then observes a typed training failure. The error trace is never returned as training input.

A successful trace and a failed trace use the same native Verifiers schema and stable id rules. Final replay or process restart does not create a second logical Trackio trace. SQLite and Doris-compatible storage both enforce that result.

TRL submits terminal traces while rollout work continues. In a multi-wave canary, trace count increases after the first wave and before the complete batch returns. Remote Trackio latency does not block the event loop; native JSONL remains complete if local provider submission fails.

veRL workers receive no tracking credentials. The parent observes complete JSONL lines while the child remains alive, submits them through `RunContext`, performs a final drain on success, failure, timeout, and cancellation, and publishes a sync receipt. A child that exits nonzero after writing a failure trace produces both that trace and the original worker failure.

Failure does not alter algorithm inputs. Failed and unscorable traces are absent from rewards, group means, standard deviations, advantages, active sampling, TIS, clipping, entropy, response masks, and optimizer batches. No-reward populations produce missing reward evidence, not zero metrics.

Population evidence is internally consistent. Requested population, terminal trace count, completed, failed, truncated, unscorable, and missing or cancelled work reconcile arithmetically according to their documented meanings. An OOM is failed but not automatically truncated. An explicit length stop can be truncated without being an execution error.

Observatory lists and opens failed rollout traces for failed training runs. It displays the safe error type and message, identifies incomplete trace synchronization, and does not draw reward or advantage values from failure records.

The focused tests, package tests, Ruff, Pyright, import-boundary checks, full pytest suite, and `git diff --check` pass. Real Trackio/Doris integration and one TRL and veRL live canary are required before release. A dirty or unpublished veRL checkout cannot satisfy the release gate.

## Idempotence and Recovery

Every native trace append uses a stable Verifiers id and one newline-terminated JSON record. Re-running parent synchronization is safe because the synchronizer deduplicates within a pass and Trackio derives a deterministic storage key from run id, trace type, and external id. If the parent restarts and loses its in-memory offset, it may replay the journal from the beginning; the logical trace population must remain unchanged.

The tailer never deletes or truncates native JSONL. An invalid line is reported as partial evidence and retained for diagnosis. A partially written final line is ignored until completed. A provider submission failure leaves the record in native storage and stops acknowledgement at that batch, so later retries do not depend on in-memory payload retention.

If a training process fails, finalization must attempt one bounded drain, publish the native trace artifact and sync receipt when possible, and then re-raise the original training error. A finalization error is attached as a note and must not replace the root cause. Do not change a failed run to succeeded because evidence reconciliation later completes.

If the new live path causes unexpected load, disable live parent draining through an internal compatibility switch only long enough to preserve the native-journal fallback; do not disable native writes. Record any such rollback in the plan and open a new run for retry. Never overwrite or reuse an existing run's evidence identity.

For TRL, rolling back the callback refactor means retaining evidence-first native persistence and falling back to final replay; do not restore projection-before-preservation. For veRL, rolling back the poll-loop delivery means retaining worker JSONL and host finalization replay. These fallbacks reduce freshness but preserve correctness.

Do not clean failed canary workspaces until provider outcome, native JSONL, Trackio traces, and sync receipts are reconciled. Use the framework purge planner after evidence decisions; do not delete provider, registry, Trackio, or local records ad hoc.

## Artifacts and Notes

Initial incident evidence, with sensitive values omitted:

    Framework run: ambient-math-olmo3-sft-coldstart-g8-1step-4090-c64-20260811-r3
    Target: NVIDIA GeForce RTX 4090, 24 GiB
    Requested population: 32 prompts x 8 generations
    Environment and inference concurrency: 64
    Terminal provider outcome: failed
    Native Verifiers JSONL records: 0
    Trackio trace records: 0
    Root cause: inference HTTP 502 caused by CUDA OOM
    Evidence loss point: VerifiersEnvironmentRolloutBridge._project before _preserve

Repository baselines at plan creation:

    posttrain: 7e55da80579df6b94deb8dfcd1944ca2d8d122a4
    trackio checkout: 4c73e8b6e71c3da65cac41fc1371830e4435ecea
    trl checkout: c9af78c1c2ea04ad271e95b26b93dfadf8b9fca1
    verl upstream base: a35908ca3c9632859c58d6a2855d858918ae21dc (dirty maintained delta)
    consumed Trackio package: carbonteq-trackio 0.31.5.post12
    consumed TRL package: trl 1.9.2.post2

Add concise focused-test transcripts, integration counts, sync receipts, canary evidence, package hashes, and image digests here as implementation proceeds. Keep large logs in run artifacts rather than this plan.

## Interfaces and Dependencies

`posttrain.common` must expose a provider-neutral append-only JSONL synchronizer or equivalent internal module. It may depend only on the standard library and common JSON-safe types. It must accept caller-supplied validation, stable-key extraction, and batch emission. It must report bounded statistics and never import Verifiers, Trackio, TRL, veRL, eval, train, or Observatory.

`posttrain.train.online_rl` must define the internal terminal-trace observer contract. `EnvironmentRolloutBridge.run()` remains the compatibility method returning trainable rollouts. The optional observed extension streams `TraceObservation` values. Public training requests and result types do not change.

`VerifiersEnvironmentRolloutBridge` must provide one evidence-first path shared by in-process TRL and portable veRL snapshots. Its native record preparation must be deterministic. Its JSONL append must remain safe across threads and Ray processes. Its projection must accept a prepared record rather than mutating evidence after it has been submitted.

The TRL adapter must attach trainer-specific trace attributes without changing payload identity. It continues to use the pinned `rollout_func` seam. No fork-specific type enters public Posttrain APIs.

The veRL agent loop must continue returning the native token ids, log probabilities, response masks, reward, and trace id only for trainable rollouts. The parent launcher owns tracking delivery, deadline enforcement, cancellation, final drain, and sync receipt. Isolated workers continue to receive a portable bridge snapshot and mounted trace path, not observer credentials.

The Trackio adapter must continue using `trackio.VerifiersTrace` and stable external ids. SQLite and Doris logical behavior must be equivalent even if their physical insertion syntax differs. A Trackio fork change is conditional, not assumed.

Observatory continues reading provider-neutral `TraceRecord` values. Failed rollout presentation uses the native `errors`, stop condition, and trace attributes. It must not require a new Trackio-only endpoint or duplicate trace table.

Revision note (2026-08-11): Created after the RTX 4090 OLMo3 qualification run demonstrated that the existing streaming callback only sees trainable projections. The plan covers the shared evidence-first correction, TRL live callbacks, veRL parent-owned JSONL tailing, truthful population metrics, Trackio idempotency, Observatory behavior, controlled failure tests, live canaries, and release sequencing.
