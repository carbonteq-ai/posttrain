# Make training artifact publication bounded and efficient

This ExecPlan is a living document and follows `docs/templates/PLAN.md`. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are updated while the work proceeds.

## Purpose / Big Picture

A completed training loop must not become a failed run merely because several checkpoint artifacts are still publishing. After this change, LoRA and QLoRA runs continue to publish only their learned adapter as the model artifact, recovery checkpoints contain only the mutable state required for exact resume, overlapping artifact files are uploaded once through Trackio's content-addressed store, and finalization waits at a configurable bounded barrier instead of a hard-coded 30 seconds. A slow or failed publication remains visible as a real evidence failure, but Trackio does not fill its queue and then require every publication to drain merely to admit one more item.

## Progress

- [x] (2026-09-10) Reproduced the production failure from retained logs and traced it to the fifth artifact encountering a four-item Trackio queue followed by a 30-second whole-queue drain.
- [x] (2026-09-10) Verified that TRL already exports an adapter-only final model and rejects full base-model weights in LoRA recovery checkpoints.
- [x] (2026-09-10) Added bounded wait-for-one-slot behavior, per-run remote publication serialization, and a configurable 600-second finalization timeout to the Trackio fork candidate.
- [x] (2026-09-10) Wired one shared timeout budget through the Posttrain Trackio adapter and prioritized the small training summary and retention manifest before model and recovery publications.
- [x] (2026-09-10) Added regression coverage for queue saturation, timeout propagation, configured final drain, artifact ordering, and adapter-only retention; 86 Trackio artifact tests and 126 focused Posttrain tests pass, including the framework running directly against the modified Trackio checkout.
- [x] (2026-09-10) Ran a real deployed API canary: two logical artifacts sharing one digest committed, both manifests read back through the Doris-backed API, the blob-presence query passed, and a fresh presigned RustFS download returned 10,309 bytes with the expected SHA-256.
- [x] (2026-09-10) Published Trackio `0.31.5.post14.dev21` from immutable commit `ce3863a61f2773d7e62cc3f2ae5acdf6f3e5cfc5`, advanced the framework and runtime locks to the exact dev-index wheel, and deployed that wheel to the production-shaped Doris plus RustFS service.
- [x] (2026-09-10) Repeated the public artifact API canary after deployment through `trackio.carbonteq.com`: two logical artifacts sharing one 1 MiB blob committed in 8.279 seconds and the presigned RustFS readback completed in 3.618 seconds with the expected SHA-256.
- [ ] Run a small real training canary that produces summary, adapter, and recovery roles, then promote the validated Trackio build from the development index to stable.

## Surprises & Discoveries

- Observation: The failed LFM run did not upload the 2.6B base model. Its final `model-adapter` was about 42 MiB, while the roughly 92 MiB recovery artifact contained the adapter plus optimizer, scheduler, trainer, and random-number-generator state required for exact resume.
  Evidence: the training backend calls `validate_adapter_only_directory` for both final and recovery output, and the retained Trackio artifact inventory reports those logical sizes.
- Observation: Trackio's two background artifact workers cannot safely improve remote upload throughput today because remote client operations are protected by one client lock, while the unlocked gap between the presence check and upload enqueue allows overlapping artifacts to decide independently that the same digest is absent.
  Evidence: `trackio/run.py::_log_artifact_sync` releases `_client_lock` after `/check_artifact_blobs` and reacquires it in `_drain_pending_uploads`.
- Observation: The deployed API path itself is healthy and uses the intended split stores.
  Evidence: the live Trackio container selects `TRACKIO_DATABASE_ENGINE=doris` and `TRACKIO_ARTIFACT_STORAGE_BACKEND=s3` with bucket `trackio-artifacts`, prefix `production`, and the RustFS endpoint. API canary manifests for `shared-payload-a-1788993025` and `shared-payload-b-1788993025` both reference digest `cc93e9b85bbbc8b8af0958a6372e28d885db3f4c0142ab07169603dca6bc84e1`; a presigned download through `artifacts.carbonteq.com` returned 10,309 bytes with the same digest.
- Observation: Trackio's production service can use stable DNS without changing its external storage contract.
  Evidence: the dev21 container uses `ai-doris.lan` and `http://ai-storage.lan:9000`, resolves them to the expected private addresses, reaches ports 9030 and 9000, and still returns presigned reads through `https://artifacts.carbonteq.com`.

## Decision Log

- Decision: Preserve adapter-only model and exact-resume checkpoint semantics rather than deleting optimizer or RNG state.
  Rationale: an adapter is the loadable descendant model, while an exact recovery checkpoint serves a different contract. Removing mutable recovery state would save space by making resume incorrect.
  Date/Author: 2026-09-10 / Codex
- Decision: Apply queue backpressure by waiting for one publication slot, not by draining the entire queue.
  Rationale: admitting the next artifact requires only one completed future. Requiring all large uploads to finish creates unnecessary head-of-line blocking.
  Date/Author: 2026-09-10 / Codex
- Decision: Use a ten-minute Posttrain publication barrier and expose it as typed Trackio settings.
  Rationale: the observed artifacts exceeded 30 seconds on the selected storage path. Ten minutes is bounded, explicit, and long enough for expected adapter/checkpoint sizes without hiding an indefinitely stuck upload.
  Date/Author: 2026-09-10 / Codex
- Decision: Serialize the remote presence-check, upload, and manifest-commit transaction per run.
  Rationale: the current client already serializes its remote calls. Extending that boundary closes the duplicate-upload race for overlapping checkpoint and adapter manifests without reducing actual supported concurrency.
  Date/Author: 2026-09-10 / Codex

## Outcomes & Retrospective

The fork release, framework pin, runtime locks, production-shaped deployment, and direct API qualification are complete. The original 30-second whole-queue failure is removed, duplicate physical upload decisions are serialized, and LoRA artifacts remain adapter-only. A small real training canary remains the final behavioral gate before stable-index promotion; direct API success alone does not prove the full training finalization path.

## Context and Orientation

The framework checkout is `/home/hammad/projects/rl-local-async` on `codex/local-async-source-env`. The maintained Trackio fork is `/home/hammad/projects/trackio` on `codex/artifact-finalization`. `packages/train/src/posttrain/train/api.py` turns backend output directories into logical artifacts. `packages/train/src/posttrain/train/backends/trl/common.py` creates adapter-only final and checkpoint model views. `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py` maps those logical artifacts to Trackio. The Trackio fork's `trackio/run.py` owns its bounded background publication queue and physical content-addressed upload.

The repositories must be changed and released in dependency order: Trackio code and tests, Trackio fork ledger and commit, Trackio package publication, then the framework adapter, consumer documentation, exact dependency pin, and lockfile. Local development may temporarily inject the Trackio checkout for tests, but the framework cannot call an unpublished API in a release commit.

## Plan of Work

In the Trackio fork, extend `Run.log_artifact` with an optional bounded queue-wait duration. When the queue is full and this duration is supplied, wait until one active future completes, surface its failure if any, prune completed work, and enqueue the new publication. Keep immediate queue-full failure as the compatibility default. Add a typed artifact finalization timeout to `trackio.init` and `Run`, and use it in `Run.finish`. Protect the remote artifact presence-check, missing-blob upload, and manifest commit with a per-run publication lock so two overlapping logical artifacts cannot upload the same absent digest concurrently.

In Posttrain, add positive finite `artifact_publication_timeout_seconds` to `TrackioSettings`, pass it to Trackio initialization, and supply it as the queue wait and final drain budget. Queue the small `training-summary` and retention manifest before model and recovery outputs. Keep model lineage and recovery artifact types unchanged. The adapter continues to fail the run if required artifact publication cannot complete within the declared bound.

Update `/home/hammad/projects/trackio/CARBONTEQ_FORK.md` and `docs/tooling/trackio/README.md` with the maintained delta and exact release identity. Publish and pin only after focused validation passes.

## Concrete Steps

From `/home/hammad/projects/trackio`, run the focused unit tests for `tests/unit/test_run.py`, then the fork's formatter/linter and relevant artifact suite. Build the distribution and verify its metadata before publication.

From `/home/hammad/projects/rl-local-async`, run:

    uv run pytest packages/tracking-trackio/tests/test_adapter.py packages/train/tests/test_api.py packages/train/tests/test_retention.py packages/train/tests/test_trl_common.py
    uv run ruff check packages/tracking-trackio packages/train
    uv run pyright packages/tracking-trackio packages/train
    uv run lint-imports
    git diff --check

After the immutable pin is updated, run the same focused tests with the locked environment. A real qualification run is a later GPU/network gate and must use a new run identity; this plan never mutates or relabels the failed production run.

## Validation and Acceptance

The Trackio regression must fill a one-item queue with a blocked publication, submit a second artifact with a bounded queue wait, release the first, and observe the second being admitted without a whole-queue drain. A separate test must observe a clear timeout when no slot becomes available. Another test must prove that two artifacts sharing a blob do not execute overlapping remote presence/upload transactions.

The Posttrain adapter regression must show the configured 600-second value reaching both queue admission and final drain. The training API regression must show artifact order `training-summary`, optional `training-retention-manifest`, model artifact, then optional recovery checkpoint. Existing tests must continue proving that LoRA model directories and recovery checkpoints contain no full base-model weight files.

Acceptance requires focused tests in both repositories, clean package metadata, an immutable Trackio release and framework lock, and no import-boundary regression. No completion claim is made from unit tests alone for remote storage throughput; the next training canary must finish with all required roles and no artifact queue timeout.

## Idempotence and Recovery

All code and tests are additive or reorder independent artifact submissions. Re-running tests and package builds is safe. If Trackio publication fails, do not advance the framework pin. If the framework integration fails after Trackio is published, retain the fork release as an unused candidate and restore only the dependency selection in a new commit; do not rewrite published history. Existing failed run evidence is immutable.

## Artifacts and Notes

The triggering run was `lfm26-grpo20-post8-vllm-shared-20260910-r1`. It completed 20 optimizer updates, then raised `TimeoutError: Trackio artifact publication drain timed out` while `posttrain.train.api._finish` queued the final summary behind four active publications.

Focused validation currently records:

    /home/hammad/projects/trackio: 86 passed
    /home/hammad/projects/rl-local-async: 126 passed
    Posttrain with PYTHONPATH=/home/hammad/projects/trackio: 126 passed
    deployed API: 2 committed manifests, 1 shared blob digest, verified RustFS download

## Interfaces and Dependencies

The Trackio fork will expose `trackio.init(..., artifact_finish_timeout: float = 600.0)` and `Run.log_artifact(..., background: bool = False, queue_timeout: float | None = None)`. `queue_timeout=None` preserves the current immediate queue-full behavior. `posttrain_tracking_trackio.TrackioSettings` will expose `artifact_publication_timeout_seconds: float = 600.0`. The Posttrain adapter is the only framework layer that translates that policy into Trackio-specific arguments.

Revision note — 2026-09-10: created this plan after the 20-step GRPO run exposed queue-wide backpressure and finalization timeout defects. The plan preserves the already-correct adapter-only and exact-resume contracts while targeting physical duplicate transfer and lifecycle behavior.

Revision note — 2026-09-10: implemented and locally validated the Trackio queue-wait/finalization contract and the Posttrain adapter integration. A deployed Doris plus RustFS API canary separated healthy storage from the client lifecycle defect. Immutable release publication and pin advancement remain.

Revision note — 2026-09-10: published and deployed dev21, advanced all workspace and transform runtime locks, moved Trackio's server-side Doris and RustFS addresses to stable private DNS, and repeated the shared-blob API canary successfully. Stable promotion remains gated by one real training finalization canary.
