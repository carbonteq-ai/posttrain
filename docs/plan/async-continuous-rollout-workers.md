# Continuous rollout workers and native asynchronous training integration

Revision 27 — 2026-09-09. The next framework release line is allocated as
`0.4.0`, covering structured-credit algorithms, native asynchronous agent
collection, and model/hardware configuration diagnostics. Allocation is not
activation: the canonical baseline still defers public async GRPO until
Milestone D closes. Release notes explicitly retain live failure-path,
checkpoint/resume, bounded 2B packaged training, maintained-fork publication,
and candidate-image gates. The older registry graph labeled `0.4.0.dev0`
predates this branch and is not release evidence.

Revision 26 — 2026-09-09. The source-composed local gate now overlays the
selected TRL, Verifiers, and AutomationBench environment checkouts and verifies
every import origin. Local TRL commit
`278af512459b6be6d9510029ad9bbfd7d7faff03` carries the corrected
learner-acknowledgement and standalone-worker behavior; it is committed but not
yet pushed or packaged. A real Posttrain Verifiers producer fed native TRL samples
through `AsyncGRPOTrainer` for one parameter-changing GPU optimizer update.
This exposed dataloader-prefetch corruption in the recovery seam: collation
acknowledged a later group before the learner consumed it. Group IDs now travel
in the batch and are acknowledged only in `training_step`; the one-step
regression proves only the trained group's siblings advance. The native
`AsyncRolloutWorker` independently completed 16 groups × 4 trajectories with
Qwen3.5-2B and 64/64 successful tool calls. Concurrency 32 took 18.7395 seconds
versus 46.7045 seconds at concurrency one, a 2.49x speedup. Its first run also
found and fixed trainer-dependent lifecycle logging and false failure reporting
for normal shutdown cancellation. These results qualify local rollout
concurrency and one small-model learner update, not 2B training,
checkpoint/resume, immutable publication, or public activation.

Revision 25 — 2026-09-09. Fast source qualification now uses a dedicated
Posttrain worktree and a worktree-local Python environment. The setup command
first installs the repository's frozen TRL/vLLM dependency closure, then
replaces only TRL with a `--no-deps` editable checkout. Imports are checked to
prove Posttrain resolves from that worktree and TRL resolves from the selected
fork. This is the inner development loop for CPU tests and local GPU harnesses;
it does not change `uv.lock`, package pins, or release evidence. The local
actual-job capsule remains the outer qualification gate. In the first setup,
34 Posttrain async integration tests and 20 selected native TRL lifecycle,
admission, and failure tests passed. Re-running `uv sync` restores the released
TRL wheel, so use `scripts/development/setup-local-trl-env TRL_CHECKOUT` to
refresh the environment and reapply the source overlay deterministically.

Revision 24 — 2026-09-09. The local executor is now the development
qualification path for candidate backend work. It packs a selected TRL or veRL
checkout as a bounded, content-addressed source snapshot into the existing
actual-job capsule, records its digest in the package manifest, and
daemon-loads that capsule without registry publication. TRL is reinstalled
without dependency resolution into the control environment; veRL replaces only
the disposable backend worktree while retaining the selected kind image's
locked dependencies. This is neither a bind mount nor an alternate launcher.
The option is restricted to the local executor, and the manifest makes the
unreleased source explicit. Publishable qualification still requires an
immutable fork artifact, consumer pin/lock update, and the normal
registry-backed actual-job gate. Focused package, manifest, Dockerfile, and
BuildKit contract tests pass.

Revision 23 — 2026-09-09. TRL's external-server changed-weight gate passed with
the tested implementation at `2308ab41aeedc082e154734f33cfe44809b1fdea` and
the evidence ledger pushed at
`b6c1206a4e7207d2243055728c4824853edab325`. The actor ran on an RTX 3070 Ti
and the vLLM server on an RTX PRO 6000 Blackwell Workstation Edition using the
immutable selected TRL runtime image. Base actor/server selected-token log-prob
delta was `0.0022419691`; after a real NCCL transfer of the changed final norm
weight it was exactly `0.0`, and the server log probability moved by
`10.6749088764`. The run exposed and fixed fail-open HTTP control calls,
Accelerate-dependent standalone client logging, and an empty-body control
response assumption. Milestone C now retains only live failure-path
qualification; optimizer updates, checkpoint/resume, publication, and public
activation remain Milestone D.

Revision 22 — 2026-09-09. Pushed TRL candidate `867885e5` adds the bounded native changed-weight actor/sampler parity probe required by Milestone C. It uses vLLM's supported NCCL weight-transfer API, changes a real actor tensor, resets the prefix cache, and teacher-forces the same token through actor and sampler with explicit parity/change thresholds. It refuses a one-GPU topology. Two exploratory local runs produced no admitted result: callable RPC cannot cross AsyncLLM's process frontend, and native NCCL correctly rejects placing trainer and inference ranks on the same GPU. Live inventory shows two idle retained workers with one GPU each and zero current RunPod two-GPU on-demand offers, even without a price ceiling. The remaining gate therefore needs a two-GPU instance or a deliberately qualified multi-node composition; this capacity constraint is not an algorithm failure.

Revision 21 — 2026-09-09. Failure containment now has deterministic coverage at every local seam. Pushed TRL candidate `d5b8cc4631c8f7adbe8296c08804205d6eb4c74c` proves a failed weight transfer leaves the old model version authoritative and never resumes inference or publishes/reopens the pending version. Posttrain proves native queue starvation reaches `check_health` and an unacknowledged upstream abort is run-fatal while local gateway resources are still released. These tests qualify classification and cleanup logic; the corresponding changed-weight and failure paths must still run in the real GPU composition before activation.

Revision 20 — 2026-09-09. The pushed veRL rollout-execution candidate `9694a6242e3590acaf58a779c1151b370f313b51` now has deterministic native prefix-resume qualification. An aborted generation resubmits the original prompt plus retained generated tokens, reduces the remaining budget, concatenates token IDs and behavior log probabilities exactly once, and reports the served policy span from version 7 through 8. The existing native AutomationBench bridge gate proves one emitted tool action executes once before the following model turn. This closes deterministic veRL partial-continuation evidence; changed-weight GPU execution, live failure paths, and public activation remain open.

Revision 19 — 2026-09-09. The deterministic TRL portion of Milestone C now has a concrete run-scoped policy gateway. It forwards Verifiers' native token protocol to the trainer-owned vLLM server, atomically gates new model turns, drains admitted requests before publication, reopens them only after version activation, and accumulates exact policy spans by native trace session. Worker projection writes that span into the native Verifiers episode and returned rollout; missing or mismatched evidence fails closed. Shared upstream failure is collection-fatal. Real changed-weight parity and veRL prefix-resume qualification remain open.

Revision 18 — 2026-09-09. Milestone B is implemented behind private, unselected candidates. TRL fork `3972dc39adf1c836f1309b263c9450c32f1863f4` adds optional learner-consumption acknowledgement and custom scheduling-state checkpoint hooks. Posttrain advances group recovery only from those acknowledgements, replays generated-but-unconsumed groups, skips terminally rejected groups, and rejects a checkpoint containing a partially consumed group. It serializes no queue, environment, request, or process state. Public activation and real checkpoint/resume qualification remain Milestone D work.

Revision 17 — 2026-09-09. Milestone B now has a concrete, internal native-async GRPO group producer. It composes run-scoped Verifiers workers, the request-admission/weight-publication authority, the existing bridge, algorithm reward shaping, group-relative advantages, and exact-token native sample projection. Recoverable episode defects reject one complete group; infrastructure failures cancel siblings and remain fatal. Startup and shutdown account for partially acquired resources. Public activation is still disabled. Milestone B remains open only for learner-consumption acknowledgement and checkpoint recovery; the current deterministic prompt cursor is intentionally ephemeral.

Revision 16 — 2026-09-08. Current direction: preserve the synchronous rollout work below and add a separate, initially internal Verifiers adapter for TRL's existing experimental `AsyncGRPOTrainer`. Do not build a second asynchronous learner. The new milestones A–D below govern asynchronous training; milestones 0–5 govern the synchronous execution path only. Neither path is production-qualified. The first new implementation slice is exact-token transport into native TRL samples, not trainer activation.

Revision 15 — 2026-09-08. Status: the TRL session passes a real vLLM 0.25.1 lifecycle gate, the framework exposes a validated loopback native token endpoint, and its fixed Verifiers worker-pool adapter enforces collection identity, deadlines, acknowledged cancellation, and immediate broker-death failure. A backend-private collection runner sequences policy synchronization, worker/endpoint admission, terminal outcomes, cancellation, inference drain, and suspension before optimizer handoff. Worker-returned native episodes now use the existing bridge's evidence projection, and a real spawned AutomationBench worker passed the complete native wire path. Native worker configuration preserves the selected chat template with LFM token parity. Shared contracts, veRL sampling/renderer propagation, native worker admission, V1 complete-group recovery, and opt-in framework translation are implemented. Changed-weight parity, existing trainer activation, immutable runtime publication, and end-to-end GPU qualification remain open. Repository: `/home/hammad/projects/rl`.

## Purpose / Big Picture

Keep inference supplied with ready work while independent agent environments execute tools, render their next requests, and score completed episodes on multiple CPU processes. A short model request must return to its environment without waiting for an unrelated long request. Support two deliberately distinct integrations: asynchronous requests inside synchronous fixed-policy rounds, and future asynchronous training where the learner consumes previously generated samples while new episodes execute. The latter reuses native trainer scheduling and requires explicit behavior-policy provenance and bounded sample age.

This plan extends [the GDPO/CAPO implementation and qualification plan](gdpo-capo-dual-backend-support.md). It changes execution, not algorithms, reward rubrics, task selection, or credit assignment. Both TRL and veRL are required implementation workstreams. They share execution semantics and validation, not a universal process manager. It must not introduce GRPO-specific environment execution that other algorithms cannot reuse.

The user explicitly excludes new instrumentation. Use existing native traces, logs, resource inspection, optimizer records, and checkpoint artifacts for qualification. Do not add a telemetry subsystem, new timing spans, or a performance dashboard. This document authorizes no changes to currently running jobs.

Actor forward/backward optimization is explicitly out of scope: no changes to actor kernels, training micro-batches, gradient accumulation, optimizer settings, or distributed training layout for performance. Existing optimizer steps remain integration correctness gates only. Rollout completion time and valid-episode throughput are the performance outcomes. Releasing rollout/judge resources at the existing training boundary is in scope because it is rollout lifecycle correctness, not actor optimization.

## Progress

- [x] (2026-09-08) Review the linked async-RL survey, released TRL documentation, and exact fork source. Select native `AsyncGRPOTrainer` and `RolloutWorkerProtocol`, not a new Posttrain learner.
- [x] (2026-09-08) Milestone A: implement exact-token, complete-group sample projection and test it against the installed native TRL sample and queue consumer. Public activation remains disabled.
- [x] (2026-09-09) Milestone B: implement a bounded Verifiers producer satisfying the native worker protocol, including concrete GRPO group production, stable prompt scheduling, independent group completion, exact group-relative advantages, bounded typed rejection, run-scoped admission, sibling cancellation, lifecycle cleanup, learner-consumption acknowledgement, and checkpoint-aware replay of only unconsumed groups. Runtime selection and live resume proof remain Milestone D.
- [ ] Milestone C: qualify served policy-version evidence across tool waits and live weight transfers (completed: backend-neutral episode spans, veRL native span propagation and deterministic prefix-resume proof, TRL's acknowledged request drain/two-phase publication gate, a run-scoped native-token admission gateway, native-episode span persistence, missing/mismatched evidence rejection, shared-upstream failure classification, and real external-server changed-weight actor/sampler parity; remaining: live failure-path qualification before activation).
- [ ] Milestone D: amend the deferred-async baseline, activate validated backend-specific settings, and qualify checkpoint/resume and publication. One real parameter-changing asynchronous update now passes through the Posttrain Verifiers producer and native TRL learner; public activation and a 2B learner run remain open. Separately qualify episode-level GDPO; do not imply CAPO/SAMPO/OLMo parity.
- [x] (2026-09-08) Inspect the current bridge, TRL request batching, native Verifiers process pool and training client, and canonical ownership contracts.
- [x] (2026-09-08) Select native environment worker processes, native token-preserving clients, a single asynchronous inference owner, and fixed-policy collection barriers.
- [x] (2026-09-08) Revise the design for veRL-native Ray workers, explicit component interfaces, and backend-specific lifecycle ownership; reject nested environment pools on veRL.
- [x] (2026-09-08) Verify canonical Verifiers checkouts after worktree cleanup and record branch/commit preflight safeguards below.
- [x] (2026-09-08) Resolve the runtime veRL source pin and specify two-stage admission, sampling precedence, global judge admission, and failure classification. Exclude actor compute optimization.
- [x] (2026-09-08) Implement and deterministically test the additive TRL `AsyncVllmSession` lifecycle foundation at fork commit `2faf864cc5728aad6c07f3871067de4f40e3acb0`: independent completion, policy-version fencing, explicit abort/drain, sleep/wake handoff, and idempotent shutdown.
- [x] (2026-09-08) Add shared `rollout_execution` identity, outcome, and capacity contracts in framework commit pending this revision: 4×8 worker capacity is constrained by the global 32-episode limit; completed outcomes are identity-fenced before admission; failed outcomes cannot manufacture a rollout.
- [x] (2026-09-08) Preserve veRL phase/per-row sampling in `VerlPolicyGenerator`: native sampling is no longer discarded, greedy validation controls are accepted, and environment output ceilings remain enforced.
- [x] (2026-09-08) Add and publish the generic veRL worker-local episode gate at fork commit `f73ca959`: the manager validates a provable global capacity contract and each Ray worker enforces its own opt-in `asyncio.Semaphore`; defaults remain unchanged.
- [x] (2026-09-08) Close the active veRL V1 path at pushed fork commit `5dbf667c`: TransferQueue sessions use the worker gate, any partially failed prompt group can be replaced as one terminal unit, and Ray CPU reservations are explicit.
- [x] (2026-09-08) Carry renderer selection and validated `backend_options.rollout_execution` through the framework adapter. The mode is revision-gated to the supporting fork and does not change the stable runtime profile.
- [x] (2026-09-08) Qualify native `AsyncLLM` request completion, explicit abort acknowledgement, sampled-token logprobs, staged weights/KV wake, sleep, and clean shutdown on vLLM 0.25.1 at pushed TRL commit `684696a22ef3d82dbf39f21a60fafa9e5f17514b`; changed-weight actor/sampler parity remains a separate open gate.
- [x] (2026-09-08) Preserve the exact selected chat template in native Verifiers `TrainClientConfig` at pushed fork commit `c6c0097ad21da845c62e4b19aba80ef6633e4d9f`, include it in renderer-pool identity, and prove worker/direct LFM historical tool-call rendering parity. HTTP endpoint and full episode wire qualification remain open.
- [x] (2026-09-08) Add a loopback-only TRL token endpoint using the native vLLM token-in/token-out schema. Tests cover model metadata, exact prompt/completion tokens and selected-token logprobs, collection fencing, explicit abort, context rejection, unsupported scheduling controls, and fatal response-identity mismatches. A real native `TrainClient` single-turn LFM request passes end to end through the endpoint. Advance framework manifests, catalogs, and candidate control-runtime inputs to the pushed Verifiers renderer revision while preserving the last published veRL backend profile.
- [x] (2026-09-08) Publish Verifiers commit `c6c0097ad21da845c62e4b19aba80ef6633e4d9f`, adding optional caller-owned native run IDs and an acknowledged `EnvClient.cancel`. Implement the framework `VerifiersWorkerPool` lifecycle with fixed native workers, validated capacity, health startup, collection fencing, stable wire IDs, deadline cancellation, fatal unacknowledged-cancel handling, and bounded shutdown. Deterministic adapter tests pass; a real spawned-worker episode is still open.
- [x] (2026-09-09) Activate `backends/trl/collection_runner.py::TrlCollectionRunner` behind explicit `request_mode: async` plus a bounded `rollout_execution` topology. A persistent serving-loop thread owns TRL's lazy `AsyncLLM`, loopback policy endpoint, and native Verifiers worker pool across synchronous fixed-policy rounds. Controlled tests prove loop reuse, trainer-thread observation replay, ordered terminal outcomes, safe drain/suspend before optimizer handoff, lifecycle shutdown, and fail-fast rejection of ambiguous batch/async configurations. GPU qualification remains open.
- [x] (2026-09-08) Route worker-returned `WireEpisode` values through the existing bridge's native preservation, enrichment, trace observation, reward projection, and token-mask logic. Logical step and rollout ordinal are explicit occurrence identity fields; no opaque id parsing reconstructs lineage.
- [x] (2026-09-08) Translate native terminal outcomes into the existing partial-rollout admission contract. Healthy completions remain available for complete-group retention/replacement, while infrastructure and identity failures remain collection-fatal. Opt-in TRL topology validation requires colocated sleeping vLLM and enforces the environment's global limit; direct mode remains the default.
- [x] (2026-09-08) Pass a real spawned-process qualification using one AutomationBench task, the selected LFM renderer/tokenizer, native `TrainClient`, loopback policy endpoint, fixed `EnvServerPool`, native episode projection, and retained episode/trace files. Add deterministic immediate broker-death handling so a lost worker pool cannot degrade into a long episode timeout.
- [x] (2026-09-08) Prove the selected TRL asynchronous engine lifecycle against a real vLLM engine before implementing worker transport.
- [x] (2026-09-08) Implement and test native-client HTTP wire compatibility and the exact-token response contract using both the native schemas and a real LFM `TrainClient` request. Multi-turn episode execution remains open.
- [x] (2026-09-08) Implement the additive TRL asynchronous generation lifecycle against the selected runtime. Trainer ownership and changed-weight parity remain open.
- [x] (2026-09-09) Integrate bounded environment workers and coordinator admission/cancellation.
- [x] (2026-09-09) Implement veRL worker budgets, model-independent rendering, typed episode failures, and pre-advantage group admission.
- [x] (2026-09-10) Repair candidate runtime construction and cache the veRL backend's checksum-pinned binary wheels across rebuilds. The real dependency layer fell from about 544 seconds to 43 seconds, the steady-state identical build completed in 0.60 seconds, and the published candidate imported the exact veRL, Verifiers, Torch, and vLLM versions. Updating `cuda-pathfinder` in the monolithic universal base is deferred until the base Torch layer is split, because republishing a multi-gigabyte layer to remove about 6.3 MiB of duplication is not a sound release trade.
- [ ] Qualify failure handling, numerical equivalence, and real GPU optimizer updates (changed-weight parity, deterministic failure semantics, and one controlled small-model optimizer update pass; live failure injection, real environment serving, and a 2B learner update remain).
- [ ] Publish fork revisions, update consumer locks and documentation, and qualify the immutable runtime image before promoting the mode. TRL `1.12.0.post6` and veRL `0.9.0.post2` are immutable GitHub prereleases published byte-for-byte to `carbonteq/dev`; Verifiers `36eac9d5` is pushed and selected by Git revision; Trackio dev20 metadata drift is repaired. Root and runtime locks are regenerated, and all seven v0.4 runtime images are published and recorded in `published.toml`. GPU qualification remains, so the milestone stays open.

## Context and source authority

Read `docs/post-training/README.md`, `04-framework.md`, `05-apis.md`, and `06-observation-and-lineage.md` before implementation. Their package boundaries and native-trace authority override historical code. Read `docs/tooling/forks.md` before any fork publication. The baseline still defers async RL: milestones A–C are private compatibility prototypes with no public selection or supported-mode claim. Milestone D must explicitly amend that baseline before public activation. Engine settings remain inference settings; training scheduling and allowed sample age belong to the training backend, never to Verifiers task configuration. The plan follows `docs/templates/PLAN.md`; the plan skill's optional `.agents/PLAN.md` does not exist in this repository.

The inspected source baseline is:

| Repository | Source baseline and role |
| --- | --- |
| `/home/hammad/projects/rl-local-async` | Active Posttrain v0.4 worktree on `codex/local-async-source-env`; it selects the development fork closure below. This is not a stable release until OCI and GPU gates pass. |
| `/home/hammad/projects/trl-async-training` | Canonical async-training worktree on `codex/posttrain-v04-dev`. Candidate source `6dfc69db939144d270cbcbbed17294262b5ac6f4` is tagged `carbonteq-v1.12.0.post8`; retained-asset workflow `34393365885` published and clean-installed the exact wheel and sdist from `carbonteq/dev`. The sibling `/home/hammad/projects/trl` checkout remains historical and is not the edit target. |
| `/home/hammad/projects/verifiers` | Canonical checkout on `codex/carbonteq-verifiers-latest`. Selected pushed commit `1f6793f7d46e8a650a54b2a585193b4010578fa6` is based on upstream main `27bbd216df0af719a43705866b2cf6139bcc95de`; the complete v1 suite passes with credential-dependent Prime tests skipped. Verifiers is selected by immutable Git revision rather than a private-index wheel. |
| `/home/hammad/projects/verifiers-environments` | Canonical checkout on `codex/verifiers-latest-support`. Commit `c7e88d7b302e6177041ac0849863d0466be77d8b` aligns all six independent environment packages on Verifiers `1f6793f7`; all 59 available tests pass with two data-dependent skips. Task semantics remain owned by this repository. |
| `/home/hammad/projects/verl-upstream` | Active branch `codex/verl-rollout-execution`. Release source `98742d3e9507318ba0b5d4944034deb7db1ec84b` is tagged `carbonteq-v0.9.0.post2`; ledger follow-up `4050fbeb3528d80492880b7c0eb16f0b3e81322d` is pushed. Posttrain workflow `34335257738` published and clean-installed the exact retained bytes from `carbonteq/dev`. The development profile selects post2; immutable OCI and GPU qualification remain. |

Resolve branches, dirty state, manifests, and lockfiles again when implementation starts. These are inspection anchors, not permission to overwrite later changes.

### Mandatory checkout preflight

The two canonical Verifiers directories above are the only active worktrees for their repositories following cleanup. Do not recreate or work in `verifiers-carbonteq-latest`, `verifiers-environments-latest`, `verifiers-environments-turns`, or `verifiers-gdpo-capo`. Those directories were moved to Trash; the last had a broken Git link to a missing temporary repository. Historical references elsewhere describe past evidence, not current edit targets.

Before changing any repository, run the following with its exact absolute path from the table:

    git -C /absolute/repository/path rev-parse --show-toplevel
    git -C /absolute/repository/path branch --show-current
    git -C /absolute/repository/path rev-parse HEAD
    git -C /absolute/repository/path status --short
    git -C /absolute/repository/path worktree list --porcelain

Compare all results with the table. A different branch, detached HEAD where a branch is expected, missing path, or unexpected commit requires reconciliation before edits; do not silently use another similarly named checkout or reset it to this historical SHA. Legitimate new commits should update this table and the relevant fork ledger. Establish and record the selected veRL implementation branch from its runtime pin before its first edit; if that branch already exists, inspect it rather than recreating or resetting it. The TRL temporary worktree is the currently verified edit location; if absent, recover its recorded branch in a deliberate worktree and update this table, never substitute sibling `trl` without checking its state.

Cleanup preserved older local changes in named stashes: `pre-worktree-cleanup-2026-09-08-fork-docs` in Verifiers and `pre-worktree-cleanup-2026-09-08-schema-index` in verifiers-environments. Do not pop or drop these as implementation setup; inspect them only if the task requires their contents. Branches and commits were retained. Restore trashed directories only for deliberate recovery, not as active parallel development targets.

## Surprises & Discoveries

TRL's released v1.12.0 documentation already describes a producer/queue/learner system, separate inference GPUs, weight transfer and stale-sample rejection. Exact fork `684696a2` exposes `RolloutWorkerProtocol` (`start`, `stop`, `update_model_version`, `check_health`, `rollout_buffer`, `metrics_queue`) and native `RolloutSample`. Its learner accepts a scalar precomputed advantage, not a reward vector or token-wise advantages. Its queue rejects samples individually by age; complete-group scoring does not mean atomic optimizer consumption of all siblings.

The real local optimizer gate found that “collated” is not equivalent to
“learner-consumed.” The Trainer dataloader prefetches the next planned
microbatch, so a callback in `DataCollatorForRollout` advanced recovery state
for work that never reached a one-step learner. The generic seam now carries
group IDs through dispatch and acknowledges them from `training_step` on the
main process. The same local composition required the `kernels` package because
the current async trainer explicitly selects Transformers' Hub-hosted
FlashAttention kernel. Online resolution succeeds, but its unpinned Hub kernel
revision remains an immutable/offline-runtime packaging gap.

The native tool-agent comparison also shows why “async” must mean independent
request admission rather than merely an async function. Qwen3.5-2B produced
the same 64 successful tool calls with zero tool failures at concurrency one
and 32, while concurrency 32 reduced wall time from 46.7045 to 18.7395 seconds.
The 2.49x rather than 32x gain is expected here: each trajectory's second model
turn depends on its tool result, and all generations share one GPU.

Milestone A's native-consumer test exposed that `RolloutQueueDataset` logs through Accelerate even when tested outside a trainer; the test substitutes only that logger and exercises the real stale-filtering iterator. Six tests pass. The implemented adapter constructs the full native sequence directly from retained token IDs, masks prompt and environment tokens, zero-pads behavior logprobs only at untrained positions, and rejects an entire group before returning anything when size, occurrence, truncation, advantage, task, logprob, or served-version evidence is invalid.

The native trainer protocol and the Verifiers group source are separate seams. `backends/trl/async_worker.py::TrlAsyncRolloutWorker` now owns only the bounded thread/event-loop lifecycle and native queues; an `AsyncGroupProducer` owns task selection, Verifiers execution, reward admission, and advantage computation. This prevents environment semantics from leaking into a TRL protocol adapter. Focused tests prove a short group is published while an unrelated group waits, producer failure reaches `check_health` without a fake sample, full-queue backpressure cannot block shutdown, startup failure closes the producer, shutdown closes it, unbounded queues are rejected, and versions cannot regress. `VerifiersWorkerPool` preserves its exact synchronous `CollectionKey` fence while exposing a mutually exclusive run-scoped admission mode for independently versioned async groups.

The run-scoped worker admission seam made the concrete producer possible without weakening synchronous collection fencing. `integrations/verifiers_async_groups.py::VerifiersAsyncGroupProducer` now creates one stable collection and sibling occurrence set per prompt, dispatches siblings concurrently, retains only complete valid groups, applies the existing GRPO reward shaping, computes population-standard-deviation group advantages, and projects exact retained tokens into native TRL samples. Expected episode defects raise `RolloutGroupRejected`; broker, identity, cancellation, and other infrastructure failures remain fatal. DAPO, GDPO, OLMo-recipe, CAPO, SAMPO, dynamic sampling, and active sampling fail closed because their credit or selection semantics are not interchangeable with this GRPO implementation. Its prompt cursor is not durable: generated or queued work is not treated as consumed until the native learner explicitly acknowledges it, which is the remaining Milestone B design seam.

In that fork, `_sync_weights` resumes inference before notifying the worker of the incremented model version, and its worker protocol has no pre-update scheduling gate. Prime-RL closes this race differently: its dispatcher records one group start version, stops new scheduling in `on_version_pending`, lets the inference pause drain active requests, applies weights, then resumes scheduling in `on_new_version`. It stamps Verifiers' existing `PolicySpan(start, end)` on completion and uses `start` as the conservative staleness version. Exact token-aligned behavior logprobs, rather than per-token version labels, support the importance ratio when a tool trajectory spans versions.

Verifiers already defines exactly the async provenance boundary Prime-RL uses: `episode.PolicySpan` and optional `TrainWorkInfo.policy`. Verifiers intentionally does not discover or enforce live policy versions; the orchestration owner stamps the span. Posttrain currently stamps `policy=None`, so the concrete async producer must supply the group's dispatch version and completion-time published version. No renderers or per-node policy-version extension is required for the selected semantics. The inspected Prime-RL source is commit `04a61d3b75c3c99f263b2c133e822f998909adf7`; its Verifiers submodule `828488ff` is an ancestor of standalone upstream main `27bbd216`, to which the CarbonTeq fork is now locally synchronized.

Research anchors: [the user-linked landscape survey](https://huggingface.co/blog/async-rl-training-landscape) motivates decoupling; [released TRL v1.12.0 documentation](https://huggingface.co/docs/trl/v1.12.0/en/async_grpo_trainer) establishes the native runtime. Main-branch documentation is not evidence of feature parity with the pinned distribution. Internal source and native-consumer tests remain executable authority.

### Backend-neutral staleness and partial-rollout boundary

The landscape's Axis 4 and Axis 5 are relevant to Verifiers, but they do not make Verifiers the training scheduler. The environment layer owns durable rollout evidence and resumable episode state; the selected training backend owns admission, queueing, weight publication, and loss policy.

| Concern | Verifiers responsibility | Training-backend / orchestrator responsibility |
| --- | --- | --- |
| Version rejection | Carry a validated episode `PolicySpan`; retain stable group and rollout identities. | Compare the conservative version against the learner, and drop or wait at whole-group granularity. |
| Depth bounding | Make cancellation acknowledged and preserve terminal episode evidence. | Bound in-flight groups, queued groups, bytes, and allowed version distance; apply backpressure. |
| Importance-sampling correction | Preserve exact sampled token IDs, policy masks, and aligned behavior logprobs without retokenizing. | Select the IS estimator, clipping, masking, and loss; Verifiers never computes policy-gradient correction. |
| Soft drain | Let an active model request finish and keep the episode/tool state alive while the next request is admission-gated. | Stop new model-request admission, drain inference, publish weights, and reopen admission. |
| Abort and prefix resume | Represent a policy-update interruption as a resumable generation checkpoint, not a task failure; concatenate retained token IDs, masks, and behavior logprobs exactly once. | Abort the inference request, reconstruct/prefill the retained prefix, choose the new serving version, and resume decoding. |
| Explicit save/resume | Preserve the same logical episode, assistant turn, environment state, and trace identity across suspension. Never repeat a completed tool action. | Decide the suspension point, persist/route the checkpoint, synchronize weights, and reactivate it. |
| Mixed-weight implicit continuation | Retain provider-reported policy span and per-token behavior logprobs. | Only enable when the inference engine can atomically update at its declared boundary and the selected loss is qualified for it. |
| Group cancellation | Expose acknowledged cancellation and finalize the cancelled episode without fabricated reward. | Choose which stale group to cancel and replace; never cancel a single sibling while training the remainder as a complete relative group. |

The initial supported capability profiles are deliberately narrower than the full taxonomy:

1. `request_drain`: current model requests finish, tool-running episodes remain alive, and their next model request waits until publication completes. This fits Prime-RL's soft-pause boundary and is the first TRL target.
2. `prefix_resume`: an in-progress assistant generation may be interrupted and resumed from exact retained tokens and logprobs. This is required before Posttrain may select veRL's partial-rollout async modes.
3. `mixed_forward`: weights can change between decode forwards without interruption. Keep this unsupported until a serving backend provides exact policy-span evidence and an algorithm-specific qualification demonstrates correct loss behavior.

Do not put these profile names into task or environment configuration. They are negotiated capabilities of the policy-generation adapter selected by the training backend. A weight update while a tool is executing does not interrupt or replay the tool: the episode keeps the observation and the next policy request follows the selected admission profile.

Latest veRL has two native async generations. Its experimental fully-async policy separates Rollouter, MessageQueue, Trainer, and ParameterSynchronizer and supports bounded staleness plus optional saved-prefix partial rollout. Its newer V1 trainer exposes `sync`, `colocate_async`, and `separate_async`; the async replay buffer applies whole-prompt-group `drop` or `wait` policy from model-version spans, while `FullyAsyncLLMServerClient` retains partial tokens/logprobs and resumes after a weight transition. Source inspection corrected an earlier interpretation: `AgentLoopManager` injects its selected `LLMServerClient` into every agent loop, and both V1 async trainers select `FullyAsyncLLMServerClient`. `PosttrainVerifiersAgentLoop` therefore already calls through native partial-resume behavior. Its actual defect was replacing the returned `min_global_steps` and `max_global_steps` with the dispatch step. The adapter now retains and merges native spans across every model turn and rejects missing span evidence in async modes.
`packages/train/src/posttrain/train/backends/trl/online_rl.py` already collects pending requests, but its `_flush_pending` calls synchronous trainer generation on the environment event loop. A controlled blocking-generation reproduction established event-loop blocking; it did not establish what fraction of a production update is spent there. `policy_rollouts.py` enters collection using `asyncio.run`, making loop ownership part of the integration problem.

The real spawned-worker gate completes in the normal framework test environment without a GPU because its session returns controlled native token output. It proves process startup, config reconstruction, native client HTTP, episode wire transport, exact sampled logprobs, bridge projection, and cleanup; it does not prove model inference throughput or changed weights. During review, the worker adapter was found to await only the episode request. It now races the request against the broker lifecycle task, cancels the local request immediately when the broker exits, and raises a collection infrastructure failure instead of waiting for the episode deadline.

The TRL fork's `trl/generation/vllm_generation.py` uses blocking colocated generation waves. vLLM already schedules continuously internally; the missing behavior is incremental request submission and independent completion across the outer harness. Renaming a batch method `async`, or putting the same whole-batch call in a thread, does not remove the wave barrier.

The first fork slice, `trl/generation/async_vllm_session.py`, is deliberately an injected-engine lifecycle wrapper rather than a replacement vLLM constructor. `tests/test_async_vllm_session.py` passes using the RL workspace test environment (2 tests): a short request completes while a long request remains pending; policy synchronization is rejected during collection; abort, drain, sleep/wake, and close are fenced. The TRL checkout's own virtual environment lacks pytest/ruff, so this is a deterministic contract test, not real-vLLM or GPU evidence.

The current `packages/train/src/posttrain/train/integrations/verifiers.py` uses an injected in-process policy client and direct `environment.run_episode` calls. Native Verifiers already provides `verifiers/v1/serve/pool.py::EnvServerPool` and serializable `RunRequest`/`RunResponse` messages. Its default elastic pool and `multiplex=128` do not imply four active workers for 32 episodes. Also, multiplex is a scaling parameter, not proof of a hard per-worker admission limit.

Native `verifiers/v1/clients/train.py::TrainClient` already performs worker-local rendering, exact-token continuation bridging, and calls `/inference/v1/generate` through `renderers.client.generate`. `TrainClientConfig` is serializable. Reusing that contract is preferable to creating a second custom policy RPC protocol. Its renderer/template behavior still needs equivalence tests against the current Posttrain renderer before migration.

Environment workers already launch isolated tool/agent subprocesses through `verifiers/v1/runtimes/subprocess.py`. Adding worker processes must not remove episode-world isolation or replicate policy/judge models. Additional CPU workers cannot by themselves accelerate GPU actor forward/backward computation.

veRL already implements the relevant scheduling layers: `verl/experimental/agent_loop/agent_loop.py` creates Ray `AgentLoopWorker` processes, divides the collection across them, and gathers their results. Each worker runs concurrent episode tasks. Posttrain's `backends/verl/agent_loop.py::VerlPolicyGenerator.generate` already awaits its server manager. The remaining gaps differ from TRL: rendering explicitly constructs `Qwen35RendererConfig`, bridge calls run one row at a time, and a missing trajectory raises an exception. A semaphore local to each bridge call cannot enforce a collection-wide budget, and an unhandled episode exception can escape the worker gather. These are source findings, not a measured veRL performance diagnosis.

`PosttrainVerifiersAgentLoop.run` had discarded veRL's `sampling_params`; the implemented adapter now validates and forwards phase/per-row controls to the already asynchronous server manager. It accepts greedy `temperature=0` and native `top_k=-1`, requires log probabilities, and rejects a maximum-token override above the environment limit. `packages/train/tests/test_verl_backend.py` passes 51 focused tests after this change. Renderer generalization, worker budgets, and terminal-outcome collection remain open.

The native veRL capacity slice is intentionally local to an `AgentLoopWorker`: it gates individual episode coroutines but does not create an environment pool or duplicate an inference engine. A collection-wide value is valid only when it is paired with a per-worker limit whose product with `num_workers` is no greater than that value. `/home/hammad/projects/verl/.venv313` provides the selected Python 3.13, Ray 2.56.1, and TransferQueue test stack; focused CPU tests execute there. This is not GPU or runtime-image evidence.

The active V1 path already supplied the correct terminal group boundary: it settles every session and publishes prompt status `failure`. The actual defects were that V1 bypassed the new worker gate and that its general failure option retained partially materializable groups. Both are fixed in `5dbf667c`; the framework now selects complete-group replacement whenever bounded rollout execution is enabled. This avoids inventing a parallel Posttrain outcome protocol inside veRL.

## Decision Log

Revision 16 decision (2026-09-08): retain fixed-policy barriers only for synchronous training. Add a private Verifiers rollout worker for native TRL async training, reusing the native learner, queue consumer, packing, and weight transfer. This supersedes earlier global deferral language for planning, but does not activate an unsupported mode. Do not reuse `TrlCollectionRunner` as the async learner's outer scheduling loop, or force its inference sleep barrier on disaggregated training.

Revision 16 decision (2026-09-08, corrected after Prime-RL inspection): transport only complete, valid groups with finite episode advantages, exact token-aligned behavior logprobs, and a caller-stamped `PolicySpan`. Siblings must share the same start version; each episode may end after later publications. Native TRL's scalar `model_version` carries the span start for conservative staleness rejection. Mixed-version episodes remain valid because the loss uses each token's recorded behavior logprob. Rubrics and reward dimensions remain scorer/algorithm concerns.

1. Use native asynchronous per-request engine generation with continuous scheduling. Do not implement another token scheduler or a new static-wave batching layer.
2. Reuse the native Verifiers process pool and training client. Extend their generic cancellation, admission, or metadata seams only where contract tests demonstrate a gap.
3. Start with four fixed environment worker processes, up to eight active episodes per worker, and a global cap of 32. This is an initial configuration, not a claim that four is optimal. Fixed startup avoids the current elastic threshold hiding parallelism.
4. Retain eight prompt groups with four generations each for the selected comparison: 32 logical trajectories per update. Worker count and inference concurrency do not multiply the algorithm batch.
5. Keep one policy inference engine and one lifecycle authority. Engine-internal processes are allowed; there must not be an engine or model copy per environment worker.
6. Keep fixed-policy barriers for synchronous collection only. In native async training, preserve each episode's observed policy-version span. TRL closes model-request admission and drains admitted requests before publishing weights; tool-running episodes wait at their next model request. veRL uses its native prefix-save/resume client and replay-buffer span policy. The trainer, not Verifiers, selects staleness rejection, waiting, or importance correction.
7. Keep judge endpoint ownership with composition and rubric ownership with the scorer. Worker processes are clients of the same judge service; they do not launch four judges.
8. Do not change sampling budgets, kernels, precision, KV-cache allocation, reward weights, timeouts, or active-sampling semantics as incidental optimizations.
9. (2026-09-08) On veRL use its existing Ray agent-loop workers as the environment processes. Do not spawn `EnvServerPool` inside each Ray worker. Existing veRL rollout replicas retain model ownership; the single-engine rule applies to the initial single-replica profile, not a prohibition on supported distributed veRL topologies.
10. (2026-09-08) Share immutable collection identities, terminal episode outcomes, and pure admission/config validation. Keep engine sessions, HTTP transport, Ray orchestration, and tensor packing backend-private. This avoids forcing TRL lifecycle methods onto veRL's trainer.
11. (2026-09-08) Use consolidated canonical Verifiers paths and mandatory Git preflight. Directory suffixes such as `latest` are not version authority; the verified branch, commit, and consumer pin are.
12. (2026-09-08) Optimize rollout execution only. Prove engine lifecycle first; preserve actor computation and use optimizer execution only as an integration gate.
13. (2026-09-08) Base veRL changes on runtime source `cec7e74c361bb973b641db8dfbb75a5544c33139`. Its worker-side packing requires two-stage collection, not merely a manager hook after packed results arrive.
14. (2026-09-08) Enforce judge concurrency at one composition-owned admission proxy shared by all environment workers. Classify episode-local failures separately from invalid global execution state.
15. (2026-09-08) Make veRL's episode cap a generic native agent-loop feature. A global cap is a guarantee only when the configured Ray worker count and local semaphore bounds prove it; otherwise reject configuration rather than silently oversubscribe.
16. (2026-09-08) Reuse V1 TransferQueue prompt status as the active veRL terminal-outcome protocol. A failed sibling invalidates its complete prompt group; replacement is bounded work within the collection, not a retry of every successful group. Keep a separate typed-envelope design deferred for the legacy non-TransferQueue manager only if that path becomes a supported Posttrain target.

## Target architecture and ownership

The process diagram and HTTP/session implementation below describe synchronous TRL training. The following veRL section is equally required for synchronous veRL and overrides those transport/process choices. Fixed-policy collection applies to these synchronous paths only. Exact-token provenance, failure classification, and native evidence authority also apply to the async integration specified next.

### Native asynchronous training: implementation milestones A–D

Asynchronous training overlaps generation and optimization. A behavior policy is the version of weights that actually generated a token; it can be older than the learner. Staleness is the difference in published weight versions, not elapsed seconds or optimizer steps. A prompt group contains sibling episodes whose rewards jointly define their advantages. An advantage is the signed learning credit produced by the selected algorithm, not another judge score.

In this plan, **run-fatal** means optimization must stop before another update because shared runtime health or sample provenance can no longer be proven. It does not imply a machine crash. Broker loss, shared inference failure, failed weight publication, missing policy-version evidence, and unacknowledged cancellation are run-fatal. A declared defect confined to one episode is recoverable by rejecting its complete sibling group within the configured bound.

**A — native sample transport.** Add private `backends/trl/async_samples.py` and `packages/train/tests/test_trl_async_samples.py`. Convert complete admitted groups of existing `EnvironmentRollout` values into the actual native `RolloutSample`, after the existing algorithm computes episode advantages. Preserve prompt and completion token IDs without text reconstruction. Prefix the native full-sequence mask with zeroes for prompt tokens and preserve the rollout mask, excluding tool observations. Carry behavior logprobs at sampled coordinates, using neutral zero padding only where the mask is zero. Require an ordered caller-owned behavior `PolicySpan`, complete unique occurrences from one task, finite advantages, and full validation before returning any samples. Siblings share the span start; their end may differ as long-running episodes finish across later updates. Put the start into native `model_version`, matching Prime-RL's conservative staleness semantics. Reject truncation and missing spans; do not invent rewards or versions. Message lists are display metadata only. Tests must feed samples through the installed native queue consumer, including its stale-sample rejection, rather than only test a lookalike dataclass. This milestone computes no estimator and starts no processes.

**B — Verifiers producer.** `backends/trl/async_worker.py::TrlAsyncRolloutWorker` implements the real native worker protocol and composes an `AsyncGroupProducer`. The protocol adapter uses an explicitly bounded thread-safe native queue and an empty bounded metrics queue, owns one background event loop, and never interprets tasks or rewards. Implement a separate private `integrations/verifiers_async_groups.py::VerifiersAsyncGroupProducer` for task selection, native environment execution, group admission, algorithm advantages, and calls to milestone A projection. CPU-intensive environment execution stays in the existing native worker processes. Refactor only the scheduling identity/admission seam needed to allow independent groups; preserve the synchronous collection fence as a separate mode. Reuse `VerifiersEnvironmentBridge.project_native_episode` and existing algorithm-specific admission/advantages before sample publication. Bound outstanding episodes, scored groups, and serialized bytes; queue size zero must never mean unbounded by accident. Queue backpressure must not block weight notification, health checks, or cancellation. Native TRL remains the only queue consumer and learner. Prove a short group can reach training while a different group waits for a tool, without changing sibling membership or rewarding incomplete groups. Cancel every owned request/tool before shutdown returns; propagate infrastructure failures through `check_health`, not fabricated low rewards. No retries of tool actions are introduced. Persist a consumed-task cursor only after native learner consumption semantics are resolved; a generated or enqueued cursor is insufficient.

**C — policy provenance and weight transfer.** Implement the backend-neutral evidence and continuation boundary above, then use each trainer's native lifecycle. For TRL, reuse the Prime-RL request-drain shape: add `prepare_model_update(next_version)` (or an equivalently explicit two-phase observer) before inference pause, close model-request admission, drain active requests, publish weights, and reopen admission through `update_model_version(version)`. For veRL, do not reuse this TRL protocol. Integrate `PosttrainVerifiersAgentLoop` through the native V1 async client/continuation seam so `min_global_steps` and `max_global_steps` reflect the real generation span and an interrupted assistant generation resumes from retained token IDs and logprobs. Tool-waiting episodes remain alive under either backend and may issue their next model request only after the new version is active. Preserve behavior logprobs for importance correction and use the oldest span version for conservative staleness. Test two tool turns spanning an update, mid-generation prefix resume, no duplicate tool execution, request drainage, cache reconstruction/invalidation, whole-group stale rejection or wait, malformed/missing span evidence, weight-transfer failure, cancellation, and queue starvation. Do not build a separate weight server, copy either learner into Posttrain, or modify Verifiers to own trainer policy state.

**D — supported activation and qualification.** Before wiring public requests, amend the canonical deferred-async statement and training API semantics narrowly: backend-selected asynchronous scheduling allows explicitly bounded older behavior policies for qualified algorithms; Verifiers remains an environment executor. Add private TRL translation in `backends/trl/policy_config.py` and native veRL V1 mode translation in `backends/verl/worker.py`; do not treat `rollout_execution` worker-count settings as the scheduling mode. Validate separate learner/inference resources, negotiated partial-rollout capability, native backend/version support, queue limits, synchronization cadence and sample age before side effects. Reject unsupported combinations rather than silently selecting synchronous behavior. Start with native async GRPO; qualify episode GDPO by transporting its independently normalized weighted advantage unchanged. CAPO, SAMPO and the OLMo recipe require separate loss/credit and feature-parity audits; a scalar sample cannot claim token-wise credit support. veRL qualification must exercise its own replay buffer, whole-group stale policy, parameter synchronizer, and partial-resume client rather than emulating TRL's protocol. Existing veRL synchronous work remains required and does not establish async-training support.

Use existing evidence for a five-update GPU test with separate inference and learner devices, then checkpoint/resume and export/reload. Verify consumed samples' behavior-policy evidence, finite losses, actual weight changes and continued rollout progress during updates. Persist the consumed task cursor and native trace references consistently; on restart discard in-flight/queued ephemeral work and regenerate from the supported cursor without replaying external tool side effects against a persistent world. Test this with isolated environment state. No local paths may enter runtime images. CPU contract tests and fake inference cannot satisfy this gate. No running jobs are cancelled or rescheduled by this plan revision.

From `/home/hammad/projects/rl`, run `uv run pytest packages/train/tests/test_trl_async_samples.py` for A, then focused producer and native integration tests as B–C land. Run `uv run ruff check`, targeted `uv run pyright`, `uv run lint-imports`, the complete train suite, and `git diff --check` before handoff. Changes are additive and unselected; retry tests without changing pins or training jobs. Promotion requires commit/push of any fork changes, fork ledger and consumer documentation updates, immutable pins/lockfiles, and the existing runtime publication gates. Reject or remove the prototype if native-consumer semantics cannot preserve the selected algorithm.

The job runtime owns the complete process tree. The trainer-side coordinator owns collection identities, group admission, and the transition between generation and optimization. A dedicated asynchronous inference frontend owns request submission and completion on its own long-lived event loop; its underlying vLLM engine may use the runtime's native engine subprocess. Environment workers contain CPU environment state, renderer/tokenizer instances, and async clients, never a trainer or CUDA model.

```text
Job runtime / trainer coordinator
  ├─ one inference owner → one vLLM engine, continuous request scheduling
  └─ native Verifiers broker, bounded admission
       ├─ environment worker 1: up to 8 episodes, CPU rendering/tools/scoring
       ├─ environment worker 2: up to 8 episodes, CPU rendering/tools/scoring
       ├─ environment worker 3: up to 8 episodes, CPU rendering/tools/scoring
       └─ environment worker 4: up to 8 episodes, CPU rendering/tools/scoring
Each worker → token-preserving inference endpoint; optional shared judge endpoint
```

The environment worker returns each finished episode immediately. Inside an episode, a completed turn resumes its own tools and next request immediately. There is no barrier requiring all 32 first turns to finish before any second turn starts. The collection barrier remains at the training boundary because group-relative rewards and optimizer inputs require the admitted round's results.

`posttrain.train` owns its private Verifiers integration, backend-neutral reward/group validation, and TRL adapter. `posttrain.environment` owns reusable neutral environment delivery and native-evidence projection, without exposing concrete Verifiers types publicly. Generic engine lifecycle work belongs in TRL; generic pool/client behavior belongs in Verifiers. Neither fork imports Posttrain. Train must not import serve, eval, or Lab to start an endpoint. Composition supplies judge connections through existing ownership seams.

### Generation API and exact token provenance

Implement a job-private loopback HTTP frontend compatible with the pinned `renderers.client.generate` wire contract used by native `TrainClient`. Before writing it, inspect the selected renderer dependency's actual request/response schema and record its immutable version. Do not infer compatibility from the endpoint name or substitute ordinary chat completion strings.

Preserve prompt token IDs, completion token IDs, sampled-token log-probabilities, finish reason, and required attribution metadata. Native `TurnTokens` and episode traces remain replay authority. Responses must round-trip through native `response_from_generate` without losing continuation provenance. Worker-local renderer configuration must be pinned to the selected base model and match the Posttrain renderer fingerprint; a LoRA adapter name is not a tokenizer identity.

Reuse native request/session identifiers. Add the smallest versioned transport metadata extension needed to carry collection ID, policy version, request ID, logical occurrence identity, and deadline. Do not put training identities into task prompts. If native request headers cannot propagate these fields per episode, extend the generic native request context rather than patching AutomationBench. The server validates the active collection/version before admitting work and echoes enough identity to reject stale or misrouted completions.

Keep `PolicyGenerator` / `PolicyTurnRequest` as the existing neutral direct-generation seam. Add a private token-level adapter below it for native worker clients; both paths converge on the same backend session and sampling validation. Do not make Verifiers serialize Posttrain dataclasses. Do not reconstruct generated tokens by re-tokenizing parsed text.

The frontend binds only within the job network namespace, uses an ephemeral job-scoped bearer token passed through protected process environment, validates payload sizes and model identity, and is not a public inference deployment. Never persist credentials in traces, configs, or image layers. Local and cloud jobs use the same packaged frontend and native worker entrypoints, with no workstation paths or downloaded-code mounts.

### Engine session and policy lifecycle

Add a reusable TRL generation session with proposed operations `open_policy(version)`, `generate(request)`, `abort(request_id)`, `drain()`, `suspend_for_update()`, `synchronize_policy(version)`, and `close()`. These are planned internal contracts, not existing public APIs. Validate actual asynchronous engine weight-update, LoRA, sleep/wake, and log-probability capabilities against the pinned runtime before committing to an implementation.

All engine mutations execute through its owning loop. The trainer may wait synchronously at round boundaries, but must not synchronously execute generation on an environment loop. Do not call a non-thread-safe engine from arbitrary worker threads. Teacher-forced parity probes and other engine consumers use the same lifecycle lock and explicit phase; they cannot race generation or weight synchronization.

The legal phase sequence is: synchronize policy version N; open collection; generate and score its episodes; close admission; drain or acknowledge all aborts; assemble admitted groups; suspend inference as required for colocated memory; perform optimizer update; synchronize version N+1; reopen admission. Any failure to drain or synchronize prevents the optimizer/next collection transition. Flush or invalidate caches as required by the engine's documented weight/adapter semantics; never reuse stale-policy KV state merely because request text matches.

A global engine concurrency ceiling of 32 limits active model requests, not total conversations forever. vLLM owns its token/KV scheduling underneath. The coordinator must not wait to accumulate a full batch before submitting an available request. Continuous batching does not require streaming partial assistant messages into tools: return a turn when that individual generation finishes.

### Environment pool and bounded execution

Use `spawn`, declarative environment activation, and native `EnvServerPool`/`RunRequest`. Do not pickle the current injected policy-client closure or inherit CUDA through `fork`. Prestart four workers with `elastic=False`; explicitly enforce the eight-episode worker limit rather than assuming native multiplex provides it. Least-active routing must respect available capacity.

Acquire global episode admission before dispatch. Hold it for the episode lifecycle, including tool waits and scoring, then release it exactly once. Judge concurrency is a separate global service budget; a per-worker semaphore of 16 would accidentally permit 64 requests and is not acceptable. Enforce it at the single composition-owned judge admission proxy specified below, preserving the configured global limit.

Bound outstanding episode requests, inference requests, and transport bytes. The existing pool's unbounded socket high-water marks are not a sufficient memory policy. Prefer application-level admission and bounded send/receive queues; prove that backpressure cannot prevent cancellation/control messages from being serviced. Size payload limits from supported prompt/output budgets and serialized native evidence, and reject oversize requests explicitly rather than truncating traces.

Workers reuse renderer/tokenizer and interpreter resources that are safe to share, but create isolated episode state. Preserve tool subprocess process-group cleanup. Start native BLAS/tokenizer thread limits conservatively at one per environment worker, then qualify under effective CPU affinity/cgroup limits. Reject or explicitly reduce an impossible worker configuration; do not infer physical CPU capacity from the host-wide count alone. No persistent mutable tool-world pooling in this change.

### Scheduling correctness, failures, and cancellation

Assign group, occurrence, seed, and collection identities before dispatch. Reassemble results in logical source order, not arrival order. Completion order must not alter reward alignment, group membership, or optimizer normalization. Seed identity remains stable across worker placement; bitwise GPU sampling equality is not promised when engine scheduling changes.

Keep existing algorithm-specific group admission. An invalid member excludes the appropriate complete group, not all otherwise-valid groups. No fastest-first replacement, generic automatic retry, or hidden resampling is added. OLMo active sampling remains an explicit algorithm policy. An all-invalid batch follows its existing typed empty-admission behavior and never produces a fabricated optimizer update.

Cancellation propagates from collection to episode, active model request, and owned tool subprocess group. HTTP disconnect alone is not proof of engine abortion: implement and test explicit abort or reliable disconnect-to-abort handling, with acknowledgment. An episode deadline includes queueing and execution; transport hops must not reset its remaining budget. All processes are local to the same job host initially, allowing a consistent monotonic deadline domain; multi-host workers are out of scope.

A worker death terminalizes its assigned episodes, preserves available native error evidence, and lets unaffected groups finish only when its owned subprocesses/requests have been cancelled or fenced and admission accounting is trustworthy. Do not replay partially executed tool actions automatically. A replacement worker may serve future work only after the failed worker is reaped and capacity accounting is repaired. If native pool behavior cannot isolate worker death, add that generic fork seam and its regression tests before enabling the mode. Inability to establish safe isolation escalates to collection failure.

Classify malformed policy output, an episode deadline, and an isolated tool/worker failure as episode-local terminal outcomes, subject to whole-group admission. Classify policy engine loss, broker corruption, provenance mismatch, failed abort/drain, failed weight synchronization, and shared judge-service failure as `CollectionExecutionError`. A malformed individual judge assessment is invalid reward evidence for that episode; an unavailable shared judge is not a batch of zero scores. Explicit job cancellation stops the collection and must not optimize a conveniently finished subset. Never catch every exception and convert it into an excluded group: only declared recoverable error classes take that path, and unexpected errors fail visibly.

### Judge admission and resource handoff

Composition owns one `JudgeAdmissionProxy` for the selected judge provider, using the existing managed-judge composition path as its integration point. Workers receive only this proxy's compatible endpoint. It forwards rubric requests unchanged to the existing provider and owns a global bounded admission gate plus active-request registry. Its contract is `start(provider, limit)`, `stop_admission()`, `drain(deadline)`, and `aclose()`. Reuse an existing proxy component if equivalent; do not implement the gate independently in each worker or scorer. Bind it in the job host without importing serve/eval into train or either fork. For multi-node veRL, use an authenticated job-private reachable endpoint, not worker-local loopback.

The proxy preserves request deadline, cancellation, provider errors, and raw assessment evidence; it does not reinterpret rubrics, average rewards, add retries, or launch a model. A request queued at the proxy counts against its end-to-end deadline. Verify 32 callers across four workers never exceed the configured judge limit and cancellation releases capacity exactly once.

At collection completion, all required assessments must be terminal. Composition drains the proxy before the existing optimizer boundary. For a colocated managed judge, its provider lifecycle must acknowledge release of GPU residency through a qualified sleep/unload operation before training proceeds; for a remote judge, only request drainage is required. If the selected managed provider cannot release residency safely, reject that colocation profile during admission rather than silently changing actor settings or killing an unrelated server. Resume the provider and acknowledge readiness before the next collection. The rubric plugin and Verifiers know nothing about GPU release.

On job shutdown, stop admission, cancel queued/active work, drain acknowledgments, close native clients and broker, join workers, and terminate remaining owned subprocess groups after the existing shutdown grace. Never kill unrelated host processes. Late completions are ignored by identity fencing; every logical episode is persisted at most once through the existing evidence sink. Return native `WireEpisode` data without adding a parallel trajectory database.

## Configuration and developer experience

Keep engine execution selection in `InferenceBinding.engine`; propose `request_mode: async` as an adapter-validated option. It means per-request asynchronous submission, not an algorithm change. Keep worker deployment in a validated private `TrainingBinding.backend_options.rollout_execution` mapping for this first train-only implementation: proposed keys are `env_workers: 4`, `episodes_per_worker: 8`, and `worker_native_threads: 1`. `EnvironmentBinding.max_concurrent: 32` remains the global episode cap. Validate these together; never multiply global concurrency by worker count.

These names are proposed implementation surfaces, not currently accepted configuration. Add typed private parsing and detached validation before enabling them in a catalog. Native options such as `elastic=False` are derived adapter details, not new algorithm fields. Preserve a direct execution mode for controlled regression comparison. Unsupported backend/mode combinations fail before scheduling, not by silently falling back. A future eval consumer can extract a shared neutral execution value when it has demonstrated requirements; do not add a public cross-capability abstraction speculatively.

On veRL, map `env_workers` to `actor_rollout_ref.rollout.agent.num_workers`, and retain native `actor_rollout_ref.rollout.mode=async` as the engine execution setting. `episodes_per_worker` must become an enforced worker bound, not a second batch-size setting. Reject conflicting native overrides. A common selection can express the same 4-worker/8-episode intent, while its adapters emit different native configuration. Allocate Ray CPU resources explicitly and account for renderer threads and tool subprocesses in the job CPU reservation; a Ray CPU reservation is scheduling accounting, not an operating-system CPU isolation guarantee.

## Concrete components and interfaces

All names in this section are selected implementation targets unless explicitly identified as existing. New files are private modules under `packages/train/src/posttrain/train`; they are not new public framework primitives. Reuse an equivalent existing type rather than creating a duplicate, recording the resolved name here. Do not create a generic `HarnessManager` with backend-name conditionals.

### Shared values and validation, without shared process ownership

Create `rollout_execution.py` with frozen, serializable values `CollectionKey(run_id, collection_id, policy_version)`, `EpisodeKey(collection, group_id, occurrence_id, seed)`, and `RolloutExecutionConfig(env_workers, episodes_per_worker, worker_native_threads)`. Logical occurrence identity is assigned before scheduling; a model request additionally has a unique turn request ID. Keep the existing native episode identity as evidence provenance rather than replacing it with these scheduling identities.

Implemented as `packages/train/src/posttrain/train/rollout_execution.py`. `EpisodeKey` also retains `example_id`, because successful results must be joined to source rows before they can enter a group. `validate_execution_config` rejects capacity greater than the global limit and native-thread reservations greater than effective CPU affinity. `validate_outcome_identity` rejects late/misrouted results and mismatched structured evidence. `packages/train/tests/test_rollout_execution.py` passes three focused tests. The values are not wired to a backend yet, so no catalog setting has changed.

Define `EpisodeOutcome(key, status, rollout, error)` with statuses `completed`, `invalid`, `cancelled`, and `failed`. A completed outcome contains an existing validated rollout value; other statuses contain a typed reason and optional native evidence reference, never a fabricated zero reward. A separate `CollectionExecutionError` represents an unusable engine, corrupt transport, or failed synchronization: it stops the round instead of pretending every such failure is ordinary bad model output.

Expose pure functions `validate_execution_config(config, global_limit, effective_cpus)` and `validate_outcome_identity(expected, outcome)`. Reuse the existing algorithm group-admission implementation through a narrow adapter accepting ordered outcomes; do not reimplement advantage estimation in this module. Return retained row indices and excluded-group reasons. Use those indices before backend advantage estimation and all tensor packing; the same indices select rewards, masks, prompts, old log-probabilities, and metadata. Preserve each algorithm's existing incomplete/empty-group policy.

No shared `start_workers()`, `update_weights()`, or `sleep_engine()` protocol is introduced. The two backends own those operations differently. Shared tests assert observable outcomes rather than requiring identical native call sequences.

### TRL components

`backends/trl/policy_endpoint.py::TrlPolicyEndpoint` owns the loopback HTTP server and its request registry, with `start(session)`, `stop_admission(collection)`, and `aclose()`. The registry maps a native request ID to its collection and active generation task. Endpoint handlers validate token requests, await generation, and abort on explicit cancellation. They do not score rewards or launch workers.

`integrations/verifiers_workers.py::VerifiersWorkerPool` owns native `EnvServerPool` startup, health, dispatch, and shutdown. Its implemented interface is `async start(activation, client_config, execution_config)`, `async open_admission(collection)`, `async run_episode(key, task, deadline) -> EpisodeOutcome`, `async cancel(key)`, `async stop_admission(collection)`, and `async aclose()`. The additional open/stop methods make the fixed-policy fence explicit while keeping workers resident. Native Verifiers owns environment execution; this adapter translates task/outcome contracts and capacity and does not inspect algorithm names. Stable wire IDs and acknowledged cancellation require Verifiers commit `c6c0097ad21da845c62e4b19aba80ef6633e4d9f`.

`backends/trl/collection_runner.py::TrlCollectionRunner` composes that pool with the endpoint and TRL session. Its implemented `async collect(collection, scheduled_episodes) -> tuple[EpisodeOutcome, ...]` returns terminal outcomes in scheduled order. It owns the phase sequence around the already-bounded worker pool and closes the collection only when every scheduled occurrence is terminal. Policy synchronization failure, endpoint fatal state, failed cancellation, and failed drain/suspend are collection errors. The synchronous trainer callback now invokes this path only when both the inference binding selects `request_mode: async` and the training binding declares a complete bounded worker topology; existing reward/group admission continues above it and optimizer execution remains synchronous. Verifiers is never asked to understand policy optimization.

`trl/generation/async_vllm_session.py::AsyncVllmSession` in the TRL fork implements the previously specified synchronous-training engine lifecycle operations. `generate` takes token-level engine input, not Posttrain or Verifiers objects. This is the only new engine lifecycle authority for the colocated path. Keep completion-logprob probes and LoRA synchronization inside this authority. Native asynchronous training instead reuses `AsyncGRPOTrainer` and its weight-transfer protocol; it does not instantiate this second colocated owner.

### veRL components and control flow

Preserve `backends/verl/agent_loop.py::PosttrainVerifiersAgentLoop` as the thin native entrypoint. Each invocation runs one Verifiers episode within an existing Ray agent-loop worker. Extract CPU renderer construction into `backends/verl/rendering.py::create_policy_renderer(model_contract, tokenizer)`, resolving the selected model's declared renderer rather than hard-coding Qwen. This helper owns no processes or model weights. Reuse renderer selection already available elsewhere through a neutral helper only if it introduces no backend import dependency.

Keep `VerlPolicyGenerator` as the adapter to the native server manager. Assign distinct per-turn request IDs while retaining stable episode identity separately; validate cancellation and routing against the pinned server-manager contract. It returns exact sampled IDs/log-probabilities through existing `PolicyTurnResult`. It does not create a TRL-style HTTP endpoint or instantiate an engine.

Add `backends/verl/collection.py` to translate shared outcome/group decisions to veRL source rows and `DataProto` fields. It must not become another collection scheduler. `worker.py` and `launcher.py` pass validated worker settings and selected renderer information to the native runtime. No renderer or CPU capacity settings belong in GRPO/GDPO algorithm settings.

In the veRL fork, extend the existing `AgentLoopWorker.generate_sequences` with a worker-wide bounded episode gate shared across all row invocations, and terminal per-row result handling. Use the manager's existing chunk allocation to assign worker budgets whose sum is at most the global limit (32 for the initial profile). Do not independently grant every worker the global allowance. Simultaneous training and evaluation collections must be serialized by the existing lifecycle unless a later explicit design supplies a shared global gate.

The pinned veRL worker currently runs `asyncio.gather` and then `_postprocess` locally before returning `DataProto`. Introduce an opt-in two-stage native path: `collect_episode_outcomes(batch)` returns terminal per-row envelopes containing successful native episode output or a typed failure; the manager gathers those envelopes and invokes global group admission; `pack_retained_outcomes(outcomes, source_rows)` then runs existing postprocessing only on retained successes. Keep the legacy `generate_sequences` return contract for callers not selecting this path. Separate any necessary per-episode CPU/reward preparation from batch tensor packing; do not transfer CUDA tensors or giant unbounded Python object lists through the new envelope path.

The collection manager owns the pre-packing admission hook, supplied by Posttrain's `collection.py` adapter. Four siblings can reside on different workers, so no worker independently drops or normalizes a partial group. Returned source-row identities must join exactly with the trainer's input batch before unioning results. Preserve distributed batch balancing and divisibility requirements using the existing backend-supported mechanism; do not silently drop extra valid groups to make shapes fit. Test the 28-retained-row case against the actual selected training layout.

This hook returns retained source indices before failed rows are packed into training tensors. It does not manufacture `AgentLoopOutput` token arrays for a failed episode. Empty admission uses the algorithm's existing explicit behavior; infrastructure failure remains fatal. Confirm every downstream field and distributed batch divisibility requirement after filtering, preserving the established retained-group correctness fixes.

### Sampling precedence and request identity

Remove `del sampling_params` from `PosttrainVerifiersAgentLoop.run`. Resolve one effective sampling value per occurrence: inference binding defaults first, explicit native trainer phase/per-row overrides second, then validate against environment and model hard limits. Environment episode limits bound remaining output but do not silently replace trainer temperature/top-p overrides. Algorithm-required sampling overrides remain authoritative and conflicting explicit settings fail validation. Validation/greedy requests must support temperature zero at the generation boundary; do not broaden training exploration rules accidentally when adjusting the current positive-temperature `PolicySampling` validation.

Pass effective sampling through `VerlPolicyGenerator` and the native TRL client unchanged except for documented engine sentinel conversion (for example, disabled top-k). Derive stable per-occurrence/per-turn seeds from the run seed and logical identity where the backend supports explicit request seeds; do not call process-global RNG reseeding from concurrent tasks. Test training defaults, greedy validation, per-row overrides, max-token remainder, distinct turn IDs, and exact log-probability semantics. Claim deterministic identity/seed assignment, not bitwise scheduling-independent GPU output.

veRL's existing rollout manager and checkpoint/weight synchronization machinery owns the phase transition. The Posttrain adapter waits for its terminal collection output, then existing reward/advantage and optimizer code proceeds. On cancellation, the manager cancels native worker tasks and in-flight server-manager requests before sleeping replicas or starting an update. Add generic abort plumbing to the fork if required; do not assume cancelling a Ray object reference cancels GPU generation. For multi-node operation, transport remaining deadline durations rather than comparing monotonic timestamps from different hosts.

### Deliberate non-abstractions

Do not merge Ray and native Verifiers pools behind a fake common worker executor. Do not share TRL's engine session class with veRL. Do not move rubric/judge code into worker orchestration, add an environment service to `common`, or change the public `PolicyGenerator` into a deployment API. Share value contracts and correctness checks; let native runtimes schedule and own resources.

## Plan of Work

### Milestone 0: verify source and engine feasibility

Read `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/verl-py313/profile.toml` and resolve `fork_revision` (`cec7e74c361bb973b641db8dfbb75a5544c33139` at this revision). Inspect that commit's agent-loop and trainer code, establish the selected clean implementation branch, and update the checkout table. Do not develop against detached historical HEAD and port later by assumption.

Before implementing HTTP or process-pool integration, run a bounded TRL GPU lifecycle proof on the pinned vLLM runtime: initialize one async engine, complete independent overlapping requests, abort one request, drain, release residency, synchronize changed LoRA weights, wake, and generate again. Verify sampled log-probability parity through the existing gate and that the second round uses updated weights. Record exact runtime versions and commands. Failure blocks the proposed async mode; it does not authorize changing precision, actor settings, or parity tolerances. The proof is small test code, not new instrumentation or a full training run.

The deterministic half of this milestone began at TRL commit `2faf864cc5728aad6c07f3871067de4f40e3acb0`. Pushed commit `684696a22ef3d82dbf39f21a60fafa9e5f17514b` additionally checks the actual vLLM 0.25.1 async API and supplies `scripts/qualify_async_vllm_lifecycle.py`. The real-engine command below passed on the local RTX 3070 Ti with the cached Qwen 0.5B model. It completed two collection rounds, explicitly cancelled one in-flight request without returning a rollout, returned one logprob entry per sampled token, drained, slept, restored weights and KV cache in separate stages, and shut down cleanly:

```bash
cd /tmp/trl-parity-probe.WIQjjv
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/tmp/trl-parity-probe.WIQjjv \
/home/hammad/projects/rl/.venv/bin/python scripts/qualify_async_vllm_lifecycle.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 512 --max-num-seqs 2 --gpu-memory-utilization 0.45
```

The first GPU attempt exposed that vLLM keeps scheduling paused after `wake_up(tags=["weights"])`; opening admission must restore `kv_cache` separately. Closing a still-sleeping CuMem engine also emitted a CUDA allocator cleanup error, so close now restores allocations before shutdown. The repeated gate passed without either defect. The command reports `weight_update_parity_tested: false` deliberately: changed LoRA/full actor weights and the existing raw actor/sampler parity gate still need to be connected to this single engine owner. Do not route requests back through `VLLMGeneration._generate_colocated_waves`, and retain `VLLMGeneration` unchanged until that remaining proof demonstrates equivalent token and log-probability behavior.

### Milestone 1: prove the native wire and rendering seam

In RL, add focused tests beside `packages/train/tests` and `packages/environment/tests` for native `TrainClient` requests and native episode projection. Cover LFM reasoning, content, tool calls, multi-turn exact-prefix bridging, sampling settings, and log-probability arrays. Compare with the current `TrlPolicyGenerator` path using controlled token fixtures. A renderer mismatch blocks migration; fix the generic renderer seam rather than introducing task-specific string surgery.

The first parity test found a real mismatch before endpoint implementation: native `TrainClient` loaded the model artifact's bundled LFM template, while Posttrain selects a corrected package template that serializes historical structured tool calls. Verifiers commit `c6c0097ad21da845c62e4b19aba80ef6633e4d9f` adds optional `TrainClientConfig.chat_template`, applies it after tokenizer load, and includes it in the renderer-pool key. Posttrain's private `create_verifiers_train_client_config` requires that capability and accepts an explicitly resolved immutable tokenizer artifact/path rather than guessing from a mutable model alias. The LFM parity test compares token IDs, message attribution, and content attribution for a user to structured assistant tool call to tool-result history. The loopback endpoint passes native request/response schema checks, preserves exact prompt and completion IDs plus selected-token logprobs, rejects context overflow before generation, acknowledges cancellation, and records engine/provenance failures as fatal collection state. A real `TrainClient` also completed a single-turn LFM request over this HTTP path with exact returned token and logprob arrays. Live multi-turn prefix bridging and native environment-worker execution are still required before this milestone is complete.

Inspect `verifiers/v1/configs/client.py`, `clients/train.py`, `serve/types.py`, and the pinned renderer client. Extend native metadata/cancellation only as demonstrated necessary. Add tests under the fork's existing test layout. Record the actual protocol and dependency version in this plan before implementing the frontend.

### Milestone 2: implement a single asynchronous engine owner

In the TRL fork, extend `trl/generation/vllm_generation.py` with a separately testable session implementation, proposed file `trl/generation/async_vllm_session.py`. Reuse existing model load, LoRA synchronization, sleep/wake, and sampling conversion behavior. Do not maintain a second independently initialized rollout model.

In RL, add a private frontend, proposed `packages/train/src/posttrain/train/backends/trl/policy_endpoint.py`, adapting the native wire format to that session. Update `online_rl.py` and `policy_rollouts.py` to manage a long-lived session rather than create an engine-facing event loop per callback. Keep existing direct generation working until the new path passes parity tests.

### Milestone 3: integrate native workers and coordinator lifecycle

Split process/client lifecycle out of `integrations/verifiers.py` into a private proposed `integrations/verifiers_workers.py`; retain reward projection and group admission at their existing ownership boundary. Use native pool dispatch, not a new multiprocessing executor around `run_episode`. Add capacity, deadline, cancellation, and late-result fencing tests. Modify the native pool only for proven hard-limit, cancellation, or worker-death gaps.

Add private config validation and catalog compatibility checks in the train adapter and its existing schema tests. Update selected comparison bindings only after the complete path is qualified. Package worker entrypoints, pinned renderer dependencies, and runtime-native async engine requirements into `packages/runtime-images`; no local checkout imports may enter the actual-job image.

### Milestone 4: implement veRL-native execution

Use the runtime-pin branch established in milestone 0. In `backends/verl/agent_loop.py`, `worker.py`, and `launcher.py`, implement renderer selection, explicit sampling precedence, execution-config translation, unique turn IDs, and typed episode outcomes. Add `rendering.py` and `collection.py`. In the veRL fork's `verl/experimental/agent_loop/agent_loop.py`, implement bounded worker admission and the two-stage collect/admit/pack path; integrate the admitted source-row join at the existing trainer boundary in `verl/trainer/ppo/ray_trainer.py`. Generic fork code must never import Posttrain.

Add proposed consumer tests `packages/train/tests/test_rollout_execution.py` and `packages/train/tests/test_verl_collection_execution.py`, plus native fork tests for concurrent worker budgets and recoverable row failures. A controlled 8-by-4 collection with one failed sibling must exclude exactly its complete group, retain the other seven groups in source order, and avoid batch-wide failure. Verify correct algorithm normalization with the retained 28 trajectories rather than assuming original batch size 32. Separately prove infrastructure failure stops optimization. Acceptance includes model renderer selection for Qwen and LFM using exact-token fixtures, not a claim of real LFM GPU compatibility yet.

### Milestone 5: validate correctness and useful concurrency

Run deterministic tests proving that short request B completes while long request A remains active, and request C can enter before A finishes. Repeat against the real selected vLLM runtime. A fake async wrapper around a blocking wave fails this acceptance test. Verify four distinct environment worker PIDs, hard global/per-worker limits, isolated task state, bounded overload, cancellation during tool/model waits, worker death, and cleanup without orphan model/tool processes.

Verify reordered completions produce the same controlled reward vectors, token masks, admitted source groups, and loss normalization on both adapters. Cover GRPO, OLMo-recipe GRPO, GDPO, CAPO, and other existing consumers where each is supported; do not claim GPU qualification for every algorithm from a shared unit test. Explicitly list unsupported algorithm/backend combinations in the qualification result rather than silently routing them to another algorithm.

Run real five-optimizer-step GRPO and OLMo canaries with the selected LFM model, eight groups by four generations, and global concurrency 32. Then run the intended 20-step comparison profiles. Check finite losses, actual parameter updates, checkpoint/resume, and exported-model inference using existing evidence. Keep the current actor/rollout log-probability parity gate; scheduling changes do not justify relaxing its tolerance or bypassing any unresolved numerical defect.

These GPU gates apply separately to TRL and veRL for supported recipes. First establish that the pinned veRL runtime actually supports the selected LFM model, precision, and adapter configuration; replacing the hard-coded renderer alone is insufficient. If that model runtime fails, report the LFM/veRL cell blocked with evidence. A supported Qwen canary may independently qualify the veRL harness, but cannot satisfy the requested LFM comparison. On both backends verify no optimizer update begins while any episode/model request remains nonterminal. On veRL verify the intended four Ray environment workers and absence of nested Verifiers pools.

Compare the current direct mode and new mode using identical task selection, model/reward settings, resource allocation, and budgets. Use existing logs/native timings and external wall-clock duration to compare complete rollout collection time and valid-episode throughput, not GPU utilization alone. Optimizer records establish safe handoff and correctness, not an actor-speed improvement claim. Report nondeterministic sampling and tool latency as limitations. Require demonstrated independent request progress and no correctness regression; measure speedup rather than promising a fixed multiplier.

## Concrete validation commands

From `/home/hammad/projects/rl`, first run `uv run pytest packages/train/tests packages/environment/tests` with the newly added test names selected during development. Before consumer promotion run:

```bash
uv sync --all-packages --locked --python 3.13
uv run ruff check .
uv run pyright
uv run lint-imports
uv run pytest
git diff --check
```

In each affected fork, run its native focused tests and supported lint/type checks from that fork's root, recording exact commands and outputs here as the new test paths exist. Before the GPU gate, record the immutable image digest and explicit canary submission/status/checkpoint commands from the current project CLI. Do not invent executable job IDs or describe a proposed command as a completed run. GPU/network tests must use existing markers and explicit missing-resource skips; a skip cannot satisfy the release gate.

## Publication, rollout, and recovery

The dependency order is native protocol/pool changes, TRL and veRL fork changes, then consumer integration and immutable runtime qualification. Local multi-repository integration tests may use explicit development sources, documented as unreleased. Commit and push each changed fork, update its `CARBONTEQ_FORK.md`, publish its required distribution, and only then update RL immutable dependency pins, runtime constraints, and `uv.lock` as applicable. Update `docs/tooling/verifiers/README.md`, `docs/tooling/trl/README.md`, and `docs/tooling/verl/README.md` in the same logical consumer change, including qualification scope and remaining release gates. Do not describe the veRL checkout SHA as published until its remote and immutable artifact are verified.

Start opt-in, with a new inference/training binding revision. Existing jobs retain their resolved selections. Promote the async mode only after all gates above; remove compatibility code in a later explicit migration, not during qualification. If qualification fails, stop the candidate job through its normal lifecycle and use the prior immutable binding for a new run. Do not overwrite prior evidence or resume a partially collected round with a different policy/renderer configuration. Recovery starts from a complete supported checkpoint, never serialized live worker state.

## Outcomes & Retrospective

Current outcome: both backends have concrete components, selected interfaces, ownership, config translation, failure handling, and separate qualification gates. TRL's asynchronous engine lifecycle is proven against the selected vLLM runtime for request execution and residency transitions; its private loopback endpoint implements the native token wire with strict provenance; its fixed worker adapter and collection runner now make cancellation and safe optimizer handoff explicit. A real spawned AutomationBench worker completed the native transport and existing bridge projection path, including retained native evidence. A separate external-server GPU gate proves that an actor weight changed through the production NCCL client reaches vLLM and preserves selected-token log-probability parity. The synchronous TRL callback now activates the continuous-batched path through an explicit paired request-mode/topology selection while retaining the old batch path as a fallback. veRL reuses Ray workers and native rollout lifecycle. Shared values and admission validation preserve algorithm semantics without a universal process manager. An integrated changed-weight optimizer update, immutable runtime publication, and measured end-to-end GPU speedup remain unproven.

Revision 4 review outcome: the runtime source mismatch is resolved in the plan; veRL's worker-side packing is explicitly addressed rather than deferred to a late hook. Engine feasibility precedes transport implementation. Sampling overrides, a single judge admission owner, and recoverable versus fatal failures have selected rules. Actor compute optimization is excluded. These are design decisions awaiting implementation and qualification, not completed runtime fixes.

Revision note: created 2026-09-08 following the user's request for a detailed plan without new instrumentation; separated execution optimization from algorithm correctness and stale-policy async RL.

Revision 2 note: updated 2026-09-08 to make veRL mandatory and replace vague shared-harness language with named components, method contracts, selected ownership boundaries, and pre-advantage failure admission. Native worker orchestration is backend-specific; verifier task semantics remain independent of optimization.

Revision 3 note: updated 2026-09-08 after worktree cleanup with canonical paths, exact active branches/commits, detached veRL warning, stash recovery notes, and mandatory checkout preflight. Older path references must not direct new implementation work.

Revision 4 note: updated 2026-09-08 following review and the user's explicit rollout-only scope. Added an engine feasibility gate, pinned veRL implementation base, two-stage collection, sampling precedence, global judge admission/resource handoff, and explicit error classification; no actor forward/backward optimization is included.

Revision 5 note: updated 2026-09-08 after the first implementation slice. TRL commit `2faf864c` adds an additive asynchronous session and deterministic lifecycle tests; no consumer pin, runtime image, environment transport, actor behavior, or GPU qualification changed.

Revision 6 note: updated 2026-09-08 after adding framework-owned rollout execution values and focused tests. This does not yet start processes or alter collection behavior; it makes the common admission and capacity invariants concrete before either backend consumes them.

Revision 7 note: updated 2026-09-08 after veRL sampling propagation. The change affects rollout sampling only; it does not alter actor update configuration, native worker count, model renderer selection, or process orchestration.

Revision 8 note: updated 2026-09-08 after fork commit `f73ca959` on pushed branch `codex/verl-rollout-execution`. It adds an opt-in native worker gate and validation only. The framework has deliberately not emitted the new settings: its immutable runtime profile still pins the parent commit, so consumer configuration must wait for a published artifact and pin update. Two-stage outcome collection/admission remains required before this gate can make partial failures safe.

Revision 9 note: updated 2026-09-08 after source inspection corrected Revision 8's assumption about the active V1 path. TransferQueue already owns two-stage prompt termination and failed-group visibility. Fork commits `c5c34bfb` and `5dbf667c` route V1 sessions through the gate, preserve complete groups, and reserve Ray CPUs. The framework emits 4×8→32 native settings only for the exact supporting revision and rejects legacy revisions or oversubscription before Ray starts. The stable runtime profile and release hashes remain unchanged pending artifact and GPU qualification.

Revision 10 note: updated 2026-09-08 after the bounded native vLLM 0.25.1 GPU lifecycle proof. It records the two staged-wake defects found by the first attempt, the clean repeated result, pushed TRL commit `684696a2`, and the still-open changed-weight parity gate without overstating this as trainer or throughput qualification.

Revision 11 note: updated 2026-09-08 after native-client renderer parity work. A generic Verifiers config seam now carries a versioned selected chat template to workers, and Posttrain constructs it from the existing model/renderer contract. This closes the demonstrated LFM template mismatch but not the endpoint, full native episode wire, or immutable package pin.

Revision 12 note: updated 2026-09-08 after implementing the loopback native token endpoint. Invalid client requests are recoverable and never reach the engine; engine failures, malformed terminal outputs, request mismatch, and prompt-token mismatch poison the active collection. Unsupported priority and cache-salt fields are rejected rather than silently discarded. Native schema/parser tests and a real LFM `TrainClient` single-turn request pass. The pushed Verifiers renderer revision is selected by framework manifests, catalogs, and candidate control locks. The published veRL backend profile remains byte-consistent with its existing OCI manifest until a replacement image is built and qualified.

Revision 13 note: updated 2026-09-08 after implementing the native worker-pool lifecycle and closing Verifiers' fire-and-forget cancellation gap. The coordinator now derives opaque stable wire IDs from full episode identity and treats missing cancellation acknowledgment as collection-fatal. Deadline and explicitly cancelled episodes remain local terminal outcomes only after native termination is accounted for. Real spawned-worker execution, bridge projection, and collection-runner integration remain separate acceptance gates.

Revision 14 note: updated 2026-09-08 after implementing the TRL collection phase coordinator in its own backend-private module rather than enlarging the existing synchronous rollout callback. It returns only after all episode tasks are terminal and inference suspension succeeds. This is deterministic lifecycle proof, not evidence that the current trainer selects the path or that a real environment process completed an episode.

Revision 15 note: updated 2026-09-08 after connecting native episode projection and terminal outcomes to the existing bridge/admission semantics. The real spawned-worker gate closes the CPU process/wire/projection milestone with controlled token generation. Broker death now fails immediately. The path remains opt-in and deliberately unselected by the trainer until the TRL fork owns one asynchronous engine with actor-to-sampler weight synchronization.

Revision 16 note: updated 2026-09-08 after reviewing TRL's released async trainer and exact fork implementation. It separates synchronous fixed-policy collection from native asynchronous learning, assigns Verifiers producer versus TRL learner ownership, records the behavior-policy provenance gap, and defines algorithm-specific promotion gates. Milestone A is implemented in `backends/trl/async_samples.py`: it transports complete admitted groups into native samples without retokenization or estimator changes, and passes six focused tests including the native queue's stale-sample behavior. No public config, running job, dependency pin, or runtime image changed.

Revision 16 implementation continuation: added `TrlAsyncRolloutWorker` as the small native protocol adapter and kept Verifiers orchestration behind `AsyncGroupProducer`. Focused tests prove bounded queues, independent group progress, failure propagation, version monotonicity, and shutdown. The concrete Verifiers producer is intentionally not connected through the synchronous pool's fixed-policy admission key; that identity boundary must be split without weakening synchronous fencing, together with served-policy evidence in milestone C.

Revision 16 validation checkpoint: 11 focused async transport/worker tests pass; after the `PolicySpan` correction, the complete train suite passes with 377 tests and 5 existing skips. Repository-wide Ruff and all eight import contracts pass. Targeted Pyright for the four new source/test files reports no errors. Repository-wide Pyright still reports 24 errors in pre-existing catalog, Observatory, environment, eval, and earlier rollout tests; none are in the revision 16 files, so they are recorded rather than mixed into this rollout-focused change.

Revision 16 provenance correction: current Prime-RL proves Verifiers' existing `PolicySpan` is the intended async contract. Its orchestrator stamps group dispatch and completion versions, stops scheduling before weight updates, and uses span start for staleness while exact behavior logprobs remain token-aligned. The earlier proposed per-token version/renderers extension was unnecessary and is superseded. Milestone A now accepts update-spanning episodes, and milestone C targets TRL's missing two-phase scheduling observer instead. This is why inspecting the native consumer changed the design before another fork was created.

Revision 16 Verifiers synchronization checkpoint: canonical branch `codex/carbonteq-verifiers-latest` was locally synchronized at `36eac9d5e04ef29b584b6fa4f027af00cd76ea19` to upstream main `27bbd216df0af719a43705866b2cf6139bcc95de`. The fork ledger records the retained CarbonTeq delta, Prime-RL's consumer-owned `PolicySpan` boundary, and which staleness/partial-rollout strategies the current evidence and cancellation contracts can support. The complete `tests/v1` suite passed; only credential-dependent Prime cases skipped. This checkpoint preceded the later task-config wire correction.

Revision 16 veRL provenance checkpoint: the backend-neutral `BehaviorPolicySpan` now lives beside the policy-turn and episode contracts. `VerlPolicyGenerator` reads veRL's native per-generation `min_global_steps` and `max_global_steps`, merges them across a multi-turn Verifiers episode, and returns the real span to veRL's replay buffer. Synchronous mode may use its dispatch step when the server emits no version; `colocate_async` and `separate_async` fail closed on missing or malformed evidence. Focused veRL plus TRL transport tests pass (count recorded by the validation checkpoint below). This does not yet activate an async trainer mode.

Revision 16 TRL request-drain checkpoint: fork commit `430215060562302e334758a0beadef5869702ee5` adds a two-phase trainer notification to async GRPO and async distillation. `prepare_model_update` atomically closes admission and waits for the child process's active model-request count to reach zero before vLLM pause; `update_model_version` publishes the new version and reopens admission after resume. This removes the event-check race while preserving tool-running episodes. Fork tests prove exact lifecycle ordering and acknowledged drain. The commit is now pushed as an ancestor of `3972dc39` but remains unselected; changed-weight GPU qualification remains open.

Revision 16 policy-lifecycle validation checkpoint: 68 focused TRL/veRL transport tests pass, including single-source episode provenance, multi-turn veRL span merging, missing-evidence rejection, group gating, and two-phase producer notification. The complete Posttrain train suite passes with 380 tests and 5 existing skips. Targeted Pyright reports no errors, Ruff passes, all eight import contracts remain intact, and `git diff --check` passes. The TRL fork's experimental pair has 89 passing tests; 9 trainer execution cases remain unavailable because this checkout lacks the optional compatible `kernels`/Flash Attention runtime, not because of a lifecycle assertion failure.

Revision 16 rejection checkpoint: `TrlAsyncRolloutWorker` now distinguishes an expected `AsyncGroupRejected` from an infrastructure failure. A rejected group publishes no native sample, increments a bounded diagnostic, and permits the next independent group to run. Consecutive rejection exhaustion remains worker-fatal, preventing an invalid environment stream from appearing healthy forever. Focused tests prove one rejected group does not poison its successor and two consecutive rejections fail at the configured bound.

Revision 16 async admission checkpoint: the existing native `VerifiersWorkerPool` now supports either one exact synchronous collection or one run-scoped async admission, never both. Run admission accepts independently identified collections and policy versions from the same training run while rejecting foreign-run work. Episode deadlines, wire cancellation acknowledgement, outcome identity checks, and native trace projection remain unchanged. Thirteen focused worker tests and all eight import contracts pass. The next slice can therefore build the concrete group source without weakening the synchronous collection fence.

Revision 17 concrete-producer checkpoint: the internal native-async path now has a real GRPO group source rather than a protocol fake. It starts the policy endpoint and worker pool in dependency order, opens run-scoped admission, selects prompts deterministically, issues sibling episodes concurrently, distinguishes recoverable group rejection from fatal runtime failure, cancels unfinished siblings, applies configured reward shaping, computes exact GRPO group advantages, and emits native samples carrying conservative behavior-policy start/end metrics. The protocol worker initializes only after producer startup succeeds and cleans partially acquired resources on failure. Twenty-three focused transport, lifecycle, and producer tests pass at this checkpoint. This is not trainer activation or recovery proof: prompt progress is still in-memory, and persisted native Verifiers episodes do not yet carry the orchestrator's final conservative policy span as replay metadata.

Revision 18 recovery checkpoint: pushed TRL candidate `3972dc39` calls an optional custom-worker hook with one group identity per sample admitted by the learner collator and saves/loads only the worker's JSON scheduling metadata alongside a trainer checkpoint. Posttrain records complete learner-consumed groups and terminally rejected groups, includes every scheduled group in a high-water cursor, and reconstructs a replay list for generated or queued groups that the learner never acknowledged. Selection order and episode seeds are stable functions of run identity, seed, group id, and example set. Because regenerating only part of a relative-reward group could change its advantages, checkpoint creation fails closed when acknowledgements show a partially consumed group; Milestone D activation must configure whole-group learner batches and prove this invariant in a real resume run. The complete Posttrain train suite passes with 395 tests and 5 skips; 19 focused fork tests pass, while the broader fork execution cases still require the absent compatible optional Flash Attention `kernels` runtime.

Revision 19 policy-provenance checkpoint: `backends/trl/async_policy_gateway.py::TrlAsyncPolicyGateway` is the missing run-scoped frontend between native Verifiers `TrainClient` workers and async TRL's trainer-owned vLLM server. It does not launch inference or interpret algorithms. It admits each token request under one live version, tracks versions by Verifiers' native trace session id, stops new turns and drains active requests during `prepare_model_update`, forwards activation only after the trainer publishes weights, and acknowledges upstream aborts during shutdown. `VerifiersWorkerPool` accepts an explicit episode-policy provider; `VerifiersEnvironmentRolloutBridge.project_native_episode` records the resolved span in `TrainWorkInfo.policy`, trace projection metadata, and `EnvironmentRollout.behavior_policy`. The producer no longer invents a span from dispatch/completion time and rejects missing or mismatched served evidence. Gateway, producer, worker, and bridge tests cover a two-turn session spanning version 0 to 1, upstream HTTP failure, the recoverable-group/run-fatal boundary, and missing policy evidence. The complete train suite passes with 400 tests and 5 skips. This is deterministic protocol evidence, not changed-weight GPU proof.

Revision 20 veRL prefix-resume checkpoint: fork commit `9694a6242e3590acaf58a779c1151b370f313b51` adds deterministic coverage around native `FullyAsyncLLMServerClient`, without replacing its scheduler or changing its runtime behavior. The controlled first request returns tokens 20–21 and is aborted under policy version 7; the resumed request receives prompt plus 20–21, requests only the remaining two tokens, returns 22–23 under version 8, and produces one ordered four-token/four-logprob result with span 7–8. Nine native continuation tests and three related global-step/weight-order tests pass. The existing real AutomationBench bridge test remains the evidence that a completed tool call appears once in episode execution. No GPU, changed-weight, pin, artifact, or supported-mode claim follows from this CPU gate.

Revision 21 deterministic failure checkpoint: TRL fork commit `d5b8cc4631c8f7adbe8296c08804205d6eb4c74c` adds a weight-publication failure regression alongside the existing success-order test. It proves `prepare` and inference pause occur before transfer, while a thrown transfer leaves `model_version` unchanged and calls neither inference resume nor rollout-worker publication. Three focused fork lifecycle tests pass. Posttrain adds explicit native health-starvation and unacknowledged-abort cases; the latter cancels the local request, releases the HTTP runner/client, and raises rather than treating uncertain cancellation as a rejected episode. Thirteen focused gateway/worker tests pass. These are deterministic gates only; real process/GPU failures remain in Milestone C.

Revision 22 changed-weight qualification checkpoint: fork commit `867885e5` provides the missing bounded probe but does not claim a pass. The first local implementation attempted `Worker.apply_model`; AsyncLLM's engine-core message boundary converted its callable dataclass to a plain dictionary, so the worker rejected it before any weight changed. The replacement uses the same typed native NCCL initialization/update requests as async TRL. NCCL then rejected the local one-device topology because trainer and inference are separate ranks. The corrected probe now requires distinct device indices before engine construction. Live dstack inventory reports idle RTX 3070 Ti and RTX PRO 6000 workers with one GPU each; a direct RunPod query for two on-demand GPUs returned no offers with or without a price ceiling. No optimizer update, parity pass, model pin, image, or public activation is claimed from these attempts.

Revision 23 external-server parity checkpoint: the production-shaped gate passed
with actor and server on distinct routable hosts. The actor used the framework
venv on the RTX 3070 Ti; the server used
`registry.carbonteq.com/carbonteq/posttrain-kind-online-rl-trl-py312@sha256:8230413ea572158e59e3f4099b218474d339869fb3eb1676ebaf23e35d35d03d`
on the RTX PRO 6000 with vLLM 0.25.1, Torch 2.11.0+cu130, and Transformers
5.14.1. For token id 387, actor/server log probabilities were
`-1.2540634871`/`-1.2563054562` before transfer and both
`-11.9312143326` afterward. The `0.0022419691` base delta is below the `0.05`
tolerance; exact post-transfer equality plus `10.6749088764` observed server
movement proves the update was applied rather than hidden by cache reuse.

Qualification also repaired three concrete client defects: HTTP control calls
now reject non-success responses, standalone clients no longer require an
initialized Accelerate logger, and a successful empty prefix-cache response is
accepted. Eight focused lifecycle/client tests pass. The slim image explicitly
disables FlashInfer's sampling kernel because it has no CUDA compiler. This
does not disable FlashInfer attention or NCCL, and vLLM's
`processed_logprobs` mode already selects the native sampler because FlashInfer
cannot return post-top-k/top-p log probabilities. This is a correctness-runtime
closure, not evidence about rollout-only sampler throughput. The dstack server
terminated cleanly and released the RTX PRO; no RunPod workload was submitted.
The live failure-boundary gate subsequently passed: the fork's real trainer
synchronization path received an injected failed `finish_weight_update`,
returned HTTP 500, kept version 0 authoritative, published no pending version,
and preserved selected-token log probability exactly (`0.0` delta before and
after). This single-rank control-path result does not qualify distributed NCCL
initialization failure propagation. Multi-rank failure propagation, optimizer
updates, checkpoint/resume, publication, and public activation remain
Milestone D.

Baseline checkpoint note (2026-09-08): the user requested commits preserving previous work. Framework changes are captured on `codex/pre-rollout-optimization-baseline`; historical veRL changes are separately preserved on `codex/verl-pre-rollout-optimization-baseline`. No fork pin is changed by these snapshots. Focused framework reward-admission, reward-advantage, and policy-message tests passed (32 tests); full release/GPU qualification is not implied. The two cleanup stashes remain separate and untouched.

Revision 24 development-publication checkpoint: the maintained fork closure is
now reproducible outside local worktrees. TRL `1.12.0.post6` and veRL
`0.9.0.post2` were published byte-for-byte to `carbonteq/dev`; Trackio remains
selected at `0.31.5.post14.dev20`; Verifiers was consumed from immutable pushed
commit `36eac9d5e04ef29b584b6fa4f027af00cd76ea19`. A system-CA-trusting local
BuildKit publication produced all seven v0.4 runtime images and recorded their
immutable digests in `packages/runtime-images/src/posttrain/runtime_images/published.toml`.
The strict release check passes and the runtime-image plus release regression
slice passes with 126 tests and one existing skip. This closes package and OCI
development publication, but not the remaining real GPU training qualification.

Revision 25 environment-closure checkpoint: pushed environment commit
`d994073b9632e73c96a57865683133d7a6ebc4bf` removes mixed upstream/fork
Verifiers URLs from all six standalone packages. Their individual lock, Ruff,
format, Pyright, test, and wheel gates pass (59 tests, two data-dependent
skips), and all six wheels install and activate together against Verifiers
`36eac9d5`. Posttrain catalogs, starter generation, the AutomationBench ledger,
and CI now select this one repository revision. This repairs the external
consumer and optional-Verifiers dependency closure exposed by CI run
`34337365149`; a new exact-source CI run must still prove the repair remotely.

Revision 26 native-tool correctness checkpoint: the five-step LFM 1.2B run
`lfm12-continuous-batching-five-step-local-20260909-r4` proved stable async
collection and optimizer lifecycle, but its retained native episodes exposed a
correctness failure: task-specific AutomationBench tool allowlists were lost
when `EnvClient` crossed into a worker, so the model received no tool schemas
and every reward was zero. Independently maintained Verifiers commits
`5304495a246e174683f5932377703e9a0a4a6926` and
`1f6793f7d46e8a650a54b2a585193b4010578fa6` make task configuration part of
the validated native wire, safely parse LFM2 structured calls without executing
sampled text, and extend tool cycles from the exact sampled token prefix even
when vLLM strips the stop token. Posttrain now selects that parser through its
versioned LFM renderer contract and forwards each derived task config. A real
spawned AutomationBench worker deterministically executes
`salesforce_note_create`, receives reward `1.0`, continues for a second model
turn on one linear token branch, and retains the native trace. The complete
Posttrain train suite passes with 411 tests and 5 skips; Ruff, all eight import
contracts, and the fork's focused tests pass. A changed-weight GPU run using an
actual job capsule remains the next gate; the RTX PRO is currently occupied by
the pre-existing 20-step GRPO comparison and must not be oversubscribed.
The environment package closure is published at
`c7e88d7b302e6177041ac0849863d0466be77d8b`; its six package suites pass with
59 tests and two data-dependent skips against Verifiers `1f6793f7`.

Revision 27 runtime-cache checkpoint: local runtime recovery previously omitted
veRL's backend identity and reproducible cache-lineage variables, so the shipped
Dockerfile could not be rebuilt through the ordinary project CLI. That request
now carries the immutable backend revision, dependency-lock digest, repository,
version, source revision, and source-date epoch, with a focused regression test.
The veRL image already shares 16 exact-version heavyweight distributions from
the parent control environment, including Torch, Triton, cuBLAS, cuDNN, NCCL,
and CUDA runtime libraries. Backend-only JIT/compiler packages remain isolated.
Explicit vLLM, CUTLASS, and DLPack wheels now come from the trusted LAN mirror,
are retained in a SHA-256-addressed BuildKit cache, are atomically populated and
verified before use, and are exposed to both the initial uv sync and ordered
repair install under valid wheel filenames. The real dependency layer improved
from approximately 544 seconds to 43 seconds; an identical rebuild returned the
same digest in 0.60 seconds. Candidate image
`sha256:d33c0e40ed3d9d01667cd06c20989014b0b87d8bf18959f05e085063241cdbd6`
imports Torch `2.11.0+cu130`, Verifiers `0.3.2.dev79`, and vLLM
`0.25.2.dev2+g7817d8457.precompiled`. This is local development publication
evidence; the clean-checkout release manifest and changed-weight GPU canary are
still open.
