# Save the last completed update when a training run is cancelled

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document must be maintained in accordance with
`docs/templates/PLAN.md`.

## Purpose / Big Picture

Today `posttrain run cancel <run>` stops a TRL online-RL training run without
saving anything new, so every optimizer update since the last periodic
checkpoint is lost. In the motivating case, run
`lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1` (job kind `train.sampo`,
TRL backend, LoRA rank 4, `checkpoint_steps: 10`, `checkpoint_limit: 3`) was
cancelled while update 44 was running. Updates 41, 42, and 43 were complete in
memory but never written; the newest saved state is update 40.

After this change, a cancelled GRPO, DAPO, OLMo 3, SAMPO, GDPO, or CAPO run on the
TRL backend first saves and publishes a checkpoint of its last completed update,
when that update is newer than the last periodic checkpoint, and only then
finalizes as `cancelled`. The checkpoint has exactly the same form as a periodic
one: a `training-checkpoint` recovery view with adapter, optimizer, scheduler,
random-number-generator (RNG), trainer, and adaptive-curriculum state, and a
`model-adapter` model view, both with content digests. A new run can continue
from it with `posttrain job run ... --resume-from-run <run> --checkpoint-step <N>`
or branch from its weights with `--model-from-run <run> --model-checkpoint-step <N>`.
`posttrain run checkpoint list <run>` shows the new step, and
`posttrain run checkpoint verify <run> --step <N>` passes. The run's events
include one `cancel_checkpoint` event that says "saved at update N" or names the
reason nothing was saved.

The change does not alter the frozen product baseline in
`docs/post-training/01` through `06`: cancellation still yields the `cancelled`
run status, checkpoints still use the two existing views and the existing
`interrupted` metadata flag, and `05-apis.md` already lets
`--resume-from-run` and `--model-from-run` select any committed checkpoint step.
The `cancel_checkpoint` event and the `train/cancel_checkpoint_step` metric are
additional observations, and the baseline's event and metric tables are minimum
sets rather than closed lists.

## Progress

- [x] (2026-09-28) Read `AGENTS.md`, the plan template, the canonical documents
  that govern checkpoints and cancellation (`05-apis.md`,
  `06-observation-and-lineage.md`), and the current runtime, tracking, and TRL
  code paths.
- [x] (2026-09-28) Inspected the pinned `transformers==5.14.1` update loop and the
  `trl==1.12.0.post10` fork's `GRPOTrainer` and `VLLMGeneration` to locate every
  point where training state changes.
- [x] (2026-09-28) Milestone 1: process-wide cancellation gate in
  `posttrain.common` and runtime signal routing through it, with unit and runtime
  tests.
- [x] (2026-09-28) Milestone 2: TRL update boundary, cancellation checkpoint save
  and publication, wiring into the online-RL adapter, fake-loop tests, and a CPU
  test through the real pinned `transformers.Trainer`.
- [x] (2026-09-28) Validation ladder on the worktree (see `Outcomes`).
- [ ] Manual GPU qualification on a real SAMPO run (procedure in `Validation and
  Acceptance`); record the run id and outcome here.
- [ ] Optional follow-up: apply the same boundary to the TRL SFT, DPO, and
  on-policy distillation adapters.

## Surprises & Discoveries

- Observation: the existing cancel path already republishes the newest periodic
  checkpoint with `interrupted: true`, and the checkpoint resolvers already
  prefer the periodic view when a step is published twice.
  Evidence: `publish_interrupted_recovery_checkpoint` in
  `packages/train/src/posttrain/train/backends/trl/common.py`,
  `_committed_over_interrupted` in `packages/train/src/posttrain/train/checkpoints.py`,
  and the matching filter in `apps/cli/src/posttrain_cli/commands/work_package.py`.
  A single cancellation checkpoint marked `interrupted: true` is therefore
  selectable without resolver changes.
- Observation: that republication rewrote the adaptive-curriculum snapshot inside
  the periodic checkpoint directory with the live controller state, which the
  interrupted update's rollout had already advanced. Same-run recovery from the
  retained workspace would then restore a curriculum ahead of the weights.
  Evidence: `checkpoint_state_writer(checkpoint)` runs unconditionally before
  republication; the regression test
  `test_fallback_republication_keeps_the_saved_controller_snapshot` shows the
  file staying at the checkpoint's own step after the fix.
- Observation: `checkpoint_limit` defaults to 1, and transformers rotates old
  checkpoints inside `_save_checkpoint` immediately after writing the new one.
  A cancellation save through the normal path would delete the newest periodic
  checkpoint before the new one is published.
  Evidence: `rotate_checkpoints(output_dir=run_dir, save_total_limit=...)` at the
  end of `Trainer._save_checkpoint` in transformers 5.14.1.
- Observation: TRL's adaptive entropy controller (`entropy_coef`) and FLOP
  counters advance during the next update's final micro-batch, before its
  optimizer step, and the trainer writes them into `trainer_state.json` and
  `entropy_ctrl_state.json`. RNG state is consumed by rollout sampling.
  Evidence: `GRPOTrainer._compute_loss` in the TRL fork updates `entropy_coef`
  when `accelerator.sync_gradients` is true; `Trainer.store_flos` folds
  `current_flos` into `state.total_flos` during a save.
- Observation: the local Docker provider stops a container with `docker stop
  --time 10`, so a local cancel has ten seconds before SIGKILL; dstack uses the
  maintained fork's bounded `stop_duration` (300-second compatibility fallback).
  Evidence: `packages/execution-local/src/posttrain_execution_local/adapter.py`
  and `docs/tooling/dstack/README.md`.
- Observation: the default `uv sync --all-packages --locked` environment lacks the
  `trl` extra, so four TRL test modules fail to import; CI installs
  `--extra trl`. Evidence: `.github/workflows/quality.yml`.

## Decision Log

- Decision: route every host cancellation signal through a small process-wide
  gate, `posttrain.common.interruption.HostCancellation`, and let reusable
  packages mark atomic updates as critical sections on it.
  Rationale: the signal handler belongs to the host (`apps/runtime`) and the
  atomic update belongs to the trainer (`packages/train`). The import contracts
  in `pyproject.toml` allow both to import `posttrain.common` but forbid runtime
  knowledge from leaking into training and training from depending on the host.
  The gate has no framework imports, so the "common contracts do not import
  execution frameworks" contract still holds. The existing cooperative
  `CancellationToken` was not reused because cancelling it makes every
  `RunContext` observation raise `OperationCancelled`, which would block the
  checkpoint publication this change needs.
  Date/Author: 2026-09-28 / Claude

- Decision (safety argument): a checkpoint is only written from a state that
  equals the end of a completed update, and the design proves that as follows.
  In the pinned transformers loop, one update runs `training_step` over the
  gradient-accumulation micro-batches (TRL generates the rollout inside the
  first one), clips gradients, calls `on_pre_optimizer_step`,
  `optimizer.step()`, `on_optimizer_step`, `lr_scheduler.step()`,
  `model.zero_grad()`, `state.global_step += 1`, `on_step_end`, and
  `_maybe_log_save_evaluate` (logging and the periodic save). Trainable
  parameters and optimizer moments change only inside `optimizer.step()`; the
  scheduler and step counter change after it. The adapter opens a critical
  section at `on_pre_optimizer_step` and closes it after `_maybe_log_save_evaluate`
  returns, so a signal that lands anywhere from the optimizer step to the end of
  the periodic save is delivered only when the update is complete and its
  periodic checkpoint, if any, is written and published. A signal that lands
  anywhere else (rollout, forward, backward) finds parameters, optimizer, and
  scheduler exactly at update `global_step`; the only extra state is partially
  accumulated gradients, which a checkpoint never contains. The remaining state
  that the next update advances before its optimizer step (RNG streams,
  `current_flos`, `total_flos`, `num_input_tokens_seen`, `entropy_coef`,
  `_last_world_entropy`, and the adaptive-curriculum controller) is copied at the
  end of every update and restored immediately before the cancellation save, so
  the files match what a periodic save at that step would have written.
  If an exception or a forced exit leaves the critical section open, the
  boundary is marked torn and nothing is written.
  Date/Author: 2026-09-28 / Claude

- Decision: bound the deferral at 60 seconds by default. A watchdog thread
  started when the host arms the gate re-sends the signal to the main thread
  with `signal.pthread_kill` once the bound expires, and the exit is then
  delivered wherever it lands, with `HostCancellation.forced` set.
  Rationale: a hung critical section (for example a colocated vLLM wake-up
  inside weight synchronization) must not turn a cancellation into a provider
  SIGKILL without tracking finalization. A real signal, unlike
  `_thread.interrupt_main`, also interrupts a blocking system call. The watchdog
  is started at arm time so a signal handler never starts a thread or takes a
  lock that the interrupted code might hold.
  Date/Author: 2026-09-28 / Claude

- Decision: write the cancellation checkpoint through the trainer's own
  `_save_checkpoint`, with `_get_output_dir` temporarily pointed at
  `<output_dir>/.cancel-checkpoint`, and rename `checkpoint-<N>` into the output
  directory only after the same completeness rule used by same-run recovery
  passes (`trainer_state.json` with the expected `global_step`, `optimizer.pt`,
  `scheduler.pt`, `rng_state*.pth`).
  Rationale: the trainer's method (including TRL's model-card and adaptive
  entropy extensions) produces the exact periodic form. The staging directory
  keeps transformers' retention from deleting the newest periodic checkpoint
  before the new one is committed, and a timed-out or failed save can never be
  mistaken for a checkpoint, because neither `get_last_checkpoint` nor
  `_prepare_interruption_recovery` looks inside a hidden directory. The
  cancellation checkpoint is not followed by a rotation: a cancelled run may
  keep `checkpoint_limit + 1` local checkpoints until workspace cleanup.
  Date/Author: 2026-09-28 / Claude

- Decision: run that save on a daemon worker thread and wait at most 60 seconds;
  on timeout, failure, or incomplete output, record a `cancel_checkpoint` event
  with the reason and fall back to the existing republication of the newest
  periodic checkpoint.
  Rationale: a LoRA rank-4 checkpoint (adapter plus AdamW moments) is tens of
  megabytes and saves in a few seconds; the bound keeps a slow disk or a
  full-parameter model from hanging cancellation. Publication itself stays on the
  main thread because tracking observers are not guaranteed to be thread-safe,
  and it is already bounded by the tracking adapter's publication timeout.
  Date/Author: 2026-09-28 / Claude

- Decision: mark the cancellation checkpoint `interrupted: true` and add no new
  metadata to its artifacts; record the trigger in a separate
  `cancel_checkpoint` event and a `train/cancel_checkpoint_step` metric.
  Rationale: keeps the artifact form identical to existing interrupted
  publications that resolvers already understand.
  Date/Author: 2026-09-28 / Claude

- Decision: also make the rollout engine's `sync_weights` a critical section.
  Rationale: weight synchronization does not change the trainable adapter (PEFT
  merges into, and in a `finally` unmerges from, the frozen base), so it is not
  needed for checkpoint correctness; deferring it avoids leaving a colocated
  engine half-woken while the checkpoint is written, and the 60-second bound
  still applies.
  Date/Author: 2026-09-28 / Claude

- Decision: scope the change to the shared TRL online-RL adapter
  (`_run_online_rl`, used by GRPO, DAPO, OLMo 3, SAMPO, GDPO, and CAPO). Leave
  TRL SFT, DPO, and on-policy distillation on the existing republication path,
  and leave veRL unchanged.
  Rationale: the three other TRL adapters each build their own trainer subclass
  and except path; they can adopt the same two helpers later with the same
  tests. veRL runs its trainer in a separate process tree launched by
  `packages/train/src/posttrain/train/backends/verl/launcher.py`, which receives
  SIGTERM from the launcher rather than the gate; it needs its own boundary in
  the veRL candidate fork and is out of scope.
  Date/Author: 2026-09-28 / Claude

## Outcomes & Retrospective

Implemented on branch `codex/cancel-checkpoint` in
`/home/hammad/projects/rl-cancel-checkpoint`. The gate, runtime routing, TRL
boundary, cancellation save, and fallback fix are covered by 10 gate tests, one
new runtime test, 10 fake-loop tests, and 2 tests through the real pinned
`transformers.Trainer` on CPU with a PEFT LoRA adapter, which also resume a new
trainer from the saved checkpoint. What remains is the real GPU qualification
below; it is the evidence that the colocated vLLM rollout, the SAMPO rollout
function, and Trackio publication behave the same on hardware. The ten-second
local Docker stop window is the main operational risk; it predates this change
but now also has to cover the cancellation save.

## Context and Orientation

A job runs inside `posttrain-runtime`
(`apps/runtime/src/posttrain_runtime/execute.py`). `_graceful_cancellation()`
installs SIGTERM and SIGINT handlers (Docker stops with SIGTERM; the dstack
runner interrupts with SIGINT) that raise `SystemExit(128 + signal)` on the main
thread. The exception unwinds through the training code into
`packages/work/src/posttrain/work/execution.py`, whose `execute_run` flushes
queued artifacts (`finalize_failure`) and whose tracked wrapper records the run
outcome `cancelled`. The provider may SIGKILL after its stop timeout.

The TRL online-RL adapter is
`packages/train/src/posttrain/train/backends/trl/policy_optimization.py`
(`_run_online_rl`). It composes a TRL `GRPOTrainer` subclass with telemetry and
adaptive-curriculum mixins and callbacks, calls
`trainer.train(resume_from_checkpoint=...)`, and on any exception publishes a
recovery checkpoint before re-raising. Periodic checkpoints are written by
transformers into `<workspace>/.../checkpoint-<step>` every `checkpoint_steps`
updates and published by `CheckpointPublicationCallback`
(`packages/train/src/posttrain/train/backends/trl/common.py`) as a
`training-checkpoint` recovery view and, for LoRA, a `model-adapter` model view
through `publish_checkpoint_views`. The adaptive curriculum
(`packages/train/src/posttrain/train/backends/trl/policy_curriculum.py`) writes
its controller snapshot into each checkpoint.

Terms used here: an "update" is one optimizer step, numbered by
`trainer.state.global_step`. A "critical section" is a span of code during which
the host defers delivering a cancellation. The "boundary" is the moment an
update is complete (after its periodic save, if any).

## Plan of Work

Milestone 1 adds `packages/common/src/posttrain/common/interruption.py` with
`HostCancellation` (a process-wide gate returned by `host_cancellation()`),
`CriticalSection`, and `CancellationRequest`, exported from `posttrain.common`.
`HostCancellation.arm()` starts honoring sections and a watchdog; `request(signal)`
raises `SystemExit(128 + signal)` immediately outside a section, records a
pending exit inside one, and ignores repeats; the outermost section's `close()`
delivers the pending exit; `close(deliver=False)` releases a section on an error
path. `_graceful_cancellation()` arms the gate, routes both signals through
`request`, and disarms it after restoring the previous handlers.

Milestone 2 adds `packages/train/src/posttrain/train/backends/trl/cancellation.py`.
`UpdateBoundary` opens the `optimizer_update` section from
`update_boundary_callback_type` (`on_pre_optimizer_step`) and closes it from
`update_boundary_trainer_type` (after `_maybe_log_save_evaluate`), capturing an
`UpdateBoundarySnapshot` at each close. The same trainer mixin wraps
`vllm_generation.sync_weights` in a `rollout_weight_sync` section.
`retain_checkpoint_after_interruption` replaces the direct call to
`preserve_recovery_checkpoint_after_error` in `_run_online_rl`: for a host
cancellation it calls `save_cancellation_checkpoint`, and otherwise, or when
nothing new was saved, it keeps the existing republication, now without
rewriting a controller snapshot the periodic save already wrote.
`CheckpointPublisher` in `trl/common.py` now owns the publication that the
periodic callback performed so the periodic and cancellation saves share it.
`AdaptiveCurriculumRuntime` gains `capture_state()` and accepts an explicit
state in `checkpoint(checkpoint, state)`.

## Concrete Steps

All commands run from `/home/hammad/projects/rl-cancel-checkpoint`.

    uv sync --all-packages --group dev --extra trl --python 3.13 --locked
    uv run --no-sync pytest -q packages/common/tests/test_interruption.py
    uv run --no-sync pytest -q packages/train/tests/test_trl_cancellation.py
    CUDA_VISIBLE_DEVICES= uv run --no-sync pytest -q packages/train/tests/test_trl_cancellation_transformers.py
    uv run --no-sync pytest -q apps/runtime/tests/test_execute.py -k cancel

Expected: 10, 10, 2, and 3 passed respectively. The transformers test skips
when `torch`, `transformers`, or `peft` is not installed.

## Validation and Acceptance

Automated acceptance is the validation ladder from `AGENTS.md`:

    uv run ruff check .
    uv run ruff format --check packages apps
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Manual GPU qualification (required before calling the feature complete; it uses
one local GPU job and must run from a clean worktree of the committed branch,
because Posttrain packs the working tree). The existing three-update SAMPO work
package saves its first periodic checkpoint only after update 3, so cancelling
during update 3 proves a checkpoint that could not exist before this change.

1. From a clean worktree of the committed branch, in `apps/lab`, launch:

       uv run --no-sync --package posttrain posttrain job run \
         .posttrain/work_packages/lfm12_automationbench_sampo_local_8gb.yaml \
         --provider local --run-id sampo-cancel-ckpt-<yyyymmdd>-r1

2. Follow the logs with `posttrain run logs sampo-cancel-ckpt-<yyyymmdd>-r1 -f`
   (or the Trackio run page). When the step-2 training metrics appear (update 2
   finished, update 3's rollout has started), cancel:

       uv run --no-sync --package posttrain posttrain run cancel sampo-cancel-ckpt-<yyyymmdd>-r1

   Before this change the run would publish no checkpoint; the log would show
   `recovery_checkpoint_unavailable`.
3. After the run is terminal, confirm its status is `cancelled` with
   `posttrain run status <run>`, and that the worker log contains
   `cancel checkpoint saved at update 2`.
4. Confirm the checkpoint views and digests:

       uv run --no-sync --package posttrain posttrain run checkpoint list sampo-cancel-ckpt-<yyyymmdd>-r1
       uv run --no-sync --package posttrain posttrain run checkpoint verify sampo-cancel-ckpt-<yyyymmdd>-r1 --step 2

   Expect `step=2  state=ready  recovery=yes  model=yes` and `"state": "verified"`.
   Add `--deep` when the tracking provider supports blob verification.
5. Confirm the run's `cancel_checkpoint` event has `outcome: saved`,
   `global_step: 2`, `previous_checkpoint_step: null`, and that
   `train/cancel_checkpoint_step` is 2.
6. Prove the checkpoint is usable by resuming it (the work package's
   `max_steps: 3` leaves exactly one more update):

       uv run --no-sync --package posttrain posttrain job run \
         .posttrain/work_packages/lfm12_automationbench_sampo_local_8gb.yaml \
         --provider local --resume-from-run sampo-cancel-ckpt-<yyyymmdd>-r1 --checkpoint-step 2

   Expect the new run to start from `global_step` 2, run only update 3, and
   succeed with a training summary whose `global_step` is 3. A LoRA branch can
   instead use `--model-from-run sampo-cancel-ckpt-<yyyymmdd>-r1
   --model-checkpoint-step 2` on an evaluation or training job.
7. Repeat once cancelling during update 3's actor phase (after the rollout
   completes and before step-3 metrics appear), and once on dstack, whose
   runner sends SIGINT, to exercise both signals.

The local Docker provider stops a container with `docker stop --time 10`: the
deferred exit, the cancellation save, and tracking finalization must all fit in
ten seconds before Docker sends SIGKILL. A rank-4 LoRA checkpoint fits, but if
step 3 shows a missing `cancelled` status or a `cancel_checkpoint` event without
the matching artifacts, raise that stop timeout (a separate change in
`packages/execution-local`) before concluding the feature failed.

Record each run id, the observed `cancel_checkpoint` event, and the
`checkpoint verify` output under `Artifacts and Notes`.

## Idempotence and Recovery

All code changes are additive and can be reapplied. A cancellation save that
times out leaves `<trainer output>/.cancel-checkpoint/`; it is removed by the
next cancellation save and is invisible to checkpoint discovery. If a real run
shows `cancel_checkpoint` with `outcome: failed`, the existing republication of
the newest periodic checkpoint still runs, so behavior is never worse than
before. Reverting is a matter of restoring the direct
`preserve_recovery_checkpoint_after_error` call in `_run_online_rl`.

## Artifacts and Notes

Fake-loop evidence (`test_trl_cancellation.py`): cancelling in update 44's
rollout or actor phase saves `checkpoint-43` beside `checkpoint-40` with
`checkpoint_limit: 1`; cancelling inside update 44's optimizer step yields a
deferred exit and `checkpoint-44`; a forced exit inside the optimizer step
writes nothing and records `cancelled_inside_optimizer_update`; a 2-second save
with a 0.2-second bound records `timed_out` in under 2 seconds and leaves only
`checkpoint-40` discoverable.

Real transformers evidence (`test_trl_cancellation_transformers.py`): the saved
adapter tensors equal the live PEFT adapter, the periodic checkpoint survives,
and `Trainer.train(resume_from_checkpoint=...)` continues from the saved step.

## Interfaces and Dependencies

In `posttrain.common` (`packages/common/src/posttrain/common/interruption.py`):

    class HostCancellation:
        def arm(self, *, max_deferral_seconds: float = 60.0) -> None
        def disarm(self) -> None
        def critical(self, name: str) -> CriticalSection
        def request(self, signal_number: int) -> None
        requested: CancellationRequest | None
        pending: bool
        forced: bool
    def host_cancellation() -> HostCancellation

In `posttrain.train.backends.trl.cancellation`:

    class UpdateBoundary:
        def __init__(self, *, gate=None, controller_state=None) -> None
        def begin(self) -> None
        def commit(self, trainer) -> None
        def abandon(self) -> None
    def update_boundary_callback_type(imports, boundary) -> type
    def update_boundary_trainer_type(parent, boundary) -> type
    def retain_checkpoint_after_interruption(context, trainer, error, *, boundary,
        publisher, imports, controller_state_writer=None, timeout_seconds=60.0) -> None
    def save_cancellation_checkpoint(...) -> bool

In `posttrain.train.backends.trl.common`: `CheckpointPublisher.publish(checkpoint,
*, step, interrupted=False, checkpoint_state_writer=None)`, and
`checkpoint_callback_type(..., publisher=None)`.

No new third-party dependencies. The code relies on transformers'
`on_pre_optimizer_step` callback event, `Trainer._maybe_log_save_evaluate`,
`Trainer._save_checkpoint`, and `Trainer._get_output_dir` as they exist in the
pinned 5.14.1; a transformers upgrade must rerun
`test_trl_cancellation_transformers.py`.
