# Make TRL GRPO actor updates observable as real optimizer stages

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

An Observatory reader must be able to distinguish rollout generation from the actor optimization that follows it. Today the TRL adapter opens an `actor_update` runtime phase around the entire `trainer.train()` call, so every rollout and every later step inflates one long actor interval. After this change, each optimizer update has its own bounded `actor_update` phase and a `train/rl/time/actor_update_seconds` metric. Rollout remains a separate phase. This makes phase duration and GPU evidence trustworthy without changing DAPO or GRPO training behavior.

## Progress

- [x] (2026-08-08 00:48Z) Confirmed from a live two-step RTX 4090 run that `actor_update` begins before the first rollout and remains open across the training session.
- [x] (2026-08-08 00:48Z) Confirmed the frozen observation baseline already assigns online-RL step metrics and runtime phases to the train adapter; no baseline amendment is required.
- [x] (2026-08-08 00:53Z) Replaced the session-wide actor phase with a lifecycle helper that starts after each rollout and ends at the optimizer-step callback.
- [x] (2026-08-08 00:53Z) Emit one canonical `train/rl/time/actor_update_seconds` duration per completed optimizer step.
- [x] (2026-08-08 00:54Z) Added focused regression tests for phase ordering, duration, callback completion, overlapping-start rejection, and failure cleanup.
- [x] (2026-08-08 01:10Z) Added actor-only processed-token and token-throughput evidence using the completed actor duration and TRL's cumulative token counter.
- [x] (2026-08-08 01:24Z) Passed 63 focused GRPO/API tests, Ruff, Pyright, all eight import-boundary contracts, and `git diff --check`.
- [x] (2026-08-08 01:19Z) Completed the two-update RTX 4090 diagnostic and retained all 256 traces plus both optimizer records in Doris.
- [x] (2026-08-08 01:24Z) Reconciled paired rollout behavior, optimizer metrics, actor/rollout system intervals, and the terminal artifact-publication failure.
- [x] (2026-08-08 02:24Z) Completed a matched two-update diagnostic from the retained 1,938-update SFT LoRA and retained 256 traces, the final adapter, the step-two recovery checkpoint, and the training summary in Trackio.
- [x] (2026-08-08 02:24Z) Diagnosed a Qwen3.5 LoRA namespace mismatch between the Transformers actor and the colocated vLLM sampler; classified the SFT-start run as bridge-failure evidence rather than DAPO learning evidence.

## Surprises & Discoveries

- Observation: `packages/train/src/posttrain/train/backends/trl/grpo.py` currently wraps `trainer.train()` in `context.phase("actor_update")` while `_rollout_function` opens nested rollout phases.
  Evidence: the live diagnostic run emitted `actor_update` and then `rollout` at the same timestamp; no actor optimizer work had occurred yet.

- Observation: the canonical metric catalog already contains `train/rl/time/actor_update_seconds`, but the TRL adapter does not emit it. The equivalent veRL native timing is already normalized.
  Evidence: `_CANONICAL_PASSTHROUGH` and `_VERL_METRICS` in `packages/train/src/posttrain/train/grpo_observations.py` contain the canonical name.

- Observation: the first diagnostic rollout produced usable policy-gradient signal without dynamic sampling: 24 of 32 groups had reward variance and the reconstructed advantages included both positive and negative values.
  Evidence: step 0 retained 128 traces; zero-variance groups were 25%, with 24.2% positive, 50.8% negative, and 25% zero reconstructed advantages.

- Observation: the Liger GRPO loss path records only aggregate clip ratio (and KL when enabled); unlike the Torch path, it does not record entropy or asymmetric lower/upper clip fractions.
  Evidence: the first live optimizer record has loss and gradient norm but no entropy, and the pinned TRL `compute_liger_loss` extracts only `metrics[-1]` as `clip_ratio`.

- Observation: the first update's importance-sampling ratio reached the configured 0.1 and 3.0 bounds, but TRL records only min, mean, and max after truncation.
  Evidence: the live optimizer record reports min 0.1, mean 0.8631, and max 3.0; the pinned TRL computes its summaries after `torch.clamp` and does not publish the fraction outside each bound.

- Observation: actor optimization is compute-heavy and rollout is capacity-heavy; the old session-wide actor phase inverted that distinction in Observatory.
  Evidence: actor intervals averaged 94.9% and 94.3% GPU activity at about 10.5 and 10.2 GiB, while rollout intervals averaged 64.0% and 64.8% GPU activity at about 20.5 and 21.2 GiB.

- Observation: the diagnostic inherited the default linear learning-rate scheduler, so update two used 5e-6 instead of the intended constant 1e-5.
  Evidence: the optimizer records report 1e-5 and 5e-6. The 4090 production profile now explicitly selects 20 steps of warmup followed by a constant rate; the diagnostic explicitly selects constant.

- Observation: training completed both updates, but post-training Verifiers trace publication failed because a post8 Trackio client used the legacy resumable endpoint against the post10 S3-backed server. The legacy route wrote the verified blob only to local CAS while manifest validation checked RustFS.
  Evidence: the missing digest was present in server-local CAS at 11,952,582 bytes and absent from S3. It was copied and SHA-256-verified in S3; the fork compatibility repair now routes legacy completion through the configured store.

- Observation: the SFT-start sampler and actor were already far apart before the first optimizer update, unlike the matched base-policy diagnostic.
  Evidence: the first SFT-start update reported mean/max absolute sampling log-probability deltas of 0.253/19.47 and an importance-sampling ratio mean of 0.128 at the configured 0.1 floor. The matched base-policy update reported 0.014/0.57 and 0.863. The second SFT-start update remained mismatched at 0.250/18.27 and 0.135.

- Observation: native PEFT LoRA names do not identify the module namespace used by vLLM's composite Qwen3.5 implementation.
  Evidence: the actor and TRL temporary export contain `base_model.model.model.layers...`; the retained serving-compatible adapter contains `base_model.model.language_model.model.layers...`. vLLM strips `base_model.model.` before applying the Qwen3-VL mapper, so the native key becomes `model.layers...` rather than the rollout module's `language_model.model.layers...` name. TRL's LoRA synchronization mode currently rejects the existing weight-name-prefix option.

- Observation: the sampled post-update population regressed while the bridge mismatch persisted.
  Evidence: paired across the same 32 prompt groups, raw environment reward changed by -0.104 with a 95% bootstrap interval of [-0.188, -0.022], triple F1 changed by -0.056 [-0.102, -0.012], submit attempts fell from 28.9% to 18.8%, and zero-variance groups rose from 21.9% to 37.5%.

## Decision Log

- Decision: Correct the phase at the TRL adapter boundary rather than teaching Observatory to subtract nested intervals heuristically.
  Rationale: the emitter knows exactly when rollout returns and when the optimizer step completes. A read product should not reinterpret an inaccurately named source interval.
  Date/Author: 2026-08-08 / Codex

- Decision: Start the actor phase only after the environment rollout, reward calculation, trace emission, and rollout metrics complete; close it in the trainer's step-end callback.
  Rationale: this interval contains actor forward/backward, gradient accumulation, clipping, and optimizer work while excluding generation and environment latency.
  Date/Author: 2026-08-08 / Codex

- Decision: Keep parameter-delta proof outside the per-step hot path by comparing recovery checkpoints for the diagnostic run.
  Rationale: copying all trainable parameters every update is cheap for some LoRA runs but unsafe for generic full fine-tuning. Checkpoint evidence proves movement without adding production overhead.
  Date/Author: 2026-08-08 / Codex

- Decision: Derive actor token throughput from TRL's cumulative processed-token counter and the newly bounded actor duration.
  Rationale: the existing generic token throughput spans rollout plus optimization and therefore cannot answer whether the actor update itself is efficient. The delta adds no model forward pass or parameter copy.
  Date/Author: 2026-08-08 / Codex

- Decision: Do not interpret the completed SFT-start run as evidence that DAPO improves or harms the SFT policy.
  Rationale: an on-policy conclusion requires the rollout sampler and actor to represent the same initial policy. The first-update parity failure predates any optimizer movement and the importance correction remained pinned near its lower bound. Correct the generic TRL LoRA synchronization path and require a pre-update actor-versus-sampler parity gate before repeating the comparison.
  Date/Author: 2026-08-08 / Codex

## Outcomes & Retrospective

The source-level telemetry correction and validation ladder are complete. The live run proved that the first optimizer update changed LoRA weights and that the following rollout reduced truncation from 11.7% to 3.9%. Paired raw task reward improved by only 0.040 with a 95% bootstrap interval crossing zero, while shaped reward improved by 0.116 because the overlong penalty fell. This is evidence of a length-control response, not yet evidence that knowledge-graph extraction learned. The second update completed, but final artifact publication failed after training and provider cleanup removed the local checkpoint, so the run is diagnostic evidence rather than a resumable model artifact.

The matched SFT-start diagnostic later completed through finalization as `train.grpo-0b8ad49f` and retained all expected artifacts, including the global-step-2 recovery checkpoint. Its first sampled population appeared better than the base run on raw reward and save rate, but the actor-to-sampler parity evidence invalidates that attribution: the SFT actor's native LoRA namespace was not reconciled with vLLM's Qwen3.5 language-model namespace. After one update, raw reward, triple F1, and save behavior regressed while the importance ratio remained near the 0.1 floor. The durable outcome is therefore a reproducible TRL bridge defect and a concrete parity gate, not a DAPO model-quality result.

## Context and Orientation

`packages/train/src/posttrain/train/backends/trl/grpo.py` translates a backend-neutral `GRPORequest` into Hugging Face TRL. Its `_rollout_function` invokes the selected Verifiers environment and already emits a bounded `rollout` phase plus rollout duration. `packages/train/src/posttrain/train/backends/trl/common.py` creates a Transformers callback that receives optimizer lifecycle hooks and forwards normalized step metrics through `RunContext`. `RunContext.phase` in `packages/common/src/posttrain/common/execution.py` records phase start, completion, and failure events. Observatory later correlates those intervals with system samples.

An actor update is the policy optimization after a rollout population has been scored: actor log-probability calculation, loss, backward passes over accumulated microbatches, gradient clipping, and the optimizer step. It does not include rollout generation or environment execution. A logical rollout step is zero-based in current traces, while the completed optimizer step reported by Transformers is one-based; actor telemetry must use the completed optimizer-step number.

The work corrects evidence semantics only. It does not change the frozen product baseline, training settings, reward, DAPO loss, batching, or checkpoint policy.

## Plan of Work

Add a small private actor-update lifecycle helper to `packages/train/src/posttrain/train/backends/trl/grpo.py`. The helper owns at most one active `RunContext.phase` context manager, records a monotonic start time, and can complete or fail the interval exactly once. Starting a second interval while one is active is an error because it means the trainer lifecycle no longer matches the adapter's assumptions.

Construct that helper before the TRL trainer. Pass it into `_rollout_function`. At the very end of a successful rollout call, start actor update number `trainer.state.global_step + 1`. Add a TRL callback whose `on_step_end` completes the matching interval and emits `train/rl/time/actor_update_seconds` at the completed global step. If training raises between rollout return and step end, fail the active phase with the original exception before re-raising. Remove the session-wide `context.phase("actor_update")` wrapper.

Extend the existing GRPO adapter tests in `packages/train/tests/test_api.py` or add a narrowly scoped test module. Prove that rollout completion precedes actor start, normal completion emits one duration at the optimizer step, repeated close is safe, an overlapping start is rejected, and a training error emits `runtime_phase_failed` rather than a false completion.

## Concrete Steps

Work from `/home/hammad/projects/rl`.

First edit the TRL adapter and tests with `apply_patch`. Then run:

    uv run pytest packages/train/tests/test_api.py -k 'grpo and (rollout or actor)' -q
    uv run ruff check packages/train/src/posttrain/train/backends/trl/grpo.py packages/train/tests/test_api.py
    uv run pyright packages/train/src/posttrain/train/backends/trl/grpo.py packages/train/tests/test_api.py
    uv run lint-imports
    git diff --check

If the broad test selector includes unrelated dirty-worktree failures, run the exact new test names and record that limitation here rather than claiming the whole package passed.

## Validation and Acceptance

The focused unit test must fail against the old wrapper because `actor_update` begins before rollout and has no per-step duration. It must pass after the change and observe this event order for one update: `rollout` starts, `rollout` completes, `actor_update` starts for optimizer step 1, `actor_update` completes, and `train/rl/time/actor_update_seconds` is emitted at step 1.

A later live GRPO run is the behavior-level acceptance gate. Its runtime phase projection must show one rollout interval and one non-overlapping actor interval per optimizer step. The actor duration must be materially smaller than the previous session-wide interval and system samples must be attributable to the correct phase.

## Idempotence and Recovery

The change adds no migration and does not mutate prior evidence. Existing runs remain historically inaccurate for actor phase duration and should be labeled as such when analyzed. If a live run fails after rollout, the helper emits a failed actor interval and resets its state; retrying creates a new run and does not reuse that phase id. Reverting the adapter and its tests restores the old instrumentation without affecting checkpoints or models.

## Artifacts and Notes

Live evidence that motivated the repair:

    00:39:57 runtime_phase_started phase=actor_update
    00:39:57 runtime_phase_started phase=rollout logical_step=0

The exact diagnostic run is `ambient-k1-dapo-diagnostic-2step-4090-g4-seq64-20260808-r1`. It is analysis evidence, not a release qualification artifact.

## Interfaces and Dependencies

No public API changes. The private helper lives in the TRL GRPO adapter and uses `RunContext.phase`, `RunContext.metric`, and `time.perf_counter`. The callback derives from the already imported Transformers `TrainerCallback` class supplied by `framework_imports()`. The canonical emitted metric is `train/rl/time/actor_update_seconds`.

Revision note (2026-08-08 00:54Z): recorded the implemented lifecycle helper, focused test evidence, and the first live rollout's advantage coverage; full validation and live analysis remain pending.
