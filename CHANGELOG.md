# Changelog

All notable changes to Posttrain are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) with a coordinated
version across first-party distributions.

## Unreleased

Targets 0.4.13, which keeps CarbonTeq veRL 0.9.0.post8 (pinned by 0.4.12); the VORTEX port
selects the post6 additions that 0.4.12 leaves off.

### Changed

- TRL is CarbonTeq 1.12.0.post12 (post11 plus float32 scoring and loss for
  float16 GRPO and RLOO training). Posttrain's own float32 subclass and LM-head
  upcast are kept and are no-ops on post12; bfloat16 runs are unchanged. The
  catalog lock `trl-fork@current` and the TRL bindings record the new lock.

- **Behaviour change for veRL GRPO and DAPO runs.** veRL now reproduces the
  semantics the selected GRPO settings always declared (TRL's), instead of
  silently running different ones. Compared with earlier veRL runs:
  - Policy loss: plain asymmetric token clipping (`token_clip`); veRL's
    default loss capped negative-advantage tokens with dual clipping.
  - KL penalty: the unclamped k3 estimator; veRL clamped it to [-10, 10].
  - Sampler correction: the selected `importance_sampling_mode` and bounds are
    applied (the GRPO default truncates sequence ratios to [0.1, 3.0]); veRL
    applied no correction.
  - Advantage scaling: the group (or batch) standard deviation plus 1e-4 with
    TRL's statistics; veRL added 1e-6.
  - Failed rollout groups: retried with the same prompt up to
    `max_admission_attempts`, then dropped with the loss averaged over the
    remaining rows; veRL either trained the incomplete group or, with
    `rollout_execution`, replaced it with a new prompt.
  - DAPO dynamic sampling: whole candidate batches from a reserved pool, with
    batch-scaled advantages using each candidate batch's standard deviation;
    veRL streamed replacement prompts.

  Runs are therefore not numerically comparable with earlier veRL GRPO runs.
  New veRL GRPO, DAPO, OLMo 3 and SAMPO runs record
  `verl_semantics: trl-parity-v1` in `grpo_runtime_resolved`, so Observatory
  can tell them apart.

### Added

- The VORTEX recipe and SAMPO on veRL: OLMo 3 loss, active sampling with
  oversampling, the adaptive curriculum (including per-checkpoint curriculum
  views for `--curriculum-checkpoint-step`), the truncation penalty, SAMPO's
  turn-level advantages, and every GRPO setting previously rejected on veRL
  (`advantage_scaling`, all importance-sampling modes and bounds,
  `mask_truncated_completions`, `max_admission_attempts`, `lr_scheduler_type:
  linear`, curriculum with DAPO). Each is checked against TRL's real code by
  CPU parity tests; see `docs/plan/verl-vortex-port.md`.
- LFM2.5 on veRL. The launcher resolves the policy's renderer exactly as the
  TRL backend does (the family's renderer config, the reasoning mode's
  template arguments, the package chat template and the tool-call protocol)
  and the veRL agent loop rebuilds that renderer, so LFM2.5's Python call
  lists are recovered as tool calls on veRL too.

## 0.4.12 - unreleased

Training-harness fixes found by auditing the LFM2.5-2.6B SAMPO continuation:
tools that behave as documented, FP16 training, fast Qwen3.5 kernels, a label
for how every episode ended, a checkpoint when a run is cancelled,
oversampled active sampling, and a KL penalty measured against the base model.

### Added

- FP16 training for TRL and veRL online RL (GRPO, SAMPO, GDPO, CAPO). On a
  training binding, `backend_options.training_precision: fp16` (default
  `bf16`) trains a LoRA adapter in float32 over a float16 base with dynamic
  loss scaling; `logits_float32: true` computes the trainer's log-probabilities
  from float32 logits (TRL). The rollout inference binding's `engine.dtype`
  (`bfloat16`, `float16` or `float32`) sets the vLLM sampler's precision.
  `posttrain work-package plan` prints the resolved precision on a
  `Precision:` line. Runs record the loss scale and skipped optimizer steps
  (`train/loss_scale`, `train/optimizer_step_skipped`), and veRL runs also
  record the rollout-versus-trainer log-probability gap. The advisor rejects
  `dtype: float32` for Qwen3.5 rollouts (vLLM's Gated DeltaNet kernel does not
  support it) and warns when an FP16 trainer samples in another precision.
  New lab work packages compare BF16 and FP16 on Qwen3.5-0.8B GSM8K (TRL and
  veRL, 8 GB GPU) and on the LFM2.5-2.6B SAMPO continuation.
- Episode ending labels. Every training and evaluation episode records how it
  ended: `completed`, `turn_limit`, `token_budget`, `time_limit`,
  `reply_token_limit`, `context_limit_reply_cut`, `context_rejected` or
  `error`. Every ending other than `completed` and `error` is a truncation, so
  the `truncated` flag, truncation penalties and masking are unchanged.
  Training writes per-update counts and rates (for example
  `train/rl/ending_reply_token_limit_rate`); Observatory shows the ending in
  the trace table and trace detail, and the semantic layer groups rollouts by
  `rollout.ending`. Trace facts v9 store it in the `episode_ending` fact
  column, and `posttrain trace-facts backfill` fills it for existing runs.
- Checkpoint on cancel. When the host cancels a TRL GRPO, DAPO, OLMo 3,
  SAMPO, GDPO or CAPO run, the last completed optimizer update is saved and
  published as a checkpoint if it is newer than the last periodic one. A
  cancellation that arrives during an optimizer step waits for that update to
  finish (at most 60 seconds). The `cancel_checkpoint` event records the
  saved step or why nothing was saved, and the run finishes as cancelled.
  `train/cancel_checkpoint_step` (value: the saved update) is recorded at the
  step of the update the cancel interrupted, the event's `cancelled_update`.
- Oversampling for active sampling. `active_sampling: {oversample: N,
  oversample_refill: M}` starts N extra prompt groups in the first round and M
  in each refill round, so an update fills in fewer rounds; the surplus is
  discarded. Both default to 0, which keeps the previous behaviour. Job plan,
  and the trainer again before the first rollout, reject a first round larger
  than vLLM `max_num_seqs`, the environment's `max_concurrent` or the rollout
  workers can run at once. TRL only.
- KL reference. GRPO and SAMPO settings take `kl_reference: base | start`
  (default `base`). When a run continues a trained adapter
  (`--model-from-run`), the KL penalty now measures distance from the base
  model instead of from the adapter the run started from; `start` keeps the
  previous behaviour on TRL. Job plan prints which reference a run uses.
- veRL can continue a trained LoRA adapter: the adapter is attached to its
  foundation model, reaches the vLLM rollout before the first collection, and
  keeps training, with the base model as the KL reference. veRL rejects
  `kl_reference: start` for a continued adapter, which it cannot provide.
- Lab: held-out AutomationBench suites `automationbench-lfm26-heldout-mix-v4`
  and `-v4-t05` on the fixed tools, with Liquid's recommended sampling at
  temperature 0.1 and 0.5, and a continuation of the LFM2.5-2.6B SAMPO run
  from its update-40 adapter on the fixed tools.

### Changed

- AutomationBench tools behave as documented: `automationbench-v1` 0.5.0
  (`61448b5d`) vendors the CarbonTeq AutomationBench tool fixes. For example, a
  Sheets row search now searches instead of returning the first ten rows,
  Drive search no longer mixes placeholder files into results, Salesforce
  search matches without a field name, list arguments are no longer corrupted
  into strings, and Sheets row updates are kept for scoring. Graders are
  unchanged; scores on the new suites are not comparable with earlier suites.
- Fast Qwen3.5 kernels: the `supervised`, `online-rl-trl-py312`,
  `online-rl-verl-py313` (backend environment) and `transform` job-kind images
  install `fla-core` 0.5.2 and CarbonTeq's `causal-conv1d` 1.7.0 build for
  PyTorch 2.13.0+cu130, so Transformers trains Qwen3.5 Gated DeltaNet layers on
  fast kernels instead of its torch fallback. A Qwen3.5-0.8B LoRA actor step on
  4,096 tokens drops from 5.0 s to 1.5 s on an RTX 3070 Ti, and FP16 steps no
  longer produce NaN gradients.
- veRL rejects GRPO settings it would otherwise silently ignore, at job plan
  and when the launch plan is built: `adaptive_curriculum` ("currently
  supported by the TRL backend only"), `advantage_scaling` other than `group`,
  `importance_sampling_mode`, `importance_sampling_clip_min` or
  `importance_sampling_clip_max` other than their defaults
  (`sequence_truncate`, 0.1, 3.0), `max_admission_attempts` other than 3, and
  `active_sampling` with the OLMo 3 recipe (until veRL has active sampling).
- veRL applies `truncation_penalty` for GRPO and DAPO with the same reward
  shaping as TRL, maps the OLMo 3 objective natively (ready for when active
  sampling lands), and rejects a veRL revision that lacks the loss or KL names
  GDPO, CAPO or OLMo 3 select. veRL `0.9.0.post5` adds the `token_clip` policy
  loss and the `k3_unclipped` KL that GDPO and CAPO use; with post3 and post4
  those runs failed at the first actor update.
- veRL runs the training loop as selected, or rejects it: `lr_scheduler_type`
  `constant` and `constant_with_warmup` map to veRL's constant schedule with
  zero or `ceil(max_steps * warmup_ratio)` warmup steps, and `linear` is
  rejected (this release does not map it to veRL); `seed` seeds prompt order,
  the rollout sampler and the FSDP engines; `logging_steps` must be 1; and
  `per_device_batch_size` becomes the per-device micro-batch, with
  `per_device_batch_size x gradient_accumulation_steps` required to equal
  prompt groups x generations and to split evenly over the devices. Behaviour
  change: veRL now trains with weight decay 0.0 (the TRL backend's value)
  instead of 0.01, seeds its prompt order, and uses the selected micro-batch
  instead of one row. It always ran a constant learning rate; the lab veRL
  settings, which left the `linear` default, now say `constant`.
- The veRL backend environment uses the framework's Verifiers (`e6a3d9bb`) and
  `carbonteq-renderers` 0.1.12.post1.dev2, the same as the control
  environment, so Verifiers environments can be packaged for veRL again; the
  image declares Verifiers as provided in both environments, and a release
  check fails if the two ever select different Verifiers.
- Trackio `0.31.5.post14.dev32`: artifact commits keep retrying while the
  server is slow, and the `episode_ending` trace-fact column (Doris schema
  version 5). Migrate the shared server to schema version 5 and run dev32
  before job images from this release write to it.
- Maintained forks: TRL `1.12.0.post11` (oversampling and `peft_reference`),
  veRL `0.9.0.post8` (loss scale, skipped steps and log-probability gap
  metrics from post4; `token_clip` and `k3_unclipped` from post5; post6's
  opt-in trainer features, which this release does not select; the LoRA
  weight-sync fix from post7; the agent-loop config defaults from post8), Trackio `0.31.5.post14.dev32`, and the `causal-conv1d`
  `1.7.0+cu130torch2.13` rebuild, which is promoted to the stable index as a
  wheel-only fork release.

### Fixed

- A cancellation checkpoint saved after the next update's rollout had started
  lost its `train/cancel_checkpoint_step` metric: it was recorded at the saved
  update's step, below the rollout metrics already logged at the next step,
  and the tracker rejects decreasing steps (seen on the local provider; the
  dstack run was cancelled before the next rollout logged). The metric is now
  recorded at the interrupted update's step, and a metric the tracker still
  rejects is named in the `cancel_checkpoint` event (`metric_error`).
- veRL LoRA training on models with fused layers: veRL `0.9.0.post5` named the
  synced LoRA tensors with vLLM's stacking weight mapper, which merges
  `q_proj`/`k_proj`/`v_proj` into `qkv_proj` and LFM2's `w1`/`w3` into `w13`,
  so the constituents collapsed onto one name and the rollout engine crashed
  or silently loaded the wrong adapter weights. The veRL kind now pins
  `0.9.0.post7`, which uses vLLM's rename-only mapper as vLLM's own adapter
  loader does (post8 carries the same fix). Every other veRL setting the
  release generates resolves as on post5.
- veRL trained on zero reward for math environments: Verifiers'
  `verify_boxed_math_answer` bounds math-verify with `signal.alarm`, which only
  the main thread may set, and veRL scores episodes inside Ray actors off the
  main thread, so every answer scored 0 (all GSM8K traces of the qualification
  run, correct replies included). Verifiers `e6a3d9bb` (`0.3.2.dev94`) scores
  such calls in a worker process that keeps the timeout. TRL scored on the
  main thread and was unaffected.
  Every environment package pins the same Verifiers, so the lab and base
  catalogs move to pin-only verifiers-environments commits: `11f4d712` (on
  `5264ec15`), `e9eacc3c` (on `3a486b0a`), `a344d127` (on `0afb73d7`) and
  `0bad6187` (on `61448b5d`, AutomationBench 0.5.0). Environment code is
  unchanged at each revision.
- veRL runs reported `train/rl/kl` as PPO's approximate KL to the rollout
  policy (`actor/ppo_kl`), which is 0 for an on-policy update, instead of the
  KL to the reference (`actor/kl_loss`); every veRL run so far logged KL 0
  whatever the policy did. `train/rl/kl` now means the same on TRL and veRL.
- veRL runs failed after their last update while recording rewards: the
  launcher attached a `rollout_step` dimension to each trace's
  `algorithm_reward` enrichment, which Trackio rejects (`a trace-fact
  enrichment may only supply algorithm_reward`). The rollout step belongs to
  the Verifiers source projection; the enrichment now carries only the
  algorithm reward, and the shared observation contract rejects any later
  trace-fact update that supplies more, so every observer catches it.
- veRL jobs without `rollout_execution` settings failed before their first
  rollout (`ConfigAttributeError: Key 'num_cpus_per_worker' is not in
  struct`): since fork post2 veRL's agent loop read three episode-capacity
  keys that its trainer config never declared, and Posttrain passes them only
  with `rollout_execution`. veRL `0.9.0.post8` declares them with their
  defaults (one reserved CPU per agent-loop worker, no episode ceiling).
- Every job container now gets an explicit `/dev/shm` size. Docker and dstack
  left the 64 MiB default, and veRL's rollout server failed at start
  (`Insufficient space in /dev/shm ... 160 MiB required, 64 MiB free`) on both
  providers: vLLM's multiprocess executor allocates a 160 MiB shared-memory
  broadcast queue at engine start. The size is 16 GiB unless the execution
  target's placement declares `shm_size_gb`; a declared `host_memory_gb` caps
  the default at half of it and rejects a larger explicit size. The local
  provider passes `docker run --shm-size` (a private tmpfs, not `--ipc=host`);
  dstack receives `resources.shm_size` and a matching `resources.memory`
  minimum so the offer can hold it. dstack applies the size on VM and SSH
  fleets; its RunPod backend creates pods whose shared memory RunPod sets, so
  the job runtime now checks `/dev/shm` at start and fails at once, naming the
  required and actual size, the target and the provider, when it is smaller
  than required (declare `shm_size_gb` on a target whose provider sets a
  smaller fixed size). Runs record the size in the worker context
  (`shared_memory_bytes`). `posttrain job plan` prints `Container shared
  memory:`, and the job run plan, provider plan and submission receipt record
  `shared_memory_gb`.
- veRL multi-turn episodes reported the prompt message spans of a bridged turn
  with one entry per new message instead of one per message of the
  conversation, so Verifiers attributed tool-result tokens to the wrong
  messages in the trace (the token sequence and loss mask were unaffected).
  They now use the TRL backend's `bridged_message_spans`.

- Observatory semantic SQL (`/api/v1/semantic/query`, MCP `query_semantics`,
  evaluations and bare run ids) read whichever source sorted first when no
  source was named; with discovered Trackio projects that was an unrelated
  project (`ai-infra-qualification` instead of `posttrain-lab`, 0 runs). A
  request now reads the source it names (`?source_id=` on HTTP, `source_id`
  on MCP), else the configured default, else the only source; with several
  sources and no default it is refused with the available sources listed.
  The default is `POSTTRAIN_OBSERVATORY_DEFAULT_SOURCE`, else the configured
  Trackio project (`POSTTRAIN_TRACKIO_PROJECT`) when projects are discovered.
  Deployments that discover projects must set one of them to `posttrain-lab`.
- Lab Verifiers environments state the harness, stage timeouts and turn and
  token limits on the agent seat, which Verifiers `cdd2ec76` requires; a test
  validates every catalog environment against the pinned Verifiers.
- FP16 TRL training no longer fails at the first gradient overflow with a
  fused optimizer (Transformers' default `adamw_torch_fused`): a skipped step
  is detected from the loss scale falling, recorded in
  `train/optimizer_step_skipped`, and its non-finite gradient norm dropped.
- Cancelling a job on the local provider gives it 300 s (was 10 s) to save,
  publish and finalize its cancellation checkpoint, and cleanup moves any
  checkpoint a stopped worker did not finalize to
  `<state>/retained-checkpoints/<run id>` instead of deleting it. The search
  runs inside the cleanup container, which can read the worker's root-owned
  scratch directories, and an unreadable directory stops cleanup.
- Oversampled active sampling runs again: the rollout function accepted at
  most prompt groups x generations rows per round, so the oversampled first
  round ((prompts + oversample) x generations) failed every OLMo 3 GRPO and
  SAMPO run, including adaptive-curriculum rounds, at its first rollout.
- Runs without an adaptive curriculum now record TRL's per-round
  active-sampling counts as
  `train/rl/active_sampling_round_<n>_{requested,generated,retained}_groups`
  (rounds bounded by `max_candidate_batches`), described in the Observatory
  metric catalog; they previously stayed in TRL's console log.
- `posttrain run cleanup` completes for dstack runs whose exact-worker cleanup
  task reclaimed more than about 2 GB: the task logged the total in awk's
  scientific notation, which cleanup rejected, leaving the run in attention
  with its placement held. The count is now printed in whole bytes, the old
  form is accepted, late logs are retried, and a finished task whose log is
  gone still completes cleanup.
- `posttrain controller run` logs why a run needs attention (exception type
  and first line, bounded), and `posttrain controller status` lists those runs.
- FP16 TRL training computes its loss in float32. TRL computed the KL term,
  importance ratios and masked sums in float16 from float16 log-probabilities,
  so `exp` of a log-ratio above about 11 overflowed; on the masked tool tokens
  of multi-turn SAMPO completions that made the loss NaN and the LFM2.5-2.6B
  fp16 canary skipped every update. FP16 training now always takes
  log-probabilities from float32 logits and casts them, and the entropies, to
  float32 before the loss; `logits_float32: false` is rejected with fp16.
- FP16 training starts its dynamic loss scaler at 1024 instead of PyTorch's
  65536 (`backend_options.fp16_initial_loss_scale` on a training binding, TRL
  and veRL): starting high, the Qwen3.5-0.8B fp16 arm skipped six of its first
  seven updates while the scale backed off. Job plan shows the starting scale
  on its `Precision:` line.
- veRL workers start on the veRL job kind again: the worker reads the source
  revision from the kind's `.posttrain-source-revision` snapshot marker instead
  of running `git`, as `posttrain-runtime` does.
- Training rollouts run the Verifiers harness scripts from the job's locked
  environment instead of installing `uv` and the harness dependencies from
  PyPI at the first rollout; `posttrain-runtime` enables this for every
  packed job, as the evaluation kind already did.

## 0.4.11 - 2026-09-28

Runs can be queried in SQL and carry notes; metrics and trace facts are correct
where they are recorded.

### Added

- Semantic layer: `runs`, `updates` and `rollouts` as SQL tables, computed
  inside Trackio's storage (Doris SQL; translated on local SQLite storage) for
  only the columns a statement reads. `posttrain query`, HTTP
  `/api/v1/semantic/*` and the MCP `query_semantics` tool take SQL or a short
  `measures`/`by`/`where` form that compiles to SQL; every result shows its SQL.
- Run notes: Markdown notes stored in Trackio with revisions, named
  ```` ```sql <name> ```` data blocks, chart/value/table views, `{{block.column}}`
  and `{{run.<dimension>}}` references and `[[run:id]]` links. Every job kind
  has a run card template. Surfaces: the Observatory run page, HTTP, MCP and
  `posttrain note`.
- One metric catalog (`posttrain_observatory.metric_catalog`) describes every
  metric job views show and every queryable measure.
- Observatory evaluation views, built from the training run and checkpoint
  step each evaluation records (`GET /api/v1/evaluations`,
  `GET /api/v1/evaluations/tasks`): the sidebar lists each training run's
  checkpoint evaluations under it with their suite and score; a training run's
  Evals tab compares the base model with each checkpoint step per suite,
  overall and per task; and the Evals page compares two training runs of the
  same base model on a suite they share.
- Observatory run pages have a Notes tab for the run card and notes.

### Changed

- The TRL trainer writes rollout metrics once per update; per-batch time is
  `train/rl/rollout_batch_seconds`. Readers return logical steps with replay
  authority applied, and combine the per-batch points of older runs.
- Trace facts v8: `task_id` is the environment's task key (for AutomationBench
  the task name), falling back to the dataset's example id, so training and
  evaluation name tasks the same way across runs.
- Trackio `0.31.5.post14.dev31` (run notes, read-only project SQL, set-oriented
  trace-fact replacement, reliable remote delivery; Doris schema version 4).
- Observatory: SAMPO runs use the GRPO overview (headline metrics, the
  Policy optimization swimlane with rollout behavior, update stability, rollout
  population, runtime, freshness, acceleration, active sampling, rollout setup
  and grouped rollouts) plus a Hierarchical credit tab. SAMPO's
  episode and turn advantage means are zero by construction; the trainer now
  records episode and turn credit magnitudes, the turn share of credit, turns
  with turn credit, and turns without a peer. The tool failure rate joins the
  Policy optimization swimlane beside tool calls per rollout.
  Chart lanes share one step range, axis labels stay short for values near
  zero, and each tab names the metrics a run did not record.
- Observatory tables page instead of growing or scrolling inside a box (note
  tables 15 rows, prompt groups 10, traces 25; the last page's Next loads older
  rows), sort by column, and note tables have a row filter. Note text reflows
  to the page width, and data tables keep ids on one line, wrap long prose
  between words and scroll sideways with the first column pinned.
- Group-policy and SAMPO run cards (templates `@3`) show first and last
  ten-update averages of reward and entropy, the best ten-update window, KL and
  truncation over the last ten updates, and the recorded error message.
- Trackio and W&B runs are named by the full run id instead of its first eight
  characters (every held-out evaluation showed as `eval.general-eval-lfm`).
  Resuming a Trackio run falls back to the old name, and runs that carry it
  are shown with their full id.
- The release candidate workflow no longer runs a GPU job: it builds,
  verifies, installs from the development index and publishes. Its packed
  transformation canary wrote a `release-candidate-<run id>` run into the lab
  Trackio project on every release and needed the RTX PRO worker idle. The
  `qualification_profile` and `run_gpu_qualification` inputs and
  `scripts/release/verify-dstack-capacity` are removed.
- Training publishes its replay authority once, compressed: the native episode
  envelope as gzip (about 5x smaller), hashed without reading it into memory.
  The derived trace view is no longer published a second time; it was streamed
  to tracking and is rebuilt from the episodes. A 150-update run's final upload
  falls from 7.7 GB to about 0.7 GB.

### Fixed

- Separate rollout batches that reported the same value were merged, which
  undercounted rollouts and rollout time.
- Observatory rollout tables showed no thinking or output tokens for training
  runs: those counts come from provider usage, which training rollouts lack.
  Trace pages now fill token and turn counts from each trace's stored facts
  (one bounded project-SQL read per page), and the grouped rollout table adds
  a Turns column beside tool calls (mean per group, exact per rollout).
- Training runs recorded no system metrics (GPU and CPU use): recoverable jobs
  open their Trackio run with resume "allow", and the adapter started Trackio's
  GPU and CPU monitors only for resume "never", which only evaluations use.
  Monitoring is now chosen by the opener: a starting or recovering job monitors,
  reopening a run to record its outcome does not.
- Traces lost on the way to Trackio: a failed request left its entries in the
  client buffer, which was then resent as one ever-growing request that a proxy
  refused every time, and was dropped when the job exited (7,546 of 11,876
  traces of a 150-update run). Trackio dev31 sends at most 8 MB per request and
  keeps what it could not send.
- SFT/DPO and veRL runs recorded no GPU metrics: their images lacked
  `nvidia-ml-py`, which Trackio's GPU monitor needs and other images received
  through vLLM. The common runtime profile now pins it.
- SAMPO, GDPO and CAPO on the TRL backend rejected their first update when the
  rollout engine used speculative decoding (DSpark, MTP): their observation
  features ignored the engine, so its speculative counters looked unexpected.
- A long artifact upload failed with 403 once its 15-minute signed part URLs
  expired, and a queued artifact timed out behind a slow upload; dev31 signs
  parts again and times out only when uploads stall.

## 0.4.10 - 2026-09-26

SAMPO can train on per-turn rewards from the AutomationBench environment itself.

### Added

- `train/sampo-turns@1`: SAMPO with direct per-turn rewards selected by a reward
  projection, without a judge.
- `automationbench-lfm12-sampo-8gb-turns-v1` uses `automationbench-v1` 0.4.2
  (`3a486b0a`), which scores the live world before every model call: each
  assistant turn earns its change in partial credit minus 0.05 per failed tool
  call. `reward/automationbench-turn-progress@1` selects that `turn_reward`, and
  `train/lfm2.5-1.2b/automationbench-sampo-turns-local-8gb` runs it on an 8 GB
  GPU. Other environments keep AutomationBench `5264ec15`.

### Changed

- The renderers consumer page records `0.1.12.post1.dev2`'s qualification
  evidence and its promotion to stable.

## 0.4.9 - 2026-09-26

SAMPO collects like VORTEX, training parses tool calls the way serving does, and
evaluations report running out of context as truncation.

### Changed

- SAMPO refills with VORTEX active sampling (dynamic sampling removed): it keeps
  prompt groups whose rewards differ and generates only the missing groups from
  a reserved candidate pool, admits groups before computing advantages, and can
  select the adaptive curriculum. The vLLM sampler correction defaults to a
  per-token cap of 2.0, an optional truncation penalty shapes the episode
  reward, and anchor-state keys drop tool-call and other sample IDs
  (`content-without-sample-ids@2`). The veRL backend rejects SAMPO, since it has
  no active sampling.
- `carbonteq-renderers` `0.1.12.post1.dev2` (`6f712616`): a tool-call opener ends
  an unclosed thought, and LFM2.5 pythonic tool calls get the vLLM fork's `lfm2`
  repairs (nested quotes, raw control characters, zero-padded integers,
  keyword-named parameters). Training dropped such calls while serving accepted
  them; LFM2.5-1.2B scored 0 in training and 1.0 in evaluation on the same task.
- Evaluation: a rollout whose final model request exceeds the context is
  truncated, not failed, and keeps the reward the environment scored (trace
  fact calculator v7). Evaluation status is `complete`, `truncated`, `partial`
  or `failed`; a run whose rollouts all fail now fails after recording its
  evidence.
- The OLMo 3 recipe selects a KL penalty (`beta`) and applies it; TRL's
  `Olmo3GRPOConfig` fixed it at 0.

### Added

- `train/lfm2.5-1.2b/automationbench-sampo-local-8gb`: SAMPO on an 8 GB GPU over
  the 57 AutomationBench tasks whose LFM2.5-1.2B attempts disagree, at an 8K
  context; and `screen/lfm2.5-1.2b/automationbench-screen-8k`, the task screen
  that chose them.
- VORTEX v5 for LFM2.5-2.6B at learning rate 5e-5 with a 0.005 KL penalty, and
  `docs/techniques/grpo/recipes/lfm2.5-2.6b-automationbench-vortex.md`, the
  learning rates, settings and efficiency changes tried.
- Launch checks: an online-RL training batch must equal prompt groups times
  generations for SAMPO too, a curriculum class field must be an environment
  facet, and a TRL candidate pool must fit the environment's tasks.

### Fixed

- Managed evaluations failed every rollout in harness setup since 0.4.7: the
  preinstalled runtime called `.setdefault` on Verifiers' new `LoopLocks`.
- A rollout-runtime shutdown failure no longer hides the training error, and
  cached GPU memory is released before an asynchronous collection wakes vLLM.

### Removed

- Six SAMPO qualification gates the new contract cannot run are retired:
  `sampo-extended`, `automationbench-sampo` and four `verl-sampo*`.

## 0.4.8 - 2026-09-26

A VORTEX v5 LFM2.5-2.6B update now takes about 276 seconds instead of 823.

### Changed

- TRL `1.12.0.post10` (`4950b99d`): GRPO scores each micro-batch at its own
  real length instead of the generation batch's padding (~25K tokens for ~9.7K
  real ones). Training bindings can compile each decoder layer
  (`compile_decoder_layers`), limit gradient checkpointing to long micro-batches
  (`gradient_checkpointing_min_tokens`), and take the vLLM importance-sampling
  ratio from the training forward instead of a separate no-grad pass
  (`importance_sampling_from_training_logps`).
- Training binding `training/lfm2.5-2.6b-trl-lora-automationbench-local-g64-w8@2`
  selects compile and the training-forward ratio; the actor update fell from
  373 to 114 seconds per 64-episode update.
- `inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c64-4k@2` rolls out
  with DSpark at c64 and a 13.75 GiB KV cache; rollout fell from 334 to about
  158 seconds per update, at 26% draft acceptance.
- Speculative-decoding evidence is expected for every drafting method, not
  only MTP.

### Added

- Actor-update timing splits into `train/rl/time/actor_forward_backward_seconds`
  and `train/rl/time/optimizer_step_seconds`.
- VORTEX v5 continuations from run r2's step-40 checkpoint: the optimizer A/B
  arms and a 150-update DSpark continuation.

### Fixed

- Resume selects the committed checkpoint when an interrupted run re-published
  the same step.

## 0.4.7 - 2026-09-26

### Added

- Configuration rules and a settings calculator (`posttrain.advisor`):
  `work-package validate`, `job plan` and `job run` report every configuration
  finding and reject unacknowledged performance errors (`--strict` also rejects
  warnings); `posttrain settings suggest` calculates engine settings, memory and
  step sizing; Observatory shows the same review on every run.
- Bindings can acknowledge a finding with a written reason
  (`performance_acknowledgements`). Bindings that recorded runs used keep their
  engine and acknowledge their findings, naming the successor.
- DSpark speculative rollout: TRL rollouts accept a DSpark drafter pinned to a
  commit, and `inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c32-4k@3`
  uses LFM2.5-2.6B-DSpark with nine tokens and a 7 GiB KV cache sized for
  target plus drafter (replay: 46 s per 32-episode collection against 84-87 s
  without speculation). The settings calculator counts drafters named in
  vLLM's speculative schema.

### Changed

- vLLM `carbonteq-v0.29.1.dev4` (`f09e4479`): LFM2 DSpark speculative
  decoding, DFlash/DSpark keep their trailing prefix-cache block, and
  session-aware prefix-cache eviction (`release_session`,
  `POST /v1/sessions/release`).

- The `posttrain init` GRPO starter uses `qwen3.5-0.8b-vllm-distill-rollout@4`
  (vLLM 0.29.1.dev3, CUDA graphs, calculator sizing for a colocated 8 GB GPU);
  the release-gate evaluation uses `qwen3.5-2b-vllm-eval@3`.
- Serving computes in the checkpoint's precision when a binding omits `dtype`
  and enables prefix caching when it omits `enable_prefix_caching`; the veRL
  worker honours the binding's prefix caching and batch invariance.
- Verifiers `cdd2ec76` and environments `5264ec15` for every base environment;
  GSM8K scores in-process.

### Fixed

- A failed GSM8K scoring call no longer poisons later calls with "bound to a
  different event loop".
- The GRPO starter's environment runs in local subprocesses (not Prime
  sandboxes) and checks its dataset at job start, not during the offline build.
- `run cleanup` releases a failed-versus-cancelled local run that never wrote
  outputs, which previously held the GPU placement forever.

## 0.4.6 - 2026-09-25

A patch release for 0.4.5.

### Fixed

- Jobs that package a Verifiers environment (GRPO and VORTEX training, and
  evaluations with environments) failed to pack on 0.4.5 with "compiled
  dependency lock contains a mutable or non-portable source". The runtime
  images declared only `verifiers` as provided, so its `carbonteq-renderers`
  dependency, pinned in the image lock as an internal-index URL, leaked into
  the portable environment lock. Kinds that provide Verifiers now also provide
  the renderers fork their lock installs.
- `main` pins the runtime images 0.4.5 published, so jobs launched from a
  source checkout resolve images that match the committed runtime locks.

## 0.4.5 - 2026-09-25

This release makes VORTEX yield-first, counts thinking tokens for every catalog
model from the renderer instead of model-specific rules, and turns Observatory
into a place to debug a rollout: where its time went, turn by turn, and which
tool calls failed.

### Training and curriculum (VORTEX)

- Yield-first curriculum: full-pool class-then-task sampling by useful-group
  probability, with an exploration lane (`exploration_share` 0.2) that adds
  aged binary-yield uncertainty. Same-step repeats are a hard capacity error.
- VORTEX v2 on the CUDA-graph c40 rollout profile, and VORTEX v5 with a
  truncation penalty and a 64-rollout batch, extended to 100 updates by resume.
- A curriculum can warm-start from another run's controller state.
- Resume works from a final checkpoint that is also a periodic checkpoint;
  rejected rollout collections are logged instead of dropped.
- Matched LFM adaptive GRPO and GDPO arms fit a colocated GPU budget and keep
  32K of Gemma judge headroom.

### Evaluation

- 20-turn 64K AutomationBench evaluations and a difficulty-tiered held-out set,
  both reserved from training; matched VORTEX evaluation packages are candidate
  gates, with a 48K canary.
- Evaluation keeps provider-error evidence, and a Trackio trace-schema
  preflight fails before submission instead of mid-run.

### Thinking-token accounting

- New maintained fork `carbonteq-renderers` `0.1.12.post1.dev1` (internal
  index). Every renderer reports `reasoning_tokens` for its completion, and
  every catalog model has one: LFM2.5, K2 Horizon, Nanbeige 4.2, Spark 2.5,
  Gemma 4 (including 12B) and Qwen 3.5.
- Verifiers `0cee0a07`: the train client records the renderer's count as
  `usage.reasoning_tokens` on each call. Verifiers environments move to
  `8b739717`, which selects the same Verifiers commit.
- Trace facts are calculator `verifiers-trace-facts.v6`. v5 added task and
  prompt-group ids; v6 takes thinking tokens only from per-call usage and drops
  the Qwen-specific recovery rules.
- `posttrain trace-facts backfill --renderer-model` re-scores historical traces
  and refuses `--apply` when it would erase recorded thinking counts.

### Tracking

- Trackio `0.31.5.post14.dev27`: server-side aggregates over stored trace
  payloads (`TracePayloadQuery`: bounded JSON paths, mean, sum, count, min, max)
  on Doris and SQLite. The tracking contract gains `aggregate_trace_payload`;
  providers without it report `unsupported`.
- Trace pages can be read newest first.

### Observatory

- Per-turn rollout transcript: one row per model call with its timing, tokens
  and finish reason; each tool call beside its result, matched by call id within
  the turn; failed calls and a missing final answer are flagged; a pinned turn
  index; timeline segments jump to their turn. The same rollout that rendered
  33,000px tall now renders about 3,200px.
- Where rollout time goes: real elapsed generation time, average rollouts in
  flight, and the per-rollout split between GPU inference and CPU tools, setup
  and scoring. Finished steps are cached, so a refresh reads only changed steps.
- Per-trajectory timeline of setup, model calls, tool execution and scoring.
- Prompt-group reward mean and standard deviation across the whole run from
  indexed facts, with explicit partial and unavailable states.
- Trace filters by step, slice, outcome and text search, backed by the server.
- VORTEX sampling evidence in plain terms: whole-run useful and tied groups
  (which add up to 100%), groups kept and discarded per step, sampling rounds,
  and first-time versus repeat tasks, with a step-range slider.
- Sticky page header and turn index now stay pinned while scrolling.

### Operations

- `posttrain run purge --orphan` previews and purges a tracking run that has no
  local submission receipt, with provider, registry, lineage and
  terminal-or-stale checks. A shared job image is retained with a warning.

### Inference and runtime

- CarbonTeq vLLM `0.29.1.dev3`: multi-turn prefix reuse for hybrid and
  sliding-window models (agentic turns no longer re-prefill the previous
  turn's generated tokens), generic SM120 batch-invariant GEMM, split-KV
  attention, GDN chunk alignment and invariant CUDA RMSNorm.
- Verifiers fork server for rollout tool servers and harness programs, faster
  port discovery, and no-delay MCP tool-server sockets. Posttrain enables it
  for Verifiers jobs by default (`VF_FORK_SERVER=0` opts out); AutomationBench
  host time per episode falls from about 6.6 s to under 1 s at concurrency 16.
- The TRL policy endpoint skips detokenization when no stop strings are set
  and no longer echoes prompt token IDs; Verifiers episode preservation runs
  off the event loop.
- Native FlashAttention 4 is rejected on SM120 targets, and attention-backend
  priority compiles to the vLLM backend setting.

### Upgrade notes

- Deploy Trackio `dev27` before using Observatory's rollout-time view; older
  servers report the view as unavailable.
- Re-score runs recorded before this release with
  `posttrain trace-facts backfill <provider-run> --renderer-model <model> --apply`
  to fill task ids, prompt-group ids and thinking tokens.

## 0.4.4 - 2026-09-20

This release promotes the release-clean Uno/SM120 runtime, adds the Gemma
DSpark comparison profile, and makes unsupported rollout combinations fail
before job submission.

### Added

- CarbonTeq vLLM `0.29.1.dev2` with native Uno, native policy-LoRA
  composition, an optional released SM120 paged-FA4 backend, and tuned
  batch-invariant SM120 linear configurations.
- A pinned Gemma 4 12B DSpark inference profile using the qualified Triton
  attention path, seven-token assistant, compiled target, 64K context, and
  16K output budget on RTX PRO 6000.
- Reusable cache, speculative-acceptance, concurrency, token-throughput, and
  judge-trace measurements for rollout qualification.

### Changed

- The K2/Uno VORTEX profile uses the second-generation K2 renderer contract
  with medium reasoning by default. Experimental Gemma-on-SM120-FA4 routing is
  excluded; SM120 FA4 remains confined to the qualified Uno path.
- Posttrain admits native Uno inference and LoRA training. Full-weight and
  QLoRA Uno refresh now fail during job compilation until they pass separate
  live optimizer/update gates.
- The vLLM runtime image uses the CUDA 13 binary wheel from the exact upstream
  base of the fork. Rejected authoritative-prefix, proposer-graph,
  deterministic-noise, and debug-switch experiments are absent from the
  released configuration.

## 0.4.3 - 2026-09-18

This release adds native Uno speculative rollout integration for the VORTEX
training path and advances the maintained TRL, veRL, and vLLM fork closure.

### Added

- A position-gated Uno proposer integrated with vLLM verification, continuous
  batching, abort, sleep, log-probability, and weight-refresh lifecycle paths.
- LoRA-aware rollout synchronization with explicit policy-version fencing and
  reward-object normalization at the active-sampling admission boundary.
- Cache, speculative-decoding, concurrency, and token-throughput observations
  for rollout qualification.

### Changed

- TRL now consumes the immutable CarbonTeq vLLM release and fixes MoE auxiliary
  scoring capability detection; veRL consumes the same fork without claiming
  native K2 training support.
- Runtime images advance to the Torch 2.13 CUDA 13 closure and rebuild the
  shared base instead of duplicating mismatched CUDA libraries in the veRL
  backend layer.
- Release preparation now permits the expected pending image materialization;
  exact runtime locks and OCI identities remain protected-candidate outputs.

## 0.4.2 - 2026-09-15

This release adds the VORTEX training profile and replaces the implicit
candidate-to-final runtime-file handoff with a verified materialization
contract.

### Added

- VORTEX (Variance-Oriented Refill and Task Exploration), a qualified
  AutomationBench profile that combines adaptive curriculum proposals with
  active oversampling and refill, retaining ten prompt groups per optimizer
  update across 40 concurrent sequences.
- Hash-addressed candidate materialization receipts that bind every generated
  runtime lock and image manifest to the exact candidate source, readiness
  receipt, target version, file set, and file content.
- A Go Task entrypoint for reproducible local quality and protected release
  dispatch commands.

### Changed

- VORTEX uses the qualified 12K episode and 4K completion budgets. Its live
  20-update qualification completed without failed rollouts; the final 12.5%
  completion-truncation observation remains a documented warning rather than
  an eliminated condition.
- Final publication now verifies and projects the candidate materialization
  into an isolated build stage instead of requiring generated runtime files to
  be committed or reconstructed from the merge commit.

## 0.4.1 - 2026-09-15

This release introduced manifest-backed adaptive curriculum selection and
held-out evaluation measurement for AutomationBench, including replayable
selection evidence in Observatory and hardened v4 runtime images.

## 0.4.0 - 2026-09-11

This release line adds structured-credit online RL and the native foundations
for overlapping agent rollout collection with learner updates. It also makes
model, inference, training, and hardware compatibility visible before remote
submission.

### Added

- First-class `train.gdpo` and `train.capo` operations with typed reward
  projection, exact sampled-token masks, complete-group admission, checkpoint
  identity, and TRL/veRL backend adapters.
- Native asynchronous Verifiers group production for TRL's async GRPO learner,
  including policy-version spans, behavior log probabilities, bounded stale
  sample admission, explicit cancellation, and recovery of generated but
  unconsumed groups.
- Model- and hardware-aware configuration diagnostics covering update method,
  training and rollout topology, serving features, memory pressure, and
  controlled readiness-preflight bypass.
- Source-capsule packing for local backend checkouts so development GPU jobs
  can qualify exact local changes without publishing temporary wheels.

### Changed

- veRL agent loops preserve phase and row sampling controls while using native
  asynchronous request handling and bounded environment-worker capacity.
- Verifiers integration isolates recoverable episode defects to complete prompt
  groups while keeping infrastructure, identity, and cancellation failures
  fatal to the affected collection.
- Hardware inventory is resolved through explicit dstack targets and live
  admission evidence rather than implicit workstation assumptions.
- Trackio multipart publication retries transient idempotent control requests,
  while failed and cancelled jobs flush their terminal status before an
  artifact-drain error is surfaced. Candidate materialization also refreshes
  the independently locked transform environment instead of carrying a stale
  tracking client into that job kind.
- Work execution delegates the bounded artifact-drain timeout to the selected
  tracking backend instead of imposing a shorter framework-wide constant, so
  large model artifacts honor Trackio's configured finalization window.
- Provider cleanup distinguishes a durably queued exact-worker cleanup from a
  fatal cleanup error. Queued dstack cleanup receives highest scheduling
  priority while terminal-marker-based infrastructure retention remains the
  fallback when a single-slot worker is occupied.

### Release gates still open

- Public async selection remains disabled until checkpoint/resume, live failure
  paths, packaged publication, and the bounded 2B learner qualification pass.
- GDPO and CAPO remain release candidates until all selected TRL/veRL cells pass
  packaged GPU update, checkpoint/reload, exported inference, and scorer
  qualification. Existing numerical and five-step toy evidence is not promoted
  into a broader support claim.
- The generated runtime-image manifest must be produced from the final
  accepted `0.4.0rcN` candidate source; older dev-only `0.4.0` package bytes
  and the earlier runtime graph predate the current release-candidate commits
  and are not reusable as release evidence.

## 0.3.26 - 2026-09-03

This release makes the infrastructure-owned remote image builder a first-class
machine setup option while keeping cloud and registry credentials out of
developer environments.

### Added

- `posttrain machine init --job-builder-endpoint` writes the remote builder
  service binding, its named credential reference, and a mode-0600 credential
  placeholder in one idempotent machine initialization flow.
- Starter projects persist `uv` system-certificate trust, so the initial and
  later dependency syncs honor an organization CA installed in the operating
  system store without requiring a shell environment variable.

### Changed

- The getting-started path now separates the developer's scoped Job Builder
  token from infrastructure-owned OCI push, OCI pull, RunPod, and cloud object
  storage credentials.
- RunPod operations documentation records the deployed dstack release and
  clarifies that Low stock means scarce capacity rather than no capacity.
  Infrastructure still ranks High and Medium first and retains bounded retry
  and failed-region cooldown.

### Qualification

- Focused and full CLI coverage validate the generated remote-builder binding,
  credential reference, protected file mode, and compatibility with existing
  machine initialization.
- External-consumer failures now retain the failed command's stdout and stderr,
  making certificate and maintained-fork resolution failures actionable.
- The deployed dstack release remained healthy through rolling replacement and
  returned live Low-stock A100 and RTX PRO 6000 offers without creating a paid
  Pod.

## 0.3.25 - 2026-08-31

This release makes digest-pinned Posttrain jobs portable between retained LAN
workers and dstack-managed RunPod spot capacity while preserving one image
identity and evidence lifecycle.

### Added

- RunPod spot qualification targets for RTX 4090 and RTX PRO 6000 capacity,
  with provider-managed run storage and explicit price/stock constraints.
- Automatic CUDA forward-compatibility selection for the veRL runtime without
  creating hardware-specific image lineages or globally forcing the payload.
- Immutable package-source sealing checks for remote actual-job builds.

### Changed

- The veRL image keeps stable parent descriptors, cache-lineage epochs, and
  backend packages while updating the shared control environment to the
  current Trackio release. Publication continues without forced recompression.
- dstack cleanup recognizes run-owned storage as provider-managed: it waits
  while the volume exists and completes from authoritative volume absence
  instead of scheduling an exact-host task on a deleted cloud worker.
- Runtime publication and remote build validation preserve registry-scoped
  package identities and sealed build contexts.

### Qualification

- The v0.3.25 veRL image passed manifest/lock validation, a clean LAN pull, the
  current Trackio import, and PyTorch CUDA execution on the local RTX 3070 Ti.
- RunPod qualification r21 completed two training steps, retained seven
  Trackio artifact records, and ended with both Pod and network volume absent;
  reconciliation now reaches admission `completed`.

## 0.3.22 - 2026-08-23

This release completes the migration path from historical retained job-pack
state to the bounded cache lifecycle introduced in the v0.3 series.

### Added

- `posttrain cache migrate-legacy-pack` previews and prepares only the selected
  project's historical assembled contexts for normal cache pruning.
- Verified LAN-registry publications retain compact materialization records;
  missing, stale, or ambiguous legacy publication metadata produces a minimal
  project-local discard record instead of retaining a heavyweight context.

### Changed

- Unleased internal OCI layouts below `.posttrain/state/cache` are consistently
  treated as disposable transport cache. Explicit user-owned local exports
  remain outside that namespace and are never pruned.
- Historical migration validates project and package identity, refuses
  symlinked or leased contexts, never mutates Trackio, registry images,
  execution evidence, or another project, and leaves deletion to the existing
  dry-run-first cache pruner.

### Qualification

- Focused CLI and pack coverage proves dry-run behavior, verified compact
  records, fallback discard journals, active-lease protection, and exact
  project-local pruning.
- Ambient migration removed 54 historical contexts and 13 internal OCI
  layouts (155,025,370,976 logical bytes), increased free disk space from about
  136 GB to 300 GB, and preserved the aggregate checksum of every execution
  evidence file.

## 0.3.21 - 2026-08-23

This release separates routine terminal cleanup, rebuildable cache pruning,
durable evidence retention, and intentional cross-plane erasure into explicit
lifecycle contracts.

### Added

- Work packages can mark run evidence as `standard` or `pinned`; the resolved
  policy is retained through admission, submission, and tracking.
- Purge previews require a non-secret reason and produce digest-bound plans,
  retry-safe per-plane journals, and privacy-bounded machine-local tombstones.
- CLI audit views can include intentionally purged runs without restoring or
  exposing deleted metrics, traces, artifacts, or diagnostics.

### Changed

- Controller and manual reconciliation automatically release only the exact
  terminal execution workspace after durable evidence has settled.
- Run purge remains non-cascading by default; cascade and project purge use
  project-scoped ownership closure, protect shared consumers and pinned runs,
  and revalidate saved action targets before mutation.

### Qualification

- Focused lifecycle coverage proves project isolation, reference protection,
  retry-safe partial failure, minimal tombstones, and equivalent local/remote
  cleanup contracts.
- Disposable Ambient local and dstack qualifications proved cross-plane purge
  and automatic workspace cleanup while preserving retained Trackio evidence.

## 0.3.16 - Unreleased

This release completes the generic sampling contract for the maintained TRL
and veRL online-training backends and qualifies the exact fork artifacts on
real GPU workers before stable promotion.

### Added

- TRL IW-OPD and veRL rollout configuration now preserve non-default min-p,
  repetition-penalty, and presence-penalty controls through their actual
  runtime generation paths, with real configuration and backend regressions.
- Lab includes bounded 0.8B veRL and 0.8B-student/2B-teacher TRL qualification
  work packages whose resolved snapshots retain exact backend, sampling,
  dependency-lock, model, and target identities.
- Retained-fork publication now has explicit development qualification,
  byte-identical server-side stable promotion, and protected dev/stable
  runtime-lock materialization paths; maintained forks still release manually.

### Fixed

- The selected TRL post11 runtime counts IW-OPD loss items after on-policy
  generation, preventing prompt-only raw batches from dividing finite token
  loss by a zero pre-generation denominator.
- Concurrent actual-job image publishers serialize by immutable publication
  identity and reuse the producer's verified receipt instead of racing after a
  shared named build context is removed.
- The veRL runtime preserves nested three-dimensional position IDs during
  minibatch selection and records the complete selected sampling policy in
  retained evidence.
- Fork promotion uses the release runner's provisioned, hash-locked devpi
  client and private CA bundle instead of a user-local or network-installed
  client.

### Qualification

- TRL post11 completed one real IW-OPD optimizer update with finite loss,
  749 teacher-scored tokens, zero teacher failures, two native traces, and
  retained checkpoint, recovery, LoRA adapter, and summary artifacts.
- veRL dev2 completed one real optimizer step with two native traces and
  retained adapter, summary, retention, and trace-sync artifacts while
  resolving the exact non-default sampling controls. Its deliberately tiny
  trajectories truncated with zero reward variance, so the run qualifies the
  backend contract rather than training quality.

## 0.3.15 - 2026-08-12

### Fixed

- `posttrain run cleanup` can now release a terminal dstack workspace when the
  provider reports `failed` while Trackio has already finalized retained
  evidence as `cancelled` (or the reverse). The disagreement remains explicit
  in the reconciliation journal and cleanup receipt; cleanup still requires at
  least one retained output and records a bounded provider diagnostic.

## 0.3.14 - 2026-08-12

### Fixed

- Buildx now streams long-running Bake progress but keeps short image-metadata
  queries quiet. Packing or submitting a job no longer prints the complete OCI
  configuration while it verifies a digest-pinned runtime image.

## 0.3.13 - 2026-08-12

### Fixed

- Local OCI job export now grants Buildx write access only to its unique
  user-selected export parent. `posttrain job pack --local` therefore works
  with current Buildx filesystem-entitlement checks without disabling them
  globally.

## 0.3.12 - 2026-08-12

### Fixed

- Local composition now rejects an inference output budget larger than the
  selected engine context window.
- Online-RL work packages now require the declared inference completion budget
  to match training settings, and reject prompt-plus-completion budgets that
  cannot fit the rollout engine context. This turns an otherwise late vLLM
  configuration failure into an actionable pre-pack validation error.

## 0.3.11 - 2026-08-12

### Fixed

- Running GRPO and OLMo 3 jobs now publish the retained reward spread and
  zero-variance-group signals at each completed optimizer step.  Observatory
  can therefore distinguish a genuinely missing signal from a job that is
  still collecting rollouts; terminal trace replay remains the recovery path
  for interrupted jobs.

## 0.3.10 - 2026-08-12

This release makes OLMo 3 active sampling auditable as a distinct rollout
population, including older runs that emitted the native metrics before their
resolved selection snapshots contained the algorithm field.

### Added

- Observatory now shows active-sampling generation rounds, retained fraction,
  and the reserved, generated, retained, and unused candidate-row populations
  without mixing those counts with rollout outcome totals.
- OLMo 3 runs owe explicit active-sampling evidence, so a missing retention or
  candidate-window metric prevents the run from presenting as fully evidenced.

### Fixed

- Run snapshots retain the resolved GRPO algorithm, prompt-group shape,
  sampling mode, clipping, advantage scaling, and bounded dynamic/active
  sampling settings needed to interpret a training run independently.
- Per-candidate TRL rollout population metrics now carry their candidate scope
  and ordinal, so readers can distinguish refill waves from an optimizer step.

## 0.3.9 - Unreleased

This release prevents a veRL training selection from being packaged against a
different backend runtime than the one it declares.

### Fixed

- veRL job packing now requires the selected source repository, full commit,
  and dependency-lock digest to match the digest-pinned runtime image.
- Runtime-image verification checks the veRL provenance labels in the registry,
  not just the shared framework dependency lock.
- The immutable two-step veRL capsule is retained in Lab's qualification
  inventory as an explicit experimental candidate.

## 0.3.8 - 2026-08-12

This release makes terminal online-RL evidence observable while rollout work
is still running and advances the maintained veRL runtime used by Posttrain.

### Added

- Terminal Verifiers traces are retained before trainability checks, including
  harness errors, inference failures, unscorable attempts, and truncated
  completions.
- The veRL parent process tails native rollout evidence from its isolated
  worker, submits complete records through the configured observer, and retains
  a compact synchronization receipt on success or failure.
- Online-RL population evidence distinguishes requested rollouts, terminal
  traces, and attempts that ended without terminal evidence.

### Changed

- The maintained TRL runtime advances to `1.9.2.post2`, including raw
  sampler/actor parity checks for native-LoRA rollouts; public CI consumes the
  matching hash-verified release wheel.
- The `online-rl-verl-py313` runtime selects CarbonTeq veRL `0.9.0.dev1` at
  immutable source revision `a6fe39c22719ec981ed8544ad8feffd59995cc13`.
- Observatory presents requested-versus-terminal rollout coverage without
  treating missing evidence as a failed or zero-reward rollout.

### Fixed

- Execution failures are no longer classified as length truncation unless the
  native trace records an actual output boundary.
- Failed or unscorable traces no longer fabricate reward spread or enter
  advantage, active-sampling, clipping, entropy, or optimizer calculations.
- TRL failure finalization retains bridge-derived rollout counters instead of
  applying successful-batch metric exclusions.

## 0.3.6 - 2026-08-10

This release tightens local job lifecycle handling, selected-input packaging,
and online-RL evidence delivery.

### Added

- Bounded cache inspection, explanation, lease-aware pruning, compact package
  records, and receipt-backed image reuse for job packing.
- Completion-time rollout observation with asynchronous Trackio delivery while
  preserving crash-recoverable native trace artifacts.
- Job-local cleanup that removes disposable provider workspaces only after
  terminal evidence reconciliation.
- The published Reasoning Gym environment revision with bounded-reasoning
  termination guidance.

### Changed

- Job packing includes only datasets reachable from the selected job's resolved
  seats; unrelated project data is rejected before materialization.
- Live-streamed traces are not replayed during finalization, while aggregate
  trace-derived metrics still cover the complete local spool.
- Local daemon image tags and temporary build material have explicit ownership
  and terminal cleanup boundaries.

## 0.3.5 - 2026-08-09

This release carries checkpoint-scoped model artifacts and the validated
Trackio `0.31.5.post12` runtime through the same immutable Python and OCI
release inputs.

### Added

- Job-scoped checkpoint inspection, verification, selection, and model-artifact
  views for recovery, evaluation, and continuation across training jobs.
- Adapter-only model artifacts for LoRA and QLoRA runs, paired with complete
  recovery state for resumable training.

### Release integrity

- The published runtime lock and OCI manifest are committed and verified
  together, including the exact Trackio post12 wheel hash.

## 0.3.3 - 2026-08-09

This release strengthens online-RL correctness and makes the maintained
Trackio and TRL distributions reproducible inputs to Posttrain jobs.

### Added

- A named `algorithm: olmo3` GRPO recipe backed by TRL's
  `Olmo3GRPOConfig`, including zero-gradient filtering, bounded active refill,
  token-level loss normalization, asymmetric clipping, mean-only advantages,
  zero KL, and truncated importance sampling.
- Portable active-sampling settings and rollout telemetry for accepted,
  rejected, replacement, exhausted, and usable prompt groups.
- DAPO advantage diagnostics covering magnitude, sign balance, group reward
  spread, zero-variance groups, scoreability, truncation, and importance-ratio
  clamping.
- Deterministic runtime-lock materialization for internally published
  dependencies. Release candidates retain the generated lock together with the
  immutable OCI manifest before merge.

### Changed

- DAPO reward scaling is configurable instead of hardcoded. Truncated
  completions can be excluded from group statistics as well as from the loss,
  preventing masked samples from changing another completion's advantage.
- LoRA rollout synchronization and trainer configuration use the maintained
  TRL `1.9.2.post1` distribution rather than an ambient Git checkout.
- Trackio advances to `0.31.5.post11`, including the S3 artifact recovery path
  used by remote jobs and Observatory evidence readers.
- Runtime images consume the same exact Trackio and TRL wheel receipts as the
  Python workspace, while GitHub-only consumers retain hash-verified public
  release fallbacks.

### Release safety

- Pull-request validation distinguishes authored dependency changes from
  candidate-generated OCI evidence. The stale-image guard remains strict after
  materialization and for every merged/default-branch build.
- Candidate artifacts include both `workspace.lock.txt` and `published.toml`,
  so final distributions cannot silently package an earlier runtime image
  graph.

## 0.3.2 - 2026-08-05

This release adds the Gemma 4 dense support matrix and qualifies the TRL
paired-assistant MTP path on the RTX PRO dstack target.

### Added

- Immutable Gemma 4 E2B, E4B, 12B Unified, and 31B model variants with exact
  checkpoint provenance and shared family-level rendering.
- TRL MTP assistant validation and snapshot materialization for the Gemma 4
  12B GRPO path, including speculative acceptance and KV-cache evidence.
- Declarative Gemma serving, SFT, and TRL qualification work packages with
  tracked dstack evidence.

### Qualification

- E2B, E4B, and 31B serving smokes returned non-empty text on the RTX PRO
  target.
- The 12B TRL run completed two non-truncated rollouts and one optimizer step
  with reward 1, MTP acceptance 0.937888, and KV-cache metrics.
- The protected LAN release transaction published the final wheelhouse and
  completed the packed dstack canary before creating tag `v0.3.2`.

## 0.3.1 - 2026-08-05

This release makes evaluation meaning part of immutable run evidence, expands
the maintained Verifiers environment library, and completes the reviewable
cross-plane purge workflow introduced after 0.3.0.

### Added

- Explicit `run purge` and `project purge` planning across provider, OCI,
  tracking, and local state, with dependency closure, digest-bound previews,
  confirmation gates, resumable receipts, and retained verification evidence.
- Independently packaged GSM8K, AutomationBench, MMLU-Pro, IFEval, Reasoning
  Gym, and Math Python Verifiers environments with immutable source/data
  revisions, reproducible subset selection, and Lab qualification packages.
- Versioned evaluation contracts that snapshot the selected population,
  success predicate, reward and metric namespaces, task facets, and structured
  compound breakdowns such as problem type by difficulty.
- Evaluation-first Observatory Overview, Compare eligibility, performance
  distributions, schema-driven reward/verifier columns, pass/fail outcomes,
  and chat-style tool-aware trace inspection.
- Reproducible Python dataset materialization and package-owned serving
  workload definitions carried forward from the post-0.3.0 release branch.

### Changed

- Evaluation images install Verifiers, environment wheels, and runtime
  dependencies during image construction. Job startup only resolves the
  snapshotted configuration and executes the worker; it performs no package
  installation or upgrade.
- Tool-using environments declare portable inference capabilities. The Qwen
  renderer contract selects the compatible vLLM reasoning/tool parser while
  subprocess or MCP transport remains environment-owned.
- Project run listings exclude foreign admissions and successfully purged runs
  by default; `--include-purged` exposes labeled retained history for audit.
- Trackio advances to the maintained post8 lifecycle API, and affected runtime
  images resolve exclusively from the CarbonTeq OCI registry.
- Release publication now runs through the protected LAN runner, private
  `pypi.lan`/`registry.lan` channels, immutable candidate receipts, and a
  verified idle RTX PRO dstack canary; GHCR and public PyPI are not part of the
  release path.

### Qualification

- Real Qwen3.5-4B thinking evaluations qualified IFEval, Reasoning Gym, and the
  full 200-task AutomationBench Simple population with native MTP, retained
  Verifiers traces, and Observatory projections. The live Math Python schema-v3
  run additionally demonstrated the frozen 500-task population, configured
  success predicate, compound problem-type-by-difficulty reporting, subprocess
  Python tools, MTP, and concurrency eight. It remains active for terminal
  reconciliation and is not stopped by this release.
- The source validation ladder passes Ruff lint and format, Pyright, all eight
  import contracts, 1,030 Python tests with 18 expected skips, 32 Observatory
  frontend tests, and the production frontend build.

## 0.3.0 - 2026-08-01

This release starts the project-owned developer-experience redesign. Static
job meaning, execution configuration, catalog composition, environment source,
and image publication are now separated so projects can be planned and packed
without silently inheriting the submitting shell or a provider connection.

### Added

- Installable `posttrain-project` and `posttrain-environment` packages with
  public project discovery, provider-free job intents, execution-setting
  provenance, portable environment activation contracts, and project-path
  environment sources.
- Deterministic catalog-family discovery through entry points. Resolved family
  provenance is locked into package identity; duplicate, absent, or undeclared
  providers fail before catalog decoding.
- Local OCI job-image export, selected transitive catalog closure staging,
  project environment scaffolding, and declared dataset builders with
  input-sensitive cache identity.
- Runtime qualification for staged activation resources, taskset loading, and
  frozen JSONL datasets before an actual job image is published.
- Manifest-only release preparation, release-neutral workspace metadata,
  staged static wheel metadata, and a single generated catalog dependency-lock
  table through `posttrain-release`.
- Durable project/control and provider locators, a foreground lifecycle
  controller, joined run views, and safe state migration/cache classification.

### Changed

- Project-root `posttrain.env` is loaded automatically and authoritatively;
  ambient shell variables no longer override project runtime configuration.
- `posttrain job plan` reports provider-free job intent. Publication and
  launch settings are selected by `job pack` and `job run` respectively.
- Read-only `--last` resolution is strictly chronological. Mutating run
  commands require the complete canonical run id.
- Machine defaults can configure local-container DNS without placing machine
  topology in project configuration. Managed inference bindings carry a
  versioned startup budget that is retained in resolved run evidence.

### Release qualification

- An external consumer installed all 24 coordinated 0.3.0 framework wheels.
  The Lab data-preparation gate and managed Qwen 3.5 2B GSM8K evaluation both
  executed from packed immutable images, reconciled provider exit 0 against
  retained Trackio artifacts, and reported complete required telemetry.
- The bounded evaluation synchronized both native Verifiers traces: 2/2
  completed successfully with mean reward 1.0 and no failed or truncated
  rollouts. Its resolved evidence records the 600-second managed-inference
  startup budget used to cover cold model/kernel initialization.

## 0.2.5 - 2026-07-31

This patch hardens high-concurrency native Verifiers GRPO and makes
Observatory discover Trackio projects dynamically.

### Fixed

- Concurrent policy turns are batched and arrivals are drained safely while a
  trainer update holds the lock.
- The TRL backend exposes Liger loss compilation as an explicit, validated
  setting.
- Verifiers rollout groups execute concurrently, with a bounded compatibility
  patch for the current MCP harness dependency.

### Changed

- Observatory discovers available Trackio projects instead of relying on a
  fixed project list.

## 0.2.4 - 2026-07-30

The maintained veRL and vLLM runtime becomes a first-class published job kind.
This release also makes dstack placement durable and visible, and carries the
runtime fixes found while qualifying two-step DAPO, SAMPO, and distillation on
the 24 GB and 96 GB workers.

### Added

- Published `online-rl-verl-py313` runtime with immutable CarbonTeq veRL and
  vLLM fork revisions.
- Persistent dstack capacity waiting plus `posttrain run queue`, requested and
  assigned worker hostnames, and worker-capacity inspection.
- Two-step DAPO, SAMPO, and distillation qualification packages, including
  MTP-only variants retained as explicit selections.
- CUDA toolkit activation and vLLM compatibility checks needed by packaged
  remote jobs.

### Fixed

- veRL distillation aligns dense and jagged teacher log probabilities to the
  exact response tokens and safely ignores fully masked padding microbatches.
- Checkpoint-free terminal model export no longer fails retention after the
  disposable checkpoint root is removed.
- Parallel Verifiers harnesses no longer race while installing container
  prerequisites.
- Colocated Qwen 3.5 rollout uses eager vLLM execution, avoiding the observed
  CUDA-graph illegal-memory-access path.
- Actual-job BuildKit targets no longer race through a mutable named context.

### Qualification

- `verl-distill-shared-pool-retentionfix-20260730` completed two optimizer
  steps on `carbonteq-ai-workstation.lan`, retained 16 native Verifiers traces,
  the trained adapter, summary, and retention manifest, and reconciled dstack
  exit `0` with no missing required artifact roles.
- Baseline DAPO and SAMPO completed two optimizer steps; dstack also proved
  concurrent capacity-based placement across the RTX 4090 and RTX PRO 6000
  workers.

### Documentation

- Produce → pin → rebind how-to for trained model handoff between work
  packages
  ([getting-started §9](docs/getting-started.md#9-pass-one-jobs-model-into-the-next),
  [developer-experience](docs/developer-experience.md#trained-model-handoff-produce--pin--rebind),
  [tooling/trackio](docs/tooling/trackio/README.md#project-developers-artifact-handoff)).

## 0.2.3 - 2026-07-28

Pin Trackio to `0.31.5.post5` (`703be380…`) so import and distribution
versions match. Do not install `0.31.5.post4` from the index — that wheel is
skewed. Public PyPI Trusted Publishing is still unconfigured, so the Git pin
remains.

### Changed

- Trackio workspace, kind-profile, and constraint pins move to `703be380…`.
- Republished kind images against the refreshed workspace lock digest
  (`bedcf309…`).

## 0.2.2 - 2026-07-28

Machine-scoped local GPU admission, Observatory/Trackio listing performance,
public developer documentation, then a required runtime-image republish so
LAN digests match the Trackio workspace lock.

### Added

- Shared local GPU admission across projects on one machine (`posttrain
  workers`); dstack placement no longer takes a host lock inside posttrain.
- Soft affinity via catalog target `placement.instances: [{hostname: …}]`
  (optional; capacity-only placement remains the default).
- CLI DX: clearer job resolve / run-id / follow paths and reduced setup
  friction for `job plan|pack|run` and run inspection.
- Trackio `0.31.5.post4` (`dc55020d…`) with bulk `run_configs` /
  `run_lifecycles` so Observatory can list runs without an N+1 history fetch.

### Documentation

Public developer-facing guides (no private ops required to start):

- [getting-started](docs/getting-started.md) (formerly consumer-setup) —
  trust, index install, local and dstack providers, doctor, plan/pack/run,
  workers
- [developer-experience](docs/developer-experience.md) — project layout,
  catalog overlays, standard jobs, datasets/envs
- [tooling/dstack](docs/tooling/dstack/README.md) — client binding, soft
  affinity, placement vs local admission
- [tooling/trackio](docs/tooling/trackio/README.md) — fork pin and project
  artifact ownership boundary
- [contributing](docs/contributing.md) — framework checkout validation ladder
- [publishing](docs/publishing.md) — cutting a release without drifting images
- Trust as a **machine** property (well-known CA path), not project config;
  service ownership (`ai-infra` operates `.lan` services; this repo is the
  framework)

### Changed

- Observatory qualification uses available runs rather than four fixed named
  runs; Observatory image builds on the interpreter the app requires.
- `published.toml` pins `registry.lan/carbonteq` kind images to lock digest
  `c93d274e…`. `0.2.1` on the index still carried prior GHCR digests / lock
  hash, which the manifest loader correctly refused until this republish.
- Publish tooling streams Buildx progress and defaults to faster push
  settings.

### Note

Install `posttrain==0.2.2` from the internal index (or the GitHub wheelhouse
when attached). Runtime images must match this release’s lock digest;
`posttrain doctor` / `runtime images verify` report drift.

## 0.2.1 - 2026-07-28

The framework was qualified from a library consumer's seat for the first
time: installed from an index, with no checkout on the machine. Eleven
defects stood between that developer and a finished job, none of which the
test suite could see, because it runs from a checkout where the source tree
exists, every package is one version, and the build definitions are inside
the project.

First stable line after `v0.1.0-rc.2`. Requires **Python 3.13**.

### Added

- An internal package index as the supported distribution channel, with the
  maintained forks constrained explicitly because uv does not resolve a
  transitive direct URL implicitly.
- Release-pinned portable runtime images (base + job-kind variants) shipped
  as package data, with registry resolution and drift refusal.
- `posttrain job diff`, explaining why two packed job packages differ.
- A `trust` readiness check reporting which certificate authority reaches
  jobs, and warning when it is absent from the machine's own store.
- `docs/consumer-setup.md`, written from steps that were executed rather
  than imagined.
- Primary-CLI work-package execution through an explicit project host.
- A reproducible remote GPU release-gate workflow.
- Durable execution lifecycle, deterministic job packing, provider adapters,
  and job capsule CLI paths needed for pack/run without a checkout.

### Documentation

Public project-developer surfaces introduced with the consumer path:

- [getting-started](docs/getting-started.md) (formerly consumer-setup) —
  install from the internal index, trust the CA, run local or dstack jobs
  (executed steps, not aspirational)
- [install](docs/install.md) and
  [release-engineering](docs/release-engineering.md) (formerly
  release-and-consumption) — how releases are installed and gated
- [remote-gpu-qualification](docs/remote-gpu-qualification.md) — remote GPU
  release-gate workflow
- [UPGRADING](UPGRADING.md), [COMPATIBILITY](COMPATIBILITY.md),
  [SECURITY](SECURITY.md), Apache-2.0 [LICENSE](LICENSE)
- Frozen product baseline under [docs/post-training/](docs/post-training/README.md)
  (workflow → primitives → work/evidence → framework → APIs → observation)

### Changed

- Framework packages pin each other exactly. Declared by bare name, they
  allowed `posttrain` to be upgraded while every sibling stayed behind, which
  was individually satisfiable, matched no release, and was packed into job
  images as though coherent.
- A job image obtains framework code as built distributions when there is no
  checkout, rather than requiring twelve source directories to copy.
- Additional certificate authorities are merged with those the job image
  already trusts, and resolved from `/etc/posttrain/trust/internal-ca.pem`
  when nothing is configured. Execution providers no longer set
  `SSL_CERT_FILE`, which replaced the trust store rather than extending it.
- Runtime floor moved to Python 3.13; images and scaffolding follow.

### Fixed

- A run that died before opening a tracking run held its machine's admission
  placement permanently, and cancel, cleanup, and reconcile each refused it
  for a different reason.
- Tracking evidence was written to the configured server but looked for in a
  local one, so a succeeded run reconciled as pending with no artifacts.
- Writing `execution.toml` for the local provider's hostname discarded
  `POSTTRAIN_REGISTRY`.
- A failed provider submission reported only that its outcome was unresolved
  and to retry, naming nothing to act on.
- `release/github-constraints.txt` omitted `trl` and `verifiers`, so the
  documented install could not resolve.

### Note

Versions 0.1.1 through 0.1.13 were development builds published while
qualifying this work, because packing downloads framework wheels from the
index and a fix could not be tested until it was published. They are not
supported releases. Development builds now stage on a separate index.

## 0.1.0-rc.2 - 2026-07-24

Makes the framework usable from a normal project while adding explicit DAPO
and multi-turn SAMPO training contracts.

### Added

- `posttrain init` creates and installs a complete project with `.posttrain/`
  configuration, catalog overlays, work packages, and a project-local
  environment.
- Primary CLI owns dataset/environment materialization, work-package
  validation and execution, run inspection, and `posttrain observatory up`.
- Standard SFT, DPO, GRPO, DAPO, SAMPO, distillation, evaluation, serving, and
  model-transform jobs from `posttrain.jobs` without requiring `posttrain-lab`
  on the common path.
- DAPO as a first-class GRPO algorithm selection (TRL and veRL) with asymmetric
  clipping, token-level loss, and related bounded-sampling contracts.
- SAMPO as a separate multi-turn tool-agent operation with hierarchical
  episode/turn advantages.
- Developer environment profiles, remote GPU qualification guidance, and
  upgrade/compatibility documentation.

### Note

Real multi-turn SAMPO GPU qualification and larger GPU release gates remained
open; this prerelease did not claim production-qualified training quality for
those paths.

## 0.1.0-rc.1 - 2026-07-23

### Added

- Portable `.posttrain` project discovery, initialization, catalog overlays,
  work packages, and ignored machine-local state.
- The `posttrain` CLI for diagnostics, project inspection, catalog inspection,
  and composition validation.
- Versioned framework catalog resources and reusable data, train, evaluation,
  serving, tracking, and work-composition packages.
- Trackio and W&B tracking adapters with provider-neutral evidence contracts.
- Observatory Python, HTTP, MCP, report, and frontend surfaces.
- Installed-wheel consumer acceptance with real local Trackio persistence and
  Observatory readback.
- GitHub Release wheelhouses with immutable fork constraints and SHA-256
  checksums.

[Unreleased]: https://github.com/carbonteq-ai/posttrain/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/carbonteq-ai/posttrain/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/carbonteq-ai/posttrain/compare/v0.1.0-rc.2...v0.2.1
[0.1.0-rc.2]: https://github.com/carbonteq-ai/posttrain/compare/v0.1.0-rc.1...v0.1.0-rc.2
[0.1.0-rc.1]: https://github.com/carbonteq-ai/posttrain/releases/tag/v0.1.0-rc.1
