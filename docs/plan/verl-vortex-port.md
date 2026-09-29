# Port the VORTEX training recipe and SAMPO to the veRL backend

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Posttrain trains agents with two backends. TRL (Hugging Face's trainer, used
through the CarbonTeq fork) runs everything today. veRL (ByteDance's
distributed RL trainer, used through the CarbonTeq fork) scales further, but
Posttrain only lets it run plain GRPO, DAPO, GDPO, CAPO and distillation on the
Qwen 3.5 model family. The recipe the team actually trains with, VORTEX, and its
multi-turn successor SAMPO, are rejected by the veRL adapter or silently lose
parts of their behavior there.

After this plan a user can select the same VORTEX or SAMPO training settings,
switch `training.backend` from `trl@...` to `verl@...`, and get the same
algorithm: the same advantages, the same loss, the same bounded refill of
uninformative prompt groups, the same curriculum choosing which tasks to
sample, on Qwen 3.5 and on LFM2.5 models. Each phase proves that with a test
that feeds one fixed batch through both backends' real code and compares the
numbers, and the last phase proves it with two short runs of the same settings
on the local 8 GB GPU whose per-update metrics are compared.

VORTEX is Posttrain's name for this combination of settings (defined by
catalog entries `lfm2.5-2.6b/automationbench-vortex-20-local-v3` and
`lfm2.5-2.6b/automationbench-vortex-yield-first-64-150-lr5e-5-kl5e-3-local-v5`
in `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`):

- `algorithm: olmo3`, the published OLMo 3 RL objective: asymmetric PPO
  clipping of the per-token probability ratio at 0.2 below and 0.272 above,
  advantages that are the reward minus its group mean with no division by the
  group standard deviation (`advantage_scaling: none`), a per-token correction
  for the difference between the vLLM sampler's and the trainer's
  probabilities that is capped at 2 (token-level truncated importance sampling,
  "TIS"), and a loss averaged over every trainable token of the whole update
  ("token-mean", DAPO-style aggregation).
- `active_sampling`: generate the update's prompt groups, drop every group
  whose rewards are all equal (it carries no learning signal), and generate
  only the missing number of groups again, from a bounded pool of candidates.
  TRL fork 1.12.0.post11 adds `oversample` (extra groups in the first round)
  and `oversample_refill` (extra groups per refill round) with a guard that the
  largest round fits the rollout engine's and environment's concurrency.
- `adaptive_curriculum`: Posttrain's own controller
  (`packages/train/src/posttrain/train/adaptive_curriculum.py`) that chooses
  which tasks every update and every refill round samples, from the reward
  evidence already observed, and checkpoints its state with the model.
- `truncation_penalty`: subtract a constant from the reward of a rollout that
  hit a length, turn or context limit before group statistics, so a truncated
  attempt ranks below an equally scored finished one.
- `beta` with the reference model being the base model (a KL penalty, k3
  estimator, added to the per-token loss).

SAMPO (`SAMPOSettings`, operation `train.sampo`) adds per-turn advantages from
the environment's turn rewards on top of active sampling and the curriculum.

This work does not change the frozen product baseline. The canonical documents
already define these settings independently of the backend
(`docs/post-training/05-apis.md`, "GRPOSettings.algorithm selects ... Backend
adapters reject unsupported semantics rather than approximating") and already
require refill rounds to be separate curriculum decisions
(`docs/post-training/02-primitives.md`, adaptive curriculum section). The plan
adds backend support that meets those contracts; no product meaning changes.

## Progress

- [x] (2026-09-28) Read AGENTS.md, `docs/tooling/forks.md`,
  `docs/tooling/verl/README.md`, `docs/templates/PLAN.md`, the canonical
  GRPO/curriculum contracts, the TRL post11 loss and active-sampling code, the
  veRL post4 loss, rollout-correction and V1 replay-buffer code.
- [x] (2026-09-28) Worktrees created: veRL fork
  `/home/hammad/projects/verl-vortex` on `codex/vortex` from
  `carbonteq-v0.9.0.post4` (`54124edf`); Posttrain
  `/home/hammad/projects/rl-verl-vortex` on `codex/verl-vortex` from
  `c35da68e` (`codex/eval-train-budget`).
- [x] (2026-09-28) Plan written.
- [x] (2026-09-28) Phase 1 fork: `token_clip` loss and `k3_unclipped` KL,
  7 new CPU tests (41 with the core-algorithm and rollout-correction suites),
  ledger entry; commit `a4d84ad30b94c11c4de41b3d915eca6399ad2b6a` on
  `codex/vortex` (not pushed).
- [x] (2026-09-28) Phase 1 Posttrain: OLMo 3 objective mapping
  (`grpo_algorithm_payload`, `_olmo3_hydra_overrides`, fork-revision gate,
  protected overrides), shared reward shaping (`shape_rollout_reward`,
  `shaped_rollout_reward`), truncation penalty accepted for GRPO/DAPO on veRL,
  OLMo 3 still rejected pending Phase 2. `test_verl_backend.py` +
  `test_truncation_penalty.py`: 80 passed. Parity test passes (see Artifacts).
- [x] (2026-09-28) Phase 1 ladder (with `--extra trl --extra verifiers`):
  ruff, format, pyright 0 errors, lint-imports 9 kept, pytest 2061 passed / 25
  skipped (the parity test skips there because veRL is not installed), diff
  check clean. Phase 1 complete; OLMo 3 acceptance on veRL waits for Phase 2.
- [x] (2026-09-28) Coordinator request: Phase 1 fork delta released as
  candidate 0.9.0.post5 for release 0.4.12 (GDPO/CAPO crash on post4). Release
  commit `9fd6e7a31396ba33a29233cc869ab05b0a9e5a80`, annotated tag
  `carbonteq-v0.9.0.post5` (tag object `3945e01a`), receipt commit
  `9c10bd1a`; branch and tag pushed to carbonteq-ai/verl. Wheel
  `c16a2ad1...` (identical across two clean-clone builds), sdist `3c9e17c2...`
  (members identical), twine check passes; retained at
  `/home/hammad/verl-release/verl-post5/dist1/`. GitHub release and publish
  workflow left to the coordinator.
- [x] (2026-09-28) Posttrain: fork-native-name gate in the worker
  (`_FORK_NATIVE_NAME_REVISIONS`, `requested_fork_native_names`,
  `fork_native_names`), GDPO/CAPO regression test on post4, strict-xfail test
  against the pinned job-kind revision, installed-veRL record check.
- [x] (2026-09-28) Phase 2 fork: `algorithm.active_sampling` config block,
  `ActiveSamplingRounds` (TRL round arithmetic and metric names),
  `ActiveSamplingReplayBuffer` (round barrier, keep by sample std of
  `seq_reward`, evict the rest including failed groups, keep the first target
  groups in dispatch order), startup capacity guard, trainer wiring; commit
  `6c7295cd411c4d3973ddc206e43816560c842336` on `codex/vortex-active-sampling`
  (not pushed). 11 new CPU tests; fork V1/core suites 162 passed, 2 skipped.
- [x] (2026-09-28) Phase 2 Posttrain (branch `codex/verl-vortex-active-sampling`,
  based on `codex/release-0.4.12` at `d93c5f78`, which already contains Phase 1
  and the post5 pin): active sampling in the veRL contract, launcher and
  worker; OLMo 3 accepted by `backend_support` and gated on a fork revision
  that registers `active_sampling` (launcher and worker); plan-time capacity
  guard on veRL (launcher and `posttrain job plan`); veRL metric mapping for
  the `active_sampling/...` names. Side-by-side test against TRL post11's real
  `_prepare_active_sampling_inputs`: 19 cases pass (see Artifacts). Ladder:
  ruff, pyright 0, lint-imports 9 kept, pytest 2192 passed / 27 skipped /
  1 xfailed.
- [ ] Phase 2 GPU check (short 8 GB Qwen3.5-0.8B OLMo 3 run with a refill
  round). Blocked: needs a veRL build that contains `6c7295cd` (a post6
  candidate and a kind image built from it); the 0.4.12 kind image is post5.
- [x] (2026-09-28) Coordinator: the VORTEX port ships in release 0.4.13; no
  post6 until phases 2-5 are code-complete with CPU parity tests, then one
  post6 candidate collects them.
- [x] (2026-09-28) Phase 3 fork: `data.prompt_selector` extension point
  (`verl/trainer/ppo/v1/prompt_selector.py`, trainer dispatch/observe/
  checkpoint hooks, round numbers passed to the dispatcher); commit
  `24920b395f8571f8f5be6b9d8469737f2355dcc9` (not pushed). V1 suite 137
  passed, 2 skipped.
- [x] (2026-09-28) Phase 3 Posttrain: `AdaptiveCurriculumRuntime` moved to the
  backend-neutral `packages/train/src/posttrain/train/adaptive_curriculum_runtime.py`
  (re-exported by the TRL module); `backends/verl/curriculum.py`
  (`PosttrainCurriculumSelector`, JSONL event journal replayed by the parent,
  final snapshot copied from the last checkpoint); worker, contract and
  launcher mapping (selector config file, warm start, state artifact
  `adaptive-curriculum-state` with the TRL path's metadata); fork-revision
  gate for `prompt_selector`; veRL accepts the curriculum with GRPO and OLMo 3
  (DAPO stays TRL-only). Decision-level parity test against the TRL path
  passes (see Artifacts). Ladder: pyright 0, lint-imports 9 kept, pytest
  2200 passed.
- [x] (2026-09-28) Phase 3 addendum (coordinator: no deferral): per-checkpoint
  curriculum views on veRL. The selector copies each checkpoint's snapshot to
  `curriculum-checkpoints/step-<N>/`; the launcher publishes every view as
  `training/<model>/<technique>/checkpoint-<N>/curriculum`
  (`adaptive-curriculum-state`, role `checkpoint-curriculum`, metadata
  `checkpoint_step`), exactly like the TRL path, so
  `--curriculum-checkpoint-step N` selects it. CLI test
  `test_a_verl_run_checkpoint_curriculum_view_is_selectable_and_warm_starts`
  publishes views from a veRL selector, selects step 1, checks it holds the
  step-1 controller, and warm-starts a new selector from it.
- [ ] Phase 3 GPU check with the other phases (post6 image).
- [x] (2026-09-28) Coordinator: replace rejections with support. Fork commits
  `c55867dcfa6ca0716bccf57b492e2f4b6f7b0717` and
  `2607b91d3cccc9d73aae924734b5104bf8cfb590`: sampler-correction lower clamp and
  exact log ratios, TRL GRPO scaling (epsilon 1e-4, batch std, NaN-excluded rows
  with TRL's own `nanstd` arithmetic), row exclusion after advantages
  (`algorithm.exclude_flagged_rows`), NaN-aware DAPO/active filters, group
  admission retries with real-row loss normalization
  (`trainer.v1.sampler.failed_group_attempts`), linear LR schedule. Fork suites
  328 passed, 2 skipped. Posttrain maps all of them for GRPO, DAPO, OLMo 3 and
  SAMPO; GRPO and DAPO now also use `token_clip` and `k3_unclipped` (TRL's
  loss and KL). Remaining rejections, each with its reason in the error: DAPO
  plus curriculum and DAPO plus batch advantage scaling. Parity: 48 CPU parity
  tests pass (see Artifacts). Ladder: pyright 0, lint-imports 9 kept, pytest
  2205 passed.
- [x] (2026-09-28) Coordinator decisions: keep TRL semantics (no option for
  the old veRL behaviour), record the behaviour change in `CHANGELOG.md`
  (Unreleased, targeting 0.4.13 / post6) and mark runs with
  `verl_semantics: trl-parity-v1` in `grpo_runtime_resolved`; port TRL's
  candidate-batch DAPO so no rejection remains. Fork commits
  `ce8e0430018204b03c009b72bfba3b58968696c7` (`CandidateBatchReplayBuffer`,
  `algorithm.filter_groups.candidate_batches`, per-candidate-batch std for
  batch scaling, one prompt-selector decision per step's candidate pool) and
  `94606019` (retry test). Posttrain maps DAPO to candidate batches and
  `verl_grpo_settings_problem` rejects nothing. DAPO parity test passes (see
  Artifacts).
- [x] (2026-09-29) Phase 4 CPU parts. Fork commit `7d850ef5` (not pushed):
  CPU test exporting a PEFT `all-linear` adapter of a tiny `Lfm2ForCausalLM`
  through `verl.model_merger` (eight projection targets, tied `lm_head`
  dropped, identical logits on reload); no fork source change is needed.
  Posttrain: the launcher resolves the policy renderer with the TRL backend's
  `renderer_config_spec` (family config, template arguments, package chat
  template, tool-call protocol) into `VerlRenderer`; the veRL agent loop
  rebuilds it with `renderer_config_from_spec`, recovers LFM2.5 Python call
  lists through the protocol and reports bridged-turn message spans with the
  TRL backend's `bridged_message_spans`; `lfm2.5` joins the veRL launcher's
  families (see Decision Log).
- [ ] Phase 4 GPU: two LFM2.5-1.2B updates on veRL on the 8 GB card, with
  the post6 image.
- [x] (2026-09-28) Phase 5 fork: `sequence_clip` policy loss (TRL's
  sequence-level ratio) and SAMPO hierarchy evidence metrics; commits
  `d344b545` and `4d37a18bc492f0f4f9c224285603740ef4a2ba54` (not pushed); PPO
  CPU suites 308 passed, 2 skipped.
- [x] (2026-09-28) Phase 5 Posttrain: `build_sampo_launch_plan` maps SAMPO
  (`sampo_algorithm_payload`): the fork SAMPO estimator, `sequence_clip`,
  `seq-mean-token-mean`, clip 0.003/0.004, `k3_unclipped` KL, token or sequence
  sampler correction truncated at the selected cap, round-based active
  sampling with oversampling, optional curriculum, truncation penalty. Rejects
  settings veRL cannot reproduce (`mask_truncated_completions`,
  `max_admission_attempts` other than 1, a lower correction bound or mask
  modes). SAMPO groups are keyed by the native prompt occurrence (`uid`).
  Job plan checks SAMPO oversampling capacity on veRL. SAMPO metric names map
  to the TRL path's hierarchy evidence. Parity test passes (see Artifacts).
- [ ] Phase 6: end-to-end TRL/veRL parity runs on the 8 GB GPU.
- [x] (2026-09-29) Fork release candidate `0.9.0.post6` prepared, not
  pushed: release commit `1badbebd22aee7af5b85185760275f697af3a073` (version
  and ledger), local annotated tag `carbonteq-v0.9.0.post6` (tag object
  `3a7eb765060cf2d691114cb8cfefd2c5a84da180`), receipt commit
  `1cd7702f6b2f6aadfa149ec261e277552178eb5a`. Two clean-clone builds with
  `SOURCE_DATE_EPOCH=1790624464`: byte-identical wheel
  `63efd613e15f0011e840fadbb67e7d053353aa7b31437c84555e0261331faee5`, sdist
  `c7a00ceb1858011799ec22f8d82d66f07632e1eb4ecb26e95bb207c4e1abeb1a` (second
  build's sdist differs in archive bytes only; identical members); `twine
  check` passes; assets in `/home/hammad/verl-release/verl-post6/dist1/`.
  Fork CPU suite: 348 passed, 2 skipped. Posttrain records the three post6
  commits in `_FORK_NATIVE_NAME_REVISIONS`; the fork-native-name test now
  checks every post6 name against the installed fork.
- [x] (2026-09-29) The coordinator pushed the fork branch and tag and
  published post6 (GitHub release `carbonteq-v0.9.0.post6`, Posttrain publish
  run 36474625071; `carbonteq/dev` serves both hashes).
- [x] (2026-09-29) Posttrain post6 pin commit. veRL is not a dependency of
  `packages/train` or the root `uv.lock` (it lives only in the isolated veRL
  kind), so the pin is the kind's release project, as for post5:
  `verl-py313/release/pyproject.toml` selects `1badbebd`, `uv lock --python
  3.13.12` changes only veRL (lock `680342f0...`, constraints `760fe709...`),
  `profile.toml`, `release/forks.toml`, the release tests, the veRL precision
  and adapter-continuation bindings follow, and the olmo3 strict xfail is
  removed. `posttrain-release images plan --registry registry.lan/carbonteq`
  (LAN trust bundle): base and every other kind reused remotely, only
  `kinds.online-rl-verl-py313` rebuilds; not blocked. Plan output saved in the
  session scratchpad (`images-plan-0.4.13-dev.json`); not published.
- [x] (2026-09-29) The coordinator published the 0.4.13.dev1 veRL kind
  image (`sha256:37d284be...f994`, post6) and committed `published.toml`
  (strict `posttrain-release check` passes). The span fix `b5b30cc3` merged
  into 0.4.12; this branch is rebased onto it.
- [x] (2026-09-29) Phase 4 and Phase 6 packages committed
  (`lfm12_verl_automationbench_check_local.yaml` and the eight
  `*_automationbench_*_parity_local.yaml`, catalog `verl-vortex-parity.yaml`,
  qualification gates registered); all plan cleanly.
- [x] (2026-09-29) Phase 4 GPU attempts on the local card, from a clean
  detached worktree (`../rl-verl-vortex-run`):
  `verl-vortex-lfm12-check-20260929-r1` and `-r2` failed at vLLM start
  (`/dev/shm` 64 MiB, see Surprises; fixed by execution-local `520131a9`,
  branch `codex/local-docker-shm`, cherry-picked here as `40491337`);
  `-r3` passed vLLM start and failed at the first LoRA sync to vLLM (stacked
  LoRA names, see Surprises; cancelled at 20:53 UTC when the Ray driver hung).
  Each run carries notes.
- [x] (2026-09-29) Fork fix `f74e84e4` on `codex/vortex-lora-sync`
  (`lora_weights_mapper`: vLLM's rename-only mapper for synced LoRA tensors)
  with a CPU regression test run in the kind image and a GPU check (below).
  Release candidate `0.9.0.post7` prepared, not pushed: release commit
  `6069abe14e2b3d27c89815a6502b849f15124e12`, local tag
  `carbonteq-v0.9.0.post7` (tag object `b60890dc...`), receipt `07ecac23`;
  wheel `9786ec44fbdba367791d8e9a4c58a895955639c79c4b9dd0e3a403a05b5e29c5`
  (byte-identical across two clean builds), sdist
  `3b0259523476d67aff9282b2b3c775c081bd866c04d369ad7a5eda8af6607088`; assets in
  `/home/hammad/verl-release/verl-post7/dist1/`.
- [x] (2026-09-29) The coordinator published post7 (GitHub release
  `carbonteq-v0.9.0.post7`, Posttrain publish run 36483660287; the index
  serves both hashes). Post7 pin commit on this branch: the kind's release
  project selects `6069abe1`, `uv lock --python 3.13.12` changes only veRL
  (lock `a9562cb1...`, constraints `8562363a...`), `profile.toml`,
  `release/forks.toml`, the release tests and every veRL binding (precision,
  adapter continuation, Phase 4 check, Phase 6) follow.
- [x] (2026-09-29) The coordinator published the 0.4.13.dev2 veRL kind image
  (post7) and committed `published.toml` as `b8f39356`; the remaining runs use
  a clean detached worktree of it (`../rl-verl-vortex-run2`).
- [x] (2026-09-29) Qwen Phase 6 TRL twins succeeded:
  `verl-vortex-p6-qwen08b-trl-vortex-r1` and
  `verl-vortex-p6-qwen08b-trl-sampo-r1` (worktree `ab929f9d`).
  `verl-vortex-p6-qwen08b-verl-vortex-r1` failed at vLLM start: it was queued
  and started by the long-running `posttrain controller` of another checkout
  (`/home/hammad/projects/rl-controller`), which lacks the `/dev/shm` fix.
  `verl-vortex-lfm12-check-20260929-r4` landed in the queue the same way and
  was cancelled before start. Runs are now submitted by a serial driver
  (session scratchpad `verl-vortex/runs/serial_driver.sh`) only when the card
  and the local admission queue are idle, so this checkout starts each
  container itself; a submission that still lands in the queue is cancelled
  and retried under the next attempt id.
- [x] (2026-09-29) Paused at the coordinator's request for the release-0.4.12
  veRL checks; no run of this plan was active.
- [x] (2026-09-29) Rebased onto `origin/codex/release-0.4.12` at `82e20d81`,
  which pins veRL 0.9.0.post8 (`ef1c3771` = post7 plus agent-loop config
  defaults). Dropped this branch's post6 and post7 pin and image commits (the
  release branch pins post8 and ships its image); kept every port, lab and
  documentation commit. `_FORK_NATIVE_NAME_REVISIONS` keeps the release
  branch's post7 and post8 entries with the full post6 name set; post8's
  development commit `bbe090b8` is not recorded because no binding or image
  selects it. The Phase 4 and Phase 6 veRL bindings pin post8 (`ef1c3771`,
  backend lock `a8391c4e`, later `cdd614f2`). The Phase 6 environment is AutomationBench
  (`partial_credit`), not a math-verify scorer, so the Verifiers
  off-main-thread `signal.alarm` bug found by the 0.4.12 checks does not
  affect these runs; none of this plan's veRL GPU runs trained (all failed at
  start or the first sync). Validated on post8: full ladder, and the seven
  parity files (49 passed) and the fork CPU suite (348 passed, 2 skipped)
  against the post8 release commit.
- [x] (2026-09-29) Rebased onto `codex/release-0.4.12` at `602725cf`
  (Verifiers `e6a3d9bb`, which scores boxed math off the main thread; veRL kind
  lock `cdd614f2...`, catalog dependency lock `3806d424...`). The Phase 4/6
  bindings record the new locks, and the 8 GB veRL bindings take the 0.4.12
  host right-sizing (`00f91a9a` on `codex/verl-8gb-ray-workers`):
  `reward.num_workers=2` and
  `transfer_queue.backend.SimpleStorage.num_data_storage_units=2`, because
  veRL's default 8 reward workers plus 8 TransferQueue storage units took about
  7.4 GB and Ray killed a run at 95% of the 62 GB host's RAM.
- [x] (2026-09-29) 0.4.13 integration: `codex/release-0.4.13` from
  `origin/main` `6d07ce75` (0.4.12 merged); this branch rebased onto it (the
  Phase 6 environment repinned to the pin-only `11f4d712`; `train/rl/kl` now
  maps `actor/kl_loss`, `07bc0329`) and merged; TRL pinned to
  1.12.0.post12 (`c4d0db05`; catalog lock `f18308d1...`); the fp16 plan
  commits `395241a8` and `b78f88e0` cherry-picked (`a944a518`/`20b0b577` did
  not apply cleanly together and were left out). Workstation twins of the four
  LFM2.5 Phase 6 packages (`lfm12_*_automationbench_*_parity_ws.yaml`) differ
  only in target (and no veRL CPU offload). The Qwen TRL twins ran on post11
  with the pre-repin environment `5264ec15`/Verifiers `cdd2ec76`; post12 and the
  pin-only repin change nothing for bf16 AutomationBench, so they stay the
  twins of the post8 veRL runs.
- [ ] GPU, when the coordinator releases the local card: Phase 4 check
  `-r5`, Qwen veRL VORTEX `-r2` and SAMPO `-r1`, then the four LFM2.5 Phase 6
  runs (only if the check succeeds); all on post8 from a clean detached
  worktree of the branch head.
- [x] (2026-09-29) `codex/verl-vortex-active-sampling` rebased onto
  `origin/codex/release-0.4.12` at `36932821` (one test-file conflict, both
  sides kept); full ladder and the seven parity files pass after the rebase.
  Rebase again onto the final 0.4.12 before merging for 0.4.13.

## Surprises & Discoveries

- Observation: every veRL run on the local Docker provider failed at vLLM
  engine start: `Insufficient space in /dev/shm for shared-memory allocation:
  160 MiB required, 64 MiB free`. The provider ran containers with Docker's
  default 64 MiB `/dev/shm`; vLLM 0.29.1.dev4's multiprocess executor allocates
  a 160 MiB broadcast queue (10 chunks of `VLLM_MQ_MAX_CHUNK_BYTES_MB=16`) and
  checks the free space. It also stopped the 0.4.12 veRL qualification
  (`q0412d-verl-qwen08b-bf16-r1`). A queued run is dispatched by whichever
  checkout's `posttrain` process drains the local queue, so `-r2` still ran
  without the fix.
  Evidence: runs `verl-vortex-lfm12-check-20260929-r1`, `-r2` (notes).
- Observation: veRL's in-memory LoRA sync (`VLLMHijack._load_adapter`) named
  LoRA modules with the full `hf_to_vllm_mapper`; its stacked maps rename
  `w1`/`w3` to `w13` and `q_proj`/`k_proj`/`v_proj` to `qkv_proj`, so the
  constituents collapsed and vLLM's merged column layer raised `IndexError:
  tuple index out of range` in `set_lora`. vLLM's own loader uses
  `get_rename_mapper()`. Qwen3.5 veRL runs never hit it: their LoRA targets
  only `o_proj`/`down_proj`. TRL was unaffected because it reloads the adapter
  from a PEFT directory through vLLM's own loader.
  Evidence: run `verl-vortex-lfm12-check-20260929-r3`; GPU check with the fix
  (RTX 3070 Ti, post6 kind image with the fork source mounted): an
  LFM2.5-1.2B all-linear adapter synced as tensors scores a 29-token text
  within 0.055 nats/token (max 0.30) of PEFT, against vLLM's base-model gap of
  0.040 and an adapter effect of 1.03 nats/token.

- Observation: veRL's DAPO dynamic sampling (streaming refill, two credits per
  filtered group, bounded by candidate prompts) is not TRL's DAPO dynamic
  sampling (whole candidate batches of the target size until filled). This
  predates the port and is the reason for the two remaining DAPO rejections;
  DAPO itself stays accepted on veRL as before.
- Observation: TRL's `nanstd` computes Bessel's factor as `count / (count - 1)`
  on integer tensors, which rounds it to float32 even for float64 rewards, so
  veRL's exact `torch.std` differed from TRL by about 1.5e-8 relative. The
  fork's TRL-mode statistics repeat TRL's operations; advantages now match
  bitwise.
- Observation: TRL rejects `*_truncate` correction with neither bound set
  (`GRPOConfig` raises), although `GRPOSettings` allows it; the veRL mapping
  accepts it as uncorrected weights. Not tested as parity because TRL cannot
  run it.
- Observation: TRL stores `admission_loss_scale` as a float32 tensor, so a
  partial admitted batch's loss agrees with veRL's real-row normalization to
  float32 precision (checked at 1e-7 relative).

- Observation: veRL's `gspo` loss is not TRL's sequence-level objective when
  advantages vary inside a row. GSPO's stop-gradient token form gives token t
  the gradient `A_t * w_t * s_i / |y|`; TRL's `importance_sampling_level=
  "sequence"` differentiates through the mean log ratio, giving every token
  `s_i * mean_t(A_t * w_t) / |y|`. SAMPO's per-turn advantages differ within a
  row, so the port adds the fork loss `sequence_clip`. Evidence: the SAMPO
  parity test fails with `gspo` or `vanilla` substituted and passes with
  `sequence_clip`.
- Observation: the Phase 2 active-sampling parity test was not re-run after
  Phase 3 changed the dispatcher signature (`dispatch(count, round_index=...)`)
  and failed with a TypeError until its fake dispatcher was updated during
  Phase 5. All five parity files now run together (24 passed).

- Observation: the veRL GRPO path silently ignores several GRPO settings.
  `advantage_scaling`, `importance_sampling_mode` and its bounds are never
  passed to veRL, so a veRL GRPO run always uses group-std scaling and no
  sampler correction. Branch `codex/release-0.4.12` (commit `af5684ed`, module
  `packages/train/src/posttrain/train/backend_support.py`) now rejects
  non-default values. This plan maps them for OLMo 3 only, where the recipe
  fixes them, and leaves the other algorithms under that rejection.
  Evidence: `packages/train/src/posttrain/train/backends/verl/worker.py`
  `build_hydra_overrides` sets no `rollout_correction` or
  `norm_adv_by_std_in_grpo` key.
- Observation: veRL's built-in PPO loss (`vanilla`) is not the OLMo 3 loss. It
  applies dual clipping (for a negative advantage the loss is capped at
  `-A * clip_ratio_c`, default 3) and clamps the log ratio to [-20, 20]. Its
  KL loss `low_var_kl` clamps the k3 estimate to [-10, 10]. TRL post11's OLMo 3
  path has neither. The fork needs a plain token-clip loss and an unclipped k3
  KL; the unpublished GDPO/CAPO candidate (`/home/hammad/projects/verl-gdpo-capo`,
  uncommitted) already defines both under the names `token_clip` and
  `k3_unclipped`, and Posttrain's worker already emits `k3_unclipped` for GDPO
  and CAPO although post4 does not contain it.
  Evidence: `verl/trainer/ppo/core_algos.py` `compute_policy_loss_vanilla` and
  `kl_penalty_forward` at `54124edf`.
- Observation: veRL's V1 trainer already has the other OLMo 3 pieces:
  token-mean aggregation normalized by the mini-batch's global token count
  (`FSDPEngine` all-reduces `batch_num_tokens`), decoupled rollout correction
  (`algorithm.rollout_correction.rollout_is=token`, `rollout_is_threshold=2.0`
  truncates `exp(old_logp - rollout_logp)` at 2 and detaches it), and GRPO
  advantages without std scaling (`algorithm.norm_adv_by_std_in_grpo=false`).
- Observation: veRL's own group filter (DAPO "filter_groups" in
  `verl/trainer/ppo/v1/replay_buffer.py`) streams: each evicted group adds two
  refill credits and new prompts start while earlier ones still run. TRL's
  active sampling works in rounds: generate, filter, then request exactly the
  missing groups (plus the optional refill oversample). The canonical
  curriculum contract makes each refill round a separate curriculum decision
  informed by earlier rounds, so the veRL port must be round-based.
- Observation: Posttrain's veRL worker already selected `token_clip` and
  `k3_unclipped` for GDPO and CAPO, but the selected post4 release does not
  register either name, so GDPO/CAPO on post4 would fail at the first actor
  update. The Phase 1 fork commit makes both names exist.
  Evidence: `git -C /home/hammad/projects/verl-vortex grep -c token_clip 54124edf -- verl` finds none.
- Observation: the first parity batch did not reach the dual-clip region or
  the KL clamp, so replacing `token_clip` with `vanilla` or `k3_unclipped`
  with `low_var_kl` still passed. The batch now forces ratios above 3 on
  negative-advantage tokens and a reference 3.2 nats above the policy, and the
  test asserts those regions are present. A mutation script (scratch
  `mutation_check.py`) confirms each of these substitutions now fails: vanilla
  loss, clamped KL, clip-high 0.28, seq-mean-token-mean aggregation, std
  advantage scaling, correction cap 3.
- Observation: the local veRL GPU path is not yet runnable through
  `posttrain job run`: the published veRL kind image cannot package a Verifiers
  environment (recorded in `docs/plan/fp16-training-precision.md` on branch
  `codex/precision-fp16-verl`). Phases 4 and 6 will run the isolated worker
  directly from a locally built veRL environment (see Phase 4) unless that
  image is rebuilt first.
  Update (2026-09-29): release 0.4.12 rebuilt the veRL kind image (post5,
  digest `sha256:523b5525...`) and its qualification runs veRL packages with
  `posttrain job run --provider local`. A kind image must be in a registry
  for packing, so the Phase 4 and Phase 6 GPU runs wait for the post6 image
  the coordinator builds, instead of a local-only image.

- Observation: the veRL agent loop did not render LFM2.5 as TRL does. It
  chose `DefaultRendererConfig` for every non-Qwen family (TRL chooses
  `LFM25RendererConfig`), kept the tokenizer's chat template (TRL installs the
  model's package template `lfm25_tool_chat.jinja`), passed no tool-call
  protocol to `parsed_policy_message` (TRL recovers LFM2.5's
  `<|tool_call_start|>[call(...)]<|tool_call_end|>` lists through it), and
  reported a bridged turn's message spans relative to the new messages only
  (TRL offsets them and pads the earlier messages with `None`). The last one
  also affected Qwen 3.5 multi-turn runs.
  Evidence: `packages/train/src/posttrain/train/backends/verl/agent_loop.py`
  before this phase; tests
  `test_verl_policy_generator_recovers_lfm25_python_calls_like_trl` and
  `test_verl_resolves_the_lfm25_renderer_exactly_as_trl`.
- Observation: the local image `posttrain-kind-online-rl-verl-py313:qwen-kernels-local-cc1d`
  (post3) has upstream `renderers` 0.1.12.dev3 installed over the CarbonTeq
  fork, so `LFM25RendererConfig` is missing there. The 0.4.12 veRL kind lock
  selects only `carbonteq-renderers`; do not use that local image for LFM2.5.

- Observation: the port sized veRL's response budget (`data.max_response_length`,
  `rollout.response_length`) as `max_completion_length`, the per-reply token
  cap. A multi-turn Verifiers response is every turn after the first prompt,
  and TRL trains all of it, bounded only by the rollout context. On veRL the
  agent loop rejected longer responses, the session raised, the group was
  marked failed and admission dropped it, so long successful episodes were
  lost and a batch of them ended the run.
  Evidence: `verl-vortex-lfm12-check-20260929-r5` (two completed LFM2.5
  episodes, reward 1.0, 1832 and 1420 sampled tokens over two replies; `rollout
  admission retained no complete groups`). The same run shows the LFM2.5
  renderer and tool-call recovery working on veRL.

- Observation: with the adaptive curriculum, a veRL run failed after training:
  the launcher replayed all trainer metrics (steps 1 to 4) and then the
  curriculum journal from step 1, and Trackio rejects a decreasing step
  (`logical metric steps must be nondecreasing`). The CPU tests used a context
  without that rule. The launcher now interleaves the journal with the trainer
  metrics by step (`CurriculumJournalReplay`).
  Evidence: `verl-vortex-p6-lfm12-verl-vortex-ws-r3`;
  `test_verl_curriculum_journal_interleaves_with_trainer_metrics_in_step_order`.
- Observation: LFM2.5-1.2B ran out of groups with reward spread in a 16-task
  pool (4 candidate batches) before filling an update, which TRL also treats as
  an error (`active sampling exhausted ... generation rounds`). The LFM2.5
  comparison settings v2 use all 24 tasks (6 candidate batches).
  Evidence: `verl-vortex-p6-lfm12-verl-vortex-ws-r2`,
  `verl-vortex-p6-lfm12-verl-sampo-ws-r1`.
- Observation: even all 24 tasks (6 candidate batches) did not reliably give
  LFM2.5-1.2B four groups with reward spread (`verl-vortex-p6-lfm12-verl-vortex-ws-r4`
  exhausted its pool; `-ws-r3` with the same settings filled four updates). The
  LFM2.5 pairs use a separate 57-task set (the tasks of
  `automationbench-lfm12-sampo-8gb-v2`, screened because LFM2.5-1.2B's attempts
  disagree) with the Qwen set's budgets and VORTEX's 10 candidate batches
  (settings v3). The job plan held only TRL to "the pool may not exceed the
  environment"; veRL is now held to it too (veRL's bounded pool would run into
  the next epoch and repeat tasks within an update).
- Observation: Phase 4 passed. `verl-vortex-lfm12-check-20260929-r6` (veRL
  post8, LFM2.5-1.2B, two updates, local 8 GB card) succeeded: rendering and
  tool-call recovery, LoRA sync to vLLM after update 1, checkpoint and export.

## Decision Log

- Decision: veRL's response budget is the rollout context (`max_model_len`,
  default prompt plus completion); `max_completion_length` stays the
  per-reply cap (vLLM `max_tokens`) and the truncation boundary, as on TRL.
  Rationale: veRL pads every response to a fixed length and cannot accept a
  longer one, and any trajectory TRL trains fits within the context. The cost
  is padding: single-turn runs now pad responses to the context length.
  Date/Author: 2026-09-29, Claude.

- Decision: `lfm2.5` joins the veRL launcher's qualified families in the
  Phase 4 code change, before its GPU run.
  Rationale: the GPU qualification runs through the launcher, which rejects
  unlisted families, and the branch ships only in 0.4.13 after the Phase 4
  and Phase 6 GPU runs; the README table marks LFM2.5 as pending until then.
  Date/Author: 2026-09-29, Claude.
- Decision: resolve the renderer on the launcher and pass it to the agent
  loop as data, rather than duplicating TRL's family table in the agent loop.
  Rationale: one function (`renderer_config_spec`) now decides the renderer
  for both backends, so they cannot drift again; the control environment does
  not need `renderers` installed because the spec is plain data.
  Date/Author: 2026-09-29, Claude.

- Decision: DAPO on veRL uses TRL's candidate-batch dynamic sampling, and the
  curriculum makes one `initial_batch` decision for the whole candidate pool
  per update, not one per candidate batch.
  Rationale: that is what the TRL backend does (`AdaptiveCurriculumTrainer`
  selects the dynamic-sampling generation batch once; only active sampling
  consults the controller per round), and parity is the requirement. The
  streaming veRL refill remains available in the fork when the flag is off.
  Date/Author: 2026-09-28, Claude.

- Decision: veRL GRPO and DAPO now run TRL's objective in full: `token_clip`
  (no dual clip), `k3_unclipped`, the selected sampler correction (the GRPO
  default is sequence-truncated to [0.1, 3.0], which veRL previously did not
  apply at all), TRL's advantage scaling (std + 1e-4, not veRL's 1e-6), TRL's
  group admission (retry the same prompt, then drop) instead of refilling
  failed groups with new prompts.
  Rationale: the settings always declared these semantics; veRL runs silently
  used different ones. Recorded veRL GRPO runs therefore do not match their
  settings, and new runs will differ numerically from them. Every mapping has
  a parity test against TRL's real code.
  Date/Author: 2026-09-28, Claude.
- Decision (superseded 2026-09-28 by the coordinator's request to remove every
  rejection; both are gone after the candidate-batch DAPO port): keep exactly two rejections. (1) `adaptive_curriculum` with DAPO:
  veRL's DAPO refill streams single prompts with no per-round decision point,
  while the curriculum contract requires decisions per refill round. (2)
  `advantage_scaling: batch` with DAPO dynamic sampling: TRL divides by the std
  of each candidate batch before filtering; veRL's streaming refill has no
  candidate batch with that population. Both reasons are in the error
  messages (`packages/train/src/posttrain/train/backend_support.py`). Porting
  TRL's candidate-batch DAPO refill to veRL would remove both and is noted
  under Surprises as a pre-existing difference.
  Date/Author: 2026-09-28, Claude.
- Decision: under active sampling (OLMo 3, SAMPO) `max_admission_attempts` is
  accepted and behaves as one attempt, because TRL forces one admission attempt
  when active sampling refills groups (`max_attempts=1 if active_sampling` in
  `backends/trl/policy_rollouts.py`).
  Date/Author: 2026-09-28, Claude.

- Decision: SAMPO on veRL uses the new `sequence_clip` loss, not `gspo`.
  Rationale: the recipe is defined by the TRL path (sequence-level ratio with
  gradient through its mean); GSPO's token form optimizes a different
  gradient when per-turn advantages vary. No veRL SAMPO selection was
  runnable before this phase (the launcher rejected SAMPO), so no recorded run
  changes meaning.
  Date/Author: 2026-09-28, Claude.

- Decision: the curriculum on veRL reuses the TRL path's
  `AdaptiveCurriculumRuntime` inside a fork prompt selector, instead of
  re-implementing decisions in the fork or in the parent process.
  Rationale: the canonical contract requires every refill round to be a
  decision informed by earlier rounds at fixed weights, which only the process
  that dispatches rounds can do; reusing the runtime makes decisions, events
  and snapshots identical by construction. The veRL trainer process has no
  `RunContext`, so the runtime writes events and metrics to
  `verl-curriculum-events.jsonl`, which the parent replays after the worker
  exits (as it already does for rollout rewards).
  Date/Author: 2026-09-28, Claude.
- Decision: keep DAPO plus curriculum TRL-only on veRL.
  Rationale: veRL's DAPO refill streams from the dataloader without decision
  boundaries; OLMo 3 active sampling is the recipe that needs the curriculum.
  Date/Author: 2026-09-28, Claude.

- Decision: base the Phase 2 Posttrain branch on `codex/release-0.4.12`
  (`d93c5f78`) instead of `54671c33`.
  Rationale: release 0.4.12 already merged Phase 1 and pinned post5, and it
  carries the TRL post11 `oversample`/`oversample_refill` settings, the
  concurrency guard and the veRL rejection module this phase edits; basing on
  the older commit would re-implement them.
  Date/Author: 2026-09-28, Claude.
- Decision: failed prompt groups count as generated but not retained in veRL
  active sampling, and a group is kept when the sample standard deviation
  (ddof 1) of `seq_reward` exceeds the epsilon (0 for OLMo 3).
  Rationale: TRL drops groups the environment could not admit and filters on
  `nanstd` of the shaped rewards; with epsilon 0 the two spread measures agree.
  Date/Author: 2026-09-28, Claude.
- Decision: record fork versions only for release commits in
  `_FORK_NATIVE_NAME_REVISIONS`; development commits carry `None`.
  Rationale: a development commit shares its parent release's version string
  without its content, so a version lookup would claim names the release lacks.
  Date/Author: 2026-09-28, Claude.

- Decision: gate fork-only native names (`token_clip`, `k3_unclipped`) by an
  explicit per-commit record in the worker and reject clean checkouts at other
  revisions; do not gate dirty candidate checkouts (`source_dirty: true`).
  Rationale: GDPO/CAPO on post3/post4 otherwise crash inside veRL after model
  loading; dirty candidates (such as the GDPO/CAPO qualification worktree) are
  identified by content digest and may register the names without a release.
  The pinned-revision test is a strict xfail restricted to `AssertionError`, so
  it cannot hide a broken test and flips to a failure when the pin moves.
  Date/Author: 2026-09-28, Claude.

- Decision: implement the OLMo 3 loss on veRL with a new registered policy
  loss `token_clip` and KL type `k3_unclipped` in the fork, not with
  `vanilla` plus a very large `clip_ratio_c`.
  Rationale: the canonical APIs require adapters to reject rather than
  approximate. `token_clip`/`k3_unclipped` reproduce TRL's formula exactly
  (no dual clip, no log-ratio clamp, no KL clamp) and use the same names as
  the unpublished GDPO/CAPO fork candidate, so the two deltas merge into one.
  Date/Author: 2026-09-28, Claude.
- Decision: map sampler correction, advantage scaling and aggregation only
  for `algorithm: olmo3` in Phase 1; plain GRPO and DAPO keep their current
  veRL mapping.
  Rationale: changing an existing algorithm's veRL behavior would change
  already-recorded selections; `codex/release-0.4.12` rejects their
  non-default values instead.
  Date/Author: 2026-09-28, Claude.
- Decision: implement the truncation penalty as reward shaping inside
  Posttrain's veRL agent loop through one shared function with the TRL path.
  Rationale: it is a reward-shaping rule, applied before group statistics in
  both backends; sharing the function makes the two paths identical by
  construction and keeps veRL's DAPO filter reading the shaped reward.
  Date/Author: 2026-09-28, Claude.
- Decision: keep rejecting `algorithm: olmo3` on veRL until Phase 2 because
  `GRPOSettings` requires OLMo 3 to use active sampling; Phase 1 delivers and
  tests the objective mapping behind that rejection. Truncation penalty for
  GRPO/DAPO is accepted once Phase 1 qualifies.
  Rationale: the user asked to remove a rejection only when its phase
  qualifies; OLMo 3 is not usable without active sampling.
  Date/Author: 2026-09-28, Claude.
- Decision: implement veRL active sampling as a round-based mode of the V1
  replay buffer (Phase 2) rather than adapting the streaming DAPO filter.
  Rationale: see the Surprises entry on streaming versus rounds; round
  semantics also give the same metric meanings as TRL post11
  (`active_sampling/round_<n>_{requested,generated,retained}_groups`).
  Date/Author: 2026-09-28, Claude.
- Decision: the Phase 1 parity test lives in Posttrain
  (`packages/train/tests/test_verl_olmo3_parity.py`) and skips unless both
  `trl` and `verl` import; it runs in a scratch environment described in
  Concrete Steps. It uses TRL's real `GRPOTrainer._compute_loss` and veRL's
  real `ppo_loss`, `compute_rollout_correction_and_add_to_batch` and
  `compute_grpo_outcome_advantage`, configured from the Hydra overrides
  Posttrain generates. TRL's group advantage is computed inline inside a large
  generation method, so the test transcribes its four-line formula from TRL
  post11 (`trl/trainer/grpo_trainer.py`, `sum_then_normalize` branch) and
  cites it.
  Rationale: neither fork may depend on the other; the adapter that maps one
  selection to both backends is Posttrain, so the equivalence claim belongs
  there.
  Date/Author: 2026-09-28, Claude.

## Outcomes & Retrospective

Not yet reached.

## Context and Orientation

Repositories and worktrees (all under `/home/hammad/projects`):

- `rl-verl-vortex`: Posttrain on branch `codex/verl-vortex`. The TRL backend
  lives in `packages/train/src/posttrain/train/backends/trl/`; the veRL backend
  in `packages/train/src/posttrain/train/backends/verl/`. Settings are frozen
  dataclasses in `packages/train/src/posttrain/train/profiles.py`
  (`GRPOSettings`, `ActiveGroupSampling`, `AdaptiveCurriculum`,
  `SAMPOSettings`, `shape_online_reward`).
- `verl-vortex`: the CarbonTeq veRL fork (remote `origin`
  `git@github.com:carbonteq-ai/verl.git`, `upstream`
  `https://github.com/verl-project/verl.git`) on branch `codex/vortex` from
  release `carbonteq-v0.9.0.post4` (`54124edfb8d0b73694696400cf07a76a14d9be65`).
  Its ledger is `CARBONTEQ_FORK.md`. The shared checkout `verl-upstream` may
  hold other agents' work and must not be edited.
- `trl` (read only here): the CarbonTeq TRL fork. Release tag
  `carbonteq-v1.12.0.post11` (`4f5eeb3d9250c902be1d158abd5faf87178fd6c0`)
  is the TRL reference for every parity check. Relevant code:
  `trl/trainer/olmo3_grpo_config.py` (the fixed recipe),
  `trl/trainer/grpo_trainer.py` `_compute_loss`,
  `_vllm_importance_sampling_ratio`, `_prepare_active_sampling_inputs`.

How a veRL run happens. `posttrain job run` builds a `GRPORequest` and the
veRL launcher (`backends/verl/launcher.py`) turns it into a
`VerlLaunchManifest` (`backends/verl/contracts.py`, pydantic, JSON on disk).
It then starts `python -m posttrain.train.backends.verl.worker MANIFEST` in a
separate interpreter that has veRL installed (`backend_options.python_executable`).
The worker (`backends/verl/worker.py`) writes the dataset parquet and an
agent-loop config, and runs `python -m verl.trainer.main_ppo` with Hydra
overrides from `build_hydra_overrides`. Inside veRL, the V1 trainer
(`verl/trainer/ppo/v1/trainer_base.py`) samples prompts, and for each prompt
runs `rollout.n` copies of Posttrain's `PosttrainVerifiersAgentLoop`
(`backends/verl/agent_loop.py`), which drives one Verifiers environment
episode against veRL's vLLM server and returns token ids, the trainable mask,
the rollout log-probabilities and a scalar reward. The replay buffer
(`verl/trainer/ppo/v1/replay_buffer.py`) collects finished prompt groups; the
trainer recomputes old log-probabilities with the actor
(`_compute_old_log_prob`), reference log-probabilities (`_compute_ref_log_prob`,
LoRA: the same model with the adapter disabled, i.e. the base model), applies
the sampler correction and advantages (`_compute_advantage`), and updates the
actor (`verl/workers/utils/losses.py` `ppo_loss`, with the policy loss chosen
by `actor_rollout_ref.actor.policy_loss.loss_mode` from
`verl/trainer/ppo/core_algos.py`).

Terms used below. A prompt group is the `num_generations` rollouts of one
task in one update. The rollout (or sampler) log-probability is the one vLLM
reported while sampling; the old log-probability is the trainer's
recomputation before the update; the reference log-probability is the base
model's. The importance ratio is `exp(current - old)`; the sampler correction
weight is `min(exp(old - rollout), 2)`.

## Plan of Work

### Phase 1: OLMo 3 objective and truncation penalty

Fork (`verl-vortex`), file `verl/trainer/ppo/core_algos.py`: register policy
loss `token_clip`: `ratio = exp(logp - old_logp)` on sampled tokens,
`loss = max(-A * ratio, -A * clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high))`,
multiplied by `rollout_is_weights` when present, aggregated with `agg_loss`
and the actor's `global_batch_info`, with the same three metrics as `vanilla`
(`actor/pg_clipfrac`, `actor/ppo_kl`, `actor/pg_clipfrac_lower` = 0). Add
`k3_unclipped` to `kl_penalty_forward`: `expm1(ref - logp) - (ref - logp)`.
Regression tests in `tests/trainer/ppo/test_token_clip_policy_loss_on_cpu.py`.
Update `CARBONTEQ_FORK.md`.

Posttrain (`rl-verl-vortex`):

- `profiles.py`: factor the reward-shaping rule into
  `shape_rollout_reward(reward, completion_tokens, *, is_truncated,
  max_completion_tokens, overlong_buffer_tokens, overlong_penalty_factor,
  truncation_penalty)`, used by `shape_online_reward` (TRL) and by the veRL
  agent loop.
- `backends/verl/contracts.py` `VerlAlgorithm`: accept
  `online_rl_algorithm="olmo3"`; add `truncation_penalty`,
  `normalize_advantage_by_std`, `rollout_importance_sampling`
  (`"token"` or None) and `rollout_importance_sampling_cap`.
- `backends/verl/launcher.py`: build the OLMo 3 algorithm payload
  (`_grpo_algorithm_payload`), pass `truncation_penalty`, stop rejecting the
  truncation penalty, keep rejecting OLMo 3 with a message naming active
  sampling as the missing piece, and record `truncation_penalty` and
  `advantage_scaling` in `grpo_runtime_resolved`.
- `backends/verl/worker.py` `build_hydra_overrides`: for OLMo 3 emit
  `actor_rollout_ref.actor.loss_agg_mode=token-mean`,
  `actor_rollout_ref.actor.policy_loss.loss_mode=token_clip`, clip ratios
  0.2/0.272, `algorithm.norm_adv_by_std_in_grpo=false`,
  `+algorithm.rollout_correction.rollout_is=token`,
  `+algorithm.rollout_correction.rollout_is_threshold=2.0`,
  `+algorithm.rollout_correction.bypass_mode=false`, and
  `actor_rollout_ref.actor.kl_loss_type=k3_unclipped`. Protect the new keys
  from `backend_options.hydra_overrides`. Require a fork revision that
  contains `token_clip` (a revision set, like the rollout-execution gate).
  Pass `truncation_penalty` to the agent-loop config.
- `backends/verl/agent_loop.py`: apply `shape_rollout_reward`.
- Tests: `packages/train/tests/test_verl_backend.py` (mapping, protection,
  truncation shaping, journal of shaped reward) and the parity test
  `packages/train/tests/test_verl_olmo3_parity.py`.

Parity test contents. One fixed batch: two prompt groups of four rollouts,
16 response tokens each, trainable masks with gaps (tool tokens) and short
rows, task rewards with one truncated rollout (so the penalty matters),
current log-probabilities that require gradients, old log-probabilities
perturbed so that some ratios cross both clip bounds, rollout
log-probabilities perturbed so that some correction weights hit the cap of 2,
and reference log-probabilities for `beta = 0.005`. TRL side: rewards shaped
by `shape_online_reward`, advantage by TRL's formula, loss by
`GRPOTrainer._compute_loss` on a stub trainer configured from
`Olmo3GRPOConfig` defaults with the same `beta`. veRL side: rewards shaped by
the agent-loop function, advantage by `compute_grpo_outcome_advantage`
configured from the generated overrides, correction weights by
`compute_rollout_correction_and_add_to_batch`, loss by `ppo_loss` with an
`ActorConfig` composed by Hydra from veRL's `ppo_trainer.yaml` plus Posttrain's
overrides. Compare advantages (exact), correction weights (1e-6), loss and the
gradient with respect to the current log-probabilities (1e-6 relative, float64).

### Phase 2: active sampling with bounded refill in the veRL fork

Fork: add an `active` mode to the synchronous V1 replay buffer, selected by a
new `algorithm.active_sampling` config block (`enable`, `max_candidate_batches`,
`oversample`, `oversample_refill`, `reward_std_epsilon`, `metric`). Semantics
copied from TRL post11 `_prepare_active_sampling_inputs`: the candidate pool is
`max_candidate_batches * train_batch_size` prompts; round 1 dispatches
`train_batch_size + oversample` prompts; the buffer waits for every in-flight
group of the round (round barrier), keeps groups whose metric spread exceeds
epsilon, then dispatches exactly `missing + oversample_refill` prompts, never
more than round 1 and never more than the remaining pool, and fails with the
TRL error text when rounds or the pool are exhausted. Selection keeps the
first `train_batch_size` retained groups in dispatch order and discards the
surplus. Metrics use TRL's names under `training/active_sampling/...` (the
Posttrain metric normalizer maps both). The concurrency guard: the largest
round's episodes, `(train_batch_size + max(oversample, oversample_refill)) *
rollout.n`, must not exceed `min(rollout.max_num_seqs, agent
max_concurrent_episodes, num_workers * max_concurrent_episodes_per_worker)`
when those are set; checked at trainer start. Posttrain maps
`ActiveGroupSampling` (after the post11 fields `oversample` and
`oversample_refill` are merged from `codex/active-sampling-oversample`) and
repeats the guard at plan time. Parity: a fake-rollout replay-buffer test in
the fork that replays TRL post11's round decisions for the same reward
sequences, and a Posttrain test that the metric names normalize identically.
Remove the OLMo 3 rejection when this phase qualifies (CPU tests plus a short
Qwen3.5-0.8B 8 GB run with at least one refill round).

### Phase 3: adaptive curriculum on veRL

The controller must choose the tasks of every round (initial and each
refill) from evidence observed so far, exclude tasks already proposed in the
update, and checkpoint its state with the model. Fork extension point: an
optional `data.prompt_selector` (class path) consulted by the V1 trainer's
refill path (`_add_prompts_to_generate`) instead of the dataloader, called
with the number of prompts wanted, the round index and stage; a callback
after each round with the finished groups' rewards; and `state_dict` /
`load_state_dict` saved next to `data.pt` in each checkpoint. Posttrain
implements the selector over `AdaptiveCurriculumController` (the same
controller class the TRL path uses through
`backends/trl/policy_curriculum.py`), emits the same decision/observation
events and writes the same snapshot name in the checkpoint. Parity: feed the
same reward sequence to the TRL runtime and the veRL selector and require the
same decisions.

### Phase 4: LFM2.5 on veRL

Qualify LFM2.5 (hybrid short-convolution plus attention blocks, tied input
and output embeddings) in veRL's FSDP2 actor and vLLM rollout: model loading
through Transformers' `Lfm2ForCausalLM`, `use_remove_padding=false`, fused
PPO head or chunked entropy on an unfused head, LoRA target modules that
exist in both Transformers and vLLM (attention `q_proj`, `k_proj`, `v_proj`,
`out_proj`; convolution `in_proj`, `out_proj`; feed-forward `w1`, `w2`,
`w3`), LoRA weight synchronization to vLLM, checkpoint and merged export with
tied embeddings. Add `"lfm2.5"` to the veRL launcher's qualified families only
after a two-update LFM2.5-1.2B run on the 8 GB card completes.

### Phase 5: SAMPO on veRL

Rebuild SAMPO on the Phase 2/3 machinery: the fork already has the SAMPO
estimator, the GSPO loss and V1 metadata transport. Posttrain maps
`SAMPOSettings` including active sampling, curriculum, sampler correction and
truncation penalty, and compares advantages with the TRL SAMPO path on one
fixed multi-turn batch.

### Phase 6: end-to-end parity runs

Short runs (two to four updates) of the same VORTEX settings on TRL and veRL
with the same seed and data on the 8 GB card, first Qwen3.5-0.8B then
LFM2.5-1.2B; compare per-update reward, KL, entropy, advantage distribution,
active-sampling counts and loss; document tolerances.

Packages (`apps/lab/.posttrain/work_packages/`, catalog
`apps/lab/.posttrain/catalog/verl-vortex-parity.yaml`): for each model
(`qwen08b`, `lfm12`) and technique (`vortex`, `sampo`) a TRL and a veRL twin,
`<model>_<trl|verl>_automationbench_<vortex|sampo>_parity_local.yaml`. Twins
share the 24-task environment `automationbench-verl-parity-v1` (20 `simple`,
two `finance`, two `operations` tasks: three curriculum classes; six turns of
up to 1,024 tokens), the settings
(`<model>/automationbench-vortex-parity-v1`: OLMo 3 objective, 4 prompts x 4
generations, active sampling with `max_candidate_batches: 4, oversample: 1,
oversample_refill: 1`, the yield-first curriculum, truncation penalty 0.1,
token-truncated correction at 2.0, seed 1729, four updates;
`<model>/automationbench-sampo-parity-v1`: the same sampling with SAMPO,
beta 0.005, truncation penalty 0.2), the LoRA update (Qwen: r8/a16 on
`o_proj`/`down_proj`, the module set veRL's Qwen3.5 bindings qualify; LFM:
r4/a8 `all-linear`), and the rollout sampling (T 0.8, top-p 0.95). They
differ only in the backend and in engine placement options that do not change
sampling semantics. One known numeric difference remains: TRL trains LoRA over
a bfloat16 base in bfloat16, veRL's FSDP computes in bfloat16 over float32
master weights.

What must agree. Rollouts are stochastic and the two stacks use different
random streams, so no rollout-dependent value can be bitwise equal. The
checks are:
1. Deterministic invariants, exactly: the learning rate, the number of
   updates, the first round's oversampled group count (1), the pool size, and
   for OLMo 3 a zero per-group mean advantage.
2. Rollout statistics of update 1 (both policies are the base model): mean
   reward within two standard errors of the group-mean difference; completion
   length and sampler entropy within 15% relative.
3. Training statistics per update: KL (SAMPO) of the same order and zero at
   update 1; advantage absolute mean, positive and negative fractions and the
   informative fraction within two standard errors given the observed reward
   spread; clip and correction fractions of the same order.
4. Active-sampling and curriculum evidence: rounds, retained and discarded
   group counts consistent with each run's own rewards (the arithmetic is
   checked exactly against TRL by the CPU parity tests); curriculum class
   shares of the same order. The runs record the observed differences in
   Artifacts.

## Concrete Steps

Parity environment (scratch, not committed). From any directory:

    S=/tmp/claude-1000/-home-hammad-projects-rl/9dcbdb8a-3c09-497d-bb22-978c505cddb5/scratchpad/verl-vortex
    uv venv --python 3.13 $S/parity-venv
    uv pip install --python $S/parity-venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
    uv pip install --system-certs --python $S/parity-venv/bin/python "trl==1.12.0.post11" "transformers>=5.14,<5.15" \
        "peft>=0.19,<0.20" "accelerate>=1.14,<1.15" "datasets>=4.6.1,<4.7" \
        --index-url https://pypi.org/simple --extra-index-url https://pypi.lan/carbonteq/dev/+simple/ \
        --index-strategy unsafe-best-match
    uv pip install --system-certs --python $S/parity-venv/bin/python "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
        hydra-core omegaconf codetiming numpy pandas pyarrow cachetools dill orjson pytest \
        "ray[default]>=2.41.0" torchdata "transferqueue==0.1.8" pylatexenc
    uv pip install --python $S/parity-venv/bin/python --no-deps -e /home/hammad/projects/verl-vortex
    uv pip install --python $S/parity-venv/bin/python -e /home/hammad/projects/rl-verl-vortex/packages/common \
        -e /home/hammad/projects/rl-verl-vortex/packages/data -e /home/hammad/projects/rl-verl-vortex/packages/train

Fork focused tests (from `/home/hammad/projects/verl-vortex`):

    $S/parity-venv/bin/python -m pytest -q tests/trainer/ppo/test_token_clip_policy_loss_on_cpu.py

Posttrain focused tests (from `/home/hammad/projects/rl-verl-vortex`):

    uv run pytest -q packages/train/tests/test_verl_backend.py packages/train/tests/test_truncation_penalty.py
    $S/parity-venv/bin/python -m pytest -q packages/train/tests/test_verl_olmo3_parity.py

Full ladder (from `/home/hammad/projects/rl-verl-vortex`): `uv sync
--all-packages --locked --python 3.13`, `uv run ruff check .`, `uv run
pyright`, `uv run lint-imports`, `uv run pytest`, `git diff --check`.

GPU rule for every GPU step: only the local RTX 3070 Ti; before starting run
`nvidia-smi --query-compute-apps=pid --format=csv,noheader` and continue only
when it prints nothing (retry later otherwise); one GPU job at a time;
experiment directories under the scratch path above, never `/tmp` directly.

## Validation and Acceptance

Phase 1 is accepted when the fork's token-clip tests pass, the Posttrain veRL
backend tests pass, and the parity test reports identical advantages and a
loss and gradient that match TRL post11 within 1e-6 relative in float64, with
the batch exercising both clip bounds, the correction cap and a truncated
rollout. Each later phase names its acceptance in its section above and in
Progress when it completes.

## Idempotence and Recovery

All code changes are on the two branches named above; the shared
`verl-upstream` checkout is never modified. The scratch environment can be
deleted and rebuilt with the commands above. Nothing is pushed or published
without reporting first; the fork release and the Posttrain pin are separate
commits so either can be dropped.

## Artifacts and Notes

Parity evidence is recorded here as each phase completes.

Phase 1 (float64, 2 groups x 4 rollouts x 16 tokens, 2 micro-batches, beta
0.005, truncation penalty 0.2):

    TRL loss 0.021341312095130  veRL loss 0.021341312095130  |diff| 0.00e+00
    max |grad diff| 1.08e-19  grad norm 0.047840
    advantages [0.675, -0.325, 0.175, -0.525, -0.4625, -0.0125, 0.7375, -0.2625]

Advantages are bitwise equal; correction weights agree to 1e-12.

Phase 2 (`packages/train/tests/test_verl_active_sampling_parity.py`, parity
environment with the fork at `6c7295cd`): 7 named scenarios (first round
full, refill only missing, oversampled surplus discarded, refill capped at the
first round, extra groups cut to the pool, pool exhausted, rounds exhausted)
and 12 random reward patterns. For every case TRL post11 and veRL dispatch the
same round sizes, keep the same candidate groups, fail for the same cause, and
report identical `active_sampling/...` metrics.

    19 passed in 9.68s

Phase 3 (`packages/train/tests/test_verl_curriculum_parity.py`): four updates
of yield-first curriculum with oversample 1 / refill 1, 24 tasks in three
classes, deterministic rewards per task and occurrence. TRL's curriculum loop
and veRL's buffer plus selector emit identical decision and observation
events (6 decisions, 2 of them refill rounds), keep the same tasks per update,
and end with byte-identical controller state. Making veRL observe only kept
groups (a plausible bug) fails the test.

Phase 5 (`packages/train/tests/test_verl_sampo_parity.py`, both advantage
normalizations): two prompt groups of three multi-turn trajectories with tool
gaps, shared and singleton anchor states, explicit and sparse step rewards and
truncated rollouts. Token advantages agree to 1e-12, the hierarchy evidence
(episode/turn advantage magnitudes, informative fraction, singleton-anchor
fraction, turn credit share, anchor group size) is equal, correction weights
agree to 1e-12, and loss and gradient agree to 1e-8 relative (veRL's
seq-mean-token-mean divides by tokens + 1e-8; TRL clamps at 1). Sequence
ratios fall both inside and outside the 0.003/0.004 clip range.

    all parity files: 24 passed in 14.55s

TRL-settings parity (`packages/train/tests/test_verl_trl_settings_parity.py`,
24 cases): sampler-correction weights for token/sequence truncate and mask
with and without lower and upper bounds, including log ratios of +24 and -26
nats, agree to 1e-12 relative; GRPO advantages for group, batch and no scaling,
with and without masked truncated completions, are bitwise equal, and loss and
gradient agree to 1e-8 relative; four admission scripts (retry then succeed,
drop after attempts, single attempt, mixed) give identical retained groups and
per-group attempt counts to Posttrain's TRL admission loop; a partial batch's
loss matches TRL's padded, rescaled loss (float32-limited); the linear schedule
gives TRL's learning rate at every step for three warmup/length settings.

    all parity files together: 48 passed in 29.53s

DAPO parity (`packages/train/tests/test_verl_dapo_parity.py`): three updates
of DAPO with the yield-first curriculum and batch advantage scaling. TRL's real
`_prepare_dynamic_sampling_inputs` (fed by one curriculum decision per step, as
the TRL backend's `AdaptiveCurriculumTrainer` does) and the fork's
`CandidateBatchReplayBuffer` with Posttrain's selector use 3, 2 and 1 candidate
batches, make identical curriculum decisions and observations, keep the same
tasks, report identical `dynamic_sampling/*` metrics, give bitwise-equal
advantages and end with identical controller state. Using the kept batch's std
instead of each candidate batch's std makes the test fail.

## Interfaces and Dependencies

Fork, `verl/trainer/ppo/core_algos.py`:

    @register_policy_loss("token_clip")
    def compute_policy_loss_token_clip(old_log_prob, log_prob, advantages, response_mask,
                                       loss_agg_mode="token-mean", config=None,
                                       rollout_is_weights=None) -> tuple[torch.Tensor, dict]

    kl_penalty_forward(logprob, ref_logprob, "k3_unclipped") == torch.expm1(ref - logprob) - (ref - logprob)

Posttrain, `packages/train/src/posttrain/train/profiles.py`:

    def shape_rollout_reward(reward: float, completion_tokens: int, *, is_truncated: bool,
                             max_completion_tokens: int, overlong_buffer_tokens: int | None,
                             overlong_penalty_factor: float, truncation_penalty: float | None) -> float
