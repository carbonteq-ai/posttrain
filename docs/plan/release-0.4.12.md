# Release 0.4.12: training-harness fixes found by the SAMPO continuation audit

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Auditing the LFM2.5-2.6B SAMPO continuation
(`lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1`) showed that the
training harness, not only the model, limited results: tools returned wrong
or fabricated data, episodes ended for reasons nobody could see, a cancel lost
every update since the last checkpoint, Qwen3.5 trained on slow fallback
kernels, the trainer and sampler disagreed numerically in bf16, and refill
rounds ran one after another. Release 0.4.12 ships the six fixes the user
scoped on 2026-09-28, each qualified before it is merged, with nothing
deferred to "later".

After this release a user can train on tools that behave as documented,
choose FP16 training, see how every episode ended, cancel a run without losing
work, train Qwen3.5 on its fast kernels, and oversample prompt groups so an
update fills in fewer rounds.

This release does not change the frozen product baseline except the narrow
observation amendment in `docs/post-training/06-observation-and-lineage.md`
made on `codex/episode-endings` (episode ending labels).

## Scope

| # | Fix | Branch / repository | Qualification gate |
| --- | --- | --- | --- |
| 1 | Fast Qwen3.5 kernels (`flash-linear-attention`, `causal-conv1d`) in the trainer image | `codex/qwen-fast-kernels` (Posttrain runtime images) + internal wheel | fast path active; LoRA fwd/bwd finite in bf16 and fp16 on sm86; actor time and log-prob gap before/after |
| 2 | FP16 training precision and rollout dtype | `codex/precision-fp16` | offline mismatch matrix; BF16/BF16 vs FP16/FP16 training on Qwen3.5-0.8B (8 GB); float32 rollout rejected for Gated-DeltaNet models |
| 3 | Tool fidelity (AutomationBench fork, adapter 0.5.0) | carbonteq-ai/AutomationBench#2, carbonteq-ai/verifiers-environments#2; catalog pins on `codex/eval-train-budget` | fork suite, adapter suite, grading-compatibility replay; first updates of `lfm26-sampo-cont40-fixed-tools-20260928-r1` |
| 4 | Trackio release: artifact-commit retries + `episode_ending` fact column | Trackio `codex/next-release`; Posttrain `codex/episode-endings` → `codex/trackio-next` | fork tests; Doris migration; backfill dry run; Observatory list/detail agree |
| 5 | Active-sampling oversampling (`oversample`, `oversample_refill`) | TRL `codex/active-sampling-oversample`; Posttrain `codex/active-sampling-oversample` | defaults byte-identical; unit tests; short 8 GB canary with oversampling on |
| 7 | KL reference = base model when a run continues a trained adapter (`kl_reference: base`, TRL `peft_reference`) | TRL post11 (published 2026-09-28); Posttrain `codex/active-sampling-oversample` | TRL test: reference log-probs equal the base model's with a non-zero starting adapter; job plan prints the reference |
| 8 | veRL: start from a trained adapter, KL to base | Posttrain (+ veRL fork if needed) | 8 GB: 2 fresh updates then 2 from that adapter; adapter loaded, KL > 0 vs base |
| 9 | veRL: FP16 training and rollout dtype | Posttrain (+ veRL fork if needed) | 8 GB Qwen3.5-0.8B bf16 vs fp16: finite grads, loss scale, smaller gap |
| 6 | Checkpoint on cancel | `codex/cancel-checkpoint` | 8 GB cancel mid-rollout and mid-actor: checkpoint listed and verified, run `cancelled` |

Out of scope: held-out re-baselines and checkpoint evaluations on the
workstation, the 2.6B FP16 check on Blackwell, and any change to training
settings of running experiments.

## Progress

- [x] (2026-09-28) #3 fork and adapter branches pushed; PRs open.
- [x] (2026-09-28) #6 implemented on `codex/cancel-checkpoint` (`edb0607d`), CPU tests pass.
- [x] (2026-09-28) Episode labels (#4 prerequisite) on `codex/episode-endings` (`cccb5570`).
- [x] (2026-09-28) #1 kernels: fla-core and the causal-conv1d rebuild published to `carbonteq/dev` and adopted in every Qwen3.5 training image; before/after measured on sm86 (`codex/qwen-fast-kernels` final `94c486a9`; see `docs/tooling/linear-attention-kernels/README.md`). Kind images are rebuilt by the release candidate.
- [x] (2026-09-28) #2 precision: code committed for TRL and veRL (`codex/precision-fp16-verl` `e7edb8ef`); veRL 0.9.0.post4 published and pinned (`969adbfd`). Training arms (GPU) still to run after the candidate images.
- [ ] #4 Trackio release prepared; publish + server migration (maintainers; no run writing).
- [x] (2026-09-28) TRL 1.12.0.post11 published (oversampling + `peft_reference`): tag `carbonteq-v1.12.0.post11` → `3135b502`, workflow run 36434036656, carbonteq/dev serves wheel `7fcea40a…` and sdist `bb3cdec9…`.
- [x] (2026-09-28) #5/#7 Posttrain pin to post11 (`930c1611`) relocked and validated on the release branch.
- [x] (2026-09-28) Trackio dev32 published (tag → `d71cf2a5`, workflow run 36432505907); Posttrain pin relocked on `codex/trackio-next` (`9a3779cf`). Server migration pending a quiet window.
- [x] (2026-09-28) #8 veRL adapter start (`842768f5`) and #9 veRL FP16 (`e7edb8ef` + post4 pin) implemented; CPU tests pass; GPU qualification waits for the candidate's veRL kind image.
- [ ] #6 GPU cancel qualification on the 8 GB card.
- [x] (2026-09-28) Integrated on `codex/release-0.4.12` (worktree `/home/hammad/projects/rl-release-0.4.12`, base `codex/eval-train-budget` `c35da68e`). Merge order and conflicts: see "Integration record" below. Version set with `posttrain-release prepare 0.4.12`; CHANGELOG `## 0.4.12` written. Full ladder green (see Validation and Acceptance).
- [x] (2026-09-28) Coordinator additions on the release branch: veRL rejects GRPO settings it would silently ignore (`af5684ed`); veRL precision pair and adapter-continuation catalog fixes and the retired veRL SAMPO-2 note (`58535c27`); causal-conv1d as a wheel-only required fork with stable promotion support (`6d8fbefa`) and its public-CI consumer wheel (`350ad262`).
- [x] (2026-09-28) veRL runs the training loop as selected or rejects it (`1a76166a`): schedule, warmup, seed, logging cadence, batch split, weight decay 0.0; lab veRL settings state `lr_scheduler_type: constant`. Ladder: pytest 2167 passed, 24 skipped; ruff, pyright, lint-imports clean.
- [x] (2026-09-28) Merged `codex/verl-vortex` (`a22257b7`: OLMo 3 objective mapping, truncation penalty on veRL for GRPO/DAPO, native-name gate; OLMo 3 stays rejected until VORTEX Phase 2) and pinned veRL 0.9.0.post5 (`d93c5f78`: tag → `9fd6e7a3`, backend lock `96d02bcc…`, constraints `e43a8d50…`; the native-name test's strict xfail removed and passing). Ladder: pytest 2188 passed, 26 skipped; consumer 2 passed; ruff, pyright, lint-imports clean.
- [x] (2026-09-28) `posttrain-release images plan` (registry.lan/carbonteq, LAN trust bundle): base reused remotely; all six kinds (supervised, online-rl-trl-py312, online-rl-verl-py313, eval, serve, transform) rebuild; not blocked.
- [x] (2026-09-28) Job-kind images published and `published.toml` committed (`d4d644ca`, by the coordinator); strict `posttrain-release check` passes.
- [x] (2026-09-28) Lab environments validate against Verifiers `cdd2ec76` (`5ef066ef`: seat-level harness, timeouts and turn/token limits in 23+5 environments; new test `apps/lab/tests/test_environment_activation_configs.py`); `gsm8k-grpo-qualification` deferred (`8eaa90ea`); rollout-topology tests independent of the host CPU count (`b61f5c18`, Quality run 36448533918). `posttrain job pack --local --allow-deferred-qualification` succeeds for the veRL adapter continuation (package `0e0790de…`), `gsm8k_qwen08b_grpo_qualification` (TRL, `3398f813…`) and `environment_library_ifeval_qualification` (`d8bb514f…`).
- [ ] Push `codex/release-0.4.12`, open the release PR, wait for Quality (dispatch it with `allow_pending_runtime_lock=true` if the push run needs it).
- [ ] Publish the 0.4.12 job-kind images (locally per `docs/publishing.md` step 7, or in the candidate) so `published.toml` records the merged locks and veRL post4; then `posttrain-release check` (strict) passes and the veRL packages can pack.
- [x] (2026-09-28 15:32Z) Trackio server: Doris backups, `migrate-doris --to 5`, dev32 deployed, episode-ending backfill (record merged from `codex/trackio-next` `c896ef58`; `docs/tooling/trackio/README.md`).
- [x] (2026-09-28) Observatory semantic SQL read the first discovered project; now the requested or configured source (`aae90bb4`), checked read-only against the dev32 server (posttrain-lab 58 runs; ai-infra-qualification 0).
- [ ] GPU gates on the new images: #1 kernels in the kind image, #2/#9 precision arms, #6 cancel on the 8 GB card, #8 veRL continuation (fresh then `--model-from-run`), #5 oversampling canary.
- [ ] Release candidate per `docs/release-engineering.md`; final.

## Integration order

Start `codex/release-0.4.12` from `codex/eval-train-budget` (tool-fixed
catalog, fixed-tool suites, plans). Merge in dependency order, running the
validation ladder after each: `codex/episode-endings`, `codex/cancel-checkpoint`,
`codex/precision-fp16`, `codex/qwen-fast-kernels`, `codex/trackio-next`
(after the Trackio wheel is published and the pin relocked),
`codex/active-sampling-oversample` (after the TRL wheel is published and the
pin relocked). Resolve conflicts in `packages/train/.../backends/trl/*`
(cancel checkpoint, precision and oversampling all touch it) by keeping each
feature's behaviour and re-running its tests.

## Integration record

All merges are `--no-ff` merge commits on `codex/release-0.4.12`; each was
followed by the touched packages' tests (run with `CUDA_VISIBLE_DEVICES=""`).

1. `codex/episode-endings` (`cccb5570`): clean.
2. `codex/trackio-next` (`9a3779cf`): clean.
3. `codex/cancel-checkpoint` (`edb0607d`): `posttrain.common.__all__`, kept
   both sides. Follow-up `e18b3126`: the transformers cancel test loaded the
   saved adapter onto CUDA when a GPU is visible (`load_peft_weights` default)
   and compared it with the CPU model; it now loads on the CPU.
4. `codex/precision-fp16-verl` (`e7edb8ef`, contains `codex/precision-fp16`):
   TRL online-RL callback list, kept the update-boundary callback
   (`on_pre_optimizer_step`) and the loss-scale callback (`on_step_end`), both
   before the metrics callback. Follow-up `d11f0485`: pyright error in
   `test_verl_precision.py`.
5. `codex/precision-fp16-verl-post4-pin` (`969adbfd`): clean.
6. `codex/qwen-fast-kernels` (`3cee9e3b`, then final `94c486a9`): veRL backend
   project took post4's fork fields and the kernels branch's Verifiers
   `cdd2ec76`, renderers and kernels; relocked with `uv lock --python 3.13.12`
   and re-exported constraints (lock `d33bdfa7…`, constraints `2c185aa1…`,
   equal to the kernels agent's cross-check). Quantization `uv.lock` relocked;
   catalog `locks.toml`, `quantization.yaml` and the lab bindings' copied
   `trl-fork@current` digest regenerated with `posttrain-release
   lock-dependencies`; root `uv.lock` and job-kind locks auto-merged and
   `lock-runtime-dependencies --check` reported them current.
7. `codex/active-sampling-oversample` (`842768f5`): TRL `policy_config` /
   `policy_optimization` (precision plus oversampling and KL reference), veRL
   launcher imports, `work-package plan` output (precision and KL reference
   lines), lab `gates.toml` (all seven new gates; counts 111 entries, 85
   candidates). The veRL adapter-continuation binding now names the post4 kind.
   Then `codex/active-sampling-oversample-post11-pin` (`930c1611`):
   `quality.yml`, `test_release.py`, `locks.toml` and the lab bindings;
   `uv lock`, `lock-dependencies` (digest `7a28a659…`, source `3135b502`) and
   `lock-runtime-dependencies` (TRL post11 in the workspace, supervised and
   TRL online-RL closures).
8. `origin/main` (`305d5269`, the causal-conv1d publisher workflow already on
   the branch): clean; keeps the branch descending from main.

## Surprises & Discoveries

- Observation: `posttrain work-package plan` cannot run on the integrated
  branch: the committed `published.toml` still records the 0.4.11 images
  (veRL post3, old lock digests), so the manifest loader refuses it. This is
  the documented pending state. Evidence: planning with the manifest the
  candidate will generate simulated in memory (lock digests and veRL identity
  from the shipped files; scratch script, nothing committed) resolves the
  2.6B fixed-tools continuation, both v4 held-out suites, the TRL and veRL
  Qwen precision pairs, the LFM2.5 fp16 canary and the veRL adapter
  continuation. A scratch oversampling package (`oversample: 4,
  oversample_refill: 2`) is rejected on the c144 bindings (168 > 144) and
  plans with 168-wide bindings.
- Observation: `posttrain[trl]` now requires causal-conv1d, so a consumer
  install needs its wheel: the public CI consumer job failed to resolve the
  SFT starter until the release wheel was added, and a stable-only install
  needs it promoted.
- Observation: the veRL launcher silently ignored several GRPO settings
  (`adaptive_curriculum`, `advantage_scaling`, the importance-sampling
  settings, `max_admission_attempts`). Loop fields remain partly ignored on
  veRL: `lr_scheduler_type` (veRL always runs its constant schedule after
  warmup, so the default `linear` is not applied), `logging_steps`, `seed`,
  and the `per_device_batch_size` / `gradient_accumulation_steps` split. Not
  changed here because rejecting the default would break every veRL package.
  Superseded by `1a76166a`: veRL maps these exactly or rejects them, and the
  lab veRL settings now state the constant schedule veRL always ran.
- Observation (coordinator): Posttrain requests veRL `policy_loss.loss_mode=token_clip`
  and `kl_loss_type=k3_unclipped` for GDPO/CAPO, which veRL 0.9.0.post4 lacks,
  so GDPO/CAPO on veRL would fail at the first update until post5 is pinned.

## Decision Log

- Decision (user, 2026-09-28): release scope is exactly fixes 1–6; fix now,
  no deferred follow-ups inside that scope.
- Decision (user): oversampling uses two fixed counts of extra prompt groups,
  `oversample` (first round) and `oversample_refill` (each refill round),
  because the right number depends on rollout concurrency; job plan checks
  that the first round fits vLLM and worker capacity.
- Decision (user): guard oversampling against concurrency. The largest round,
  (prompts + max(oversample, oversample_refill)) × generations episodes, must
  not exceed min(vLLM max_num_seqs, environment max_concurrent, env_workers ×
  episodes_per_worker); a refill can need every prompt group if round 1 kept
  none. Checked as an error at job plan and again in the trainer before the
  first rollout.
- Decision (user): no automatic stop on KL or entropy.
- Decision (user): KL is measured against the base model, also when a run continues a trained adapter; veRL must support starting from an adapter (not reject it) and FP16 training, in this release.
- Decision (coordinator, 2026-09-28): reject GRPO settings veRL does not
  receive at job plan and in the launcher rather than ignore them.
- Decision: causal-conv1d is a required wheel-only fork release (no sdist; the
  retained wheel is the identity); the promotion workflow accepts an empty
  `sdist_sha256` only for it.
- Decision: the SAMPO-2 veRL package stays retired (veRL rejects SAMPO until
  the VORTEX port lands) and is annotated rather than repinned.
- Decision (user): FP16 A/B replicates the paper's headline pair (BF16/BF16 vs
  FP16/FP16, arXiv 2510.26788 Section 4.4) on Qwen3.5-0.8B.

## Outcomes & Retrospective

Not yet released.

## Context and Orientation

Posttrain worktrees under `/home/hammad/projects`: `rl-perf-guard`
(`codex/eval-train-budget`), `rl-episode-endings`, `rl-cancel-checkpoint`,
`rl-precision`, `rl-qwen-kernels`, `rl-trackio-next`, `rl-oversample`. Forks:
`trackio-release-next` (Trackio), `trl-oversample` (TRL), `automationbench`,
`verifiers-environments-turns-v2`. Internal indexes: `carbonteq/dev`,
`carbonteq/stable` on `pypi.lan`.

## Concrete Steps

From each worktree root: `uv sync --all-packages --locked --python 3.13`
(add `--extra trl --extra verifiers` for train tests), `uv run ruff check .`,
`uv run pyright`, `uv run lint-imports`, `uv run pytest`, `git diff --check`.
Release: `docs/release-engineering.md` candidate and final workflows.

## Validation and Acceptance

Each row of the Scope table passes its gate before merge; the integrated
branch passes the full ladder; the release candidate canary passes.

Integrated branch, 2026-09-28, from `/home/hammad/projects/rl-release-0.4.12`:
`uv sync --all-packages --locked --python 3.13 --extra trl --extra verifiers`;
`ruff check .` and `ruff format --check .` clean; `pyright` 0 errors;
`lint-imports` 9 kept; `CUDA_VISIBLE_DEVICES="" pytest` 2155 passed, 24
skipped; `tests/consumer` 2 passed with the four fork wheels in
`POSTTRAIN_CONSUMER_EXTRA_WHEELS`; Observatory frontend `npm test` 120 passed
and `npm run build` succeeds with an unchanged generated schema; `posttrain
catalog validate` valid (100 base, 249 project entries); `posttrain-release
lock-runtime-dependencies --check` current; `posttrain-release check
--allow-pending-runtime-lock` passes (runtime lock pending candidate
materialization); strict `check` fails only on the veRL kind identity until
the images are republished; `git diff --check` clean.

## Idempotence and Recovery

Each fix is on its own branch and can be dropped from the integration branch
without affecting the others, except `codex/trackio-next`, which builds on
`codex/episode-endings`.

## Artifacts and Notes

Audit and research notes: session scratchpad `tool_audit/`, `trace_audit/`,
`research/`, `precision/`.

## Interfaces and Dependencies

New pins: CarbonTeq Trackio (next dev), CarbonTeq TRL (1.12.0.post11),
`causal-conv1d` and `flash-linear-attention` in the TRL online-RL image,
automationbench-v1 0.5.0 (`61448b5d`).
