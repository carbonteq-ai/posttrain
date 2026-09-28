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
| 6 | Checkpoint on cancel | `codex/cancel-checkpoint` | 8 GB cancel mid-rollout and mid-actor: checkpoint listed and verified, run `cancelled` |

Out of scope: held-out re-baselines and checkpoint evaluations on the
workstation, the 2.6B FP16 check on Blackwell, and any change to training
settings of running experiments.

## Progress

- [x] (2026-09-28) #3 fork and adapter branches pushed; PRs open.
- [x] (2026-09-28) #6 implemented on `codex/cancel-checkpoint` (`edb0607d`), CPU tests pass.
- [x] (2026-09-28) Episode labels (#4 prerequisite) on `codex/episode-endings` (`cccb5570`).
- [ ] #1 kernels: wheel built, image rebuilt, before/after measured.
- [ ] #2 precision: code committed; training arms after #1.
- [ ] #4 Trackio release prepared; publish + server migration (maintainers; no run writing).
- [ ] #5 TRL release prepared; publish; Posttrain pin.
- [ ] #6 GPU cancel qualification on the 8 GB card.
- [ ] Integrate on `codex/release-0.4.12` in the order below, full validation ladder, changelog.
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

## Surprises & Discoveries

(recorded as each fix qualifies)

## Decision Log

- Decision (user, 2026-09-28): release scope is exactly fixes 1–6; fix now,
  no deferred follow-ups inside that scope.
- Decision (user): oversampling uses two fixed counts of extra prompt groups,
  `oversample` (first round) and `oversample_refill` (each refill round),
  because the right number depends on rollout concurrency; job plan checks
  that the first round fits vLLM and worker capacity.
- Decision (user): no automatic stop on KL or entropy.
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
