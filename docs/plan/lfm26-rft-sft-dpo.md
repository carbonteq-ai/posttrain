# RFT warm start for LFM2.5-2.6B on AutomationBench: SFT, then a narrow DPO pass

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

SAMPO runs from the raw LFM2.5-2.6B base model show flat training reward and small held-out changes, while the ARLArena paper (arXiv 2602.21534) starts SAMPO from a model fine-tuned on its own high-scoring rollouts (Qwen3-4B-rft). This work builds that warm start from data we already have: the best and worst episodes of 25 LFM training runs stored in Trackio. After it, the lab can (1) SFT the base on the model's own best trajectories, (2) run a narrow DPO pass on same-task best-vs-worst pairs from that SFT adapter, and (3) evaluate both on the 64K luna-v2 held-out suite next to base, the old cont100 run, the v2 checkpoints and the H100 run. Success is a held-out table with SFT and SFT+DPO rows; the hypothesis is higher partial credit and goal rate with no more harmful actions than base.

The frozen product baseline is unchanged: this adds data scripts, catalog entries and work packages, and fixes two DPO backend defects.

## Progress

- [x] (2026-10-08 10:20Z) Harvest via read-only SQL on Trackio's Doris store (`scripts/data/harvest_agent_trajectories.py`, run inside the Trackio container with its own connection): about 180,000 LFM training episodes in 25 runs.
- [x] (2026-10-08 10:30Z) Datasets (`scripts/data/trajectories_to_training_data.py`): SFT 753 examples over 153 tasks (>= 0.8 partial credit, no verified harms, at most 8 per task, 3 per simple task, deduplicated by tool-call sequence, rotated across runs); DPO 1,118 same-task pairs over 133 tasks (chosen >= 0.7 without harms, rejected <= 0.2, harmful rejected first, each episode in at most two pairs). 40 held-out evaluation tasks excluded; 0 leaks.
- [x] (2026-10-08 10:40Z) SFT run `lfm26-rft-sft-r8-20261008-r1` submitted on the workstation (LoRA rank 8, 190 steps x 8 examples, LR 1e-4 linear, 28,672 tokens).
- [x] (2026-10-08 11:10Z) DPO backend: tool results masked out of the preference loss; a continued adapter's frozen policy is the DPO reference (commit `86bcda57`).
- [x] (2026-10-08) Render-check DPO pairs at 32,768 tokens; register `datasets/lfm26-agent-dpo@1`.
- [x] (2026-10-08) SFT v1 eval (`luna2-heldout-64k20t-t05-rft-sft-r8-20261008-e2`): success 0.10, partial credit 0.473, harms 0.72 per episode.
- [x] (2026-10-08) DPO v1 from the SFT adapter, eval: success 0.08, partial credit 0.487, harms 0.99 per episode (harms rose).
- [x] (2026-10-11) SFT v2 (32 per task, weighted by task) relaunched as `lfm26-rft-sft-r8-v2-20261011-r2`; eval chained.
- [x] (2026-10-11) Contrast DPO tooling: `scripts/data/episode_signals.py` (per-episode failure signals and setup fingerprint from Doris, 138,773 episodes), `scripts/data/select_dpo_contrasts.py` (perfect vs categorized failure, same task and setup), harvester `--selection`, converter category pairing.
- [x] (2026-10-11) DPO `sft_weight` setting (supervised loss on the chosen side, TRL `loss_type ["sigmoid", "sft"]`).
- [x] (2026-10-11) Offline harm re-scoring (`scripts/data/harm_rescore/`): replays each episode's recorded tool calls with its own environment revision's tools and runs the current guard manifests. 5,658 episodes of 48 guarded tasks re-scored in about 7 minutes; 3,453 of 3,647 with recorded harms match exactly, every mismatch is a higher re-scored count (186 of 194 from r9/r10).
- [x] (2026-10-12) SFT v2 eval (`luna2-heldout-64k20t-t05-rft-sft-r8-v2-20261011-e2`): success 0.05, partial credit 0.474, goals 0.608, harms 0.86 per episode, harm-free 0.56. Neutral against the base model (0.05 / 0.485 / 0.594 / 0.75 / 0.59); SFT v1 kept as the DPO starting point.
- [x] (2026-10-12) DPO v2 data `datasets/lfm26-agent-dpo@2`: 413 harm-checked contrast pairs over 38 tasks (wrong-outcome 93, tool-error 91, out-of-room 73, empty-results 58, near-miss 53, harmful 45); all fit 32,768 tokens (max 27.3K).
- [x] (2026-10-12) DPO v2 attempts: r1 OOM at step 1 (TRL's `sft` term copied the full-vocabulary chosen logits; fixed in TRL 1.12.0.post15, adopted with republished `supervised`/`online-rl-trl-py312` images); r2 OOM at step 16 on the longest pairs (pairs capped at 22,000 tokens: `datasets/lfm26-agent-dpo@3`, 380 pairs, 38 tasks); r3 cancelled in favour of the researched recipe.
- [x] (2026-10-12) DPO v3 `lfm26-rft-dpo-r8-v3-20261012-r1` from SFT v1 (`agent-dpo-v3`: `sigmoid_norm` beta 5 + sft weight 1, LR 5e-6, 10% warmup, reference precomputed, 95 steps): trained end to end in 47 minutes (about 22 s per step after an 8-minute reference pass), loss flat at about 0.85, reward accuracy 53% -> 61%.
- [x] (2026-10-12) DPO v3 evals (100 episodes): final success 0.07, partial credit 0.456, goals 0.546 (1.68 met per episode), harms 0.79, harm-free 0.56; step 48 success 0.07, partial credit 0.491, goals 0.548 (1.65), harms 1.00, harm-free 0.52. Within noise of SFT v1 (0.10 / 0.473 / 0.542 / 0.72 / 0.62); final harms well below DPO v1's 0.99.
- [ ] Ten-attempt held-out evals (`lfm26_automationbench_luna2_heldout_eval_64k20t_t05_x10`, 200 episodes each) of base, SFT v1, DPO v3 final and step 48, DPO v1, SFT v2 and SAMPO cont100, then task-resampled confidence intervals.

## Surprises & Discoveries

- Observation: some older runs record a hashed task id in `fact_task_id`; held-out exclusion by name missed them until the harvester read the name from the payload.
- Observation: training traces in Trackio keep results only (no assessment batches); harms are read from `info.posttrain_turn_rewards` turn components (`manifest_harms`), so harm filtering applies to manifest-era runs only.
- Observation: rendered SFT examples are long because the tool schemas sit in the system prompt (median 10.6K tokens, p95 18.8K, max 27.8K); the conversation itself is median about 2.5K tokens.
- Observation: only 3,203 of 10,088 perfect episodes carry manifest turn evidence, so for most "perfect" episodes the absence of harm was never assessed.
- Observation: without a setup match, about 30% of rejected episodes had no chosen partner with an identical system prompt and tool list (different environment versions).
- Observation: recorded harm counts of the r9/r10 manifest-steps runs under-count: their environment revisions recorded call arguments in a form some guards could not match. Re-scoring with the current manifests finds more harms (for example 124 vs 110 on a 100-episode sample).
- Observation: of the 111 tasks with trustworthy harm status (48 guarded, 63 unguarded, held-out excluded), only 51 ever had a perfect training episode; most perfect episodes belong to `simple.*` tasks, which have no manifest.
- Observation: SFT on more of the model's own successes (v2: 3,093 examples, 32 per task, task-weighted) did not move held-out results beyond noise; 100 episodes cannot separate differences of a few points.
- Observation: at step 0 the frozen merged reference and the policy adapter disagree slightly (bf16 rounding accumulated over thousands of tokens), so logged DPO rewards and reward accuracy start offset from zero; the loss is unaffected in expectation, and the offset was left as is.
- Observation: TRL's DPO collator trains every completion token, including tool results, and with a PEFT model and no `ref_model` uses the base model as reference.

## Decision Log

- Decision: contrast DPO pairs a perfect episode (passed, full credit, no tool mistakes, no harm, not cut off) with a failure of the same task, given one category in the order harmful, out-of-room, tool-error, empty-results, near-miss, wrong-outcome; up to 4 per task and category. Gave-up-early and plain failure were left out (few, or overlapping the others).
  Date/Author: 2026-10-11, user (categories, including out-of-room as base behaviour) and Claude (order, caps).
- Decision: only episodes whose harm status is known are used, on both sides: guarded tasks use offline re-scored harms (exact replays only, recorded flags not trusted), tasks whose reviewed manifest defines no guard count as harm-free (they still teach the other failure kinds), and tasks without a manifest are excluded. This replaced an earlier "minharm" variant that admitted unchecked perfect episodes; it was never run.
  Date/Author: 2026-10-11, user (no unchecked harm; keep unguarded-manifest tasks) and Claude (exact-replay filter).
- Decision: DPO v2 starts from SFT v1, not SFT v2: their partial credit is equal (0.473 vs 0.474) and SFT v1 has fewer harms (0.72 vs 0.86 per episode) and higher success (0.10 vs 0.05); DPO v2 aims to cut harms.
  Date/Author: 2026-10-12, user and Claude.
- Decision: a pair may come from any run or rollout group but must share the setup (system prompt and tool list); each perfect episode is used in at most 4 pairs (hard tasks with one success otherwise reused it up to 20 times); perfect episodes with identical tool calls are kept, since their reasoning differs.
  Date/Author: 2026-10-11, user (any group) and Claude (setup match, reuse cap).
- Decision: contrast DPO adds a supervised loss on the chosen side (weight 1, as in RPO) and runs two passes, because v1's plain DPO raised harms (likelihood displacement).
  Date/Author: 2026-10-11, Claude.

- Decision: LoRA rank 8 for SFT and DPO (requested); DPO continues the SFT adapter rather than stacking a new one.
  Date/Author: 2026-10-08, user and Claude.
- Decision: SFT threshold 0.8, DPO thresholds 0.7 / 0.2 (requested); "best" excludes episodes with verified harms.
  Date/Author: 2026-10-08, user.
- Decision: cap simple tasks at 3 per task and rotate picks across runs for task and behaviour diversity (requested diversity; simple tasks were 37% of the uncapped set).
  Date/Author: 2026-10-08, Claude.
- Decision: own trajectories rather than GPT Luna traces, which lack the full reasoning; SFT on reasoning-free targets would teach acting without thinking.
  Date/Author: 2026-10-08, Claude.

## Outcomes & Retrospective

Interim (2026-10-12): no SFT or DPO variant beats the base model beyond 100-episode noise on partial credit or goals. Plain DPO (v1, 2e-5, summed loss, loosely labelled pairs) raised harms (0.72 -> 0.99); the researched recipe on harm-checked contrast pairs (v3) did not (0.79) but also did not improve success, partial credit or goal rate, consistent with its nearly flat training loss. The step-48 checkpoint had higher partial credit (0.491) but more harms (1.00) than the final model. Ten-attempt evals are running to separate real differences from sampling noise.

## Context and Orientation

`train.sft` (`train/trl-sft@1`) and `train.dpo` (`train/trl-dpo@1`) render with the same `lfm2.5-tools-thinking@2` renderer as RL. Catalog entries are in `apps/lab/.posttrain/catalog/lfm26-rft-warm-start.yaml` (registered in `layer.yaml`); work packages `apps/lab/.posttrain/work_packages/lfm26_automationbench_rft_sft.yaml` and `lfm26_automationbench_rft_dpo.yaml`. Dataset JSONL files are local inputs under `apps/lab/.posttrain/datasets/` (not committed, 28 MB and larger) and enter jobs as materialized datasets.

## Concrete Steps

On the control host, inside `ai-control-trackio-1`:

    python /tmp/harvest_agent_trajectories.py --out /tmp/rft-sft --exclude-tasks /tmp/heldout-tasks.txt --per-task 8 --domain-cap simple=3 --best-min 0.8 --worst-max -1
    python /tmp/harvest_agent_trajectories.py --out /tmp/rft-dpo --exclude-tasks /tmp/heldout-tasks.txt --per-task 8 --domain-cap simple=3 --best-min 0.7 --worst-max 0.2

Locally:

    python scripts/data/trajectories_to_training_data.py sft/trajectories.jsonl.gz --out sft --sft
    python scripts/data/trajectories_to_training_data.py dpo/trajectories.jsonl.gz --out dpo --dpo

From `apps/lab`:

    posttrain job run .posttrain/work_packages/lfm26_automationbench_rft_sft.yaml --job train --provider dstack --timeout-seconds 43200 --run-id lfm26-rft-sft-r8-20261008-r1
    posttrain job run .posttrain/work_packages/lfm26_automationbench_rft_dpo.yaml --job train --provider dstack --timeout-seconds 43200 --model-from-run lfm26-rft-sft-r8-20261008-r1 --run-id lfm26-rft-dpo-r8-20261008-r1

## Validation and Acceptance

Both adapters evaluate on `lfm26_automationbench_luna2_heldout_eval_64k20t_t05` with 0 dropped attempts, scored with the same component scorer as the other rows (pass, partial credit, goal rate, harms per episode, harm-free).

## Idempotence and Recovery

The harvest is read-only and deterministic for a given Doris state. Training runs use new run ids; a failed run is resubmitted under a new id.

## Artifacts and Notes

    harvest: 25 runs, ~180k episodes; SFT 753 / 153 tasks; DPO 1,118 pairs / 133 tasks

## Interfaces and Dependencies

`RenderedPreferenceExample` gains `chosen_mask` and `rejected_mask`; `posttrain.train.backends.trl.dpo.masked_completion` rebuilds TRL's completion mask; `posttrain.train.backends.trl.common.load_frozen_adapter_reference` returns the merged, frozen adapter policy.
