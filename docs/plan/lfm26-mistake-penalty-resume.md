# Continue LFM2.5-2.6B SAMPO training with a tool-mistake penalty and relaxed length penalty

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

The VORTEX KL run (`lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2`) and the SAMPO
run (`lfm26-sampo-turns-150-lr5e5-kl5e3-20260927-r2`) raised training reward
only by truncating less. Reward on finished episodes stayed at about 0.44 for
150 updates, and failed tool calls rose (KL run 4.4% -> 8.4% of rollouts).
Held out, the KL run's update-140 checkpoint gains +0.056 over base at the
training sampling (temperature 0.8: truncation 15% -> 5%, finished-episode
reward unchanged) and nothing at temperature 0.1.

After this change a new training run continues SAMPO from its update-120
checkpoint with a
signal that rewards doing the task: a small capped penalty for real tool
mistakes, a smaller truncation penalty, lower training temperature, a slightly
stronger KL anchor, and evaluation at both 0.1 and the training temperature.
Success is finished-episode reward and held-out score rising above the
step-140 checkpoint, with mistakes per episode falling.

## Progress

- [x] (2026-09-28) Diagnosed flat task skill from Trackio rollouts (finished
  vs truncated reward per 25-update window) and three evaluation variants.
- [x] (2026-09-28) Classified 48 failed tool calls in 84 late KL-run traces:
  25 invalid arguments (8 of them the Airtable decoding bug below), 8 missing
  required arguments, 7 invented ids, 7 unknown tools, 1 empty search.
- [x] (2026-09-28) Settings agreed with the user: learning rate 3e-5 with a
  5-update warmup, temperature 0.5, KL beta 0.01, truncation penalty 0.1,
  mistake penalty 0.02 per mistake capped at 0.1, evaluation every 20 updates
  at 0.1 and 0.5. (Start point and length changed below.)
- [x] (2026-09-28) Adapter automationbench-v1 0.4.3 (`0afb73d7` on branch
  `codex/automationbench-mistake-penalty` of `verifiers-environments`):
  optional-string argument fix, mistake classification, episode mistake
  penalty, turn rewards penalize mistakes only (scorer version 2). 47 tests.
- [x] (2026-09-28) User changed the start: continue SAMPO from its update-120
  checkpoint for 100 updates instead of the KL run's update 140.
- [x] (2026-09-28) Posttrain lab catalog: `automationbench-lfm26-sampo-turns-v2`,
  projection `turn-progress@2`, work package
  `lfm26_automationbench_sampo_turns_continue120_100_local_v1.yaml`, and the
  held-out suite at temperature 0.5 (`..._64k_v3_t05.yaml`); gates registered.
- [x] (2026-09-28) User raised the batch from 16 x 4 to 24 prompt groups x 6
  attempts (144 rollouts) with every episode of a round in flight at once:
  settings `automationbench-sampo-turns-100-g24x6-lr5e-5-kl1e-2-t05-local-v1`,
  training binding `g144-w12@1` (12 workers x 12 episodes), rollout binding
  `c144-4k-t05@1` (`max_num_seqs` 144, 41 GiB KV cache), environment
  `max_concurrent` 144. `posttrain job plan` passes.
- [ ] Launch and monitor.

## Surprises & Discoveries

- The limited toolset registers tools with FastMCP, which decodes any
  JSON-looking string argument unless the parameter is declared exactly
  `str`. `airtable_create_record(fields_json: str | None)` therefore received
  a dict and failed validation although the model passed a correct string: 8
  of 48 late failures were ours, not the model's.
- The existing turn-reward penalty (`tool_result_failed`) counts every
  `success: false` result, including empty searches ("No matching meeting
  found"), so it punished legitimate searches.
- Evaluation at temperature 0.1 hides the only change training made: at 0.1
  the base model truncates 3 of 60 attempts under the training budget, at 0.8
  15 of 100.

## Decision Log

- Decision: penalize invalid arguments, missing required arguments, unknown
  tools and not-found ids on id lookups; never empty search results or
  find/list/search misses. Rationale: those are mistakes the model controls;
  an empty search is information. Date: 2026-09-28.
- Decision: 0.02 per mistake, capped at 0.1 per episode, in the episode
  reward. Rationale: the 0.2 truncation penalty became the largest reward gap
  in a group; a capped penalty stays below typical partial-credit
  differences. Date: 2026-09-28.
- Decision: truncation penalty 0.1 (user: 0.05 relaxed it too far). Date:
  2026-09-28.
- Decision (user, 2026-09-28): continue SAMPO from its update 120, where
  entropy (about 0.28) and KL to the base model (about 0.05) were lower than
  the KL run's at the same point, for 100 updates. The per-turn mistake
  penalty is 0.02, matching the episode penalty. Date: 2026-09-28.
- Decision: resume by `--model-from-run` and `--model-checkpoint-step 120`,
  not an exact resume, so the new rate and scheduler apply; TRL uses the
  frozen starting adapter as the KL reference. SAMPO keeps no curriculum
  state to warm-start from, so `--curriculum-from-run` is not passed.
  Date: 2026-09-28.
- Decision (user): 24 x 6 per update, with vLLM (`max_num_seqs`) and the
  Verifiers workers (12 x 12) sized for all 144 episodes at once so a round
  takes one episode wave, as 16 x 4 at 64 concurrency did. KV cache: at 64
  episodes 14.8 GB (539K tokens) peaked at 100%, about 8.4K tokens per
  episode; 41 GiB (about 1.6M tokens) gives 11K per episode at 144.
  Date: 2026-09-28.
- Decision (user): learning rate 5e-5, the SAMPO run's own rate, instead of
  3e-5. `job plan` flagged 3e-5 at LoRA alpha 8 as 1.3x below the lowest
  Tinker RL recipe, and the continuation should not learn slower than the run
  it extends. Date: 2026-09-28.
- Decision (user): keep the 160-task mix the SAMPO run trained on
  (`lfm26-automationbench-mix-v2`). Refill rounds draw from those 160
  candidates, so active sampling allows 6 rounds (24 x 6 = 144 reserved
  prompts); the earlier runs needed 2 rounds on most updates and at most 5.
  Date: 2026-09-28.

## Outcomes & Retrospective

Not yet run.

## Context and Orientation

- Adapter: `environments/automationbench_v1` in
  `carbonteq-ai/verifiers-environments`, branched from 0.4.2 (`3a486b0a`,
  per-turn rewards). `limited_tools.py` registers tools; `turn_rewards.py`
  classifies failures for SAMPO; `taskset.py` holds the rewards.
- Posttrain: `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`
  (environments, inference bindings, training settings, evaluation plans) and
  `apps/lab/.posttrain/work_packages/`. Truncation shaping is
  `shape_online_reward` in `packages/train/src/posttrain/train/profiles.py`.

## Plan of Work

1. Adapter 0.4.3: in `limited_tools.py`, accept a decoded object for
   parameters declared `str | None` and re-encode it as the JSON string the
   tool expects; add `classify_tool_result` (mistake kinds and empty results);
   add an optional `mistake_penalty` task config with per-mistake value and
   cap, applied as a negative reward component and reported as metrics;
   make turn rewards penalize mistakes only. Tests beside the adapter.
2. Posttrain lab: `automationbench-lfm26-sampo-turns-v2` (SAMPO v1's task
   mix plus adapter 0.4.3 and the mistake penalty, sampling temperature 0.5),
   rollout and training bindings for 144 concurrent episodes, settings
   (24 x 6, 5e-5, `constant_with_warmup`, warmup 5/100, beta 0.01,
   truncation 0.1, 100 updates), a work package, and held-out evaluation at
   temperature 0.5 beside the existing 0.1 suite.
3. Launch from a clean worktree with `--provider dstack --model-from-run
   lfm26-sampo-turns-150-lr5e5-kl5e3-20260927-r2 --model-checkpoint-step 120
   --timeout-seconds 172800`.

## Concrete Steps

Adapter: `uv run pytest environments/automationbench_v1/tests -q` in the
`verifiers-environments` checkout. Posttrain: `posttrain job plan` on the new
work package, then `posttrain job run ... --provider dstack`.

## Validation and Acceptance

- Adapter tests: an Airtable `fields_json` string reaches the tool as a string
  even after FastMCP decoding; each observed error text classifies as shown in
  Surprises; empty searches are not penalized; the penalty caps at 0.1.
- Run: mistakes per episode, finished-episode reward, truncation, entropy and
  tool calls per episode are tracked; held-out evaluations at 0.1 and 0.5 every
  20 updates are compared with base and KL update 140 in Observatory.

## Idempotence and Recovery

Catalog additions are new ids; nothing existing changes. A failed launch is
retried under a new run id.

## Artifacts and Notes

Trace sample: 160 trace ids from KL updates 126-150; classifier script kept in
the session scratchpad.

## Interfaces and Dependencies

New adapter revision pinned only by the new lab catalog entries.
