# Run SAMPO with the VORTEX curriculum on LFM2.5 AutomationBench, developed on an 8 GB GPU

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds. This document follows `docs/templates/PLAN.md`. It builds
on the checked-in plan `docs/plan/sampo-agentic-training.md`, which added SAMPO to
the framework; everything needed from that plan is repeated here.

## Purpose / Big Picture

Today a Posttrain user can train an LFM2.5 agent on AutomationBench with VORTEX, a
curriculum that picks which tasks to sample and refills groups of attempts whose
rewards are all equal. Every token of an attempt then receives the same credit,
derived from one number at the end of the episode. SAMPO (Stable Agentic
Multi-turn Policy Optimization, from the ICML 2026 ARLArena paper) adds a
per-turn signal: when two attempts at the same task reach the same observation,
the turn that led to a better outcome gets a positive advantage and the other a
negative one. No judge model is involved.

After this plan, a user can select SAMPO for the same LFM2.5 AutomationBench
workload and keep what makes VORTEX work: the yield-first curriculum, refilling
only the missing groups, tolerating a failed episode, the truncation penalty and
the per-token vLLM correction. They can develop and profile it on an 8 GB GPU with
LFM2.5-1.2B, then run it on the 96 GB RTX PRO 6000 with LFM2.5-2.6B. The run's
evidence shows how many turns received a step-level signal and how the update
time splits between rollout and training.

To see it working, run the 8 GB work package in this plan for three updates. The
run finishes, Trackio records `train/rl/anchor_group_size_mean` above 1 and a
nonzero step-advantage share, and a deliberately failed episode does not abort the
update.

## Progress

- [x] (2026-09-26 13:30Z) Measured anchor-state matching on 520 healthy VORTEX v5
  attempts: 41% of turns share a state with another attempt in their group, 50%
  after removing per-sample IDs and UUIDs.
- [x] (2026-09-26 13:40Z) Mapped the current SAMPO path and its gaps (see Context
  and Orientation).
- [x] (2026-09-26 15:30Z) Milestone 1: baseline SAMPO on 8 GB. After three setup
  fixes (task domains, vLLM memory share, shutdown error masking) run
  `lfm12-sampo-8gb-20260926-r6` failed with "dynamic sampling exhausted 3 candidate
  batches ... retained rows 0": all 12 LFM2.5-1.2B attempts scored 0.
- [x] (2026-09-26 16:40Z) Milestone 2 code: SAMPO refills only with VORTEX active
  sampling (dynamic sampling removed), goes through group admission, selects the
  adaptive curriculum, and computes advantages on admitted groups. 1969 tests pass.
- [x] (2026-09-26 21:10Z) Milestone 2 run: `lfm12-sampo-8gb-20260926-r13` completed
  three updates (800 s) on the 57 screened tasks, 8K context. Every group had
  differing rewards in the first round, so no refill ran (8 rows generated per
  update, 0 rejected); trainer peak memory 4.44 GiB. Update time 396 s (first,
  with warm-up), 155 s, 247 s, of which rollout 367, 132 and 217 s and the actor
  update 13-17 s. vLLM's KV cache held 21,670 tokens and peaked at 45-60%, so at 4
  concurrent sequences it is not the limit.
- [x] (2026-09-26 17:30Z) Milestone 3 code: the vLLM correction is selectable and
  defaults to VORTEX's per-token cap of 2.0; an optional truncation penalty shapes
  the episode reward before advantages; anchor keys drop the tool-call id and
  UUIDs (scheme `content-without-sample-ids@2`, recorded on the run). KL was
  already selectable. The 8 GB package uses a 0.2 truncation penalty.
- [x] (2026-09-26 18:10Z) Milestone 2 collection: run `lfm12-sampo-8gb-20260926-r11`
  filled its 8-row batch in three VORTEX rounds (8 rows generated, 0 kept; 8
  generated, 4 kept; only the missing group's 4 regenerated, 4 kept), with no
  failed or unscorable rollouts. The first actor update then ran out of memory in
  backward on a 12,288-token episode (see Surprises).
- [x] (2026-09-26 19:00Z) Lowered the 8 GB package to an 8,192-token context (12
  turns and 3,072 tokens per reply kept; prompt budget 5,120) and added a task
  screen, `screen/lfm2.5-1.2b/automationbench-screen-8k`: every simple-domain task
  and 14 short multi-step tasks, three attempts each, at the training budgets.
- [x] (2026-09-26 20:30Z) Screen `lfm12-screen-8k-20260926-r2`: 642 episodes,
  mean reward 0.54, median 3,313 tokens, none over 7K. Of 200 simple tasks, 56
  gave attempts with different rewards, 82 were always solved, 54 always failed
  and 8 gave the same partial credit; of the 14 multi-step tasks only
  `finance.vendor_spend_analysis` differed. The 8 GB package now uses
  `automationbench-lfm12-sampo-8gb-v2`, those 57 tasks, with up to 10 candidate
  batches.
- [x] (2026-09-26 22:10Z) Released renderers `carbonteq-v0.1.12.post1.dev2`
  (fork commit 6f712616, GitHub pre-release, published to `carbonteq/dev` by
  Posttrain run 36259417685) and pinned it, so training accepts the tool calls
  evaluation accepts.
- [x] (2026-09-26 22:40Z) Evaluation: a rollout whose final request exceeds the
  context is truncation and keeps its reward (fact calculator v7); status
  separates `truncated` from `partial`; an evaluation with no successful rollout
  fails the run. Canonical doc 06 amended.
- [x] (2026-09-26 22:50Z) Retired the six SAMPO gates the new contract cannot run
  (`sampo-extended`, `automationbench-sampo`, four `verl-sampo*`); their work
  packages stay as history.
- [x] (2026-09-26 21:10Z) Milestone 3 run (r13): the vLLM importance ratio averaged
  0.98 (range 0.59-1.46) and was never clamped at 2.0; turns shared an anchor
  state with 3.0-4.6 attempts on average (groups of 4); KL 0.0004 from update 2;
  entropy 0.43-0.53; gradient norm 0.03-0.06, never clipped. 6 of 24 rollouts
  were truncated, by a first reply reaching 3,072 tokens (4) or the 8K context
  (2). Training still used renderers 0.1.12.post1.dev1, without the tool-call
  repairs.
- [x] (2026-09-26 23:40Z) Released Posttrain 0.4.9 (PR #124, merge `2c21eabf`,
  tag `v0.4.9`): candidate run 36262557103 (8 GB GPU canary), renderers dev2
  promoted to stable (run 36263033868), final run 36263079405. Run
  `lfm12-sampo-8gb-20260926-r15` on the 0.4.9 images (renderers dev2 installed)
  completed three updates and parsed 3 tool calls that needed a repair.
  Run r14 had still used dev1, because job images take third-party packages from
  the job-kind image built from the committed runtime lock.
- [ ] ~~Milestone 4: speed on 8 GB~~ Out of scope (user decision, 2026-09-26): the
  8 GB work is for correctness; TurboQuant was only considered to fit training.
- [ ] Milestone 5: environment turn rewards in the AutomationBench adapter
  (separate repository).
- [ ] ~~Milestone 6: qualification on the RTX PRO 6000 with LFM2.5-2.6B~~ Out of
  scope for this round (user decision, 2026-09-26): the RTX PRO runs the KL job.

## Surprises & Discoveries

- Observation: on the 8 GB GPU a failed training step surfaced as a vLLM out-of-
  memory error. The rollout runtime's shutdown wakes the colocated engine, which
  ran out of memory and replaced the training error. Fixed in `trainer_lifecycle`
  (commit 5d705d64): the training error is kept, the shutdown failure attached as
  a note, and cached CUDA memory is released before shutdown.
- Observation: r11's out-of-memory allocation (384 MiB) is one 12,288 × 8,192
  fp32 copy that PEFT makes of an adapter input, because it keeps LoRA weights in
  fp32. Skipping the copy saves 0.47 GiB at 12,288 tokens and is exact under bf16
  autocast, but TRL's chunked-logits path (`logits_chunk_size`, used by every
  binding here) calls the inner model outside autocast, so there the LoRA matmul
  really runs in fp32 and skipping the copy fails with a dtype error (run r12).
  The change (ea8f156e) was reverted; the 8 GB package uses an 8K context instead.
  A bf16 chunked path would need a TRL change and alters numerics slightly.
- Observation: the first screen, `lfm12-screen-8k-20260926-r1`, reported success
  although all 642 rollouts failed in harness setup before reaching the model:
  Verifiers cdd2ec76 (pinned since 0.4.7) replaced the runtime's per-script lock
  dict with `LoopLocks`, and Posttrain's preinstalled-runtime override still called
  `.setdefault`. Every managed evaluation on 0.4.7 and 0.4.8 is affected; training
  does not use the override. Fixed by calling `LoopLocks.get`, with the test fake
  now using the real Verifiers type. The job status does not consider
  `eval/run/rollouts_failed`; whether it should is an open product question.
- Observation: training and evaluation parse the same LFM2.5 tool-call text
  differently. Evaluation uses the vLLM fork's `lfm2` parser, which repairs
  near-valid Python (an unescaped apostrophe in `subject='Let's Get Started'`,
  JSON with nested quotes, `month=07`, `from=`); training parses sampled tokens
  with the renderers fork's `LFM2ToolParser`, which had no repairs and dropped
  such calls silently. `simple.gmail_onboarding_welcome` scored 0 on all 4
  training attempts (every call dropped) and 1.0 on all 3 evaluation attempts.
  Fixed in the renderers fork by copying vLLM's repair helpers unchanged.
- Observation: of 56 LFM2.5-1.2B episodes on the 8 GB runs, 14 (25%) ended with a
  malformed tool call (usually bad quoting of a JSON string argument) that was
  dropped without an error, so the model never retried; 40 ended with a plain-text
  answer, often claiming work it had not done. The 1.2B scored 0 on 7 of the 8
  tasks it saw, including simple ones LFM2.5-2.6B always solves.
- Observation: LFM2.5-2.6B lost the final tool call of 11.7% of healthy VORTEX v5
  episodes (61 of 520): 28 calls sampled before `</think>` were swallowed as
  reasoning, 20 were malformed Python, 13 were cut off by the 4096-token reply cap.
  The first cause is fixed in the renderers fork (commit f1952b2, not yet
  released); the others end the episode silently instead of returning an error.

- Observation: anchor states match more often than expected on AutomationBench,
  although attempts diverge after the first turn.
  Evidence: exact keys 41% of turns (turn 1 100%, turn 2 43%, turn 3 44%, turn 4
  32%, turn 5 25%, turn 6+ 13-14%); keys without `tool_call_id` and UUIDs 50%.
  Measured on traces of run `lfm26-vortex-v5-150-dspark-opt-20260926-r1`,
  updates 41-56, 130 groups of 4.
- Observation: AutomationBench already exposes per-turn signals. 12.6% of tool
  calls return `success: false`; the failed share is 16.8% in zero-reward
  episodes, 10.2% in partial ones and 2.6% in solved ones. Almost every assertion
  checks one write action (email sent, sheet row, Slack message, SMS), so the turn
  that satisfied it can be credited.
  Evidence: same traces; assertion types from `info.automationbench.assertions`.
- Observation: `gmail_message_not_sent_to` assertions passed 0 of 121 times
  because the agent really emailed the forbidden address; it is a skill the
  episode-level reward never isolates, not an evaluator bug.

- Observation: six older SAMPO qualification packages fail `work-package validate`
  since SAMPO jobs gained the online-RL static checks (commit 9e2d83ea). The SAMPO
  runtime applies the same sampling rule, so they would already fail at launch:
  `qwen08b_sampo_10_qualification` and `automationbench_sampo_qualification`
  declare inconsistent sampling (the latter also a batch mismatch), and the four
  `qwen08b_verl_sampo_2*` packages also target veRL, which no longer runs SAMPO.
  They need new binding revisions or retirement; they do not block this plan.

## Decision Log

- Decision: do not return a parse error to the model for an unrecoverable tool
  call; the episode keeps ending there.
  Rationale: serving turns such a call into plain text and ends the episode, so
  a retry in training would make training and evaluation differ again; ending
  with the episode's reward also keeps a direct penalty on the malformed call,
  where a retry would place it inside a positively rewarded trajectory. The
  common near-valid calls are now repaired identically on both paths.
  Date/Author: 2026-09-26, Claude (recommended to the user).

- Decision: retire the six older SAMPO qualification gates instead of giving
  them new bindings.
  Rationale: the four veRL ones cannot run SAMPO (no active sampling); the Qwen
  alphabet-sort one samples greedily (environment temperature defaults to 0);
  the Qwen AutomationBench one has a batch mismatch and is covered by the LFM2.5
  8 GB package. All were experimental candidates, so release gating is
  unchanged.
  Date/Author: 2026-09-26, Claude, under the user's instruction to finish the
  SAMPO work.

- Decision: SAMPO's vLLM sampler-mismatch correction defaults to VORTEX's
  per-token truncation capped at 2.0 instead of the former hard-coded
  `sequence_truncate` in [0.1, 3.0].
  Rationale: the sequence mode multiplies token ratios over the whole episode;
  over 4,000-20,000 sampled tokens small differences compound and the clamp
  engages often. The correction is independent of SAMPO's sequence-level policy
  ratio, which is unchanged. Existing SAMPO capsules pick up the new default.
  Date/Author: 2026-09-26, Claude.

- Decision: remove dynamic sampling from SAMPO; it refills only with VORTEX active
  sampling (keep groups with differing episode rewards, generate only the missing
  groups from a reserved candidate pool), optionally with the adaptive curriculum.
  Rationale: dynamic sampling regenerates whole candidate batches and stopped the
  8 GB baseline after three rounds with no retained group; VORTEX's refill is the
  collection behaviour that works on AutomationBench. The veRL backend has no
  active sampling, so it now rejects SAMPO instead of running a different
  collection policy; its SAMPO capsules are not runnable until it gains one.
  Date/Author: 2026-09-26, user decision.

- Decision: develop on the 8 GB RTX 3070 Ti with LFM2.5-1.2B-thinking, then
  qualify on the RTX PRO 6000 with LFM2.5-2.6B.
  Rationale: LFM2.5-2.6B training plus colocated vLLM does not fit 8 GB.
  LFM2.5-1.2B shares the model family and renderer, and bindings
  `training/lfm2.5-1.2b-trl-lora-changed-weight-local@1` and
  `inference/lfm2.5-1.2b-vllm-changed-weight-local@1` already run AutomationBench
  on this card.
  Date/Author: 2026-09-26, Claude with the user.
- Decision: no external judge in this plan.
  Rationale: SAMPO's episode advantage comes from AutomationBench partial credit
  and its step advantage from anchor-state matching; both come from the
  environment. Judge-based turn grades (the Jev rubric work) belong in the
  AutomationBench rubric system later and can enter through the same per-turn
  reward slot.
  Date/Author: 2026-09-26, user decision.
- Decision: SAMPO keeps the VORTEX curriculum's statistics on the episode reward.
  Rationale: the curriculum ranks tasks by how often they produce groups with
  differing rewards. If step signals counted, almost every group would look
  useful and the curriculum would lose its signal.
  Date/Author: 2026-09-26, Claude.
- Decision: learning rate 5e-5 and KL 0.005 as the starting point for LFM2.5-2.6B.
  Rationale: see `docs/techniques/grpo/recipes/lfm2.5-2.6b-automationbench-vortex.md`;
  1e-5 barely learned, 2e-4 and 1e-4 drifted into entropy collapse.
  Date/Author: 2026-09-26, user decision.

## Outcomes & Retrospective

None yet.

## Context and Orientation

Posttrain is a Python 3.12 `uv` workspace. Training code lives in
`packages/train/src/posttrain/train`. Catalog selections for this lab live in
`apps/lab/.posttrain/catalog/*.yaml`, and runnable work packages in
`apps/lab/.posttrain/work_packages/*.yaml`. Runs are recorded in Trackio and read
with `tracking_source_for_project` from `posttrain_cli.execution_provider`.

Terms used in this plan:

- An **attempt** or **rollout** is one episode of the agent on one task. A
  **group** is several attempts at the same task in the same update (4 today).
- An **advantage** is the weight a token receives in the policy update: positive
  pushes its probability up, negative down.
- An **anchor state** is the observation (the last user or tool message) that a
  turn responded to. `_anchor_state_key` in
  `packages/train/src/posttrain/train/integrations/verifiers.py` hashes the whole
  message, including `tool_call_id`.
- **SAMPO advantages** are computed by `compute_sampo_advantages` in
  `packages/train/src/posttrain/train/sampo_advantages.py`: an episode advantage
  (reward centred within the group) plus `step_advantage_weight` times a turn
  advantage (the discounted return centred among turns of the group that share an
  anchor key). A turn whose key matches no other attempt gets turn advantage 0.
  If every turn carries an explicit `step_reward`, those rewards replace the
  sparse final reward in the discounted return.
- **VORTEX** is the OLMo 3 GRPO algorithm with active sampling (refill of groups
  whose rewards are all equal) and the adaptive curriculum
  (`adaptive_curriculum` on `GRPOSettings` in
  `packages/train/src/posttrain/train/profiles.py`, run by
  `packages/train/src/posttrain/train/backends/trl/policy_curriculum.py`).
- **The vLLM correction** re-weights tokens by the ratio between the trainer's
  and vLLM's probability of the sampled token, because vLLM samples with slightly
  different numerics.

How SAMPO runs today (TRL backend): `sampo()` in
`packages/train/src/posttrain/train/api.py` calls `run_sampo` in
`backends/trl/policy_optimization.py`, the same function GRPO uses, with plain
TRL `GRPOTrainer`. `backends/trl/policy_config.py` sets
`use_precomputed_advantages`, forces TRL's DAPO-style `dynamic_sampling`, sets
`importance_sampling_level="sequence"` and hard-codes the vLLM correction to
`sequence_truncate` clamped to [0.1, 3.0]. Rollouts are collected by the same
runner as VORTEX (`backends/trl/policy_rollouts.py`, `collection_runner.py`) and
SAMPO advantages are computed after collection in `policy_rollouts.py`.

The gaps this plan closes, all observed in code:

1. The curriculum wrapper applies only to `GRPORequest`
   (`policy_optimization.py`), and `SAMPOSettings` has no `adaptive_curriculum`,
   `active_sampling`, `truncation_penalty` or `importance_sampling_mode` field.
   Its schema forbids unknown fields.
2. TRL's dynamic sampling regenerates whole candidate batches, keeps groups with
   reward spread, and raises `RuntimeError` after `max_candidate_batches` (default
   3). At AutomationBench solve rates that is likely to stop a run.
3. SAMPO never goes through `admit_rollout_groups` (`policy_rollouts.py`), so one
   failed or timed-out episode raises `VerifiersRolloutFailure` and aborts the
   update. `_rollout_batch` also gives SAMPO no prompt-group IDs, so traces cannot
   be grouped.
4. `sequence_truncate` multiplies token ratios over the whole episode. Over
   4,000-20,000 sampled tokens small per-token differences compound, so the clamp
   will engage often. VORTEX uses `token_truncate` with maximum 2.0.
5. With one optimizer step per generation batch the policy ratio is exactly 1, so
   SAMPO's 0.003/0.004 sequence clip never engages. Stability must come from the
   learning rate and KL.
6. SAMPO has never run on LFM2.5 or with the speed options (async rollout
   execution, DSpark, `compile_decoder_layers`,
   `importance_sampling_from_training_logps`). Liger must stay off; TRL rejects it
   with precomputed advantages.

This plan changes the frozen product baseline: `SAMPOSettings` gains curriculum,
refill, truncation-penalty and vLLM-correction fields. Milestone 2 starts by
amending `docs/post-training/05-apis.md` in the SAMPO settings description, in
the same commit as the code.

## Plan of Work

Milestone 1 establishes the baseline on the 8 GB card without code changes. Add
catalog entries in `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`:
a `sampo-settings` entry `lfm2.5-1.2b/automationbench-sampo-local-8gb-v1` with 3
updates, 2 prompt groups of 4 attempts, `max_completion_length` 1024, learning rate
5e-5, `beta` 0.005, and the existing 1.2B training and inference bindings. Add a work
package `apps/lab/.posttrain/work_packages/lfm12_automationbench_sampo_local_8gb.yaml`
on the SAMPO job definition `train/trl-sampo@1`. Run it on the local provider and
record what happens: whether it completes, whether dynamic sampling exhausts its
candidates, how often the vLLM correction clamps, `train/rl/anchor_group_size_mean`,
peak GPU memory and the time split. Write the findings into Surprises & Discoveries.
This is a prototyping milestone: its output is evidence, not a supported
configuration.

Milestone 2 makes collection robust. In `profiles.py`, add optional
`adaptive_curriculum` and `active_sampling` fields to `SAMPOSettings`, with the same
types and validation as on `GRPOSettings`, and make `dynamic_sampling` optional
and mutually exclusive with `active_sampling`. In `policy_optimization.py`, apply
the adaptive-curriculum trainer to SAMPO requests when the settings carry a
curriculum. In `policy_config.py`, select TRL's active-sampling path (refill only
missing groups) instead of dynamic sampling when `active_sampling` is set. In
`policy_rollouts.py`, route SAMPO through `admit_rollout_groups` so a failed episode
drops its group rather than the update, and return prompt-group and rollout IDs for
SAMPO like GRPO. Extend `reward_admission.py` to accept `SAMPOSettings`. A group is
useful when its episode rewards differ, matching VORTEX; record separately how many
dropped groups had nonzero turn advantages, so a later decision can use it.

Milestone 3 fits the objective. Add `importance_sampling_mode` and its clip bounds
to `SAMPOSettings`, defaulting to the current `sequence_truncate` [0.1, 3.0] for
compatibility, and pass them through `policy_config.py`. Add `truncation_penalty`
with the GRPO semantics. `beta` already exists. In `verifiers.py`, change
`_anchor_state_key` to drop `tool_call_id` and replace UUID-shaped strings before
hashing, behind a versioned key scheme recorded in run attributes, so earlier runs
remain explainable. Log `train/rl/step_advantage_token_share` (the share of sampled
tokens whose turn advantage is nonzero) and the vLLM clamp fraction.

Milestone 4 profiles and speeds up the 8 GB loop. Use the actor-update split added
in Posttrain 0.4.8 (`train/rl/time/actor_forward_backward_seconds` and
`train/rl/time/optimizer_step_seconds`) plus rollout time. Try, one at a time and
recording each result, the TRL 1.12.0.post10 options `compile_decoder_layers` and
`importance_sampling_from_training_logps`, async `rollout_execution`, and CUDA graphs
instead of eager decoding in a new inference binding revision (the recorded
revision uses `enforce_eager`). Keep only changes that lower update time without
raising peak memory past the card.

Milestone 5 adds per-turn environment rewards in the separate
`carbonteq-ai/verifiers-environments` repository (environment
`environments/automationbench_v1`), not in this repository. After each turn the
adapter re-evaluates the task assertions and emits the change in assertions passed
minus a small penalty for each failed tool call, as a per-turn value under a stable
key. The per-turn values sum to the episode's partial credit plus the penalties.
Posttrain consumes it through a turn reward projection (`turn_reward_key` on the
SAMPO job), which already exists. That repository is committed and pinned first,
then the pin is updated here; the exact order is in Concrete Steps.

Milestone 6 qualifies on the RTX PRO 6000: LFM2.5-2.6B, bindings
`training/lfm2.5-2.6b-trl-lora-automationbench-local-g64-w8@2` and
`inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c64-4k@2`, 16 prompt groups
of 4, learning rate 5e-5, KL 0.005, yield-first curriculum, truncation penalty 0.2,
10 updates, then a full run compared with the VORTEX control
`lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r1`.

## Concrete Steps

All commands run from the repository root `/home/hammad/projects/rl-perf-guard`
unless stated. Launch runs from a clean worktree of committed code, because
Posttrain packs the working tree.

Validate a work package:

    cd apps/lab
    uv run --no-sync --package posttrain posttrain work-package validate \
      .posttrain/work_packages/lfm12_automationbench_sampo_local_8gb.yaml

Run it on the local 8 GB card (one GPU job at a time on this machine):

    cd apps/lab
    uv run --no-sync --package posttrain posttrain job run \
      .posttrain/work_packages/lfm12_automationbench_sampo_local_8gb.yaml \
      --provider local --run-id lfm12-sampo-8gb-<date>-r1

Run the focused tests after each milestone:

    uv run --no-sync pytest -q packages/train/tests/test_sampo.py \
      packages/train/tests/test_api.py apps/lab/tests

Milestone 5 order: commit and push the adapter change in
`carbonteq-ai/verifiers-environments`, pin its full commit in the framework
catalogs, dependency constraints and `uv.lock` here, then update the SAMPO work
packages to name the turn reward key.

## Validation and Acceptance

Milestone 1 is accepted when its findings are written down, whether or not the run
completes. Milestones 2-4 are accepted when the 8 GB work package completes three
updates with SAMPO, the curriculum and active sampling selected; when a test in
`packages/train/tests/test_sampo.py` shows one failed episode dropping its group
without aborting (failing before Milestone 2 and passing after); when Trackio shows
`train/rl/anchor_group_size_mean` above 1 and `train/rl/step_advantage_token_share`
above 0; and when the time split per update is recorded before and after Milestone
4. Milestone 6 is accepted when 10 LFM2.5-2.6B updates finish with entropy within
twice its early level. The full ladder must pass at each commit:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

## Idempotence and Recovery

Catalog entries and work packages are additive; new behaviour is opt-in through
new settings fields whose defaults keep existing SAMPO runs unchanged. A failed
local run leaves only its run record and workspace; rerun with a new `--run-id`. If
the 8 GB run exhausts GPU memory, lower `max_completion_length` or concurrency in a
new binding revision rather than editing a recorded one.

## Artifacts and Notes

The anchor-matching measurement used traces already downloaded from run
`lfm26-vortex-v5-150-dspark-opt-20260926-r1`; the script hashed the last user or
tool message before each sampled assistant turn exactly as `_anchor_state_key`
does, then again without `tool_call_id` and with UUIDs replaced.

## Interfaces and Dependencies

At the end of Milestone 3, `SAMPOSettings` in
`packages/train/src/posttrain/train/profiles.py` has, in addition to its current
fields:

    adaptive_curriculum: AdaptiveCurriculum | None = None
    active_sampling: ActiveGroupSampling | None = None
    dynamic_sampling: DynamicGroupSampling | None = DynamicGroupSampling(3)
    truncation_penalty: float | None = None
    importance_sampling_mode: str = "sequence_truncate"
    importance_sampling_clip_min: float | None = 0.1
    importance_sampling_clip_max: float | None = 3.0

with exactly one of `active_sampling` and `dynamic_sampling` set. The catalog
schema in `packages/train/src/posttrain/train/catalog_schema.py` accepts the same
fields for `sampo-settings`. No new external dependency is introduced. The TRL
fork `1.12.0.post10` configuration accepts active sampling together with
precomputed advantages (`trl/trainer/grpo_config.py`, the `active_sampling`
validation block); Milestone 2 must verify with a test that refill rounds carry
the precomputed advantages of retained and refilled groups through unchanged.
