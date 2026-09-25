# Qualify VORTEX next variance-aware curriculum

This is a living execution plan under `docs/templates/PLAN.md`. Its present milestone is a research simulation. Production adoption is a later milestone with its own qualification gate.

## Purpose / Big Picture

VORTEX currently allocates a discovery quota to tasks never seen before and ranks familiar tasks partly by whether their reward groups varied. The next version should allocate every candidate from two lanes: 80% favor reliable, meaningful variation and 20% add a bias toward uncertain or stale estimates. Both lanes can choose every task. When evidence is absent, classes are equally weighted and tasks within a class are equally weighted. The user can inspect a seven-run replay and a policy-change stress test before any production behavior changes.

## Progress

- [x] (2026-09-23) Read current controller, canonical observation boundary, and existing seven-run replay; preserved the dirty production worktree.
- [x] (2026-09-23) Exported a new compact snapshot with four numeric rewards per group and run provenance.
- [x] (2026-09-23) Built a research selector with two soft lanes, bounded variance magnitude, repeated-group stability, class/task pooling, and policy-step aging; 12 behavioral tests for old and new simulations pass.
- [x] (2026-09-23) Completed paired simulation across six replayable runs and two scenarios; audited task coverage and saturation behavior.
- [x] (2026-09-23) Recorded the negative efficiency result and production qualification gate in `docs/research/proposals/simulations/vortex-next-analysis.md`.
- [x] (2026-09-23) Verified the retained VORTEX profile uses unscaled centered group advantages, so raw variance must be evaluated separately from binary admission.
- [x] (2026-09-23) Completed paired yield-first soft-routing ablation, group-std-per-output-token comparison, and exploration-bonus sensitivity. Recorded the coverage tradeoff and no-duplicate replay test in `docs/research/proposals/simulations/vortex-simple-analysis.md`.
- [x] (2026-09-23) Amended the frozen curriculum contract for a versioned yield-first policy and hard same-step task uniqueness.
- [x] (2026-09-23) Implemented yield-first scoring with checkpointed aged binary-yield moments; old quota selection remains selectable and version-4 quota snapshots remain readable.
- [x] (2026-09-23) Enforced no duplicates across initial and refill decisions for all curriculum policies; impossible batches fail before a partial decision.
- [x] (2026-09-23) Added and statically validated the matched v4 LFM selection/work package; configured population is 160 tasks against a worst-case 100-task step.
- [x] (2026-09-23) Replayed the production controller on six sources and two scenarios; it reduced candidates per retained group in all 12 source/scenario pairs. Focused tests, job-intent plan, Ruff, targeted Pyright, import contracts, and diff checks pass.
- [x] (2026-09-23) Created a 15-minute thread follow-up to watch the running v3 job and apply publication/launch gates after it is terminal; no follow-on job has been submitted.
- [x] (2026-09-23 14:55Z) Confirmed exact v3 Trackio run succeeded at 14:44:48Z without recorded error or alert; no matching yield-first run exists.
- [x] (2026-09-23 15:10Z) Reran the actual production-controller replay with ten seeds; the generated JSON is byte-for-byte identical to `vortex_simple_results.json`.
- [x] (2026-09-23 15:10Z) Materialized the v4 environment through the exact pinned AutomationBench adapter commit `9bbd3116e6b5a444d6cf103dce18b1866ae32787` in a disposable source archive: 160 selected tasks, 160 distinct indices, and 160 distinct task names. Removed the temporary archive after the check.
- [x] (2026-09-23) Isolated the candidate in branch `codex/vortex-yield-first-release`; source-ready checks and the quality workflow passed at `583f21e7`. Canceled release workflow `35881521633` before its protected candidate job could publish the stale 4K profile.
- [x] (2026-09-23) Read the separate run-analysis task's truncation report. The repeated-task comparison at step 19 had 23/44 truncated trajectories versus 13/44 at earlier observations; truncated trajectories averaged 0.028 reward versus 0.434 for non-truncated ones. The report associates about 0.089 of a 0.198 paired reward drop with the changed truncation mix, without establishing causality. Independent full-run trace inspection found 492/1284 truncated trajectories and sampled `finish_reason=length` calls at exactly 4096 tokens. V3's 5 GiB KV pool reached 100% peak usage.
- [x] (2026-09-23) Revised the unpublished v4 work package to use an 8K per-call output cap, 16K episode cap, 32K training/serving context, and a 9 GiB KV pool. This changes both controller and rollout budget; it is not a policy-only A/B. Catalog and detached job resolution pass locally.
- [x] (2026-09-23) Resolved the earlier dirty-checkout release blocker by using the clean isolated branch, which retains the coherent dev24 lock. `posttrain-release check` passes. The shared checkout's dev25 work remains untouched.
- [x] (2026-09-23) Revised v4 source validation: 1,750 tests passed, 31 skipped; Pyright, Ruff, import contracts, detached work-package planning, and release consistency check passed.
- [ ] Push the revised source and wait for a new quality workflow, protected release approval, exact-profile GPU qualification, and immutable publication. Then submit one fresh run and monitor three optimizer updates.

## Surprises & Discoveries

The older compact snapshot retained only a binary useful flag, so variance magnitude could not be recovered from it. Trackio still holds the numeric rewards, and the read-only export projects four scalars per group without prompts or tool transcripts. The latest VORTEX run grew from 680 to 840 traces between snapshots; each result must name its exact source digest. Reward variance is a proxy for group-relative advantage, not proof of downstream model improvement. Raw variance magnitude is also not proportional to normalized GRPO update size.

The retained VORTEX v3 selection actually sets `advantage_scaling: none`. The previous sentence about normalized GRPO applies to other settings, not that run: here centered reward magnitude can affect update strength. Offline comparisons report within-group standard deviation per rollout cost as an amplitude proxy alongside binary usable-group admission; variance remains its squared diagnostic.

The cross-task report was generated in the separate Codex task titled “Analyze approved run purges” from local Observatory's repeated-task view. Its fixed-stratum calculation is observational: truncation can lower reward, but the same tasks may also become harder for other reasons. The rollout evidence distinguishes a 4096-token *model-call* limit from aggregate multi-turn completion tokens, which can exceed 4096. Raising the call limit therefore requires coordinated prompt/context and KV headroom. The 9 GiB KV setting is a candidate, not live-qualified capacity.

## Decision Log

- Decision: Keep the 80/20 lane mixture but let both lanes select from the whole inventory. Rationale: the 20% lane should smoothly shift from novel tasks to revisits as evidence accumulates, with no hard gate requiring a new task. Date/Author: 2026-09-23, Codex.
- Decision: Bound within-group population variance as `variance / (variance + scale²)` and penalize unstable history across groups. Rationale: tiny gaps should not count as equal to meaningful gaps, while one lucky outlier should not dominate repeated evidence. `scale=0.1` is a testable research setting on rewards normalized to [0,1], not a qualified reward threshold. Date/Author: 2026-09-23, Codex.
- Decision: Age observation weight by optimizer step, and shrink task estimates toward class peers. Rationale: an apparently mastered or stuck task can change after policy updates, while a cold task should inherit class evidence. Date/Author: 2026-09-23, Codex.
- Decision: Compare against the real current controller and simple ablations under identical empirical draws. Rationale: a synthetic win against uniform alone would not establish incremental value. Date/Author: 2026-09-23, Codex.
- Decision: Add a simpler yield-first full-pool ablation without changing the production controller. Rationale: the VORTEX selection uses `advantage_scaling: none`, so admission frequency and centered reward magnitude need separate accounting. Date/Author: 2026-09-23, Codex.
- Decision: A task may appear at most once within an optimizer step; later-step reuse is necessary for rechecks. If the distinct eligible inventory cannot fill a requested batch, report a shortfall instead of silently repeating a task. Date/Author: 2026-09-23, Codex.
- Decision: Preserve the original quota policy as the baseline and introduce `yield_first` as a separate selectable policy. Rationale: full-pool exploration cannot silently reinterpret the existing cumulative discovery floor. Date/Author: 2026-09-23, Codex.
- Decision: Qualify weight 4 as the coverage-preserving exploratory setting and weight 0.2 as an efficiency sensitivity, not a production default. Rationale: the paired replay found an efficiency-versus-coverage tradeoff. Date/Author: 2026-09-23, Codex.
- Decision: Amend the unpublished yield-first v4 selection to test the new controller with larger output and context budgets. Rationale: v3 had substantial length-truncated trajectories and its 5 GiB KV pool saturated; releasing the inherited 4K cap would knowingly repeat a major limitation. This sacrifices a clean policy-only causal comparison; a separate matched-budget baseline is needed for controller attribution. Date/Author: 2026-09-23, Codex.

## Outcomes & Retrospective

The two-lane full-pool mechanism behaved as designed, but the proposed scoring rule lost to the existing controller: across six source runs, stationary candidates per retained group were 1.536 versus 1.471, and synthetic drift candidates per retained group were 1.965 versus 1.731. We therefore do not recommend a production change from this prototype. The stronger next candidate should keep the current calibrated useful-group probability as its primary score while softening the hard novelty pool. The simulator is offline and does not update weights, so it cannot certify learning rate, final reward, or GPU-hour savings. Before adoption, require a same-checkpoint live comparison with GPU-time accounting, retained group quality, class coverage, and held-out evaluation.

The subsequent yield-first soft-routing replay improves candidates per retained group to 1.383 versus 1.471 stationary and 1.604 versus 1.731 under synthetic drift at exploratory bonus 0.2, but reduces mean covered tasks. Bonus 4 recovers most coverage while keeping a smaller efficiency gain. The VORTEX job's unscaled advantages make group std per output-token cost a relevant *amplitude proxy*; neither it nor admission efficiency proves downstream learning. The no-duplicate contract passed a four-round research test, which motivated replacing the production controller's exhausted-inventory duplicate fallback with a hard capacity error. See `docs/research/proposals/simulations/vortex-simple-analysis.md`.

The production `yield_first` implementation at bonus 4 produced 1.428 versus 1.471 candidates per retained group stationary and 1.664 versus 1.731 under synthetic drift; all 12 source/scenario pairs improved. It preserved average coverage of 151.1 and 155.4 tasks. This is a replay of the code intended for launch, still not a learning or GPU-hour result. The hard same-step uniqueness rule is implemented and tested, including ten refills of ten distinct tasks from a 160-task inventory. The v3 run succeeded at 14:44:48Z on 2026-09-23. The previous full-suite failures and release check mismatch came from the shared dirty checkout, not the isolated candidate: the candidate now passes 1,750 tests, Pyright, Ruff, import contracts, and the release check. The job image containing the revised code and output budget has not yet been immutably published, so the follow-on run remains unsubmitted.

## Context and Orientation

`packages/train/src/posttrain/train/adaptive_curriculum.py` owns the current production controller. `docs/research/proposals/simulations/automationbench_replay.py` exports compact evidence from seven retained runs. `docs/research/proposals/simulations/vortex_next.py` is the new research selector and paired simulator. A group is four rollouts of one task. The reward mean distinguishes constant maximum-reward groups from constant low-reward groups for diagnosis; variance controls the sampling score. A class is the AutomationBench domain. A stale estimate has accumulated few effective observations after exponential down-weighting by policy step.

## Plan of Work

Preserve the numeric reward vector and source identity in `vortex_next_groups.json`. In the VORTEX next sampler, use a prior on bounded group variation and borrow peer-task moments within each class. Calculate a task's estimated variation and uncertainty from recent weighted observations. Its normal-lane weight is the estimated variation tempered by instability between groups. Its exploration-lane weight adds an uncertainty bonus. Normalize at the class level first, using each class's mean task weight, then normalize tasks within the chosen class. This gives classes equal cold-start weight even if class sizes differ. Simulate 100 steps with eight retained groups and up to four sampling rounds, comparing uniform, current controller, binary-score and magnitude-score ablations, and VORTEX next. The stress scenario makes a stable task cohort reach maximum reward after step 40 and a previously flat cohort regain variation after step 60; these artificial changes qualify aging behavior only.

## Concrete Steps

From `/home/hammad/projects/rl`, run `uv run python docs/research/proposals/simulations/automationbench_replay.py export --output docs/research/proposals/simulations/vortex_next_groups.json`, then `uv run python docs/research/proposals/simulations/vortex_next.py --input docs/research/proposals/simulations/vortex_next_groups.json --output docs/research/proposals/simulations/vortex_next_results.json --seeds 10`. The export reads Trackio and writes only numeric group evidence. The simulator is fully offline. Run `uv run pytest -q docs/research/proposals/simulations/tests`, `uv run ruff check docs/research/proposals/simulations/vortex_next.py docs/research/proposals/simulations/automationbench_replay.py`, and `git diff --check`.

## Validation and Acceptance

The snapshot must reconcile per-run trace counts and have exactly four finite rewards in each eligible group. Cold-start tests must show class equality despite unequal class sizes; a zero uncertainty bonus must make both lane distributions identical. A low mean reward group with variation must score the same as a high mean reward group with equal variation. One outlier must rank below repeated consistent variation after sufficient history. Policy aging must restore recheck uncertainty. Every replayed optimizer step must have zero duplicate task IDs across all refill rounds; later-step reuse remains allowed. The paired replay reports retained groups, candidate groups per retained group, completion, round count, new-task share, coverage, and constant max/low groups. Any improvement claim must state its source run, scenario, seed spread, and observational limitations.

## Idempotence and Recovery

The export can be rerun with the same explicit output path; a live run may have grown, so preserve a copy if comparing snapshots. The simulation is deterministic for a fixed snapshot and seed. It does not mutate remote runs, the production controller, or model weights. If an export fails partway, the old output remains until the final write.

## Artifacts and Notes

The earlier binary replay and its report remain under `docs/research/proposals/simulations/`. The next-version results cite source and prototype SHA-256 digests. The compact source contains numeric rewards and task identifiers only; it excludes raw prompts and tool payloads.

## Interfaces and Dependencies

The exporter uses the pinned `trackio.Api` read path. The simulator uses the existing `posttrain.train.AdaptiveCurriculumController` as its baseline and Python standard library for its proposed selector. The original research milestone did not amend the frozen baseline. The subsequent production milestone does: `docs/post-training/README.md`, `02-primitives.md`, and `05-apis.md` now state the new selection option and hard step-uniqueness invariant.

## Production milestone: yield-first selection and delayed live qualification

Implement this in `/home/hammad/projects/rl`; no sibling fork owns task selection. In `packages/train/src/posttrain/train/profiles.py` and `catalog_schema.py`, give `AdaptiveCurriculum` an explicit `quota` or `yield_first` policy and policy-specific exploration share, uncertainty weight, and evidence-aging controls. Keep old settings and jobs resolving to `quota`. In `packages/train/src/posttrain/train/adaptive_curriculum.py`, share inventory, useful-group prediction, state journaling, and checkpointing, but route `yield_first` through full-inventory class-then-task sampling. The ordinary score remains the current useful-group prediction; the exploration lane adds uncertainty based on aged binary group-yield observations shrunk toward class peers. Do not use raw reward-variance magnitude or a between-group stability penalty for ranking. Record lane, probability, policy, and coefficients in evidence. Refuse a batch before mutating decision state when remaining distinct tasks are fewer than requested. Keep selected IDs in the checkpoint so a resumed step cannot repeat one.

The target job requests ten retained groups of four rollouts and permits up to ten candidate batches, so worst-case refill demand is 100 distinct task identities in one optimizer step. The pinned task-mix fixture describes a without-replacement 160-task selection from 800 available tasks, leaving 60 identities even after all ten maximum-size rounds. A production-shaped ten-round unit test confirms the controller selects 100 unique identities. Shortfall is therefore a defensive integrity check, not an anticipated operating mode. This differs from the research replay's eight groups and four rounds. Add tests in `packages/train/tests/test_adaptive_curriculum.py` and catalog tests for cold-start equality, exploration bias and aging, all-round uniqueness, exhausted-inventory failure without partial decision, and snapshot/resume equivalence. From `/home/hammad/projects/rl`, run `uv run pytest -q packages/train/tests/test_adaptive_curriculum.py apps/lab/tests/test_catalog.py`, `uv run ruff check packages/train/src/posttrain/train packages/train/tests/test_adaptive_curriculum.py`, `uv run lint-imports`, and `git diff --check`. From `/home/hammad/projects/rl/apps/lab`, validate and compile the exact new work package before submission.

V3 is terminal. Publish an immutable framework revision and verify that the selected runtime image contains the revised controller and 8K/16K budget. Before submission, qualify 32K context and 9 GiB KV on the RTX PRO with the exact c40 serving profile, plus trainer memory headroom; the v3 pool reached 100%, so catalog resolution alone is insufficient. Verify distinct-task capacity and all source gates. Then submit one fresh run ID and confirm three real optimizer updates, zero same-step duplicate selections, retained-group admission, truncation trend, GPU use, and a checkpoint. Submission must use one idempotent run identity; if a gate fails, do not launch or delete evidence, and record the failed gate.

## Revision note

2026-09-23: Created for the user-requested VORTEX next design and empirical simulation. Keep the same plan updated as results or design decisions change.

2026-09-23: Extended the existing research plan to a versioned production policy after same-step uniqueness became a hard requirement and a follow-on run was requested. Launch is gated on terminal status and immutable publication.
