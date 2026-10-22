# Test 4B-generated teacher guidance in two bounded SDPO experiments

SUPERSEDED before any GPU training/update by the newer user authorization in
sdpo-guidance-five-mtp.md: five experiments of ten updates, native MTP instead
of the proposed DSpark path. Keep this preparation/data audit as historical
evidence; do not launch the old two-arm proposal or count its budget twice.

This living plan follows docs/templates/PLAN.md. Maintain Progress, Surprises &
Discoveries, Decision Log and Outcomes & Retrospective. Work in
/home/hammad/projects/worktrees/rl-rft, codex/rft-warm-start. Project research
only; no frozen framework contract or dependency-pin changes. Preserve dirty
work, old attempts, and held queues. No27B model participates in any new phase.

## Purpose / Big Picture

The user authorizes20 optimizer updates TOTAL, two experiments of at most10
updates each, concurrency40 and a compatible DSpark draft. A local Qwen3.5-4B
first interprets historical Luna execution and current goal/harm declarations
into compact task guidance. Only the self-teacher receives this guidance.
Student rollouts use ordinary task/tool context. The aim is better native
decision making, not lower loss or prompt-following with student hints.

## Progress

- [x] Workstation GPU/Docker checked free. Existing SDPO scripts inspected;
  they hardcode32slots and must explicitly qualify40 rather than relabel32.
- [x] Identified Qwen3.5-4B DSpark candidate
  shanjiaz/qwen3_5_4b_perfectblend_regen_dspark revision
  3842177e5e0f0445328cb1ceda44ff32e1da530b. Native target architecture and
  vocab match metadata; actual runtime/LoRA/parity support is NOT yet qualified.
- [ ] CPU screen training-eligible Luna sources for current manifest authority,
  available reference tools and task instruction binding. Freeze a diverse
  task list and explain historical-state differences. Never silently copy old
  identifiers/answers into a current task. Exclude all protected evaluation sets.
- [ ] Prepare4B-only compact guidance, retain rejected stages and missing task
  coverage. Historical variants may guide procedure only, not establish current
  state or correctness. No external model/judge calls or27B generation.
- [ ] Pin/cache draft weights without executing repository remote code; qualify
  selected vLLM,40inflight,batch invariance,target logprob parity,native tools,
  adapter refresh and full-context SDPO backward with zero optimizer updates.
- [ ] Experiment A: fresh4B,10updates,task-wide compact teacher guidance plus
  factual execution feedback. Native evaluation at0/10. Persist exact receipt.
- [ ] Inspect A training signals/errors before fixing B; never tune B using
  evaluation-task outcomes. Experiment B: fresh same base,up to10updates,
  decision-focused guidance/feedback. Hold other settings matched.
- [ ] Reconcile updates<=20combined, native evaluation identities/coverage,
  errors, obligations/harms, exclusions and adapter versions; paired task
  uncertainty against each fresh base. Final comparison and pause; no scaling.

## Surprises & Discoveries

The previous guided Sales trace passed legacy completion while the native
audit-note obligation failed. Context selection must inspect manifest findings,
not infer correctness from the legacy success bit. Existing SDPO association
re-evaluates original student action tokens with added teacher context; guidance
alone does not guarantee corrective gradients at a bad reasoning prefix.

## Decision Log

- Decision: No27B critic, solver or context generator. All new context generation
  and the self-teacher use the pinned local4B; DSpark is only a token proposer.
  Rationale: direct user correction. Date:2026-10-19.
- Decision:20updates TOTAL, at most10per experiment; qualification has zero
  optimizer updates and must not covertly consume/reset that budget. Compare two
  summarized-context variants rather than spending an arm on already studied
  raw dumps. No claim of a new matched raw-context causal control.
- Decision: Proper evaluation means fixed native heldout62,62x5,20turns,8192
  reply,32768episode,65536context,greedy thinking for both arms at0/10. This
  benchmark has been inspected repeatedly and is research monitoring, not a
  pristine final test. Do not choose B or replace missing tasks from its results.
- Decision: Candidate training profile32K/4Kreply,12288episode,8turns,
 40episode/server slots,thinking,LoRA16alpha32dropout0,batch32,micro1,
 LR1e-6 constant,reverseKL1,top20+tail,teacher update rate.01. Keep fixed across
  arms; whole incomplete/overlong transition exclusion, no truncation or gold CE.
  Each arm hard cap7200seconds including startup/gates/training/evaluation.

## Context and Orientation

Existing scripts/qualification/sdpo_agent_ablation.py owns native student
collection, teacher rescoring, top-k SDPO loss, EMA adapter and saved updates.
self_feedback_vllm.py owns the optimized native rollout and refresh lifecycle.
luna_solver_stages.py and luna_policy_guidance.py supply source-separated
guidance construction. Extend only opt-in project paths; do not mutate running
capsules or resume old SELF/SDPO/LFM jobs. Use immutable new capsules under
/var/lib/posttrain/qualifications/sdpo-guidance20-20261019-* and local evidence
docs/research/evidence/sdpo-guidance20-20261019. Actual names/digests/receipts
must be recorded immediately. Shared checkpoint/pin publications are prohibited.

## Plan of Work and Concrete Steps

First audit data in the existing pinned offline image, preserving source hashes
and every exclusion. Use existing training exclusions, plus an exact overlap
check against fixtures/automationbench_heldout62_v1.json. Prepare guidance with
current tools/manifest and historical procedures explicitly distinguished.
Then add explicit40-slot and speculative-config options, qualify them without
updates, and implement teacher-only guidance construction and audited native
obligation feedback. Preserve original student prefixes/actions and same-mode
gradient/Adam diagnostics. Freeze each arm before launch; inspect real GPU and
retained owned states before every submission. No automatic duplicate/restart.

SSH uses timeout25, -F /dev/null, BatchMode=yes, IdentitiesOnly=yes,
IdentityAgent=none,GSSAPIAuthentication=no,ConnectTimeout=8,key
/home/hammad/projects/ai-infra/.state/ssh/dstack_worker,dstack@192.168.110.12.
Use existing offline LAN image sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337,
runtime self-campaign-2048-20261017-r5 mounted read only, Qwen4B revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a BF16. No paid capacity.

## Validation and Acceptance

CPU checks must cover source/task binding, protected-set exclusion, context
leakage, no target substitution, complete transition admission, per-arm and
combined update caps,40-slot declaration, and repeatable context rendering.
GPU gates require actual40peak,correct target logprobs and refreshed LoRA,
native call/result/continuation,finite backward and unchanged weights in zero-
update gates. Then verify exactly10updates per completed arm, independent fresh
initializations, fixed evaluation populations, no hidden-context student input,
native obligation/harm coverage and conservative missing-score treatment.
Lower loss/teacher preference alone is not acceptance or reward improvement.

## Idempotence and Recovery

Never overwrite an existing capsule or resubmit a retained container. Failed
preparation/qualification remains evidence; repair under a new identity only
after diagnosing it. Never silently reduce concurrency, omit the requested
draft, change task/turn/token budgets or weaken parity. If an essential gate
cannot be met, report the concrete blocker and preserve the remaining updates.
No crash resume or full training automatically follows a local pass.

## Outcomes & Retrospective

Preparation only. No new SDPO update or GPU job has been launched yet.

## Interfaces and Dependencies / Artifacts

All code is project-owned under scripts/qualification. Keep native Verifiers
traces authoritative and compute summaries from them. DSpark's public model
card is provenance, not runtime qualification. Pin its exact weight revision;
do not run remote config.py or change existing framework dependency pins.
