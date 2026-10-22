# Audit and repair heldout62 information access before Qwen27B reevaluation

This living execution plan follows docs/templates/PLAN.md and the canonical
post-training baseline. The baseline meaning and package boundaries are unchanged.

## Purpose / Big Picture

Determine whether each of the62 frozen tasks supplies the tools and simulated
date needed to satisfy its requirements. Repair demonstrated information-access
defects without leaking rubric answers, changing task identities, or weakening
checks. Run a separately labelled Qwen3.5-27B evaluation on the repaired local
snapshot. Existing frozen runs remain historical evidence, never overwritten.
No training, external judge calls, commits or pushes are part of this work.

## Progress

- [x]2026-10-18: User requested "ditch the queue". Cancellation requested for
  queued repaired run qwen35-27b-heldout62-observable-20261018-r1 only. Do not
  automatically resubmit or resume it. Original running evaluation is preserved.
- [x]2026-10-18: User authorized all62-task audit, local repairs and reevaluation.
- [x] Located isolated environment checkout at exact frozen base with existing
  tested local LinkedIn serialization repair; preserve that fix and its tests.
- [x] Inventory62 prompts, initial clocks/resources, allowed native tools and
  manifest requirements; record explicit per-task findings and limitations.
- [x] Repair demonstrated missing-tool access and enable existing explicit world
  time context in a new selection. Real native tool probes retrieve the tax
  exemption and prepaid Slack correction;36 environment/codec tests pass.
- [x] Build immutable local snapshot, retain patch/tree/wheel digests and compose
  separate Qwen27B package preserving model, sampling and episode budgets.
  All six environment wheels build/install together; installed wheel repeats
  all62-task audit. Full image passed offline qualification and publication.
- [x] Repaired run accepted and currently queued behind original GPU owner:
  qwen35-27b-heldout62-observable-20261018-r1, provider
  pt-66e62a0ecce949676f225c4d, image
  sha256:7cc71d59589dc48e06e139d1b8351310ad1b6257a1b75638b67c8b3006310be1.
- [ ] Monitor accepted job to completion; audit all310 episodes and report
  remaining model errors and benchmark limitations. Never resubmit accepted ID.

## Context and Orientation

Framework: /home/hammad/projects/worktrees/rl-rft, codex/rft-warm-start.
Environment: /home/hammad/projects/worktrees/abv1-heldout62-scoring-fix,
base a0c7378b179e2b267382b8125192876502d82f89. Its standalone package
environments/automationbench_v1 owns taskset loading, tools, schemas and scoring.
Read its AGENTS.md; commit/tag/publication requires explicit authorization. User
previously chose keep-local. Local project-path packaging is an experiment
snapshot, not a published source pin. Do not alter any running image/capsule.

Current historical run qwen35-27b-heldout62-20261018-r1, provider
pt-7baaaedecc3f76db86edbdb3, uses frozen source. Do not interrupt or modify it.
Task names SHA256 ed81cf1fe88f68bb8420bb46c5fdf3afde24660c7043812da680bb1c7bc9196f.
Native traces are replay authority. Task hashes will change when public tools or
clock context change, so results belong to a new comparison cohort.

## Plan of Work / Milestones

First generate a reproducible CPU audit from the exact selected taskset, then
review task-specific procedures and initial resources. Static service coverage
is a screening heuristic; it does not prove semantic solvability. For each
confirmed omission retain a witness showing the hidden prerequisite and the
missing retrieval route. Review all62 entries, not only observed failures.

Second make minimal opt-in environment changes, backed by regression tests that
exercise the real tool implementation. Existing world_time_context supplies the
declared timestamp without host-time fallback; enable it for all selected tasks
  so every declared clock is exposed.47 tasks lack declared clocks; preserve
  explicit dates or supplied durations instead of inventing host dates. Never expose goals, expected answers or hidden
scoring state as public context. Include the existing LinkedIn reload repair
to avoid known missing scores. If a task lacks a clock, report/resolve explicitly.

Third create a separate source snapshot and binding through supported project-path
packaging, with exact source-tree/patch/wheel hashes. Preserve BF16 model revision,
native thinking,65536context,8192reply,32768episode,20turns,62tasksx5 and8/8
concurrency. Validate composition and actual packaged imports before GPU launch.
Use fresh qwen35-27b-heldout62-observable-20261018-r1 only after checking retained
states, scheduler and real GPU occupancy. The normal dstack full-GPU allocation
may queue while the original run owns the only GPU; never start a direct second
GPU process or interrupt the original run.

## Concrete Steps

CPU work and evidence live under scripts/qualification and
docs/research/evidence/heldout62-observability-20261018 in rl-rft. Run tests from
the environment package with its .venv and exact local source. CLI plans take no
provider/registry. Normal job pack/run uses registry.lan/carbonteq/posttrain-lab;
run takes provider dstack and fresh local framework wheelhouse containing the
block_size fix. Use process-scoped bwrap DNS workaround if still required; do
not change system DNS or hosts. SSH direct192.168.110.12 with the protected key
and no agent/config, as documented in the Qwen27B plan.

## Validation and Acceptance

Audit covers62 exact names, both original and repaired tool sets, date and
resource visibility findings. Added tools are registered, serializable, limited
and actually retrieve required resources through normal native tool calls.
Clock context exactly matches initial meta.current_time; no host fallback.
Old defaults remain unchanged. Tests preserve rubric behavior and unknown-field
strictness. Repaired benchmark retains310 distinct episodes with5 per task,
reports score/errors/limits, and separately labels original versus repaired scores.
No missing scores zero-filled, no successful-episode pooling or general claim
that every remaining failure must be model error merely because static audit passes.

## Idempotence and Recovery

Inspect accepted executions, packaging processes and GPU containers before every
submission. Never reuse an accepted ID or mutate staged snapshots. Preserve any
failed test/run and repair under a new identity. No commit/push, public source or
registry release; only protected experiment packaging to existing workstation.

## Surprises & Discoveries

- Initial tax failure requires Gmail exemption notice but offers no email reader.
- world_time_context already exists opt-in at frozen source but current eval
  leaves it false. Prepaid task clock is February1 while the model guessed an
  earlier month from message timestamps.
- The existing strict clock flag rejects timeless tasks. Added a separate
  world_time_context_if_declared option; strict behavior/defaults unchanged.
- Real project-path packaging reproduced a framework digest mismatch: planner
  and CLI retain a canonical project-relative context digest, but wheel builder
  compared a Git-style package-root hash. Fixed comparison to the exact staged
  context, retaining pre/post-build drift rejection. Regression fails before
  fix;16 wheel-builder tests pass. Nine import-boundary contracts pass.
- Current snapshot leaves initial states, user prompts, all assertions and
  checks unchanged. Residual unscored support-manager identity and meeting
  timezone omissions are documented, not answered with invented fixture data.
  Four policy-conflict tasks require per-outcome authority review; remaining
  failures cannot all be labelled model mistakes from this static audit alone.

## Decision Log

-2026-10-18: Cancel queued repaired evaluation at user's request. Preserve audit,
  source snapshot/image and cancellation evidence; any future rerun needs new
  user authorization. The original active evaluation is not part of this stop.
-2026-10-18: Repair observability, not expected answers or thresholds. Keep the
  running frozen arm intact, then create a new comparison cohort. User authorized
  local repair/rerun; prior keep-local instruction still applies to source publication.
-2026-10-18: All audit gates pass; image7cc71d59589dc48e06e139d1b8351310ad1b6257a1b75638b67c8b3006310be1
  published through normal protected LAN packaging. Queue one exclusive GPU
  job behind active original provider pt-7baaaedecc3f76db86edbdb3; scheduler and
  direct processes show no unrelated active workload. Preserve the existing
  no-offers cleanup job and never attempt direct GPU concurrency.

## Outcomes & Retrospective

Audit/repair qualification complete:62 per-task reviews,15 clocks exposed,two
retrieval additions, native world/tool-definition validation for62/62, unchanged
states/user requests/assertions,36 environment regression tests and16 packager
tests. No Git source/release publication. Protected LAN experiment image is
published and repaired run accepted, currently queued behind the original run.
No repaired benchmark outcome is available yet. Inspect accepted-run.json and
retained execution before any action; never duplicate this run.

The immutable environment snapshot tree SHA is
532f4db53e2b88c4d1d88c9fc13e87763eaf1d71fe4892ee465ff6c3ddcdf641.
It includes taskset.py access additions and prior PostContent fix. All62 rows
and limitations are in audit.json; installed-wheel-audit.json repeats the audit
through the installed wheel. No global claim of semantic solvability is made.

## Artifacts and Notes / Interfaces and Dependencies

Existing local scoring repair plan: docs/plan/heldout62-scoring-fix-rerun.md.
Original baseline plan: docs/plan/qwen35-27b-heldout62-evaluation.md.
No dependency pins or framework capability contracts change.

Revision2026-10-18: completed audit and local access/clock repairs; repaired
the measured local-package digest comparison needed for normal job packaging.
