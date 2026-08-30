# Agent runbook: implement dstack cloud execution without scope drift

This runbook is the execution companion for
`docs/plan/dstack-r2-registry-routing.md`. The plan remains authoritative for
architecture, interfaces, acceptance, and current progress. This file only
defines how an autonomous goal should execute that plan efficiently.

Do not create a second design, speculatively reopen completed decisions, or
duplicate the plan's evidence. This is not a smallest-diff-at-any-cost rule:
make evidence-backed improvements that increase correctness, maintainability,
operability, or developer experience when they are adjacent to the active
milestone. When implementation changes what is known, update the living plan
first and keep this runbook limited to execution mechanics.

## Authority and fixed decisions

Read these sources in order:

1. `AGENTS.md`
2. `docs/plan/dstack-r2-registry-routing.md`
3. The current milestone's named source, tests, and deployment files
4. Only the canonical `docs/post-training/` document governing a contract that
   the current milestone actually changes

The following decisions are presumptively closed. Revisit one only when current
implementation evidence shows it is incorrect, unsafe, or needlessly degrades
the active milestone's quality or developer experience; record that evidence
and revise the living plan before changing direction:

- Posttrain owns logical jobs, immutable image identity, evidence contracts,
  artifacts, and reconciliation.
- dstack owns placement, provider attempts, interruption truth, RunPod compute,
  run-scoped volume ownership, and cleanup state.
- ai-infra owns service deployment, credentials, registry topology, RustFS,
  ingress, hooks, backups, and production rollout.
- Cloudflare R2 stores only the external OCI registry's bytes.
- Trackio stores artifact bytes in CarbonTeq's self-hosted RustFS through its
  S3-compatible direct multipart contract. Trackio remains the artifact and
  digest authority.
- The universal base image stays unchanged. Preserve inherited OCI blobs and
  publish with no forced recompression.
- Images and owned resources are selected by immutable digests or stable
  logical/provider IDs. Timestamps belong only in receipt observations; never
  use them as image identity, cache lineage, or cleanup selectors.
- CUDA compatibility stays in the veRL child and is selected automatically by
  the image runtime before CUDA consumers import.
- dstack receives generic provider-neutral contracts. CarbonTeq hostnames,
  Ansible playbooks, R2, RustFS, and Trackio do not enter dstack core models.
- Security hardening is Milestone I, after functional qualification and before
  promotion. Preserve the existing credentials; do not regenerate, replace,
  or revoke them as part of this goal.
- Do not create new Git worktrees. Preserve unrelated dirty changes in every
  repository.

## Command to run as a goal

Paste the following block into a goal-capable Codex task. Starting the goal with
this command authorizes plan-scoped source, test, documentation, candidate
deployment, qualification, and the scoped commits and pushes needed to publish
immutable producer revisions on `codex/*` branches. It does not authorize
merging to main, deleting non-qualification resources, cancelling unrelated
jobs, weakening a release gate, or exposing additional private services.

```text
GOAL: Fully implement docs/plan/dstack-r2-registry-routing.md. First resolve its
remaining Milestone 0 edit-isolation preflight, then execute Production
Milestones A, B, C, D, E, F, G, H, I, and J in order. Continue until every
required Progress item is complete, every milestone's acceptance evidence is
retained, the final validation gates pass, bounded cloud fallback is promoted,
and rollback remains proven. Use
docs/plan/dstack-r2-cloud-execution-agent-runbook.md as the mandatory operating
loop. The living plan remains the architecture authority.

OPERATING LOOP:

1. Read AGENTS.md, this runbook, and only these parts of the living plan:
   Progress, the first incomplete Production Milestone, Concrete Steps,
   Validation and Acceptance, Idempotence and Recovery, and the latest Revision
   Note. Do not load unrelated historical milestones unless the current step
   explicitly depends on them.

2. Inspect git status and exact HEAD in rl, ../dstack, ../trackio, and
   ../ai-infra. Preserve unrelated changes. Do not create a worktree. If the
   living plan's Milestone 0 isolation item is still incomplete, resolve and
   record ownership/overlap before editing milestone files. Then select the
   first incomplete milestone in A -> B -> C -> D -> E -> F -> G -> H -> I -> J.
   Work on that milestone only.

3. Before editing, write a five-line milestone brief in the task commentary:
   objective; owning repository or repositories; files/interfaces in scope;
   focused validation; live-resource and rollback bound. Treat every other idea
   as deferred unless it blocks acceptance or qualifies as an adjacent quality
   improvement under this runbook.

4. Implement the smallest coherent release unit. Use apply_patch for authored
   edits. Put generic fixes in their owning fork, update its CARBONTEQ_FORK.md
   and framework consumer page, and update immutable pins only after the
   producer revision is committed and published. Never mix changes from two
   release units in one commit. Include adjacent improvements when they reduce
   complexity or operator effort, clarify typed errors or state, strengthen
   idempotency/recovery, improve tests and fixtures, or make the milestone's
   plan/apply/verify/cleanup workflow easier to use. Validate those improvements
   in the same release unit.

5. Run focused tests first. Fix them until green. Then run the milestone's
   integration or live qualification exactly once per candidate revision. Every
   external apply must have a read-only plan, bounded cost/resource intent,
   exact owned IDs, verification, and idempotent cleanup. Never create a second
   resource while the first resource's ownership or deletion is uncertain.

6. Retain a safe receipt under ai-infra .state/artifacts. Record component
   revisions, immutable image/artifact digests, provider and cleanup IDs, state
   transitions, timings, cost, and acceptance result. Never retain secrets,
   authorization headers, environment values, signed query strings, checkpoint
   bodies, or source contents.

7. Update the living plan's Progress, Surprises & Discoveries, Decision Log,
   Outcomes & Retrospective, and Revision Note as appropriate. Mark a milestone
   complete only after its focused tests, live acceptance, cleanup confirmation,
   and receipt all pass. Then return to step 1 for the next milestone.

8. After H, perform I without changing the working credentials: publish
   diagnostic redaction, validate least privilege and protected state, harden
   public method/path boundaries, run secret scans, and pass negative tests.
   Do not start J if any security result is red.

9. For J, wait until both retained LAN workers are healthy and idle. Do not
   cancel work or restart a busy worker. Promote immutable revisions, repeat LAN
   and RunPod canaries, enable fallback only for the allow-listed bounded
   project, observe the declared window, and prove rollback. If the fleet is
   busy, continue safe non-production work or wait; do not weaken the gate.

COMMUNICATION:
- Give one concise update when starting a milestone, after focused validation,
  before a billed or production mutation, and after cleanup/receipt.
- Do not narrate routine searches or repeat the plan.
- Ask the user only for new authority, an unavailable secret, an architectural
  choice that changes the plan, or a destructive/external action outside this
  command's scope.

DONE WHEN: Milestone 0 and Production Milestones A-J are complete in the living
plan; all provider Pods and run-scoped volumes owned by qualification are
absent; the same digest works on LAN and RunPod; Trackio/RustFS recovery is
verified; interruption, hooks, inventory, security, promotion, and rollback
receipts exist; no unrelated work was changed; and final repository validations
pass.
```

## Goal stack and unlock conditions

Only one goal may be active. Complete its `Done when` condition before unlocking
the next row.

| Goal | Owner sequence | Done when |
| --- | --- | --- |
| P0 — Edit isolation | All four repositories | Exact heads and dirty-file ownership recorded; milestone edit surfaces do not overwrite unrelated work |
| A — CUDA runtime | `rl` | One new veRL digest selects native/compat automatically on local and RunPod; inherited descriptors match; Pod absent |
| B — Registry admission | `dstack` -> `ai-infra` | Pending actual-job mirror creates no Pod; verified digest creates exactly one Pod; restart/timeout behavior proven |
| C — Trackio/RustFS | `trackio` if needed -> `rl` pin -> `ai-infra` | Public Trackio writes work; multipart checkpoint reaches self-hosted RustFS directly; digest, interruption, migration, backup, and restore pass |
| D — Provider attempts | `dstack` -> `ai-infra` | RunPod state is authoritative; forced spot disappearance records one interrupted attempt without terminating the logical run incorrectly |
| E — Lifecycle hooks | `dstack` -> `ai-infra` | Transactional outbox survives restart; receiver deduplicates; fixed argv/signed webhook and dead-letter paths pass |
| F — Run storage | `dstack` -> `ai-infra` | One logical spot run owns one fenced volume across attempts; terminal cleanup confirms Pod and volume absence |
| G — Recovery/inventory | `dstack` + integration owners -> `ai-infra` | Same-volume and cross-data-center recovery use immutable Trackio artifacts; inventory stays truthful through transition and cleanup |
| H — Resilience matrix | Candidate deployments only | Every declared functional failure/restart path passes with bounded resources and no leaks; production fallback remains off |
| I — Security | Owning forks -> `ai-infra` | Redaction, least privilege, ingress restrictions, secret scans, and negative tests pass; existing credentials are unchanged |
| J — Promotion | `ai-infra` | Idle-fleet gate passes; immutable release is deployed; bounded fallback, observation window, and rollback are proven |

Current completion: P0, A, and B are complete. B proved waiting-to-ready,
cold-pull CUDA execution, restart-resume, and readiness-timeout behavior with
published dstack commit `e1e0921007297e19d39dd2e189b94b6761663d60`.
An explicit-volume spot canary also recovered the same marker after the first
Pod was deleted directly through the RunPod API, but it exposed the still-open
provider-authoritative observation and automatic run-storage work in D and F.
Resume C next. Milestone A's retained
exact actual-job digest is
`sha256:38412b847e7977f5c0747d88d2399feabf0a7f5ab2c3a33bd976b703cda50bb9`;
its successful RunPod receipt is protected under
`../ai-infra/.state/qualifications/runpod-runtime/successful-canary.json`.

## Focused execution protocol

At the start of each milestone, create a scratch checklist in task state—not a
new repository document—with exactly these items:

```text
[ ] preflight and dirty-change ownership recorded
[ ] smallest code/config delta implemented
[ ] focused unit tests green
[ ] integration/candidate test green
[ ] live mutation planned and bounded
[ ] live acceptance green
[ ] exact provider cleanup confirmed
[ ] rejected image and uniquely owned cache cleanup confirmed
[ ] safe receipt retained
[ ] living plan updated
```

Use this repository order when a milestone spans repositories:

```text
producer fork code/tests
-> producer fork ledger
-> immutable producer revision
-> framework pin only when required
-> ai-infra candidate configuration
-> read-only plan
-> bounded apply
-> verify
-> cleanup
-> exact rejected-publication cleanup
-> receipt
```

Do not run the full framework validation ladder after every edit. Run the
smallest owning tests while developing, the milestone command block before
acceptance, and full validations only before an immutable release or final
promotion.

## Distraction filter

An improvement belongs in the active milestone when all of these are true:

- it improves the code, tests, diagnostics, documentation, CLI, configuration,
  or operator workflow already being changed;
- it preserves the plan's ownership and public-contract boundaries;
- its benefit is concrete and can be covered by focused validation;
- it does not introduce a new service, backend, image lineage, migration, or
  unrelated dependency;
- it can ship and roll back with the same release unit.

Prefer improvements that remove repeated manual commands, provide safe
`plan|apply|verify|cleanup` wrappers, make retry state visible, replace prose
errors with typed actionable failures, reduce configuration duplication, or
turn a fragile qualification step into a deterministic test. Record material
judgment in the living plan's Decision Log and the result in its Progress or
Surprises section.

If an improvement fails any criterion, preserve it as a concise discovery and
continue the milestone. Promote it into the plan only after the active gate is
green, unless ignoring it would create an unsafe or fundamentally wrong result.

Defer an observation when all of the following are true:

- it does not prevent the current milestone from reaching acceptance;
- it does not risk data loss, credential disclosure, provider leakage, or an
  incorrect public contract;
- it belongs to another milestone or repository owner;
- a safe note in `Surprises & Discoveries` is enough to preserve it.

Stop the current action and recover when any of these occurs:

- an owned Pod or volume cannot be proved absent;
- a writer lease or provider state is ambiguous;
- a manifest or artifact digest changes unexpectedly;
- an inherited OCI descriptor is recompressed or replaced;
- a secret or signed query string appears in output or retained evidence;
- a live command targets an unresolved variable, broad prefix, or unowned ID;
- a production worker becomes busy before a restart or promotion;
- implementation would require changing the frozen Posttrain product meaning.

Recovery priority is always: prevent new mutations, identify exact owned
resources, reconcile or clean them, retain a safe failure receipt, update the
living plan, then resume the same milestone. Do not advance around a red gate.

## Anti-patterns

- Working on two production milestones at once
- Re-reading the entire 900+ line plan every turn
- Running broad test suites before focused tests
- Rebuilding or republishing an image for a scheduler-only change
- Adding provider, Trackio, RustFS, R2, or Ansible vocabulary to generic dstack
  contracts
- Sending checkpoint bodies through Trackio when presigned RustFS transport is
  the selected path
- Using Cloudflare R2 for Trackio artifacts
- Changing the universal base image or forcing OCI recompression
- Replacing working credentials during the security milestone
- Treating runner disconnect as proof of spot interruption
- Creating a replacement volume before the prior writer is fenced
- Calling provider deletion complete before observing provider absence
- Cancelling an unrelated job or weakening the idle-fleet promotion gate
- Marking Progress complete without a retained receipt
- Rejecting an obvious adjacent quality or developer-experience improvement
  merely because the first implementation would have been a smaller diff

## Maintaining this runbook

Update this file only when the operating loop, milestone order, authorization
boundary, or completion protocol changes. Put implementation discoveries and
architecture decisions in the living plan. Record the date and reason below so
future agents can distinguish an intentional workflow change from drift.

- 2026-08-30: Created from the authoritative post-canary A-J plan. Optimized
  execution for one-milestone focus, bounded live mutations, receipt-driven
  progress, self-hosted RustFS artifacts, OCI-only R2, preserved credentials,
  security-last hardening, and idle-gated production promotion.
- 2026-08-30: Clarified that focus does not forbid improvement. The goal may
  include adjacent, evidence-backed quality and developer-experience work in
  the active release unit while deferring broader redesigns that would create
  scope drift.
