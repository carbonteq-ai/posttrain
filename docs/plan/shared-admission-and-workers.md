# Shared admission ledger and workers visibility

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to
date as work proceeds.

This document must be maintained in accordance with
`docs/templates/PLAN.md`.

## Purpose / Big Picture

Two local projects on the same GPU must not both be active at once. dstack jobs
must not invent host locks inside posttrain. `posttrain workers` must name who
holds a placement and who waits.

## Progress

- [x] Amend `docs/post-training/03-work-and-evidence.md` admission section
- [x] Resolve machine admission root (`POSTTRAIN_ADMISSION_ROOT` /
  `/var/lib/posttrain` / XDG); wire CLI
- [x] Retarget admission tests for dstack `run:` keys; singular-queue cases on
  local-docker; shared-root cross-project test
- [x] Add `posttrain workers` (human + JSON) with orphan project-ledger warning
- [x] Validation: ruff/pyright/lint-imports clean; 90 focused tests passed;
  `posttrain workers` JSON smoke OK
- [x] Land Observatory #18/#19: Trackio `0.31.5.post4` pin
  (`dc55020d779147612b32c0aced34a8868b91aa71`) + bulk `run_configs` /
  `run_lifecycles` listing; sources health probe stays at `limit=1`
- [x] Docs #20–#28: dstack consumer section, README/release links, trust wording,
  remote GPU gate, lifecycle bite list

## Surprises & Discoveries

- Uncommitted admission code already had `Placement`, `placements()`, and
  dstack `run:` keys; the remaining work was the machine root, test retarget,
  and CLI surface.
- Trackio wheel builds from Git still require `SKIP_FRONTEND_BUILD=1` (or a
  local frontend dist) because `frontend/dist` is not in the published tree.

## Decision Log

- Decision: admission ledger root is CLI-resolved
  (`POSTTRAIN_ADMISSION_ROOT` → writable `/var/lib/posttrain` → XDG), while
  `ExecutionAdmissionService` still takes an absolute `state_root`.
  Rationale: keep the neutral package free of host path policy.
- Decision: no automatic migration of project-local `admission/queue.json`.
  Rationale: cutover is reconcile/cancel then ignore orphans; `workers` warns.

## Outcomes & Retrospective

Delivered machine-scoped host admission, dstack self-scheduling keys with
retargeted tests, and `posttrain workers`. Follow-ons remain Observatory
#18/#19, docs #20+, and remaining DX items.