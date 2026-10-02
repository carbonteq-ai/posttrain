# ADR 0020 — Resolved policy updates independent of execution packing

## Status

Accepted for implementation, 2026-10-02. Native qualification remains open.
Plan: `docs/plan/hierarchical-policy-update-engine.md`.

## Context

Collection, statistical credit populations, optimizer minibatches and execution
packs have different meanings. Current adapters constrain native minibatching.
Thinking spans and externally scored steps also require explicit original-token
provenance, without changing native replay ownership or existing algorithms.

## Decision

Training resolves one optimizer update over immutable native action/context
references, named population memberships, prepared credit and versioned objective
terms. Scheduling chooses contribution occurrences; packs preserve their values,
derivatives and global weights. Reuse native TRL/veRL machinery where equivalent.
Support qualified definitions and combinations; reject unsupported semantics.

Qualified supplied semantic annotations may overlap and contain multiple original
sampled-token intervals. Policy, KL, ratio and reduction support remain distinct.
Scorer quality is evidence; a qualified estimator supplies advantages with scope,
normalization and complete-population provenance. No reward-model lifecycle or
universal objective compiler is introduced. This narrowly supersedes ADR 0018's
semantic-segmentation deferral, preserving `assistant-turns@1` and CAPO union rules.

Fresh collection/refills retain task uniqueness within one population. Reusing
frozen contributions creates update occurrences, not duplicate generation.
Applied updates commit model/optimizer state and cursor together; overflow and
incomplete accumulation do not advance policy versions or learning schedules.

## Consequences

Pure contracts can validate without ML imports. Context/ratio dependency closure
may require more work than selected loss tokens; unsupported capacity fails
explicitly. Existing selections retain semantics during one migration release.
CPU math establishes only mathematical contracts, not native backend support.

## Alternatives Considered

A strict group/episode/turn tree cannot represent overlapping statistical groups.
A generic mask collapses different objective meanings. Replacing native schedulers
adds needless duplication. Freely composable loss switches imply support without
qualification. None addresses the required correctness boundary.

## Implementation Notes

Canonical 02/04/05/06 and README are amended before code. Preserve native replay
authority and package independence. Publish necessary generic fork changes before
immutable consumer pin updates. Qualify native gradients, optimizer transitions,
BF16/FP16, distributed weighting and recovery as specified in the living plan.

## Revision History

- 2026-10-02: Initial decision following reviewed general-engine proposal and
  implementation authorization.
## Explicit SAMPO turn-row amendment, 2026-10-02

The author ARL-Arena source at a25a2a229c85431b421ac785fa5f375a99b2072a
uses individual agent-step rows, length-normalized sequence ratios with local
token derivatives and seq-mean-token-mean reduction. Add `sampo-turns@1` as an
explicit opt-in objective with turn-wide ratios and equal-turn weighting. Keep
original complete-group SAMPO credit and the existing episode objective intact.
Native packing and schedule selection must not substitute for this objective
choice. Qualify moderate ratios against the author formula and retain our
explicit rejection of nonfinite ratios; the author's log-weight cap is not
silently adopted. Public adoption remains subject to native execution gates.
