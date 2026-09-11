# ADR 0018 — Explicit reward evidence between native rollouts and algorithms

## Status

Accepted for implementation, 2026-09-06. Runtime qualification remains open.
Related plan: `docs/plan/gdpo-capo-dual-backend-support.md`.

## Context

GDPO needs independently declared reward components; CAPO needs outcome and
resolved critique over original sampled tokens. Inferring these from an algorithm
name inside the Verifiers bridge couples environment semantics to trainers and
can silently interpret missing critique as successful zero error.

## Decision

- Custom Verifiers judge/scoring plugins own rubrics and extraction using injected
  inference. Scorer is a role, not a new product primitive or mandatory separate
  class. Rubric dimensions are not algorithm fields.
- Initial local credit is assistant-turn addressed with original sampled-token
  provenance. Tool observations are not credited. Context scope is explicit;
  SAMPO, CAPO and GDPO retain distinct mappings. Semantic segmentation is deferred.
- Migrated runtimes retain native Episode envelopes. A detected legacy reader
  supports old artifacts without fabricating tasks or rewriting source data.
  Native advantages are not judge scores; validate absent annotations before
  native flattening can fill them with zeros.
- Environment/scorer implementations own reward meaning and retained judgments.
  A serializable versioned `RewardProjection` selects their evidence explicitly.
  It is a training selection, not a new concrete provider dependency in common.
- Native Verifiers traces remain replay authority. Group/response identities are
  assigned before enrichment; async enrichment is awaited. Terminal traces are
  retained before projection and after enrichment failure or cancellation.
- Critique resolution consumes retained step identities and original token spans,
  not retokenized text. Incomplete panels, ambiguous votes and unavailable scores
  fail closed. Valid zero remains valid. Judge configuration belongs to the
  scorer/composition layer, never to a backend-specific algorithm implementation.
- Admission operates on complete logical prompt groups with bounded replacement.
  Raw evidence is exchanged before global normalization. Scalar-variance filtering
  is not a substitute for evidence validity. Observation/padding positions remain
  outside the loss; veRL may repad masks but preserves sampled-token ordering.
- Numerical backend profiles are explicit. New token clipping/KL implementations
  do not silently change legacy objectives. Framework packages remain independent
  of one another's concrete providers.

## Consequences and verification

Versioned selections and retained evidence make future scorer changes explicit.
Resume now rejects missing or changed selected projection/environment/numerical
contract digests. Binding the actual frozen judge/scorer configuration into this
identity and qualifying native interrupted recovery remain gates in the plan. Bounded collective
admission requires distributed runtime tests, especially tensor-parallel
generation when some ranks have no local replacements. CPU formula equivalence
does not establish distributed lifecycle qualification.

Regression ownership: framework projection/admission/bridge tests under
`packages/train/tests`; native veRL numerical, loss and replay tests in its fork.
Qualify actual GDPO/CAPO optimizer, resume and exported inference on both backends
before publishing support or promoting consumer pins.

## Alternatives Considered

A new scorer service duplicates native Verifiers responsibilities. Fixed rubric
fields couple algorithms to one task. Retokenizing text loses provenance, while
rewriting legacy artifacts risks replay evidence. None is needed for turn support.

## Implementation Notes

Amend the canonical baseline before code. Stage native readers before switching
writers and pins; preserve the qualified runtime until modern activation,
rollout, scoring and reload pass together. Follow the plan's R7 release gates.

## Revision History

- 2026-09-06: Initial structured evidence and complete-group admission decision.
- 2026-09-06: Clarified plugin ownership, native turns and episode migration after
  user approval of Revision 7.
