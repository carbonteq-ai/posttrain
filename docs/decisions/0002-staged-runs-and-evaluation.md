# ADR 0002 — Staged runs and evaluation boundaries

## Status

Superseded by [ADR 0004](./0004-lifecycle-driven-mvp-platform.md). Retained as prototype history.

## Context

The lab needs to support experiments that compose techniques, such as SFT followed by GRPO and another SFT stage. The initial scripts used mutable `*-latest` outputs, task code inside trainer entrypoints, reference-only YAML, and no shared distinction between recovery checkpoints and completed stage outputs.

The repository is still a playground. It needs consistent conventions without introducing a training platform or hiding TRL behind a local framework.

## Decision

- Treat each trainer invocation as a stage with executable YAML configuration.
- Keep technique runners in `apps/train` and task-specific transforms, prompts, and rewards under `tasks/`.
- Store each run under a stable run ID with resolved config, metadata, local events, recovery checkpoints, and completed output.
- Distinguish `new` adapter creation from continuing an existing adapter.
- Organize evaluation into general capability suites and task-specific suites.
- Use Trackio for interactive metrics while retaining a lightweight local record.
- Keep multi-stage pipeline definitions simple and defer a full orchestrator until real runs justify it.

## Consequences

- Stages can be compared and chained without overwriting their parent outputs.
- SFT, GRPO, and future techniques share run conventions but continue to use their native TRL configuration and trainer APIs.
- Task code can be reused by training and evaluation.
- Some configuration and metadata glue is added locally.
- Pipeline execution remains partly manual until its repeated behavior is understood.

## Alternatives considered

### Continue with independent scripts and `*-latest` outputs

Rejected because lineage, resumption, and comparison become ambiguous when techniques are stacked.

### Build a generic trainer and pipeline framework now

Rejected because the lab has not yet accumulated enough distinct jobs to justify those abstractions.

### Store observability only in Trackio

Rejected because local runs must remain understandable and recoverable when an external UI or logging integration is unavailable.

## Implementation notes

- `docs/architecture.md` is the revision-aware description of these boundaries.
- Stage YAML currently supplies defaults to the small existing argparse CLIs. Reconsider TRL's parser before expanding the local configuration surface.
- Recovery checkpoints live below `checkpoints/`; completed stage artifacts live below `output/`.
- General and task evaluations are selected independently so every stage need not run a large release suite. Results live in the shared evaluation system from ADR 0003 and reference the originating artifact/run.

## Revision history

- 2026-07-19: Superseded as current platform guidance by ADR 0004; retained as prototype history.
- 2026-07-19: Moved evaluation observations out of training-run directories into the backend-neutral evaluation system defined by ADR 0003.
- 2026-07-19: Initial decision.
