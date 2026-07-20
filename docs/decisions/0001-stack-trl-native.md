# ADR 0001 — Stack: TRL-first, native uv workspace, Qwen3.5-2B

## Status

Superseded by [ADR 0004](./0004-lifecycle-driven-mvp-platform.md). Retained as prototype history.

## Context

Single **RTX 3070 Ti (8 GB)**. Goal: post-train ~2B open models (SFT + GRPO) locally, then scale later.

## Decision

| Choice | Detail |
| --- | --- |
| Trainer | **TRL** (SFT + GRPO), not PrimeRL yet |
| Layout | **uv workspace**: `packages/common`, `tasks`, `apps/{train,eval,serve}` |
| Tracking | **Trackio** (`report_to=trackio`) |
| Eval | **LightEval** (benchmarks) and **Verifiers** (task envs) as conflicting extras |
| Inference | **vLLM** native in train/serve apps |
| Model default | `principled-intelligence/Qwen3.5-2B-text-only` |
| Tooling | **mise** (Python 3.12) + **uv** |
| Docker | Not for training |

## Consequences

- One shared lockfile; declared conflicts when pins disagree (train vs Verifiers; LightEval vs Verifiers).
- 8 GB forces QLoRA, short completions, small `num_generations`.
- PrimeRL / multi-GPU deferred until hardware allows.
- Prefer Hub Verifiers envs over long-lived custom reward code.

## Alternatives considered

- A single unstructured Python project was rejected because training, evaluation, and serving have different dependency and execution needs.
- PrimeRL was deferred because its distributed focus does not match the current single-GPU lab.
- Custom data, environment, and evaluation frameworks were rejected in favor of upstream libraries plus thin task modules.

## Implementation notes

- The root `uv.lock` covers workspace packages and declared conflicting extras.
- `tasks/` owns task-specific transforms and interim rewards shared by training and evaluation.
- Stage composition, run layout, and evaluation boundaries are specified separately in ADR 0002.

## Revision history

- 2026-07-19: Superseded as current platform guidance by ADR 0004; retained as prototype history.
- 2026-07-19: Added the `tasks` workspace boundary and linked stage conventions to ADR 0002.
- 2026-07-19: Accepted the TRL-first uv workspace decision.

## See also

- [overview](../overview.md)
- [mise-uv/workspace](../tooling/mise-uv/workspace.md)
- [HANDOFF](../HANDOFF.md)
