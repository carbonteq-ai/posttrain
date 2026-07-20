# Decisions (ADRs)

Architecture and stack choices for this lab.

| ID | Title |
| --- | --- |
| [0001](./0001-stack-trl-native.md) | TRL-first, uv workspace, Qwen3.5-2B, Trackio / LightEval / Verifiers |
| [0002](./0002-staged-runs-and-evaluation.md) | Staged runs, checkpoint semantics, observability, and evaluation boundaries |
| [0003](./0003-backend-neutral-evaluation-data.md) | Backend-neutral model, runtime, result, and reporting data |
| [0004](./0004-lifecycle-driven-mvp-platform.md) | Code-first lifecycle platform boundaries (current) |
| [0005](./0005-trackio-verifiers-traces.md) | Compatible Trackio fork with queryable Verifiers traces |
| [0006](./0006-trackio-observation-model.md) | Trackio-only observability and evidence model |
| [0007](./0007-trl-vllm-025-fork.md) | TRL 1.8 fork pinned with validated vLLM 0.25.1 support |
| [0008](./0008-model-conversation-contracts.md) | Shared model-native conversation contracts and backend-owned parsers |
| [0009](./0009-native-verifiers-environment-packages.md) | Independently publishable native Verifiers v1 environments |

ADRs 0001–0003 describe the prototype and are superseded by ADR 0004 for current platform architecture.
