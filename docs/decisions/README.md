# Decisions (ADRs)


> **STALE — pending reconciliation (2026-07-21).**
> Canonical design: [docs/post-training/](../post-training/README.md).
> Individual ADRs may still be historically useful, but product vocabulary and contracts follow the post-training baseline until each ADR is reconciled. Gap list: [architecture/RECONCILIATION.md](../architecture/RECONCILIATION.md).

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
| [0010](./0010-environment-driven-online-rl-bridge.md) | Environment-driven online-RL bridge with trainer-owned policy generation |
| [0011](./0011-canonical-posttraining-data.md) | Canonical SFT/preference data with HF, NeMo, and Verifiers adapters |
| [0012](./0012-observatory-read-product.md) | Observatory as the single post-training read and analysis product |
| [0013](./0013-portable-project-layout.md) | Portable `.posttrain` project layout and runtime-state boundary |
| [0014](./0014-attested-release-promotion-graph.md) | Attested dependency-to-platform release promotion graph |
| [0015](./0015-checkpoint-scoped-model-artifacts.md) | Checkpoint-scoped recovery and model artifacts |
| [0016](./0016-site-wide-remote-builder-authorization.md) | Site-wide remote-builder authorization with project namespace isolation |
| [0017](./0017-dstack-run-scoped-storage-and-lifecycle-hooks.md) | dstack-owned run-scoped storage, durable lifecycle hooks, spot recovery, and truthful inventory |

ADRs 0001–0003 describe the prototype and were superseded by ADR 0004 for the
previous platform architecture. ADR 0004–0006 are themselves pending rewrite
against the [post-training baseline](../post-training/README.md); see
[RECONCILIATION.md](../architecture/RECONCILIATION.md).
