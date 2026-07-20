# Tooling (workspace)

Runtime lives in the **uv workspace** (`packages/common`, `packages/train`, `packages/eval`, and `packages/serve`). These notes are about upstream tools.

| Tool | Used by | Notes |
| --- | --- | --- |
| [trl](./trl/) | `packages/train` | SFT / GRPO |
| [vllm](./vllm/) | `packages/train`, `packages/serve` | Optional colocate + serve |
| [sglang](./sglang/) | planned inference adapter | Backend-specific runtime, shared evaluation data |
| [triton](./triton/) | planned server/kernel work | Clarifies server versus kernel implementations |
| [verifiers](./verifiers/) | `packages/eval` extra `envs` | Environments Hub / `prime` |
| [trackio](./trackio/) | lab observation context and reports | CarbonTeq fork, immutable pin, Verifiers trace copy |
| [mise / uv / setup](./mise-uv/) | whole repo | Python 3.12 + uv workspace |
| [hardware](./hardware/) | — | 8 GB constraints |
| [primerl](./primerl/) | deferred | Multi-GPU later |

Setup: [mise-uv/setup-environment.md](./mise-uv/setup-environment.md) · Workspace: [mise-uv/workspace.md](./mise-uv/workspace.md).
