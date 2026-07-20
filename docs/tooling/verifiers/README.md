# Verifiers (library)

- Installed via the **`packages/eval` `verifiers` extra**: `uv sync --package eval --extra verifiers --python 3.12`
- Also pulls **`prime`** CLI (Environments Hub: `prime env install` / `prime eval run`)
- The workspace intentionally uses upstream `main`, pinned to commit `284a868d6a9022109b749710672a0460e8a996d4` (2026-07-19), rather than the older `0.2.0` release
- The pinned code exposes the current architecture under `verifiers.v1`; the old environment API is documented upstream as legacy
- The current package composes an **Environment** from a **Taskset**, **Harness**, and runtime configuration; evaluation and online RL consume this same environment model
- `Taskset.load()` is the task-dataset abstraction: it loads or generates typed `TaskData` rows wrapped in `Task` behavior
- A `Task` owns lifecycle hooks, tools, user simulation, validation, metrics, rewards, and group rewards; these should not be duplicated in our eval or trainer code
- Treat reusable task/environment implementation as independently installable Verifiers packages in their own repositories or package workspaces; keep source-data cards under [datasets/](../../datasets/)
- Prefer the same versioned taskset for held-out evaluation and online RL when the semantics match
- Keep rewards that affect training distinct from diagnostic metrics used only for observability
- Preserve Verifiers' native resolved config, traces, and logs; normalized lab summaries should link to them rather than replace them
- Treat the v1 `Trace` as the rollout observability schema: it already records message graphs, model-call sampling/usage/timing/errors, tools, runtime and phase timing, rewards/metrics, stop conditions, and version/run provenance.
- Log bounded trace-derived aggregates to Trackio; attach `traces.jsonl`, `config.toml`, `eval.log`, and the platform driver log as the evaluation artifact.
- Resolve and record the endpoint's serve profile. The current model defaults
  give general evals a 32K TurboQuant K8V4 endpoint, and the `eval` package
  forwards that limit to its internal Verifiers adapter as `max_total_tokens`.
- Use task-shaped response ceilings and inspect call/rollout truncation rates. A score produced by repeated `finish_reason=length` is not clean capability evidence.
- Conflicts with `packages/train` (datasets pin) — switch sync when using Hub environments

Product-level ownership and jobs are defined in [Product capabilities and jobs](../../functional/overview.md). The target integration is described in [Lab architecture](../../architecture.md). The former TRL reward callbacks and separate evaluation-suite implementation were removed; new environments should use the current Verifiers model directly.

When advancing Verifiers, pin a reviewed commit, refresh `uv.lock`, run a taskset smoke evaluation, and record the new SHA here. Do not depend on an unpinned moving branch.
