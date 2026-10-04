# Recovered errors remain visible

Seven actual executions of the pinned SDK passed against a local synthetic provider. No signed-in authentication or real model inference was used. These fresh journals preserve the earlier accounting evidence without rewriting accounting-01 or accounting-02.

The added case sends a synthetic `response.failed` stream event with `server_error` and missing usage. The pinned runtime publishes a scoped `error` notification with `willRetry: true`, retries, then completes two responses reporting cumulative output 12 and reasoning 5. Three physical fixture requests occurred. The worker retains the public error notification and reports `provider_errors_observed`, `provider_retries_observed`, and `output_budget_state: unqualified_provider_errors_observed`, even though the actual terminal status is `completed`. A recovered run remains usable evidence; its terminal totals do not prove the failed request's output budget.

The other six cases cover known cumulative usage, an intermediate `usage: null`, completion and interruption at threshold, exact 16,384 crossing, and one token of overshoot. All seven passed. Existing SDK tests: eight passed, seven opt-in cases skipped. Scoped Ruff and `git diff --check` passed.

`summary.json` records seven unique journals, checksums, native response usage, public errors, terminal states and the tested source hashes. `source-evidence.json` retains pinned public error schema and runtime emission excerpts. `raw-evidence.zip` contains the raw journals and physical fixture request records; pytest symlink aliases are excluded.

The source proves public retry notifications are observable when emitted. This does not establish complete visibility into every HTTP retry, failed partial stream or billed token. Missing usage and observed errors must decline reported-budget qualification. A clean completion does not establish task success, physical usage completeness or a hard pre-generation cap.

Reproduce from `/home/hammad/projects/verifiers`, using a fresh directory:

```sh
VF_RUN_CODEX_SDK_USAGE=1 uv run --frozen pytest tests/v1/test_configs.py -k actual_unpaid_usage_accounting -q --basetemp=/tmp/sdk-usage-accounting-03-check
```
