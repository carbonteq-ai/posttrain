# Per-response usage evidence

Six executions of the actual pinned `openai-codex==0.160.0` worker passed against a local synthetic provider. No signed-in authentication or real model inference was used. Existing SDK tests also passed: eight passed, with the six opt-in executions skipped in the ordinary run.

The worker now requests `experimentalRawEvents` at thread creation and retains `rawResponse/completed` for the exact thread and turn. Each notification carries a response ID and optional usage. The published SDK exposes this experimental notification through the public `UnknownNotification.params` object; the worker uses that object without modifying SDK internals. Other raw-item and account/setup notifications are excluded.

The decisive counterexample is an intermediate response with missing usage. Two responses completed, but the first reported `usage: null`. The second reported output 5 and reasoning 2. The final thread total consequently reported only output 5 and reasoning 2, despite the first synthetic response representing another 7 output tokens. A valid-looking terminal cumulative count cannot establish completeness. The preserved null response makes this omission inspectable.

The other cases retain exact output/reasoning counts for both responses and cover completion below threshold, terminal completion at threshold, active interruption, exact 16,384 crossing and one token of overshoot. See `summary.json` for native response records, final counters, termination status and tested source hashes. `raw-evidence.zip` includes the original SDK compatibility failure and fixed execution journals; symlink aliases are excluded.

Pinned source at `a956835d020762cb2b570053af06f643a11c0ecc` defines this interface in `codex-rs/app-server-protocol/src/protocol/v2/thread.rs` and translates it in `app-server/src/bespoke_event_handling.rs`. `core/src/session/mod.rs::record_observed_response_completed` emits the response event even when usage is missing; cumulative token accounting updates only when usage is present. The decoder evidence from `../sdk-usage-accounting-01/decoder-source.json` shows that reasoning is preserved as a detail of provider output rather than added again.

This qualifies evidence collection for **reported completed responses**. It does not prove physical/billed token completeness for failed partial streams, transport retries without completion usage, or provider misreporting. It does not create model-call nodes, token coordinates or a hard pre-generation output cap. Existing traces without raw completion events cannot acquire this evidence retroactively. Environment budget reconciliation and signed-in qualification remain separate checks.

Reproduce from `/home/hammad/projects/verifiers`:

```sh
VF_RUN_CODEX_SDK_USAGE=1 uv run --frozen pytest tests/v1/test_configs.py -k actual_unpaid_usage_accounting -q --basetemp=/tmp/sdk-usage-accounting-02-check
```
