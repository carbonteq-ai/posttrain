# Record how every rollout episode ended

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It is maintained according to `docs/templates/PLAN.md`.

## Purpose / Big Picture

Reinforcement-learning (RL) training on AutomationBench runs many rollout *episodes*: one agent attempt at one task, recorded as one native Verifiers trace. Until this change, every episode that did not finish normally was recorded only as `truncated`. A 210-trace audit of run `lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1` (job kind `train.sampo`) found four very different endings behind that one flag: a model reply cut at the per-call `max_tokens` (4096), a next request refused because its prompt exceeded the served context (24,576 tokens), a reply shortened because prompt plus reply reached the context length, and the normal turn limit. None of them is an execution error, and the first three are fixable in different ways (a larger per-call budget, context management, shorter thinking), so a researcher needs to see them apart.

After this change every training and evaluation episode records one *ending label*. A researcher can group rollouts by it in Observatory's semantic SQL (`select ending, count(*) from rollouts group by ending`), read per-update rates such as `train/rl/ending_reply_token_limit_rate` next to the existing `train/rl/rollouts_truncated`, and see the label in Observatory's trace table and trace detail. The trace list and trace detail also stop disagreeing about context-rejected episodes: both show them as truncated, with the reward the environment scored, not as errors.

Truncation keeps its exact meaning. Every ending other than `completed` and `error` is a truncation, so the `truncated` flag, truncation penalties, masking, and rewards are unchanged.

## Progress

- [x] (2026-09-28 06:40Z) Read `AGENTS.md`, the plan template, `docs/post-training/06-observation-and-lineage.md`, and the code that derives truncation: `packages/environment/src/posttrain/environment/verifiers_evidence.py`, the train bridge `packages/train/src/posttrain/train/integrations/verifiers.py`, the TRL rollout adapter `packages/train/src/posttrain/train/backends/trl/policy_rollouts.py`, the eval adapter `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py`, the Trackio adapter `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`, and Observatory's `traces.py` and `semantic_layer/`.
- [x] (2026-09-28 06:50Z) Confirmed that the pinned Trackio fork accepts only a closed set of trace-fact dimension names, so a new fact column needs a Trackio change; chose the trace metadata path instead (see Decision Log).
- [x] (2026-09-28 07:00Z) Milestone 1: defined the vocabulary in `posttrain.common.episodes`, derived the label in `verifiers_episode_ending`, recorded it as the trace attribute `episode_ending`, and redefined `verifiers_trace_is_truncated` on top of it. Tests use trimmed real traces. Commit `37f63e67`.
- [x] (2026-09-28 07:10Z) Milestone 2: per-update counts and rates in training metrics, both live (TRL rollout adapter) and replayed from native traces. Commit `78f8a5e1`. Evaluation traces carry the label (test only; the eval adapter already spreads the shared attributes). Commit `223296c0`.
- [x] (2026-09-28 07:25Z) Milestone 3: Observatory list and detail agree; the ending is shown in both; the semantic layer gains `rollout.ending` and the rate measures; OpenAPI and the frontend schema are regenerated. Commit `8b2babd7`.
- [x] (2026-09-28 07:31Z) Canonical baseline amendment in `docs/post-training/06-observation-and-lineage.md` and this plan.
- [x] (2026-09-28 07:40Z) Validation ladder run on the whole repository (results under Validation and Acceptance).
- [x] (2026-09-28 15:40Z) Milestone 4, Trackio side (repository `../trackio`, worktree `/home/hammad/projects/trackio-release-next`, branch `codex/next-release` from tag `carbonteq-v0.31.5.post14.dev31`): fast-forwarded the unreleased f3be77d7 (artifact-commit retries), added the `episode_ending` fact dimension and Doris schema version 5 (commit `f0abb2c0`), and bumped to `0.31.5.post14.dev32` with the fork ledger (commit `d71cf2a5cbcc561bd72e3932d8032200a791d79c`). Unit suite: 516 passed, 7 skipped, 1 failed (`test_gpu_hardware.py`, `pynvml` not installed; the ledger's separate GPU gate). `ruff check trackio tests` passed. The real-Doris test file skipped (no isolated Doris credentials in this environment).
- [x] (2026-09-28 15:45Z) Built dev32 from a fresh clone of `d71cf2a5` with `uv build` (the same command rebuilt dev31's published wheel and sdist byte for byte): wheel `78e3ecf207c074c43281edff75343086bf379a22cc281308cc12436e5a5260a3`, sdist `94d3f3bb7084346f441ce39b8e9d4cd775d1197d07ca6e5fba87726dfbddddc0`; `twine check` passed; a clean Python 3.12 install reports matching distribution and import versions, schema version 5 and the dimension. Retained under `/home/hammad/projects/trackio-release-next/dist/` (git-ignored) with `SHA256SUMS`.
- [x] (2026-09-28 16:10Z) Milestone 4, Posttrain side (branch `codex/trackio-next` from `codex/episode-endings`): pin commit `b28b0dc5` (assumes dev32 is published; locks written from the hashes), code commit `55eb9ed3` (calculator v9 emits `episode_ending`; `rollout.ending` reads `fact_episode_ending` with the attribute as fallback; backfill reports endings). Validated with the local dev32 wheel installed into the worktree venv (`uv pip install --no-deps`), not through the lock.
- [x] (2026-09-28 16:15Z) Read-only backfill preview against the shared server: `posttrain-lab` run `aab30e6e079b4974bb82512a703e8592` (`train.sampo-lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1`), first 50 traces: 50 complete, endings `completed` 49 and `turn_limit` 1, nothing written.
- [x] (2026-09-28 14:05Z) Release steps 1-3: pushed fork branch `codex/next-release` and annotated tag `carbonteq-v0.31.5.post14.dev32` (both at `d71cf2a5`), created the GitHub prerelease with the two retained files (downloaded assets hash to `78e3ecf2…` and `94d3f3bb…`), and ran Posttrain workflow `publish-trackio-internal.yml` run `36432505907` (success). `https://pypi.lan/carbonteq/dev/+simple/carbonteq-trackio/` lists both files under `../../+f/78e/3ecf207c074c4/` and `../../+f/94d/3f3bb7084346f/`, the paths the pin commit predicted, and both download with the expected SHA-256. `uv lock`, the quantization `uv lock`, `posttrain-release lock-dependencies` and `posttrain-release lock-runtime-dependencies` (and its `--check`) changed nothing. Ladder from the lock (`uv sync --all-packages --locked --python 3.13 --extra trl --extra verifiers`): 2052 passed, 24 skipped; ruff, pyright (0 errors), lint-imports (9 kept), frontend tests and build pass. `posttrain-release check` fails only because the job-kind images predate the new closures (republish them in step 5); `--allow-pending-runtime-lock` passes.
- [x] (2026-09-28 14:53-15:33Z) Release step 4 in the maintenance window (training cancelled 13:52Z, last evaluation ~14:49Z; newest metric/trace rows 14:47Z, one CLI note at 14:51Z). Retained, restore-checked Doris backups with exact before/after row counts (production `trackio` 2,107,556 rows, snapshot `trackio_production_pre_v5_20260928`; `trackio_candidate` 29,238 rows, snapshot `trackio_candidate_pre_v5_20260928`) using ai-infra `scripts/qualify_trackio_doris_backup.py --retain-backup`; receipts `ai-infra/.state/artifacts/trackio-doris-{production,candidate}/pre-v5-backup-receipt.json`. `trackio storage migrate-doris --to 5 --preview` listed only the `ADD COLUMN`; `--apply` migrated candidate then production at 14:56Z. ai-infra `scripts/deploy-trackio plan|apply` (branch `deploy/trackio-dev32`, commit `c35ad96`) rebuilt both services; the image build took about 35 minutes because PyPI downloads ran at about 7 KB/s. Both `/version` endpoints report dev32 since 15:32Z; both databases report schema version 5 with `fact_episode_ending varchar(128)`. `scripts/qualify_trackio.py` (metrics and artifact round trip), a run-note add/delete on the qualification run, `scripts/qualify-observatory`, and authenticated Observatory reads of runs, notes and traces passed.
- [x] (2026-09-28 15:35-17:00Z) Release step 6: backfilled 26 runs from `/home/hammad/projects/rl-trackio-next` with `outputs/qualification/episode-ending-backfill/backfill.py` (the `backfill_verifiers_trace_window` function behind `posttrain trace-facts backfill`, 200-trace pages, the internal URL `http://192.168.110.53:7860` because 1,000-trace pages through the public edge timed out). Previews ran for the three largest training runs and all 22 evaluation runs (equal endings to the applied result); the vortex preview was stopped to reduce server load, since apply refuses pages with unscored thinking and none had any. Storage check: all 36,897 traces of these runs have a `fact_episode_ending` and v9 facts. Endings: cont40-fixed-tools 5,940 (completed 5,358, reply_token_limit 382, turn_limit 94, context_limit_reply_cut 59, context_rejected 36, token_budget 11); cont120 6,654 (5,955 / 266 / 154 / 149 / 120 / 10); turns-150 10,787 (9,174 / 567 / 512 / 255 / 226 / 53); vortex-v5-150 11,876 (9,996 / 624 / 585 / 324 / 274 / 73); the 22 held-out evaluations (1,680 traces) are all completed except 6 (reply_token_limit 4, turn_limit 1, token_budget 1 in v3/v3-t05 runs).
- [ ] Release steps 5 and 7: job images with calculator v9, and promotion of dev32 to `carbonteq/stable`.
- [ ] Other follow-ups (see Outcomes): evaluation run counters, veRL live counters, and passing `max_model_len` to the classifier.

## Surprises & Discoveries

- Observation: the 12,288-token rollout output budget did end some episodes. The audit note said no stop condition existed for it, but Verifiers v1 (`verifiers/v1/session.py`, `RolloutLimits`) stops an episode with stop condition `max_output_tokens` when the budget is reached between turns, and the run has such episodes.
  Evidence: grouping this run's 6,654 traces by the native stop condition through Observatory's SQL endpoint returned `agent_completed` 6,490, `max_turns` 154, `max_output_tokens` 10. These now carry the label `token_budget`. Nothing about the budget itself was changed.
- Observation: every context-shortened reply in the audit sample ended exactly at the served context length, and every one stopped short of the call's recorded `sampling.max_tokens`, so the rule "finish_reason `length` below `max_tokens` means the context cut it" needs no `max_model_len`.
  Evidence: over the 211 cached audit traces the classifier gives completed 115, reply_token_limit 43, turn_limit 21, context_rejected 16, context_limit_reply_cut 16; all 16 context cuts have prompt plus completion tokens equal to 24,576; passing `max_model_len=24576` changes no label; the derived truncation flag equals the recorded `is_truncated` attribute on all 211.
- Observation: the Observatory disagreement came from two payload shapes, not two rules. Trackio's trace list pages return a summary payload that drops `calls` (`trackio/sqlite_storage.py`, `_trace_payload_for_read`), so the list never saw the refused final call, while the detail page (full payload) did and reported `ProviderError (HTTP 400)`, clearing the reward. The shared evidence had already decided (fact calculator v7) that a final context refusal is truncation, not error, and recorded `has_error: false`.
  Evidence: trace `06d0f371052643dcbc7f250d8f8127e5` (optimizer step 43) listed as truncated with reward −0.02, detail as outcome error with reward null, before this change; both truncated with reward −0.02 after it (checked against the shared Trackio server with this branch's Observatory on port 7871).
- Observation: Trackio's trace-fact dimension names are a closed set (`trackio/trace_facts.py`, `_DIMENSION_NAMES`), and the fact columns are fixed in `trackio/project_sql.py` (`FACT_COLUMNS`). An `episode_ending` dimension would be rejected by the pinned server.
  Evidence: reading the installed `carbonteq-trackio` package in `.venv/lib/python3.13/site-packages/trackio/`.
- Observation: the worktree's default `uv sync` omits the `trl` and `verifiers` extras, so 20 train tests fail on import (`torch`, `trl`, `aiohttp`, `renderers`). CI installs `--extra trl` and `--extra verifiers`.
  Evidence: `uv sync --all-packages --locked --python 3.13 --extra trl --extra verifiers` then `uv run --no-sync pytest packages/train/tests` passes.

- Observation: a second `uv build` in the same tree gives a different sdist (the first build's `dist/` and build products are picked up), while a fresh clone reproduces the published bytes.
  Evidence: rebuilding dev31 from a fresh clone of `6f292fe2` gave exactly the published hashes (`4980ee67…`, `7ce09dc7…`); building dev32 twice in one clone gave the same wheel and sdists `94d3f3bb…` then `faadcc98…`.
- Observation: Trackio's Doris schema check is strict in both directions, so the migration and the server switch are one maintenance step, and rolling back the server after migrating needs the schema row set back to 4.
  Evidence: `trackio/doris_schema.py`, `negotiate_schema`: a recorded version above the runtime's is "newer than this Trackio runtime", below is "requires an explicit migration".

## Decision Log

- Decision: the labels are `completed`, `turn_limit`, `token_budget`, `time_limit`, `reply_token_limit`, `context_limit_reply_cut`, `context_rejected`, `error`, defined once in `packages/common/src/posttrain/common/episodes.py`.
  Rationale: the audit's four endings map to `reply_token_limit`, `context_rejected`, `context_limit_reply_cut`, and `turn_limit`. `token_budget` (rather than `output_budget`) covers all three Verifiers rollout-wide token caps (`max_output_tokens`, `max_input_tokens`, `max_total_tokens`), which are distinct stop conditions with the same remedy. `time_limit` keeps the legacy stop condition `harness_timeout` from falling into another bucket, so every episode the old flag called truncated still has a truncated label. The vocabulary lives in `posttrain.common` because Observatory, train, and eval all need the names while only the environment package understands native Verifiers records.
  Date/Author: 2026-09-28, Claude.
- Decision: detection rules, first match wins: (1) `error` when `verifiers_trace_has_error` is true; (2) a limit stop condition (`max_turns` to `turn_limit`; the three token caps to `token_budget`; legacy `context_length` to `context_rejected`; legacy `harness_timeout` to `time_limit`); (3) `context_rejected` when the final model call's error message contains "maximum context length" (the existing `final_call_overflowed_context`; vLLM's "Prompt length (N) exceeds maximum context length (M)." from the renderer pre-flight `OverlongPromptError`, and its older "This model's maximum context length is N tokens"); (4) when the last successful call has `finish_reason == "length"`: `context_limit_reply_cut` if its `usage.completion_tokens` is below its `sampling.max_tokens`, or, when a caller passes `max_model_len`, prompt plus completion reached it; otherwise `reply_token_limit`; (5) `completed`.
  Rationale: all inputs are structured fields of the native record except the context refusal, whose `ProviderError` has no structured reason field (status 400 alone is not specific), so the message match is used as the task allowed. A limit stop condition outranks a cut last reply because Verifiers checks limits before serving another turn, so the agent was still working. A missing `usage` or `max_tokens` defaults to `reply_token_limit` because the length finish then has no other evidence.
  Date/Author: 2026-09-28, Claude.
- Decision: record the label as the trace attribute `episode_ending`, not as a new trace-fact dimension, and have the semantic layer read it from Trackio's `traces.metadata` column (`json_extract_string(t.metadata, '$."episode_ending"')`).
  Rationale: Posttrain already spreads trace attributes into Trackio trace metadata, which project SQL exposes in both Doris and local SQLite, so no Trackio or schema change is needed. A fact column would need a Trackio fork change and a pin update; that is recorded as a follow-up. `TraceFactSet` dimensions and the fact calculator version (`verifiers-trace-facts.v8`) are unchanged, so fact projection ids and backfills are unaffected.
  Date/Author: 2026-09-28, Claude.
- Decision: training writes both counts (`train/rl/rollouts_ending_<ending>`) and shares (`train/rl/ending_<ending>_rate`) per update, for every ending including zeros, over the attempted population.
  Rationale: the task asked for rates. `06-observation-and-lineage.md` prefers irreducible counts, so counts are written too and the rates are recomputed from summed counts when an update spans several rollout batches (`RolloutUpdateTotals.flush`). The live TRL path counts the attribute of every observed trace (each attempted rollout, including failed and replaced ones), which is the same population the trace replay (`_trace_metrics`) counts from the native records, so replay authority replaces live points with equal values.
  Date/Author: 2026-09-28, Claude.
- Decision: Observatory shows only the label the producer recorded; traces recorded before the label show none in both list and detail. For the error/outcome decision both views follow the recorded `has_error` attribute first, and the payload fallback skips a final context refusal.
  Rationale: list pages have no model calls, so deriving the label in Observatory would give the detail page a label the list cannot have, recreating the disagreement this plan removes. The recorded `has_error` exists on historical training traces too, so the disagreement on run `lfm26-sampo-cont120-…-r1` is fixed without a relabel.
  Date/Author: 2026-09-28, Claude.
- Decision: evaluation records the label on every trace (the eval adapter already merges `verifiers_trace_attributes`) but its run counters (`eval/run/*`) are unchanged.
  Rationale: evaluation rollouts are already rows of the `rollouts` table, so `group by ending` answers the question; adding eval counters is a separate, optional change.
  Date/Author: 2026-09-28, Claude.

- Decision: the fact calculator derives `episode_ending` from the native record (the same `verifiers_episode_ending` the trace attribute uses), not from caller attributes, and the version moves to `verifiers-trace-facts.v9`.
  Rationale: the backfill re-projects from retained native traces, so the fact must be a pure function of the record to be reproducible; the new dimension changes every projection id, which the version bump makes explicit.
  Date/Author: 2026-09-28, Claude.
- Decision: `rollout.ending` reads `COALESCE(t.fact_episode_ending, <metadata attribute>)` through a new optional `Source.fallback_attribute` on `trace_fact` sources.
  Rationale: traces recorded between the attribute (v8) and v9, and runs not yet backfilled, keep their label; after the backfill the column answers alone. A fallback-free switch would blank those rows until every run is backfilled.
  Date/Author: 2026-09-28, Claude.
- Decision: Trackio validates the ending only as bounded text (at most 128 characters, `VARCHAR(128)` on Doris), not against Posttrain's vocabulary.
  Rationale: the fork keeps generic storage; the vocabulary belongs to `posttrain.common.episodes`.
  Date/Author: 2026-09-28, Claude.
- Decision: the Posttrain pin commit writes `uv.lock`, the quantization lock and the job-kind closures from the dev32 hashes (carbonteq/dev paths are `+f/<sha[:3]>/<sha[3:16]>/`) rather than leaving the dev31 lock or pointing at a local wheelhouse.
  Rationale: dev32's dependency list equals dev31's, so the entries are fully determined by the hashes; relocking after publication must then be a no-op, which checks both the prediction and the published bytes. Local validation installed the wheel into the venv instead.
  Date/Author: 2026-09-28, Claude.

## Outcomes & Retrospective

All three milestones are complete. New training and evaluation traces carry `episode_ending`; training metrics carry per-update counts and rates; Observatory's list and detail agree for context-rejected episodes and show the label; the semantic layer can group rollouts by `rollout.ending` and read `ending_<ending>_rate` per update. Truncation, rewards, penalties, and trace facts are unchanged.

Milestone 4 (the Trackio fact column, formerly follow-up 1) is implemented and built but not released; the steps that need the maintainers are listed in "Release and deployment of the fact column". What remains, each a separate follow-up:

1. (Implemented in Milestone 4; release pending.) Trackio fact column (repository `../trackio`, then the pin here). Add `episode_ending` to `_DIMENSION_NAMES` in `trackio/trace_facts.py`, a nullable `fact_episode_ending` column to the SQLite and Doris trace schemas (a Doris schema migration), to `FACT_COLUMNS` in `trackio/project_sql.py`, and to the restricted aggregation; publish the fork, update the exact pin and `uv.lock`; then add `episode_ending` to the dimensions `project_verifiers_trace_facts` emits (bump the calculator to v9), switch `rollout.ending` in `apps/observatory/src/posttrain_observatory/semantic_layer/framework.py` to `Source(kind="trace_fact", name="episode_ending")` with a metadata fallback for older rows, and run `posttrain trace-facts backfill` for retained runs.
2. Labels for historical traces. Trace metadata cannot be rewritten through the current Trackio API, so traces recorded before this change (including run `lfm26-sampo-cont120-…-r1`) show no label. Follow-up 1 plus a backfill makes the label available for them. Separately, Trackio's summary payload for list pages could retain the last call's `finish_reason`, `usage`, `sampling.max_tokens`, and `error` so readers can classify without the full payload.
3. Evaluation run counters `eval/run/rollouts_ending_<ending>` if screens want them next to `eval/run/rollouts_truncated`.
4. veRL live counters: the veRL agent loop (`packages/train/src/posttrain/train/backends/verl/agent_loop.py`) writes no live ending counts; the trace replay covers veRL runs whose native traces are retained.
5. Passing the served `max_model_len` (known in `policy_rollouts.py` for the async runtime) into `verifiers_episode_ending` would also catch a context cut by a client that records a clamped per-call `max_tokens`. The pinned Verifiers records the unclamped value, so this is not needed today.
6. The 12,288-token output budget is left as it is. It ended 10 of 6,654 episodes in the audited run (stop condition `max_output_tokens`, now `token_budget`), contrary to the audit note.

## Context and Orientation

The repository is a Python 3.12+ `uv` workspace. The pieces involved:

`packages/common/src/posttrain/common/episodes.py` (new) defines the `EpisodeEnding` literal type, `EPISODE_ENDINGS`, their one-sentence meanings in `EPISODE_ENDING_DESCRIPTIONS`, `TRUNCATED_EPISODE_ENDINGS`, the attribute name `EPISODE_ENDING_ATTRIBUTE = "episode_ending"`, and `episode_ending(value)` which returns a known label or None. `posttrain.common` must not import Trackio, TRL, Verifiers, or vLLM.

`packages/environment/src/posttrain/environment/verifiers_evidence.py` reads native Verifiers v1 trace records (plain JSON mappings) without importing Verifiers. A record has `stop_condition`, `ok`, `errors`, and `calls`; each call has `finish_reason`, `usage` (`prompt_tokens`, `completion_tokens`), `sampling.max_tokens`, and `error` (`type`, `status_code`, `message`). `verifiers_trace_attributes(record)` returns the attributes every producer attaches to the trace (`model`, `is_truncated`, `has_error`, now `episode_ending`, `trace_schema_version`). `verifiers_episode_ending(record, *, max_model_len=None)` is the classifier. `project_verifiers_trace_facts` builds the `TraceFactSet` that Trackio stores as fact columns; it is unchanged.

The train bridge `packages/train/src/posttrain/train/integrations/verifiers.py` builds each training `TraceObservation` from those attributes (`_observation_from_record`) and, when a run finishes, replays per-step metrics from the preserved native traces (`_trace_metrics`, marked `observation_source: verifiers`; replayed points replace live ones for the same update). The TRL rollout adapter `packages/train/src/posttrain/train/backends/trl/policy_rollouts.py` observes every attempted rollout's trace and adds per-batch sums to `RolloutUpdateTotals` (`backends/trl/update_totals.py`), which writes one value per update. Metric names and validation live in `packages/train/src/posttrain/train/grpo_observations.py`, which now also holds `EPISODE_ENDING_COUNT_METRICS`, `EPISODE_ENDING_RATE_METRICS`, `episode_ending_counts`, `episode_ending_metrics`, and `episode_ending_rates`.

The eval adapter `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py` (`_emit_batch`) merges `verifiers_trace_attributes(record)` into each evaluation trace, so evaluation traces get the label without code changes.

The Trackio adapter `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py` (`TrackioTrackedRun.trace`) writes trace attributes as top-level keys of the Trackio trace `metadata` (and under `posttrain_attributes`). Trackio's project SQL exposes that `metadata` column in the `traces` table.

Observatory (`apps/observatory`) is the read-only evidence product. `src/posttrain_observatory/traces.py` turns a `TraceRecord` into a `TraceSummary` (`_summary`) for both list pages and the detail page; `models.py` defines `TraceSummary`. The semantic layer (`src/posttrain_observatory/semantic_layer/`) defines SQL views `runs`, `updates`, and `rollouts` over Trackio's tables; `framework.py` declares dimensions and measures, `views.py` generates the SQL, and measures of metric series come from `metric_catalog.py`. The committed `apps/observatory/openapi.json` and the generated `apps/observatory/frontend/src/lib/api-schema.ts` describe the HTTP contract. The React frontend shows traces in `frontend/src/components/TraceTable.tsx` and the selected trace in `frontend/src/App.tsx`.

## Plan of Work

Milestone 1 (vocabulary and classifier). Add `episodes.py` to `posttrain.common` and export its names. In `verifiers_evidence.py`, replace the truncating stop-condition set with a mapping from limit stop conditions to endings, add `verifiers_episode_ending` and `_reply_cut_by_context`, add `episode_ending` to `verifiers_trace_attributes`, and define `verifiers_trace_is_truncated` as "the ending is a truncation". Export the classifier from `posttrain.environment`. Tests: `packages/environment/tests/test_episode_endings.py` with trimmed real records in `packages/environment/tests/fixtures/automationbench_episode_endings.json` (one context rejection, two context cuts, two per-call cuts, one turn limit, one completion, about 11 KB) plus synthetic rule cases.

Milestone 2 (training metrics). In `grpo_observations.py` add the metric-name maps and helpers, and register the names as passthrough, ratio, and non-negative metrics. In `policy_rollouts.py` collect `trace.attributes["episode_ending"]` in `observe_trace` and add `episode_ending_counts(...)` to the batch sums; in `update_totals.py` recompute the rates from the summed counts on flush. In `integrations/verifiers.py` add `episode_ending_metrics(...)` to `_trace_metrics`. Tests in `packages/train/tests/test_api.py` (live, four batches in one update) and `packages/train/tests/test_verifiers_grpo_bridge.py` (replay). Add an evaluation assertion in `packages/eval/tests/test_api.py`.

Milestone 3 (Observatory). Add `ending: EpisodeEnding | None` to `TraceSummary`; in `traces.py` make `_wire_error` follow the recorded `has_error` and skip a final context refusal, make `_wire_truncated`'s payload fallback count a final context refusal, and fill `ending` from the recorded attribute. In the semantic layer add the source kind `trace_attribute` (read from `t.metadata`), the dimension `rollout.ending`, and catalog entries `ending_<ending>_rate`. Regenerate `openapi.json` and the frontend schema, add `episodeEndingPresentation` in `frontend/src/lib/trace-presentation.ts`, an Ending column and outcome-icon label in `TraceTable.tsx`, and an Ending chip in the trace detail in `App.tsx`. Tests: `apps/observatory/tests/test_trace_errors.py` (list and detail from the same trimmed trace agree), `test_semantic_sql.py` (group by ending on a local Trackio project written by Posttrain), `test_semantic_layer.py`, and `frontend/src/components/TraceTable.test.tsx`.

## Concrete Steps

All commands run from the repository root (`/home/hammad/projects/rl-episode-endings` for this branch) unless stated.

    uv sync --all-packages --locked --python 3.13 --extra trl --extra verifiers
    uv run --no-sync pytest packages/common/tests packages/environment/tests -q
    uv run --no-sync pytest packages/train/tests packages/eval/tests packages/data/tests apps/cli/tests -q
    uv run --no-sync pytest apps/observatory/tests -q
    uv run --no-sync posttrain-observatory schema --openapi apps/observatory/openapi.json --mcp apps/observatory/mcp-schema.json
    (cd apps/observatory/frontend && npm ci && npm test && npm run build)
    uv run --no-sync ruff check .
    uv run --no-sync pyright packages/common packages/environment packages/train packages/eval apps/observatory
    uv run --no-sync lint-imports
    git diff --check

Use `uv run --no-sync` after syncing with extras; a plain `uv run` re-syncs to the default extras and removes `trl`.

## Validation and Acceptance

Unit acceptance: the test `test_real_automationbench_episodes_get_their_ending` labels each trimmed real trace as its audit class and keeps `is_truncated` and `has_error` equal to what was recorded; `test_ending_rules` covers every rule; `test_grpo_rollout_adapter_writes_episode_ending_counts_and_rates_once_per_update` sees counts 2/1/1 and rates 0.5/0.25/0.25 for one update of four batches; `test_replayed_step_metrics_count_every_episode_ending` sees one of each audit ending in a five-trace step; `test_context_rejection_is_the_same_truncated_scored_trace_in_list_and_detail` fails before the Observatory change (detail reported `ProviderError (HTTP 400)` and reward None) and passes after; `test_rollouts_group_by_the_recorded_episode_ending` groups a real local Trackio project by `rollout.ending`.

Observed results on this branch: common and environment 94 passed; train, eval, data, and cli 811 passed with the extras installed; Observatory 182 passed; frontend 120 tests passed and the production build succeeded. The whole repository suite (`uv run --no-sync pytest -q`, extras installed) gave 2051 passed, 24 skipped; `ruff check .` passed; `pyright` on the touched packages reported 0 errors; `lint-imports` kept all 9 contracts; `git diff --check` was clean.

Live acceptance (read-only, against the shared Trackio server): start this branch's Observatory with the same source settings as the deployed one on another port,

    POSTTRAIN_OBSERVATORY_SOURCE=trackio POSTTRAIN_OBSERVATORY_SOURCE_ID=trackio-carbonteq \
    POSTTRAIN_TRACKIO_PROJECT=posttrain-lab POSTTRAIN_TRACKIO_SERVER_URL=https://trackio.carbonteq.com \
    uv run --no-sync --package posttrain-observatory posttrain-observatory serve --port 7871

then request the detail `GET /api/v1/runs/<key>/traces/06d0f371052643dcbc7f250d8f8127e5` and the list `GET /api/v1/runs/<key>/traces?step=43&limit=250` for run key `WyJ0cmFja2lvLWNhcmJvbnRlcSIsImxmbTI2LXNhbXBvLWNvbnQxMjAtZzI0eDYtbHI1ZTUta2wxZTItdDA1LTIwMjYwOTI4LXIxIl0`. Both return outcome `truncated`, truncated true, reward −0.02, error null, ending null (the trace predates the label). `POST /api/v1/semantic/query` with `{"measures":["rollouts"],"by":["rollout.ending","rollout.truncated"],"runs":["lfm26-sampo-cont120-g24x6-lr5e5-kl1e2-t05-20260928-r1"]}` runs on Doris and returns `[[null, false, 5955], [null, true, 699]]`; a run trained after this change returns one row per ending.

## Release and deployment of the fact column

These steps need the maintainers' credentials or touch shared services; nothing here has been run. A training run is writing to the shared Trackio now, so do steps 4-5 only after it finishes (its last artifact commits have landed and the job is terminal), not while it runs.

1. Trackio fork (`/home/hammad/projects/trackio-release-next`): review, push branch `codex/next-release` to `origin`, merge or keep as the release commit, tag `carbonteq-v0.31.5.post14.dev32` at `d71cf2a5cbcc561bd72e3932d8032200a791d79c`, and create the immutable GitHub Release with the two retained files from `dist/` (verify `sha256sum -c dist/SHA256SUMS` first). If the commit changes, rebuild from a fresh clone with `uv build` and update every hash below.
2. Dispatch Posttrain's `Publish retained Trackio candidate to development` (`.github/workflows/publish-trackio-internal.yml`) with `release_tag=carbonteq-v0.31.5.post14.dev32`, `wheel_sha256=78e3ecf207c074c43281edff75343086bf379a22cc281308cc12436e5a5260a3`, `sdist_sha256=94d3f3bb7084346f441ce39b8e9d4cd775d1197d07ca6e5fba87726dfbddddc0`.
3. On `codex/trackio-next`: `uv lock`, `(cd tools/quantization && uv lock)`, `uv run posttrain-release lock-dependencies`, `uv run posttrain-release lock-runtime-dependencies`, then `uv sync --all-packages --locked --python 3.13 --extra trl --extra verifiers` and the ladder. Expect no lock change; commit any difference.
4. Server (ai-infra `scripts/deploy-trackio`, candidate database `trackio_candidate` first, then production): take a retained, restore-verified Doris backup; run `trackio storage migrate-doris --to 5 --preview` (it must list only the `ALTER TABLE traces ADD COLUMN fact_episode_ending VARCHAR(128) NULL`), then `--apply --backup-receipt <receipt>`; deploy the dev32 server immediately after, because the dev30 server refuses a v5 database on its next start. Run the fork's real-Doris test `tests/integration/test_doris_storage.py::test_episode_ending_fact_dimension_on_real_doris` against an isolated database, and check `/version` reports dev32.
5. Only then release job images built from this branch (calculator v9); older images keep writing v8 facts, which the new server accepts.
6. Backfill each retained run, preview then apply, from `apps/lab`: `uv run posttrain trace-facts backfill <provider run id> --window-size 1000` and then the same with `--apply`, resuming with `--cursor <next cursor>` until it prints `done`. Runs whose traces carry reasoning text without reasoning-token usage need `--renderer-model <model id>` (the command refuses otherwise). Reconcile a run that was training during the backfill again after it ends.
7. Promote dev32 to `carbonteq/stable` with `promote-retained-fork-candidate.yml` after qualification.

Rollback: before step 4 nothing shared changed; revert the Posttrain branch. After step 4, the column is additive and old servers never read it: redeploy the dev30/dev31 server after setting the schema row back with `INSERT INTO schema_versions (component, version, applied_at) VALUES ('trackio', 4, UTC_TIMESTAMP())` (the table is keyed by component), and keep job images on calculator v8 (a dev31 client rejects `episode_ending`). Dropping the column (`ALTER TABLE traces DROP COLUMN fact_episode_ending`) is optional and loses backfilled labels; the native traces and the `episode_ending` trace attribute still hold them.

Milestone 4 validation (branch `codex/trackio-next`, dev32 wheel installed into the venv): whole suite with extras `uv run --no-sync pytest -q` 2052 passed, 24 skipped; `ruff check .` passed; `pyright` 0 errors; `lint-imports` 9 contracts kept; `posttrain-release check --allow-pending-runtime-lock` OK; frontend 120 tests passed and the production build succeeded. `test_rollouts_group_by_the_recorded_episode_ending` now covers a label present only as a fact (rollout-1) and only as a trace attribute (rollout-3); `test_backfilled_facts_are_accepted_by_the_pinned_trackio_client` fails with a Trackio client older than dev32.

## Idempotence and Recovery

All steps are code and test changes; rerunning them is safe. Regenerating `openapi.json` and `api-schema.ts` is deterministic. The live check only reads from the shared Trackio server; stop the port-7871 server afterwards. To back out, revert the four commits; no stored data was changed, and traces written with the attribute remain readable by older code (the extra metadata key is ignored).

## Artifacts and Notes

Distribution of the 211 cached audit traces under the new classifier:

    completed 115, reply_token_limit 43, turn_limit 21, context_rejected 16, context_limit_reply_cut 16
    all 16 context cuts: prompt_tokens + completion_tokens == 24576

Native stop conditions of the whole audited run (6,654 traces), from trace metadata on Doris:

    agent_completed 6490, max_turns 154, max_output_tokens 10

Generated rollout view column:

    JSON_EXTRACT_STRING(t.metadata, '$."episode_ending"') AS `ending`

## Interfaces and Dependencies

In `posttrain.common` (`packages/common/src/posttrain/common/episodes.py`):

    type EpisodeEnding = Literal["completed", "turn_limit", "token_budget", "time_limit",
        "reply_token_limit", "context_limit_reply_cut", "context_rejected", "error"]
    EPISODE_ENDING_ATTRIBUTE = "episode_ending"
    EPISODE_ENDINGS: tuple[EpisodeEnding, ...]
    EPISODE_ENDING_DESCRIPTIONS: Mapping[EpisodeEnding, str]
    TRUNCATED_EPISODE_ENDINGS: frozenset[EpisodeEnding]
    def episode_ending(value: object) -> EpisodeEnding | None
    def episode_ending_is_truncated(ending: EpisodeEnding) -> bool

In `posttrain.environment`:

    def verifiers_episode_ending(record: Mapping[str, object], *, max_model_len: int | None = None) -> EpisodeEnding

In `posttrain.train.grpo_observations`:

    EPISODE_ENDING_COUNT_METRICS: Mapping[EpisodeEnding, str]   # train/rl/rollouts_ending_<ending>
    EPISODE_ENDING_RATE_METRICS: Mapping[EpisodeEnding, str]    # train/rl/ending_<ending>_rate
    def episode_ending_counts(endings: Iterable[object]) -> dict[str, float]
    def episode_ending_metrics(endings: Iterable[object]) -> dict[str, float]
    def episode_ending_rates(counts: Mapping[str, float]) -> dict[str, float]

In Observatory: `TraceSummary.ending: EpisodeEnding | None`; semantic `Source.kind` gains `trace_attribute`; dimension `rollout.ending`; measures `ending_<ending>_rate` (update entity). Milestones 1-3 changed no pin. Milestone 4 pins `carbonteq-trackio==0.31.5.post14.dev32` and adds `Source.fallback_attribute: str | None` (only on `trace_fact` sources), the `episode_ending` literal in `posttrain.tracking.TraceFactsQuery` and `PayloadDimension`, and `endings: Mapping[str, int]` on `TraceFactBackfillPage` and `TraceFactBackfillWindow`.

Revision note (2026-09-28): plan created and completed in one pass; all milestones implemented, validation recorded, and the Trackio follow-up described.

Revision note (2026-09-28, later): Milestone 4 (Trackio `episode_ending` fact column, dev32, calculator v9, fact-backed `rollout.ending`, backfill reporting) implemented and validated locally; release and deployment steps recorded for the maintainers.
