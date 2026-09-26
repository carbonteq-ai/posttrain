# Add run notes: Markdown notes on runs, with references to recorded data and job-type templates

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Today a run is only its recorded evidence: configuration, metrics, events, traces and artifacts. Why it was launched, what went wrong, what changed afterwards and what it proved live in chat history, pull request bodies and plan documents, so a reader of a run in the Observatory (the web app that shows runs) cannot tell a crashed setup attempt from a qualification result. The tracking project `posttrain-lab` holds 77 runs; 25 training runs failed before ten updates, and several evaluations are known to be invalid (one served the base model instead of its checkpoint; one failed all 642 rollouts), yet nothing on those runs says so.

After this change anyone can attach notes to any run. A note is Markdown text with an author, a kind, and a revision history: editing a note adds a revision and never destroys the earlier text. Notes can be written from the command line (`posttrain run note add`), by an agent through the Observatory's MCP server (the Model Context Protocol endpoint that language-model agents call as tools), and by a person in the Observatory web page.

Notes can refer to the run's recorded data instead of copying numbers. `{{run.settings.loop.learning_rate}}` shows the learning rate the run was configured with; `{{metric "train/rl/entropy" last}}` shows its last recorded entropy; `{{run:"lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2" metric "train/rl/reward_mean" mean}}` pulls a value from another run, so a comparison note stays correct. Every job type (for example `train.grpo`, `train.sampo`, `eval.general`) has a versioned Markdown template built from these references. Rendered for a run, the template is that run's "run card": model, algorithm, learning rate, KL penalty, batch, environment, budgets, parent run, and key results. Because templates exist per job type from the start, every run, old or new, gets a correct card without anyone writing it, and a language model writing a narrative note cites real values through references instead of typing numbers it might get wrong.

To see it working: `posttrain run card lfm12-sampo-turns-8gb-20260926-r1` prints the SAMPO card with that run's settings and results; `posttrain run note add lfm12-sampo-8gb-20260926-r11 --kind finding --body "..."` attaches a note; the run's Observatory page shows the card and the note, and the note's history after an edit shows both revisions.

## Progress

- [x] (2026-09-27 02:10Z) Mapped the Trackio fork (storage, HTTP registry, auth, run deletion paths, MCP, run page, release process) and the Posttrain side (canonical read-only statements, tracking protocols, Observatory service/HTTP/MCP/frontend, CLI, run snapshot, import-linter contracts). Findings are in Context and Orientation.
- [ ] Milestone 0: amend the canonical baseline to allow author-attributed notes.
- [ ] Milestone 1: Trackio fork `run_notes` store, server endpoints and client methods (backend only); fork release and Posttrain pin.
- [ ] Milestone 2: Posttrain `RunNote` model, `RunNoteStore` protocol and Trackio adapter.
- [ ] Milestone 3: reference language and renderer.
- [ ] Milestone 4: job-type templates and run cards.
- [ ] Milestone 5: Observatory service, HTTP, MCP and web page.
- [ ] Milestone 6: CLI `posttrain run note` and `posttrain run card`.
- [ ] Milestone 7: notes backfill for the existing runs, then the run cleanup.

## Surprises & Discoveries

- Observation: the Observatory is read-only by contract, not only by implementation.
  Evidence: `docs/post-training/04-framework.md` (the `apps/observatory` row, "Dedicated read product", and dependency rule 7, "Observatory is read-only with respect to execution state"), `docs/post-training/05-apis.md` ("tracking readers and Observatory are read-only"), `docs/post-training/06-observation-and-lineage.md` ("one read-only product", "No writes while reading").
- Observation: the self-hosted Trackio server authorizes writes with one shared secret, so it cannot tell who wrote a note.
  Evidence: `trackio/server.py` `check_write_access` accepts `TRACKIO_WRITE_TOKEN` from a header, cookie or query parameter; per-user identity exists only on Hugging Face Spaces through `whoami`.
- Observation: the Observatory has no user identity today.
  Evidence: `apps/observatory/src/posttrain_observatory/settings.py` `auth_mode: Literal["none", "ingress"]`; nothing reads an identity header.
- Observation: Posttrain pins Trackio `0.31.5.post14.dev27`, which lives on fork branch `codex/trace-payload-aggregates`, not on the branch the local fork checkout has open (`codex/artifact-finalization`, dev25).
  Evidence: `packages/tracking-trackio/pyproject.toml`; `git branch -r --contains carbonteq-v0.31.5.post14.dev27` in `/home/hammad/projects/trackio`.

## Decision Log

- Decision: store notes in Trackio next to the run, with revisions; write them from the CLI, MCP and the Observatory web page; do notes before cleaning up runs.
  Rationale: notes then appear wherever the run appears and are deleted with it; revisions let a person or an agent correct a note without destroying what was there; noting runs first makes the cleanup decision per run obvious (runs with a purpose worth recording are kept).
  Date/Author: 2026-09-27, user decision on Claude's recommendation.
- Decision: notes are Markdown, rendered sanitized (no raw HTML, no scripts, links only to http(s) and in-app run links).
  Rationale: user requirement; sanitization because notes are written by agents and shown in a browser.
  Date/Author: 2026-09-27, user requirement.
- Decision: notes may reference the recorded data of their run and of other runs; job-type templates exist from the first release, and a run card is the rendered template.
  Rationale: user requirement ("refer to a lot of things from original runs as templates ... support job type templates right from the start"). References keep numbers exact and let a language model write about runs without inventing values.
  Date/Author: 2026-09-27, user requirement.
- Decision: Trackio stores notes and knows nothing about references or templates; Posttrain parses references and renders templates.
  Rationale: Trackio is a generic tracker (fork policy in `AGENTS.md`: generic storage and API work goes to the fork, Posttrain-specific views stay here); only Posttrain knows the shape of each job type's recorded settings.
  Date/Author: 2026-09-27, Claude.
- Decision: the Trackio fork gets backend support only (storage, server endpoints, client methods). Notes are shown and edited in the Observatory, and agents write them through the Observatory's MCP server; the fork's own dashboard and MCP server are unchanged.
  Rationale: user decision ("we dont need to build this into trackio FE only its backend and observatory FE"); the Observatory is the product people and agents use, and one write path keeps authorship and rendering in one place.
  Date/Author: 2026-09-27, user decision.
- Decision: the author is supplied by the writer and required: `user:<email>` from the CLI (git `user.email` or `--author`), `agent:<model id>` from MCP (a required argument), and the identity header from the Observatory's ingress (the reverse proxy in front of it) for the web page, which refuses writes without an identity. Trackio also records a `verified_author` when it can derive one (Spaces).
  Rationale: the self-hosted Trackio token is shared, so the server cannot attribute writes; recording the claimed author plus how it was established is honest about what is known.
  Date/Author: 2026-09-27, Claude.
- Decision: the reference language is deliberately small (paths into the recorded run, a fixed set of metric reductions, and formatting filters) and never executes code; an unresolved reference renders as a visible marker, never as an empty string.
  Rationale: notes are rendered in the browser and by agents; a general template engine (for example Jinja) could execute arbitrary expressions, and a silent blank would hide a wrong reference.
  Date/Author: 2026-09-27, Claude.

## Outcomes & Retrospective

Not started.

## Context and Orientation

Posttrain is this repository, a Python 3.12 `uv` workspace (see `AGENTS.md`). A run is one execution of one job (for example a training or evaluation job) and is recorded by a tracking backend. The backend used here is a CarbonTeq fork of Trackio, checked out at `/home/hammad/projects/trackio` (remote `carbonteq-ai/trackio`), deployed as a self-hosted server at `https://trackio.carbonteq.com`, project `posttrain-lab`. Posttrain reads runs through the `RunDataSource` protocol in `packages/tracking/src/posttrain/tracking/contracts.py` (methods `list_runs`, `get_run`, `metric_series`, `traces`, and aggregates), implemented for Trackio by `TrackioDataSource` in `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`, which calls the fork's client object `trackio.Api` (`runs`, `run_configs`, `run_lifecycles`, `capabilities`). One writer already exists outside a live run: `TrackioTraceFactWriter` in the same adapter file builds a `RemoteClient(base_url, write_token=...)` from the environment variable `TRACKIO_WRITE_TOKEN` and calls a server endpoint without reopening the run. Notes follow that pattern.

`RunDataSource.get_run` returns a `RunDetail` (`packages/tracking/src/posttrain/tracking/models.py`): `summary` (a `RunSummary` with run id, work package id, job kind, status, start and finish times, error), `resolved_inputs` (the resolved selection for each seat of the job, for example `model`, `environment`, `settings`, `training`, `rollout_inference`), `source_metadata` (execution provider, job image, target, worker, job package), `metric_names`, `events` and `trace_count`. The checkpoint a run started from (`source_run_id`, `checkpoint_step`) is inside the `model` seat's resolved selection. `packages/advisor/src/posttrain/advisor/snapshot.py` already extracts seats deterministically (`seats`, `settings_seat`, `training_seat`, `environment_seat`, `target`), and the Observatory uses those helpers in `apps/observatory/src/posttrain_observatory/configuration.py`.

The Observatory is `apps/observatory`. `ObservatoryService` (`apps/observatory/src/posttrain_observatory/service.py`) composes run views from one or more `RunDataSource` objects; `http.py` exposes them under `/api/v1` (every POST today is a read, and CORS allows GET and POST); `mcp.py` exposes read tools (`list_runs`, `get_run_view`, `get_metric_series`, `summarize_run`, and others); checked-in contract snapshots `apps/observatory/mcp-schema.json` and `apps/observatory/openapi.json` must be regenerated when these change. The web page is a React app in `apps/observatory/frontend/src`; `App.tsx` chooses an overview component per run (`ServingBenchmarkOverview`, `EvaluationOverview`, `GenericOverview`) inside its `Overview` section, and `components/ContentRenderer.tsx` renders message content. `settings.py` has `auth_mode` `none` or `ingress`; production refuses `none` on a non-loopback host. The Observatory must not import `trackio` or `wandb` (import-linter contract in `pyproject.toml`), so it reaches notes only through a protocol in `posttrain.tracking`.

The command line is `apps/cli`. `apps/cli/src/posttrain_cli/commands/run_cmd.py` registers the `posttrain run` group and nests `posttrain run checkpoint` as a sub-group; `posttrain run show` builds an `ObservatoryService` through `project_observatory_settings` and `create_service`. Tracking credentials come from `posttrain_cli.tracking_config.project_tracking_environment(layout)`; writers read `TRACKIO_WRITE_TOKEN` from that mapping. Never print or log that value.

In the Trackio fork, storage is chosen by `trackio/storage.py` `get_storage()`: `SQLiteStorage` (`trackio/sqlite_storage.py`, one database per project, schema created idempotently in `init_db`, columns added with guarded `ALTER TABLE`) or `DorisStorage` (a SQL warehouse backend with an explicit schema version in `trackio/doris_schema.py`, `SCHEMA_VERSION = 3`, migrations in `migration_statements()`). The `alerts` table is the pattern to copy: rows keyed by run, an optional client-supplied id with a unique index for idempotent writes, a bulk write (`bulk_alert`) and reads (`get_alerts`). The server is Starlette: `POST /api/{api_name}` dispatches to functions registered in `_api_registry()` in `trackio/server.py`, binding arguments by signature and injecting `request`. Writes are authorized by `check_write_access` and the guards `assert_can_write_metrics` and `assert_can_mutate_runs`. Run deletion, purge, rename and cross-project move each list their tables explicitly (`delete_run`, `purge_runs`, `rename_run`, `move_run` in `sqlite_storage.py`; `delete_run`, `purge_runs`, `rename_run`, `delete_project` in `doris_storage.py`; `AUTHORITATIVE_TABLES` and `_RUN_SCOPED_TABLES` in `trackio/doris_migration.py`; parquet export tables in `sqlite_storage.py`). The fork also has its own dashboard and MCP server; this plan does not change them. Versions live in both `pyproject.toml` (distribution `carbonteq-trackio`) and `trackio/_version.py` and must match; releases are tagged `carbonteq-v<version>`, built locally, published as an immutable GitHub pre-release, then copied to the internal index `carbonteq/dev` by Posttrain's workflow `.github/workflows/publish-trackio-internal.yml`, then promoted to stable (`docs/tooling/forks.md`, the fork's `CARBONTEQ_FORK.md`).

Terms used below. A "note" is one Markdown text attached to a run or to a project, identified by a `note_id` that stays the same across edits. A "revision" is one saved version of a note; revision 1 is the first, each edit adds the next number, and a deletion is a final "tombstone" revision that hides the note without erasing history. A "reference" is a `{{...}}` expression or `[[...]]` link inside note Markdown that the renderer replaces with a value from recorded run data. A "template" is a versioned Markdown file for one job kind, written with references. A "run card" is a template rendered for one run.

## Plan of Work

Milestone 0 amends the canonical baseline before any code, because `AGENTS.md` requires the product documents to change first when a change alters product meaning. In `docs/post-training/04-framework.md`, amend the `apps/observatory` package row and dependency rule 7 to read that the Observatory is read-only with respect to execution state and evidence, with one exception: it may create and revise author-attributed notes, which never change a run's configuration, metrics, events, traces, artifacts or status. In `docs/post-training/05-apis.md`, add the same exception where tracking readers and the Observatory are called read-only, add `posttrain run note add|edit|delete|show|history` and `posttrain run card` to the CLI command list, and add the note and card surfaces to the tracking and Observatory API listing. In `docs/post-training/06-observation-and-lineage.md`, add a short "Run notes" subsection: what a note is, revisions and tombstones, authorship (claimed and verified), references and templates, sanitized rendering, and that notes follow the run on deletion. Commit this amendment alone.

Milestone 1 implements notes in the Trackio fork, in `/home/hammad/projects/trackio`, on a new branch created from tag `carbonteq-v0.31.5.post14.dev27` (the version Posttrain pins). Add a `run_notes` table to `SQLiteStorage.init_db` with columns `id` (integer primary key), `note_id` (text, stable across revisions), `revision` (integer from 1), `scope` (`run` or `project`), `run_id` and `run_name` (null for project notes), `kind` (text), `title` (nullable text), `body_md` (text), `author` (text), `verified_author` (nullable text), `created_at` (time of revision 1, copied forward), `revised_at`, `deleted` (0 or 1), `parent_revision` (nullable integer) and `metadata` (JSON text), with a unique index on `(note_id, revision)` and indexes on `run_id` and `(scope, revised_at)`. Add the same table to Doris as schema version 4 with `UNIQUE KEY(project_id, note_id, revision)`, update `MANAGED_TABLES`, `migration_statements()`, `AUTHORITATIVE_TABLES` and `_RUN_SCOPED_TABLES`. Add storage methods `add_run_note`, `revise_run_note`, `delete_run_note` (tombstone), `get_run_notes` (latest non-deleted revision of each note, filtered by run, scope and kind) and `get_run_note_history`. A revise or delete must pass `expected_revision`; if the latest revision differs, fail with a conflict error that names the current revision, so two editors never overwrite each other silently. Adding with a client-supplied `note_id` that already exists at revision 1 with identical content is a no-op, which makes retries safe. Add `run_notes` to every run deletion, purge, rename and cross-project move path and to parquet export. Register server endpoints `get_run_notes`, `get_run_note_history`, `add_run_note`, `revise_run_note` and `delete_run_note` in `_api_registry`; the three writes call `assert_can_mutate_runs` and are written synchronously. Add `Api.run_notes`, `Api.run_note_history`, `Api.add_run_note`, `Api.revise_run_note` and `Api.delete_run_note`, plumb an optional `write_token` through `Api.__init__`, and add `"run_notes": True` to `Api.capabilities()`. The fork's dashboard and MCP server are not changed: people read and edit notes in the Observatory, and agents write them through the Observatory's MCP server. Update `CARBONTEQ_FORK.md`, bump the version in both places to `0.31.5.post14.dev28`, release, publish to `carbonteq/dev`, and pin it in Posttrain (`packages/tracking-trackio/pyproject.toml`, `uv.lock`, Quality workflow wheel, `release/forks.toml`, `docs/tooling/trackio/README.md`) following the same sequence used for renderers `0.1.12.post1.dev2`. The deployed Trackio server must be upgraded before notes can be written in production; that is an ai-infra deployment, confirmed with the user before it is done.

Milestone 2 adds the Posttrain tracking surface. In `packages/tracking/src/posttrain/tracking/models.py`, add `RunNote` (note id, revision, scope, run id, kind, title, Markdown body, author, verified author, created and revised times, deleted flag) and add `run_notes: bool = False` to `TrackingCapabilities`. In a new `packages/tracking/src/posttrain/tracking/notes.py`, define the protocol `RunNoteStore` with `list_notes`, `note_history`, `add_note`, `revise_note` and `delete_note`, and a `NoteConflict` error. Keep it separate from `RunDataSource` so readers stay write-free. In `packages/tracking-trackio`, implement `TrackioRunNotes` using the `trackio.Api` read methods and a `RemoteClient` with the write token for writes, following `TrackioTraceFactWriter`, and report the capability only when the server's `capabilities()` says `run_notes`. The W&B adapter reports the capability as unsupported.

Milestone 3 adds the reference language and renderer in a new module `apps/observatory/src/posttrain_observatory/note_render.py` (the Observatory owns rendering; it reads through `RunDataSource`). The grammar is small and fixed. `{{run.<path>}}` reads a dotted path from the run's `RunDetail`, with friendly roots: `run.id`, `run.status`, `run.job_kind`, `run.work_package`, `run.started_at`, `run.finished_at`, `run.duration`, `run.settings.<...>` (the settings seat), `run.training.<...>`, `run.environment.<...>`, `run.model.<...>`, `run.inference.<...>` (the rollout or evaluation inference seat), `run.parent` (the source run and checkpoint step), `run.image`, and `run.inputs.<seat>.<...>` for any other seat. `{{metric "<name>" <reduction>}}` reads a metric series, with reductions `first`, `last`, `min`, `max`, `mean`, `count`, `delta` (last minus first), `at <step>`, and `argmax`/`argmin` (the step of the extreme). Either form may be addressed to another run with a prefix, `{{run:"<run id>" ...}}`. Filters after a pipe format the value: `round <n>`, `percent`, `duration`, `sci` (scientific notation), and `default "<text>"`. `[[run:<run id>]]` renders a link to that run's Observatory page, and `[[run:<run id>|label]]` sets its text. Parsing produces a list of text and reference nodes; nothing is evaluated as code. A reference that cannot be resolved renders as a visible marker `⟦unresolved: <reference> — <reason>⟧`. Rendering fetches each referenced run and metric once per render. Values are escaped before they are placed into Markdown. The rendered result is Markdown; the Observatory page renders it to HTML with a sanitizing Markdown component (no raw HTML, no scripts, links only to http(s) and in-app run links), and the CLI prints it as terminal text.

Milestone 4 adds job-type templates and run cards. Framework templates live in `apps/observatory/src/posttrain_observatory/note_templates/<job kind>.md`, one per job kind the framework registers (`train.sft`, `train.dpo`, `train.grpo`, `train.sampo`, `train.gdpo`, `train.capo`, `train.distill`, `eval.general`, `eval.domain`, `serve.benchmark`, `serve.smoke`, `model.transform`, `data.*` kinds as registered), plus `generic.md` for any kind without its own. A project can override a template by placing a file of the same name in its `.posttrain/note_templates/` directory, the same way the project catalog overlays the framework catalog. Each template begins with a front-matter line naming its template id and revision (for example `template: train.sampo@1`) so a card records which template produced it. The SAMPO template, for example, shows model and parent checkpoint, environment and task count, the settings (updates, prompt groups by rollouts, learning rate, KL penalty, discount, step advantage weight, candidate batches, curriculum policy, truncation penalty), the training and inference bindings, and results (updates completed, reward first and last, entropy first and last, KL last, mean update time, share of turns with step rewards, truncation rate, whether the vLLM correction clamped). `run_card(detail, source)` renders the job kind's template for a run. A template that references a setting a run does not have (for example an older configuration) shows the unresolved marker for that line rather than failing the whole card; a test renders every framework template against a recorded run of that kind.

Milestone 5 wires notes and cards into the Observatory. `ObservatoryService` gains an optional `RunNoteStore` per source and methods `run_notes(run_key)`, `run_note_history(run_key, note_id)`, `add_run_note`, `revise_run_note`, `delete_run_note`, `render_note` and `run_card(run_key)`. `http.py` adds `GET /api/v1/runs/{run_key}/notes`, `GET /api/v1/runs/{run_key}/notes/{note_id}/history`, `POST /api/v1/runs/{run_key}/notes` (add), `PUT /api/v1/runs/{run_key}/notes/{note_id}` (revise, with `expected_revision`), `DELETE /api/v1/runs/{run_key}/notes/{note_id}`, `GET /api/v1/runs/{run_key}/card`, and `POST /api/v1/notes/preview` (render Markdown with references for a run without saving); CORS gains PUT and DELETE. Writes require an author: in `ingress` mode, from a configured identity header (a new setting, for example `identity_header: X-Forwarded-Email`); a write without identity is refused with 401. `mcp.py` adds read tools `list_run_notes`, `get_run_note_history`, `get_run_card`, `render_note_preview` and write tools `add_run_note`, `revise_run_note` (both require an `author` argument such as `agent:claude-opus-5-5`); write tools are registered only when the deployment enables note writing and holds the write token server-side. The frontend adds `components/RunNotesPanel.tsx`, rendered in the `Overview` section for every overview kind: the run card at the top, then notes newest first with author, kind and time, an editor with a live preview (through the preview endpoint), and a history view. Regenerate `openapi.json` and `mcp-schema.json`.

Milestone 6 adds the CLI: a nested `note` sub-group in `run_cmd.py` with `posttrain run note add <run id> --kind <kind> --body <text> | --file <path> [--title] [--author]`, `edit <run id> <note id> --expected-revision <n> ...`, `delete`, `show <run id> [--raw]` (rendered by default), and `history <run id> <note id>`; and `posttrain run card <run id>`. The author defaults to `user:<git user.email>`.

Milestone 7 backfills notes and then cleans up. For every run in `posttrain-lab`, write one `summary` note with its purpose and outcome, generated by a language model that reads the run card, its events and error, and the plan documents that cite it, and that uses references for every number. Record known problems as `correction` notes (for example that `eval-lfm26-automationbench-heldout20x3-olmo3-adaptive-oversample10x4-12k-lfmrec-20260915-r1` served the base model and that `lfm12-screen-8k-20260926-r1` failed every rollout). Then propose the cleanup list to the user: training runs with fewer than ten updates and evaluations with no valid result, excluding runs cited as evidence in the repository and runs whose checkpoints other runs consume; delete only after the user confirms the list, through `posttrain run purge` and `posttrain purge apply`.

## Concrete Steps

Commands run from `/home/hammad/projects/rl-perf-guard` unless stated; jobs and demonstrations run from a clean worktree of committed code (Posttrain packs the working tree).

Milestone 1, in the fork:

    cd /home/hammad/projects/trackio
    git fetch origin
    git switch -c codex/run-notes carbonteq-v0.31.5.post14.dev27
    python -m pytest -q tests/unit

Milestones 2 to 6, in Posttrain:

    uv sync --all-packages --locked --python 3.13 --group dev --extra trl --extra verifiers
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    (cd apps/observatory/frontend && npm ci && npm test && npm run build)

Demonstration against a local Trackio server started from the fork with SQLite storage (so nothing touches the shared server until it is upgraded):

    uv run posttrain run card lfm12-sampo-turns-8gb-20260926-r1
    uv run posttrain run note add lfm12-sampo-8gb-20260926-r11 --kind finding --body 'Rollout collection worked; the first update ran out of memory at {{metric "train/rl/rollout_selected_tokens" max}} tokens.'
    uv run posttrain run note show lfm12-sampo-8gb-20260926-r11

## Validation and Acceptance

Milestone 1 is accepted when the fork's unit suite passes with new tests that fail before the change: adding a note and reading it back; revising with the right `expected_revision` creates revision 2 and history returns both; revising with a stale revision fails with a conflict naming the current revision; deleting adds a tombstone that `get_run_notes` hides and history keeps; an add retried with the same `note_id` and body is a no-op; deleting, renaming and moving a run carries or removes its notes; writes without the write token are refused; the client methods round-trip against a local server.

Milestone 3 is accepted when renderer tests cover every reference form and reduction against a fixture run, an unknown path and an unknown metric render the unresolved marker, a reference to another run resolves through the same source, and a body containing `{{__import__("os")}}` renders as unresolved text without evaluation.

Milestone 4 is accepted when every framework template renders against a recorded run of its kind (fixtures) with no unresolved markers for settings that run records, and `posttrain run card` on the real runs `lfm12-sampo-turns-8gb-20260926-r1` (SAMPO), `lfm26-vortex-v5-150-lr5e5-kl5e3-20260926-r2` (GRPO, OLMo 3) and `eval-lfm26-heldout-64k-base-20260924-r1` (evaluation) prints their true learning rate, KL penalty, batch and final reward (compare with the values in their Trackio configuration and metrics).

Milestone 5 is accepted when, against a local Trackio server, the Observatory run page shows the card and a note rendered from Markdown (and a note containing a `<script>` tag renders without it), editing the note in the page produces revision 2 with history, a write without an identity header in `ingress` mode returns 401, and an MCP client calling `add_run_note` with `author: agent:test` creates a note that the page shows.

Milestone 6 is accepted when the three demonstration commands above work and `posttrain run note edit` with a stale `--expected-revision` prints the conflict.

Milestone 7 is accepted when every run in `posttrain-lab` has a summary note, the user has reviewed the cleanup list, and only the confirmed runs are purged.

## Idempotence and Recovery

Schema creation in SQLite is idempotent; the Doris migration is versioned and refuses to start on a mismatch, so it is applied deliberately by the operator with the new server version. Note writes are safe to retry: an add with a client `note_id` is a no-op when it already exists with the same content, and revisions are guarded by `expected_revision`. Nothing in this plan deletes notes' history except deleting the run itself. The shared Trackio server is not modified until its upgrade is confirmed with the user; all development uses a local fork server. If the upgrade must be rolled back, the previous server version ignores the extra table in SQLite; in Doris, version 4 must be rolled back with the version 3 server only after dropping `run_notes`, so keep a database backup before migrating.

## Artifacts and Notes

The run cleanup that motivated this plan inventoried the project on 2026-09-27: 77 Trackio runs (76 with Posttrain metadata and one orphan without it), of which 42 are training runs (25 failed, 15 succeeded, 1 running, 1 cancelled), 23 evaluations and 11 release-candidate canaries. Training runs with fewer than ten updates include the 8 GB SAMPO setup attempts `lfm12-sampo-8gb-20260926-r2` to `-r12` and the VORTEX v5 launch attempts; several short runs (`lfm12-sampo-8gb-20260926-r13`, `-r15`, `lfm12-sampo-turns-8gb-20260926-r1`, `lfm26-v5-opt-ab-dspark-20260926-r1`, `lfm26-v5-opt-breakdown-20260926-r1`) are cited as evidence in plans, recipes and release notes.

## Interfaces and Dependencies

In `packages/tracking/src/posttrain/tracking/models.py`:

    class RunNote(TrackingModel):
        note_id: str
        revision: int
        scope: Literal["run", "project"]
        run_id: str | None
        kind: str
        title: str | None
        body_md: str
        author: str
        verified_author: str | None
        created_at: datetime
        revised_at: datetime
        deleted: bool = False

In `packages/tracking/src/posttrain/tracking/notes.py`:

    class NoteConflict(Exception): ...

    class RunNoteStore(Protocol):
        async def list_notes(self, run_id: str | None = None, *, kind: str | None = None) -> tuple[RunNote, ...]: ...
        async def note_history(self, note_id: str) -> tuple[RunNote, ...]: ...
        async def add_note(self, *, run_id: str | None, kind: str, body_md: str, author: str,
                           title: str | None = None, note_id: str | None = None) -> RunNote: ...
        async def revise_note(self, note_id: str, *, expected_revision: int, body_md: str, author: str,
                              kind: str | None = None, title: str | None = None) -> RunNote: ...
        async def delete_note(self, note_id: str, *, expected_revision: int, author: str) -> RunNote: ...

In `apps/observatory/src/posttrain_observatory/note_render.py`:

    def parse(body_md: str) -> tuple[TextNode | ReferenceNode | LinkNode, ...]: ...
    async def render(body_md: str, *, run_id: str, source: RunDataSource, link_for: Callable[[str], str]) -> str: ...
    async def run_card(run_id: str, *, source: RunDataSource, templates: TemplateSet) -> RenderedCard: ...

Dependencies: the Observatory frontend renders Markdown with a sanitizing Markdown component (reuse its existing content renderer if it already renders Markdown safely, otherwise add one library with a sanitizer and commit the lockfile); the Trackio fork needs no frontend or new dependency; no new Python dependency is required for the reference language (hand-written parser).
