# Discover Trackio projects dynamically in Observatory

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`. The implementation spans two sibling repositories: `/home/owayys/Projects/carbonteq-ai/posttrain` owns the Observatory product and Trackio adapter, while `/home/owayys/Projects/carbonteq-ai/ai-infra` owns its deployed configuration and qualification. Make separate, reviewable commits in those repositories; no dependency pin changes or changes to the CarbonTeq Trackio fork are required.

## Purpose / Big Picture

After this change, an operator can create a project in the shared Trackio service and see it become an Observatory evidence source without editing or redeploying a JSON project list. Observatory performs one discovery before serving normal traffic, repeats it at a configurable interval, and offers `POST /api/v1/sources/refresh` so an authenticated user can request the same reconciliation immediately. The browser calls Observatory only. It never calls Trackio or receives Trackio connection details.

A successful refresh atomically replaces only the discovery-owned Trackio portion of the source registry. Explicitly configured Trackio, W&B, and fixture sources remain intact. A failed refresh records a safe error and retains the last successful registry snapshot. A project omitted by a later successful Trackio response is removed immediately from the discovery-owned portion; omission during a failed response removes nothing.

The frozen post-training product baseline does not need an amendment. The canonical documents already assign provider composition to `apps/observatory`, provider integration to `packages/tracking-trackio`, and source identity to provider-neutral Observatory locators. This change makes composition dynamic without changing the meaning of runs, jobs, projects, metrics, traces, artifacts, or lineage.

## Progress

- [x] (2026-07-30 11:07Z) Re-read repository instructions, the ExecPlan template, canonical framework/API/observation documents, and the affected source, frontend, test, and deployment files on pulled `posttrain/main` at `9225a36` and `ai-infra/main` at `2d343c0`.
- [x] (2026-07-30 11:07Z) Verified the deployed Trackio discovery contract at `https://trackio.lan/api/get_all_projects`; it returned eight project names while the current Observatory compose template contains four.
- [x] (2026-07-30 11:19Z) Synced pulled `main` with its locked Python 3.13 environment and launched a current-state loopback Observatory preview against all eight live Trackio projects; readiness reported eight healthy sources, the UI returned HTTP 200, and six sources contained 28 canonical posttrain runs.
- [x] (2026-07-30 11:36Z) Implemented and tested the Trackio project catalog adapter and strict discovery settings in `posttrain`.
- [x] (2026-07-30 11:36Z) Implemented atomic source reconciliation, safe refresh state, startup refresh, periodic refresh, serialized manual refresh, and clean shutdown in `posttrain`.
- [x] (2026-07-30 11:36Z) Added and tested the POST refresh HTTP contract and the frontend refresh control inside the `Evidence backend` popover header.
- [x] (2026-07-30 11:36Z) Removed the deployed hardcoded list, enabled 300-second discovery, rewrote dynamic qualification, and added a credential-free deployment self-test in `ai-infra`.
- [ ] Final release verification (completed: 45 focused backend tests, 802 full tests with 17 expected skips, 21 frontend tests, production frontend build, focused Pyright, Ruff, import contracts, deployment self-test, Ansible syntax, and real `trackio.lan` startup/manual refresh all pass; remaining: user browser acceptance and immutable-image deployment/live authenticated qualification).

## Surprises & Discoveries

- Observation: The pinned Trackio Python `Api` supports runs for a known project but does not expose a public project-list method, even though the server and `RemoteClient` expose `/get_all_projects`.
  Evidence: `posttrain/.venv/lib/python3.12/site-packages/trackio/server.py` defines `get_all_projects()`, and `remote_client.py` supports `predict(api_name="/get_all_projects")`; `api.py::Api` has no corresponding method.

- Observation: Constructing `trackio.Api(server_url=...)` negotiates with the remote server synchronously. Discovery and discovered-source construction must therefore run off the event loop and must complete before publishing a new registry snapshot.
  Evidence: a direct `TrackioDataSource` construction enters `trackio.remote_client.RemoteClient` and performs remote transport setup.

- Observation: Deployed authentication is a Caddy ingress boundary, not FastAPI credential validation. Caddy applies Basic Auth to the entire `observatory.lan` site before reverse proxying to Observatory.
  Evidence: `ai-infra/ansible/roles/control/templates/Caddyfile.j2` wraps `reverse_proxy observatory:7861` in `basic_auth`, and `POSTTRAIN_OBSERVATORY_AUTH=ingress` is a safety declaration in settings.

- Observation: The frontend derives backend/project selectors from `/api/v1/runs`; it does not contain a fixed project allowlist. A refresh button only needs to call Observatory and then reload the run list.
  Evidence: `apps/observatory/frontend/src/App.tsx` builds `sourceOptions` and `projects` from the `runs` state.

- Observation: On 2026-07-30, Trackio returned `ai-infra-qualification`, `ambient-agent`, `artifact-transport-qualification`, `foundation-models`, `grpo-demo`, `infrastructure-executions`, `occupancy-research`, and `sft-demo`. The compose template configures only four of them.
  Evidence: `curl -H 'content-type: application/json' -d '{"args":[],"kwargs":{}}' https://trackio.lan/api/get_all_projects` returned those eight values.

- Observation: The pulled `main` workspace requires Python 3.13 and pins Trackio commit `703be3808cb6cac52259cb15e614cad971978d2e`. The pre-pull virtual environment was Python 3.12 with an older Trackio API, so a first live probe failed at `Api.run_configs`; a locked Python 3.13 sync corrected the local dependency mismatch.
  Evidence: `uv sync --all-packages --locked --python 3.12` rejected every workspace package, while the Python 3.13 sync installed `posttrain` 0.2.4 and `carbonteq-trackio` 0.31.5.post5 from the pinned commit. This was an environment mismatch, not a defect to work around in discovery code.

- Observation: The execution sandbox blocks Python 3.13 worker threads, causing existing `asyncio.to_thread` Trackio tests to hang rather than fail. The same tests complete normally outside that restricted sandbox.
  Evidence: a minimal `asyncio.to_thread(lambda: 42)` timed out in the sandbox; the focused suite passed 45 tests in 2.50 seconds and the full suite passed 802 tests in 15.68 seconds with the approved test runner outside it.

- Observation: Repository-wide Pyright currently reports unrelated pre-existing symbol errors across `apps/lab`, `apps/runtime`, and several reusable packages on pulled `main`; the changed Observatory and Trackio scope is clean.
  Evidence: `uv run pyright apps/observatory packages/tracking-trackio` reports zero errors, while unscoped `uv run pyright` reports 104 errors in files untouched by this plan. Ruff and all eight import-linter contracts pass repository-wide.

## Decision Log

- Decision: Use each exact Trackio project name as the discovered `source_id` and as `TrackioDataSource.project`.
  Rationale: This is the requested stable identity, removes an unnecessary translation table, and makes qualification a direct set-containment comparison with Trackio.
  Date/Author: 2026-07-30 / Codex

- Decision: Keep configured sources and discovery-owned sources as separate registry layers, with explicitly configured sources winning on an ID collision.
  Rationale: Replacing only the discovered layer guarantees preservation of manual Trackio and non-Trackio sources. Explicit configuration is intentional operator policy and must not be silently displaced by a coincidentally named Trackio project.
  Date/Author: 2026-07-30 / Codex

- Decision: Remove a discovery-owned source immediately after, and only after, a successful discovery response omits its project.
  Rationale: `/get_all_projects` is Trackio's authoritative current catalog. Delayed tombstones add policy and state without protecting against the important failure case, which is already handled by retaining the last successful snapshot. Tests will distinguish a successful empty/contracted response from an exception.
  Date/Author: 2026-07-30 / Codex

- Decision: Complete source construction before one atomic registry swap; readers capture one immutable snapshot per registry operation.
  Rationale: No request should observe a partially populated project list. If discovery or construction of any new `TrackioDataSource` fails, the previous map remains published.
  Date/Author: 2026-07-30 / Codex

- Decision: Serialize startup, periodic, and HTTP-triggered refreshes through one `asyncio.Lock` and return the result of the actual refresh to every caller.
  Rationale: This prevents overlapping remote calls and last-writer races while keeping the 300-second normal interval inexpensive.
  Date/Author: 2026-07-30 / Codex

- Decision: Perform startup discovery synchronously in the FastAPI lifespan before yielding, but do not abort process startup when Trackio is unavailable.
  Rationale: The first normal request sees either the discovered catalog or an explicit failed refresh status. A temporary Trackio outage should leave manual sources usable and allow periodic/manual recovery.
  Date/Author: 2026-07-30 / Codex

- Decision: Treat the deployed POST endpoint as authenticated through the existing Caddy boundary and prove both unauthenticated rejection and authenticated success in qualification.
  Rationale: Adding a second application credential would conflict with the current ingress-auth design. Local `auth_mode=none` remains intentionally usable on loopback for development.
  Date/Author: 2026-07-30 / Codex

- Decision: Add a compact refresh button inside the open Backend popover, at the right edge of its `Evidence backend` header row; after a successful POST it fetches `/api/v1/runs` again without navigating away.
  Rationale: Source discovery is contextual to the backend list, and placing the action in the popover avoids adding permanent sidebar chrome. The action refreshes all discovered evidence backends, not only the selected entry, and ensures the browser never contacts Trackio.
  Date/Author: 2026-07-30 / Codex

## Outcomes & Retrospective

Implementation is complete in both worktrees. A discovery-only loopback server performed startup discovery and an explicit POST refresh against `https://trackio.lan`; both reported the same eight stable source IDs, all eight source health probes passed, and readiness remained `ready`. Unit tests prove successful removal, failure retention, configured-source preservation, periodic execution, concurrent serialization, and whole-snapshot reads. The frontend action is inside the agreed popover header and its success/failure behavior passes component tests. Deployment configuration and qualification are implemented but have not been applied because immutable-image publication and infrastructure deployment require a committed posttrain revision. User browser acceptance and deployed authenticated qualification remain release gates.

## Context and Orientation

`posttrain/apps/observatory/src/posttrain_observatory/settings.py` reads process configuration. Today `configured_sources()` either parses `POSTTRAIN_OBSERVATORY_SOURCES` or creates one legacy source from `POSTTRAIN_OBSERVATORY_SOURCE`. Discovery must be opt-in so existing local and W&B compositions do not change.

`posttrain/apps/observatory/src/posttrain_observatory/composition.py` is the only module allowed to select concrete adapters. It currently creates every `TrackioDataSource` once and hands a plain mapping to `ObservatoryService`.

`posttrain/apps/observatory/src/posttrain_observatory/sources.py` owns `RunSourceRegistry`. Its current dictionary never changes. A registry snapshot in this plan means an immutable mapping from stable source ID to `RunDataSource`. Reconciliation constructs a complete replacement mapping and publishes it in one assignment, so concurrent readers see the complete old mapping or the complete new mapping, never an in-between state.

`posttrain/packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py` owns knowledge of Trackio. Add project catalog access here rather than teaching Observatory Trackio's raw `/get_all_projects` protocol. The adapter may use the pinned Trackio `RemoteClient` because the higher-level `trackio.Api` currently lacks this call. Validate the returned value as a list of unique, non-empty strings and return it sorted as a tuple. Run the synchronous call via `asyncio.to_thread` from the discovery coordinator so startup and refresh do not block unrelated asynchronous requests.

`posttrain/apps/observatory/src/posttrain_observatory/http.py` creates the FastAPI app. Its lifespan will own the initial refresh, periodic task, cancellation, and shutdown. All HTTP routes, including the new refresh route, are protected by Caddy in production. The response must not contain the Trackio URL or secrets.

`posttrain/apps/observatory/frontend/src/App.tsx` loads all runs once and derives its source and project controls from those run records. `apps/observatory/frontend/src/lib/api.ts` is the browser's only transport layer. The refresh UI belongs beside `SourceSelector`; it calls the new Observatory POST and then calls `api.runs()`.

`ai-infra/ansible/roles/observatory/templates/compose.yml.j2` currently lists four JSON sources. Replace that list and the legacy single-project variables with discovery configuration while retaining `POSTTRAIN_TRACKIO_SERVER_URL=http://trackio:7860`. `ai-infra/ansible/playbooks/qualify-observatory.yml` currently asserts an exact four-source set. It must instead invoke refresh through authenticated Observatory, read Trackio's catalog through the control-plane Trackio endpoint for qualification only, and prove every Trackio project is present and healthy in Observatory. This direct Trackio call is an Ansible operator-side assertion, not a browser behavior.

## Plan of Work

First, extend `ObservatorySettings` in `apps/observatory/src/posttrain_observatory/settings.py` with `discover_trackio_projects: bool = False` and `trackio_discovery_interval_seconds: int = Field(default=300, ge=1)`. Parse `POSTTRAIN_OBSERVATORY_DISCOVER_TRACKIO_PROJECTS` with an explicit strict boolean parser accepting only conventional true/false forms and parse `POSTTRAIN_TRACKIO_DISCOVERY_INTERVAL_SECONDS` as an integer. When discovery is true, require `trackio_server_url`, and make `configured_sources()` return only explicitly supplied `sources` rather than manufacturing the legacy single Trackio source. When discovery is false, retain all current fallback behavior. Add settings tests for default-off compatibility, enabled values, invalid booleans/intervals, the required Trackio URL, and coexistence with explicit W&B/fixture/Trackio sources.

Second, add a small exported `TrackioProjectCatalog` in `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`. It stores the remote server URL and has a synchronous `list_projects() -> tuple[str, ...]` that calls Trackio's remote `/get_all_projects`, validates the payload, deduplicates defensively, and sorts deterministically. Do not put refresh scheduling or Observatory models in this package. Add adapter tests that fake the Trackio remote client and cover a normal list, duplicates/order, malformed values, and propagated transport failure. If direct use of `trackio.remote_client.RemoteClient` is required, document that this is pinned-fork API usage in the class docstring; do not add `httpx` as a direct dependency merely to duplicate Trackio's transport.

Third, rework `RunSourceRegistry` in `apps/observatory/src/posttrain_observatory/sources.py`. Store `_configured_sources`, `_discovered_sources`, and one immutable `_snapshot`. Permit an empty initial mapping so discovery-only startup can represent “not yet discovered.” Add `reconcile_discovered(sources: Mapping[str, RunDataSource]) -> tuple[str, ...]`; validate IDs, discard discovered entries whose IDs collide with configured entries, build the full map, and atomically assign a `MappingProxyType` snapshot while holding a short `threading.Lock`. Never await while holding that lock. Make `source_ids`, `resolve`, `sources`, `list_runs`, and `work_package_view` capture one snapshot at entry. Refactor the composite `work_package_view` helper so its list and subsequent resolves use that same captured mapping. Preserve the current deterministic source sorting and source-isolation behavior.

Add `apps/observatory/src/posttrain_observatory/discovery.py` to own orchestration rather than putting lifecycle state in the registry. Define a strict `SourceRefreshStatus` model in `models.py` with `enabled`, `state` (`disabled`, `pending`, `refreshing`, `succeeded`, or `failed`), `last_attempt_at`, `last_success_at`, `error`, and `discovered_source_ids`. Define `TrackioSourceDiscovery` with the Trackio catalog, server URL, interval, registry, source factory, clock, and sleeper injection points. `refresh()` acquires one async lock, marks an attempt, runs catalog lookup and construction away from the event loop, reconciles only after all construction succeeds, and updates timestamps/status. Catch ordinary backend exceptions at this boundary, sanitize them to a concise message, retain the registry snapshot, and return failed status; do not catch cancellation. `run_periodically()` sleeps for the configured interval before each subsequent refresh and exits cleanly on cancellation. Disabled composition exposes a disabled status and does not create a task.

Fourth, update `composition.py` to construct configured adapters as it does today, build the registry, and conditionally build `TrackioSourceDiscovery`. Pass both into `ObservatoryService`. Adjust `ObservatoryService` in `service.py` to accept an already-created registry without breaking current mapping/source callers, and add narrow methods `start_source_discovery()`, `stop_source_discovery()`, `refresh_sources()`, and `source_refresh_status()`. Keep provider selection out of the service.

Fifth, give `create_http_app` in `http.py` an `asynccontextmanager` lifespan. On entry, await the first discovery refresh and start the periodic task; on exit, cancel and await it without leaking `CancelledError`. Add `POST /api/v1/sources/refresh` to perform and return a refresh. Include the current safe status under a `source_refresh` field in `/health/ready`, while preserving the existing `status` and `sources` fields. Readiness is `ready` when at least one source is healthy and `degraded` otherwise; a refresh error does not erase healthy sources or independently force degradation. Update the checked-in `apps/observatory/openapi.json` and generated frontend API schema.

Focused backend tests belong in `apps/observatory/tests/test_sources.py` and `test_http.py`, with a small `test_discovery.py` if that keeps responsibilities clearer. Test startup discovery before the first request, periodic reconciliation with an injected controllable sleeper, manual POST, GET status, error serialization, and shutdown cancellation. Test that a successful contracted catalog removes an old discovered project, a failed catalog retains it, and a configured fixture/W&B-equivalent fake survives both. For concurrency, coordinate readers and reconciliation with events/barriers and assert every observed `source_ids` value is exactly the complete old or complete new snapshot, never a mixed set; also exercise `work_package_view` across a swap. Test simultaneous periodic/manual requests are serialized. Avoid timing-based sleeps in unit tests.

Sixth, add `SourceRefreshStatus` and `api.refreshSources()` to `frontend/src/lib/api.ts`. In `App.tsx`, place a compact icon button at the right end of the `Evidence backend` header row inside the open Backend popover. Give it the accessible label and tooltip `Refresh evidence backends`, because it refreshes the complete discovered registry rather than only the active entry. Keep the popover open when it is clicked, disable the button and replace the icon with a spinner while the request is active, and show a concise safe failure directly beneath the header. On click, POST only to `/api/v1/sources/refresh`, then reload `/api/v1/runs`; preserve the currently selected run when it still exists, and otherwise select the first available run using the existing `chooseRun` path. Add Vitest coverage that asserts the placement and accessible name, the POST target, the popover remaining open, disabled/loading behavior, successful run-list update, selection preservation, and visible failure. Run the production build so OpenAPI-derived types and the checked-in `dist` artifact follow existing repository policy.

Seventh, edit `ai-infra/ansible/roles/observatory/templates/compose.yml.j2` to remove `POSTTRAIN_OBSERVATORY_SOURCES`, `POSTTRAIN_OBSERVATORY_SOURCE_ID`, and `POSTTRAIN_TRACKIO_PROJECT`. Set `POSTTRAIN_OBSERVATORY_DISCOVER_TRACKIO_PROJECTS: "true"` and `POSTTRAIN_TRACKIO_DISCOVERY_INTERVAL_SECONDS: "300"`; retain the internal `POSTTRAIN_TRACKIO_SERVER_URL`. Keeping `POSTTRAIN_OBSERVATORY_SOURCE` is unnecessary in discovery-only deployment and should also be removed to make the absence of a fallback source explicit.

Update `ai-infra/ansible/playbooks/qualify-observatory.yml` in this order: prove unauthenticated readiness and refresh are HTTP 401; POST authenticated `/api/v1/sources/refresh` and require `state == "succeeded"`; read authenticated readiness; POST Trackio's `/api/get_all_projects` using `https://trackio.lan`, the internal CA, and `{"args": [], "kwargs": {}}`; compare sets rather than ordering; assert every Trackio project is present among Observatory source IDs and healthy source IDs, allow extra explicitly configured sources, require unique Observatory IDs, and require readiness's refresh status to show a successful timestamp and no error. Keep the image digest and revision assertions. Change qualification prose from “complete exact source set” to dynamic discovery.

Add `ai-infra/scripts/self_test_observatory_discovery.py`, a credential-free deployment contract test that renders the compose Jinja template with a dummy immutable image, parses the YAML using the Ansible environment's YAML support, and asserts the discovery variables and absence of the hardcoded list/legacy project variables. It should also parse or structurally inspect the qualification playbook to ensure both authenticated refresh and set-containment assertions remain present. Add a small launcher if consistent with existing self-test naming, and run it explicitly in validation; do not couple it to the destructive infrastructure preflight.

Finally, update `ai-infra/docs/plan/reproducible-unraid-ai-infrastructure.md` with the deployment decision and qualification result, as required by that repository's agent guide. Do not record credentials. Commit `posttrain` first, build/publish or otherwise select its immutable Observatory image following the existing deploy script, then commit `ai-infra` with the new source commit/image reference workflow. Never claim deployed success until the live qualification uses that exact image revision.

## Concrete Steps

Work in `/home/owayys/Projects/carbonteq-ai/posttrain` for product changes. Before editing, confirm the pulled baseline remains clean:

    git status --short --branch
    git rev-parse HEAD

Run the smallest backend tests during development:

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv sync --all-packages --locked --python 3.13
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run pytest packages/tracking-trackio/tests/test_adapter.py apps/observatory/tests/test_settings.py apps/observatory/tests/test_sources.py apps/observatory/tests/test_discovery.py apps/observatory/tests/test_http.py -q

Run frontend tests and regenerate/build the frontend from `apps/observatory/frontend`:

    npm test -- --run
    npm run build

Regenerate the checked-in OpenAPI contract from the repository root using the existing CLI:

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run posttrain-observatory schema --openapi apps/observatory/openapi.json

Then run the posttrain validation ladder:

    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run ruff check apps/observatory packages/tracking-trackio
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run pyright
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run lint-imports
    UV_CACHE_DIR=/tmp/posttrain-uv-cache uv run pytest
    git diff --check

Work in `/home/owayys/Projects/carbonteq-ai/ai-infra` for deployment changes. Run:

    uv run python scripts/self_test_observatory_discovery.py
    uv run ansible-playbook --syntax-check -i ansible/inventory/generated.yml ansible/playbooks/qualify-observatory.yml
    uv run ruff check scripts/self_test_observatory_discovery.py
    git diff --check

For a safe real-backend development run, do not send credentials to the browser and do not point browser code at Trackio. Start the backend on loopback with discovery enabled:

    POSTTRAIN_OBSERVATORY_DISCOVER_TRACKIO_PROJECTS=true \
    POSTTRAIN_TRACKIO_DISCOVERY_INTERVAL_SECONDS=300 \
    POSTTRAIN_TRACKIO_SERVER_URL=https://trackio.lan \
    POSTTRAIN_OBSERVATORY_HOST=127.0.0.1 \
    POSTTRAIN_OBSERVATORY_PORT=7861 \
    UV_CACHE_DIR=/tmp/posttrain-uv-cache \
    uv run posttrain-observatory serve

Open `http://127.0.0.1:7861/`, `http://127.0.0.1:7861/api/v1/sources`, and `http://127.0.0.1:7861/api/v1/sources/refresh`. Use the UI button or:

    curl -sS -X POST http://127.0.0.1:7861/api/v1/sources/refresh

The POST should report `state: succeeded`; the source list should contain all current Trackio projects using the project names as IDs. Use the in-app browser to click refresh, verify the UI remains functional, verify the network request targets Observatory, inspect console errors, and capture a screenshot if useful.

For deployed qualification, use the repository's existing immutable-image deployment workflow and then run from `/home/owayys/Projects/carbonteq-ai/ai-infra`:

    ./scripts/qualify-observatory

Expect the play recap to have zero failures and the report to state that authenticated dynamic Trackio discovery, source health, product contract, image digest, and source revision passed.

## Validation and Acceptance

The setting is backward compatible: with discovery unset or false, current single-source and JSON multi-source behavior and tests pass. With discovery true and no JSON sources, service creation accepts an initially empty registry and startup populates it from Trackio. An enabled configuration without `POSTTRAIN_TRACKIO_SERVER_URL`, an invalid boolean, or an interval below one second fails settings validation before serving.

Given a fake Trackio catalog returning `alpha` and `beta`, the first request after application startup sees source IDs `alpha` and `beta`. Given a later successful response returning only `beta`, `alpha` is absent and `beta` remains. Given a subsequent exception, `beta` remains, refresh status becomes failed, `last_success_at` is unchanged, and the error contains no traceback or server credentials. A configured source remains present throughout. Concurrent readers see only whole snapshots.

`POST /api/v1/sources/refresh` uses the same serialized operation as startup and periodic refresh and returns its safe status; `/health/ready` exposes the current status without triggering a refresh. In production, unauthenticated requests to both readiness and refresh receive 401 from Caddy, while authenticated refresh succeeds. There is never a frontend request to `trackio.lan` or `trackio:7860`.

The frontend button appears at the right end of the `Evidence backend` header inside the Backend popover, is exposed to assistive technology as `Refresh evidence backends`, keeps the popover open when activated, and visibly disables while refresh is in flight. After success, newly available canonical runs appear in the selectors without a page reload, and a still-existing selection remains selected. A backend refresh error appears beneath the popover header and does not erase the current run view.

The rendered deployment compose contains `POSTTRAIN_OBSERVATORY_DISCOVER_TRACKIO_PROJECTS=true`, interval `300`, and the internal Trackio URL, and contains no hardcoded project JSON or legacy single project. Live qualification compares the dynamic Trackio set as a subset of Observatory's unique healthy sources instead of asserting exactly four. With the current `trackio.lan` catalog, Observatory should expose at least the eight project IDs recorded in Surprises & Discoveries.

All focused tests, frontend tests/build, repository linters/type checks/import checks, full pytest suite, Ansible syntax check, deployment contract self-test, and `git diff --check` pass. The browser opens the loopback Observatory URL, refresh works, and there are no new console errors.

## Idempotence and Recovery

Every refresh is read-only against Trackio and replaces an in-memory discovered layer; it can be repeated safely. The periodic task performs no persistent writes. A failed refresh keeps the last successful map. Restarting Observatory reconstructs configured sources and discovers afresh; there is deliberately no on-disk cache, so a startup outage yields only configured sources until periodic or manual recovery.

A successful Trackio response that omits a project removes that discovery-owned source immediately. Recovery is to restore the project in Trackio and refresh again, or explicitly configure a source if it must survive catalog omission. An explicit source ID collision wins and can be removed only by changing configuration, not by discovery.

If the dev port is already occupied, stop only the process launched for this plan or select a different loopback port and update the browser URL. Do not kill unrelated services. If `uv` cannot write its default cache, retain `UV_CACHE_DIR=/tmp/posttrain-uv-cache`. If sandboxed Python cannot resolve `trackio.lan`, rerun the server with the required network approval rather than weakening TLS or changing application code.

Deployment rollback is one immutable Observatory image/configuration rollback through the existing `ai-infra` workflow. The previous static-list image remains usable until the new image passes qualification. Do not deploy the discovery-only compose against an old image that does not understand the discovery variables, because it would fall back to a manufactured single source.

## Artifacts and Notes

The verified Trackio project-catalog request and abbreviated response are:

    POST https://trackio.lan/api/get_all_projects
    {"args": [], "kwargs": {}}

    {"data":["ai-infra-qualification","ambient-agent","artifact-transport-qualification",
    "foundation-models","grpo-demo","infrastructure-executions","occupancy-research","sft-demo"]}

The intended safe status shape is:

    {
      "enabled": true,
      "state": "succeeded",
      "last_attempt_at": "2026-07-30T11:15:00Z",
      "last_success_at": "2026-07-30T11:15:00Z",
      "error": null,
      "discovered_source_ids": ["ai-infra-qualification", "ambient-agent"]
    }

The timestamps above are illustrative, and the returned IDs are abbreviated. The implementation must use timezone-aware UTC values serialized by Pydantic.

## Interfaces and Dependencies

In `packages/tracking-trackio/src/posttrain_tracking_trackio/adapter.py`, provide:

    class TrackioProjectCatalog:
        def __init__(self, server_url: str) -> None: ...
        def list_projects(self) -> tuple[str, ...]: ...

Export it from `posttrain_tracking_trackio.__init__`. It returns validated project names and raises on transport or contract failure.

In `apps/observatory/src/posttrain_observatory/sources.py`, provide:

    class RunSourceRegistry:
        def reconcile_discovered(
            self, sources: Mapping[str, RunDataSource]
        ) -> tuple[str, ...]: ...

The returned tuple is the sorted set actually installed after configured-ID collisions are removed.

In `apps/observatory/src/posttrain_observatory/models.py`, provide `SourceRefreshStatus` with the fields described above. In `discovery.py`, provide one coordinator whose public async surface is:

    async def refresh(self) -> SourceRefreshStatus: ...
    async def run_periodically(self) -> None: ...
    def status(self) -> SourceRefreshStatus: ...

In `ObservatoryService`, provide:

    async def start_source_discovery(self) -> None: ...
    async def stop_source_discovery(self) -> None: ...
    async def refresh_sources(self) -> SourceRefreshStatus: ...
    def source_refresh_status(self) -> SourceRefreshStatus: ...

The disabled implementation returns disabled status and does not raise merely because discovery is off. The refresh interface is `POST /api/v1/sources/refresh`, returning `SourceRefreshStatus` JSON; `/health/ready` includes the same current status without refreshing.

Use only existing runtime dependencies: Python `asyncio`, `threading`, `types.MappingProxyType`, Pydantic, FastAPI lifespan, the pinned CarbonTeq Trackio client, and existing React/Phosphor components. Do not add a database, cache, scheduler, browser Trackio client, or Trackio fork change.

Revision note (2026-07-30): Initial plan created after the user switched `posttrain` to pulled `main`; the plan records the new commit and revalidated current files and live Trackio contract. It was then corrected to use the pulled branch's Python 3.13 requirement and pinned Trackio environment after a stale pre-pull virtual environment was detected during live startup, and the current-state live preview result was recorded. The frontend decision was refined to place refresh inside the Backend popover's `Evidence backend` header rather than beside the closed selector. Implementation and local/live verification results were added; immutable deployment and user browser acceptance remain explicitly open.
