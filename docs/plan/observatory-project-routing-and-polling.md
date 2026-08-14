# Make Observatory project-first and route each run

This ExecPlan is a living document and must be maintained according to
`docs/templates/PLAN.md` and `.agents/PLAN.md`.

## Purpose / Big Picture

Opening Observatory must show the Trackio project catalog before it reads any
project's runs. Choosing one project loads only that project's bounded run list.
Choosing a run gives it a stable browser path that another user can open, and an
open run refreshes its currently visible evidence every sixty seconds. This
avoids the current failure mode where a busy project's recent activity chooses
the initial project and hides other projects from the navigation.

## Progress

- [x] (2026-08-13 13:15Z) Reproduced that the global `/api/v1/runs` default
  limit showed only `occupancy-engine`, even though dynamic Trackio discovery
  reported 22 projects and Ambient Agent GRPO runs existed.
- [x] (2026-08-13 13:15Z) Verified that `/api/v1/sources` already represents
  the discovery-owned Trackio project catalog without loading the global run
  feed; discovered Trackio source IDs equal their Trackio project names.
- [x] (2026-08-13 13:27Z) Add source-scoped run-list and direct run-summary
  HTTP contracts.
- [x] (2026-08-13 13:27Z) Replace the run-derived backend/project selectors
  with a fixed Trackio backend and a separate project picker with no
  preselected project.
- [x] (2026-08-13 13:27Z) Add URL routing, browser history handling, and the
  one-minute active-run refresh loop.
- [x] (2026-08-13 13:29Z) Validate Python API contracts, frontend behavior,
  production build, and a live local Ambient Agent project-scoped query.
- [x] (2026-08-13 14:05Z) Keep a selected empty Trackio project visible and
  render its distinct empty state instead of resetting the project picker.

## Surprises & Discoveries

- Observation: the global list endpoint honours its 100-item default after
  merging per-project results, so 100 recent runs from one busy Trackio project
  can remove every other project from the frontend selector.
  Evidence: local `GET /api/v1/runs` returned 100 `occupancy-engine` rows;
  the same endpoint with `limit=1000` returned 137 rows across Ambient Agent,
  Posttrain Lab, policy-prism, occupancy, and conformance projects.

- Observation: source discovery has a catalog identity independent of a run
  list. In the Trackio discovery composition, source IDs are exact Trackio
  project names and the source list may include an empty project.
  Evidence: the local readiness response contained 22 healthy discovered
  Trackio source IDs, including `ambient-agent`, before any project runs were
  selected.

- Observation: the in-app browser initially continued to call the older local
  Python server after the frontend had been rebuilt. That process ignored the
  new `source_id` query and made `ambient-agent` appear to select
  `occupancy-engine`.
  Evidence: after restarting the local loopback server, the identical
  `source_id=ambient-agent` request returned only `ambient-agent` runs.

- Observation: `ai-infra-qualification` exists in Trackio but has zero
  retained runs. The initial empty-state UI treated every missing selected run
  as if no project had been chosen.
  Evidence: `GET /api/v1/runs?source_id=ai-infra-qualification&limit=1000`
  returned an empty list while the route remained
  `/projects/ai-infra-qualification`.

## Decision Log

- Decision: use `/api/v1/sources` as the initial Trackio project catalog, and
  do not call `/api/v1/runs` until a user selects a healthy Trackio source.
  Rationale: this preserves empty projects, avoids a global evidence read, and
  uses the dynamic source discovery boundary already owned by Observatory.
  Date/Author: 2026-08-13 / Codex

- Decision: present Trackio as fixed backend context and call the discovery
  entries “Trackio projects” in the product, even though the provider-neutral
  registry currently stores each discovered project as a source ID.
  Rationale: a user chooses a Trackio project, not an evidence backend; this
  preserves the generic registry implementation without exposing its internal
  terminology in the UI.
  Date/Author: 2026-08-13 / Codex

- Decision: retain the picker value for an empty selected project and state
  “No retained runs in <project>”.
  Rationale: empty is valid project state and must not be indistinguishable
  from the initial no-project state.
  Date/Author: 2026-08-13 / Codex

- Decision: extend the run list with an optional `source_id` and make the
  registry resolve only that source before asking it for runs.
  Rationale: Trackio uses one datasource per project in dynamic discovery, so
  source scoping is the narrowest physical read and remains provider-neutral.
  Date/Author: 2026-08-13 / Codex

- Decision: make the browser route `/runs/<run-key>`, where `run-key` is the
  existing URL-safe `RunLocator` encoding of source ID and provider run ID.
  Rationale: the key already expresses the provider-neutral locator and avoids
  relying on a human-readable project name or globally unique run ID.
  Date/Author: 2026-08-13 / Codex

- Decision: poll only a routed run page, at sixty-second intervals, and refresh
  the run summary plus the evidence corresponding to the active tab.
  Rationale: project discovery and project lists remain user-driven, while an
  opened running job stays current without background work on the project
  picker or unrelated browser routes.
  Date/Author: 2026-08-13 / Codex

## Outcomes & Retrospective

Implemented and validated. The root path calls only the Trackio project
catalog; selecting a project makes one bounded, project-scoped run request;
opening a run uses its shareable `/runs/<run-key>` route. The local server is
running with dynamic Trackio discovery and the local CA for a real-source
check. The focused frontend suite includes root, direct-route, and polling
coverage; the full Observatory Python suite passes.

## Context and Orientation

`apps/observatory/src/posttrain_observatory/sources.py` owns the immutable
provider registry. `RunSourceRegistry.list_runs()` currently fans a query out
to every configured source, merges the result, and applies one global limit.
`apps/observatory/src/posttrain_observatory/http.py` exposes that behavior at
`GET /api/v1/runs`. `apps/observatory/frontend/src/App.tsx` currently calls it
at startup, derives both project and backend controls from those rows, and
selects the first row automatically.

`RunLocator` in `apps/observatory/src/posttrain_observatory/models.py` holds a
source ID and a provider run ID. Its `key` is a URL-safe, reversible opaque
identity. A direct summary route will resolve it against exactly one source;
it must not use the older `locate_run` scan, because a run ID can be ambiguous
between sources.

The existing dynamic Trackio discovery represents each Trackio project as one
source. The project picker should show only healthy Trackio sources. The
product has no selected run on `/`; it should render a neutral prompt telling
the user to choose a Trackio project. Other provider-specific source choices
remain an explicit future extension rather than a hidden first-project default.

## Plan of Work

First, add an optional `source_id` argument to `RunSourceRegistry.list_runs()`
and `_list_runs()` in `sources.py`. If supplied, validate and resolve that
source from the snapshot, execute only that datasource's `list_runs()` call,
and still wrap the result in `LocatedRunSummary`. Retain the existing all-source
behavior when omitted for exports and CLI consumers. Add a `get_run(locator)`
registry method that resolves exactly the locator's source and returns a
`LocatedRunSummary` from `source.get_run`.

Second, expose both behaviors through `http.py`: accept an optional
`source_id` on `GET /api/v1/runs`, and add `GET /api/v1/runs/{run_key}` before
the current nested run routes. The direct summary response uses the existing
`LocatedRunSummary` transport shape. Update `openapi.json` through the existing
schema command and add HTTP tests that prove a source-scoped list never calls
another source and a direct route resolves only the encoded source.

Third, change `frontend/src/lib/api.ts` so `sources()` reads `/api/v1/sources`,
`runs(sourceId)` sends `source_id`, and `run(runKey)` reads the direct summary
route. Regenerate its OpenAPI types. Remove the temporary `limit=1000` client
workaround because the UI will never use global rows for navigation.

Fourth, refactor `App.tsx` around a small browser-route helper. Parse only
`/` and `/runs/<run-key>`; unknown paths render the same selection surface plus
a visible safe not-found error. Start by loading source summaries, filter to
healthy Trackio entries, and keep `selectedSourceId`, `selectedProject`, and
`selected` unset on `/`. Selecting a project requests `api.runs(sourceId)`,
sets project/run shell state only from those returned rows, and pushes the first
selected run to `/runs/<run-key>`. Selecting another row pushes its own route.
A direct route loads `api.run(runKey)` first, then that run's project-scoped
list so the sidebar is populated. Handle `popstate` so browser back/forward
restores the route instead of silently retaining React-only selection state.

Fifth, replace the source selector with a fixed Trackio label and a project
selector whose choices come from source summaries, not run rows. Preserve the
refresh action there: after source refresh, re-fetch the project catalog; do
not automatically choose a newly discovered project. On a selected project,
refresh its current scoped list and retain the routed run when it still exists.
When no project is selected, render the normal Observatory shell with a
project-selection empty state instead of an error or a preloaded run.

Sixth, add a route-only polling effect. When the browser route contains the
currently selected run key, start one 60,000 ms interval and clean it up on
route/run change or component unmount. Each tick re-reads the direct run
summary and refreshes the selected evidence tab: overview/metrics use the view
endpoint, system metrics use its endpoint, and trace tabs reload their first
page plus aggregate evaluation when applicable. Guard every response with the
current route/run key so a late poll never overwrites a newer navigation.

## Concrete Steps

Run from `/home/hammad/projects/rl` during implementation:

    uv run pytest apps/observatory/tests/test_sources.py apps/observatory/tests/test_http.py -q
    uv run posttrain-observatory schema --openapi apps/observatory/openapi.json

Run from `/home/hammad/projects/rl/apps/observatory/frontend`:

    npm test -- --run src/App.test.tsx
    npm run build

Then from the repository root:

    uv run ruff check apps/observatory
    git diff --check

For live validation, start only the local loopback server with discovery and
the local CA, then open `http://127.0.0.1:7861/`. The first paint shows no
selected project; selecting `ambient-agent` loads only that project; selecting
a GRPO row changes the location to `/runs/<run-key>`; reloading that location
returns to the same run. Keep the run page open for at least one polling
interval and confirm the active evidence request repeats without a global
`/api/v1/runs` request.

## Validation and Acceptance

The source-scoped registry test must prove that a request for `ambient-agent`
does not call `posttrain-lab`. The HTTP test must prove the direct route returns
the encoded locator and a malformed or unknown locator gives the existing safe
404 or 422 response. Frontend tests must prove startup requests sources but not
runs, project selection requests one scoped run list, a direct URL rehydrates
the selected run, back/forward restores navigation, and fake timers produce one
refresh cycle only while a `/runs/<run-key>` route is active.

Human acceptance is: open `/`, select a Trackio project, open a run, copy its
URL into a new tab, and see the same run. The active run view refreshes after a
minute; the empty project picker does not issue run-list calls.

## Idempotence and Recovery

All new reads are GET-only. Restarting the local server performs discovery
again and returns to the empty `/` selection state. A source refresh failure
retains the existing source catalog by the existing discovery contract. If a
routed run is removed, show a safe not-found message and retain the project
picker; do not choose a different run automatically. Reverting the frontend
and HTTP changes restores the previous global-feed behavior without touching
Trackio data.

## Artifacts and Notes

The local diagnostic that motivated this work was:

    GET /api/v1/runs                 -> 100 rows, all occupancy-engine
    GET /api/v1/runs?limit=1000      -> 137 rows across nine projects
    GET /api/v1/sources              -> 22 discovered Trackio projects

## Interfaces and Dependencies

In `apps/observatory/src/posttrain_observatory/sources.py`, extend the public
registry with:

    async def list_runs(self, query: RunQuery, *, source_id: str | None = None) -> tuple[LocatedRunSummary, ...]: ...
    async def get_run(self, locator: RunLocator) -> LocatedRunSummary: ...

In `apps/observatory/src/posttrain_observatory/service.py`, mirror those
methods so HTTP remains a thin transport. In `http.py`, expose:

    GET /api/v1/runs?source_id=<Trackio-project>&limit=<bounded>
    GET /api/v1/runs/<run-key>

In `apps/observatory/frontend/src/lib/api.ts`, expose:

    sources(): Promise<SourceSummary[]>
    runs(sourceId: string): Promise<RunItem[]>
    run(runKey: string): Promise<RunItem>

`App.tsx` owns browser-route parsing and must not add a second router library
for two routes. The existing static-file fallback in `http.py` already serves
the frontend at a direct `/runs/<run-key>` path.

Revision note (2026-08-13): created after reproducing the global-feed project
selection bug in the live local Trackio-backed Observatory.
