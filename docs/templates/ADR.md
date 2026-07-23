# Architecture Decision Records

This folder contains Architecture Decision Records, or ADRs. An ADR records a durable technical decision, the context that made the decision necessary, the options considered, and the consequences the team accepts by choosing it.

ADRs are revision-aware documents. Like ExecPlans in `.agents/PLAN.md`, they must stay useful to a future reader who has only the current working tree and the ADR file. If a decision changes, revise the ADR rather than relying on chat history or memory.

## When to Write or Update an ADR

Create or update an ADR when work introduces or changes a meaningful technical direction, including:

- data ownership, schema, persistence, or migration strategy;
- authentication, authorization, security, or tenancy boundaries;
- integration architecture for services such as Supabase, Stripe, Meta, or Vercel;
- framework, runtime, deployment, or testing strategy;
- cross-cutting conventions that future implementation should follow.

Not every plan needs an ADR. After implementing a plan, review its `Decision Log`, `Surprises & Discoveries`, and `Outcomes & Retrospective`. If the final result establishes a durable decision that future contributors need to understand, create a new ADR or update the relevant existing ADR.

## Naming

Use numbered, kebab-case files:

- `0001-use-custom-cookie-sessions.md`
- `0002-seed-preview-databases-with-supabase-branches.md`

Keep `ADR.md` as the process guide and template.

## Current Records

- `0001-use-nextjs16-react19-app-router-patterns.md` — Next.js 16 and React 19 App Router implementation conventions.
- `0002-use-drizzle-schema-with-supabase-postgrest-runtime.md` — **Superseded by 0008.** Original direction: Drizzle as schema/type source with Supabase PostgREST runtime data access.
- `0003-standardize-e2e-testing-on-gauge-and-playwright.md` — Gauge specs backed by Playwright for E2E coverage.
- `0004-centralize-route-security-in-nextjs-middleware.md` — centralized route protection, CSP, and rate limiting in Next.js middleware.
- `0005-use-custom-app-sessions-with-supabase-identity.md` — custom app sessions backed by Supabase identity records.
- `0006-standardize-frontend-client-state-and-ui-primitives.md` — SWR, context, frontend form, primitive, and token conventions.
- `0007-use-capacitor-remote-nextjs-shell-for-mobile.md` — Capacitor native shell loading a live Next.js origin plus PWA metadata.
- `0008-supabase-migrations-as-canonical-schema-source.md` — `supabase/migrations/` is the single canonical schema source; Drizzle removed.
- `0009-seed-non-production-databases-with-typescript-runner.md` — TypeScript seed runner under `scripts/seed/` for local, development, and preview targets, with hard guards against production.
- `0010-keep-organizer-events-list-uncached.md` — `/organizer/events` fetches fresh organizer-scoped event rows instead of using legacy Redis read-through caching.

## Required Sections

Each ADR must include these sections:

```md
# ADR 0000: Short Decision Title

Status: Proposed | Accepted | Superseded | Deprecated
Date: YYYY-MM-DD
Deciders: Name(s) or role(s)
Related Plan: `plans/example-plan.md` or `None`
Supersedes: `docs/technical/adr/0000-old-decision.md` or `None`
Superseded By: `None`

## Context

Explain the current state and the forces that make a decision necessary. Define terms that are specific to this repository. Name the files, services, tables, routes, commands, or workflows that matter.

## Decision

State the decision directly. A future implementer should be able to read this section and know what direction to follow.

## Consequences

Describe what improves, what tradeoffs are accepted, and what becomes harder. Include operational, security, data, testing, and product consequences when relevant.

## Alternatives Considered

List the serious alternatives and why they were not chosen. Keep this focused on options that were plausible at the time.

## Implementation Notes

Link to the code, migrations, docs, plans, or commands that implement or validate the decision. If the ADR is written before implementation is complete, state what must still happen.

## Revision History

- YYYY-MM-DD: Initial version. Reason: ...
```

## Revision Rules

When revising an ADR:

- update the `Status`, `Supersedes`, or `Superseded By` fields if the decision changed materially;
- append a dated note to `Revision History` explaining what changed and why;
- keep the current decision understandable without reading old chats;
- preserve useful historical context, but remove stale implementation instructions that would mislead a future contributor.
