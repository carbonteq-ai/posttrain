# Pull Request Creation Guide

This folder contains reusable writing guides for durable engineering artifacts. This file defines how pull requests should be written for this repository so teammates can quickly understand the change without reconstructing it from commit history or file diffs.

PR bodies are not implementation scratch notes. They are teammate-facing summaries for reviewers scanning GitHub or Slack. The body should explain what changed from a product and behavior perspective, call out important fixes, and record the validation that actually happened.

## When to Use This Guide

Use this guide whenever you create or substantially update a pull request.

This guide applies to:

- feature work
- refactors
- bug fixes
- follow-up cleanup after a larger migration
- PR updates where the scope changed after the original body was written

If the PR introduces a durable architectural decision, update the relevant ADR as well. If the PR executes a larger staged rollout, keep the linked plan current too.

## Default PR Structure

Every PR body should use this structure unless the repository or team explicitly asks for something narrower:

```md
## Summary
Short orientation paragraph.

## Product-facing changes
- Surface or workflow level change.
- Surface or workflow level change.

## Important fixes and behavior corrections
- Important fix or behavior correction.
- Important fix or behavior correction.

## Developer notes
- Shared contract or component change that matters to future implementation work.
- Migration, API, or shared-layer note that reviewers should know.

## Validation
- Command or check that actually ran.
- Command or check that actually ran.
```

Optional sections may be added when relevant:

- `## Developer notes`
- `## Risks and follow-ups`
- `## Infra / analytics / rollout notes`
- `## Screenshots`

Do not add optional sections just to make the PR look fuller.

## Writing Rules

### 1. Start with the outcome, not the implementation

The opening `Summary` should explain what area changed and why it matters. A reviewer should understand the purpose of the PR before reading any bullets.

Good:

- “This refactor moves User Settings and Institution Settings onto the same shared settings shell and aligns the sidebar/search behavior across both surfaces.”

Bad:

- “This PR updates several files related to settings.”

### 2. Group by product surface or workflow

Describe changes the way a reviewer experiences the product:

- Account Settings
- Institution Settings
- Library search
- Organizer public event page
- Checkout flow

Do not organize the PR body as a file list.

### 2a. Keep shared API notes out of product-facing bullets

`Product-facing changes` should describe user-visible outcomes:

- screens that were improved
- workflows that changed
- behavior that is now more consistent

Do not put shared implementation details here, such as:

- “shared select now owns option rendering”
- “autocomplete now supports renderOptionContent”
- “button theme weight was normalized”

Those belong in `Developer notes` or `Important fixes and behavior corrections`, depending on whether the point is mainly for future implementers or for reviewers evaluating behavior risk.

### 3. Call out behavior corrections explicitly

If the PR fixes drift, inconsistencies, or broken behavior, say that directly in `Important fixes and behavior corrections`.

Examples:

- removed inconsistent selected-state styling between two settings rails
- fixed a modal closing before the create mutation could render an error
- corrected stale cache behavior on organizer events

This section is especially important for refactors, because reviewers need to know whether the change is only structural or also behavioral.

### 4. Only claim validation that actually happened

The `Validation` section must only include checks that were actually run or observed.

Good:

- `yarn tsc --noEmit --pretty false`
- “Verified the updated screen in the browser”

Bad:

- “Tested locally” when nothing specific was run
- “All tests pass” unless that exact test set was run

### 5. Keep it teammate-facing

Write for a teammate scanning quickly, not only for the original implementer. Prefer short sentences, concrete nouns, and product language over repo-internal shorthand.

Avoid:

- long diff narration
- file-by-file bullet dumps
- vague statements like “cleanup” without telling the reviewer what got cleaner

### 6. Use `Developer notes` when shared contracts changed

Add `Developer notes` when the PR changes shared layers that future engineers need to know about, for example:

- shared form primitives
- shared autocomplete or select contracts
- theme-level button behavior
- reusable modal or layout conventions

This section should stay concise. It is not a changelog. Its job is to point reviewers and future contributors at the important shared-layer deltas without polluting `Product-facing changes`.

## Default Expectations

Unless the user or repository asks otherwise, a good PR body should:

- summarize product-facing changes by surface or workflow
- call out important fixes and behavior corrections
- separate shared API/component changes into `Developer notes` when relevant
- mention infra, analytics, or deployment effects when relevant
- record actual validation
- stay concise enough to scan in one pass

## Example PR Body

```md
## Summary
This refactor brings the settings experience onto a shared shell and nav/search contract so User Settings and Institution Settings behave like one coherent product area instead of two parallel implementations.

## Product-facing changes
- User Settings and Institution Settings now use the same split settings shell with a shared left rail structure.
- Settings sidebars now share the same selected, hover, and destructive-action treatment.
- Institution Settings search now uses the shared compact search field with the embedded AI toggle action.

## Important fixes and behavior corrections
- Removed the old left-accent selected-state treatment from the newer settings nav path.
- Tightened the nested General sidebar so it no longer wastes horizontal space.
- Polished the main sidebar identity/header treatment so the institution and user context read as one block.

## Developer notes
- Shared settings surfaces now rely on the same shell and nav/search contract, so follow-up settings work should build on the shared primitives instead of reviving screen-local wrappers.
- The PR also updates the shared settings/search lane, so future sidebar or search changes should be made in the common layer first.

## Validation
- yarn tsc --noEmit --pretty false
```

## PR Update Rule

When a pull request evolves meaningfully after the first draft, update the PR body so it still describes the current scope. Do not leave an outdated summary in place after the implementation direction changes.

Examples that require a PR body update:

- the PR started as styling only but now includes behavior fixes
- shared infrastructure was introduced during implementation
- validation changed
- a new product surface was added to the scope

## Checklist For Authors

Before posting or updating a PR, confirm:

- the `Summary` explains the outcome clearly
- product-facing changes are grouped by surface or workflow
- product-facing changes do not contain purely shared API/internal implementation bullets
- important fixes are called out explicitly
- shared contract changes are in `Developer notes` when relevant
- validation is accurate and real
- the body is not just a file list or diff summary
