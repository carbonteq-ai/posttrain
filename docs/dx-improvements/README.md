# Developer experience improvement reviews

This directory holds release-scoped developer experience critiques. Each
subdirectory is named after an immutable launched Posttrain version and records
what a project developer could actually do with that release, where the
experience breaks down, and the proposed direction for a later release.

These reviews are evidence and planning input. They do not replace the frozen
[post-training product baseline](../post-training/README.md), an accepted
architecture decision, or an implementation plan. If a proposal changes the
meaning of a canonical product contract, amend the baseline before implementing
it.

## Reviews

| Released version | Published | Review | Status |
| --- | --- | --- | --- |
| `v0.2.5` | 2026-07-31 | [Developer experience critique](./v0.2.5/README.md) | Current assessment |

## Versioning convention

- Create one directory named `v<major>.<minor>.<patch>` only after that version
  has been launched.
- Anchor the review to the release tag and commit, not to a later dirty working
  tree or an unreleased branch.
- Keep findings in the original release review even after they are fixed. A
  later launched release gets a new directory that records the new observed
  behavior.
- Append corrections to the review's revision history when evidence about that
  exact release changes; do not silently rewrite the assessment.
