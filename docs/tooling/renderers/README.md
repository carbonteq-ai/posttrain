# Renderers (CarbonTeq fork)

A renderer turns chat messages into model tokens and parses generated tokens
back into a message: thinking text (`reasoning_content`), the answer
(`content`) and tool calls. Verifiers' train client and Posttrain's TRL training
path use it for every model. Posttrain selects the CarbonTeq fork,
`carbonteq-ai/renderers`, published as the distribution `carbonteq-renderers`
with the unchanged import package `renderers`. The fork's own record is
`CARBONTEQ_FORK.md` at its repository root.

## Why the fork

The fork makes the renderer the single owner of model output accounting. Every
parse result carries `ParsedResponse.reasoning_tokens`: how many completion
tokens were thinking, counting generated thinking markers. A reply cut off
inside its thought counts entirely as thinking, and a reply without thinking
counts 0. Verifiers copies the count into each reply's `usage.reasoning_tokens`,
so Posttrain's trace evidence and Observatory need no model-specific rules. The
plan, including dedicated renderers for every catalog model, is
`docs/plan/renderers-fork-thinking-token-accounting.md`.

None of this delta is submitted to upstream (user decision, 2026-09-25).

## Selection

Posttrain 0.4.5 selects `carbonteq-renderers==0.1.12.post1.dev1` from
`carbonteq-dev` (`packages/train/pyproject.toml`, with its identity under
`[tool.posttrain.renderers]`). It is built from fork commit
`6aba28a8c9a597475addc2c123dd18b28a462766` (tag
`carbonteq-v0.1.12.post1.dev1`, based on upstream `20f2b38c`); wheel
`2e3231784729b9177bfc25eb06e3a6a958f9ba420ff8e241c010266cc9422d0b`, sdist
`bf529fc910f66494770a96e0ee5ec986344ed5f8b48258b50a55b82baa876fa3`. Verifiers
`0cee0a07` depends on it without an index pin, so public consumers (including
the Quality `external-consumer` job) install the GitHub Release wheel instead of
reaching `pypi.lan`. `release/forks.toml` and the stable-index fork check cover
it.

## Publication

The fork follows `docs/tooling/forks.md`. After the fork change is pushed, a
maintainer creates the immutable GitHub release `carbonteq-v<version>` with its
wheel and sdist and records their SHA-256. The maintainer then dispatches
`.github/workflows/publish-renderers-internal.yml` with the tag and both hashes.
That workflow publishes those exact bytes to `https://pypi.lan/carbonteq/dev/`,
proves the stored files match, and installs them cleanly. Promotion to
`carbonteq/stable` uses `.github/workflows/promote-retained-fork-candidate.yml`
after qualification; its first promotion needs the workflow fix in Posttrain
0.4.5 that treats a missing stable page as a first promotion. A workflow can only
be dispatched once it exists on the repository's default branch.

## Qualification evidence and remaining gates

None recorded yet. Selection requires the fork's conformance suite for every
catalog model, Verifiers' tests on the fork, and a live LFM2.5 rollout whose
Observatory rows show Thinking + Output equal to completion tokens.
