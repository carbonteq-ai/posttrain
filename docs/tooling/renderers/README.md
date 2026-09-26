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

Posttrain selects `carbonteq-renderers==0.1.12.post1.dev2` from `carbonteq-dev`
(`packages/train/pyproject.toml`, with its identity under
`[tool.posttrain.renderers]`). It is built from fork commit
`6f712616fa88073919827a695af4f318b8c2d24e` (tag
`carbonteq-v0.1.12.post1.dev2`, based on upstream `20f2b38c`); wheel
`fdc65e9ed1a8a2f877c127a3456834d5ded996f32a4c70fe22bebe615073a87e`, sdist
`2c83d94bd1fd81f5fdfe95cd8390ac1d5057355bc278bf14e642ab3f8b2c26bd`. Over dev1
(Posttrain 0.4.5-0.4.8) it adds two parsing fixes that stop training from
dropping tool calls serving accepts: a tool-call opener ends an unclosed thought,
and LFM2.5 pythonic calls get the vLLM fork's `lfm2` parser repairs (nested
quotes, raw control characters, zero-padded integers, keyword-named parameters).
Verifiers `cdd2ec76` depends on `carbonteq-renderers>=0.1.12.post1.dev1` without
an index pin, so public consumers (including the Quality `external-consumer`
job) install the GitHub Release wheel instead of reaching `pypi.lan`. `release/forks.toml` and the stable-index fork check cover
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
