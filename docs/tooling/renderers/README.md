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

Posttrain selects `carbonteq-renderers==0.1.12.post1.dev3` from `carbonteq-dev`
(`packages/train/pyproject.toml`, with its identity under
`[tool.posttrain.renderers]`). It is built from fork commit
`7fe5d06b9840ef0e4c7d43419cbd7f9afc1b727a` (tag `carbonteq-v0.1.12.post1.dev3`,
branch `codex/lfm-sampled-mask`); wheel
`57dcc6f8ba2db7bec21c75d40177059f704bb817a772558435e4d34c262ab5eb`, sdist
`650954a172b93524e871f873fad0d5448bc9e578bd8da2fe3e46d48b9307aa39`, published to
`carbonteq/dev` by
[run 36996039659](https://github.com/carbonteq-ai/posttrain/actions/runs/36996039659).
Over dev2 it fixes LFM SFT targets: the template renderer previously trained
injected assistant headers and separators (`1aafe24`); targets now cover actual
assistant output through the turn stop, handling 2.6B's prefilled `<think>`
versus 1.2B Thinking's sampled marker. `7fe5d06` masks assistant turns that the
template rewrites when a later user message follows (2.6B drops their
reasoning) in that rewritten form; without it the 2.6B reasoning-history
offsetless case raised. The fork's full suite passes (11,834 cases) and
Posttrain's `test_rendering.py` LFM header/tool mask regressions now pass.
Real LFM three-step SFT and DPO math checks pass on `1aafe24`; this does not
establish task quality. The veRL kind's release lock selects dev3 too; its
published image keeps dev2 until rebuilt.
Evidence: [LFM mask repair](../../research/evidence/correctness-matrix/lfm-renderer-mask-repair.md).

Previously selected: `carbonteq-renderers==0.1.12.post1.dev2` from `carbonteq-dev`
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

`0.1.12.post1.dev2` ships in Posttrain 0.4.9 and was promoted byte-for-byte to
`carbonteq/stable` (Posttrain run 36263033868).

- Fork suite: 11,812 passed, 107 skipped at `6f71261`; the six LFM2.5 tool-call
  repair tests fail without the change.
- Parser parity: on the calls LFM2.5-1.2B sampled in AutomationBench training,
  the fork and the vLLM fork's `lfm2` parser give identical results, including
  the ambiguous cases both reject.
- Live rollout: `lfm12-sampo-8gb-20260926-r15` (0.4.9 job-kind images, which
  install dev2) parsed 3 calls that are not valid Python and would have been
  dropped by dev1; the 3 it rejected repeat a keyword argument or were cut off.

Remaining gate: an Observatory check that Thinking + Output equals completion
tokens on a live LFM2.5 rollout has not been recorded for dev2.
