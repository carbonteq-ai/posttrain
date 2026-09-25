# Make the renderer the single owner of thinking-token accounting for every catalog model

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md` and the fork rules in `docs/tooling/forks.md`.

## Purpose / Big Picture

Observatory's rollout tables show a "Thinking" and an "Output" token count for every rollout. For LFM2.5 training runs both columns show "—" ("Not recorded on loaded rollouts"), for example on run `lfm26-vortex-v5-100-from-r4-step20-20260925-r2`. The reason is architectural. The only component that knows exactly which generated tokens are thinking is the renderer: the small library that turns chat messages into model tokens and parses generated tokens back into a message with `reasoning_content` (thinking text), `content` (the answer) and tool calls. The renderer throws that knowledge away, so this repository re-derives thinking counts after the fact with a hard-coded Qwen 3.5 rule in two places. Every other model gets nothing, and output tokens are computed as completion minus thinking, so they disappear too.

After this plan, every model in the Posttrain catalog has a renderer that reports, for every generated reply, how many tokens were thinking. Verifiers copies that number into the per-reply usage record it already has (`usage.reasoning_tokens`), and the framework and Observatory read it without any model-specific code. A human can see it working by opening any new LFM2.5, Qwen 3.5, Gemma 4, K2-Horizon, Nanbeige 4.2 or Spark X2.5 rollout in Observatory and reading non-empty Thinking and Output columns whose sum equals the rollout's completion tokens.

The same change makes the renderer the one home for model-specific output knowledge. Today that knowledge is spread across the upstream renderers package, Verifiers' `renderer_extensions.py` (LFM2 and K2 tool and thinking parsers, LFM tool-result bridging) and this repository's `packages/train/src/posttrain/train/rendering.py`. Because every consumer (Verifiers' train client, TRL training, trace replay, historical re-scoring) goes through the renderer, a per-token thinking/answer/tool split there benefits all of them, including future reward shaping such as a thinking-length penalty.

## Progress

- [x] (2026-09-25) Traced the blank columns. Verifiers' train client (`../verifiers/verifiers/v1/clients/train.py`, the `Usage(prompt_tokens=..., completion_tokens=...)` construction near line 152) records no thinking count, with a comment that the subset is unknown. `packages/environment/src/posttrain/environment/verifiers_evidence.py` (`Qwen35ThinkingTokenRule`) and `apps/observatory/src/posttrain_observatory/traces.py` (`_trace_reasoning_tokens`) both re-derive thinking only when the model name contains "qwen3.5". Trackio stores what it is given (`fact_thinking_tokens`, `fact_model_output_tokens`).
- [x] (2026-09-25) Resolved upstream. Package `renderers` on PyPI is `https://github.com/PrimeIntellect-ai/renderers` (Apache-2.0). The installed `renderers==0.1.12.dev3` is byte-identical to upstream commit `06bcf635f4b582216fbb225eb4ed2e5c84fad7fe`. Upstream `main` was `20f2b38` on 2026-09-24. A read-only clone is at `/home/hammad/projects/renderers`.
- [x] (2026-09-25) Prototype 1, on the installed version: a renderer-agnostic prefix-parse boundary found LFM2.5's `</think>` exactly, but a reply cut off mid-thought parsed as no thinking at all (see Surprises).
- [x] (2026-09-25) Prototype 2, on upstream `main`: `DefaultRenderer` keeps a cut-off LFM2.5 thought (`reasoning_complete=False`), so the fork starts from upstream `main`.
- [x] (2026-09-25) Milestone 1: created the public GitHub fork `carbonteq-ai/renderers` (parent `PrimeIntellect-ai/renderers`). Local clone `/home/hammad/projects/renderers` has remotes `origin` (fork) and `upstream`. Branch `carbonteq/thinking-token-accounting` from upstream `20f2b38c`. The distribution is renamed `carbonteq-renderers`; the version comes from `renderers/_version.py` (`0.1.12.post1.dev1`); ledger `CARBONTEQ_FORK.md` added (commit `582d1ec`). Clean upstream baseline: 11100 passed, 124 skipped.
- [x] (2026-09-25) Milestone 2: `ReasoningBoundary.token_count` and `ParsedResponse.reasoning_tokens` for every scan-based parser; Gemma 4, DeepSeek V4 and Llama 3 special cases (commit `e9d65aa`). Full suite on that commit: 11247 passed, 99 skipped, 0 failed.
- [x] (2026-09-25) Milestone 3 (uncommitted in worktree `/home/hammad/projects/renderers-catalog`, branch `carbonteq/catalog-renderers`):
  - LFM2.5, K2-Horizon, Nanbeige 4.2 and Spark X2.5 renderers in `renderers/catalog_models.py`;
  - LFM2/K2 parsers moved into `renderers/catalog_parsers.py`;
  - Gemma 4 12B mapping and prefill set;
  - `generate()` returns `reasoning_tokens`;
  - `tests/test_catalog_model_renderers.py` (32 passed);
  - the catalog models added to upstream's `MODEL_CATALOG`.

  Full-suite run pending.
- [x] (2026-09-25) Milestone 4, Verifiers: worktree `/home/hammad/projects/worktrees/verifiers-thinking-tokens`, branch `codex/renderer-thinking-tokens`, commit `c7388dbe` on `b71ade0a`. `Usage.reasoning_tokens` is filled from the renderer; `renderer_extensions.py` and the LFM bridge fallback are removed; the dependency is now `carbonteq-renderers`; its ledger is updated. Tests against the local fork: 114 passed, 76 skipped (need `PRIME_API_KEY`). `uv.lock` not yet updated (needs the published fork).
- [x] (2026-09-25) Milestone 6, this repository: worktree `/home/hammad/projects/rl-thinking-tokens`, branch `codex/renderer-thinking-tokens`, commits `9e02fd77` and `e3a3e817` on `49c43a4e`.
  - The Qwen rules are removed from `verifiers_evidence.py` (calculator `verifiers-trace-facts.v5`) and from Observatory `traces.py`.
  - `create_renderer_config` maps every catalog family to the fork's renderer; the LFM bridge copies are removed.
  - `posttrain trace-facts backfill --renderer-model` re-scores historical traces.
  - Against the local fork and Verifiers branch: 1270 passed, 14 skipped (only the known vLLM loopback test deselected); ruff, pyright and import contracts clean.
- [ ] Milestone 5: publish.
  - Push the fork, then create release `carbonteq-v0.1.12.post1.dev1` with wheel and sdist hashes.
  - Merge `codex/renderers-fork-publisher` (`91c436d5`, cut from `origin/main`) into `main` so `publish-renderers-internal.yml` can be dispatched, then dispatch it.
  - Push the Verifiers branch and lock it against the published fork.
  - Replace `renderers==0.1.12.dev3` with `carbonteq-renderers==0.1.12.post1.dev1` in `packages/train/pyproject.toml`, move the Verifiers pin to the pushed commit, regenerate `uv.lock` and the runtime-image locks, rebuild images, and register the fork in `release/forks.toml`.
- [ ] Re-score the v5 runs with `posttrain trace-facts backfill <provider run> --renderer-model LiquidAI/LFM2.5-2.6B --apply` after Milestone 5. `lfm26-vortex-v5-100-from-r4-step20-20260925-r2` is Trackio run `f2fa4f777fa341e79aa2af3350e2a16a`.
- [ ] Milestone 7: live acceptance on a short LFM2.5 VORTEX run and a Qwen 3.5 eval.

## Surprises & Discoveries

- Observation: Upstream already moved partway in this direction. Commit #153 added `ParsedResponse.reasoning_complete`, #152 preserves unfinished reasoning across renderers and bridges, and #158 treats tool-call openers as implicit reasoning ends. No upstream issue or pull request asks for a token count or a token boundary; `gh search issues --repo PrimeIntellect-ai/renderers "reasoning tokens"` returned nothing on 2026-09-25.
  Evidence: `git -C /home/hammad/projects/renderers log --oneline renderers-v0.1.11..20f2b38`.
- Observation: On the installed 0.1.12.dev3, a renderer-agnostic count defined as "the shortest generated prefix from which the renderer recovers the full thinking text" is exact and monotonic for LFM2.5, but a reply cut off inside thinking parses as no thinking (count 0, which is wrong).
  Evidence:

      think+answer: total=27 boundary=17 last-thinking-token='</think>' monotonic=True 0.05ms
      truncated in thought: total=10 reasoning=None boundary=0

- Observation: Upstream `main` fixes that case for LFM2.5 through the generic renderer.
  Evidence:

      truncated in thought: reasoning='\nI should look up the sheet and then' complete=False content=''
      no thinking: reasoning=None complete=None content='Done.'

- Observation: Upstream has dedicated renderers for Qwen 3.5 and for Gemma 4 E2B, E4B, 26B-A4B and 31B. `google/gemma-4-12B-it` (in our catalog) is not in its model map, and LFM2.5, K2-Horizon, Nanbeige 4.2 and Spark X2.5 fall back to the generic `DefaultRenderer`.
  Evidence: `grep -n "gemma-4" /home/hammad/projects/renderers/renderers/base.py`.
- Observation: The eval path does not use our renderer for parsing. Verifiers' relay (eval) client calls vLLM's OpenAI-compatible chat endpoint, and vLLM reports `completion_tokens_details.reasoning_tokens` itself whenever the server runs with a reasoning parser (`vllm/entrypoints/openai/chat_completion/serving.py`, `_include_reasoning_tokens_details = bool(reasoning_parser)`). Verifiers' `Usage.from_openai` already keeps it.
- Observation: LFM2.5's `<think>` and `</think>` are single special tokens (64400 and 64401 in `LiquidAI/LFM2.5-1.2B-Thinking`), so an LFM renderer can find the boundary by token id rather than by decoding text.
- Observation: Model output knowledge is duplicated outside the renderer, and the fork removes it. The LFM tool-result bridge exists twice: `bridge_lfm25_tool_cycle` in `packages/train/src/posttrain/train/rendering.py` (TRL training) and `bridge_lfm2_tool_cycle` in `../verifiers/verifiers/v1/clients/renderer_extensions.py`. Both work around the LFM template adding a newline after `<|im_end|>` when it renders history. `create_renderer_config` in `rendering.py` picks parsers per model. `../verifiers/verifiers/v1/utils/score.py` cuts answers at the `</think>` string.
- Observation: The Verifiers context-budget prototype (`codex/context-budget-clamp`, `e9edbeb3`) estimates prompt tokens from UTF-8 byte length on the relay (eval) path, because that path does not render. A renderer there could count prompt tokens exactly. This is a separate follow-up, not part of this plan.

- Observation: The pinned Verifiers (`b71ade0a`) registers only the LFM2 parser. The K2 `k2-ifm` parsers exist only as uncommitted 9/18 edits in the main Verifiers checkout, yet `rendering.py` selected `k2-ifm` for K2-Horizon. So the K2 native-worker path could not have resolved its parser on the pinned stack. The fork now provides it.
- Observation: The generic renderer mishandles two catalog families on its own. Spark's generation prompt ends in `<|Bot|><think>`, not the Qwen header it assumes, so a prompt-opened thought was missed. K2's turn ends with `<|ifm|im_end|>` but its `eos_token` is `<|ifm|endoftext|>`, so the turn end leaked into content. Dedicated renderers declare both.
- Observation: `google/gemma-4-12B-it` ships the same template revision as 26B/31B, which pre-closes an empty thought when thinking is off. Upstream's Gemma renderer deliberately keeps that prefill on history for prefix stability. Upstream parity lists 26B/31B as exceptions for this; 12B joins them.
- Observation: A real VORTEX v5 LFM2.5 trace re-scored through the renderer ends every call's count exactly at `</think>`. 1682 of its 2304 output tokens were thinking.

## Decision Log

- Decision: Fork the renderers package instead of computing thinking counts in Verifiers or keeping per-model rules in this repository.
  Rationale: Every parser already computes the exact token where thinking ends and then discards it. Reporting it from the parser is exact, covers every renderer at once, and makes the renderer the single owner of model output formats. A Verifiers-only boundary search works for finished replies, but it is a derived definition and failed the cut-off case on the installed version. A per-model marker table would duplicate the renderer's knowledge. vLLM's own reasoning parsers are not on the token-in/token-out training path and may split differently from the renderer that defines training tokens.
  Date/Author: 2026-09-25, user.
- Decision: Base the fork on upstream `main` (`20f2b38` or newer at fork time), not on the installed `06bcf635`.
  Rationale: Upstream `main` contains #152, #153 and #158, which make cut-off thinking and tool-call-ended thinking parse correctly. Prototype 2 shows the LFM2.5 case working.
  Date/Author: 2026-09-25, Claude.
- Decision: Define `reasoning_tokens` as the number of generated completion tokens that come before the answer, counting the thinking markers the model generated, such as `</think>`. For a reply cut off inside thinking it is every generated token (and `reasoning_complete` is False). For a reply with no thinking it is 0. It is None only when a parser cannot tell.
  Rationale: With this definition, thinking plus output always equals completion tokens, so Observatory's "Output = completion − thinking" never goes negative or double-counts. Markers are generated tokens that cost context and time exactly like thinking text. A template-prefilled `<think>` is part of the prompt, not the completion, and is never counted.
  Date/Author: 2026-09-25, Claude.
- Decision: Publish the fork as the distribution `carbonteq-renderers` (import name stays `renderers`) through the same retained-asset route as `carbonteq-trackio`, with release tags `carbonteq-v<version>`.
  Rationale: A distinct distribution name prevents a public `renderers` release from replacing the fork during resolution, and the retained-asset publisher is the repository's approved fork publication path (`docs/tooling/forks.md`). A PEP 440 version must put `.post` before `.dev`, so fork releases are `<upstream base>.post<n>.dev<m>`, starting at `0.1.12.post1.dev1`. That sorts above every upstream `0.1.12.devN` and still satisfies Verifiers' `>=0.1.12.dev2`.
  Date/Author: 2026-09-25, Claude.
- Decision: Keep the chat template selection in this repository. The fork's LFM2.5 renderer accepts the tokenizer's chat template as its caller sets it.
  Rationale: `lfm25_26b_tool_chat.jinja` is a Posttrain renderer contract (`lfm2.5-tools-thinking@2`) selected per model variant. The renderer owns parsing and token accounting, not which template a project pins.
  Date/Author: 2026-09-25, Claude.
- Decision: Do not submit the `reasoning_tokens` change or the catalog renderers to upstream. The whole delta stays fork-only.
  Rationale: User decision. The fork ledger must therefore list every change and its conflict-sensitive files so that upstream rebases stay routine.
  Date/Author: 2026-09-25, user.

- Decision: Dedicated renderers for LFM2.5, K2-Horizon, Nanbeige and Spark subclass `DefaultRenderer` (template rendering) instead of hand-coding templates like upstream's typed renderers.
  Rationale: Rendering through each model's own template keeps parity by construction. The families differ only in parsing: header, markers, turn ends and tool format. Like `DefaultRenderer`, they reject an explicit `thinking_retention`, and only LFM bridges (tool results). Other turns re-render.
  Date/Author: 2026-09-25, Claude.
- Decision: The `default` training-renderer implementation now selects the fork's renderer for the model family, and the generic renderer only for families without one. Renderer ids such as `lfm2.5-native-v1` are unchanged, and `structured_output` is ignored because dedicated renderers always parse their own tool calls.
  Rationale: Token output is unchanged by design. The previous LFM behavior came from a fallback bridge and parser registration spread across Verifiers and this repository.
  Date/Author: 2026-09-25, Claude.
- Decision: Historical traces are re-scored by filling `usage.reasoning_tokens` before projection, through the existing `trace-facts backfill` command, not by keeping model rules in projection.
  Rationale: Projection stays model-independent. Trackio's `TraceFactUpdate` replaces a trace's facts by trace id, so re-scoring is idempotent.
  Date/Author: 2026-09-25, Claude.

## Outcomes & Retrospective

Code is complete and tested locally across the three repositories (see Progress). Nothing is pushed or published yet. The remaining work is publication and pinning (Milestone 5), re-scoring the v5 runs, and live acceptance.

## Context and Orientation

Four repositories take part. They are normally checked out as siblings of this one under `/home/hammad/projects`.

`renderers` (upstream `PrimeIntellect-ai/renderers`, to be forked as `carbonteq-ai/renderers`) turns chat messages into model tokens and parses generated tokens back. Its central types live in `renderers/base.py`. `ParsedResponse` is the parse result, with fields `content`, `reasoning_content`, `tool_calls` and, on upstream `main`, `reasoning_complete`. A renderer's `parse_response(token_ids, tools=...)` produces it. Model-specific renderers include `qwen35.py`, `gemma4.py` and `glm5.py`. `default.py` is the generic `DefaultRenderer`, which uses the tokenizer's chat template plus optional named tool and reasoning parsers from `parsers.py`. Shared parsing helpers are in `parsing.py`, including `scan_reasoning` (finds the thinking region) and `_reasoning_end_token_index` (finds a text marker's token boundary by binary search over decoded prefixes). The model-name-to-renderer map is `MODEL_RENDERER_MAP` in `base.py`.

`verifiers` (fork `carbonteq-ai/verifiers`, checked out at `/home/hammad/projects/verifiers`) runs agent episodes. Its train client, `verifiers/v1/clients/train.py`, sends raw tokens to vLLM's `/inference/v1/generate` endpoint through the renderers package's `generate` helper. It then builds a `Response` whose `usage` is a `verifiers/v1/types.py` `Usage` with an optional `reasoning_tokens` field that is currently always empty on this path. `verifiers/v1/clients/renderer_extensions.py` adds the `lfm2` and `k2-ifm` parsers and `bridge_lfm2_tool_cycle`. Verifiers declares `renderers[multimodal]>=0.1.12.dev2` in its `pyproject.toml`.

This repository selects models and renderers. `packages/common/src/posttrain/common/variants/*.py` defines each catalog model and its renderer contract. `packages/train/src/posttrain/train/rendering.py` (`create_renderer_config`) picks `Qwen35RendererConfig` for Qwen 3.5 and `DefaultRendererConfig` plus the `lfm2` or `k2-ifm` parsers otherwise. `packages/environment/src/posttrain/environment/verifiers_evidence.py` turns each native Verifiers trace into per-rollout values called trace facts, which are sent to Trackio. The `thinking_tokens` and `model_output_tokens` facts come from provider usage when present and otherwise from `Qwen35ThinkingTokenRule`. `apps/observatory/src/posttrain_observatory/traces.py` shows those values and has its own Qwen-only fallback, `_trace_reasoning_tokens`.

Catalog models and their renderers today:

- Qwen 3.5 0.8B and 2B: upstream `qwen35` renderer.
- Gemma 4 E2B, E4B and 31B: upstream `gemma4` renderer. 12B is missing from the map.
- LFM2.5 1.2B-Thinking and 2.6B: `DefaultRenderer` plus Verifiers' `lfm2` tool parser.
- K2-Horizon 7B: `DefaultRenderer` plus Verifiers' `k2-ifm` parsers.
- Nanbeige 4.2 3B and Spark X2.5 4B: `DefaultRenderer`, XML `<tool_call>` tool calls.

A "trace" is Verifiers' native record of one episode: messages, per-reply token ids with a sampled mask (true for tokens the model generated), and per-reply usage. Because traces keep token ids, a renderer can re-parse an old trace and recover its thinking count.

## Plan of Work

Milestone 1 creates the fork without behavior changes. With the user's approval (it creates a public GitHub repository under `carbonteq-ai`), create `carbonteq-ai/renderers` from upstream `main`, and set remotes `origin` to the fork and `upstream` to `PrimeIntellect-ai/renderers`. Rename the distribution in `pyproject.toml` to `carbonteq-renderers` and keep the import package `renderers`. Configure the version from `carbonteq-v*` tags. Add `CARBONTEQ_FORK.md` with every section `docs/tooling/forks.md` requires, and add `docs/tooling/renderers/README.md` here. At the end, the fork's own test suite and Verifiers' renderer tests pass against it unchanged.

Milestone 2 adds the accounting. In `renderers/base.py`, add `reasoning_tokens: int | None = None` to `ParsedResponse` with a docstring stating the definition from the Decision Log. In every parser in `renderers/parsing.py` and every renderer's `parse_response`, set it from the boundary the parser already computes:

- single-token closers (Qwen 3/3.5, GLM, LFM2.5 in Milestone 3): the index just past the close token;
- text markers: `_reasoning_end_token_index`;
- tool-call openers that end reasoning (#158): the opener's index;
- open reasoning (`reasoning_complete=False`): the full completion length;
- no reasoning: 0.

Tokens that are not sampled never count. Add regression tests under `tests/` for each parser.

Milestone 3 makes coverage explicit. Add `renderers/lfm25.py` (`LFM25Renderer`): token-id thinking boundaries (`<think>`, `</think>`), the pythonic tool-call parser and the tool-result bridge now in Verifiers' `renderer_extensions.py`. Add `renderers/k2_horizon.py` with the `k2-ifm` tool and reasoning parsers. Add renderers or parser registrations for Nanbeige 4.2 and Spark X2.5 after reading each tokenizer's chat template, and add `google/gemma-4-12B-it` to `MODEL_RENDERER_MAP`. Add `tests/test_thinking_accounting_conformance.py`, parametrized over every catalog model id. For each model, fixed replies (a thought then an answer, a thought cut off by the length limit, no thought, a thought then a tool call) must yield the expected `reasoning_tokens` and `reasoning_complete`, and `reasoning_tokens` must never exceed the completion length. Tokenizers load from the Hugging Face cache; the tests skip with a clear reason when a tokenizer is not cached and are marked `network`.

Milestone 4 changes Verifiers in its own fork, on a new branch. Depend on `carbonteq-renderers[multimodal]` instead of `renderers[multimodal]`. In `verifiers/v1/clients/train.py`, pass the parse result's `reasoning_tokens` into `Usage(..., reasoning_tokens=...)`, replacing the "unknown" comment. The renderers `generate` helper must return it alongside `reasoning_content`; if it does not, Milestone 2 adds it to `renderers/client.py`. Delete the LFM2 and K2 pieces from `renderer_extensions.py` and select the new renderers instead. Keep Verifiers' public configuration names working for one release by mapping them to the new renderers.

Milestone 5 publishes and pins. Follow `docs/tooling/forks.md` steps 6–13:

- in the fork: push, then create the immutable GitHub release `carbonteq-v0.1.12.post1.dev1` and record its wheel and sdist SHA-256;
- in this repository: add a retained-asset publisher workflow modelled on `.github/workflows/publish-trackio-internal.yml` and register the fork in `release/forks.toml`;
- then push the Verifiers change and update its commit pin in `packages/eval/pyproject.toml` and `packages/data/pyproject.toml`;
- replace `renderers==0.1.12.dev3` in `packages/train/pyproject.toml` with `carbonteq-renderers`;
- regenerate `uv.lock` and the runtime-image locks, and rebuild the runtime images.

Milestone 6 removes the model-specific code here. In `verifiers_evidence.py`, delete `Qwen35ThinkingTokenRule`, `ThinkingTokenRule` and `DEFAULT_THINKING_TOKEN_RULES`, so that thinking comes only from usage. In `traces.py`, delete `_trace_reasoning_tokens` and `_QWEN35_THINKING_END_TOKEN_ID`, and read thinking from `calls[].usage.completion_tokens_details.reasoning_tokens` or `calls[].usage.reasoning_tokens` (whichever Verifiers writes; confirm on a real trace). Add a re-scoring command, `posttrain run rescore-thinking RUN_ID`, in `apps/cli`. It loads a run's native traces, re-parses each reply's sampled tokens with the run's renderer, and re-emits the two trace facts. Before writing it, confirm with the Trackio fork maintainers whether re-emitted facts replace or duplicate existing ones.

Milestone 6 also removes the other model-specific code the fork now owns. Delete `bridge_lfm25_tool_cycle` from `packages/train/src/posttrain/train/rendering.py` and let TRL training use the LFM2.5 renderer's `bridge_to_next_turn`. Reduce `create_renderer_config` to choosing a renderer for the model variant, with no per-model parser selection. `../verifiers/verifiers/v1/utils/score.py` stays unchanged: it scores free text from any provider, for example API judge replies that never pass through a renderer, and its `</think>` split is a text-level guard, not token accounting. Verifiers' truncation reporting can also record `reasoning_complete`, so runs can tell a reply cut off mid-thought from one cut off mid-answer. That lets a later truncation penalty target runaway thinking.

Milestone 7 proves it end to end: a two-update LFM2.5 VORTEX run and a two-task Qwen 3.5 eval, both viewed in Observatory, plus re-scoring the v5 run.

## Concrete Steps

Prototype reproduction (safe to rerun; run from `/home/hammad/projects/rl`):

    HF_HUB_OFFLINE=1 PYTHONPATH=/home/hammad/projects/renderers uv run python - <<'EOF'
    from renderers import create_renderer, DefaultRendererConfig
    from renderers.base import load_tokenizer
    tok = load_tokenizer("LiquidAI/LFM2.5-1.2B-Thinking")
    r = create_renderer(tok, DefaultRendererConfig())
    p = r.parse_response(tok.encode("<think>\nI should look", add_special_tokens=False))
    print(p.reasoning_content, p.reasoning_complete)
    EOF

Expected: the partial thought and `False`.

Milestone commands are added here as each milestone runs, with the working directory for each.

Pin-bump inventory (verified 2026-09-25 on `49c43a4e`). `renderers==0.1.12.dev3` appears in:

- `packages/train/pyproject.toml` and `uv.lock`;
- the runtime-image locks `locks/{workspace,supervised,eval,online-rl-trl-py312,online-rl-verl-py313}.lock.txt` and profile `profiles/supervised.txt`;
- the veRL image's `verl-py313/release/{uv.lock,backend-constraints.txt}`.

All runtime-image paths are under `packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/`. The Verifiers commit `b71ade0a` is pinned in `packages/{data,eval}/pyproject.toml`, the profiles `{eval,online-rl-trl-py312,online-rl-verl-py313-control}.txt`, the matching `.lock.txt` files, and `verl-py313/{profile.toml,release/pyproject.toml}`. Regenerate these through the repository's runtime-image lock tooling, not by hand.

## Validation and Acceptance

The fork's conformance suite passes for every catalog model id and fails on the unmodified base, because `reasoning_tokens` does not exist there. Verifiers' tests pass, including a new test that a train-client response built from an LFM2.5 thought-then-answer completion carries `usage.reasoning_tokens` equal to the position just past `</think>`.

In this repository, the full ladder in `AGENTS.md` passes. `packages/environment/tests/test_verifiers_evidence.py` shows thinking coming from usage for a non-Qwen model. The Observatory tests show Thinking and Output for an LFM2.5 fixture trace whose usage carries `reasoning_tokens`.

Live acceptance: on the Milestone 7 LFM2.5 run, every rollout row in Observatory's rollout table shows numeric Thinking and Output, and Thinking + Output equals the rollout's completion tokens. Cut-off rollouts show Output 0 and Thinking equal to their completion tokens. After re-scoring, `lfm26-vortex-v5-100-from-r4-step20-20260925-r2` shows the same.

## Idempotence and Recovery

Creating the fork and its release are one-time, outward-facing steps that need the user's approval. A release tag is immutable; a bad release is superseded by the next `.devN`, never overwritten. Pin changes are ordinary commits and can be reverted together with `uv.lock`. Re-scoring is repeatable only if Trackio replaces facts by identity; until that is confirmed, run it once per run and record it in this plan.

## Artifacts and Notes

Upstream base candidates: installed `06bcf635f4b582216fbb225eb4ed2e5c84fad7fe` (0.1.12.dev3), and upstream `main` at `20f2b38` on 2026-09-24. Our remote: `carbonteq-ai/renderers` (to be created).

## Interfaces and Dependencies

In the fork's `renderers/base.py`:

    @dataclass
    class ParsedResponse:
        content: str
        reasoning_content: str | None = None
        tool_calls: list[ParsedToolCall] = field(default_factory=list)
        reasoning_complete: bool | None = True
        reasoning_tokens: int | None = None

In Verifiers, `Usage.reasoning_tokens` keeps its meaning: a subset of `completion_tokens`, not added to totals. The distribution name becomes `carbonteq-renderers`; the import name stays `renderers`.
