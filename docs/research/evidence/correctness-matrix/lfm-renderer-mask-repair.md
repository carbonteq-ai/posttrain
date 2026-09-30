# LFM SFT target repair: headers are context, not sampled output

2026-09-30. The reproduced LFM SFT masking failure is corrected in published
source candidate `1aafe24595a7f2d2f31d24afb4b1bb7a6c6dd076` on the renderers
fork's `codex/lfm-sampled-mask` branch. Production still selects the previous
dev2 wheel. The canonical renderer-owned token provenance is preserved; this
is a correction to sampled masks, not a new training recipe or product meaning.

## What was wrong

`LFM25Renderer` inherited template message attribution without a sampled mask.
`build_training_sample` therefore fell back to all tokens attributed to a
trainable assistant message. That includes the template-injected
`<|im_start|>assistant` header and trailing separator. The inference prompt
already contains the header; teaching the model to produce it adds targets
outside its sampled completion.

System, user, and tool-result content were already excluded. The bug affects
renderer-built SFT labels; the direct native rollout action mask must be
qualified separately. It does not prove that this was the cause of the active
SAMPO run's outcome.

## Correction and edge cases

The LFM renderer now identifies each assistant span from its exact generation
prompt token prefix and marks output through the emitted turn stop. It leaves
headers, BOS, appended generation prompts, and post-stop separators unscored.
Literal role-marker text inside an answer remains trainable; this is a token
boundary repair rather than decoded-text deletion.

LFM family templates differ. The 1.2B Thinking tokenizer does not prefill
`<think>`; the model samples it. The 2.6B template prefills that marker on
generation prompts. The correction excludes the prefill when present and
supports curated plain answers without a reasoning block. Assistant-only SFT
examples retain their content and stop without training BOS/header. Inconsistent
prefix boundaries raise a clear error instead of silently marking scaffolding.
The initial implementation failed eight existing 2.6B cases; that failure was
resolved before publication and is not hidden by dependency skips.

The fix leaves serialized token IDs unchanged. It adds template prefix checks
to an already incremental renderer; long-history preprocessing overhead is an
open measurement. It changes neither parsing nor the native tool-history bridge.

## Matched token and target evidence

The before arm imports the exact dev2 wheel, SHA-256
`fdc65e9ed1a8a2f877c127a3456834d5ded996f32a4c70fe22bebe615073a87e`.
The after arm imports the isolated source candidate. Both use the same immutable
cached tokenizers and role-content fixtures. Every pair has identical input
token hashes; only LFM target masks change. Qwen is an unchanged control.

| Multi-turn selection | LFM trained tokens before | After | Qwen before / after |
|---|---:|---:|---:|
| Both assistant messages | 24 | 16 | 51 / 51 |
| Final assistant only | 12 | 8 | 13 / 13 |
| First assistant only | 12 | 8 | 38 / 38 |

All six after cases include the selected assistant content, exclude unselected
assistant content and system/user/tool sentinels, and exclude injected assistant
openers. Artifact files `renderer-mask-audit-baseline-current.json` and
`renderer-mask-audit-after.json` retain input/mask/source hashes.

## Actual three-update model experiment

The matched local LFM1.2B experiment uses the same immutable weights, seed 42,
BF16 base, FP32 rank-4/alpha-8 q/v adapters, LR 1e-4, deterministic attention,
gradient norm cap 1, and arithmetic preference fixture from the preceding
report. SFT and DPO reset adapter initialization and Adam state independently.
Input-token hashes match exactly across before/after arms.

| Measurement | Before | After |
|---|---:|---:|
| SFT scored tokens per update | 7 | 3 |
| SFT loss, update 1 | 8.11609 | 9.17900 |
| SFT loss, update 3 | 8.02190 | 9.03374 |
| Initial adapter gradient norm | 9.38787 | 11.55811 |
| Independent parameter-gradient relative error | 0 on all steps | 0 on all steps |

Every update has finite gradients and adapter movement near 1e-4. The initial
loss increases after correction because the scored population and its mean
denominator change. Comparing those averages as a quality improvement would
be invalid. Removing injected targets is independently justified by inference
boundaries; this three-step fixture does not establish task convergence.

The DPO control is exactly identical across all three update records, including
loss, gradients, margins, and parameter movement. The renderer's full IDs and
DPO preference completion boundaries are unchanged. Aggregate artifacts are
`lfm12-sft-dpo-renderer-baseline.json` and `lfm12-sft-dpo-renderer-fixed.json`.

## Verification and remaining gates

- All 20 new regressions fail on the isolated old wheel and pass on the candidate.
- The final renderer slice passes 59 cases, with 320 unrelated cases deselected.
- Posttrain rendering/SFT-validation integration passes 15 cases, including three new multi-turn mask regressions.
- Lint and whitespace checks pass with each repository's own configuration.

The fixture tokenizer revisions are
`95053d21d8e0b7ca99421a2127ae39c64f685ff3` for 1.2B Thinking and
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0` for 2.6B. The latter is tokenizer
validation, not a 2.6B model training run on the 8GB GPU.

The framework integration ran in the research Python with synchronous tests
and a minimal pytest config because that environment lacks the project's
asyncio plugin. The normal root venv skipped the renderer module due to absent
optional ML dependencies; that skip was not counted as integration evidence.

A new immutable wheel, consumer pins, native Verifiers multi-turn rollout
extraction, checkpoint/resume, broad task comparisons, and other algorithms
remain campaign gates. Existing jobs and production runtimes are unchanged.
