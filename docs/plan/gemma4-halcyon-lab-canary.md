# Prove Gemma 4 Halcyon GraphQL SFT in the lab

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

After this change, a developer with Hugging Face access can run one bounded, validation-aware LoRA update of `google/gemma-4-12B-it` against the pinned private Halcyon GraphQL corpus through `posttrain-lab`. The experiment proves the existing text-only TRL SFT path, tokenizer-owned Gemma tool-call rendering, assistant-only loss masking, W&B tracking, and artifact export without claiming portable framework support.

## Progress

- [x] (2026-07-28 06:46Z) Re-audited the pinned baseline and selected a lab-local programmatic scenario.
- [x] (2026-07-28 07:12Z) Added the Gemma protocol identifier and lab-owned model, renderer, target, training binding, and canary settings.
- [x] (2026-07-28 07:18Z) Added pinned train and validation data sources with strict corpus-shape validation.
- [x] (2026-07-28 07:25Z) Composed the validation-aware SFT scenario in the lab CLI.
- [x] (2026-07-28 07:41Z) Added synthetic CPU tests and credential-gated tokenizer, corpus, rendering, architecture, and PEFT tests.
- [x] (2026-07-28 08:22Z) Passed focused and lab test suites, Ruff, targeted Pyright, import-linter, and diff checks. Repository-wide Pyright exposes pre-existing compatibility-module errors, and the full pytest run stalled in an unrelated long-running portion after the lab suite had passed.
- [x] (2026-07-28 08:05Z) Completed the one-step GPU/W&B canary and a separate adapter reload/generation probe.

## Surprises & Discoveries

- Observation: Transformers 5.14.1 maps the pinned `Gemma4UnifiedConfig` through `AutoModelForCausalLM` to `Gemma4UnifiedForConditionalGeneration`.
  Evidence: both causal-LM and image-text auto mappings resolve to the same concrete class, so no loader dispatch is required.
- Observation: the existing default renderer successfully rendered all cached pinned examples at length 2048 with no supervised-token truncation.
  Evidence: train p50/p90/p99/max were 711/752/755/758; validation values were 702/751/752/752.
- Observation: a language-model-scoped rank-eight LoRA regex selects 32,784,384 trainable parameters and no multimodal tower parameter in a metadata-only model.
- Observation: cached, credential-gated tokenizer/rendering and metadata-only architecture/PEFT integration tests pass without loading the 12B weights.
  Evidence: two guarded tests passed with the pinned tokenizer/config cache; live private-corpus materialization could not run because the environment had no network-capable Hugging Face credential.
- Observation: the prescribed repository-wide Pyright invocation reports 93 errors in existing compatibility re-export modules and optional integrations, while targeted Pyright over every changed Python module reports zero errors.
  Evidence: `uv run pyright packages/common/src/posttrain/common/models.py apps/lab/src/posttrain_lab/gemma4_halcyon.py apps/lab/src/posttrain_lab/data/halcyon_graphql.py apps/lab/tests/test_gemma4_halcyon_sft.py apps/lab/tests/test_halcyon_graphql_data.py` completed with zero errors.
- Observation: the GPU canary completed with 28.49 GiB peak allocated VRAM, zero supervised-token truncation, and a validation loss change from 2.64068 to 2.48794.
  Evidence: W&B run `369fff7c-8434-4b5e-8979-7b675d90a598` finished with `posttrain/status=succeeded` and committed adapter, checkpoint, summary, and history artifacts.
- Observation: the exported adapter reloads over `Gemma4UnifiedForConditionalGeneration`, attaches 328 language-model-only LoRA modules, and performs finite text-only generation.
  Evidence: W&B run `0hyffszz` consumed the adapter artifact and recorded `reload/status=succeeded`.

## Decision Log

- Decision: Keep the renderer, model, target, settings, data sources, and work package under `apps/lab`; only extend the shared closed tool-protocol identifier literal.
  Rationale: this proves feasibility without adding global catalog or renderer registration.
  Date/Author: 2026-07-28 / Codex and user.
- Decision: Reuse `sft_definition(..., with_validation=True)`, the default renderer, `AutoModelForCausalLM`, the public dataset adapter, and existing W&B execution.
  Rationale: these primitives already implement the required behavior.
  Date/Author: 2026-07-28 / Codex and user.
- Decision: Defer OCI image construction, portable YAML work packages, the 98-step run, and adapter generation/reload.
  Rationale: the current milestone is a narrow code-level and one-step GPU feasibility test.
  Date/Author: 2026-07-28 / user.

## Outcomes & Retrospective

The lab-local canary is complete. CPU checks passed, the live private corpus materialized as 392 train and 31 validation examples, the one-step 96 GB GPU run finished without OOM or truncation, and W&B holds its adapter, tokenizer, checkpoint, summary, metrics, and reload lineage. The standalone reload probe generated beyond `<tool_call|>` because it did not use that delimiter as a stop token; this is a probe configuration issue and is corrected in the full-run acceptance procedure.

## Context and Orientation

`apps/lab/src/posttrain_lab/cli.py` composes project-owned selections into programmatic `WorkPackage` values and executes them through optional Trackio or W&B backends. Its existing SmolTalk scenario demonstrates validation-aware SFT. `packages/train/src/posttrain/train/backends/trl/sft.py` renders canonical `SupervisedExample` records, emits sequence/truncation metrics, runs TRL, and saves the model update and tokenizer. The private corpus is static supervised data; no GraphQL server participates.

The tokenizer's own chat template is authoritative. Gemma tool calls use `<|tool_call>call:<name>{...}<tool_call|>`. Enabling thinking changes template control tokens but does not fabricate a reasoning field when one is absent from the assistant message.

## Plan of Work

Extend `ToolCallProtocol.id` with `gemma4_native`. Add `posttrain_lab.gemma4_halcyon` with the exact pinned `RendererContract`, `ModelVariant`, 96 GB target, language-model-only LoRA binding, and one-step validated settings. Do not register these values in the global catalog.

Add `posttrain_lab.data.halcyon_graphql` with a frozen source selected by `train` or `test`. Load the pinned Hugging Face revision, normalize through `supervised_from_huggingface(format="messages")`, then reject incorrect row counts or protocol shape before training.

Add `gemma4-halcyon-graphql-sft-canary` to the lab CLI. Compose one job using the existing five-seat SFT definition factory and `_one_job_package`, then execute through the existing W&B-aware path.

Add always-on unit tests using generated synthetic rows. Add `HF_TOKEN`-guarded network tests for the pinned tokenizer, corpus, rendering/masking, architecture mapping, and PEFT scope. Do not commit private rows.

## Concrete Steps

From the repository root, run focused validation during implementation:

    uv run pytest -q apps/lab/tests/test_halcyon_graphql_data.py apps/lab/tests/test_gemma4_halcyon_sft.py
    uv run ruff check packages/common/src/posttrain/common/models.py apps/lab/src/posttrain_lab apps/lab/tests
    uv run pyright
    uv run lint-imports
    git diff --check

Then run `uv run pytest`. On a 96 GB GPU with `HF_TOKEN`, `WANDB_API_KEY`, and `WANDB_ENTITY`, run:

    uv sync --package posttrain-lab --extra gpu-train --locked --python 3.12
    uv run --package posttrain-lab posttrain-lab gemma4-halcyon-graphql-sft-canary \
      --project halcyon-graphql-sft --project-root . \
      --tracking-backend wandb --wandb-entity "$WANDB_ENTITY"

## Validation and Acceptance

CPU acceptance requires exact model/data revisions, five resolved seats, strict corpus-shape checks, native tokenizer delimiters, assistant-only labels, deterministic rendering, zero supervised truncation at 2048, the expected unified architecture mapping, and 32,784,384 trainable LoRA parameters confined to `.language_model.`.

GPU acceptance requires 392 train and 31 validation rows, finite initial/final validation and training losses, one completed optimizer step without OOM, zero supervised-token truncation, and W&B publication of parameters, length profiles, supervision ratio, step time, peak VRAM, summary, tokenizer, and adapter artifacts.

## Idempotence and Recovery

Tests and the canary are safe to rerun. Each run receives a new tracked identity. A failed training run must retain its evidence and error; fix the cause and start a new run rather than overwriting it. Secrets remain environment variables. No migration or destructive repository operation is required.

## Artifacts and Notes

Pinned inputs are Posttrain `15c1da7aae4fb297df3de4155175525a4d0734b6`, model revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, and dataset revision `a69e1c0c6ebb1f565be91cd0b6d95bd2b0e9110c`.

## Interfaces and Dependencies

The new public lab scenario is `gemma4-halcyon-graphql-sft-canary`. The new data interface is `HalcyonGraphQLSupervisedSource(split: Literal["train", "test"])`. No dependency versions change. The existing `posttrain-lab[gpu-train]`, Transformers 5.14.x, TRL fork, PEFT, datasets, renderers, and W&B adapter are reused.

Plan revision note: created for the lab-local Gemma 4 Halcyon canary after explicitly deferring image construction; updated after CPU implementation and validation to record environment-gated and repository-baseline checks accurately.
