# Qualify the Gemma 4 31B dense checkpoint

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`. It is self-contained so another contributor can resume the work from the repository without chat history.

## Purpose / Big Picture

After this change, posttrain can identify and qualify the exact instruction-tuned `google/gemma-4-31B-it` checkpoint through the Gemma 4 family support already proven by the 12B Unified model. A user can resolve `models/gemma4-31b-it@bf16`, run a bounded text-only vLLM generation smoke, and perform two LoRA supervised fine-tuning optimizer updates on the Lab 96 GiB target. The successful training run must retain a PEFT adapter that can be loaded in a clean process and, if supported by the pinned vLLM runtime, served for a generation smoke.

This is checkpoint qualification, not a new architecture API. The upstream checkpoint is the tower-dense `gemma4` topology, while the existing 12B checkpoint is `gemma4_unified`; both resolve through `AutoModelForMultimodalLM` and use the same Gemma 4 renderer contract. The concrete topology remains immutable checkpoint provenance. No caller-authored `architecture` field, new renderer, new training loader, dependency update, multimodal training path, MoE support, quantization path, or full-context claim is introduced.

The observable proof mirrors the completed 12B qualification. The serving work package must load the pinned foundation weights and return nonempty generated text. The SFT work package must complete two finite optimizer updates, retain an adapter and training summary, report a positive trainable subset, and survive clean adapter reload. A real tool-bearing request must produce a structured `tool_calls` response before the checkpoint is described as fully qualified for the existing Gemma tool protocol.

## Progress

- [x] (2026-08-05 08:12Z) Read the repository instructions, complete plan template, canonical workflow, work/evidence, framework, and API documents relevant to model qualification.
- [x] (2026-08-05 08:12Z) Inspected the completed 12B implementation, catalog bindings, work packages, release gates, tests, and retained qualification plan.
- [x] (2026-08-05 08:12Z) Resolved the exact 31B Hub revision and model facts; proved tokenizer vocabulary equivalence, family loader compatibility, dense topology, and the existing LoRA target expression against the pinned config.
- [x] (2026-08-05 08:20Z) Added the exact 31B model variant, base-catalog row, exports, and identity/template tests.
- [x] (2026-08-05 08:20Z) Added checkpoint-specific Lab training, serving, and evaluation bindings plus serving and SFT work packages and candidate gates.
- [x] (2026-08-05 08:20Z) Ran focused tests, tokenizer-aware Gemma tests, both static work-package validations, locked sync, full Ruff, import boundaries, targeted Pyright, the full repository test suite, and diff checks.
- [ ] Submit and track the serving and SFT work packages on `targets/carbonteq-rtx-pro-6000-96gb`.
- [ ] Prove clean PEFT reload, structured tool calling, and adapter serving where supported; record immutable evidence and finish the retrospective.

## Surprises & Discoveries

- Observation: The 31B checkpoint uses a different upstream topology from 12B without requiring a different posttrain loader or renderer.
  Evidence: `google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475` declares `model_type=gemma4`, `Gemma4ForConditionalGeneration`, and `AutoModelForMultimodalLM`; the 12B checkpoint declares `gemma4_unified` and `Gemma4UnifiedForConditionalGeneration` under the same auto-model factory. Both pinned tokenizers expose equal 262,144-entry vocabularies and equal special-token ids.

- Observation: The existing dense Gemma LoRA target expression matches the 31B tower language model without selecting a modality tower.
  Evidence: constructing the exact 31B config under empty weights selected 410 modules under `model.language_model.layers` with `^model[.]language_model[.]layers[.]\d+[.](self_attn[.](q_proj|k_proj|v_proj|o_proj)|mlp[.](gate_proj|up_proj|down_proj))$`; no selected name was outside the language model. PEFT rank 8 creates 61,214,720 trainable parameters against the config-instantiated model.

- Observation: The exact stored checkpoint parameter count differs from a fresh config-instantiated parameter count.
  Evidence: the official Hub safetensors metadata reports 31,273,088,876 BF16 parameters, while a Transformers 5.14.1 empty-weight construction reports 32,682,372,656 parameters before PEFT. The `ModelVariant.parameters` field records the exact stored checkpoint count, matching the precedent used by 12B, while runtime evidence will record the trainer's actual loaded total.

- Observation: The existing tokenizer fingerprint can be reused for 31B.
  Evidence: the pinned 12B and 31B tokenizers have identical vocabularies and special-token ids, which are the canonical inputs documented for the existing Gemma fingerprint `059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416`. Their canonical Gemma 4 chat templates are also identical at the pinned revisions.

- Observation: No reusable Python implementation changed beyond adding the exact model value.
  Evidence: both work packages pass static composition, the 31B binding resolves through the existing `gemma4` renderer and family loader, and the complete repository suite reports 1,043 passed and 20 skipped. Ruff passes, all eight import contracts remain intact, and targeted Pyright reports zero diagnostics.

## Decision Log

- Decision: Keep `family="gemma4"` and reuse `gemma4-tools@1`, `gemma4-off-v1`, and family-level `AutoModelForMultimodalLM` dispatch.
  Rationale: All compatibility surfaces are family-level and the immutable checkpoint resolves its tower-dense class. Adding an external topology axis would duplicate upstream metadata without improving static validation.
  Date/Author: 2026-08-05 / Codex

- Decision: Add a distinct exact model selection and checkpoint-named Lab bindings instead of broadening the 12B entries.
  Rationale: Model identity, resource settings, job-package identity, and qualification evidence are checkpoint-specific even when framework behavior is shared.
  Date/Author: 2026-08-05 / Codex

- Decision: Reuse the proven dense language-model LoRA regular expression, subject to real-run parameter and memory evidence.
  Rationale: Empty-weight module discovery proves the expression selects all present 31B dense attention and MLP projections and excludes multimodal towers. A separate expression would encode no real divergence.
  Date/Author: 2026-08-05 / Codex

- Decision: Begin with the same bounded 8,192-token serving shape and two-step, 512-token SFT shape as 12B, but use a lower initial vLLM GPU-memory utilization if required by startup evidence.
  Rationale: Qualification should isolate checkpoint size while preserving workload meaning. The 31B BF16 weights consume roughly 62.5 GB before runtime allocations, so the 96 GiB target must retain more headroom than the 12B model.
  Date/Author: 2026-08-05 / Codex

## Outcomes & Retrospective

Implementation and real qualification are pending. Completion requires successful foundation serving, two-step SFT, clean adapter reload, structured tool-call parsing, and an honest adapter-serving result or explicitly recorded unsupported boundary.

## Context and Orientation

`packages/common/src/posttrain/common/variants/gemma4.py` owns the Gemma renderer contract and exact Python model variants. `packages/common/src/posttrain/common/variants/__init__.py` exports variants and builds `FOUNDATION_VARIANTS`. `packages/catalog/src/posttrain/catalog/base/models.yaml` is the reusable base catalog; it must contain exact model facts but no Lab target or qualification policy.

The family-aware training loader is already implemented in `packages/train/src/posttrain/train/backends/trl/common.py`: every model whose `family` is `gemma4` loads through `AutoModelForMultimodalLM`. `packages/train/src/posttrain/train/profiles.py` already exposes `gemma4-off-v1` through the generic default renderer. No reusable train or serve code change is expected.

`apps/lab/.posttrain/catalog/gemma4-unified-qualification.yaml` currently owns 12B-specific settings and bindings. Rename this overlay to a family-level filename only if doing so is a clean tracked rename with all references updated; otherwise add the 31B entries to it and record that the legacy filename is merely a catalog shard. `apps/lab/.posttrain/work_packages/` contains declarative work packages. `apps/lab/src/posttrain_lab/qualification/gates.toml` inventories candidate release gates. The target `targets/carbonteq-rtx-pro-6000-96gb` is a single NVIDIA RTX PRO 6000 Blackwell Workstation Edition with approximately 96 GiB VRAM, submitted through dstack.

The exact checkpoint is `google/gemma-4-31B-it` at commit `842da3794eaa0b77d5f08bae87a17459d91ff475`. It is instruction-tuned, BF16, dense, and stores 31,273,088,876 parameters. Its upstream model type is `gemma4`, architecture is `Gemma4ForConditionalGeneration`, native context is 262,144 tokens, and it supports text and image-family visual inputs but no audio tower. The executable bindings in this plan are text-only. The model's separate speculative assistant checkpoint is not part of this variant, so `mtp` remains false.

## Plan of Work

First add `GEMMA_4_31B_IT` beside the 12B constant. Reuse the renderer object and tokenizer fingerprint, but pin the distinct repo, commit, stored parameter count, modalities, context, and upstream provenance. Export it and add it to `FOUNDATION_VARIANTS`. Mirror the same value in `packages/catalog/src/posttrain/catalog/base/models.yaml`. Extend common tests so Python and YAML identities agree, registry membership is exact, the shared renderer is explicit, and the locally cached 31B tokenizer produces the same off/thinking/tool boundaries. The cache-dependent test must skip clearly when the exact tokenizer is absent.

Then add checkpoint-specific Lab entries. Define `gemma4-31b-it/sft-qualification-v1` with two steps and length 512. Define `training/gemma4-31b-it-trl-lora-qualification@1` using the existing renderer, rank, alpha, exact dense target expression, target, TRL fork revision, and dependency lock digest. Define screen and evaluation vLLM bindings for the 31B selection. Start the serving screen at 8,192 maximum model length, one sequence, 4,096 batched tokens, eager mode, chunked prefill, BF16, text-only modality limits, skipped multimodal profiling, and Gemma tool/reasoning parsers. Use a conservative memory-utilization value that leaves enough space for the loaded BF16 checkpoint; amend the plan with observed startup evidence if it changes.

Create `gemma4_31b_serve_smoke_qualification.yaml` and `gemma4_31b_sft_qualification.yaml` by following the 12B work-package shapes while using 31B identities and descriptions. Add candidate serving and SFT gates under a `gemma4-31b-dense-support` experiment family. Extend Lab tests to resolve every selection, validate target and renderer compatibility, and load both work packages without importing GPU dependencies.

Run focused validation and then the normal repository ladder. Commit logical slices separately: exact model contract, Lab qualification composition, and plan/evidence updates. Do not change dependency pins unless runtime proof demonstrates a blocker and this plan is amended first.

Finally submit serving and SFT through the normal dstack execution path. Track each canonical run through provider completion and retained-evidence reconciliation. Inspect generated output, metrics, artifacts, image digest, GPU identity, and peak memory. Materialize the exact retained adapter into a clean process and prove finite logits or generation. Send a real tool-bearing OpenAI-compatible request to foundation serving and preserve the structured `tool_calls` result. Attempt adapter serving with the pinned vLLM stack; if it is unsupported, record that boundary rather than weakening acceptance silently.

## Concrete Steps

Use `/home/owayys/Projects/carbonteq-ai/posttrain` as the working directory.

Run focused model and catalog tests after the first slice:

    uv run pytest packages/common/tests/test_model_variants.py packages/common/tests/test_model_chat_templates.py packages/common/tests/test_contracts.py packages/catalog/tests -q

Run Lab composition checks after the second slice:

    uv run pytest apps/lab/tests/test_catalog.py apps/lab/tests/test_work_packages.py apps/lab/tests/test_qualification.py -q
    uv run --package posttrain-lab posttrain --project-root apps/lab work-package validate gemma4_31b_serve_smoke_qualification.yaml
    uv run --package posttrain-lab posttrain --project-root apps/lab work-package validate gemma4_31b_sft_qualification.yaml
    uv run --package posttrain-lab posttrain-lab qualification list --project-root apps/lab --json

Run the repository validation ladder:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

The known full-workspace Pyright configuration currently scans an ignored nested environment and may not be a usable local terminal gate. If that remains true, do not conceal it: record the exact behavior, run Pyright over every changed production module, and leave the repository-wide configuration debt outside this model-support diff.

Submit the real jobs with the existing target embedded in their bindings:

    uv run --package posttrain-lab posttrain --project-root apps/lab --json job run gemma4_31b_serve_smoke_qualification.yaml --job smoke --provider dstack --env HF_TOKEN
    uv run --package posttrain-lab posttrain --project-root apps/lab --json job run gemma4_31b_sft_qualification.yaml --job train --provider dstack --env HF_TOKEN

Record the canonical run IDs from these commands before tracking them with `posttrain run status`, `posttrain run logs`, `posttrain run wait`, and `posttrain run reconcile` as appropriate.

## Validation and Acceptance

Contract acceptance requires `models/gemma4-31b-it@bf16` to resolve identically from the Python registry and base YAML, with the exact immutable Hub commit, stored BF16 parameter count, family renderer, tokenizer fingerprint, context, modalities, and upstream topology provenance. Existing 12B behavior must remain unchanged and no architecture field may appear.

Training composition acceptance requires the 31B binding to use `AutoModelForMultimodalLM` through existing family dispatch and to select only actual language-model attention and dense MLP projections. The real run must finish both optimizer updates with finite loss and gradient norms, report positive trainable parameters below the loaded total, retain the adapter and summary, and identify the 31B foundation selection as lineage. A clean process must load the exact retained adapter and produce finite logits or a token.

Serving acceptance requires vLLM 0.25.1 to load the pinned foundation checkpoint on the 96 GiB target, expose health and model endpoints, and return nonempty chat content. A separate tool-bearing request must return `finish_reason=tool_calls` with a structured function name and JSON arguments. Adapter serving must also return nonempty content unless the pinned engine rejects it; any rejection must be retained and called out as an incomplete portion of qualification.

Scope acceptance requires no changes to model-family contracts, generic loader dispatch, renderer implementations, online request shape, package boundaries, dependency manifests, or the frozen product baseline. This plan does not qualify multimodal execution, 256K serving, MoE, MTP speculation, DPO, GRPO, distillation, full fine-tuning, QLoRA, or quantized weights.

## Idempotence and Recovery

All code and catalog edits are additive and safe to validate repeatedly. Hub probes and jobs pin immutable revisions. Interrupted Hub downloads may be retried against the same revision without deleting shared caches. A failed provider run must remain as evidence; retry by submitting a new framework run rather than overwriting or relabeling the prior attempt.

Before each commit inspect `git status --short`, `git diff`, and `git diff --check`, and stage only files named in this plan. Never commit `.posttrain/state`, tokens, signed URLs, model weights, adapter payloads, or raw provider credentials. If the 96 GiB target cannot fit a required BF16 operation, preserve the failure, narrow runtime settings only when workload meaning remains intact, and update this plan before retrying. Do not silently switch to quantization, CPU offload, multiple GPUs, or a different checkpoint.

## Artifacts and Notes

Known immutable inputs:

    branch: feat/gemma4-support
    model: google/gemma-4-31B-it
    model revision: 842da3794eaa0b77d5f08bae87a17459d91ff475
    stored BF16 parameters: 31,273,088,876
    upstream model type: gemma4
    upstream architecture: Gemma4ForConditionalGeneration
    tokenizer fingerprint: 059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416
    dense LoRA target modules discovered: 410
    rank-8 LoRA trainable parameters under empty weights: 61,214,720
    transformers: 5.14.1
    vllm: 0.25.1
    TRL fork revision: 6e7739b8ec741d21ecd79c0c212694cd15ff20d8
    target: targets/carbonteq-rtx-pro-6000-96gb

Add real run IDs, provider IDs, actual-job image digests, catalog snapshots, GPU identity, startup duration, peak memory, generated-answer summary, structured tool call, optimizer metrics, loaded total/trainable parameters, adapter identity and content digest, and reload/adapter-serving results here as work proceeds. Do not paste secrets or large model outputs.

Local validation before GPU submission:

    focused common/catalog tests: 55 passed, 2 skipped
    focused Lab tests: 35 passed, 1 skipped
    tokenizer-aware Gemma/TRL tests: 5 passed, 6 unrelated uncached-tokenizer skips
    static work-package validation: both complete
    uv sync --all-packages --locked: resolved 307 packages
    full Ruff: passed
    import contracts: 8 kept, 0 broken
    targeted Pyright: 0 errors, 0 warnings, 0 informations
    full pytest: 1,043 passed, 20 skipped, 4 warnings
    git diff --check: clean

## Interfaces and Dependencies

At completion `packages/common/src/posttrain/common/variants/gemma4.py` exposes both `GEMMA_4_12B_IT` and `GEMMA_4_31B_IT`, backed by the single `GEMMA4_RENDERER_CONTRACT`. `FOUNDATION_VARIANTS` contains both exact ids. The base catalog contains `models/gemma4-31b-it@bf16` as an exact `ModelVariant` value.

No new reusable interface is expected. `packages/train/src/posttrain/train/backends/trl/common.py` continues to return `AutoModelForMultimodalLM` for `family == "gemma4"`, and `GEMMA4_RENDERER` remains the generic default renderer with reasoning off. Lab provides checkpoint-specific `SFTSettings`, `TrainingBinding`, `InferenceBinding`, work packages, and candidate gates. All dependencies remain at the locked versions listed above.

Revision note (2026-08-05): Created the plan after reconciling the completed 12B qualification with the exact 31B checkpoint. The feasibility probe proved that family-level renderer and loader behavior generalize, while model identity, resource policy, and qualification evidence remain checkpoint-specific.

Revision note (2026-08-05): Updated after implementation and local validation. The result adds only an exact reusable model value plus checkpoint-specific Lab policy; no new family, architecture flag, renderer, loader, dependency, or backend branch was needed.
