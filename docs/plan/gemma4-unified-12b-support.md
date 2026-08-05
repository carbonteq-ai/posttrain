# Add the Gemma 4 Unified 12B support plane (historical scope)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `docs/templates/PLAN.md`. It is intentionally self-contained: a contributor should be able to implement and qualify this change from this file and the current repository without relying on chat history.

The active scope is now maintained in `docs/plan/gemma4-0.3.2-support-and-release.md`.
That plan expands this work to the E2B, E4B, 12B, and 31B dense matrix and
records the accepted non-truncated 12B MTP run. This file remains useful for
the original 12B implementation history; do not use its old “unsupported
31B” or “128-token accepted proof” wording as the current release contract.

## Purpose / Big Picture

After this change, posttrain can identify, render, run a bounded text-only serving smoke, perform LoRA supervised fine-tuning (SFT), and run a bounded TRL GRPO rollout with paired-assistant MTP for the instruction-tuned `google/gemma-4-12B-it` checkpoint. The exact supported plane is the immutable `models/gemma4-12b-it@bf16` selection in the `gemma4` family. Its upstream `gemma4_unified` model type remains checkpoint provenance rather than a separately authored framework axis. A user will pair that selection with the Gemma-specific TRL bindings and get the same catalog resolution, job execution, evidence, and artifact behavior used by existing model families.

This historical slice did not add multimodal training or evaluation, DPO,
distillation, quantization, or a full-context benchmark. The active 0.3.2 plan
adds serving smoke selections for the E2B, E4B, and 31B dense checkpoints while
keeping every executable profile text-only and bounded. The 12B model facts
record text, image, and audio capability; video is not part of its pinned
configuration. The accepted GRPO proof is now the non-truncated
`gemma4-trl-mtp-qualification-4` run; the older 128-token run is retained only
as historical execution evidence.

The observable proof has three parts. First, a Lab serving work package launches the exact pinned foundation checkpoint with vLLM, exposes it through the OpenAI-compatible endpoint, and returns a nonempty concise answer while using Gemma 4's parser and template settings. Second, a Lab SFT work package performs bounded LoRA optimizer updates using the shared smoke dataset, records nonzero trainable parameters and finite training metrics, and emits a loadable PEFT adapter plus training summary. Third, a Lab TRL GRPO work package starts colocated vLLM with the pinned paired assistant, completes two rollout attempts and one optimizer step, and persists nonzero speculative-decoding evidence. This last proof validates the MTP execution path; it is not a quality or throughput benchmark.

## Progress

- [x] (2026-08-04 12:35Z) Read the canonical post-training baseline, the plan template, current model/renderer/train/serve/catalog/Lab code, dependency pins, and existing qualification surfaces.
- [x] (2026-08-04 12:35Z) Fix the implementation boundary to Gemma 4 Unified 12B, text-only vLLM serving, and TRL LoRA SFT.
- [x] (2026-08-04 12:35Z) Record the exact upstream model identity and the existing repository hardware target for real qualification.
- [x] (2026-08-04 13:02Z) Ran the pinned dependency, tokenizer/processor, renderer-mask, meta-model, forward-signature, PEFT target, and tokenizer-fingerprint probes.
- [x] (2026-08-04 13:15Z) Initially added a required architecture axis across model contracts and consumers.
- [x] (2026-08-04 13:15Z) Added the exact Gemma renderer contract and 12B variant; common tests pass and cached Gemma template tests exercise off/thinking/tools.
- [x] (2026-08-04 13:15Z) Added Gemma-aware TRL loading, explicit Gemma SFT rendering, language-model-only LoRA targeting, and focused tests.
- [x] (2026-08-04 13:15Z) Added Lab-only Gemma training/inference selections, candidate gates, declarative SFT/serving work packages, and a model-neutral generation-smoke job.
- [x] (2026-08-04) Reassessed the architecture axis against offline validation and upstream auto-model behavior; removed the field, its propagation, catalog migration, and architecture-dispatch utility in favor of family-aware loading.
- [x] (2026-08-04) Validated the corrected family-only contract with focused tests, the broader owning-package suite, the full repository suite, targeted Ruff and Pyright, import boundaries, both primary-CLI static work-package validations, and `git diff --check`.
- [x] (2026-08-04) Audited the complete branch delta for minimality: reused the generic training renderer instead of adding a Gemma implementation enum/branch, removed a redundant Gemma HTTP request-shape test, and made the existing text-only vLLM translation disable audio as well as image/video.
- [x] (2026-08-05) Planned both real qualification packages against the 96 GiB target; the first serving submission exposed and then fixed a checkout-source packaging omission before any remote run was created.
- [x] (2026-08-05) Submitted both qualification packages through dstack. The serving run `96152bfa-08c3-4bdb-bd07-286f1ec0b8e9` completed successfully; the SFT run `6b04480c-0df5-4ec6-ba00-3201bd9953e3` is submitted and provider-queued for the same single workstation.
- [x] (2026-08-05) Tracked SFT run `6b04480c-0df5-4ec6-ba00-3201bd9953e3` to terminal success. It completed two optimizer updates with finite loss and gradient norms and published the required model and summary artifacts plus a recovery checkpoint.
- [ ] Validate the smallest package suites, then the repository-wide static and test ladder (completed: locked sync, full Ruff, import boundaries, targeted Pyright, focused ownership suites, full pytest, and diff check; remaining: full-workspace Pyright wrapper does not terminate locally).
- [x] (2026-08-05) Closed the remaining real qualification acceptance on `carbonteq-ai-workstation.lan`: raw Trackio metrics prove the LoRA parameter ratio and peak SFT memory, a clean process materialized and reloaded the exact adapter with finite logits, the Gemma parser returned a structured tool call, and pinned vLLM served the retained adapter successfully.
- [x] (2026-08-05) Added the paired-assistant MTP contract, pinned assistant materialization, and a bounded Gemma TRL GRPO work package. The successful run `6579a33b-d78e-4162-a38e-8371a93b2351` completed two rollouts and one optimizer step with nonzero MTP counters; a follow-up 512-token attempt `c8cf775c-9ebe-415a-9be0-68c5c3bed66e` reached the verifier harness but failed on setup timeout before producing a trainable branch. The latter is retained as a retry caveat, not an MTP failure.

## Surprises & Discoveries

- Observation: The canonical primitive requires a `ModelVariant` to establish architecture compatibility, but that does not require a caller-authored `architecture` field.
  Evidence: a variant already pins an immutable checkpoint, whose `config.json` declares `model_type` and concrete architecture. Static work-package validation uses catalog facts and family compatibility without fetching the checkpoint, while Transformers and vLLM resolve concrete architecture at runtime. The initially added field was not consumed by static validation and duplicated checkpoint metadata.

- Observation: Gemma 4 Unified is a composite conditional-generation architecture even for text-only use, so the current unconditional `AutoModelForCausalLM` training loader is not a safe loading path.
  Evidence: the pinned checkpoint declares `Gemma4UnifiedForConditionalGeneration`, `model_type=gemma4_unified`, and the `AutoModelForMultimodalLM` auto class. `packages/train/src/posttrain/train/backends/trl/common.py::load_trainable_model` currently always calls `AutoModelForCausalLM.from_pretrained`.

- Observation: The current vLLM adapter already has the required generic controls for this slice.
  Evidence: `packages/serve/src/posttrain/serve/profiles/base.py::VllmEngineConfig` translates `text_only` to zero multimodal limits and supports `skip_mm_profiling`; `packages/serve/src/posttrain/serve/backends/vllm/bindings.py::frontend_args` already emits configured tool-call and reasoning parsers. vLLM 0.25.1 supports `Gemma4UnifiedForConditionalGeneration` and parser name `gemma4`.

- Observation: Lab already declares a reusable 96 GiB NVIDIA target and machine-local state records a prior direct vLLM launch for this exact model on an 80 GiB-or-larger GPU.
  Evidence: `apps/lab/.posttrain/catalog/execution-targets.yaml` defines `targets/carbonteq-rtx-pro-6000-96gb`; the ignored `apps/lab/.posttrain/state/gemma4-12b-it-latest.dstack.yml` launches `google/gemma-4-12B-it` with bfloat16, a 32K cap, eager mode, and multimodal profiling disabled. The ignored file is historical evidence only and must not become a framework contract.

- Observation: The pinned `renderers==0.1.8` package does not advertise a dedicated Gemma renderer, while the pinned model tokenizer carries a canonical chat template and response schema. Whether `DefaultRendererConfig` produces correct assistant-only masks and tool boundaries is therefore a feasibility question, not an assumption.
  Evidence: `packages/train/src/posttrain/train/rendering.py` routes non-Qwen families through `DefaultRendererConfig`. The first milestone below makes exact render/mask behavior a stop/go gate.

- Observation: Online serving and evaluation currently serialize renderer keyword arguments differently.
  Evidence: `packages/serve/src/posttrain/serve/online.py::generate` spreads values such as `enable_thinking` at the top level, while the vLLM OpenAI contract and the evaluation path use a nested `chat_template_kwargs` object. Gemma's declared `thinking` mode is not a truthful online contract until this is normalized.

- Observation: `renderers==0.1.8` has no typed Gemma renderer, but its documented `DefaultRenderer` path is sufficient for the deliberately SFT-only slice.
  Evidence: the exact tokenizer probe rendered ordinary and tool-bearing conversations with every user and tool-response content token masked and selected assistant spans trainable. `DefaultRenderer` has no `sampled_mask`, so assistant scaffolding such as `<|turn>model` is also trained and `ensure_final_stop` is a no-op; the Gemma template already emits `<turn|>`. This is acceptable for bounded SFT but not a claim of loss-exact multi-turn RL rendering.

- Observation: The unified model exposes a token-only causal-loss forward path and the text transformer is cleanly separable from multimodal embedders.
  Evidence: a meta-weight `Gemma4UnifiedForConditionalGeneration` accepts `input_ids`, `attention_mask`, and `labels`. The regex `^model[.]language_model[.]layers[.]\d+[.](self_attn[.](q_proj|k_proj|v_proj|o_proj)|mlp[.](gate_proj|up_proj|down_proj))$` selected 328 linear modules and PEFT created 656 LoRA parameters, all under `language_model.layers`; none selected `model.embed_audio` or `model.embed_vision`.

- Observation: The exact tokenizer and processor share one 262144-entry vocabulary, and a canonical ordered-vocabulary/special-id digest is now known.
  Evidence: `GemmaTokenizer` and `Gemma4UnifiedProcessor.tokenizer` returned equal vocabularies. SHA-256 over compact UTF-8 JSON containing vocabulary entries ordered by `(token_id, token)` and sorted unique `all_special_ids` is `059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416`.

- Observation: The full-workspace Pyright command did not complete normally in the local process wrapper and emitted no diagnostics before manual interruption.
  Evidence: `uv run pyright` reproduced the behavior both before and after `uv sync --all-packages --locked --python 3.13`; interrupting the wrapper produced a `KeyboardInterrupt` while Python waited for its Node subprocess. The six changed production modules pass targeted Pyright with `0 errors, 0 warnings, 0 informations`. Resolve or run full Pyright in CI before completion; this is a validation-runner issue, not a code diagnostic.

- Observation: All locally executable behavioral and boundary checks pass after the implementation.
  Evidence: focused ownership suites reported `295 passed, 11 skipped`; the full repository suite reported `928 passed, 18 skipped`; Ruff reported `All checks passed!`; import-linter kept all 8 contracts; both Gemma work packages passed primary-CLI static composition validation; `git diff --check` is clean.

- Observation: Removing the architecture field preserved the complete local compatibility surface.
  Evidence: focused removal regressions reported `78 passed, 1 skipped`; the broader common/catalog/train/serve/jobs/Lab suite reported `290 passed, 10 skipped`; after the minimality audit the full repository suite reported `926 passed, 18 skipped`; full Ruff and targeted Pyright passed; all 8 import contracts remain intact; and both Gemma work packages still report complete static composition validation without loading model metadata or an ML runtime.

- Observation: Gemma 4 Unified and tower checkpoints diverge in their multimodal internals, not in the family renderer or required Transformers auto-model factory.
  Evidence: Unified uses direct vision/audio embedders while tower models use modality towers, but both official model types are registered under `AutoModelForMultimodalLM`; the pinned 12B and 31B canonical chat templates are byte-identical. Posttrain can therefore keep family-level renderer and loader selection while retaining the exact upstream model type and class as provenance.

- Observation: No Gemma-specific `TrainingRenderer.implementation` is required for the bounded SFT path.
  Evidence: the generic renderer path already forwards the selected reasoning mode's chat-template kwargs into `DefaultRendererConfig`, which is the exact configuration proven by the feasibility probe. After removing the redundant enum/branch, the pinned train environment executed the Gemma chat-template and SFT rendering cases successfully; the reported skips were only uncached Qwen and LFM tokenizers.

- Observation: The existing `text_only` vLLM translation disabled image and video inputs but not audio.
  Evidence: Gemma 4 Unified 12B declares image, video, and audio processors, and vLLM 0.25.1 accepts `audio` in `limit_mm_per_prompt`. The generic translation now sets all three supported multimodal inputs to zero, making the existing `text_only` name truthful for this checkpoint.

- Observation: Source-checkout actual-job packaging omitted the environment-contract distribution even though its consumers declared it.
  Evidence: the serving qualification image failed its pre-publication smoke check with `ModuleNotFoundError: No module named 'posttrain.environment'`. `posttrain-catalog` and `posttrain-runtime` both declare `posttrain-environment`, and the wheel-based framework distribution list already included it, but `_FRAMEWORK_INSTALL_ROOTS` omitted `packages/environment`. Adding that install root and a regression assertion makes checkout and wheel staging agree; no dstack run was created by the failed attempt.

- Observation: The SFT run recorded the required model-size and memory metrics even though the job-specific Observatory projection does not display them.
  Evidence: direct Trackio retrieval for `train.sft-6b04480c` returned `train/parameters_total=11,992,514,560`, `train/parameters_trainable=32,784,384`, `train/parameters_trainable_fraction=0.0027337372688584836`, and `train/peak_gpu_memory_gib=28.37083673477173`.

- Observation: The exact retained Gemma adapter works through both backend consumption paths in the pinned runtime stack.
  Evidence: dstack task `gemma4-adapter-reload-validation-v3` materialized content digest `dfc511859f94a6c98a2d3a1d7552699c7e471a2ae8ba78999e999ce86af01796`, loaded `PeftModelForCausalLM` over `Gemma4UnifiedForConditionalGeneration`, produced finite logits and one generated token, and peaked at 22.521 GiB. Task `gemma4-serve-contract-validation-v3` loaded the same digest through vLLM LoRA serving and returned nonempty final content.

- Observation: The Gemma vLLM parser produces a genuine OpenAI-compatible structured tool response rather than only accepting parser flags at startup.
  Evidence: `gemma4-serve-contract-validation-v3` sent a real tool-bearing request and received `finish_reason=tool_calls` with function `get_weather` and JSON arguments `{"city":"Paris"}`.

- Observation: The repository-wide Pyright command is scanning an ignored nested environment and is not currently a usable terminal gate on this checkout.
  Evidence: verbose Pyright found 12,748 source files and entered `apps/lab/environments/skyrl_bird_sql_v1/.venv`; the configured exclusions contain root-relative `.venv` and `.venvs`, not recursive patterns. A temporary recursive exclusion completed and exposed 146 broader workspace diagnostics dominated by namespace re-export resolution, while the ten Gemma production modules pass with zero diagnostics. This remains repository validation debt and was not changed as part of model support.

- Observation: Paired-assistant MTP is active in the TRL/vLLM worker and its telemetry is persisted as canonical Trackio metrics.
  Evidence: run `6579a33b-d78e-4162-a38e-8371a93b2351` on `carbonteq-ai-workstation.lan` completed two rollouts and one optimizer step. Trackio recorded `serve/backend/speculative_draft_tokens=135`, `serve/backend/speculative_accepted_tokens=120`, `serve/backend/speculative_acceptance_rate=0.8889`, `serve/backend/kv_cache_capacity_tokens=22080`, and `serve/backend/kv_cache_peak_usage_ratio=0.02302`; the image used the pinned assistant revision from the resolved run attributes. No CUDA or vLLM error was emitted.

- Observation: A longer 512-token retry did not disprove MTP; it failed earlier at the Verifiers harness setup boundary.
  Evidence: run `c8cf775c-9ebe-415a-9be0-68c5c3bed66e` ended with `HarnessError: harness setup timed out` and `got 0` trainable branches. It produced no MTP counters and no CUDA/vLLM failure. The accepted MTP proof therefore remains the bounded 128-token run, while a non-truncated quality qualification is still future work.

## Decision Log

- Decision: Support only the immutable `google/gemma-4-12B-it` instruction checkpoint in the `gemma4` family.
  Rationale: The checkpoint identity and revision bound the support claim precisely. Its Unified topology matters to backend qualification and LoRA targeting but need not be repeated as a public selection axis.
  Date/Author: 2026-08-04 / Codex and user.

- Decision: Do not add a required `architecture` field to `ModelVariant` or catalog rows; retain `upstream_model_type` and `upstream_architecture` as Gemma checkpoint provenance.
  Rationale: Posttrain's offline validation does not consume an architecture field, the immutable checkpoint is the authority, and Transformers/vLLM resolve its concrete class. A second independently authored value creates drift without enabling an additional rejection. This supersedes the initial architecture-field decision made earlier on 2026-08-04.
  Date/Author: 2026-08-04 / Codex.

- Decision: Name the public model selection `models/gemma4-12b-it@bf16`, the Python variant `GEMMA_4_12B_IT`, and the renderer contract `gemma4-tools@1`.
  Rationale: These follow existing selection and constant conventions. The renderer name is family-level because the conversation protocol is a Gemma 4 protocol; the immutable checkpoint resolves its concrete model class.
  Date/Author: 2026-08-04 / Codex.

- Decision: Represent both `off` and `thinking` conversation modes, default to `off`, but introduce only an `off` SFT training binding in this slice.
  Rationale: The pinned tokenizer template exposes `enable_thinking`, so omitting the mode would make the renderer contract inaccurate. Training the thinking form is a separate behavior and evidence obligation.
  Date/Author: 2026-08-04 / Codex.

- Decision: Keep the reusable model fact accurate with upstream modalities `(text, image, audio, video)` and context window 262144, but set every new executable profile to text-only with a bounded 8192-token context.
  Rationale: Model capability facts and qualified framework behavior are different concerns. This avoids falsely claiming multimodal or long-context post-training while retaining truthful discovery metadata.
  Date/Author: 2026-08-04 / Codex.

- Decision: Set `mtp=False` and do not add a speculative binding.
  Rationale: Superseded by the paired-assistant amendment below; this was correct for the SFT-only scope but is no longer the desired model capability contract.
  Date/Author: 2026-08-04 / Codex.

- Decision: Set Gemma 4's `mtp` capability true only with a paired assistant
  checkpoint declared by `assistant_model` and a full-commit
  `assistant_revision` in the colocated TRL inference engine mapping.
  Rationale: vLLM 0.25.1 recognizes Gemma's assistant architecture but cannot
  take a revision field in `speculative_config`. The TRL adapter therefore
  resolves the full SHA with `snapshot_download`, passes the resulting local
  path as vLLM's `model`, and rejects incomplete or unpinned paired mappings.
  Native MTP mappings for existing models remain unchanged.
  Date/Author: 2026-08-05 / Codex.

- Decision: Use `AutoModelForMultimodalLM` for the `gemma4` family and retain `AutoModelForCausalLM` for existing families.
  Rationale: Both official Gemma 4 Unified and tower model types are registered with the multimodal auto factory, which resolves their concrete classes from checkpoint configuration. Family dispatch is the smallest framework-owned distinction required by the current loader.
  Date/Author: 2026-08-04 / Codex.

- Decision: Put the reusable model variant and backend capability in framework packages, but put the 96 GiB target, qualification settings, inference/training bindings, and work packages in the Lab project overlay.
  Rationale: Concrete host capacity and release evidence belong to the composition host. The base catalog must not depend on CarbonTeq workstation names or claim that every installation owns a 96 GiB GPU.
  Date/Author: 2026-08-04 / Codex.

- Decision: Do not vendor Gemma's large chat template. Use `ChatTemplate("tokenizer")` against the immutable checkpoint revision.
  Rationale: The template is part of the pinned tokenizer artifact and includes its response schema. Copying it would create two independently versioned sources.
  Date/Author: 2026-08-04 / Codex.

- Decision: Send all online renderer mode values through the model-neutral `chat_template_kwargs` request field and update existing Qwen serving tests to the same contract.
  Rationale: This matches vLLM's OpenAI-compatible request schema and the existing evaluation adapter. Keeping a top-level special case would make Gemma thinking mode backend-dependent and preserve two serialization conventions.
  Date/Author: 2026-08-04 / Codex.

- Decision: Treat the renderers probe as a hard implementation gate. If `renderers==0.1.8` cannot produce correct Gemma assistant masks and tool turns, stop, document the failure here, and amend this plan with an upstream dependency change before production edits.
  Rationale: A local ad hoc parser in `posttrain.train` would violate renderer ownership and would make “support” appear complete while corrupting labels.
  Date/Author: 2026-08-04 / Codex.

- Decision: The renderer gate passes for SFT through explicit `DefaultRendererConfig`; do not add or fork a typed renderer in this change.
  Rationale: Role attribution is correct for the selected SFT examples, the template supplies its own final stop, and DPO/RL/bridge support is explicitly outside this plane. A future Gemma online-training plane must first add a typed upstream renderer with `sampled_mask`, parsing, and bridge parity.
  Date/Author: 2026-08-04 / Codex.

## Outcomes & Retrospective

The surgical Gemma 4 Unified 12B support plane is implemented and its model-specific qualification is complete for serving, SFT, and the bounded paired-assistant MTP execution path. The exact foundation checkpoint generated successfully through vLLM; LoRA SFT completed two finite updates with 32,784,384 of 11,992,514,560 parameters trainable and retained an immutable adapter; a clean PEFT process reloaded that adapter and produced finite logits; the Gemma parser emitted a structured tool call; and pinned vLLM served the retained adapter with nonempty output. The MTP GRPO run completed two rollouts, one optimizer step, and emitted 135 draft tokens with 120 accepted (0.8889 acceptance), with 2.3% peak KV-cache use and no CUDA/vLLM error. Its 128-token cap truncated both completions and produced zero reward/gradient, so it proves execution and telemetry, not learning quality. A 512-token retry failed with `HarnessError: harness setup timed out` before a trainable branch; preserve this as an environment qualification caveat. The initially introduced required architecture field was removed after review showed that it duplicated immutable checkpoint metadata and enabled no current static validation. Focused tests, the complete pytest suite, Ruff, import boundaries, targeted Pyright, and whitespace checks pass. Repository-wide Pyright remains a pre-existing workspace-configuration gate rather than a Gemma diagnostic, so gate promotion should record that distinction instead of broadening this model-support change.

## Context and Orientation

The canonical product baseline is `docs/post-training/README.md` and the six documents beside it. This change uses the paired-assistant MTP amendment in the canonical README and API document. It does not add a new job kind or training objective; it adds a bounded value to the existing model capability and inference-engine contracts. Architecture compatibility is established by the exact pinned target and assistant checkpoints, backend support, and qualification evidence rather than a separately authored required field.

`packages/common/src/posttrain/common/models.py` owns immutable model facts. A `ModelVariant` is one exact loadable weight state, not a loose marketing family. Its `RendererContract` describes roles, reasoning modes, tool boundaries, and the source of the chat template. `packages/common/src/posttrain/common/catalog_schema.py` validates untrusted YAML before it becomes those Python values. `packages/common/src/posttrain/common/catalog.py` decodes validated rows. Family variants live under `packages/common/src/posttrain/common/variants/` and base model selections live in `packages/catalog/src/posttrain/catalog/base/models.yaml`.

`packages/train` owns reusable training behavior. `TrainingRenderer` is the training binding's selection of a model renderer and reasoning mode. `packages/train/src/posttrain/train/rendering.py` uses the external `renderers` library to create token IDs and a loss mask; a true value in the loss mask means that token contributes to the training loss. For SFT, only explicitly trainable assistant messages may be true. `packages/train/src/posttrain/train/backends/trl/common.py` loads the tokenizer, base weights, and PEFT adapters. PEFT means parameter-efficient fine-tuning; the LoRA form trains small low-rank matrices while freezing base weights.

`packages/serve` owns provider-neutral serving requests plus the private vLLM adapter. `VllmEngineConfig` translates the catalog's engine values into vLLM Python and CLI options. The engine's `text_only` value prevents image, video, and audio inputs, while `skip_mm_profiling` avoids reserving multimodal profiling memory. A parser converts model-specific reasoning and tool syntax into OpenAI-compatible response fields. The existing parser fields are data-driven and need no Gemma-specific branch.

`packages/jobs` owns standard `train.sft` and `serve.smoke` job definitions. They are already model-neutral and must not gain Gemma branches. `apps/lab` is the reference composition and qualification host. Its tracked overlay lives in `apps/lab/.posttrain/catalog/`, its declarative work packages live in `apps/lab/.posttrain/work_packages/`, and `apps/lab/src/posttrain_lab/qualification/gates.toml` inventories reviewed qualification surfaces. This plan uses those surfaces rather than adding another hard-coded `posttrain-lab` command.

The exact upstream model facts to encode are:

- Hugging Face repository: `google/gemma-4-12B-it`.
- Immutable revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Transformers architecture class: `Gemma4UnifiedForConditionalGeneration`.
- Transformers model type retained as provenance: `gemma4_unified`.
- Total parameters: `11_959_730_224`.
- Weight precision: bfloat16.
- Native context window: 262144 tokens.
- License provenance value: `apache-2.0`.
- Tool parser and reasoning parser name in vLLM 0.25.1: `gemma4`.

Do not guess the tokenizer fingerprint or LoRA module regular expression. The first milestone computes and proves both against the exact pinned revision, then records them in this plan before production catalog entries are added.

## Plan of Work

### Milestone 1: Prove the pinned dependencies can express this slice

Before changing a public contract, create an uncommitted diagnostic script under `/tmp`, never under the repository, and run it in the `posttrain-train` environment. Load `AutoConfig`, `AutoTokenizer`, and `AutoProcessor` from the exact revision. Assert `config.model_type == "gemma4_unified"`, confirm the processor uses the same tokenizer, print the auto-map/architecture metadata, and compute the tokenizer fingerprint with the same canonical byte selection used for current Qwen fingerprints. If the repository has no reusable fingerprint helper, define the precise inputs in this plan before implementing one; at minimum the fingerprint must cover the immutable tokenizer files and chat/response template state, not a Python object representation with unstable ordering.

Probe `DefaultRendererConfig(enable_thinking=False)` through `renderers==0.1.8` with three short conversations: user then assistant, user then assistant tool call, and user then assistant tool call then tool response then assistant. For each, inspect token IDs, decoded text, and the loss mask produced by `build_training_sample`. Acceptance requires user, system, and tool-response tokens to be masked; the selected assistant answer or tool-call tokens to be trainable; Gemma boundary tokens to appear exactly once in their expected places; and `ensure_final_stop=True` to add exactly one terminal stop. Repeat ordinary chat rendering with `enable_thinking=True` to prove the declared conversation mode changes the generation prompt as the tokenizer contract specifies. Do not claim thinking-mode SFT from this probe.

Inspect the exact Transformers model source or instantiate it under an empty/meta-weight context so no full checkpoint allocation is required. Record the paths of the text transformer's attention and MLP projections. Choose a regex that selects only the language-model `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` modules and excludes audio/image encoders, projectors, embeddings, output heads, and any expert/router modules. Instantiate `LoraConfig` against the empty model and print selected module names and trainable parameter count. Acceptance requires at least one module in every intended projection class and zero modules outside the language model.

Finally, test the model factory with the pinned Transformers 5.14.1 API. Confirm `AutoModelForMultimodalLM` exists, recognizes the config without `trust_remote_code`, accepts token-only `input_ids`, `attention_mask`, and `labels`, and exposes a causal language-model loss. This may use empty/meta weights for signature validation; the real forward/backward proof belongs to the GPU milestone.

Record all exact outputs and chosen values under `Surprises & Discoveries` and `Artifacts and Notes`. If the renderer probe fails, do not add a local Gemma parser. Determine whether a released `renderers` version already fixes the behavior. If not, the work becomes a multi-repository or upstream-package change: amend this plan to name that repository, tests, commit order, immutable pin, `uv.lock`, runtime dependency lock, and qualification rerun. If Transformers or PEFT cannot load the architecture with the pinned versions, follow the same stop-and-amend rule. This gate keeps speculative dependency work out of the surgical path.

### Milestone 2: Keep concrete architecture checkpoint-owned

Do not add a required architecture field to `ModelVariant`, its catalog schema, or derived model constructors. The exact immutable artifact revision is the authority for concrete `model_type` and architecture. Preserve those upstream values in Gemma provenance for inspection and evidence, but do not create a second independently authored compatibility axis.

Static compatibility remains expressed through existing model-family, renderer, binding, capability, and execution-target contracts. Runtime adapters must choose only the broad auto-model API required by the family and allow the backend to resolve the concrete class from the checkpoint. Tests must prove that legacy catalog rows remain unchanged and that Gemma provenance retains the upstream model type and class.

### Milestone 3: Add the Gemma renderer contract and exact model variant

Create `packages/common/src/posttrain/common/variants/gemma4.py`. Define `GEMMA4_RENDERER_CONTRACT` with id `gemma4-tools@1`, family `gemma4`, tokenizer-sourced chat template, roles `system`, `user`, `assistant`, and `tool`, reasoning modes `off` and `thinking`, default `off`, and `strips_past_reasoning=True` only if the milestone-1 template probe proves that behavior. Map `off` to `enable_thinking=False` and `thinking` to `enable_thinking=True`. Extend `ToolCallProtocol.id` with `gemma4_structured` and encode the exact tool-call start/end tokens and assistant format observed in the pinned tokenizer template. Do not reuse the Qwen XML or LFM Python protocol names merely because all three support tools.

In the same module define `GEMMA_4_12B_IT` from the exact upstream facts in `Context and Orientation`, including `family="gemma4"`, an immutable `HubModelRef`, the computed tokenizer fingerprint, accurate modalities, `mtp=True`, and provenance containing source, license, upstream model type, upstream architecture class, and the paired assistant identity. Export the new contract and variant through `packages/common/src/posttrain/common/variants/__init__.py` and whatever package-level export pattern the current variants follow.

Add `models/gemma4-12b-it@bf16` to `packages/catalog/src/posttrain/catalog/base/models.yaml`. The base catalog row must contain only reusable model facts; it must not reference a Lab target or qualification binding. Update `packages/common/tests/test_model_variants.py`, `packages/common/tests/test_model_chat_templates.py`, and exact registry assertions such as `packages/common/tests/test_contracts.py`. Tests must prove immutable identity, upstream topology provenance, default-off and explicit-thinking template kwargs, role support, tool boundaries, registry resolution, and equality between the Python constant and YAML-decoded selection. Use the locally cached pinned tokenizer where token-level behavior is required and skip with the repository's existing clear cache-miss pattern.

### Milestone 4: Add only the TRL LoRA SFT training path

In `packages/train/src/posttrain/train/profiles.py`, add a public `GEMMA4_RENDERER` using id `gemma4-off-v1`, family `gemma4`, the existing `default` implementation, and reasoning mode `off`. Export it from `packages/train/src/posttrain/train/__init__.py`; do not add a new renderer implementation literal.

Use the existing generic `DefaultRendererConfig` path, which already forwards `enable_thinking` from the selected conversation mode. Add Gemma only to the assistant-only SFT mask test in `packages/train/tests/test_rendering.py`; do not add it to the DPO prefix parameterization because DPO is outside this support plane. Add a tool-bearing supervised example that proves the tool response is context-only and the assistant tool call/final answer are trained only when their message indices are selected.

In `packages/train/src/posttrain/train/backends/trl/common.py`, import `AutoModelForMultimodalLM` lazily alongside the existing auto classes. Add a small model-factory helper that returns the multimodal auto class for the `gemma4` family and otherwise preserves the existing causal-model loader. Use that helper in `load_trainable_model`; Transformers resolves Unified versus tower from the pinned checkpoint configuration. Retain text-only `AutoTokenizer` loading because the renderer consumes token IDs; do not switch all training data through `AutoProcessor` or add multimodal dataset fields.

Add a guarded paired-assistant path in the same TRL adapter. For Gemma MTP mappings, require `assistant_model` and a full-commit `assistant_revision`, resolve that exact revision with the already-pinned `huggingface-hub` package into the worker's model cache, and pass only the resulting local path as vLLM's `model` plus the speculative method and token count. Do not download or upgrade Python packages at runtime. Native mappings without assistant fields must keep their current behavior; incomplete Gemma mappings must fail before trainer construction.

Use the milestone-1 language-model-only target regex in the Lab training binding. Keep `task_type="CAUSAL_LM"` only if the PEFT probe and real run prove correct adapter injection and save/reload; otherwise use the smallest PEFT-supported configuration demonstrated by the probe and document the decision. Add fake-import unit tests around family-based factory selection, unchanged Qwen/LFM dispatch, exact load options, LoRA target selection, and adapter resume behavior. Do not add QLoRA, full-parameter, DPO, or distillation bindings. Add one Gemma TRL GRPO binding solely for the bounded MTP qualification described below.

Create `packages/train/tests/test_trl_common.py` for these loader tests rather than growing algorithm-level API or online-RL tests. The file should exercise the private adapter seam through fake auto-model classes and must not download weights.

### Milestone 5: Compose Lab selections and declarative proofs

Create `apps/lab/.posttrain/catalog/gemma4-unified-qualification.yaml` and list it in `apps/lab/.posttrain/catalog/layer.yaml`. Define a bounded SFT settings selection, initially `gemma4-12b-it/sft-qualification-v1`, with two optimizer steps, max length 512, per-device batch size 1, gradient accumulation 1, bfloat16 runtime behavior inherited from the TRL adapter, logging every step, and one retained final checkpoint. Define `training/gemma4-12b-it-trl-lora-qualification@1` using backend `trl@1.8.0`, renderer `gemma4-off-v1`, rank 8, alpha 16, the proven language-model-only target regex, and `targets/carbonteq-rtx-pro-6000-96gb`. Pin the existing TRL source revision and dependency-lock digest in the same manner as other Lab qualification bindings; do not change dependency versions unless milestone 1 forced a recorded plan amendment.

In that overlay define `inference/gemma4-12b-it-vllm-screen@1` for `models/gemma4-12b-it@bf16`, vLLM 0.25.1, renderer `gemma4-tools@1`, and the same Lab target. Use startup timeout 900 seconds, bfloat16, max model length 8192, one sequence, at most 4096 batched tokens, GPU utilization 0.80, eager mode, chunked prefill, automatic KV cache dtype, `text_only=true`, `skip_mm_profiling=true`, `tool_call_parser=gemma4`, and `reasoning_parser=gemma4`. Give it purposes `[screen, smoke]`, max output 128, and temperature 0.0. Do not copy the historical 32K machine-local launch into the qualified profile.

Also define a Gemma MTP-only TRL qualification in the same overlay: one GSM8K prompt group, two generations, one optimizer step, max prompt length 256, max completion length 512, training `max_length` at least 1024, colocated vLLM with eager mode, sleep during optimization, max model length 4096, one sequence, and a conservative GPU-memory fraction on the 96 GiB target. Its `speculative_config` must contain `method: mtp`, `num_speculative_tokens: 1`, `assistant_model: google/gemma-4-12B-it-assistant`, and the immutable assistant revision `364bd03c9952e5b7da73665ee30c9eccfc408345`. Run `gemma4-trl-mtp-qualification-4` is the accepted non-truncated execution proof; the earlier 128-token run and the setup-timeout retry remain diagnostic evidence. This is a TRL `train.grpo` proof, not a standalone serving binding.

In `packages/serve/src/posttrain/serve/online.py::generate`, replace the top-level spreading of reasoning-mode values with `chat_template_kwargs=extra` when the selected mode has values. Update `packages/serve/tests/test_online.py` and `packages/serve/tests/test_api.py` so existing Qwen requests assert the nested shape and add a Gemma off/thinking case. Keep tools in the normal OpenAI `tools` and `tool_choice` fields. This is a request-shape correction shared by all renderer contracts, not a Gemma conditional.

Create `apps/lab/.posttrain/work_packages/gemma4_unified_serve_smoke_qualification.yaml` using the standard `serve/vllm-smoke@1` definition and the new inference selection. Create `apps/lab/.posttrain/work_packages/gemma4_unified_sft_qualification.yaml` using standard `train/trl-sft@1`, `datasets/posttrain-sft-smoke@1`, the new SFT settings, and the new training binding. Add both to `apps/lab/src/posttrain_lab/qualification/gates.toml` as candidate gates during implementation. Promote them to extended gates only after real GPU evidence satisfies acceptance; a failed or unrun candidate must not silently become a release claim.

The standard `serve.smoke` job currently proves health and model exposure, while `apps/lab/src/posttrain_lab/jobs/foundation_screening.py::run_online_smoke` proves a generated answer but is wired by a hard-coded scenario command. Keep the primary work package declarative. If the standard smoke still performs only a health probe at implementation time, narrowly strengthen its model-neutral operation in `packages/jobs` or add a versioned model-neutral `serve/vllm-generation-smoke@1` definition that sends the existing “What is 2 + 2?” request and rejects empty content. Do not add `if gemma` behavior and do not add another Lab CLI branch. The selected acceptance must demonstrate one generation, not merely process startup.

Add catalog and work-package tests in `apps/lab/tests/test_catalog.py`, `apps/lab/tests/test_work_packages.py`, and qualification inventory tests. They must resolve the exact Gemma model, training, inference, settings, dataset, and target seats; assert family and upstream architecture provenance; assert the text-only/parser engine values; and validate both YAML work packages without a GPU.

### Milestone 6: Validate code and collect real integration evidence

Run focused tests after each milestone, then the full validation ladder from the repository root. A cache-dependent tokenizer test may skip on a clean CPU host, but the release-gate GPU run must have network access or a cache containing the exact revision. No test may silently fall back to a moving branch or `latest` revision.

On the Lab target, validate both work packages through the primary CLI, then run the serving work package and the SFT work package through the normal tracked execution path. The serving run succeeds only if vLLM loads the pinned architecture, the endpoint exposes the selected model, a chat completion returns nonempty final content, and the run records terminal status and runtime versions. Also issue one explicit tool-bearing OpenAI-compatible request and preserve the response showing a structured `tool_calls` field; this is evidence for parser support, not a separate benchmark.

The SFT run succeeds only if both optimizer steps finish; `train/parameters_trainable` is positive and less than `train/parameters_total`; losses and gradient norms are finite; the summary reports two updates; and the PEFT adapter artifact is reconciled into tracking with model lineage pointing to `models/gemma4-12b-it@bf16`. Materialize the produced adapter and run the existing PEFT reload path at least once. If vLLM supports the resulting Gemma adapter in the pinned stack, run the same generation smoke against it and record that evidence. If it does not, do not hide the gap: keep foundation serving and adapter training results, leave adapter-serving qualification incomplete, and amend the support claim and this plan before declaring the full train-then-qualify flow complete.

Record run IDs, immutable runtime image digests, selected catalog digests, artifact versions/digests, GPU identity, peak memory, and concise acceptance output in `Artifacts and Notes`. Remove no machine-local cache after a successful run. Never commit tokens, signed URLs, `.posttrain/state/`, or raw model weights.

## Concrete Steps

Use `/home/owayys/Projects/carbonteq-ai/posttrain` as the working directory for every command unless a command explicitly says otherwise.

Before implementation, confirm branch and cleanliness:

    git status --short --branch

Expected branch prefix:

    ## feat/gemma4-support

Run the dependency probe through the train package environment. Store the diagnostic script in a fresh `mktemp -d` directory and keep any Hugging Face token only in the environment. The exact probe command must be added here after the script exists; do not commit the script unless it becomes a reusable test helper.

After the checkpoint-owned topology milestone, run:

    uv run pytest packages/common/tests/test_model_variants.py packages/common/tests/test_contracts.py packages/common/tests/test_catalog.py -q
    uv run pytest packages/train/tests/test_api.py packages/train/tests/test_transform.py -q
    uv run pytest apps/lab/tests/test_catalog.py apps/lab/tests/test_work_packages.py -q

After the renderer and TRL milestone, run:

    uv run pytest packages/common/tests/test_model_chat_templates.py packages/train/tests/test_rendering.py packages/train/tests/test_trl_common.py packages/serve/tests/test_online.py packages/serve/tests/test_api.py -q

The tokenizer-dependent tests should pass when `google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` is cached and otherwise skip with that exact cache-miss reason. Factory and catalog tests must always run offline.

Validate the declarative Lab work packages:

    uv run --package posttrain-lab posttrain --project-root apps/lab work-package validate gemma4_unified_serve_smoke_qualification.yaml
    uv run --package posttrain-lab posttrain --project-root apps/lab work-package validate gemma4_unified_sft_qualification.yaml
    uv run --package posttrain-lab posttrain-lab qualification list --project-root apps/lab --json

The first two commands must report valid work packages. The qualification listing must contain each new YAML exactly once and show candidate state before real evidence is accepted.

Run the focused owning-package suites:

    uv run pytest packages/common/tests packages/catalog/tests packages/train/tests packages/serve/tests packages/jobs/tests apps/lab/tests -q

Then run the normal repository ladder:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Do not run `uv sync` with an unlocked dependency resolution unless the plan has been amended for a dependency change. Expected acceptance is zero lint/type/import-boundary errors, all non-environmental tests passing, only clearly explained network/GPU/cache skips, and no whitespace errors.

The successful submission commands use the target already pinned by each binding; the CLI rejects repeating it as a no-op override:

    uv run --package posttrain-lab posttrain --project-root apps/lab --json job run gemma4_unified_serve_smoke_qualification.yaml --job smoke --provider dstack --env HF_TOKEN
    uv run --package posttrain-lab posttrain --project-root apps/lab --json job run gemma4_unified_sft_qualification.yaml --job train --provider dstack --env HF_TOKEN

For the paired-assistant MTP qualification, use the explicit deferred-environment
waiver only because the GSM8K package performs a network-backed dataset metadata
check during the offline image smoke stage. The worker still loads the pinned
environment package and records its revision in Trackio:

    uv run --package posttrain-lab posttrain --project-root apps/lab --json job run gemma4_unified_grpo_mtp_qualification.yaml --job grpo --provider dstack --allow-deferred-qualification

## Validation and Acceptance

Contract acceptance is met when existing model contracts and catalog rows require no new architecture field, `models/gemma4-12b-it@bf16` resolves to the exact immutable identity in this plan, and its provenance records the upstream model type and class. Loader tests must show that Gemma uses the multimodal auto factory while existing Qwen and LFM behavior remains causal-model based.

Renderer acceptance is met when ordinary, thinking-enabled, tool-call, tool-response, and multi-turn examples render with the pinned tokenizer; assistant-only SFT labels are nonempty; no user/system/tool-response token receives loss unless explicitly represented as a trainable assistant message; and the training binding selects reasoning mode `off`. Token-level tests must check boundaries and masks rather than comparing only decoded prose.

Training acceptance is met when Qwen and LFM tests remain unchanged in behavior, Gemma dispatches through `AutoModelForMultimodalLM`, LoRA touches only proven language-model projections, two real SFT updates produce finite evidence, and a PEFT adapter can be saved, materialized, and reloaded. Merely constructing a config does not count as training support.

Serving acceptance is met when vLLM 0.25.1 launches the pinned foundation weights on the Lab target under the text-only 8192-token profile, `/health` is successful, `/v1/models` exposes the selected model, a streamed chat completion returns nonempty final content, and a tool-bearing request is parsed into structured tool calls with parser `gemma4`. A health-only probe does not count as the final serving proof.

Scope acceptance is equally important: the diff must contain no Gemma tower-dense or MoE architecture values, no 31B model, no multimodal dataset path, no DPO/distillation selection, no QLoRA/full update, no quantization, no standalone Gemma MTP serving claim, no new dependency import from common into an ML backend, and no model-family conditional in `packages/jobs` or `apps/lab` orchestration. The single Gemma GRPO selection is a bounded qualification only.

The implementation is complete only after focused and full validation pass and both real qualification results are recorded. If adapter serving is unsupported by the pinned vLLM stack, the plan must explicitly narrow the final outcome before completion rather than implying the trained candidate passed a serve-smoke gate.

## Idempotence and Recovery

All catalog, contract, and test edits are additive and may be rerun safely. The dependency probe uses immutable revisions and a temporary directory; retrying it should reuse the Hugging Face cache without changing repository state. If a download is interrupted, rerun against the same revision. Never delete the shared cache as a recovery step.

GPU work packages are safe to retry because job and artifact identities are versioned. Before retrying, inspect the prior run and preserve its failure evidence. Use a new run attempt rather than overwriting tracked artifacts. Stop and reconcile a partially uploaded adapter through the normal job lifecycle; do not manually rename provider objects into success.

If a dependency change becomes necessary, pause implementation and revise this ExecPlan first. Name every repository and file affected, pin immutable commits or compatible package ranges as required by `AGENTS.md`, update `uv.lock` and relevant runtime dependency locks, and validate the producer before its consumer. Do not mix an uncommitted sibling-repository change into this branch.

Unrelated dirty work belongs to the user. Do not revert it. Before each commit, inspect `git status --short` and `git diff --check`; stage only files named by this plan or subsequently recorded here.

## Artifacts and Notes

Known immutable inputs at plan creation:

    branch: feat/gemma4-support
    model: google/gemma-4-12B-it
    model revision: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    MTP assistant: google/gemma-4-12B-it-assistant
    MTP assistant revision: 364bd03c9952e5b7da73665ee30c9eccfc408345
    transformers architecture: Gemma4UnifiedForConditionalGeneration
    model type: gemma4_unified
    total parameters: 11,959,730,224
    transformers: 5.14.1
    vllm: 0.25.1
    renderers: 0.1.8
    TRL fork revision: 6e7739b8ec741d21ecd79c0c212694cd15ff20d8
    Lab target: targets/carbonteq-rtx-pro-6000-96gb

Qualification submissions on 2026-08-05:

    serving run: 96152bfa-08c3-4bdb-bd07-286f1ec0b8e9
    serving dstack provider id: pt-e4558f9a1b3e3b006259d4cd
    serving job image: registry.lan/carbonteq/posttrain-job@sha256:a5810b7134472b7846103c6ebc49f1e1ecb5f4bcc6bf8b8f69f414495fc575ed
    serving result: succeeded; generated-output and server-log artifacts recorded in Trackio
    serving GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 102641958912 bytes

    SFT run: 6b04480c-0df5-4ec6-ba00-3201bd9953e3
    SFT dstack provider id: pt-dcbe24384c8b212565cad89b
    SFT job image: registry.lan/carbonteq/posttrain-job@sha256:e0eff255602a176fb82b63e90b82ec3bda8cb0bdfdf05a1b751b5844c43dea62
    SFT result: succeeded on attempt 1; requested target targets/carbonteq-rtx-pro-6000-96gb@1
    SFT runtime: 2 updates; train runtime 1.452 s; aggregate train loss 11.49
    SFT step losses: 10.477238655090332, 12.507713317871094
    SFT gradient norms: 20.59836196899414, 24.06705093383789
    SFT peak throughput: 92.09031549222547 non-padding tokens/s
    SFT parameters: 32,784,384 trainable / 11,992,514,560 total (0.2733737269%)
    SFT peak allocated GPU memory: 28.37083673477173 GiB
    SFT adapter: trackio/posttrain-lab/training-models-gemma4-12b-it-bf16-sft-lora-adapter:v0
    SFT adapter artifact digest: 36f0dd670efca07674dbc4fb835f2ffc24f58f8bb3088e277f6985215bd39088
    SFT adapter content digest: dfc511859f94a6c98a2d3a1d7552699c7e471a2ae8ba78999e999ce86af01796
    SFT recovery checkpoint digest: 7c31be77d895c85a17a97790aee5add5a8a860b2f2af5eb67af9cf6cd1f1c99e
    SFT summary digest: d9bb2842754dd4b26aa516afe9568e97db6461a1db8d0a124dbba70118c73481

    clean PEFT reload dstack task: gemma4-adapter-reload-validation-v3
    clean PEFT reload result: passed; finite logits; one generated token
    clean PEFT reload model classes: Gemma4UnifiedForConditionalGeneration -> PeftModelForCausalLM
    clean PEFT reload peak allocated GPU memory: 22.5210223197937 GiB

    structured tool and adapter-serving dstack task: gemma4-serve-contract-validation-v3
    structured tool result: passed; get_weather({"city":"Paris"}); finish_reason=tool_calls
    adapter serving result: passed; exact retained content digest; nonempty final content; finish_reason=stop

    MTP GRPO run: 6579a33b-d78e-4162-a38e-8371a93b2351
    MTP dstack provider id: pt-b96982ab638a77323412711d
    MTP job image: registry.lan/carbonteq/posttrain-job@sha256:01168917f1321b4be16115729a22a0dddee93b3802abf4c8612f74696edb7096
    MTP result: succeeded; two rollouts completed; one optimizer step; two Verifiers traces
    MTP counters: 135 draft tokens, 120 accepted tokens, acceptance rate 0.8889, accepted length 1.889
    MTP KV evidence: capacity 22,080 tokens; peak usage ratio 0.02302
    MTP runtime: 84.78 s; rollout time 82.21 s; peak GPU memory 54.13 GiB
    MTP caveat: both 128-token completions were truncated, reward and gradient were zero; this is an execution/telemetry proof only

    MTP extended retry: c8cf775c-9ebe-415a-9be0-68c5c3bed66e
    MTP extended retry image: registry.lan/carbonteq/posttrain-job@sha256:f5fe84a0663ad94becfea7e52494d9bd2416eb481138a0edcafedca0050a4a01
    MTP extended retry result: failed in the Verifiers harness with `HarnessError: harness setup timed out` before a trainable branch; no CUDA/vLLM error observed

Add the following evidence during milestone 1: tokenizer fingerprint inputs and digest; response-template behavior with thinking off/on; decoded ordinary and tool conversations; assistant loss-mask spans; selected LoRA module names and count; excluded multimodal module names; auto-model class resolution; and token-only forward signature result.

Add the following evidence during milestone 6: serving run ID and image digest; SFT run ID and image digest; resolved catalog snapshot digests; GPU model/VRAM; startup duration; peak allocated/reserved memory; generated answer summary; structured tool-call summary; optimizer update count; initial/final loss; gradient norm samples; trainable/total parameters; adapter artifact URI/version/digest; and adapter reload or adapter-serving result.

Local validation evidence at the implementation stopping point:

    uv sync --all-packages --locked --python 3.13
    Resolved 286 packages

    uv run ruff check .
    All checks passed!

    uv run lint-imports
    Contracts: 8 kept, 0 broken.

    uv run pyright <all ten changed production modules>
    0 errors, 0 warnings, 0 informations

    uv run pytest packages/common/tests packages/catalog/tests packages/train/tests packages/serve/tests packages/jobs/tests apps/lab/tests -q
    328 passed, 11 skipped

    uv run pytest
    1042 passed, 19 skipped, 4 warnings

    uv run pyright <all ten Gemma production modules>
    0 errors, 0 warnings, 0 informations

    uv run --package posttrain-train --extra trl pytest packages/common/tests/test_model_chat_templates.py packages/train/tests/test_rendering.py -q -rs
    3 passed, 6 skipped (all skips are uncached Qwen/LFM tokenizers; Gemma cases passed)

    posttrain work-package validate gemma4_unified_serve_smoke_qualification.yaml
    Work package composition valid: screen/gemma4-12b-it/generation-smoke-qualification

    posttrain work-package validate gemma4_unified_sft_qualification.yaml
    Work package composition valid: train/gemma4-12b-it/sft-qualification

Do not paste secrets, complete model outputs containing sensitive prompts, full dependency logs, or large traces here. Store authoritative evidence through the configured observer and include only concise identifiers and acceptance excerpts.

## Interfaces and Dependencies

At the end of the implementation, `packages/common/src/posttrain/common/models.py` must retain the existing family-only compatibility surface. Concrete upstream model type and class live in the exact Gemma variant's provenance and in the pinned checkpoint configuration; they are not required `ModelVariant` fields.

`ToolCallProtocol.id` must include `gemma4_structured`. `packages/common/src/posttrain/common/variants/gemma4.py` must expose:

    GEMMA4_RENDERER_CONTRACT: RendererContract
    GEMMA_4_12B_IT: ModelVariant

`packages/train/src/posttrain/train/profiles.py` must reuse the existing default renderer implementation and expose:

    GEMMA4_RENDERER = TrainingRenderer(
        id="gemma4-off-v1",
        model_family="gemma4",
        implementation="default",
        reasoning_mode="off",
    )

`packages/train/src/posttrain/train/backends/trl/common.py` must contain one small family-dispatch interface. Its exact private name may follow local style, but its behavior is fixed:

    def trainable_model_factory(model: ModelVariant, imports: dict[str, Any]) -> Any:
        if model.family == "gemma4":
            return imports["AutoModelForMultimodalLM"]
        return imports["AutoModelForCausalLM"]

No new direct dependency is expected. The implementation must use the currently locked Transformers 5.14.1, PEFT 0.19.x, CarbonTeq TRL fork revision `6e7739b8ec741d21ecd79c0c212694cd15ff20d8`, renderers 0.1.8, and vLLM 0.25.1 if milestone 1 passes. Framework packages remain independent: common imports no ML backend; train does not import serve or Lab; serve does not import train or Lab; standard jobs remain model-neutral; and Lab composes the concrete target and evidence path.

Revision note (2026-08-04): Created the initial implementation-ready plan after repository and dependency-surface research. It fixes the scope to Gemma 4 Unified 12B, uses a hard dependency probe before production changes, separates reusable framework support from Lab-only 96 GiB qualification policy, and normalizes online renderer kwargs through vLLM's nested `chat_template_kwargs` contract. Updated after the feasibility gate to record the exact tokenizer digest, model/PEFT projection evidence, and the decision to use `DefaultRenderer` only for the bounded SFT plane while reserving typed-renderer work for future online training. Updated again after implementation with completed milestones, local validation evidence, the full-Pyright wrapper limitation, and the remaining real-GPU qualification work. The initial required architecture-axis decision was subsequently superseded: immutable checkpoint metadata and backend auto-resolution own concrete topology, while posttrain retains family-level contracts and upstream provenance.

Revision note (2026-08-05): The user expanded the support request to TRL-based Gemma MTP. The plan now treats MTP as rollout-only paired-assistant capability, pins `google/gemma-4-12B-it-assistant` by full revision, resolves it into the worker cache before TRL/vLLM construction, and adds one bounded Gemma GRPO qualification. The prior SFT-only `mtp=False` decision is retained as historical context and superseded for this implementation slice.

Revision note (2026-08-05, MTP qualification): The paired-assistant contract and TRL adapter are implemented. The first RTX PRO run succeeded with nonzero speculative counters and complete rollout/optimizer lifecycle, but its deliberately small 128-token cap truncated both GSM8K completions and therefore cannot support a learning-quality claim. A 512-token retry was submitted with a 4096-token engine context and failed at the Verifiers harness setup timeout before a trainable branch; it did not emit a CUDA/vLLM error. The plan retains the successful bounded execution proof and records the non-truncated quality retry as follow-up work. The GSM8K environment is marked deferred because its package performs a network-backed dataset metadata check during offline image qualification; the run used the explicit deferred waiver and immutable Verifiers revision.

Revision note (2026-08-05, scope superseded): E2B, E4B, 12B, and 31B are now
covered by `docs/plan/gemma4-0.3.2-support-and-release.md`. The corrected 12B
profile uses training `max_length=1024`; run
`gemma4-trl-mtp-qualification-4` completed two non-truncated traces with reward
and MTP acceptance telemetry. The old 128-token and setup-timeout runs remain
diagnostic evidence only.
