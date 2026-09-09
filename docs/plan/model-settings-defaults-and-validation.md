# Job, algorithm, and model settings: explicit defaults and early validation


Revision 4 — 2026-09-09. The preceding async/local-qualification work is now
committed independently at Posttrain `68f1c812` and TRL `278af512`. The local
setup installs every workspace package and overlays the intended TRL,
Verifiers, and AutomationBench source checkouts, with import-origin checks for
the packages used by this plan. The focused configuration/catalog suite passes
in that environment. Implementation can proceed without mixing the two change
sets or relying on ad-hoc `PYTHONPATH` entries.

Revision 5 — 2026-09-09. M1 and the static portion of M2 are implemented:
owner-specific serving and training resolvers preserve explicit values and
origins, prepared jobs carry a digest-bound validation report, and declared
hardware facts produce deterministic compatibility findings without probing a
machine. vLLM server and benchmark translation now consume the one resolved
serving configuration. `job plan --explain` renders the report, and the
framework catalog supplies explicit RTX PRO 6000, H100, and H200 profiles.
The bounded local GPU qualification passed one source-composed optimizer update
with 27 parameter tensors changed. `--skip-preflight`, live serving identity
verification, and full backend/model-family GPU qualification remain release
gates rather than being represented as no-op flags.

Revision 6 — 2026-09-09. M4 is complete. `--skip-preflight` now flows through
the provider-free project/job planning seam. Static validation always runs; a
host may supply only optional bounded readiness checks, which are recorded as
passed or skipped in the digest-bound report. Hosts with no readiness probe
record `not_applicable`, so the flag cannot falsely claim that an unperformed
check passed or weaken runtime guards.

Revision 3 — 2026-09-09. Implementation started. M0 added the canonical API amendment and `docs/model-settings.md` inventory, now expanded to job-type and algorithm settings. Defaults, hard validity, and efficiency advice remain separate at each layer. Hardware-aware recommendations cover architecture, residency, MTP, and TurboQuant independently of full-weight/LoRA/QLoRA selection. No new LoRA-rank default is introduced.

## Purpose / Big Picture


A developer should select a model and its intended use, inspect the complete effective configuration, and discover avoidable incompatibilities before building an image or reserving a GPU. The same configuration should retain its meaning in serving, evaluation, training, teacher scoring, and LLM judging, locally or on a remote worker. Different model roles may deliberately use different settings.

This work separates two questions that must never be conflated. **Defaults** answer “what happens when I omit this setting?” **Validation** answers “is this explicit or resolved combination supported and internally consistent?” A value can differ from the recommended default and still be perfectly valid. Conversely, a default can become invalid when combined with another selection; the resolver must explain the conflict, not silently change the user's selection.

The outcome is an inspectable model-configuration resolution and validation path using existing Posttrain selections and adapter boundaries. It is not a new universal model mega-config. The developer can run `posttrain job plan ... --explain` to see values, origins, compatibility findings, and checks deferred until a runtime exists. Job submission runs additional bounded readiness checks by default. A narrowly scoped `--skip-preflight` option bypasses those additional checks, not schemas, immutable identity, known-incompatible combinations, or runtime correctness guards.

## Progress


- [x] (2026-09-09) Corrected scope after user clarification: model configuration across roles and operations, not async rollout scheduling.
- [x] (2026-09-09) Inspected current model contracts, family renderers, inference schemas, serving and trainer translation, standard-job validators, project planning, and canonical ownership rules.
- [x] (2026-09-09) Wrote separate default decisions, validity rules, runtime verification, developer workflows, migration, and acceptance gates.
- [x] (2026-09-09) Extended the plan for RTX PRO 6000/H100/H200-class hardware, model architecture, phase residency, training method/rank visibility, and independent MTP/TurboQuant eligibility.
- [x] (2026-09-09) M0: added `docs/model-settings.md` and amended the canonical API with explain, bypass, per-use reasoning, hardware-advice, and independent acceleration semantics.
- [x] M1: implement effective-setting resolution and provenance without changing existing selections.
- [x] M2: implement uniform static validation and error reporting across model roles and entry points; dynamic readiness remains M3.
- [ ] M3: align native adapter translation, generation settings, rendering, and runtime verification (serving translation is complete; live identity verification remains).
- [x] M4: expose explainable CLI/Python behavior and the bounded preflight bypass.
- [ ] M5: migrate versioned catalog defaults and qualify supported model/backend paths (hardware profiles and one bounded local training qualification complete; serving/model-family gates remain).

## Scope and explicit exclusions


In scope are job-type, algorithm, model, inference, training-binding, and hardware settings. This includes required role topology and lifecycle; algorithm population, sequence, objective, reward, clipping, KL and schedule invariants; model artifact and revision; base/adapter relationships; architecture and modalities; tokenizer and processor identity; chat templates and special tokens; thinking modes; tool-call protocols and parsers; storage, compute and KV-cache precision; persistent versus load-time quantization; context and generation budgets; sampling, stop, seed and structured-output controls; model-loading and attention options; engine limits, parallelism and memory settings; speculative decoding including MTP and DSpark; model-specific training compatibility, adapters, and frozen model roles. Hardware-aware recommendations use the selected topology, model residency by phase, workload bounds, full-weight/LoRA/QLoRA selection and existing rank. They do not change those choices or implement a new scheduler. The inventory includes backend-native options already exposed through namespaced configuration, not just portable fields.

Dataset/task diversity, semantic reward-rubric design, prompt scheduling implementation, async policy staleness, environment subprocess architecture, provider queueing, job cleanup, and performance instrumentation are out of scope. Algorithm settings—including learning schedule and reward weights—are now in scope for explicit defaults and validity, but algorithms are not redesigned and the advisor never changes scientific settings automatically. Model-related memory configuration is covered; automatic micro-batch tuning and actor-kernel optimization are not.

The plan covers the framework surface broadly. It does not promise that every model family, modality, quantization method, or acceleration combination will become supported. Unsupported combinations must be represented honestly and rejected or left explicitly unqualified according to the existing qualification policy.

## Context, authority, and working location


The implementation worktree is `/home/hammad/projects/rl-local-async`, branch `codex/local-async-source-env`, with the preceding local-source baseline committed at `68f1c812b21aae6e6d619fe9b351fe34039d3a36`. `/home/hammad/projects/rl` is a different checkout on `codex/pre-rollout-optimization-baseline`; do not edit it by accident. All relative paths below are relative to the local-async worktree unless a different repository is named.

The preceding async qualification changes are no longer dirty: they were tested and committed separately at `68f1c812b21aae6e6d619fe9b351fe34039d3a36`. This plan remains an independent workstream; it does not declare the async plan complete or depend on publishing that unreleased baseline.

Canonical authority is `docs/post-training/README.md`, especially `02-primitives.md`, `03-work-and-evidence.md`, `04-framework.md`, and `05-apis.md`. These establish separate model, inference, training, environment, and execution selections; static detached planning without native ML/environment activation; and runtime validation immediately before the operation. `docs/developer-experience.md` supplies the project-developer golden path. This plan follows `docs/templates/PLAN.md`; the planning skill's optional `.agents/PLAN.md` is absent in the inspected checkout.

**Baseline impact:** clarifying and consistently enforcing existing constraints does not change product meaning. Publishing the new explain/report contracts, preflight-bypass semantics, and any explicit per-use reasoning-selection field does require a narrow amendment to `05-apis.md` before code changes to that public surface. Amend `02-primitives.md` only if the representation of a selection changes, and `04-framework.md` only if ownership changes. Do not silently unfreeze async training or move backend options into common model facts.

No fork change is required merely to resolve and validate configuration. If M3 proves a native library change necessary, first resolve that fork's current branch, dirty state, upstream base, manifest selection, and lock. The inspected TRL candidate is `/home/hammad/projects/trl-async-training` at local committed tip `278af512459b6be6d9510029ad9bbfd7d7faff03`, which is not yet pushed or packaged; Verifiers is `/home/hammad/projects/verifiers` at `36eac9d5e04ef29b584b6fa4f027af00cd76ea19`. These development sources are not equivalent to the released dependency selection: the inspected eval/data/train manifests still select Verifiers `c6c0097ad21da845c62e4b19aba80ef6633e4d9f`. Re-derive every active pin at implementation time. Follow `docs/tooling/forks.md`: generic fix and regression in the owning fork, fork ledger, tests, commit/push and publication, consumer pin/lock, consumer tooling page, then immutable qualification. Do not introduce fork changes speculatively.

## Current architecture and observed gaps


`packages/common/src/posttrain/common/models.py` owns `ModelVariant`, `ModelCapabilities`, `RendererContract`, `ConversationProfile`, `ReasoningMode`, and `ToolCallProtocol`. `ModelVariant` identifies weights and model-interface facts. `packages/common/src/posttrain/common/selections.py::InferenceBinding` identifies a model/backend/target/purpose combination with engine and sampling mappings. `packages/train/src/posttrain/train/bindings.py` owns update plans and training bindings. Keep these ownership boundaries.

`packages/common/src/posttrain/common/variants/{qwen35,lfm25,gemma4,nanbeige42}.py` contains family-specific interface facts and reasoning defaults. The inspected Qwen default renderer selects `off`, while a second renderer selects `thinking`; Gemma's inspected default is `off`, Nanbeige's is `thinking`, and LFM has native-only modes in its current renderer contracts. These are repository selections, not a claim about every upstream checkpoint in each family. Never infer a switchable thinking mode from a model name.

`packages/serve/src/posttrain/serve/profiles/base.py` contains `VllmEngineConfig` and `VllmSamplingConfig`. Their defaults include FP16 compute and greedy sampling. `packages/train/src/posttrain/train/backends/trl/common.py::load_trainable_model` defaults to BF16 and SDPA and rejects model dtypes outside its implemented set. Its tokenizer loader chooses the base tokenizer, sets padding side, and falls back from a missing pad token to EOS. These choices need to be visible and validated; difference alone does not prove a bug because actor and inference precision may legitimately differ.

`packages/serve/src/posttrain/serve/backends/vllm/bindings.py` separates `engine_config`, `sampling_config`, and `frontend_args`. Tool/reasoning parser values are removed from the engine mapping and handled separately. Tool parser conflicts are checked in `frontend_args`, not by constructing the common inference binding alone. A Nanbeige/DSpark/TurboQuant incompatibility is already scoped to an exact backend revision. Preserve its precision instead of generalizing it to all versions.

The serving sampling dataclass, `packages/environment/src/posttrain/environment/requests.py::SamplingPolicy`, `packages/train/src/posttrain/train/online_rl.py::PolicySampling`, and request-specific options expose overlapping but non-identical fields and validation. Some numeric checks accept booleans or fail to reject non-finite values consistently. This needs a field-by-field audit, not blanket coercion or one schema forced onto every operation.

`packages/train/src/posttrain/train/backends/trl/policy_config.py` and `packages/train/src/posttrain/train/backends/verl/worker.py` separately translate model/runtime choices. veRL currently derives rollout dtype and context defaults at translation time. Model identity, token limits, reasoning mode, and precision must be resolved before this translation rather than acquire invisible adapter-specific changes.

`packages/project/src/posttrain/project/service.py::JobService.plan` uses `packages/work/src/posttrain/work/runner.py::prepare_work_package_job`. Standard definitions use `JobDefinition.static_validator` in `packages/jobs/src/posttrain/jobs/definitions.py`. `_validate_online_rl_batch_seats` already checks GRPO sampling/context relationships. Equivalent early coverage is not wired to every standard model-consuming definition. Reuse these extension points; do not add CLI-only correctness checks or another job launcher.

`apps/cli/src/posttrain_cli/commands/work_package.py::plan_work_package_cmd` now resolves provider-free job intent. `apps/cli/src/posttrain_cli/execution_planning.py` handles packaging and launch decisions separately. Preserve this split: explanation must not unexpectedly build, download weights, contact a GPU, or submit a job.

## Decision A: defaults, independently of validity


### Resolution rules


Resolve settings within their owning seat. A project overlay may replace a catalog selection as today; it is not an implicit deep merge of unrelated mappings. An explicit supported per-request override has precedence over that role's inference defaults only where the operation allows overrides. Training-owned sampling constraints are checked against the resolved result rather than arbitrarily winning a merge. Conflicting values at authoritative seats are errors with both origins displayed.

Apply defaults only to omitted fields. Explicit `false`, `0`, an empty sequence, and `null` are not interchangeable with omission. For a nullable field, `null` has its documented meaning, such as disabling an optional feature or delegating to a pinned backend policy. Otherwise it is invalid. Do not use truthiness-based fallback for numeric or boolean model settings. Resolve recognized aliases once, reject conflicting aliases, and retain the supplied spelling in diagnostic provenance.

The default policy is versioned by the selected binding/recipe and framework release. It does not float with a mutable model card, Hub `generation_config.json`, runtime auto-detection, or environment variables. Pinned artifact generation defaults may be imported deliberately at authoring time, with their revision and field origins recorded; backend auto-loading must not become an undocumented second default layer.

Every effective setting has an origin: explicit selection, allowed request override, family contract, versioned purpose default, or qualified adapter default. Hardware-resolved choices have a deferred rule before launch and a concrete observed value at runtime. Never claim a concrete dtype, attention kernel, or available-memory value before it has actually been resolved.

### Default decisions by setting family


The following are selected policies for new or newly versioned bindings. Existing immutable bindings are not silently rewritten. Exact numeric defaults that depend on workload or target are deliberately required inputs or named binding choices, not universal guesses.

| Setting family | Default policy when omitted | What does not follow from that default |
| --- | --- | --- |
| Model identity and revision | Required exact artifact/base identity; no automatic model or revision selection | A missing revision is not permission to use `main` |
| Tokenizer/processor | Inherit the selected artifact's declared compatible tokenizer/processor source; base inheritance is explicit for adapters | A derived checkpoint cannot silently lose modified vocabulary or processor files |
| Template and special tokens | Use the model's versioned renderer/template; preserve declared special-token and stop semantics | No generic template replacement or arbitrary token insertion |
| Thinking mode | Preserve current renderer default for existing bindings; new purpose bindings choose explicitly where the family supports a choice | Thinking is neither globally enabled nor globally disabled; `native` is not synonymous with `off` |
| Tool calling | Derive the backend parser from the selected qualified protocol when tool calling is required | A parser setting cannot grant capability the model/backend lacks |
| Weight precision | Use the selected artifact representation unchanged | No automatic quantization to make a model fit |
| Compute dtype | New bindings prefer artifact-compatible BF16 for BF16 weights on a qualified BF16 runtime; otherwise require an explicit supported choice | Do not silently convert to FP16, enable mixed precision, or reject an explicitly supported alternative |
| KV-cache dtype | Recommend TurboQuant K8V4 independently of the training update method; enable by default in new bindings only for qualified model/backend/hardware/use combinations, otherwise retain qualified native KV with a visible eligibility explanation | TurboQuant is not LoRA or weight quantization; recommendation is not proof of compatibility or quality |
| Context and output limits | Require bounded limits from the purpose binding/workload; derive context from a known input allowance plus output reserve when both are declared | Native maximum context is a capability ceiling, not the default allocation |
| Generation sampling | Preserve family-qualified, purpose-specific sampling; use neutral controls for omitted optional penalties/stops | Do not install a universal temperature across all reasoning models |
| Reproducibility | Qualification recipes explicitly seed supported sampling paths; snapshots identify unseeded/backend-limited behavior | Temperature zero alone does not guarantee reproducibility |
| Speculative decoding | Enable MTP by default for new bindings whose complete model/backend/hardware/operation combination is qualified; use that binding's explicit qualified draft depth | `mtp=True` alone does not establish serving support, target/draft compatibility, or good throughput; explicit disable remains valid |
| Parallelism | Single-device model parallelism for new single-device bindings; larger topologies selected explicitly | Tensor/expert/context parallelism cannot be inferred merely from available GPUs |
| Memory and engine batch limits | Explicit target/purpose binding values; retain existing settings until revised and qualified | A single global VRAM fraction or sequence cap is not sensible for every topology |
| Attention and compilation | Hardware/model-specific qualified policy, including optimized attention and graph execution where validated; expose the chosen implementation and prerequisites | Do not force one FlashAttention version across architectures or install/build missing kernels as a recovery action |
| Training updates | Require a selected update plan; retain current plan-specific LoRA/QLoRA defaults until independently revised | Never switch full training to LoRA or quantize a source because of a memory estimate |
| Remote code | Disabled; any future permitted path requires explicit reviewed source and execution policy | `--skip-preflight` cannot authorize remote code |

For new general serving/evaluation/judge bindings, select deterministic sampling only when compatible with the family's documented/qualified recommendation; otherwise specify the family recommendation and report that choice. For training rollouts, use the algorithm's supported stochastic default and validate it against family constraints. Existing positive-temperature policy requirements remain in place until a separate algorithm decision changes them. This distinction prevents “greedy is valid for inference” from becoming “greedy is supported for every training algorithm.”

A new reasoning-quality judge binding should explicitly select thinking when the chosen checkpoint supports it, with an adequate declared output reserve. A latency-oriented serving binding may explicitly choose non-thinking. Both are valid. The rubric does not set engine defaults, and the judge does not inherit the policy model's settings accidentally.

EOS remains enabled by default. `ignore_eos`, forced minimum output, custom stop strings, logits processors, and constrained decoding are explicit choices. They may be useful for qualification or structured output, but their semantic effects must appear in the explanation and compatibility checks. Seed propagation and top-k disabling sentinels are translated according to the pinned backend, not assumed identical across engines.

### Hardware-aware defaults and recommendations


Select an effective configuration from **model architecture + exact backend/runtime + hardware profile + operation/role + workload + training update plan**, not the GPU name alone. Hardware capability is a necessary input, not permission to enable every advertised optimization. Explicit valid overrides still win; existing versioned bindings retain their meaning. Automatic choices are documented defaults for omitted settings, never a rewrite of explicit input.

Use distinct hardware identities. RTX A6000 is not RTX PRO 6000 Blackwell, and RTX 6000 Ada is another product. The selected profile must retain SKU/edition, architecture/compute capability, usable memory, GPU count, partition/MIG constraints when applicable, interconnect, and known runtime requirements. A generic `nvidia-cuda` target with only memory capacity cannot justify architecture-specific kernel selection. Unknown facts produce a deferred check or a conservative qualified default, not an invented capability. Workstation display/other-process headroom and server-exclusive headroom may differ.

| Hardware family | Candidate defaults/recommendations to qualify | Limits on automatic selection |
| --- | --- | --- |
| RTX PRO 6000 Blackwell | Blackwell-compatible optimized kernels and graph policy; appropriate engine batch/token limits; qualified MTP and TurboQuant; BF16 baseline or explicitly selected supported lower precision | Do not infer a datacenter Blackwell kernel, NVLink fabric, or full unpartitioned device from the family name |
| H100 | Hopper-compatible optimized attention and graph policy; BF16 baseline; qualified FP8 execution as a separately selected precision path; evaluate available parallelism and engine capacity | H100 variants differ in memory/form factor; FP8 hardware support does not authorize silently quantizing BF16 weights |
| H200 | Hopper-compatible kernel policy with H200-specific memory/bandwidth estimates; warn about inherited low capacity caps or avoidable offload | More memory does not require larger task budgets, altered learning batches, or full-weight training |
| RTX A6000 / other older GPUs | Architecture-compatible precision and kernels; the same independent MTP/TurboQuant eligibility assessment | Never inherit Hopper/Blackwell-only defaults from a similar product name |

The initial optimized binding candidates should assess graph execution, chunked prefill, attention backend, prefix caching, token batching, and precision independently. Prefix caching for changing policy weights requires a qualified invalidation path; more hardware does not make stale cache reuse correct. Head dimension, attention pattern, hybrid recurrent state, expert layout, quantized cache format, and speculative method constrain the valid kernel set. If eager execution or CPU offload is explicitly selected, retain it and warn about a potentially better qualified alternative instead of overriding it.

MTP defaults apply to qualifying generation roles, not teacher-forced scoring or every reference-model operation. Resolve the required head or paired assistant from pinned provenance, include its memory and verification cost, and check native weight-update compatibility when the policy is trained. Qualify MTP plus TurboQuant as a combination; independent passes do not prove composability. Draft depth remains a qualified binding value, not a quantity derived from GPU size. A high-concurrency workload may receive a throughput warning despite MTP being supported. The current vLLM documentation positions speculative decoding particularly for memory-bound, medium-to-low request rates; support is not a universal speedup guarantee. [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

TurboQuant K8V4 is a KV-cache recommendation for every eligible generation binding, regardless of full-weight, LoRA, QLoRA, or inference-only use. Always report its assessment: `enabled`, `recommended_available`, `not_applicable`, `unsupported`, or `qualification_blocked`, with a reason. No KV cache or no supported attention/cache path means not applicable or unsupported, not “requires LoRA.” Existing explicit native-KV settings remain valid and receive a recommendation where appropriate. For new optimized bindings, qualified TurboQuant is the preferred default. If quality or compatibility evidence is missing or adverse, leave it disabled with an explicit gate; do not silently turn a recommendation into a release-qualified default.

The current `docs/tooling/trl/README.md` records failed Qwen3.5 K8V4 long-context recall checks while normal KV passed, and keeps combined MTP/K8V4 qualification separate. That is a recorded restriction for the tested configurations, not proof that every future Qwen/backend/GPU combination fails. M0 must reconcile exact evidence identities before promoting a new default. Merely moving that model to H100/H200 does not clear the quality gate. TurboQuant K8V4 describes cache precision, not adapter rank; the user's correction expressly rules out bundling this recommendation with LoRA rank 8.

### Memory reservation, useful work, and sequential execution


An engine setting of `gpu_memory_utilization: 0.90` is a memory reservation/budget, not 90% compute utilization. vLLM defines it per engine instance; it does not coordinate independent engines on one GPU. The advisor must distinguish configured reservation, estimated useful memory demand, observed allocation, compute utilization, and workload throughput. Before a run, only estimates and known configuration bottlenecks are available; report “possible underutilization,” not a measured utilization percentage. [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)

Compute a phase-aware estimate: loading/warmup, generation, judging/teacher scoring, optimization, and checkpointing, using only phases the job actually declares. For each phase/device, account for resident weights, trainable/frozen parameters, gradients, optimizer/master states, adapters, KV/recurrent state, draft models, activations, graph/workspace buffers, and headroom. Show ranges and unknown terms. Count shared allocations once only when sharing is established; do not assume actor and sampler share weights merely because they use the same checkpoint.

For full-weight training, include state associated with all trainable parameters under the selected optimizer and sharding/offload policy. For LoRA/QLoRA, include the frozen base plus adapter training state and activations; do not estimate total memory from adapter rank alone. For an adapted matrix of dimensions `d_in` and `d_out`, ordinary LoRA contributes approximately `rank * (d_in + d_out)` adapter parameters; sum over actual target modules and account for implementation-specific extras. Display rank, alpha, target modules, and estimated trainable fraction. The user has not selected a new LoRA/rank default in this revision. Spare VRAM is not evidence that increasing rank or switching to full-weight training improves the task.

Architecture metadata must drive estimates. Dense attention KV depends on KV heads and head dimensions, not simply total model parameters; grouped/multi-query attention differs from full multi-head attention. Sliding-window layers and hybrid recurrent/state-space layers need their own state accounting. For MoE models, distinguish total resident expert weights from active compute per token, and include expert placement/communication. Multimodal encoders and input-dependent activation costs cannot be inferred from the language model parameter count. An unsupported estimate is `unknown` with a reason, not the dense formula applied anyway.

Show three separate notions of “sequential”: models executed in non-overlapping phases, inference request concurrency constrained to one, and several selected GPUs with only one assigned useful work. The advisor reports what is actually selected rather than assuming which meaning the user intended. Model residency includes devices, intervals, dependencies, and whether independent roles overlap. A judge that consumes a completed trajectory is not independent of that trajectory's generation, although other trajectories might overlap where the existing execution mode supports it. Selecting H200 does not itself authorize overlapping model phases or async training.

Warn when retained evidence or declared bounds indicate an avoidable constraint: one inference slot despite independent ready requests; a small engine sequence/token cap on a larger selected device; a cache reservation much larger than bounded demand; unnecessary offload when a conservative phase estimate fits; forced eager mode despite a qualified graph path; or unused selected devices. Suggest smaller hardware, a different versioned execution binding, or capacity settings as alternatives. Do not automatically increase training group size, sequence budget, LoRA rank, parallelism, or cloud spend. Hardware-aware advice can recommend concurrency without implementing rollout scheduling.

Assess shared-device budgets jointly. Two concurrently resident engines each reserving 90% of the same device are not independent valid reservations. A provable allocation conflict is an error; uncertain peak memory is a warning/deferred probe. Sequential phases still count any actor/optimizer allocation that remains resident. A budget of 90% is an explicit candidate for a dedicated inference profile only after non-engine headroom fits; it is not a default applied separately to every actor, judge, and reference engine.

### Architecture-aware warning contract


Each recommendation or warning includes a stable code, affected role/phase/device, selected values, architecture/backend rule, evidence or estimation basis, confidence, proposed alternative, and whether the change alters model numerics or training semantics. The default resolver applies qualified omission defaults; the advisor only recommends changes to already selected explicit settings. Unsupported kernels or contradictory identity/budget constraints are errors. Poor expected efficiency is a warning. Missing capability information is deferred. Known failed quality qualification is a release gate, not a claim of syntactically invalid configuration.

Proposed warning codes include `HARDWARE_PROFILE_INCOMPLETE`, `POSSIBLE_ENGINE_UNDERUTILIZATION`, `EXCESS_CACHE_RESERVATION`, `UNUSED_SELECTED_GPU`, `SEQUENTIAL_RESIDENCY_OPPORTUNITY`, `OFFLOAD_POSSIBLY_UNNECESSARY`, `MTP_THROUGHPUT_UNQUALIFIED`, and `TURBOQUANT_QUALIFICATION_BLOCKED`. The report must never invent exact tokens/second or GPU compute utilization from capacity alone.

Hardware references checked while revising this plan: NVIDIA distinguishes RTX A6000's 48 GB from RTX PRO 6000 Blackwell's 96 GB; its H100 specifications distinguish variants, and H200 lists 141 GB. These facts motivate exact SKU profiles, not universal hardcoded budgets. Runtime visible memory/partition constraints remain authoritative. [RTX A6000](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/), [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/), [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/)

## Decision B: invalid configurations, independently of defaults


### Always-enforced static checks


Reject malformed types, unknown owned keys, non-finite numeric values, invalid enum choices, impossible ranges, mutually exclusive options, and conflicting aliases. Python dataclasses and catalog YAML must accept and reject the same logical values. Validate a backend-native option against its selected adapter schema; do not reject a legitimate native option simply because it is not portable. Unrecognized options must not disappear during translation.

Validate artifact/revision consistency, adapter/base compatibility, tokenizer/processor provenance, renderer family, allowed reasoning modes, required modality and tool protocol, backend version, and supported operation. A model name, successful import, or one general capability boolean is insufficient. Missing qualification is distinct from a known incompatibility. Existing `--allow-deferred-qualification` policy remains distinct from the new readiness bypass; neither turns a known-invalid combination into a valid one.

Validate precision and quantization separately: stored weight format, loader format, compute dtype, KV-cache dtype, trainable parameters, and update method. For example, the current TRL loader's unsupported dtype should fail during planning, not after allocation. Keep current rejection of persistent weight-quantized sources for unsupported training paths and of double-quantizing a source via QLoRA. Validate adapter ranks/modules and tokenizer vocabulary compatibility against retained metadata when available; otherwise defer the exact structural check to model loading.

Check context arithmetic only between quantities with the same meaning. A request's serialized prompt, tool schemas, multimodal input cost, retained context, and output reserve must fit the selected engine context. The declared input allowance plus output reserve must fit when both are known. A lower operational context than the model's native ceiling is valid. Exceeding the native ceiling requires an explicitly supported scaling selection; no automatic RoPE scaling. An output reserve larger than the operational context is invalid.

Per-request output limits, total sampled episode output, cumulative input/output billed across turns, training sequence limits, and judge request budgets are different values. Do not equate a Verifiers cumulative token limit with a model context window. Do not claim that a soft episode limit provides a hard maximum for judge input. A judge must budget its own instructions, rubric, retained trace/tool evidence, and response; the trace's generation output alone is not an input-size bound. This plan validates and explains the configured budgets; it does not introduce trace compaction or change task behavior.

Validate draft/target pairing, required assistant artifacts, immutable revisions, tokenizer/protocol constraints, supported speculative method and depth, attention and KV-format compatibility, and backend-specific exclusions. Scope rules to the versions and architectures they actually constrain. For Gemma and other paired-assistant paths, an assistant is not interchangeable with an arbitrary smaller model. DSpark, MTP, and TurboQuant remain inference choices, not training algorithms.

Validate parallelism against known architecture facts and target constraints where the backend imposes those requirements. Do not apply dense-attention divisibility rules blindly to hybrid/MoE models. Validate text-only execution against declared input modalities and processor requirements. A model supporting images can validly be used text-only; disabling image support while selecting an image task is a conflict.

### Valid but potentially poor settings


An unusually small output budget, high temperature, conservative memory utilization, disabled acceleration, or less-than-optimal sequence cap is normally a warning, not an error. It becomes an error only when it contradicts an explicit operation requirement or known backend constraint. Warnings explain risk and possible alternatives without changing values or making submission interactive.

Memory/throughput estimates are not proofs. Do not reject a valid model solely from `parameters × dtype` or assume that a model fitting its weights will fit KV cache, activations, reference/judge models, adapters, workspace, and runtime overhead. Summed budgets matter only for models that actually coexist on the same device. A sequentially loaded judge should not be charged as permanently resident; a colocated judge must not be omitted. Unknown placement or available memory is deferred, not zero.

Uniform judge scores, poor task success, truncation on one example, and low GPU utilization are not automatically invalid model configuration. They are runtime or quality evidence. Preflight proves configuration and readiness, not model intelligence or algorithm quality.

## Validation stages and bypass semantics


**Stage 1: pure resolution and validation.** Runs through `Project.jobs.plan`, `work-package validate`, and before every pack/run path. Uses installed framework metadata and declared selections only. It may import lightweight framework modules, not torch, vLLM, TRL, veRL, Verifiers, or selected model code. It requires no GPU, network, credentials, or model download. Mandatory errors stop before packaging. Unknown runtime facts are explicitly deferred.

**Stage 2: additional model readiness preflight.** Runs by default on `job run` after pure validation and before avoidable build/submission costs. It may inspect available pinned local metadata, cached tokenizer/config manifests, declared source availability, and an explicitly configured existing endpoint's identity/capability metadata. No weight download, paid generation, arbitrary remote-code import, or tool execution is implicit. Network checks use existing authenticated clients, bounded timeouts, and secret-safe reporting. A worker-created endpoint is deferred until that endpoint exists; a missing developer cache is not a configuration failure.

`--skip-preflight` skips only Stage 2 and records exactly which checks were skipped. Proposed CLI help: “Skip additional model readiness probes; schema, compatibility, artifact integrity, and runtime checks still apply.” The option is invocation-scoped, not an environment variable or an inherited project default. It must not alias `--allow-deferred-qualification`, suppress warnings, disable TLS verification, alter resource requests, or bypass required image integrity checks.

**Stage 3: selected-runtime verification.** Validate installed package/source identity, actual artifact/config/tokenizer/processor fingerprints, device and kernel support, native parser availability, resolved auto settings, and endpoint configuration. Check cheap metadata before heavyweight allocation when possible; perform hardware-dependent checks when the actual worker is known. Fail with the same diagnostic codes before real workload traffic. A health response alone does not verify model identity or template semantics. Reused external endpoints need a declared identity contract; unavailable introspection is reported as unverified, never guessed from a model alias.

**Stage 4: per-request and per-load correctness.** Enforce exact serialized request limits and supported request options; retain truthful stop reasons and actual generation/compute configuration. Validate reference and teacher model identities when loading or scoring. These checks cannot be skipped. Unknown future request lengths cannot all be proven statically, especially for multi-turn or multimodal work.

Public direct-library operations retain mandatory validation even when the caller does not use a project or CLI. They do not implicitly acquire network permissions or provision an inference service just to validate a request. Job-level readiness is composition; adapter correctness belongs to the capability package.

## Architecture and concrete interfaces


### Shared facts and diagnostic values


Extend existing model/interface values in `packages/common/src/posttrain/common/models.py` only where genuinely portable facts are missing. Keep engine-native schemas in their owners. Do not encode backend-version exception tables in `ModelCapabilities` or add task/rubric settings to a model.

Proposed new `packages/common/src/posttrain/common/validation.py` contains immutable JSON-serializable `ConfigurationIssue` and `SettingOrigin` values, not a global validator registry. An issue has `code`, `severity`, `stage`, `role`, `path`, `message`, `hint`, and related source paths. No prompts, credentials, signed URLs, or full native exception objects are included. Setting origins identify the source selection/revision and whether a value was explicit, defaulted, derived, or runtime-resolved. Reuse an existing equivalent value if M0 discovers one, and update this document with the final path.

Proposed `packages/work/src/posttrain/work/validation.py` owns `JobValidationReport`: report schema version, resolved-input digest, safe role summaries, setting origins, findings, and check outcomes (`passed`, `failed`, `deferred`, `skipped`, `not_applicable`). Severity and check outcome are different: a deferred hardware check is not a warning proving failure. Stable ordering makes JSON suitable for CI. Timestamps and probe freshness belong to evidence, not semantic job hashes.

Do not place backend logic in this report module. Each owning package exposes pure resolution/validation functions returning shared issues or a typed resolved configuration. Existing `ContractError` remains the public failure family; add a structured subclass carrying findings where necessary, preserving callers that catch it. Aggregate independent errors, but do not manufacture cascades from a seat that could not be parsed.

Hardware facts belong with execution-target contracts and versioned catalog metadata, not model family conditionals in CLI. Add a typed optional hardware-capability descriptor to `ExecutionTarget` and its schema only after the canonical amendment; keep generic targets valid with deferred capability findings. `packages/catalog/src/posttrain/catalog/base/targets.yaml` currently contains generic 8/24GB entries, not a complete SKU inventory. Add exact profile entries without reinterpreting those generic IDs.

Proposed `packages/jobs/src/posttrain/jobs/model_advice.py::assess_model_hardware_configuration(...)` composes owner-provided estimates and declared phase residency into `JobValidationReport` findings. It accepts resolved model roles, target capabilities, workload bounds, update selection, and runtime identity; it does not poll GPUs, launch models, import ML libraries, or schedule work. Train owns training-state estimates; serve owns inference/cache estimates; common owns neutral facts and issue values. Unknown phase residency must be reported, not reconstructed from an algorithm name. Existing `packages/jobs/src/posttrain/jobs/inference_services.py` is the first place to inspect for declared role lifecycle; extend only the description needed by advice, not the lifecycle itself.

### Capability-owned resolvers


In `packages/serve/src/posttrain/serve/backends/vllm/bindings.py`, introduce a pure `resolve_binding_configuration(binding)` result containing the validated engine, sampling, frontend settings, and setting origins. Have `engine_config`, `sampling_config`, and `frontend_args` delegate to it during migration. All supported parser, dtype, context, reasoning, speculative, and generation settings must participate in one resolution. Avoid recursion by extracting private raw parsers. No vLLM import or model download is allowed in this function.

In proposed `packages/train/src/posttrain/train/model_configuration.py`, define `resolve_training_model_configuration(model, training, *, inference=None, role="policy")`. It uses backend-owned lightweight schemas to validate supported loading/update choices, renderer/reasoning configuration, precision and model-role compatibility. It must not import `posttrain.serve`. It returns the selected values that TRL and veRL adapters consume rather than re-deriving defaults in workers. Preserve backend-specific native settings in explicit namespaces.

Training request validation applies common model checks to SFT, DPO, GRPO/DAPO, SAMPO, GDPO, CAPO, and distillation without treating their settings as `GRPOSettings`. Keep algorithm-specific rules in their existing request validators. For token-level teacher scoring, tokenizer compatibility is mandatory; an independently judging LLM is allowed to have a different vocabulary. Reference models may require a stricter same-token-space contract than judges. Enumerate each role from standard definitions, not by assuming the seat is named `model`.

Use existing `ReasoningMode`/`ConversationProfile` to resolve per-use reasoning. During M0, make one explicit public representation decision: add an optional `reasoning_mode` to `InferenceBinding`, defaulting to the renderer's current default, while retaining `TrainingRenderer.reasoning_mode` for actor rendering. Require policy/rollout agreement where exact-token training demands it. This avoids creating a new weight variant merely to switch thinking mode. Preserve existing renderer identities and aliases; do not collapse genuinely different templates or history-stripping behavior into one mode. Record the baseline amendment before adding the field.

In `packages/jobs/src/posttrain/jobs/definitions.py`, attach pure validators for all relevant standard definitions. Composition compares selected roles and required capabilities, then delegates to capability-owned rules. `packages/work/src/posttrain/work/runner.py` aggregates reports through the existing preparation path. Both activated and detached paths must run the mandatory logical checks; an installed `seat_resolver` must not accidentally suppress them. Do not activate environments just to validate a model.

### Native translation and execution


TRL `backends/trl/common.py`, `policy_config.py`, and rendering paths consume resolved values. veRL `backends/verl/worker.py` and its launch manifest consume the equivalent resolution. Serving online and offline paths consume the same serving resolution. Tests compare logical effective settings, not byte-identical native configurations across different engines.

Define a field-consumption check at translation boundaries: every owned selected field is emitted, used in validation/derived output, explicitly unsupported, or documented as inapplicable. Never drop an explicit reasoning parser, sampling penalty, template kwarg, precision override, or generation limit merely because another path has a smaller schema. Conversely, reject duplicate native options that overwrite selected artifact, tokenizer, target, or model-role identity.

Special-token and tokenizer adjustments must be explicit adapter transformations with provenance. A missing pad token may use the existing qualified EOS fallback only when the attention/loss masking path is valid and tested; otherwise reject it. Never resize embeddings or rewrite a chat template silently. Load tokenizer/processor changes from derived artifacts when declared rather than unconditionally assuming the base tokenizer.

M3 must check CLI versus in-process engine parity. In particular, explicit `false` options must be carried through when a pinned backend default is `true`; omitting a flag is not equivalent to setting false. Backend `generation_config` handling must not override the resolved sampling values. Pinned backend automatic attention or dtype selection is allowed only as an explicit recorded policy, followed by observed-resolution evidence.

## Developer experience


Extend existing `job plan` output with a short model-role summary by default and a proposed `--explain` for field origins and deferred checks. Reuse the global `--json` surface. Do not make a new command family merely for validation. `catalog show` should describe supported modes, qualified bindings, and exact identities; `work-package validate` should report all enabled jobs' independent model configuration errors.

Illustrative output after implementation, not a claim that the flags exist today:

    $ posttrain job plan .posttrain/work_packages/model-check.yaml --job check --explain
    Policy: qwen3.5-2b @ 15852e8c... / rollout
      reasoning_mode: off       [renderer default: qwen3.5-tools@1]
      compute dtype: bfloat16   [explicit inference binding]
      context: 12288            [explicit inference binding]
      output reserve: 8192      [explicit operation setting]
      input allowance: 4096     [explicit operation setting]
    Static compatibility: passed
    GPU/kernel and exact tokenizer verification: deferred to selected runtime
    No model downloaded, image built, or run submitted.

An invalid example must name the conflict, not merely print a Python traceback:

    MODEL_CONTEXT_BUDGET_CONFLICT [error]
    rollout_inference.engine.max_model_len = 8192
    settings.max_prompt_length + settings.max_completion_length = 4096 + 8192
    Required capacity is 12288 tokens. Choose a supported larger context or
    explicitly reduce the declared budgets. No settings were changed.

An off-default valid setting reports its origin without an error. An unsupported thinking mode reports supported modes for that exact renderer. An unknown option reports the owning namespace and a suggested spelling only when unambiguous. Dtype diagnostics distinguish checkpoint precision from compute dtype and KV-cache dtype. Do not print private paths, endpoint credentials, or environment-variable contents in hints.

Python uses the same job intent/report as CLI: `Project.open(...).jobs.plan(...)` gains an inspectable report; invalid configurations raise the structured contract error. Direct capability APIs use the same owned validators. The CLI remains thin. Exit codes stay consistent with the canonical contract: zero for valid/planned or successful operation, one for expected contract/readiness failure, two for invalid CLI syntax. Deferred checks alone do not change a valid plan's exit code; they also do not justify printing “runtime ready.”

Model readiness results are digest-bound to model, tokenizer/renderer, backend/runtime identity, selected settings, and known target constraints. Changing any relevant input invalidates reuse. Successful earlier probes do not suppress mandatory runtime guards. Skip choices belong in the launch receipt and run evidence; they must not change learned-model identity or force an otherwise identical image rebuild.

## Plan of Work


### M0 — freeze the inventory and amendment


Create `docs/model-settings.md` as the developer-facing reference and field inventory. For every current portable and backend-native job-type, algorithm, model, inference, and training setting, record owner, consumers/roles, omission semantics, default origin, validity relationships, validation stage, and translation destination. Include standard job role topology; SFT, DPO, GRPO/DAPO/OLMo3, SAMPO, GDPO, CAPO, and distillation settings; `ModelVariant`, renderers, `InferenceBinding`, serving profiles, training bindings/update plans, model loader options, sampling controls, and speculative configuration. Enumerate fields programmatically where possible and review dictionary-backed options manually.

Include exact hardware SKU descriptors, architecture-sensitive estimation inputs, current full-weight/LoRA rank defaults, phase residency, and separate eligibility records for MTP, TurboQuant, and their combination. Map manufacturer capabilities to qualified backend implementations rather than asserting that a Tensor Core feature automatically activates through the framework. Retain current failed/pending quality gates in the default-selection matrix.

Record the per-use reasoning field, report structure, and readiness-bypass amendment in `05-apis.md`. Review catalog recipes and family defaults against pinned model metadata and existing qualification evidence before selecting changed numeric defaults. If external documentation is needed, use official model/backend sources at the selected version and retain exact references in the field inventory. This is not an upgrade task. Acceptance: every in-scope accepted field has an owner/default/validity/stage entry, with no requirement to inspect chat history, and no new settings advertised before their validators exist.

### M1 — effective configuration and provenance


Implement the shared diagnostic values, owner-specific pure resolvers, reasoning selection, and additive report on prepared job intent. Start with a no-behavior-change mode reproducing existing selected values and exposing hidden defaults. Preserve explicit values and compare old/new effective settings in tests. Do not change catalog IDs in this milestone. Acceptance: equivalent YAML/Python selections yield equivalent effective settings and origins; omission differs from false/zero/null; modifying a value changes the semantic digest; explaining a model does not import native ML modules or create a run.

Implement hardware-aware advice as pure composition over the same resolution. Golden reports must display training method/rank, phase residency, selected devices, memory estimate ranges, MTP/TurboQuant eligibility, and whether a throughput recommendation is measured, estimated, or unknown. Preserve explicit off settings while explaining available optimizations.

### M2 — complete early model validation


Harden owned numeric/enum/mapping validation, parser and reasoning checks, identity relationships, context arithmetic, quantization/dtype compatibility, and supported model-role constraints. Wire every standard model-consuming job and direct public operation. Move known late adapter errors earlier without deleting runtime checks. Explicitly test existing `seat_resolver` and custom project-entry paths. Acceptance: invalid examples fail before image preparation/provider calls, while valid non-default alternatives pass. Unsupported model/backend combinations identify the exact missing capability and do not silently fall back.

### M3 — runtime agreement and native parity


Refactor serving/TRL/veRL translation to consume resolved settings, with field-consumption coverage. Verify actual tokenizer/processor/template, special tokens, loaded precision, runtime source, and selected parser. Implement bounded readiness probes and runtime-resolution evidence using existing clients, run snapshots, and diagnostics. No new telemetry service. Add real tokenizer-rendering fixtures, target/draft compatibility tests, and endpoint negative tests. Acceptance: CLI/offline/online/trainer request paths retain the same intended model semantics, explicit false values survive native translation, and mismatched deployed model/template identity is not reported healthy merely because HTTP works.

### M4 — explain and controlled bypass


Add CLI `--explain` and `--skip-preflight` at the existing project/job execution seams, plus equivalent typed execution options for Python callers of that seam. Extend `apps/cli/src/posttrain_cli/commands/job.py`, `commands/work_package.py`, `execution_planning.py`, and report rendering without moving validation into CLI. Test default, skipped, deferred, failed, and successful probes. Acceptance: skip mode can avoid a bounded optional endpoint/cache-readiness probe, still rejects malformed or known-invalid settings, still verifies runtime identity, and leaves a clear receipt. No effect on qualification bypasses, security, or artifact integrity.

### M5 — versioned defaults, migration, and real qualification


Create new binding/recipe revisions only for intentional default changes justified by M0. Preserve existing model artifacts and renderer identities; update consumers explicitly. Add a local fixture project/work package covering policy, judge, teacher/reference roles and negative configurations. Qualify model configuration through Posttrain's normal local executor using exact source snapshots or existing pinned runtimes, not a raw backend CLI as final evidence. Start with the pinned Qwen3.5-2B on an available suitable GPU for rendering, short generation, and tool-call transport. These are model integration tests, not a benchmark or async training experiment.

Exercise LFM, Gemma, and Nanbeige schema/rendering rules with pinned lightweight metadata/tokenizer fixtures; run real model loads for any changed model-specific runtime behavior before claiming that path qualified. Use suitable capacity for larger models; no forced 12B load on an 8GB device. For changed trainer-loading semantics, run a tiny supported model optimizer step for each affected backend and verify the intended parameters update while frozen roles remain frozen. This is correctness qualification, not a 20/50-step comparison. A GPU-skipped case remains an open release gate.

Prove local/remote packaging carries the same resolved model contract and fingerprints. An actual remote run is required only if deployment-specific model readiness or loading behavior changed; then use an authorized target and preserve evidence. Do not launch cloud jobs merely to validate a pure config change. Acceptance: one documented developer golden path from selection through explain, early rejection, and real local execution; all changed supported paths have evidence; unsupported paths remain clearly rejected or unqualified.

## Concrete validation commands


Before edits, use the active worktree and inspect its state:

    cd /home/hammad/projects/rl-local-async
    git status --short
    git branch --show-current
    git rev-parse HEAD

Focused existing tests to extend, using the already prepared local environment without replacing source overlays:

    .venv/bin/python -m pytest packages/common/tests/test_model_variants.py packages/common/tests/test_model_chat_templates.py packages/common/tests/test_model_artifact_descriptor.py
    .venv/bin/python -m pytest packages/serve/tests/test_vllm_bindings.py packages/serve/tests/test_vllm_compat.py packages/serve/tests/test_vllm_offline.py
    .venv/bin/python -m pytest packages/train/tests/test_trl_common.py packages/train/tests/test_trl_online_rl.py packages/train/tests/test_verl_backend.py packages/train/tests/test_rendering.py packages/train/tests/test_sft_validation.py
    .venv/bin/python -m pytest packages/project/tests/test_service.py packages/work/tests/test_work.py packages/jobs/tests/test_jobs.py apps/cli/tests/test_cli.py

Proposed new tests, created in their milestones:

    .venv/bin/python -m pytest packages/common/tests/test_configuration_issues.py packages/train/tests/test_model_configuration.py packages/work/tests/test_model_validation.py apps/cli/tests/test_model_preflight.py

For final locked validation use a separate clean qualification checkout/environment so `uv sync` does not replace the active editable forks unexpectedly. Follow the repository ladder there: `uv sync --all-packages --locked --python 3.13`, `uv run ruff check .`, `uv run pyright`, `uv run lint-imports`, `uv run pytest`, and `git diff --check`. Record exact failures/skips rather than implying the full ladder passed from focused tests.

During M5 create `.posttrain/work_packages/model-settings-qualification.yaml` and a documented selected job ID `model-smoke`, with an exact matching project overlay. The intended command is then `posttrain job plan .posttrain/work_packages/model-settings-qualification.yaml --job model-smoke --explain`, followed by `posttrain job run .posttrain/work_packages/model-settings-qualification.yaml --job model-smoke --provider local`. These files/flags are planned artifacts, not runnable at revision 1. Resolve actual interpreter, GPU availability, source snapshot, and executor options first; update this section with the exact successful command and evidence path before marking M5 complete.

## Validation and Acceptance


The test matrix must cover defaults and validity independently. For every setting family, test omitted, explicit recommended, explicit valid alternative, malformed, incompatible combination, and runtime-unknown inputs. For aliases test agreement and contradiction; for numbers test booleans, strings, NaN, infinities, boundaries, and backend sentinels. Test environment/request versus inference sampling conflicts without changing task selection.

Use parameterized coverage for all registered standard model-consuming jobs, including serving smoke/benchmark, evaluation, supervised/preference training, all existing online-RL definitions, distillation, and model transforms where applicable. Assert role-specific validation: a judge with a different tokenizer is valid, a teacher used for exact token scoring with an incompatible vocabulary is rejected, and the rollout model cannot silently differ from the policy model. Existing operations unsupported by a backend remain unsupported.

Test model families with different reasoning/tool protocols, derived adapters and full weights, text-only and multimodal declarations, and required processor artifacts. Test explicit `thinking`/`off` versus `native`, package versus tokenizer template sources, past-reasoning stripping, EOS/pad fallback, custom stop controls, and exact token-ID preservation. A mocked parser declaration alone is not a rendering-parity test.

Native-translation tests must cover explicit false options, unsupported/dropped keys, precision triplets, draft depth and pairing, and known revision-scoped exclusions. Model metadata and runtime metadata disagreements must fail at the earliest stage that has evidence. Instrument tests with spies proving zero provider, builder, model-download, and heavyweight-import activity during pure planning. These are test assertions, not a production instrumentation subsystem.

Hardware-advice fixtures must compare RTX A6000, RTX PRO 6000 Blackwell, H100 variants, H200, generic unknown CUDA, partitioned devices, and multi-GPU placements. Cross these with dense/GQA, hybrid/recurrent, MoE and multimodal metadata; full-weight, LoRA and QLoRA updates; and sequential versus overlapping residency. Prove TurboQuant recommendations do not depend on LoRA or rank. Test qualified MTP default enablement, explicit disable preservation, missing paired assistant, failed TurboQuant quality gate, and individually supported features with an unsupported combination. A 90% engine memory setting alone must never produce “90% compute utilization.” High available VRAM must not change rank, update method, training batch, sequence limits, or topology. Runtime tests are required on each exact hardware path whose optimized default is newly promoted; offline matrix tests alone cannot qualify new kernels or quantized numerics.

Preflight tests distinguish absent local cache, network-unavailable optional checks, known incompatibility, stale evidence, different source revisions, different target, and changed options after a previous successful plan. TOCTOU protection means packaging/execution checks the exact resolved digest it was given rather than re-resolving a mutable catalog behind the user's back. Secret values must never enter reports or snapshots. Skip mode must remain incapable of disabling integrity, tokenizer compatibility, or runtime request-budget checks.

Completion requires the documented CLI/Python golden path, focused and boundary tests, relevant real runtime evidence, versioned default migrations, and an explicit list of remaining unqualified combinations. “All model configs are safe” and “OOM cannot happen” are not acceptable completion claims.

## Idempotence, migration, and recovery


Begin additively: preserve old constructors, selected revisions, and effective behavior while exposing origins. Add compatible report fields, then wire early checks, then migrate intentional defaults via new bindings. Unknown fields that were previously silently ignored become targeted errors; provide a migration hint rather than continuing silent behavior. Do not broadly allow unknown configuration for backward compatibility.

Do not alter a running job or re-resolve a queued immutable plan under new defaults. Failed static validation creates no provider submission or GPU allocation. Failed runtime verification records failure and releases only run-owned resources through existing cleanup. Readiness retries must not repeat paid inference, model-changing operations, or tool actions. Successful probe caches are bounded, identity-scoped conveniences, not a new artifact store.

Rollback means selecting a previous qualified binding/framework revision for a new run, not overwriting old evidence or removing validation on a known-invalid request. Keep source edits and test evidence separated from unrelated dirty work. Commit this work in milestone-sized changes only when implementation is requested and verified; no automatic push or release is part of this planning turn.

## Surprises & Discoveries


The original conversation focus on rollout failures led to an initially too-narrow investigation. The user's clarification explicitly replaces that scope with model settings across the framework. Environment discovery and async worker redesign are not deliverables here.

Serving FP16 and training BF16 defaults differ in inspected code, but different role precision can be intentional. The defect to prevent is unexplained or incompatible resolution, not all precision differences. Similarly, family-specific thinking defaults must not be replaced with one global switch.

The model's maximum advertised context is not a sensible operational context default. The existing Qwen3.5 catalog declares 262,144 native tokens while recent local tests selected much smaller engine windows. Resolution should explain this relationship, not allocate maximum context by default or reject every smaller setting.

Python type hints alone do not enforce runtime configuration. Several dataclass validations are weaker than corresponding schema or adapter assumptions. Exact accepted types, boolean handling, finite floats, and translation-consumption tests are necessary for YAML/Python parity.

Existing native options are split across frontend arguments, engine settings, request settings, and trainer attributes. A field can appear in the selected configuration but fail to reach one execution path. The resolution/consumption tests address this without forcing backend schemas into common contracts.

The inspected TRL consumer ledger still records Qwen3.5 K8V4 recall regressions for tested long-context configurations. This blocks blanket promotion of TurboQuant despite its potential cache-capacity benefit. The user wants TurboQuant recommended independently of the update method, not bundled with LoRA rank 8; this supersedes the intermediate interpretation of that phrase.

## Decision Log


2026-09-09: Scope is all model-related settings across policy, serving/evaluation, judge, teacher, reference, and draft roles. Async scheduling and harness performance remain in their existing plan.

2026-09-09: Defaults and validity are separate deliverables and tests. Explicit supported alternatives remain valid even when not recommended. No silent clamping, quantization, model substitution, thinking-mode change, or native-option dropping.

2026-09-09: Reuse existing model/interface selections, capability-owned adapters, standard-job validators, and provider-free `Project.jobs.plan`. Add small shared diagnostic values, not a new model orchestration framework.

2026-09-09: Make reasoning an explicit per-use selection while preserving versioned renderer/token semantics and existing defaults. A narrow canonical API amendment precedes implementation.

2026-09-09: The proposed bypass is limited to additional readiness probes. This deliberately does not skip statically known compatibility errors; “skip checks” must not mean “accept invalid model semantics.”

2026-09-09: Resolve target-sensitive defaults using qualified bindings or an explicit recorded runtime policy. Performance estimates are warnings, not universal invalidity rules. Real GPU qualification is required for changed model-runtime behavior, not for every pure validator test.

2026-09-09, revision 2: Enable MTP by default in new qualified model/backend/hardware/operation bindings. Recommend TurboQuant independently of full-weight/LoRA/QLoRA, and prefer it in newly qualified optimized defaults. Preserve explicit overrides and known quality/compatibility gates. Do not introduce a new LoRA-rank default from the user's corrected wording.

2026-09-09, revision 2: Add phase-aware hardware advice using architecture, role residency and training method/rank. Hardware determines eligible implementations and capacity recommendations, not scientific training choices. Memory reservation, useful cache demand and compute utilization are separate quantities.

## Outcomes & Retrospective

Revision 5 implementation outcome: provider-free resolution now has one
stable report shape shared by Python and CLI planning. The report records
origins, known incompatibilities, target capabilities, and a semantic digest;
it does not import vLLM, load a model, or make a provider call. The native
adapter boundary consumes the resolved vLLM configuration so command and
benchmark behavior cannot independently re-derive defaults. Numeric hardening
also rejects booleans and non-finite values in engine, sampling, and adapter
fields. The real local qualification covers an optimizer update but not the
new serving readiness path. No claim is made for untested 12B-family loading
or production target availability.


Revision 1 delivers the implementation plan and its separate policy decisions only. No model default, adapter behavior, canonical baseline, dependency pin, runtime, or active run has been changed by this plan. Implementation begins with M0; release requires all applicable acceptance gates, not just creation of schemas or new CLI flags.

Revision 2 added concrete hardware/default eligibility and warning requirements, independent TurboQuant recommendations, and architecture-aware capacity/placement tests. Hardware-specific defaults were not enabled or qualified by that document update.

Revision 3 begins implementation with the field inventory and canonical API amendment. It broadens the settings audit to job type and algorithms while preserving the boundary that advice cannot silently modify algorithm meaning.

Revision note: created after the user clarified the scope from async rollout configuration to all model-related settings. The initial investigation is retained only where it informs existing model-resolution and submission boundaries; harness work is excluded.

Revision 2 note: extended the same plan after the user requested RTX PRO 6000/H100/H200-aware configuration, utilization warnings, visibility into sequential execution and full-weight/LoRA/rank choices, and MTP/TurboQuant defaults. The latest clarification removes any implied coupling between TurboQuant and LoRA.

Revision 3 note: implementation began at M0 and the user expanded the settings surface to job-type and algorithm defaults. The inventory and canonical amendment now state that separation explicitly.
