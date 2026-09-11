# Configuration defaults, validation, and hardware advice


**Status:** implementation inventory — 2026-09-09. **Authority:** subordinate
to `docs/post-training/`. The living implementation plan is
`docs/plan/model-settings-defaults-and-validation.md`.

## Mental model


Posttrain has no universal configuration object. A model variant identifies
immutable weights and their interface. An inference binding defines how one
model role generates or scores tokens. A training binding defines parameter
updates and a training target. Job type determines required roles and their
lifecycle. Algorithm settings define the learning objective and its required
population. The resolved job combines these selections without moving their
ownership.

A **default** fills an omission. A **validation rule** rejects a contradiction.
A **recommendation** describes a valid alternative without applying it. A
**runtime check** uses facts unavailable during detached planning. Reports must
preserve these meanings.

## Resolution and precedence


Catalog selection happens before field resolution. Within one role, an
operation-supported explicit request override precedes that role's inference
binding, which precedes a versioned family/purpose default, which precedes a
qualified adapter default. Algorithm requirements do not silently win a merge:
the selected settings must agree or validation fails.

Omission differs from `null`, `false`, zero, and an empty collection. Explicit
valid values remain unchanged. Recognized aliases resolve once and conflicting
aliases fail. Mutable model cards, Hub `main`, unpinned generation config, and
ambient environment variables are not default sources.

## Model and interface fields


| Field | Owner and roles | Omission/default | Validation stage |
| --- | --- | --- | --- |
| artifact, base, revision/digest | `ModelVariant`; every role | immutable identity required; adapter base explicit | relationships statically; files at runtime |
| form and weight precision | `ModelVariant`; every load | form required; catalog precision currently BF16 | form/artifact/quantization statically; artifact truth at runtime |
| family and parameters | `ModelVariant`; loaders/advice | required | positive count and renderer family statically; architecture details at runtime |
| instruction-tuned and modalities | `ModelVariant`; conversational/task roles | required | role requirements before submission; processor at runtime |
| native context | `ModelCapabilities`; generation | required positive ceiling | operational window may be smaller; extension must be explicit and qualified |
| MTP capability | `ModelCapabilities`; generation | false | eligibility only; complete binding qualification enables it |
| tokenizer/processor identity | model artifact; exact-token roles | pinned artifact or explicit inherited base | fingerprint shape statically; exact files at runtime |
| renderer/template/roles | `RendererContract` | versioned model contract | family, template source and protocol statically; parity at runtime |
| reasoning modes | conversation plus per-use inference/training selection | renderer default | requested mode must exist; exact-token policy/rollout agreement |
| tool-call protocol | conversation and inference frontend | absent means unsupported | parser/backend/environment compatibility and runtime parser discovery |
| quantization/provenance | model artifact | empty | weight-quantized form requires metadata; loader support separately |

A reasoning-mode change does not create new weights, but it changes effective
behavior and belongs in the versioned inference binding/run snapshot. Policy,
judge, teacher, reference, and draft roles are resolved independently.

## Inference and generation fields


| Field | Owner | Default rule | Main checks |
| --- | --- | --- | --- |
| backend, renderer, target, purpose | `InferenceBinding` | required | exact version, model renderer, job role and target compatibility |
| capabilities | `InferenceBinding` | empty | environment requirements; tool protocol/parser when applicable |
| max model length | engine binding | bounded purpose value for new profiles | prompt plus output reserve; native ceiling; no automatic maximum |
| compute dtype/load format | engine binding | qualified artifact/hardware policy | separate from stored and cache precision |
| GPU memory fraction | engine binding | explicit target/purpose value | finite `(0,1]`; per-engine memory budget, not compute utilization |
| sequence/token batch caps | engine binding | target/purpose value | positive; low values may warn without being invalid |
| explicit KV-cache bytes | engine binding | absent delegates to pinned policy | precedence over fraction-based sizing is explained |
| KV-cache dtype | engine binding | recommend TurboQuant K8V4; default only after exact qualification | independent of update method and adapter rank |
| eager/graphs/compilation | engine or trainer | qualified architecture policy | explicit values retained; kernel/runtime support checked |
| chunked prefill/prefix cache | engine | qualified workload policy | architecture and weight-update invalidation compatibility |
| text-only/multimodal profiling | engine | multimodal-safe unless explicitly restricted | selected tasks and processors must agree |
| tensor/data/context/expert parallelism | owning binding | one on new single-device profiles | topology, device count, architecture-specific constraints |
| offload/sleep/cache lifecycle | owning binding | qualified operation policy | residency and latency advice; never auto-enabled to make an estimate fit |
| tool/reasoning parsers | frontend | derive from qualified protocol | conflicting explicit parser fails; existence verified at runtime |
| output and sampling controls | inference/operation | family- and purpose-qualified | finite ranges, context, stops, EOS and algorithm requirements |
| seed | qualification/operation | explicit where reproducibility is required and supported | backend propagation and limitations retained |
| speculative method/depth/assistant | inference binding | MTP on only for qualified new bindings | model eligibility, exact draft/head, weight sync, hardware and combined features |

Per-request output, cumulative episode input/output/total, training sequence,
and judge input/output are different budgets. They are never substituted. A
multi-turn episode limit checked between turns may be exceeded by its final
completed turn, and the actual stop condition remains evidence.

## Training and algorithm fields


| Setting family | Owner | Existing/default policy | Validation/advice |
| --- | --- | --- | --- |
| update kind | `TrainingBinding.update` | explicit full/LoRA/QLoRA/QAT | never selected from available VRAM; backend/model form checked |
| LoRA rank/alpha/dropout/targets | update plan | current schema 8/16/0/all-linear | show trainable fraction using actual modules; hardware never changes rank |
| QLoRA quantization | update plan | NF4/BF16/double | requires unquantized source and supported compute |
| training renderer/reasoning | training binding | explicit | model family/mode and exact-token rollout agreement |
| topology/offload/runtime | training binding | one node, no offload; device count may be required | selected hardware, model state and job lifecycle |
| training loop | algorithm settings | kind-specific versioned values | finite learning controls, positive counts, sequence and checkpoint relationships |
| SFT validation | SFT settings | optional | positive bounded validation work and dataset compatibility |
| DPO beta/loss kernel | DPO settings | beta 0.1; Torch unless selected | positivity, backend support and reference semantics |
| RL prompt groups/generations | GRPO/DAPO/OLMo3/SAMPO/GDPO/CAPO | kind-specific; at least two generations | effective/global batch and complete logical-group relationships |
| prompt/completion lengths | online algorithms | bounded kind-specific values | sum fits loop and inference context; sampling max agrees |
| KL, clipping and IS | online algorithm | kind-specific | finite ranges and exact backend support; no approximation |
| dynamic/active sampling | DAPO/OLMo3/SAMPO | algorithm-specific | bounded attempts and correct algorithm only |
| reward components/weights | GDPO/CAPO | explicit algorithm contracts | names match environment/scorer evidence; valid population and dominance rules |
| teacher temperature/generations | distillation | positive versioned values | effective batch, teacher/student token compatibility |

Job-type defaults and algorithm defaults are separate. A standard job defines
required roles and invokes an operation; it does not choose a project's learning
rate or task population. An algorithm can mandate mathematical invariants while
allowing several valid operating points. Recommendations may identify inefficient
settings but cannot change the algorithm, effective batch, reward weighting, or
number of training steps.

## Hardware facts, residency, and advice


An exact target records `accelerator_model` (the scheduler-facing SKU),
accelerator architecture or compute
capability, visible usable memory, count, partition, interconnect, and runtime
requirements. Generic memory-only targets remain valid but cannot authorize
architecture-specific defaults. Exact profiles must distinguish RTX A6000,
RTX PRO 6000 Blackwell variants, H100 variants, and H200 variants.

Hardware selects eligible implementations, not scientific settings. New exact
profiles should assess optimized attention, graph execution, BF16 or explicit
qualified lower precision, engine token/sequence capacity, MTP, and TurboQuant.
Hopper FP8 support does not silently quantize BF16 weights. More VRAM does not
select full-weight training, increase LoRA rank, enlarge context, or change the
algorithm batch.

Memory advice is phase-aware: loading/warmup, generation, judging/teacher
scoring, optimization, and checkpointing. Each device/phase accounts for
weights; trainable/frozen state; gradients and optimizer/master state; adapters;
KV or recurrent state; speculative assistants; activations; graphs/workspace;
and headroom. Dense attention, GQA/MQA, sliding-window, recurrent/hybrid, MoE,
and multimodal models require different estimates. Unknown architecture inputs
remain unknown rather than receiving a dense-model formula.

Reports distinguish three meanings of sequential: models resident in
non-overlapping phases, inference concurrency one, and selected GPUs with no
assigned work. They warn about provable reservation conflicts and possible
underuse such as tiny sequence/token caps, unnecessary offload, forced eager
mode, unused devices, or cache capacity far above bounded demand. Suggested
alternatives do not apply themselves.

`gpu_memory_utilization: 0.90` is a per-engine reservation/budget, not measured
GPU compute utilization. Two concurrent engines each requesting 90% of one GPU
conflict; sequential roles still include allocations that remain resident. A
pre-submission report can estimate probable underuse but cannot invent measured
utilization or tokens per second.

## MTP and TurboQuant


MTP defaults on only for new bindings whose exact model, backend, hardware,
operation, draft depth and update path are qualified. Explicit off remains
valid. Support and measured speedup are separate; high concurrency can change
the benefit.

TurboQuant K8V4 is recommended for every eligible generation binding regardless
of inference-only, full-weight, LoRA, or QLoRA use. It is KV-cache storage, not
adapter rank. Report `enabled`, `recommended_available`, `not_applicable`,
`unsupported`, or `qualification_blocked`, with a reason. Existing Qwen3.5
consumer evidence records long-context K8V4 recall failures for the tested path,
so that exact path remains blocked from default promotion until matched quality
gates pass. MTP plus TurboQuant requires combined qualification; individual
passes do not establish composition.

## Validation and developer output


Pure planning checks types, identities, ownership, role relationships, known
compatibility, arithmetic, and declared hardware without native ML imports,
downloads, GPUs, or provider calls. Optional readiness checks use bounded cached
or existing-endpoint metadata. Runtime checks verify actual files, fingerprints,
devices, kernels, parsers, and resolved automatic choices. Per-load/request
guards remain mandatory.

Errors stop known contradictions. Warnings describe valid but potentially poor
choices. Deferred findings identify missing runtime facts. Recommendations are
non-mutating. Skipped readiness probes are retained separately. The proposed
`--skip-preflight` never bypasses schema, compatibility, integrity, security, or
runtime guards.

`posttrain job plan ... --explain` shows each role, effective setting and origin;
job type; algorithm; update kind and adapter rank; phase/device residency;
MTP/TurboQuant eligibility; context and batch arithmetic; memory range; and
warning confidence. It labels measured, estimated, unknown, deferred, and
skipped facts distinctly.
