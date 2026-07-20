# Profiles and model variants

Status: target MVP architecture  
Last revised: 2026-07-20

## Purpose

Profiles make proven model and engine setup reusable without turning
configuration into a workflow language or storing results beside inputs.

## Core model

There are three different things:

- a **model artifact** is immutable weights, an adapter, a quantized model, a
  speculator, or a promoted checkpoint;
- a **model profile** is a typed, version-controlled starting definition for a
  loadable model artifact or composition;
- an **engine profile** is a typed, implementation-owned set of reusable
  execution defaults.

```text
model profile + engine profile + explicit caller overrides
  -> typed package operation
  -> resolved JSON-safe snapshot in Trackio
  -> execution
  -> optional descendant artifact
```

Profiles describe how to start. Trackio observes what happened.

## Code-first definitions

Profiles are normal importable Python objects. Each owning package defines its
own Pydantic model or frozen dataclass:

```python
QWEN_35_2B = ModelProfile(
    id="qwen3.5-2b",
    artifact=HubModel(repo="Qwen/...", revision="<commit>"),
    family="qwen3.5",
    context_window=262_144,
    modalities={"text"},
    reasoning_modes={"thinking", "non_thinking"},
    recommended={
        "serve": "posttrain.serve.profiles.qwen35.VLLM_STANDARD",
        "general_eval": "posttrain.eval.programs.GENERAL_SMALL_MODEL",
    },
)
```

Engine profiles remain concrete:

```python
VLLM_TURBOQUANT_K8 = VllmServeConfig(
    max_model_len=32_768,
    kv_cache="turboquant-k8",
)

TRL_SFT_QLORA = TrlSFTConfig(
    use_peft=True,
    load_in_4bit=True,
)
```

The job may import a named definition directly or select a recommended
reference from the model profile. It can make an explicit copy with overrides.
There is no generic resolver that treats model, train, eval, and serve schemas
as one profile family.

## Where definitions live

| Definition | Location | Why |
| --- | --- | --- |
| Model profile | `posttrain.common.profiles` | Cross-engine entry point with lightweight model facts and references |
| Train profile | `packages/train` | Versioned with the reusable package operation and validation that understand it |
| Eval program | `packages/eval` | Versioned with environment loading and execution policy |
| Serve profile | `packages/serve` | Versioned with backend adapter, kernels, and compatibility tests |
| Workload suite/corpus | `posttrain.serve.benchmarks` | Shipped and versioned with the public operation that validates and executes it |

The existing YAML files may be migrated into these typed definitions. They are
not retained as compatibility contracts.

## Foundation and derived profiles

A **foundation profile** points to an immutable upstream model revision. A
**derived profile** points to a descendant intentionally promoted as a reusable
entry point, such as a broadly useful adapter, merged model, quantized release,
or selected post-training checkpoint.

Most checkpoints remain artifacts only. A derived profile is created when:

- another job should be able to import the descendant directly;
- serving or evaluation owners need a stable target;
- the artifact is an intentionally named release; or
- it requires durable compatibility or composition information.

Adding a profile does not create lineage. It points to lineage already observed
through the producing and consuming runs.

## Model facts versus engine choices

| Model profile owns | Engine profile owns |
| --- | --- |
| immutable artifact reference | package operation and backend implementation ID |
| family and architecture | runtime dtype and load format |
| artifact form and required base | weight/KV-cache quantization |
| native context limit | context actually exposed by an endpoint |
| modalities | kernels, scheduling, caching, and parallelism |
| native reasoning/MTP capability | speculative-decoding configuration |
| tokenizer/chat-template identity | workload and resource defaults |

MTP support is a model fact; enabling MTP in vLLM is a serve-profile
choice. TurboQuant KV cache is entirely a serve-profile choice. A weight-changing
quantization produces another model artifact; a runtime-only cache or kernel
setting does not.

## Recommended variants

A model profile can expose stable names for combinations proven by the owning
teams:

```python
QWEN_35_2B.recommended == {
    "serve.standard": "posttrain.serve.profiles.qwen35.VLLM_STANDARD",
    "serve.turboquant_k8": "posttrain.serve.profiles.qwen35.VLLM_TURBOQUANT_K8",
    "serve.mtp": "posttrain.serve.profiles.qwen35.VLLM_MTP",
    "train.sft": "posttrain.train.profiles.qwen35.TRL_SFT_QLORA",
    "train.dpo": "posttrain.train.profiles.qwen35.TRL_DPO",
    "train.grpo": "posttrain.train.profiles.qwen35.TRL_GRPO",
    "eval.general": "posttrain.eval.programs.GENERAL_SMALL_MODEL",
}
```

These are discoverability references, not copied configs. Updating a shared
serve profile and publishing the serve package makes the tested setup available
to job code after it pulls that version.

## Job-local overrides

Jobs should express meaningful differences, not duplicate complete profiles:

```python
serve_config = VLLM_TURBOQUANT_K8.model_copy(
    update={"gpu_memory_utilization": 0.90}
)

ctx.run(serve.benchmark, model=QWEN_35_2B, config=serve_config, suite=LONG_CONTEXT)
```

If several jobs repeat the override, it is promoted into the owning engine
package as a named, tested definition. If it remains specific to one objective
or machine, it stays in that job.

## Serialization and provenance

Python source is authoritative for the definition. At run start the executor
records:

- fully qualified symbol or stable definition ID;
- source package and version;
- repository commit and dirty digest;
- schema name/version;
- fully resolved JSON-safe configuration;
- explicit job overrides.

Trackio can therefore reproduce and compare executions without becoming the
profile registry. It never stores executable callables.

## Model evidence

A model evidence view joins an exact model artifact to known serving,
evaluation, and training observations. It is calculated by `packages/reports`.
It is not another profile and does not copy results into Python or YAML.

## MVP constraints

- typed Python definitions and explicit composition;
- no arbitrary multi-parent inheritance;
- no universal engine config schema;
- no mutable descendant list on a model profile;
- no automatic profile creation for checkpoints;
- no Trackio-backed profile registry;
- optional TOML/YAML only as input to a concrete typed config.

## Revision history

- 2026-07-20: Clarified that profiles configure reusable package operations;
  framework runners/adapters remain internal implementation details.
- 2026-07-20: Replaced generic YAML profile resolution with typed Python model
  and engine definitions owned by their packages; clarified recommended
  standard, TurboQuant, MTP, and training variants.
- 2026-07-19: Defined foundation and deliberately promoted derived profiles separately from artifact lineage.
