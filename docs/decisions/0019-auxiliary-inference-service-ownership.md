# ADR 0019 — Auxiliary inference is an independently owned service dependency

## Status

Accepted for implementation, 2026-09-07; amended 2026-09-09 to cover API-only
external providers. Remote service-handle qualification remains open. Related
plans: `docs/plan/gdpo-capo-dual-backend-support.md` and
`docs/plan/openrouter-default-judge-service.md`.

## Context

Episode-level GDPO and CAPO can ask an environment-owned judge to score policy
trajectories. The algorithm consumes validated reward evidence; it does not own
the judge model. A local composition may launch a small judge beside training,
but a 12B judge and a 2.6B training/rollout stack do not have a safe implicit
reservation merely because both inference bindings name the same 96 GiB target.
`serve.launch` starts a child process inside the job's existing allocation; it
does not provision the judge binding's target or isolate its GPU memory.

Treating that child process as remotely managed would make a portable package
appear safe while relying on unbudgeted colocation. It would also couple judge
availability to optimizer lifecycle and make service reuse, independent scaling,
failure attribution, and cleanup ambiguous.

## Decision

- Production remote judged training uses a separately provisioned inference
  service and attaches to it. The service owner reserves its target, starts the
  selected model/runtime, authenticates the endpoint, proves readiness, retains
  immutable model and engine provenance, and stops only the service it owns.
- API-only providers and routers are a third, unowned lifecycle variant. The
  hosted inference binding must explicitly select both the API model and the
  provider slug. The composition host validates and probes that exact pair but
  does not pretend to deploy, stop, or own remote weights. A hosted-model
  selector is not a `ModelVariant` and creates no model-lineage edge.
- A training job receives a named attached-service connection plus the expected
  `InferenceBinding`. The volatile address and scoped credential reference are
  composition inputs, not catalog primitives and never algorithm settings.
- Attachment must verify deployment provenance against the selected model
  artifact/revision, renderer, backend, and relevant engine identity. `/models`
  alone proves only an alias and is insufficient for reproducible qualification.
- Environment judge plugins continue to own prompts, rubrics, schemas, retries,
  parsing, and named reward annotations. GDPO, CAPO, SAMPO, TRL, and veRL see only
  the existing validated reward-evidence contract.
- Judge plugins call one narrow `openai-chat@1` structured-completion contract.
  Managed vLLM, attached endpoints, and API providers all resolve to that same
  contract. Provider adapters must normalize routing and capability differences
  before exposing the endpoint; provider names and router payloads do not enter
  judge prompts, schemas, algorithms, or plugin sampling configuration.
- The implementation uses Verifiers `Judge.complete()` and the official
  `AsyncOpenAI` client as the wire adapter. Posttrain owns endpoint resolution,
  capability admission, credentials, cost controls, and request adaptation. A
  general agent framework or multi-provider proxy is not part of this boundary.
- Hosted-model profiles contain intrinsic model/interface facts only. Provider
  endpoint profiles contain route-dependent transport facts such as whether
  structured output is accepted as JSON Schema or JSON object. Composition
  probes the selected model-provider pair and resolves both profiles into one
  internal immutable judge client; jobs do not author that resolved value.
- Chat templates remain token-serialization contracts. Verifiers plugins retain
  prompts, rubrics, and response schemas. Model-native wording may be retained
  as versioned model or environment metadata, but it does not change provider
  transport or create another chat template.
- Managed child-process inference remains valid for explicitly qualified local
  or partitioned compositions. Remote colocation is rejected unless the host
  supplies an explicit resource partition and lifecycle controller covering all
  resident models; independent `gpu_memory_utilization` fractions are not such a
  reservation.
- Service readiness, saturation, timeout, and termination remain distinct from
  scorer-output validity. A failed endpoint never becomes a numeric zero reward.
- A router route is explicit and frozen per optimizer run. Price- or
  availability-based automatic provider selection is not admission policy.
  Provider fallback that would
  change the judge implementation is disabled by default; a route change is a
  new explicit provider attempt with separate evidence rather than a silent
  continuation of the same reward population.
- The default OpenRouter policy permits provider retention and does not require
  zero-data-retention routing. Posttrain records that policy but does not mutate
  account-level prompt logging or privacy settings during job execution.
- Every API-paid judge binding carries a run-wide hard cost ceiling. The shared
  default is USD 4.99. Composition derives a worst-case call/token projection
  from resolved loop and judge limits, prices it using the selected live
  endpoint, and rejects an over-budget run before the paid readiness probe.
  Admitted calls use an atomic run-local reservation guard, so concurrency and
  retries cannot spend through the ceiling. Missing usage is charged
  conservatively. Raising the ceiling requires an explicit versioned binding;
  it is never an algorithm knob or implicit retry behavior.
- The same auxiliary-service boundary applies to any algorithm whose Verifiers
  environment uses a judge, including scalar-reward GRPO, DAPO, and OLMo-style
  GRPO. Algorithms expose only a generic maximum trajectory-consumption
  envelope; Verifiers configuration exposes calls and tokens per trajectory;
  composition combines them without moving judge semantics into training.
- The host/work-package orchestrator owns dependency ordering: service ready,
  training admitted, training drained, then owned service release. Attached
  services are never stopped by the consumer. Interrupted training may reattach
  only to the same qualified deployment identity or must create a fresh run.

## Consequences

The initial Gemma 4 12B comparison needs an additional service deployment and
attachment gate before GDPO can run. This may consume a second GPU, but it avoids
an unmeasured OOM risk and lets judge concurrency scale independently from policy
rollouts. Vanilla GRPO and OLMo3 remain single-allocation jobs and do not wait for
or pay for a judge.

Portable packaging still contains the expected judge inference selection and
task-owned rubric. It does not contain a workstation path or a fixed ephemeral
URL. Evidence must connect the training run to the exact service deployment and
retain endpoint latency/error observations without credentials.

The conservative projection may reject a large judged run even when its likely
cost is lower. That is intentional: an operator must either reduce the bounded
population or explicitly select a higher ceiling. The selected limit and final
metering receipt make that choice auditable without coupling dollars to reward
weights.

Provider transport declarations become explicit catalog data. This adds one
small required nested profile to hosted-inference bindings, but removes the
misleading practice of attaching endpoint behavior to a reusable hosted model.
Service receipts make the requested contract, effective transport, and
validation strategy inspectable.

## Alternatives Considered

### Launch the judge in the training process allocation

Rejected for the 12B comparison because the target declaration does not reserve
a second device or coordinate vLLM sleep/wake phases. Previous scalar GRPO
evidence already reached the 96 GiB limit during actor log-probability work.

### Put model launch logic in the Verifiers judge plugin

Rejected because plugins own scoring semantics, not GPU placement, credentials,
or process lifecycle. It would also make the same rubric behave differently by
trainer backend.

### Put judge model and decoding fields in GDPO/CAPO settings

Rejected because those are inference/composition concerns and would prevent
multiple rubrics or algorithms from sharing one service safely.

### Add LiteLLM or PydanticAI inside judge plugins

Rejected for the current OpenRouter and vLLM services because both already expose
the same OpenAI-compatible Chat Completions protocol used by Verifiers. LiteLLM
would duplicate Posttrain's route, retry, spend, and usage ownership; PydanticAI
would add an agent abstraction above a scorer that needs only one typed
completion. Reconsider a broader adapter only when an admitted provider cannot
implement `openai-chat@1` without lossy behavior.

### Trust a caller-supplied URL and model alias

Rejected for qualification because it cannot prove which weights or engine
served the request. It remains useful only for explicitly non-reproducible
exploration labeled as such.

## Implementation Notes

Extend the existing `AttachedInferenceService` composition path rather than
adding a scorer provider to `posttrain.train`. Add a provider-owned service
deployment receipt/handle and an attachment resolver in `posttrain.jobs` or the
Lab host. Keep execution-provider types out of reusable capability packages.
Tests must cover distinct policy/judge artifacts, credential redaction, endpoint
identity mismatch, readiness failure, consumer cleanup that leaves attached
services running, and interrupted reattachment. The live GDPO comparison remains
blocked until service placement and teardown are observed on RunPod.

`HostedModel`, `ProviderEndpointProfile`, and `HostedInferenceBinding` live in
`posttrain.common`. `ResolvedInferenceService.judge_client` is the internal
composition view and is intentionally not a catalog family. OpenRouter resolves
`json-schema` directly or adapts `json-object` with the exact schema preserved
in the system instruction; the caller's typed local validation remains the
acceptance authority.

## Revision History

- 2026-09-11: Separated intrinsic hosted-model behavior from provider-endpoint
  transport capabilities and defined the internal resolved judge-client
  contract. Kept chat templates and Verifiers rubric semantics outside this
  service-resolution boundary.

- 2026-09-10: Standardized judge invocation on `openai-chat@1`, assigned
  capability adaptation to service resolvers, and explicitly declined an
  additional multi-provider agent SDK at the scoring boundary.

- 2026-09-09: Added the USD 4.99 default hard ceiling, live-price admission,
  and concurrency-safe run-local enforcement for API-paid judges.

- 2026-09-09: Added API-only external services, run-scoped route stability, and
  the retention-allowed OpenRouter default without changing judge or algorithm
  ownership.

- 2026-09-07: Accepted independent ownership and attached-service composition
  for production remote judged training.
