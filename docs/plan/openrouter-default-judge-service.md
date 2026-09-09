# Make OpenRouter the default judge service without coupling judging to training

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current
as implementation proceeds.

Maintain this document in accordance with `docs/templates/PLAN.md`. The template
refers to `.agents/PLAN.md`, but that file is not present in this checkout. The
repository instructions in `AGENTS.md`, the template, and the frozen product
baseline in `docs/post-training/` are therefore the governing instructions.

Implementation starts from the clean worktree
`/home/hammad/projects/rl-local-async`, branch
`codex/local-async-source-env`, commit
`4bf6470a0f962bed339b5d058dd2e4624d6208f6`. This is the current Posttrain v0.4
development-release line. It already contains the published v0.4 development
runtime images, current fork closure, hardware-aware job planning, local-source
capsules, and asynchronous-agent work. Do not implement this plan in the older
`/home/hammad/projects/rl` checkout on
`codex/pre-rollout-optimization-baseline`.

## Purpose / Big Picture

After this change, a judged training work package can name a judge once and let
the composition host satisfy it in either of two ways. The default path calls
OpenRouter using the versioned model slug
`deepseek/deepseek-v4-flash-0731`. The alternate path starts or attaches to a
self-hosted judge such as Gemma 4 12B. The Verifiers judge plugin, episode-level
seven-dimensional reward schema, GDPO/CAPO algorithms, and trainer backends do
not change when the service changes.

“Alongside the training run” means a named dependency in the work-package
lifecycle: resolve the judge, prove that it is usable, run training, drain judge
requests, then release only resources owned by the work package. It does not
mean silently loading a judge into the trainer process or placing an unbudgeted
judge on the policy GPU. An OpenRouter dependency performs no deployment. A
managed self-hosted dependency may be deployed as a sibling service on its own
declared execution target. An already-running endpoint is attached and is never
stopped by the training consumer.

The default is intentionally operational, not algorithmic. OpenRouter and
DeepSeek must not appear in `GDPOSettings`, `CAPOSettings`, reward evidence, or
Verifiers rubric semantics. A project can replace the service or model through
a versioned selection without editing its algorithm or judge implementation.

The first visible proof is a local comparison command that replays the same
frozen AutomationBench judge inputs through the OpenRouter DeepSeek default and
the existing Gemma 4 12B reference, produces comparable quality, latency,
token, and cost evidence, and reserves no training GPU. The release proof then
runs a bounded five-update judged GDPO qualification through the default service
and shows a real optimizer update, nonuniform reward components, retained judge
provenance, and an exported adapter.

## Progress

- [x] (2026-09-09 11:51Z) Resolve the correct v0.4 development baseline and
  confirm that commit `4bf6470a` is a strict descendant of the older rollout
  optimization checkout.
- [x] (2026-09-09 11:51Z) Inspect the frozen API, framework, and evidence
  boundaries; ADR 0019; existing managed/attached inference lifecycle; native
  judge injection; remote evaluation service types; development runtime-image
  manifest; current judged AutomationBench work package; and release status.
- [x] (2026-09-09 11:51Z) Resolve the initial product decisions: OpenRouter is
  the default service, `deepseek/deepseek-v4-flash-0731` is the default judge,
  provider retention is allowed, and managed self-hosting remains supported.
- [x] (2026-09-09 12:25Z) Amend the frozen baseline narrowly so an external hosted-model descriptor
  may identify a composition-owned judge dependency while remaining invalid as
  a trainable model, rollout policy, or model-lineage node.
- [x] (2026-09-09 12:25Z) Generalize the inference-service contract and retain a one-release
  compatibility path for the current evaluation-only remote types.
- [x] (2026-09-09 12:25Z) Implement provider-neutral external-service resolution and an OpenRouter
  adapter with credential isolation, capability admission, route freezing,
  cost/usage capture, and redacted evidence.
- [x] (2026-09-09 12:25Z) Make standard judged SAMPO, GDPO, and CAPO definitions accept named
  managed, attached, or external service dependencies without changing trainer
  request contracts.
- [x] (2026-09-09 12:25Z) Publish the default OpenRouter/DeepSeek selections and change the Lab
  judged AutomationBench example to use that default; retain Gemma as an
  explicit alternate profile.
- [x] (2026-09-09 12:25Z) Add local machine credential selection and remote execution secret
  forwarding without putting a secret in catalog, packages, images, logs, or
  run snapshots.
- [x] (2026-09-09 12:25Z) Add the service-neutral episode replay harness and
  prove its managed-versus-external request mapping with unit tests. Retain the
  real fixture and historical-corpus executions as qualification work below.
- [x] (2026-09-09 15:10Z) Add provider-neutral paid-judge cost policy: a USD
  4.99 default hard ceiling, conservative run-wide usage projection, live-price
  admission, an atomic run-local metering gateway, CLI visibility, and focused
  regression tests.
- [x] (2026-09-09 15:35Z) Generalize optional paid-judge composition to the
  full GRPO family and split projection into a training-owned maximum trajectory
  envelope plus a Verifiers-owned per-trajectory judge-call envelope.
- [ ] Replace the one-off comparison scripts with a service-neutral local
  replay harness and qualify DeepSeek against the reviewed fixture and the
  retained multi-run corpus.
- [x] (2026-09-09 15:45Z) Run focused common/catalog/jobs/train/CLI tests,
  Ruff, Pyright, import-boundary checks, diff checks, and the full repository
  suite. The changed surface is green; the full suite retains two unrelated
  baseline failures recorded below.
- [ ] Complete a clean consumer install and resolve or separately baseline the
  two pre-existing full-suite failures before release qualification.
- [ ] Run a five-update real judged GDPO qualification through OpenRouter,
  inspect its reward variation and optimizer evidence, then update the v0.4
  release ledger. Do not promote v0.4 solely from this cell; the other release
  gates in `docs/releases/v0.4.md` remain binding.

## Surprises & Discoveries

- Observation: the v0.4 development artifacts are already published, but the
  public `0.4.0` release is intentionally not tagged.
  Evidence: `packages/runtime-images/src/posttrain/runtime_images/published.toml`
  records framework version `0.4.0` and immutable development image digests,
  while `docs/releases/v0.4.md` says qualification and publication are not
  complete and lists the remaining GPU gates.

- Observation: the existing named judge-service seam is not yet capable of
  representing OpenRouter correctly.
  Evidence: `packages/jobs/src/posttrain/jobs/inference_services.py` requires
  every `ResolvedInferenceService` to carry a local `InferenceBinding`, whose
  identity includes downloadable `ModelVariant` weights, renderer, engine, and
  execution target. An API-only DeepSeek model has no framework-owned weight
  artifact and must not be fabricated as a `ModelVariant`.

- Observation: remote model and service types already exist, but their contract
  is explicitly evaluation-only.
  Evidence: `RemotePolicy` and `ExternalInferenceService` live in
  `packages/eval/src/posttrain/eval/requests.py`; `docs/post-training/05-apis.md`
  says they are not accepted by train, serve, or token-level rollout APIs.
  Reusing them directly from judged training would violate their documented
  meaning even though the wire protocol happens to fit.

- Observation: judged standard jobs currently know only how to start a managed
  local judge.
  Evidence: `structured_rl_definition` and `sampo_definition` construct
  `ManagedInferenceService` objects from extra `InferenceBinding` seats and call
  `bind_managed_native_judge_services`. The lower-level
  `bind_native_judge_services` is already service-neutral once a service is
  resolved, so the refactor can preserve the Verifiers injection boundary.

- Observation: OpenRouter normally routes a model across available providers.
  That uptime behavior is undesirable inside one optimizer run because two
  provider implementations can grade the same episode differently.
  Evidence: OpenRouter documents provider `order`, `allow_fallbacks`, and
  `require_parameters`; disabling fallback after selecting a compatible
  provider creates a stable run-scoped route. The endpoint inventory is
  available from `GET /api/v1/models/:author/:slug/endpoints`.

- Observation: allowing retention does not require Posttrain to turn on remote
  prompt logging.
  Evidence: OpenRouter documents prompt/response logging as an account-level
  opt-in and retains request metadata by default. This plan permits endpoints
  that retain or train on data and does not require ZDR, but it does not mutate
  account privacy or observability settings as a side effect of job submission.

- Observation: the current workstation has no configured OpenRouter credential
  source and the submitting shell has no `OPENROUTER_API_KEY`.
  Evidence: the read-only job plan reports
  `runtime_credentials.OPENROUTER_API_KEY=unavailable`; the opt-in live contract
  test skips before opening a paid request. The implementation and offline
  tests can proceed, but fixture replay and five-update qualification require
  the operator to populate the protected machine credential source.

- Observation: the environment-owned judge performs its completion calls after
  Posttrain injects a generic OpenAI-compatible endpoint, so catalog metadata
  alone cannot enforce spend.
  Evidence: Verifiers `Judge.complete()` creates its own async client and records
  returned usage on the native trace. Posttrain now gives it a loopback metered
  endpoint whose upstream credential and atomic cost ledger remain composition
  owned; the judge and algorithm APIs remain unchanged.

- Observation: the existing 50-update, 16-by-4 GDPO work package is correctly
  too large for the new default ceiling when every bounded recovery path is
  included.
  Evidence: its resolved projection is 19,200 judge requests, 157,286,400 input
  tokens, and 314,572,800 output tokens. At the currently tested endpoint rates
  of USD 0.00000005/input token and USD 0.00000016/output token, the conservative
  maximum is USD 58.195968. It must not run on the USD 4.99 selection; a five-step
  cell must reduce its bounded population or explicitly select a higher ceiling.

- Observation: the complete repository suite no longer reports an ownership
  gap for the new replay harness, but it is not globally green for two failures
  outside this change.
  Evidence: `uv run pytest -q` reports 1,626 passed and 24 skipped, with failures
  in `test_terminal_rollout_evidence.py` (an older fixture lacks the newer
  `trace.info` field) and `test_work_packages.py` (the base-only catalog cannot
  resolve a Lab overlay distillation selection). Focused changed-surface tests
  report 497 passed and 10 skipped; Ruff, Pyright, all eight import contracts,
  and `git diff --check` pass.

## Decision Log

- Decision: continue from v0.4 development commit `4bf6470a`, not from the
  older `codex/pre-rollout-optimization-baseline` branch.
  Rationale: all recent async, release, hardware-planning, source-capsule, and
  runtime-image work is already committed and pushed on the newer branch.
  Date/Author: 2026-09-09 / Codex and user.

- Decision: make OpenRouter the default judge service and pin the default model
  selector to `deepseek/deepseek-v4-flash-0731`.
  Rationale: it avoids reserving a second GPU for ordinary judged training and
  is substantially cheaper for the expected judge-token volume. The dated slug
  is more reproducible than `deepseek/deepseek-v4-flash` or a `latest` alias.
  Date/Author: 2026-09-09 / user, with Codex specifying the versioned slug.

- Decision: treat provider retention as allowed, set no ZDR requirement, and do
  not reject providers because they may retain or train on request data.
  Rationale: the user explicitly does not require a data-retention restriction.
  This is a routing policy, not permission to expose secrets or to silently
  enable account-level prompt logging.
  Date/Author: 2026-09-09 / user and Codex.

- Decision: require every hosted inference binding to name both an exact API
  model and an exact provider slug. Validate that pair against live inventory,
  freeze it for the training run, and set `allow_fallbacks` to false.
  Rationale: model-name stability is insufficient because provider drift can
  introduce reward-distribution drift inside an optimizer run. Automatically
  choosing the cheapest or currently available provider also makes the reward
  implementation an undeclared runtime choice. A missing or incompatible pair
  fails admission; a provider change is allowed only as a new provider attempt
  with an explicit evidence boundary.
  Date/Author: 2026-09-09 / Codex.

- Decision: use one provider-neutral hosted-model/service identity shared by
  eval and composition, but do not make it a `ModelVariant`.
  Rationale: an API-only model has no loadable artifact digest, tokenizer, or
  framework-owned lineage. Duplicating the same descriptor in eval and jobs
  would create drift; pretending it is a local model would create false
  reproducibility.
  Date/Author: 2026-09-09 / Codex.

- Decision: keep lifecycle variants distinct: managed, attached, and external.
  Rationale: managed services are started and stopped by the host; attached
  services are pre-existing deployments whose selected artifacts can be
  verified; external services such as OpenRouter expose a hosted policy but no
  Posttrain-owned deployment. A single `owned: bool` cannot express all three
  provenance and cleanup rules safely.
  Date/Author: 2026-09-09 / Codex.

- Decision: preserve judge-plugin ownership of prompts, rubrics, response
  schema, episode compaction, parsing, and the seven episode-level scores.
  Rationale: model hosting and request transport must be replaceable without
  changing task semantics or algorithm design. GDPO continues to consume the
  validated component vector and its configured weights; task completion keeps
  its larger configured credit without being confused with service defaults.
  Date/Author: 2026-09-09 / Codex.

- Decision: compare judges locally before spending GPU time on training.
  Rationale: frozen judge inputs can measure correctness, saturation, schema
  validity, latency, tokens, and cost through a remote API without a policy GPU.
  A GPU is required only for the later optimizer qualification.
  Date/Author: 2026-09-09 / user and Codex.

- Decision: cap every API-paid judge service at USD 4.99 per run by default.
  Before a paid readiness probe, price the conservative maximum derived from
  loop steps, prompt groups, generations, collection attempts, judge attempts,
  context fan-out, and input/output budgets. Route admitted traffic through an
  atomic reservation gateway and require an explicit versioned hosted-inference
  binding to raise the ceiling.
  Rationale: a warning or post-hoc usage report cannot prevent concurrent calls
  and retries from crossing a spend limit. The limit is operational service
  policy and must not alter reward weights, rubrics, GDPO, CAPO, or SAMPO.
  Date/Author: 2026-09-09 / user and Codex.

- Decision: paid judge composition is algorithm-independent.
  Rationale: Verifiers owns judge models, rubrics, retries, and reward shape. A
  scalar-reward GRPO/DAPO/OLMo run can use the same judge service as SAMPO,
  GDPO, or CAPO. Training publishes only a generic maximum trajectory count;
  Verifiers publishes calls and token bounds per trajectory; composition joins
  them for cost control.
  Date/Author: 2026-09-09 / user and Codex.

## Outcomes & Retrospective

Implementation outcome: the v0.4 development line now has a concrete boundary
between algorithm, judge plugin, service orchestration, provider transport, and
paid-service cost control. Offline tests prove that an over-budget projection is
rejected before the paid probe, concurrent reservations cannot cross the limit,
the upstream credential stays behind the loopback gateway, and the standard job
projection includes bounded retries. Real OpenRouter fixture comparison and GPU
qualification remain blocked on the protected runtime credential.
The same service is now composable with the whole GRPO family, and the replay
harness has a declared Lab ownership/exit path. Full release validation still
has the two unrelated baseline failures described above.

## Context and Orientation

Posttrain separates a training algorithm from the system that generates judge
responses. The external `carbonteq-ai/verifiers-environments` package owns the
AutomationBench judge plugin. That plugin constructs the rubric and returns
seven whole-episode components: five reasoning-quality dimensions plus action
quality and answer quality. `packages/train` validates those components and
implements GDPO or CAPO credit assignment. Neither layer should know whether
the answer came from OpenRouter, a local vLLM process, or a separately deployed
Gemma service.

That seven-component plugin is one use, not the definition of a judge. Another
Verifiers plugin may return one scalar reward to GRPO, DAPO, or OLMo-style GRPO,
or annotate a SAMPO trajectory. The same service composition and cost control
applies; only the algorithm-side reward consumer differs.

`packages/jobs` is the composition layer. Its
`packages/jobs/src/posttrain/jobs/inference_services.py` resolves named service
dependencies. Its `packages/jobs/src/posttrain/jobs/native_judges.py` injects a
resolved endpoint and ephemeral API-key variable into a copied Verifiers
configuration. Its `packages/jobs/src/posttrain/jobs/definitions.py` defines
the standard GRPO-family, SAMPO, GDPO, and CAPO jobs. Their judged variants
accept the same named service seat and can resolve a managed local model, an
attached deployment, or an API-only external service.

`packages/eval` already supports an API-only evaluation subject through
`RemotePolicy`, `ExternalInferenceService`, and `RemoteEvaluationBinding` in
`packages/eval/src/posttrain/eval/requests.py`. Those values correctly avoid a
fake local weight artifact, but the frozen baseline deliberately limits them to
evaluation. This plan extracts only the generic hosted-model and OpenAI-service
identity into a framework-neutral location and leaves
`RemoteEvaluationBinding` in eval. Judged training consumes the external model
only as a composition dependency, never as the trainable policy or rollout
binding.

A service selection is a secret-free, versioned declaration. A service
connection is an ephemeral runtime value containing an address and resolved
credential. A service receipt is retained evidence describing what was
selected and observed without containing the credential. A route receipt is the
external-provider part of that evidence: requested model, requested provider,
resolved endpoint,
parameter capabilities, price snapshot, routing policy, and request protocol.

The current Lab example is
`apps/lab/.posttrain/work_packages/lfm26_automationbench_gdpo_episode_50.yaml`.
It binds `judge_inference` to a managed Gemma 4 12B vLLM selection from
`apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`. The new
default example must instead bind a named external judge service. The Gemma
binding remains as an explicit alternate work package so the two paths can be
compared without editing YAML in place.

ADR `docs/decisions/0019-auxiliary-inference-service-ownership.md` already
requires production judged training to use an independently owned dependency,
requires readiness before training, forbids turning service failure into zero
reward, and keeps lifecycle in composition. Amend it to cover external API
providers and route receipts; do not replace its ownership decision.

## Plan of Work

### Milestone 1: amend the contract before code

Update `docs/post-training/02-primitives.md`,
`docs/post-training/04-framework.md`, `docs/post-training/05-apis.md`, and
`docs/post-training/06-observation-and-lineage.md`. State that an API-only
hosted-model identity may be used for an evaluation subject or an auxiliary
inference dependency such as a judge, but remains invalid for model lineage,
training policy, token-level rollout, serving, or export. State that composition
owns managed/attached/external service resolution, while the environment owns
judge semantics and train owns credit. Define the route evidence that must be
retained and the rule that a provider change cannot be hidden within one
optimizer run.

Update ADR 0019 rather than creating a competing ADR. Add external-provider
resolution as a third lifecycle variant. Clarify that “alongside” is dependency
ordering, not same-process or same-GPU colocation. Record that the default
OpenRouter policy allows retention and does not require ZDR. This is a narrow
baseline amendment because the current API text says the hosted-policy
descriptor is evaluation-only; it does not change reward or training semantics.

Acceptance for this milestone is a documentation review in which every one of
these cases has one unambiguous owner: managed Gemma, attached Gemma, external
OpenRouter DeepSeek, provider change, missing credential, invalid judge output,
and work-package cancellation.

### Milestone 2: introduce one truthful external inference identity

Move the provider-neutral portions of `RemotePolicy` and
`ExternalInferenceService` out of `packages/eval` into
`packages/common/src/posttrain/common/`. Use names that describe their broader
meaning, such as `HostedModel` and `ExternalInferenceService`. The final name
must not mention OpenRouter and must not imply owned weights. A `HostedModel`
contains a stable selection id, a provider/model revision string, the exact API
model identifier, context window, and declared capabilities. It has no artifact
URI, tokenizer digest, execution target, or model-lineage methods.

An `ExternalInferenceService` contains stable id/revision, protocol
`openai-chat@1`, credential reference, safe base URL, safe non-secret headers,
safe request defaults, and explicit provider policy. The provider policy must be
JSON-compatible but validate the cross-provider controls Posttrain depends on:
whether fallback is allowed, whether parameter support is required, retention
restriction if any, and a run-stability policy. Keep provider-specific endpoint
inventory parsing out of common.

Keep `RemoteEvaluationBinding` in `packages/eval`. Re-export the old
`RemotePolicy` name as a deprecated alias for one v0.4 compatibility interval,
update catalog schemas to decode old and new data, and add a removal note. Do
not make `packages/train` import common external inference types; only jobs and
eval need them.

Generalize `packages/jobs/src/posttrain/jobs/inference_services.py` so
`ResolvedInferenceService` carries a discriminated selection:

    type InferenceServiceSelection = InferenceBinding | HostedInferenceBinding

    @dataclass(frozen=True, slots=True)
    class HostedInferenceBinding:
        id: str
        revision: str
        model: HostedModel
        service: ExternalInferenceService
        max_cost_usd_micros: int = 4_990_000
        sampling: Mapping[str, JsonValue]
        purpose: tuple[Literal["judge"], ...]

    @dataclass(frozen=True, slots=True)
    class ExternalInferenceServiceRequest:
        binding: HostedInferenceBinding
        usage: ExternalInferenceUsageProjection

The exact class names may change once existing catalog conventions are applied,
but the distinctions may not. Managed and attached local services retain an
`InferenceBinding`; an external request retains a `HostedInferenceBinding` and
has no execution target. `ResolvedInferenceService.trace_identity()` returns
one stable schema with `lifecycle` equal to `managed`, `attached`, or
`external`, plus type-appropriate model identity. It must never invent an
artifact digest for a hosted model.

Add catalog schema support in `packages/catalog` or the current owning schema
module and add serialization round-trip tests. The core types remain provider
neutral and import no OpenRouter SDK.

### Milestone 3: resolve OpenRouter as an external service

Add an internal adapter under
`packages/jobs/src/posttrain/jobs/providers/openrouter.py` (or an equivalently
private provider directory). Use the repository's existing HTTP dependency; do
not add a provider SDK unless inspection proves it reduces code and lock weight.
The adapter must:

1. resolve `OPENROUTER_API_KEY` only at runtime;
2. query the versioned model endpoint inventory and reject a missing or changed
   requested model before training;
3. retain declared context, supported parameters, endpoint/provider slug,
   quantization when reported, latency, availability, and current input/output
   price metadata as a timestamped snapshot;
4. require the hosted binding's explicit provider slug, verify that the
   requested model/provider pair has a healthy compatible endpoint, set
   `provider.order` to that provider, set `allow_fallbacks: false`, and set
   `require_parameters: true` for the judge's structured-output and reasoning
   requirements;
5. omit `zdr` and permit data-collection/retention routes by default, while
   preserving an explicit optional restrictive policy for projects that need
   it later;
6. send a minimal structured readiness request that exercises the exact system
   role, response schema, reasoning control, and output-budget fields used by
   the judge rather than trusting a model-list response alone;
7. return an ephemeral connection plus a secret-free route receipt; and
8. classify 401/403 as credential/admission failures, unsupported parameters as
   configuration failures, 429/5xx/timeouts as service failures, and invalid
   model output as scorer failures.

Before item 6 performs its paid readiness request, the adapter must price the
request's aggregate input/output-token projection against the selected endpoint
and reject any value above `HostedInferenceBinding.max_cost_usd_micros`. Once
admitted, expose a run-local OpenAI-compatible gateway rather than the upstream
URL. It atomically reserves each request's conservative maximum charge before
dispatch, reconciles valid reported usage, and conservatively retains the
reservation when usage is unavailable. A 402 response with code
`posttrain_judge_cost_limit` is a service-budget failure, not a judge score.

Do not add automatic semantic retries at the algorithm layer. The existing
judge plugin may perform its declared bounded parse/transport attempts under a
stable request identity. Retain every attempt's status, token usage, latency,
request id, reported model, and provider. If a response reports a different
provider or model from the frozen route, quarantine that assessment and fail or
drop the affected rollout according to existing group-admission policy; never
coerce it to zero or mix it into the group.

An external service has no process to stop. Its context manager emits resolved,
ready, and released events for consistent lifecycle evidence, but release only
closes the HTTP client and drains outstanding requests. Managed services still
stop in reverse order; attached services remain untouched.

### Milestone 4: compose judged jobs from named service seats

Refactor `packages/jobs/src/posttrain/jobs/native_judges.py` so validation and
injection work from the normalized service identity rather than assuming
`service.inference.model` is a `ModelVariant`. Validate a local judge against
its model revision and sampling. Validate an external judge against its hosted
model revision, exact API model id, and service sampling. Preserve the copied
activation, ephemeral per-service credential variables, multi-plugin sharing,
and cleanup behavior.

Refactor `sampo_definition` and `structured_rl_definition` in
`packages/jobs/src/posttrain/jobs/definitions.py` to accept declarative named
service seats. Do not add a provider argument to these algorithm job factories.
A host-side resolver converts each selected service seat into a managed,
attached, or external request before entering `bind_inference_services`.
Retain the existing managed-only helper as a compatibility wrapper, then migrate
all Lab callers and remove it only after no catalog or test uses it.

The execution graph must be observable as:

    resolve seats
      -> resolve/provision every named judge service
      -> run exact capability probes
      -> inject ephemeral connections into copied Verifiers config
      -> admit and run training
      -> drain scoring requests and retain native traces
      -> close external clients and stop only owned managed services

If any required judge is unavailable before training, no policy rollout or
optimizer work starts. If it fails during a batch, existing relaxed admission
drops only invalid rollouts/groups within declared bounds; a batch is not
discarded merely because one assessment failed. If the admissible population
cannot be completed, the update does not occur and the run reports an explicit
incomplete-population failure.

### Milestone 5: publish defaults and credential DX

Add versioned hosted-model, external-service, and hosted-judge binding entries
to the appropriate base catalog YAML. The intended initial selections are:

    hosted-models/deepseek-v4-flash-0731@1
      api_model: deepseek/deepseek-v4-flash-0731
      revision: 0731
      capabilities: [system-role, reasoning, structured-output]

    external-services/openrouter@1
      base_url: https://openrouter.ai/api/v1
      protocol: openai-chat@1
      api_key_var: OPENROUTER_API_KEY
      retention: allowed
      require_zdr: false

    hosted-inference/deepseek-v4-flash-openrouter-judge@1
      model: hosted-models/deepseek-v4-flash-0731@1
      service: external-services/openrouter@1
      provider: open-inference/fp8
      purpose: [judge]
      route: {freeze_provider_per_run: true, allow_fallbacks: false,
              require_parameters: true}

Use the repository's actual catalog family naming rules; the illustrative ids
above describe required semantics, not permission to invent an inconsistent
family. Do not put a mutable price into selection identity. Capture price from
the live route receipt because provider prices can change without a Posttrain
release.

Extend the machine-configuration path documented in
`docs/plan/dx-configuration-authority.md` so a named `openrouter-default`
credential source permits only `OPENROUTER_API_KEY`. Local execution loads it
from a mode-0600 machine file. Dstack or another remote executor forwards only
that named variable through its secret mechanism. The packer and image builder
must reject the secret as file content, image environment, catalog data,
command-line argument, event field, or serialized run snapshot. A plan command
must report `OpenRouter credential: configured/unavailable`, never its value.

Change the judged LFM AutomationBench work package to the OpenRouter binding and
rename its description from “managed Gemma” to provider-neutral wording. Add a
second explicit Gemma work package/profile for self-hosted comparison. Do not
delete the immutable Gemma model or inference binding because it remains the
self-hosted option and reference judge.

The default output budget remains the judge plugin's 16,384 tokens with an
8,192-token input allowance and a 24,576-token minimum context contract. The
hosted model advertises a larger context, but Posttrain records and enforces the
smaller purpose-specific budget. A provider that cannot honor the exact output
budget or structured schema is rejected during preflight rather than silently
truncating judgments.

### Milestone 6: compare judge quality locally

Turn `scripts/qualification/replay_episode_judge.py`,
`scripts/qualification/calibrate_general_episode_prompt.py`, and related helper
logic into one service-neutral replay harness. Preserve existing command
compatibility where practical. It must accept a catalog judge binding rather
than a hard-coded model/base URL pair, materialize one frozen input corpus once,
and execute several bindings against byte-identical judge inputs. The harness
runs on the local CPU and network; OpenRouter performs DeepSeek inference. A
self-hosted Gemma endpoint may be attached if already available, but the harness
must not reserve the RTX PRO 6000 merely to test OpenRouter.

Use two input sets. First, run
`scripts/qualification/fixtures/general_episode_judge_candidate_v1.json` (or
its reviewed successor) to test known defects and score bounds. Second, collect
the already-retained episode inputs from the prior Nanbeige/Gemma runs into a
content-addressed manifest without rewriting or retokenizing trajectories. Do
not expose native benchmark oracle answers to the judge if the prior harness
excluded them.

For every model, report first-pass schema validity, final validity, timeout and
truncation counts, per-dimension score distributions, pairwise discrimination
on seeded defects, uniform-score saturation, input/reasoning/output tokens,
p50/p95 latency, concurrency achieved, provider route, and estimated cost from
the route snapshot. Preserve raw responses and parse errors by immutable
reference. A judge passes only if it satisfies every reviewed fixture bound,
does not collapse all seven dimensions to one value across the defect corpus,
has no unexplained route drift, and meets the plugin's output contract. Report
DeepSeek and Gemma side by side; do not average the seven reward dimensions into
one judge-quality number.

### Milestone 7: qualify the default in real training and close release evidence

After the local comparison passes, run one bounded five-update GDPO job using
the existing LFM 2.5 2.6B AutomationBench policy configuration, 8 prompt groups
by 4 generations, rollout concurrency 32, and the OpenRouter judge service. The
judge concurrency is independently capped and tuned from the local OpenRouter
latency/rate evidence; do not assume the old Gemma concurrency of 16 or 32 is
valid for an external account limit.

Acceptance requires five completed optimizer updates, complete admitted prompt
groups, nonuniform raw seven-component reward vectors on a task-diverse sample,
the configured task-completion-heavy GDPO weights, finite nonzero learning
signal where reward variation exists, checkpoint/reload, exported adapter
inference, native Verifiers traces, and a secret-free service/route receipt.
Compare observed time and cost with the retained Gemma reference. Do not claim
algorithm correctness from judge agreement; numerical GDPO correctness remains
covered by deterministic tests, while this run qualifies the integration.

Update `docs/plan/gdpo-capo-dual-backend-support.md`,
`docs/releases/v0.4.md`, and this plan with exact commits, run ids, immutable
artifacts, observed metrics, and remaining gates. Regenerate and publish a new
development candidate image only after source and lock changes pass locally.
Do not overwrite the already-published manifest by hand and do not tag v0.4
until all independent release gates are complete.

## Concrete Steps

All commands below run from `/home/hammad/projects/rl-local-async` unless a
different directory is shown. Before every implementation session, verify the
checkout:

    test "$(git rev-parse --show-toplevel)" = /home/hammad/projects/rl-local-async
    git branch --show-current
    git status --short

Expect branch `codex/local-async-source-env`. If the branch has advanced, record
the new commit in Progress; do not reset it to `4bf6470a`.

After the contract/type milestone, run focused tests first:

    uv run pytest packages/common/tests packages/eval/tests/test_api.py \
      packages/jobs/tests/test_inference_services.py \
      packages/jobs/tests/test_native_judges.py -q
    uv run lint-imports
    uv run pyright packages/common packages/eval packages/jobs

Add HTTP-adapter tests using a local fake server and run:

    uv run pytest packages/jobs/tests/test_openrouter.py \
      packages/jobs/tests/test_inference_services.py \
      packages/jobs/tests/test_native_judges.py -q

Add a planning/preflight command that performs no paid completion and confirm a
missing credential fails before job submission. Then run the one paid readiness
probe only when `OPENROUTER_API_KEY` is available:

    uv run pytest -m network packages/jobs/tests/test_openrouter_live.py -q

The expected live evidence names the exact requested model, one provider slug,
`allow_fallbacks=false`, structured-output support, token usage, and no secret.

Run the local frozen-corpus comparison with the final CLI exposed by the
harness. The implementation must update this command to its exact catalog ids:

    uv run python scripts/qualification/compare_episode_judges.py \
      --work-package apps/lab/.posttrain/work_packages/lfm26_automationbench_gdpo_episode_50.yaml \
      --fixture scripts/qualification/fixtures/general_episode_judge_candidate_v1.json \
      --output artifacts/judge-comparison/openrouter-deepseek-v4-flash-0731

If a Gemma endpoint is available, run the same materialized input manifest with
its attached binding and then generate one comparison report. Do not regenerate
the judge inputs between arms.

Before GPU qualification, execute the repository validation ladder:

    uv sync --all-packages --locked --python 3.13
    uv run ruff check .
    uv run pyright
    uv run lint-imports
    uv run pytest
    git diff --check

Then use `posttrain job plan` against the five-update work package. The plan
must show one policy/rollout target, one external judge dependency with no GPU
target, an available named credential, the frozen model/provider policy, and no
secret. Submit only after that output is correct. Record the exact final command
and provider job id here before launch because the current command surface may
advance during implementation.

## Validation and Acceptance

The type and catalog tests must prove that hosted models never acquire local
artifact lineage; old remote-evaluation YAML still resolves; new external judge
YAML round-trips; a hosted binding cannot occupy the policy, rollout, or serve
seat; and a local `InferenceBinding` cannot be silently treated as an external
model.

The lifecycle tests must prove all three variants. Managed services start once,
probe once, share across named plugins, and stop in reverse order. Attached
services probe but are never stopped. External services resolve one route,
probe the actual judge capability, share one client/credential variable, drain,
and close without a deployment stop. Every exceptional exit restores the
process environment exactly.

The security tests must prove that credentials do not appear in catalog
serialization, work-package snapshots, packed source, OCI build inputs,
subprocess arguments, events, exception strings, native traces, or service
receipts. Only the ephemeral judge client receives the bearer value.

The OpenRouter contract tests must cover authentication failure, unknown model,
no compatible endpoint, unsupported structured output, response truncation,
429, timeout, invalid JSON, provider drift, and a valid response with separate
reasoning and output token counts. Retryable transport failure and invalid
assessment must remain different states. Neither may produce numeric zero.

The local comparison is accepted when all reviewed defect fixtures satisfy
their declared per-dimension bounds, the seven score columns show meaningful
variation where fixtures differ, every scored row cites raw response evidence,
and cost/latency/token totals reconcile to individual attempts. “DeepSeek is
better” is not a required result; if it fails calibration, the framework work
can still merge with Gemma remaining the selected project override, but the
OpenRouter/DeepSeek pair cannot be called qualified. Because the user selected
it as the product default, public default activation is blocked until it passes.

The five-update training run is accepted only with observed optimizer steps 1
through 5, a checkpoint and reload, an exported adapter that generates through
the selected renderer, nonuniform judge components, and complete retained
service provenance. A successful HTTP request, a zero-loss update, or five
loop iterations without parameter change is insufficient.

## Idempotence and Recovery

All catalog entries and tests are additive until consumers migrate. Running the
local comparison again writes a new run directory keyed by selection and input
manifest digest; it never overwrites prior evidence. Price and provider
inventory are snapshots, so a repeat may differ and must retain a new receipt.

The OpenRouter adapter must reuse one idempotent logical request id for a bounded
transport retry. A process restart may replay an unconsumed assessment but must
not duplicate an assessment already acknowledged by the training producer.
Existing structured-reward consumption markers remain authoritative.

If a provider route disappears between planning and execution, fail before the
first optimizer update and create a new provider attempt after re-resolution.
Do not silently enable fallback. If the external service fails during scoring,
drain completed assessments, retain the failure evidence, and let existing
group admission decide whether a complete update population remains. Resume
only from a complete checkpoint whose scorer, rubric, hosted model, route
policy, and reward-schema identities match.

If implementation of the shared hosted types breaks remote evaluation, retain
the compatibility alias and decoder until both old fixtures and the new
external-judge path pass. Do not delete the prior eval schema as a shortcut.

Do not delete or retag the current development images. Source changes produce a
new content-addressed candidate after local validation. If publication fails,
repair the builder/trust path and retry from the immutable plan; never bypass
TLS or hand-edit `published.toml`.

## Artifacts and Notes

The current implementation baseline is:

    worktree: /home/hammad/projects/rl-local-async
    branch: codex/local-async-source-env
    commit: 4bf6470a0f962bed339b5d058dd2e4624d6208f6
    v0.4 dev image publication commit: 291cbd35
    v0.4 status: development artifacts published; public release not tagged

The default service policy is:

    service: OpenRouter
    base URL: https://openrouter.ai/api/v1
    model: deepseek/deepseek-v4-flash-0731
    credential variable: OPENROUTER_API_KEY
    provider route: explicit open-inference/fp8 endpoint selection, validate once, pin for the run, no cross-provider fallback
    required parameters: true
    zero-data-retention requirement: false
    provider retention/training restriction: none
    account-level prompt logging: unchanged by Posttrain
    default hard run cost ceiling: USD 4.99
    higher ceiling: explicit versioned hosted-inference binding only

Mutable prices are evidence, not configuration. At qualification time, record
the endpoint inventory's current rates and timestamp in the route receipt and
compute cost from actual token usage. Do not embed a chat-era price estimate in
algorithm settings or acceptance logic.

Relevant existing files are:

    docs/releases/v0.4.md
    docs/decisions/0019-auxiliary-inference-service-ownership.md
    packages/eval/src/posttrain/eval/requests.py
    packages/jobs/src/posttrain/jobs/inference_services.py
    packages/jobs/src/posttrain/jobs/native_judges.py
    packages/jobs/src/posttrain/jobs/definitions.py
    apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml
    apps/lab/.posttrain/work_packages/lfm26_automationbench_gdpo_episode_50.yaml
    scripts/qualification/replay_episode_judge.py
    scripts/qualification/calibrate_general_episode_prompt.py
    scripts/qualification/fixtures/general_episode_judge_candidate_v1.json

The external provider behavior used by this plan was checked on 2026-09-09
against OpenRouter's official provider-routing, endpoint-listing, privacy, and
Responses API documentation. Treat those network APIs as versioned integration
surfaces and protect them with live contract tests; do not assume documentation
observations remain forever true.

## Interfaces and Dependencies

The stable public concepts after implementation are a hosted model identity, an
external OpenAI-compatible service descriptor, a hosted judge inference
binding, and one normalized resolved service. The exact Python names must follow
the repository's naming review, but their ownership is fixed:

    posttrain.common
      HostedModel
      ExternalInferenceService

    posttrain.eval
      RemoteEvaluationBinding
      RemotePolicy  # temporary compatibility alias only

    posttrain.jobs
      HostedInferenceBinding
      ManagedInferenceService
      AttachedInferenceService
      ExternalInferenceServiceRequest
      ExternalInferenceUsageProjection
      ResolvedInferenceService
      bind_inference_services(...)
      bind_native_judge_services(...)

Provider adapters implement an internal resolver protocol shaped like:

    class ExternalServiceResolver(Protocol):
        def resolve(
            self,
            context: RunContext,
            request: ExternalInferenceServiceRequest,
        ) -> AbstractContextManager[ResolvedInferenceService]: ...

`ResolvedInferenceService` exposes a secret-bearing ephemeral endpoint only to
the binding context and a separate `trace_identity()` that is guaranteed
secret-free. The receipt schema includes lifecycle, selection ids/revisions,
requested and reported model ids, protocol, safe origin, route policy,
provider, capability snapshot, price snapshot, readiness result, and timestamps.

OpenRouter remains a private adapter selected by the service id. No OpenRouter
type or request field enters `posttrain.train`, the environment reward schema,
or GDPO/CAPO settings. The Verifiers judge continues to receive only its generic
OpenAI-compatible `model`, `base_url`, `api_key_var`, `sampling`, and safe
headers. TRL and veRL continue to consume the same validated structured reward
contract.

The standard registry also exposes `train/grpo-family-judged@1`. It accepts the
same optional named judge inference seat for scalar GRPO, DAPO, and OLMo-style
GRPO without adding judge fields to `GRPOSettings`.

Revision note, 2026-09-09: created this plan after confirming that the correct
baseline is the v0.4 development worktree at `4bf6470a`. It replaces the earlier
informal idea of “deploy OpenRouter alongside training” with three explicit
lifecycle variants, an OpenRouter/DeepSeek default, retention-allowed routing,
local judge comparison, and a five-update release qualification gate.

Revision note, 2026-09-09: added the user-required sub-USD-5 default judge cost
control. The plan now distinguishes mutable live prices from an immutable
selected cost ceiling and requires both conservative preflight admission and
concurrency-safe runtime enforcement, because either mechanism alone can still
permit an unintended bill.

Revision note, 2026-09-09: clarified that judges are Verifiers-owned and usable
with every compatible RL algorithm. Projection now joins two neutral envelopes
at composition instead of branching on an algorithm to understand judge work.
