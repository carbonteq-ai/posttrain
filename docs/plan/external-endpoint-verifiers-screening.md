# External-endpoint Verifiers screening without a framework fork

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` accurate as the work proceeds.
It follows [docs/templates/PLAN.md](../templates/PLAN.md), the execution-plan
format required by the repository's agent guide.

## Purpose / Big Picture

After this work, a project can screen a model that is served by OpenAI,
OpenRouter, or another supported external service through the same native
Verifiers environment used for local evaluation. The resulting run retains the
native Verifiers trace bundle, records the requested and observed remote model
identity, and states whether the exact model-service combination is suitable
for a tool-using, multi-turn environment. It does not turn an external API
model into a trainable policy and it does not make Posttrain own a second agent
loop.

The behavior must be visible without a paid API call through fake-client tests
and a local OpenAI-compatible test server. A real, credential-gated OpenRouter
probe is the final integration gate. The first supported remote protocol is
OpenAI Chat Completions. OpenRouter uses that protocol, including its tool-call
and reasoning-detail extensions, so it is a normal binding of this protocol,
not a Posttrain-specific provider fork.

## Progress

- [x] (2026-07-31 14:06Z) Inspected the frozen framework baseline, the current
  `posttrain.eval` request and Verifiers adapter, the standard job definitions,
  and the pinned upstream Verifiers client/dialect implementation.
- [x] (2026-07-31 14:06Z) Created this implementation plan and recorded the
  decision not to create an owned Verifiers fork for OpenAI-compatible
  screening.
- [x] (2026-07-31 14:18Z) Amended the frozen primitive/API wording to
  distinguish an evaluation subject from a trainable `ModelVariant` while
  preserving the local model path.
- [x] (2026-07-31 14:18Z) Added typed remote-policy, external-service, and
  remote-binding values plus the detached `remote-evaluation` catalog decoder.
- [x] (2026-07-31 14:18Z) Routed remote bindings through `posttrain.eval`, the
  standard external evaluation job definition, and Verifiers' existing eval
  client configuration.
- [x] (2026-07-31 14:18Z) Added requested subject/service/protocol attributes
  to events, artifacts, and traces; trace attributes include the observed model
  only when the native record reports one.
- [x] (2026-07-31 14:18Z) Added fake and local native-config tests. Focused
  tests passed with the pinned Verifiers extra; static checks and import-boundary
  checks are clean.
- [x] (2026-07-31 14:42Z) Made the immutable source snapshot reject
  `posttrain.env` in addition to `.env`, so a project-local secret binding
  cannot be accidentally included by a future broad source include.
- [x] (2026-07-31 14:44Z) Updated the consumer setup guide and starter
  `.gitignore` for project-owned `posttrain.env`; planning now resolves its
  `POSTTRAIN_REGISTRY` value as project-owned configuration.
- [x] (2026-07-31 14:44Z) Made the configured project environment authoritative
      and added `--registry` as an explicit one-command publication destination
      override. Ambient shell exports no longer alter a configured project.
- [x] (2026-08-01) Preserved the implementation on
      `codex/external-endpoint-screening`. Ruff, Pyright, and all eight import
      contracts pass. The pre-merge full suite reports 15 legacy runtime-image
      tests that still inject an ambient registry despite the new authority
      contract, plus one Lab test whose AutomationBench package is absent from
      the main-branch environment; these are resolved against the newer 0.3.0
      DX branch rather than weakening the configuration contract here.
- [ ] Add Ambient Agent's first OpenRouter screening work package without
  embedding a credential.
- [ ] Run the credential-gated OpenRouter probe when `OPENROUTER_API_KEY` is
  intentionally supplied, update this plan with the result, and publish the
  framework release only after review.

## Surprises & Discoveries

- Observation: the pinned upstream Verifiers package already has a generic
  external evaluation client; it is not a local-vLLM-only client.
  Evidence: Ambient Agent pins upstream Verifiers commit
  `284a868d6a9022109b749710672a0460e8a996d4`. Its `EvalClient` accepts an
  absolute `base_url`, API-key environment-variable name, and extra headers.

- Observation: the pinned chat dialect already preserves the important
  OpenRouter reasoning continuation state.
  Evidence: `verifiers.v1.dialects.chat` recognizes `reasoning`,
  `reasoning_content`, and `reasoning_details`; it places the latter in opaque
  `AssistantMessage.provider_state` and replays the verbatim assistant message
  on a chat continuation. This is the correct Verifiers-owned behavior.

- Observation: current Posttrain external evaluation is incomplete as a
  product surface.
  Evidence: `packages/eval/.../requests.py` has `EvaluationEndpoint` with only
  URL, model string, and key-variable fields. The Verifiers adapter forwards
  none of the existing client's optional headers or provider request defaults.
  `packages/jobs/.../definitions.py` constructs a local `ServeLaunchRequest`
  endpoint even for the explicitly external definition, so an external service
  cannot yet be selected as a first-class work-package seat.

- Observation: Verifiers accepts provider request extensions through its
  `SamplingConfig` rather than a separate request-default field.
  Evidence: the pinned `SamplingConfig` permits extra JSON keys, and the chat
  dialect applies its validated sampling object after setting the requested
  model. Posttrain now rejects evaluation-owned fields (`model`, messages,
  tools, and normal sampling controls) from service defaults and passes safe
  provider extensions such as `provider` through this established channel.

- Observation: the existing framework job definition for external evaluation
  was not an external endpoint seat in practice.
  Evidence: it constructed `ServeLaunchRequest(inference)`, whose endpoint is
  `http://127.0.0.1:8000/v1`. The new
  `eval/verifiers-remote-general@1` definition accepts a
  `remote-evaluation` seat and passes `endpoint=None`; `EvaluateRequest`
  resolves the endpoint only from its declared remote service.

- Observation: source snapshots rejected `.env` but not the framework's
  documented `posttrain.env` name.
  Evidence: the package snapshotter and staged-source validator both compared
  only against `.env`; an explicit broad source include could therefore have
  copied a project execution credential file into an image. Both guards now
  reject `posttrain.env`, with focused coverage.

- Observation: the protected execution file was forwarded to jobs but its
  registry value was not used while planning an image.
  Evidence: `load_local_execution_config` derived the registry from
  `os.environ` before callers loaded `environment_file`. It now resolves the
  file for derived registry configuration. A configured project file is
  authoritative rather than silently falling back to a shell variable.

## Decision Log

- Decision: Do not fork Verifiers for the initial OpenAI/OpenRouter path.
  Rationale: upstream already owns the generic client, chat dialect, trace
  graph, tool continuation, and opaque reasoning-provider state. A Posttrain
  agent loop or trace fork would duplicate the authority that native traces
  deliberately provide.
  Date/Author: 2026-07-31 / Codex and user.

- Decision: Keep `ModelVariant` trainable/materialized-only and add an
  evaluation-only remote policy subject instead of pretending a proprietary
  endpoint has local weights, a tokenizer fingerprint, or a renderer.
  Rationale: a model selector, a service URL, and token rendering are different
  things. GRPO requires exact local sampled token IDs and remains unavailable
  for remote policies. Evaluation needs a requested remote model identity, not
  weight ownership.
  Date/Author: 2026-07-31 / Codex and user.

- Decision: Make the service binding, not the model alone, own external wire
  protocol, secret-variable name, headers, request defaults, and routing.
  Rationale: the same logical model can be local, direct-provider hosted, or
  OpenRouter-routed. Tool support and reasoning continuation are observed
  properties of the policy-service binding.
  Date/Author: 2026-07-31 / Codex and user.

- Decision: Retain native Verifiers traces as evaluation authority and record
  remote compatibility as evidence, not a parallel score database.
  Rationale: leaf count, tool paths, provider errors, and rewards belong to the
  trace population. Run-level attempted/complete/failed/truncated counters and
  qualification events are the irreducible summary.
  Date/Author: 2026-07-31 / Codex and user.

- Decision: Introduce one `remote-evaluation` catalog family whose value nests
  separate policy and service records, rather than three globally shared
  selection families.
  Rationale: this is evaluation-only functionality today. Nesting keeps the
  model/service distinction explicit without widening train, serve, or common
  catalog contracts beyond one additive opaque selection family. If another
  capability later needs either record independently, promote that record only
  with a new cross-capability contract and migration.
  Date/Author: 2026-07-31 / Codex.

- Decision: Treat `posttrain.env` as a reserved secret filename for immutable
  source snapshots.
  Rationale: it is a convenient project-local execution binding, but it is not
  source code or configuration that can safely enter an image. The framework
  must protect that boundary even if a project accidentally broadens its pack
  includes.
  Date/Author: 2026-07-31 / Codex.

- Decision: A configured `posttrain.env` is authoritative; one-off publication
  destination changes use the explicit `--registry` CLI option.
  Rationale: allowing a shell export to silently override project runtime
  configuration makes the command irreproducible and defeats project ownership.
  Date/Author: 2026-07-31 / Codex and user.

## Outcomes & Retrospective

The framework implementation now provides an external endpoint screening path
with no private Verifiers fork, and its immutable packer rejects the project
secret file used by Ambient Agent. Its focused validation passed:

    114 focused tests passed
    Ruff: all checks passed
    Pyright: 0 errors
    lint-imports: 8 contracts kept, 0 broken

The remaining product work is intentional: publish the framework change, add
the Ambient project overlay, and run a credential-gated OpenRouter probe. If a
real provider loses a continuation-critical field, this plan will name the
exact field and propose a small upstream Verifiers dialect/client change,
pinned temporarily by immutable commit until released; it will not create a
long-lived `posttrain-verifiers` repository.

## Context and Orientation

`packages/common` owns reusable immutable selections. A `ModelVariant` is an
exact local weight state and carries a tokenizer renderer contract. It is the
right seat for local serving and online training, where Posttrain must tokenize
and optimize exact sampled tokens. It is not the right representation for a
closed model called over HTTPS.

`packages/eval` owns evaluation requests and the Verifiers adapter.
`packages/eval/src/posttrain/eval/requests.py` currently defines
`EvaluateRequest` as a local `ModelVariant`, local `InferenceBinding`, and a
thin `EvaluationEndpoint`. `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py`
creates an upstream `EvalConfig`, invokes `run_eval`, synchronizes
`traces.jsonl`, and emits the native trace population into the observer.

`packages/jobs/src/posttrain/jobs/definitions.py` owns reusable job
definitions. It already distinguishes a managed endpoint, which a job launches,
from an external-endpoint definition. The external definition still synthesizes
a local vLLM endpoint, which is the behavior this plan corrects.

Verifiers owns the environment interaction. Its client receives native provider
requests, returns native responses, builds the trace graph, executes tools, and
preserves provider-specific continuation state. Posttrain must configure that
client and retain its evidence; it must not reimplement the loop.

The new terms are:

- An **evaluation subject** is the policy being judged. It is either a local
  `ModelVariant` or a `RemotePolicy`, which is an opaque provider model selector
  with an immutable requested revision/variant policy and declared context
  limit. A remote policy has no local artifact or renderer.
- An **external inference service** is where a remote policy is called. It owns
  an OpenAI-compatible protocol version, base URL, secret environment-variable
  name, safe headers, safe request defaults, and optional routing policy.
- A **remote evaluation binding** joins exactly one remote policy and one
  external service for `screen`/`eval`. It is the unit that can be qualified.
- A **compatibility probe** is a small native Verifiers environment cell that
  proves direct response, tool call, tool-result continuation, and configured
  reasoning behavior before a remote binding is used by the product extract
  screen.

## Plan of Work

First amend the frozen primitive/API baseline narrowly. The amendment must say
that evaluation accepts an `EvaluationSubject`; local subjects still use the
existing model and inference selections, while remote subjects are evaluation
only. The amendment must expressly preserve the existing rule that train and
token-level rollout adapters require `ModelVariant` plus a renderer. Add the
remote binding to the evaluation package rather than `posttrain.common` because
it is not a cross-capability train/serve primitive yet.

In `packages/eval/src/posttrain/eval/requests.py`, add immutable dataclasses
named `RemotePolicy`, `ExternalInferenceService`, and
`RemoteEvaluationBinding`. Validate stable ids/revisions, absolute secret-free
URLs, uppercase secret-variable names, JSON-only header/default maps, OpenAI
Chat protocol version `openai-chat@1`, and positive context length. Reject
authorization headers, URL user-info/query/fragment, and request defaults that
try to replace the selected model, messages, tools, or sampling. Headers and
defaults are safe configuration, never secret values.

Make `EvaluateRequest` use a discriminated evaluation subject/binding pair.
For a local subject, retain the present `ModelVariant` plus `InferenceBinding`
checks unchanged. For a remote subject, require a matching
`RemoteEvaluationBinding`, require only `screen` or `eval` purpose, and derive
the context limit from the remote policy. This is additive; no train or serve
request imports remote evaluation types.

In `packages/eval/src/posttrain/eval/backends/verifiers/adapter.py`, map a
remote binding to Verifiers' existing `client.type=eval` config with its
base URL, key-variable name, headers, and request defaults. Do not parse or
re-render assistant turns in Posttrain. Preserve the raw native trace bundle.
Record requested policy id/revision, service id/revision, protocol, and safely
redacted endpoint origin in events, trace attributes, and the evaluation
artifact metadata. Extract the observed `trace.agent.model` and any native
provider field already in the trace into trace attributes when present; do not
infer a provider that was not reported.

Add a versioned compatibility probe program under `packages/eval/src/.../programs`
or a test-only native fixture if the public environment package does not belong
in this repository. It must execute a tool call and tool result continuation,
not merely `/models` or a one-turn completion. The ordinary domain evaluation
remains the source of product reward; the probe only decides whether the
binding can participate.

Extend the catalog decoder in `packages/eval/src/posttrain/eval/catalog_schema.py`
with evaluation-package-owned remote policy/service/binding records. Do not add
a fake weight artifact to the framework `model` family. Update jobs so the
external evaluation definition resolves the endpoint from its remote binding;
managed definitions remain local `ServeLaunchRequest` paths. The old external
job definition gets a compatibility error when it is supplied only a local
inference binding with no externally supplied service, preventing accidental
calls to `127.0.0.1`.

Finally, add an Ambient Agent overlay that names OpenRouter as a service and
uses a pinned model slug selected by the project. The overlay contains only
`OPENROUTER_API_KEY` as a variable name; the key enters the dstack job through
the existing secret mechanism. Start with a small held-out extract screen and a
separate compatibility probe. Do not use a moving alias, frozen evaluation
tasks, or any training/GRPO work package.

## Concrete Steps

From `/home/hammad/projects/rl`:

1. Amend `docs/post-training/02-primitives.md` and
   `docs/post-training/05-apis.md` before changing code. Explain the new
   evaluation-only remote subject and keep the train invariant explicit.

2. Implement the typed evaluation values, adapter mapping, catalog decode, job
   definition wiring, and focused unit tests. Run:

       uv sync --all-packages --locked --python 3.13
       uv run pytest packages/eval/tests apps/lab/tests/test_work_packages.py packages/jobs/tests
       uv run ruff check packages/eval packages/jobs apps/lab
       uv run pyright packages/eval packages/jobs apps/lab
       uv run lint-imports

   The expected focused test output includes a fake remote binding whose
   configuration contains an OpenAI-compatible URL and an `OPENROUTER_API_KEY`
   variable name, while the serialized config and observer events contain no
   credential value.

3. Build an in-process or local HTTP fixture that returns a Chat Completions
   tool call followed by a successful tool-result continuation. Run the
   Verifiers adapter against it and assert one completed native trace, a native
   `traces.jsonl` artifact, and matching requested/observed remote-policy
   attributes.

4. From `/home/hammad/projects/ambient-agent`, after the framework change is
   published or intentionally consumed from an immutable local source revision,
   add the remote service, policy, binding, and `screen` work package. Run:

       UV_SYSTEM_CERTS=1 uv run posttrain catalog validate
       UV_SYSTEM_CERTS=1 uv run pytest -q packages/episode-qa-v1/tests/test_episode_qa_v1.py

5. Only when an operator intentionally supplies a non-empty
   `OPENROUTER_API_KEY`, launch the one-task compatibility probe and then the
   bounded held-out extract screen. Record the dstack run id, Trackio run id,
   native trace artifact, requested/observed model identity, tool success,
   continuation success, leaf count, reward distribution, latency, and provider
   failures in this plan. Never print the key or request authorization header.

## Validation and Acceptance

The framework change is accepted when all focused tests pass and prove these
observable behaviors:

1. A remote policy cannot be supplied to train, serve, or token-level rollout
   operations.
2. An external service URL with credentials, query parameters, or an
   authorization header is rejected before execution.
3. A remote binding with `openai-chat@1` produces a native Verifiers config
   containing its base URL, key-variable name, allowed headers, and allowed
   request defaults, but no secret value.
4. The local fake tool loop completes through Verifiers and preserves native
   trace authority; Posttrain creates no parallel interaction graph.
5. A managed local evaluation still constructs and uses its vLLM endpoint as
   before.
6. A real OpenRouter probe, when credentials are deliberately present, completes
   at least one tool-result continuation or returns a retained, classified
   native failure trace. A provider failure is valid evidence, not a reason to
   silently retry with another model.

## Idempotence and Recovery

All selections are additive and versioned. A failed remote probe writes a
normal evaluation artifact and native traces, so retries use a new run id and
do not overwrite evidence. Catalog validation and fake tests are safe to rerun.
Do not add secrets to YAML, the catalog, the plan, artifacts, log output, or
Git. If an upstream Verifiers gap is found, create a small upstream pull request
with a regression test, pin its immutable commit only while necessary, and
remove that pin after a release; do not create a permanent mirror fork.

## Artifacts and Notes

Current upstream evidence relevant to the design:

    Ambient Agent Verifiers pin: 284a868d6a9022109b749710672a0460e8a996d4
    verifiers.v1.clients.EvalClient: base_url, api_key, headers
    verifiers.v1.dialects.chat: reasoning/reasoning_content/reasoning_details
    AssistantMessage.provider_state: opaque signed/encrypted reasoning state
    ChatDialect.extend: appends verbatim upstream assistant message

The upstream client behavior is why OpenRouter uses a normal OpenAI-chat
service binding. A future native Anthropic or Responses binding is a separate
protocol adapter, not a condition hidden inside an OpenRouter selection.

## Interfaces and Dependencies

At the end of the implementation, `posttrain.eval` must expose:

    @dataclass(frozen=True, slots=True)
    class RemotePolicy:
        id: str
        revision: str
        model: str
        context_window: int
        capabilities: Mapping[str, JsonValue]

    @dataclass(frozen=True, slots=True)
    class ExternalInferenceService:
        id: str
        revision: str
        protocol: Literal["openai-chat@1"]
        base_url: str
        api_key_var: str
        headers: Mapping[str, str]
        request_defaults: Mapping[str, JsonValue]

    @dataclass(frozen=True, slots=True)
    class RemoteEvaluationBinding:
        id: str
        revision: str
        policy: RemotePolicy
        service: ExternalInferenceService
        purpose: tuple[Literal["screen", "eval"], ...]

The final `EvaluateRequest` must accept either the existing local model plus
`InferenceBinding` path or the new `RemoteEvaluationBinding` path, but it must
make mixed pairs a validation error. The Verifiers adapter must use existing
upstream `EvalClient`; no Posttrain client implementation, renderer, or trace
graph is introduced.

## Plan revision note

2026-07-31: Created after inspecting current framework and pinned upstream
Verifiers behavior. The plan intentionally separates local token rendering,
remote model selection, service transport, and native environment ownership.

2026-07-31: Implemented the framework-owned contract, catalog family, standard
job definition, Verifiers config mapping, and fake/local tests. Ambient work is
deferred until this uncommitted framework revision is intentionally published
or otherwise consumed by immutable source revision.
