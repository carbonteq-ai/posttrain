# Tool-using environment execution

**Status:** active implementation architecture, 2026-08-04. The canonical
product contract is the tool-using environment amendment in
`docs/post-training/README.md` and the fields in `05-apis.md`. Revise this
document in place when a new transport or inference backend is qualified.

## Problem

AutomationBench exposed two independent defects that ordinary single-turn
evaluations could not reveal. First, the Verifiers harness and its MCP tool
servers ran from different Python dependency graphs. After that was corrected,
the model endpoint rejected `tool_choice="auto"` because its vLLM binding did
not enable a tool-call parser. The first parser-qualified run then exposed a
third boundary: the model emitted valid structured calls, but the selected
meta-tool surface invited it to call discovered concrete names that the server
did not expose. Treating any failure as an AutomationBench-specific exception
would reproduce the same defects for future tool-using environments.

The architecture separates portable interaction meaning from runtime
mechanism:

```text
EnvironmentBinding                  InferenceBinding
requires: tool-calling   matches    provides: tool-calling
        |                                  |
        |                                  +-- model renderer protocol: qwen3_xml
        |                                  +-- vLLM parser: qwen3_xml
        |
        +-- Verifiers activation
            +-- harness client
            +-- MCP stdio tool servers
            +-- environment package
            all use /opt/posttrain/venv/bin/python
```

`tool-calling` is the compatibility contract. MCP is only one possible
environment-side transport. A future environment may execute tools in-process
or through another protocol without changing the inference contract.

## Contracts and ownership

`packages/environment/src/posttrain/environment/requests.py` owns
`EnvironmentBinding.required_inference_capabilities`. An environment declares
`tool-calling` when completing its tasks requires the model to emit structured
tool calls. The declaration is portable and belongs with task meaning.

`packages/common/src/posttrain/common/selections.py` owns
`InferenceBinding.capabilities`. A capability describes the complete
model-renderer-backend binding, not a foundation-model claim. The binding's
renderer identity must equal the selected model's renderer contract, so it
cannot choose a second chat template. A tool-capable binding must use a model
renderer with a declared `ToolCallProtocol`.

`packages/work/src/posttrain/work/runner.py` checks environment requirements
against the selected evaluation or rollout inference binding during detached
planning. `packages/eval/src/posttrain/eval/requests.py` repeats the check when
constructing local or remote evaluation requests, so direct API callers receive
the same failure semantics.

`packages/serve/src/posttrain/serve/backends/vllm/bindings.py` owns the vLLM
translation. A binding advertising `tool-calling` resolves its vLLM parser from
the selected model's `ToolCallProtocol`; the adapter emits
`--enable-auto-tool-choice` and `--tool-call-parser`. Qwen3.5's renderer
declares `qwen3_xml`, which maps to vLLM's `qwen3_xml` parser. A conflicting
explicit override fails. This backend mapping does not belong on the
environment or need repetition in every inference catalog entry.

`packages/eval/src/posttrain/eval/backends/verifiers/runtime.py` owns packed
Verifiers runtime activation. It materializes immutable harness script bytes
and runs the harness, MCP client, tool servers, environment package, and
framework from `/opt/posttrain/venv/bin/python`. It does not create a PEP 723
environment, invoke `uv sync`, or install packages at execution time.

The environment binding also owns the semantic tool surface. For the Qwen3.5
4B AutomationBench qualification it selects `limited_zapier`, which exposes
only the concrete tools declared by each task. The model renderer owns how
schemas and calls are serialized; it does not decide whether an environment
presents concrete tools or discovery and execution meta-tools.

## Renderer resolution

The model catalog is the renderer authority. `ModelVariant.renderer` owns the
chat-template source, reasoning modes, and tool-call wire protocol. An
`InferenceBinding` may only fingerprint that same renderer identity; it cannot
select a different template.

Managed evaluation uses Verifiers' OpenAI evaluation client. Verifiers sends
typed messages and tool schemas, while the managed vLLM endpoint applies the
template resolved from the selected model. Posttrain materializes a
package-owned template for vLLM when the model contract supplies one; otherwise
vLLM loads the immutable model revision's tokenizer template. Evaluation
reasoning kwargs such as `enable_thinking` are derived from the same model
contract and forwarded with the request.

Token-level training has a different need: exact rendering, response parsing,
and turn bridging happen client-side. That path reuses the pinned `renderers`
package through Posttrain's training adapter, but its concrete renderer config
is still derived from the selected model contract. Switching normal eval to the
training renderer client would change the endpoint protocol and token-evidence
semantics; it is not required to make the model catalog authoritative.

## Admission sequence

1. Catalog decoding validates stable capability names.
2. Work-package planning resolves the environment and the relevant inference
   binding and rejects missing capabilities before packing or submission.
3. The backend adapter resolves the model protocol to its concrete parser. For
   vLLM, an unknown protocol or conflicting override is an error.
4. The job image supplies one locked interpreter and dependency closure.
5. Verifiers initializes the environment and MCP session, then sends tools to
   the OpenAI-compatible endpoint.
6. vLLM parses model output according to the selected renderer protocol and
   returns structured tool calls to Verifiers.
7. Verifiers dispatches only calls admitted by the task-scoped environment
   surface and records tool results in the native trace.

The failure messages identify the owning layer: capability mismatch during
planning, invalid parser during backend translation, dependency/runtime failure
during environment initialization, or provider/model failure during rollout.

## Qualification

AutomationBench is the first cross-layer qualification because it exercises
the full path: catalog requirement, detached compatibility, packed dependency
closure, MCP initialization, OpenAI tool schema, model tool output, tool
execution, multi-turn continuation, native reward components, and explicit
task success. A qualifying run must show at least one structured tool call and
must persist AutomationBench's native `partial_credit` reward and
`task_completed_correctly` metric. Starting the MCP server alone is necessary
but insufficient.

The qualifying evidence is run
`observatory-qwen4b-automationbench-success-v2e-20260804`, packaged as
`45c547e6297374b2828c0580a282d41a6192331e338e8c0ff55504e5effc76e0`
and executed from private-OCI image digest
`sha256:5f9527356e4be72864a2d0f8ae3ac60bf6978034394ad697776fac5d8dfc9075`.
Both selected tasks passed with successful concrete tool results, no unknown
tools, no truncation, and native reward, success, assertion, token, latency,
response, and thinking evidence persisted through Observatory.

For a new tool-using environment:

- publish and immutably pin the environment package;
- declare `required_inference_capabilities: [tool-calling]`;
- select an inference binding whose renderer and backend parser are qualified
  for the same tool-call protocol;
- verify planning rejects a plain inference binding;
- run one real end-to-end qualification and inspect native traces for tool
  calls, tool results, continuation, and environment-owned success evidence.

## Revision history

- 2026-08-04: established the environment requirement, inference capability,
  detached admission, vLLM parser, and single locked Verifiers/MCP runtime
  boundaries from the AutomationBench failures.
- 2026-08-04: assigned task-specific tool-surface selection to the environment
  binding after the first parser-qualified run showed that renderer correctness
  does not imply semantic compatibility with discovery meta-tools.
