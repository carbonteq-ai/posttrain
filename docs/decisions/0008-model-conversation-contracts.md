# ADR 0008 — Model conversation contracts and backend parsers

## Status

Accepted.

## Context

Post-training, serving, and evaluation must render the same messages in the
format expected by a pinned model. A model profile that records only a renderer
name or a reasoning boolean is insufficient: tool definitions, assistant tool
calls, tool responses, thinking controls, and prior reasoning all affect the
token sequence.

These semantics are partly model-native and partly backend-specific. The model
template defines the wire grammar, while vLLM selects parsers that project
native generations into OpenAI-compatible reasoning and tool-call fields.
Putting both concerns in a serving profile would make training and direct model
evaluation depend on vLLM. Putting vLLM parser names in the shared model profile
would leak one backend into every consumer.

The pinned Qwen3.5 template supports its XML tool-call grammar and explicit
`enable_thinking` flag. The pinned LFM2.5 Thinking template advertises tools but does not
reconstruct an OpenAI assistant `tool_calls` history: `content=None` renders as
`null`, losing the call on the next turn. That breaks multi-turn tool-use
environments even though vLLM can parse newly generated LFM calls.

LFM2.5-350M and LFM2.5-1.2B-Instruct use a newer, byte-identical upstream
template with explicit assistant-generation boundaries. They are instruction
models, so binding them to the Thinking renderer or reasoning parser would
change both their token contract and response semantics.

## Decision

- A shared `ModelProfile` owns a typed `ConversationProfile` containing
  supported roles, reasoning modes and template keyword arguments, default
  reasoning mode, native tool-call protocol, prior-reasoning policy, and a
  `ChatTemplate` source.
- The canonical cross-package message schema is OpenAI-compatible messages and
  JSON function definitions. Model templates project that schema into native
  tokens.
- Use the pinned tokenizer template when it round-trips the canonical schema.
  A model-specific package template is allowed when the upstream template is
  incomplete, provided it is versioned with the model profile and covered by
  tokenizer-level golden tests.
- Qwen3.5 uses its pinned tokenizer template, XML function/parameter elements,
  and explicit `native`, `off`, and `thinking` modes.
- LFM2.5 Thinking uses a package-owned template derived from its pinned tokenizer
  template. The override preserves OpenAI assistant tool-call history as the
  model's native Pythonic call list between `<|tool_call_start|>` and
  `<|tool_call_end|>`.
- LFM2.5 Instruct uses a separate package-owned, byte-exact copy of the pinned
  upstream Instruct template. Its renderer has reasoning disabled, retains the
  native Pythonic tool grammar, and exposes the template's generation markers
  to assistant-only SFT masking.
- A `VllmServeProfile` owns vLLM frontend parser selection. Qwen uses
  `qwen3_xml` and `qwen3`; LFM uses the `lfm2` tool parser. Only a reasoning
  model receives a reasoning parser; the LFM Instruct bindings omit it.
- Training renderers consume the shared conversation contract and add
  technique-specific loss masks. Evaluation environments provide tools and
  scoring semantics but do not redefine model formatting.

## Consequences

- Serving, SFT, DPO, GRPO, and evaluation can share one model-native
  conversation definition without sharing framework code.
- A serving backend remains free to choose its own parser or adapter while the
  underlying tool grammar stays stable.
- Multi-turn tool evaluations cannot silently lose LFM assistant calls.
- Package chat-template overrides become maintained compatibility code and need
  golden tests whenever the model revision or tokenizer stack changes.
- A tool parser being configured does not prove tool quality; tool-use
  environments must still evaluate call selection, arguments, state changes,
  and final answers.

## Alternatives Considered

### Keep chat and tool formatting only in serving profiles

Rejected because training renderers and in-process evaluation need identical
model-native tokens without depending on vLLM.

### Trust every tokenizer template without validation

Rejected because the pinned LFM template demonstrably loses OpenAI tool-call
history on multi-turn conversations.

### Normalize all models to one custom JSON tool grammar

Rejected because it would diverge from the formats learned by the foundation
models and invalidate their native parser support.

### Put vLLM parser names in the shared model profile

Rejected because parser identifiers are backend implementation details and can
change independently of the model's native protocol.

## Implementation Notes

- Shared types and profiles live in `posttrain.common.models` and
  `posttrain.common.profiles`.
- The LFM override is
  `posttrain/common/templates/lfm25_tool_chat.jinja` and is included in the
  `posttrain-common` wheel.
- The LFM Instruct template is
  `posttrain/common/templates/lfm25_instruct_chat.jinja`; its pinned SHA-256 is
  `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6`.
- vLLM parser choices live in `posttrain.serve.profiles` and are emitted only
  when an online vLLM endpoint is launched.
- Tests load the exact pinned tokenizers when locally available and verify
  Qwen tool/thinking markers plus LFM assistant-call and tool-result history.
- Upstream references:
  [Qwen pinned template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/chat_template.jinja),
  [LFM pinned template](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/blob/95053d21d8e0b7ca99421a2127ae39c64f685ff3/chat_template.jinja),
  [vLLM LFM parser](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/tool_parsers/lfm2_tool_parser.py), and
  [vLLM parser registry](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/tool_parsers/__init__.py).

## Revision History

- 2026-07-20: Established shared conversation contracts, backend-owned parser
  choices, and the tested LFM tool-history template override.
- 2026-08-21: Added a distinct, hash-pinned LFM2.5 Instruct contract and kept
  reasoning parsing exclusive to the Thinking variant.
