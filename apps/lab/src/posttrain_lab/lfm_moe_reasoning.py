"""Project-local optional-thinking parser for the pinned LFM2.5-8B experiment.

Unlike Qwen3's default, the native 8B template does not prefill a think block.
Keep plain answers as content and enter reasoning only on an explicit marker.
The 24B experiment continues to use the stock qwen3 parser.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import replace
from typing import Any


def register_optional_reasoning() -> None:
    """Register an opt-in parser; never replace the runtime's qwen3 parser."""
    from vllm.parser.engine.adapters import make_adapters
    from vllm.parser.engine.parser_engine_config import ParserState, Transition
    from vllm.parser.qwen3 import Qwen3Parser, qwen3_config
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

    class LfmOptionalThinkingParser(Qwen3Parser):
        def __init__(self, tokenizer: Any, tools: Any = None, **kwargs: Any) -> None:
            config = qwen3_config(thinking=False, name="lfm2_optional_thinking")
            transitions = dict(config.transitions)
            transitions[ParserState.CONTENT, "THINK_START"] = Transition(ParserState.REASONING)
            # Keep DelegatingParser in its extraction phase until actual output
            # disambiguates optional thinking. Starting in CONTENT alone would
            # otherwise bypass extraction for the whole HTTP stream.
            kwargs["parser_engine_config"] = replace(config, transitions=transitions, wait_for_reasoning=True)
            super().__init__(tokenizer, tools, **kwargs)
            # This controls extraction, not whether the model is asked to think.
            self.thinking_enabled = True

    reasoning_adapter, _ = make_adapters(LfmOptionalThinkingParser)

    class OptionalReasoningAdapter(reasoning_adapter):
        def extract_reasoning_streaming(self, *args: Any, **kwargs: Any) -> Any:
            delta = super().extract_reasoning_streaming(*args, **kwargs)
            if delta is not None and delta.content:
                self._plain_content_seen = True
            return delta

        def has_engine_confirmed_reasoning_end(self) -> bool:
            return super().has_engine_confirmed_reasoning_end() or getattr(self, "_plain_content_seen", False)

    ReasoningParserManager.register_module(name="lfm2_optional_thinking", module=OptionalReasoningAdapter, force=True)


def qualify_optional_reasoning(tokenizer: Any) -> None:
    """Exercise real pinned parser extraction before any model is allocated."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.abstract_parser import DelegatingParser
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    register_optional_reasoning()
    parser_type = ReasoningParserManager.get_reasoning_parser("lfm2_optional_thinking")
    request = ChatCompletionRequest(model="probe", messages=[{"role": "user", "content": "probe"}])
    for text, expected_reasoning in [("READY-2047", ""), ("<think>check</think>READY-2047", "check")]:
        reasoning, content = parser_type(tokenizer).extract_reasoning(text, request)
        if (reasoning or "", content) != (expected_reasoning, "READY-2047"):
            raise RuntimeError("optional-thinking nonstream parser regression")
    for pieces, expected_reasoning in [(["READY-2047"], ""), (["<think>", "check", "</think>", "READY-2047"], "check")]:
        parser = parser_type(tokenizer)
        previous = ""
        previous_ids: list[int] = []
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for piece in pieces:
            delta_ids = tokenizer.encode(piece, add_special_tokens=False)
            current_ids = previous_ids + delta_ids
            delta = parser.extract_reasoning_streaming(
                previous, previous + piece, piece, previous_ids, current_ids, delta_ids
            )
            if delta is not None:
                reasoning_parts.append(delta.reasoning or "")
                content_parts.append(delta.content or "")
            previous += piece
            previous_ids = current_ids
        if ("".join(reasoning_parts), "".join(content_parts)) != (expected_reasoning, "READY-2047"):
            raise RuntimeError("optional-thinking streaming parser regression")

    class NativeStreamParser(DelegatingParser):
        reasoning_parser_cls = parser_type
        tool_parser_cls = ToolParserManager.get_tool_parser("lfm2")

    prompt_ids = tokenizer.apply_chat_template(request.messages, tokenize=True, add_generation_prompt=True)
    for text, expected_reasoning in [("READY-2047", ""), ("<think>check</think>READY-2047", "check")]:
        parser = NativeStreamParser(tokenizer)
        reasoning_parts = []
        content_parts = []
        ids = tokenizer.encode(text, add_special_tokens=False)
        for index, token in enumerate(ids):
            delta = parser.parse_delta(
                tokenizer.decode([token]),
                [token],
                request,
                prompt_token_ids=prompt_ids,
                finished=index == len(ids) - 1,
            )
            if delta is not None:
                reasoning_parts.append(delta.reasoning or "")
                content_parts.append(delta.content or "")
        if ("".join(reasoning_parts), "".join(content_parts)) != (expected_reasoning, "READY-2047"):
            raise RuntimeError("optional-thinking native delegating stream regression")
