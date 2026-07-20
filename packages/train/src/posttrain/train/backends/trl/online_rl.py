"""TRL policy-generation adapter for environment-driven online-RL bridges."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal, cast

from posttrain.common import ModelProfile

from ...online_rl import PolicyTurnRequest, PolicyTurnResult
from ...profiles import GRPOProfile
from ...rendering import create_renderer


class TrlPolicyGenerator:
    """Expose the trainer's already-loaded policy as a backend-neutral turn generator."""

    def __init__(self, trainer: Any, tokenizer: Any, model: ModelProfile, profile: GRPOProfile) -> None:
        self._trainer = trainer
        self._renderer = create_renderer(tokenizer, model, profile.renderer)
        self._max_completion_length = profile.max_completion_length
        self._lock = asyncio.Lock()

    async def generate(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        expected = (
            self._max_completion_length,
            float(self._trainer.temperature),
            float(self._trainer.top_p),
        )
        actual = (request.sampling.max_tokens, request.sampling.temperature, request.sampling.top_p)
        if actual != expected:
            raise ValueError(f"environment sampling {actual!r} does not match the loaded TRL generator {expected!r}")
        messages = [cast(dict[str, Any], dict(message)) for message in request.messages]
        tools = [cast(dict[str, Any], dict(tool)) for tool in request.tools]
        rendered = None
        if request.previous_prompt_ids:
            rendered = self._renderer.bridge_to_next_turn(
                list(request.previous_prompt_ids),
                list(request.previous_completion_ids),
                messages[request.tail_start :],
                tools=tools or None,
            )
        if rendered is None:
            rendered = self._renderer.render(messages, tools=tools or None, add_generation_prompt=True)
            spans = tuple(rendered.message_token_spans())
        else:
            spans = _bridged_message_spans(rendered, request.tail_start, len(request.previous_token_ids))

        async with self._lock:
            completion_ids, logprobs = self._trainer._generate_single_turn(  # noqa: SLF001 - pinned adapter
                [rendered.token_ids],
                None,
                {},
            )
        token_ids = tuple(int(value) for value in completion_ids[0])
        if not token_ids:
            raise RuntimeError("the policy generator returned an empty completion")
        sampled_logprobs = () if logprobs is None else tuple(float(value) for value in logprobs[0])
        parsed = self._renderer.parse_response(list(token_ids), tools=tools or None)
        tool_calls = [
            {
                "id": item.id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": item.arguments
                    if isinstance(item.arguments, str)
                    else json.dumps(item.arguments or {}, separators=(",", ":")),
                },
            }
            for index, item in enumerate(parsed.tool_calls)
            if item.name is not None and item.status.value == "ok"
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": parsed.content or None,
        }
        if parsed.reasoning_content is not None:
            message["reasoning_content"] = parsed.reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = _finish_reason(
            token_ids,
            frozenset(self._renderer.get_stop_token_ids()),
            bool(tool_calls),
            self._max_completion_length,
        )
        raw_response = _openai_response(message, finish_reason)
        return PolicyTurnResult(
            message=message,
            prompt_ids=tuple(int(value) for value in rendered.token_ids),
            completion_ids=token_ids,
            completion_logprobs=sampled_logprobs,
            finish_reason=finish_reason,
            prompt_message_spans=spans,
            prompt_is_content=tuple(bool(value) for value in rendered.is_content),
            raw_response=raw_response,
        )


def _bridged_message_spans(rendered: Any, tail_start: int, prefix_tokens: int) -> tuple[tuple[int, int] | None, ...]:
    from renderers import RenderedTokens  # pyright: ignore[reportMissingImports]

    tail = RenderedTokens(
        message_indices=rendered.message_indices[prefix_tokens:],
        message_roles=rendered.message_roles,
    )
    return tuple(
        [None] * tail_start
        + [
            None if span is None else (span[0] + prefix_tokens, span[1] + prefix_tokens)
            for span in tail.message_token_spans()
        ]
    )


def _finish_reason(
    completion_ids: tuple[int, ...],
    stop_token_ids: frozenset[int],
    has_tool_calls: bool,
    max_completion_length: int,
) -> Literal["stop", "length", "tool_calls"]:
    if has_tool_calls:
        return "tool_calls"
    if completion_ids[-1] in stop_token_ids:
        return "stop"
    if len(completion_ids) >= max_completion_length:
        return "length"
    return "stop"


def _openai_response(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "id": "posttrain-policy-turn",
        "object": "chat.completion",
        "created": 0,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


__all__ = ["TrlPolicyGenerator"]
