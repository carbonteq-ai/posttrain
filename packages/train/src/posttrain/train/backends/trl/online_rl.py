"""TRL policy-generation adapter for environment-driven online-RL bridges."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, cast

from posttrain.common import ModelVariant

from ...bindings import TrainingBinding
from ...online_rl import PolicyTurnRequest, PolicyTurnResult
from ...profiles import GRPOSettings, OnPolicyDistillationSettings, SAMPOSettings
from ...rendering import create_renderer


class TrlPolicyGenerator:
    """Expose the trainer's already-loaded policy as a backend-neutral turn generator."""

    def __init__(
        self,
        trainer: Any,
        tokenizer: Any,
        model: ModelVariant,
        settings: GRPOSettings | SAMPOSettings | OnPolicyDistillationSettings,
        training: TrainingBinding,
    ) -> None:
        self._trainer = trainer
        self._renderer = create_renderer(tokenizer, model, training.renderer)
        self._model_family = model.family
        self._max_completion_length = settings.max_completion_length
        self._lock = asyncio.Lock()

    async def generate(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        expected_sampling = (float(self._trainer.temperature), float(self._trainer.top_p))
        actual_sampling = (request.sampling.temperature, request.sampling.top_p)
        if actual_sampling != expected_sampling:
            raise ValueError(
                f"environment sampling {actual_sampling!r} does not match the loaded TRL generator "
                f"{expected_sampling!r}"
            )
        if request.sampling.max_tokens > self._max_completion_length:
            raise ValueError(
                f"environment max_tokens={request.sampling.max_tokens} exceeds the loaded TRL generator cap "
                f"{self._max_completion_length}"
            )
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
            with _turn_generation_config(self._trainer, request, self._model_family):
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
                "name": item.name,
                "arguments": item.arguments
                if isinstance(item.arguments, str)
                else json.dumps(item.arguments or {}, separators=(",", ":")),
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
            request.sampling.max_tokens,
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


_MISSING = object()
_GEMMA_JSON_WHITESPACE_PATTERN = r" ?"
_XGRAMMAR_UNSUPPORTED_JSON_SCHEMA_KEYS = frozenset(
    {
        "multipleOf",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "patternProperties",
        "propertyNames",
    }
)


def _xgrammar_json_schema(value: Any) -> Any:
    """Remove constraints that vLLM's XGrammar backend cannot compile.

    The environment retains and validates its canonical schema.  This copy is
    used only for constrained token generation, so removing semantic
    constraints such as array uniqueness does not bypass environment-owned
    validation.
    """

    if isinstance(value, list):
        return [_xgrammar_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _xgrammar_json_schema(item)
        for key, item in value.items()
        if key not in _XGRAMMAR_UNSUPPORTED_JSON_SCHEMA_KEYS
    }


@contextmanager
def _turn_generation_config(trainer: Any, request: PolicyTurnRequest, model_family: str) -> Iterator[None]:
    """Apply one staged turn's token/schema controls and restore shared trainer state."""

    generation_config = getattr(trainer, "generation_config", None)
    prior_max_tokens = (
        getattr(generation_config, "max_new_tokens", _MISSING) if generation_config is not None else _MISSING
    )
    vllm_generation = getattr(trainer, "vllm_generation", None)
    prior_generation_kwargs = (
        getattr(vllm_generation, "generation_kwargs", _MISSING) if vllm_generation is not None else _MISSING
    )
    if request.response_format is not None and vllm_generation is None:
        raise RuntimeError("structured policy turns require the configured vLLM rollout engine")
    try:
        if generation_config is not None:
            generation_config.max_new_tokens = request.sampling.max_tokens
        if vllm_generation is not None:
            if prior_generation_kwargs is _MISSING or not isinstance(prior_generation_kwargs, dict):
                raise TypeError("TRL vLLM generation_kwargs must be a dictionary")
            overrides = dict(prior_generation_kwargs)
            overrides["max_tokens"] = request.sampling.max_tokens
            if request.response_format is not None:
                structured_outputs: dict[str, Any] = {
                    "json": _xgrammar_json_schema(dict(request.response_format.schema))
                }
                if model_family == "gemma4":
                    structured_outputs["whitespace_pattern"] = _GEMMA_JSON_WHITESPACE_PATTERN
                overrides["structured_outputs"] = structured_outputs
            else:
                overrides.pop("structured_outputs", None)
            vllm_generation.generation_kwargs = overrides
        yield
    finally:
        if generation_config is not None:
            if prior_max_tokens is _MISSING:
                try:
                    del generation_config.max_new_tokens
                except AttributeError:
                    pass
            else:
                generation_config.max_new_tokens = prior_max_tokens
        if vllm_generation is not None and prior_generation_kwargs is not _MISSING:
            vllm_generation.generation_kwargs = prior_generation_kwargs


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
    wire_message = dict(message)
    tool_calls = wire_message.get("tool_calls")
    if isinstance(tool_calls, list):
        wire_message["tool_calls"] = [
            {
                "id": str(call["id"]),
                "type": "function",
                "function": {
                    "name": str(call["name"]),
                    "arguments": str(call["arguments"]),
                },
            }
            for call in tool_calls
        ]
    return {
        "id": "posttrain-policy-turn",
        "object": "chat.completion",
        "created": 0,
        "choices": [{"index": 0, "message": wire_message, "finish_reason": finish_reason}],
    }


__all__ = ["TrlPolicyGenerator"]
