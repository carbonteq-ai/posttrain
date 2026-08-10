"""TRL policy-generation adapter for environment-driven online-RL bridges."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal, cast

from posttrain.common import ModelVariant

from ...bindings import TrainingBinding
from ...online_rl import PolicyTurnRequest, PolicyTurnResult
from ...profiles import GRPOSettings, OnPolicyDistillationSettings, SAMPOSettings
from ...rendering import create_renderer

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
_LOGGER = logging.getLogger(__name__)


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
        self._tokenizer = tokenizer
        self._renderer = create_renderer(tokenizer, model, training.renderer)
        self._model_family = model.family
        self._max_prompt_length = settings.max_prompt_length
        self._max_completion_length = settings.max_completion_length
        self._lock = asyncio.Lock()
        self._pending: list[
            tuple[
                Sequence[int],
                int,
                dict[str, object] | None,
                asyncio.Future[tuple[Sequence[int], Sequence[float] | None]],
            ]
        ] = []
        self._flush_task: asyncio.Task[None] | None = None

    async def generate(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        expected = (float(self._trainer.temperature), float(self._trainer.top_p))
        actual = (request.sampling.temperature, request.sampling.top_p)
        if actual != expected:
            raise ValueError(f"environment sampling {actual!r} does not match the loaded TRL generator {expected!r}")
        if request.sampling.max_tokens > self._max_completion_length:
            raise ValueError("environment max_tokens exceeds the loaded trainer max_completion_length")
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

        prompt_tokens = len(rendered.token_ids)
        prompt_limit = request.max_prompt_tokens or self._max_prompt_length
        sequence_limit = request.max_sequence_tokens or (
            self._max_prompt_length + self._max_completion_length
        )
        if prompt_tokens > prompt_limit:
            raise ValueError(
                f"rendered policy prompt has {prompt_tokens} tokens; limit is {prompt_limit}"
            )
        effective_max_tokens = min(
            request.sampling.max_tokens,
            self._max_completion_length,
            sequence_limit - prompt_tokens,
        )
        if effective_max_tokens <= 0:
            raise ValueError("rendered policy prompt leaves no completion capacity")
        completion_ids, logprobs = await self._generate_tokens(
            rendered.token_ids,
            effective_max_tokens,
            _structured_outputs(request.response_format, model_family=self._model_family),
        )
        token_ids = tuple(int(value) for value in completion_ids)
        if not token_ids:
            raise RuntimeError(
                "the policy generator returned no completion token ids; "
                "an immediate model stop is not a valid policy turn"
            )
        if len(token_ids) > effective_max_tokens:
            raise RuntimeError("the policy generator exceeded the effective completion-token limit")
        if prompt_tokens + len(token_ids) > sequence_limit:
            raise RuntimeError("the policy generator exceeded the effective sequence limit")
        sampled_logprobs = () if logprobs is None else tuple(float(value) for value in logprobs)
        parsed = self._renderer.parse_response(list(token_ids), tools=tools or None)
        if request.response_format is not None:
            content = parsed.content or ""
            try:
                json.loads(content)
            except (json.JSONDecodeError, TypeError, ValueError):
                decoded = self._tokenizer.decode(list(token_ids), skip_special_tokens=False)
                _LOGGER.warning(
                    "structured policy completion is not valid JSON; "
                    "token_count=%d; token_ids_prefix=%r; token_ids_suffix=%r; "
                    "decoded_prefix=%r; parsed_content_prefix=%r",
                    len(token_ids),
                    list(token_ids[:16]),
                    list(token_ids[-16:]),
                    str(decoded)[:160],
                    content[:160],
                )
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
            effective_max_tokens,
        )
        raw_response = _openai_response(
            message,
            finish_reason,
            prompt_tokens=prompt_tokens,
            effective_max_tokens=effective_max_tokens,
        )
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

    async def _generate_tokens(
        self,
        prompt_ids: Sequence[int],
        max_tokens: int,
        structured_outputs: dict[str, object] | None,
    ) -> tuple[Sequence[int], Sequence[float] | None]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[Sequence[int], Sequence[float] | None]] = loop.create_future()
        self._pending.append((prompt_ids, max_tokens, structured_outputs, future))
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_pending())
        return await future

    async def _flush_pending(self) -> None:
        # Let concurrently scheduled environment turns reach the queue before
        # issuing the synchronous trainer call.
        await asyncio.sleep(0)
        pending: list[
            tuple[
                Sequence[int],
                int,
                dict[str, object] | None,
                asyncio.Future[tuple[Sequence[int], Sequence[float] | None]],
            ]
        ] = []
        try:
            # Waiting for the trainer lock yields to the event loop. Drain until
            # no later turn remains queued so none can be stranded behind this
            # already-scheduled flush task.
            while self._pending:
                pending, self._pending = self._pending, []
                active = [item for item in pending if not item[3].cancelled()]
                if not active:
                    continue
                heterogeneous_generate = getattr(
                    self._trainer,
                    "_generate_policy_turns",
                    None,
                )
                if callable(heterogeneous_generate):
                    async with self._lock:
                        completion_ids, logprobs = cast(Any, heterogeneous_generate)(
                            [list(item[0]) for item in active],
                            [
                                {
                                    "max_tokens": item[1],
                                    "structured_outputs": item[2],
                                }
                                for item in active
                            ],
                        )
                    _resolve_generation_results(active, completion_ids, logprobs)
                    continue
                groups: dict[tuple[int, str], list[tuple[Any, ...]]] = {}
                for item in active:
                    schema_key = json.dumps(item[2], sort_keys=True, separators=(",", ":"))
                    groups.setdefault((item[1], schema_key), []).append(item)
                for (max_tokens, _schema_key), group in groups.items():
                    structured_outputs = group[0][2]
                    async with self._lock:
                        with _generation_overrides(
                            self._trainer,
                            max_tokens=max_tokens,
                            structured_outputs=structured_outputs,
                        ):
                            completion_ids, logprobs = self._trainer._generate_single_turn(  # noqa: SLF001 - pinned adapter
                                [list(item[0]) for item in group],
                                None,
                                {},
                            )
                    _resolve_generation_results(group, completion_ids, logprobs)
        except BaseException as exc:
            for _prompt_ids, _limit, _structured, future in [*pending, *self._pending]:
                if not future.done():
                    future.set_exception(exc)
            self._pending = []
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.create_task(self._flush_pending())


def _resolve_generation_results(
    pending: Sequence[tuple[Any, ...]],
    completion_ids: Sequence[Sequence[int]],
    logprobs: Sequence[Sequence[float]] | None,
) -> None:
    if len(completion_ids) != len(pending):
        raise RuntimeError(
            "the policy generator returned a different completion count from the request batch"
        )
    if logprobs is not None and len(logprobs) != len(pending):
        raise RuntimeError(
            "the policy generator returned a different logprob count from the request batch"
        )
    for index, item in enumerate(pending):
        future = item[3]
        if not future.cancelled():
            future.set_result(
                (
                    completion_ids[index],
                    None if logprobs is None else logprobs[index],
                )
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


def _openai_response(
    message: dict[str, Any],
    finish_reason: str,
    *,
    prompt_tokens: int,
    effective_max_tokens: int,
) -> dict[str, Any]:
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
        "posttrain_generation": {
            "prompt_tokens": prompt_tokens,
            "effective_max_tokens": effective_max_tokens,
        },
    }


def _structured_outputs(
    response_format: Mapping[str, object] | None,
    *,
    model_family: str,
) -> dict[str, object] | None:
    if response_format is None:
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return {"json_object": True}
    if kind != "json_schema":
        raise ValueError(f"unsupported environment response_format type {kind!r}")
    contract = response_format.get("json_schema")
    if not isinstance(contract, Mapping) or contract.get("strict") is not True:
        raise ValueError("environment JSON Schema must be strict")
    schema = contract.get("schema")
    if not isinstance(schema, Mapping):
        raise ValueError("environment JSON Schema response_format has no schema")
    structured_outputs: dict[str, object] = {
        "json": _xgrammar_json_schema(dict(schema))
    }
    if model_family == "gemma4":
        # Gemma can otherwise spend its constrained prefix on unrestricted
        # whitespace/control-token sequences without ever entering the JSON
        # object. This bounded pattern is the proven XGrammar contract used by
        # the earlier live Gemma distillation path.
        structured_outputs["whitespace_pattern"] = _GEMMA_JSON_WHITESPACE_PATTERN
    return structured_outputs


def _xgrammar_json_schema(value: object) -> object:
    """Create the generation-only schema supported by XGrammar.

    Policy environments retain and validate the canonical schema. Only the
    temporary wire copy used for constrained generation loses keywords that
    XGrammar 0.2.3 cannot compile.
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
def _generation_overrides(
    trainer: Any,
    *,
    max_tokens: int,
    structured_outputs: dict[str, object] | None,
) -> Iterator[None]:
    generation = getattr(trainer, "vllm_generation", None)
    generation_config = getattr(trainer, "generation_config", None)
    old_vllm_limit = getattr(generation, "max_completion_length", None)
    old_generation_kwargs = getattr(generation, "generation_kwargs", None)
    old_model_limit = getattr(generation_config, "max_new_tokens", None)
    if generation is not None:
        generation.max_completion_length = max_tokens
        kwargs = dict(old_generation_kwargs or {})
        if structured_outputs is None:
            kwargs.pop("structured_outputs", None)
        else:
            kwargs["structured_outputs"] = structured_outputs
        generation.generation_kwargs = kwargs
    if generation_config is not None:
        generation_config.max_new_tokens = max_tokens
    try:
        yield
    finally:
        if generation is not None:
            generation.max_completion_length = old_vllm_limit
            generation.generation_kwargs = old_generation_kwargs
        if generation_config is not None:
            generation_config.max_new_tokens = old_model_limit


__all__ = ["TrlPolicyGenerator"]
