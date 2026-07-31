"""TRL policy-generation adapter for environment-driven online-RL bridges."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
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
        self._max_completion_length = settings.max_completion_length
        self._lock = asyncio.Lock()
        self._pending: list[
            tuple[
                Sequence[int],
                asyncio.Future[tuple[Sequence[int], Sequence[float] | None]],
            ]
        ] = []
        self._flush_task: asyncio.Task[None] | None = None

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

        completion_ids, logprobs = await self._generate_tokens(rendered.token_ids)
        token_ids = tuple(int(value) for value in completion_ids)
        if not token_ids:
            raise RuntimeError("the policy generator returned an empty completion")
        sampled_logprobs = () if logprobs is None else tuple(float(value) for value in logprobs)
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

    async def _generate_tokens(
        self,
        prompt_ids: Sequence[int],
    ) -> tuple[Sequence[int], Sequence[float] | None]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[Sequence[int], Sequence[float] | None]] = loop.create_future()
        self._pending.append((prompt_ids, future))
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
                asyncio.Future[tuple[Sequence[int], Sequence[float] | None]],
            ]
        ] = []
        try:
            # Waiting for the trainer lock yields to the event loop. Drain until
            # no later turn remains queued so none can be stranded behind this
            # already-scheduled flush task.
            while self._pending:
                pending, self._pending = self._pending, []
                active = [(prompt_ids, future) for prompt_ids, future in pending if not future.cancelled()]
                if not active:
                    continue
                async with self._lock:
                    completion_ids, logprobs = self._trainer._generate_single_turn(  # noqa: SLF001 - pinned adapter
                        [list(prompt_ids) for prompt_ids, _future in active],
                        None,
                        {},
                    )
                if len(completion_ids) != len(active):
                    raise RuntimeError(
                        f"the policy generator returned {len(completion_ids)} completions for {len(active)} prompts"
                    )
                if logprobs is not None and len(logprobs) != len(active):
                    raise RuntimeError(
                        f"the policy generator returned {len(logprobs)} logprob rows for {len(active)} prompts"
                    )
                for index, (_prompt_ids, future) in enumerate(active):
                    if not future.cancelled():
                        future.set_result(
                            (
                                completion_ids[index],
                                None if logprobs is None else logprobs[index],
                            )
                        )
        except BaseException as exc:
            for _prompt_ids, future in [*pending, *self._pending]:
                if not future.done():
                    future.set_exception(exc)
            self._pending = []
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.create_task(self._flush_pending())


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
