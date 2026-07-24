"""veRL agent loop that delegates episode ownership to a portable Verifiers bridge."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

from ...integrations.verifiers import load_verifiers_bridge_snapshot
from ...online_rl import PolicyTurnRequest, PolicyTurnResult, RolloutBatch
from ...profiles import shape_soft_overlong_reward

try:
    from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
except ImportError as error:  # pragma: no cover - imported only by the isolated veRL runtime
    raise RuntimeError("PosttrainVerifiersAgentLoop must run inside the pinned veRL environment") from error


class VerlPolicyGenerator:
    """Expose veRL's already-loaded rollout server through the framework policy contract."""

    def __init__(self, server_manager: Any, tokenizer: Any, *, enable_thinking: bool) -> None:
        try:
            from renderers import Qwen35RendererConfig, create_renderer  # pyright: ignore[reportMissingImports]
        except ImportError as error:  # pragma: no cover - isolated runtime dependency
            raise RuntimeError("the veRL environment requires renderers with Qwen 3.5 support") from error
        self._server_manager = server_manager
        self._renderer = create_renderer(tokenizer, Qwen35RendererConfig(enable_thinking=enable_thinking))

    async def generate(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        messages = [cast(dict[str, Any], dict(message)) for message in request.messages]
        tools = [cast(dict[str, Any], dict(tool)) for tool in request.tools]
        renderer_messages = cast(Any, messages)
        renderer_tools = cast(Any, tools or None)
        rendered = None
        if request.previous_prompt_ids:
            rendered = self._renderer.bridge_to_next_turn(
                list(request.previous_prompt_ids),
                list(request.previous_completion_ids),
                renderer_messages[request.tail_start :],
                tools=renderer_tools,
            )
        if rendered is None:
            rendered = self._renderer.render(renderer_messages, tools=renderer_tools, add_generation_prompt=True)
        output = await self._server_manager.generate(
            request_id=request.session_id or uuid4().hex,
            prompt_ids=list(rendered.token_ids),
            sampling_params={
                "max_tokens": request.sampling.max_tokens,
                "temperature": request.sampling.temperature,
                "top_p": request.sampling.top_p,
                "logprobs": True,
            },
        )
        token_ids = tuple(int(value) for value in output.token_ids)
        if not token_ids:
            raise RuntimeError("the veRL rollout server returned an empty completion")
        logprobs = tuple(float(value) for value in (output.log_probs or ()))
        if len(logprobs) != len(token_ids):
            raise RuntimeError("veRL rollout log probabilities are not aligned with completion token ids")
        parsed = self._renderer.parse_response(list(token_ids), tools=renderer_tools)
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
        message: dict[str, Any] = {"role": "assistant", "content": parsed.content or None}
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
        return PolicyTurnResult(
            message=message,
            prompt_ids=tuple(int(value) for value in rendered.token_ids),
            completion_ids=token_ids,
            completion_logprobs=logprobs,
            finish_reason=finish_reason,
            prompt_message_spans=tuple(rendered.message_token_spans()),
            prompt_is_content=tuple(bool(value) for value in rendered.is_content),
            raw_response={
                "id": "posttrain-verl-policy-turn",
                "object": "chat.completion",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": _openai_message(message),
                        "finish_reason": finish_reason,
                    }
                ],
            },
        )


class PosttrainVerifiersAgentLoop(AgentLoopBase):
    """Run one native Verifiers trajectory for each veRL dataset row."""

    def __init__(
        self,
        *args: Any,
        bridge_snapshot: str,
        enable_thinking: bool = False,
        mask_truncated_completions: bool | None = False,
        max_completion_tokens: int,
        overlong_buffer_tokens: int | None = None,
        overlong_penalty_factor: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._bridge = load_verifiers_bridge_snapshot(Path(bridge_snapshot))
        self._generator = VerlPolicyGenerator(
            self.server_manager,
            self.tokenizer,
            enable_thinking=enable_thinking,
        )
        self._mask_truncated_completions = bool(mask_truncated_completions)
        self._max_completion_tokens = max_completion_tokens
        self._overlong_buffer_tokens = overlong_buffer_tokens
        self._overlong_penalty_factor = overlong_penalty_factor

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        del sampling_params
        example_id = str(kwargs["example_id"])
        step = int(kwargs.get("global_steps", 0))
        model_id = str(kwargs["model_id"])
        started = perf_counter()
        rollouts = await self._bridge.run(
            RolloutBatch(example_ids=(example_id,), step=step, model_id=model_id),
            self._generator,
        )
        if len(rollouts) != 1:
            raise RuntimeError("a veRL agent-loop row must produce exactly one Verifiers trajectory")
        rollout = rollouts[0]
        trace_calls = rollout.trace.payload.get("calls", [])
        num_turns = len(trace_calls) if isinstance(trace_calls, list) else 0
        if len(rollout.prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError("Verifiers trajectory prompt exceeds the selected veRL prompt length")
        if len(rollout.completion_ids) > self.rollout_config.response_length:
            raise ValueError("Verifiers trajectory response exceeds the selected veRL response length")
        reward = rollout.reward
        if self._overlong_buffer_tokens is not None:
            if self._overlong_penalty_factor is None:
                raise ValueError("veRL DAPO overlong shaping requires a penalty factor")
            reward = shape_soft_overlong_reward(
                reward,
                len(rollout.completion_ids),
                max_completion_tokens=self._max_completion_tokens,
                buffer_tokens=self._overlong_buffer_tokens,
                penalty_factor=self._overlong_penalty_factor,
            )
        response_mask = [int(value) for value in rollout.env_mask]
        if rollout.is_truncated and self._mask_truncated_completions:
            response_mask = [0] * len(response_mask)
        return AgentLoopOutput(
            prompt_ids=list(rollout.prompt_ids),
            response_ids=list(rollout.completion_ids),
            response_mask=response_mask,
            response_logprobs=list(rollout.sampling_logprobs),
            reward_score=reward,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(generate_sequences=perf_counter() - started),
            extra_fields={
                "rollout_trace_id": rollout.trace.external_id,
                "example_id": rollout.example_id,
                "is_truncated": rollout.is_truncated,
                "task_reward": rollout.reward,
                "algorithm_reward": reward,
                "min_global_steps": step,
                "max_global_steps": step,
            },
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


def _openai_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert the normalized Verifiers message back to OpenAI's wire shape."""

    projected = dict(message)
    tool_calls = projected.get("tool_calls")
    if isinstance(tool_calls, list):
        projected["tool_calls"] = [
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
    return projected


__all__ = ["PosttrainVerifiersAgentLoop", "VerlPolicyGenerator"]
