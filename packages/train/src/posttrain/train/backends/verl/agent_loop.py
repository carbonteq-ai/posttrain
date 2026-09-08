"""veRL agent loop that delegates episode ownership to a portable Verifiers bridge."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

from ...integrations.verifiers import load_verifiers_bridge_snapshot
from ...online_rl import EnvironmentRollout, PolicySampling, PolicyTurnRequest, PolicyTurnResult, RolloutBatch
from ...policy_messages import parsed_policy_message
from ...profiles import shape_soft_overlong_reward
from .reward_fields import streaming_reward_extra_info, structured_reward_metadata, training_response_mask

try:
    from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
except ImportError as error:  # pragma: no cover - imported only by the isolated veRL runtime
    raise RuntimeError("PosttrainVerifiersAgentLoop must run inside the pinned veRL environment") from error


_SAMPLING_OVERRIDE_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "logprobs",
    }
)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"veRL sampling override {name!r} must be a finite number")
    return float(value)


def _validated_sampling_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Validate native veRL sampling controls before they reach a rollout server."""
    unknown = set(overrides).difference(_SAMPLING_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"unsupported veRL sampling overrides: {', '.join(sorted(unknown))}")
    result: dict[str, Any] = {}
    if "max_tokens" in overrides:
        value = overrides["max_tokens"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("veRL sampling override 'max_tokens' must be a positive integer")
        result["max_tokens"] = value
    if "temperature" in overrides:
        value = _number(overrides["temperature"], "temperature")
        if value < 0:
            raise ValueError("veRL sampling override 'temperature' cannot be negative")
        result["temperature"] = value
    if "top_p" in overrides:
        value = _number(overrides["top_p"], "top_p")
        if not 0 < value <= 1:
            raise ValueError("veRL sampling override 'top_p' must be in (0, 1]")
        result["top_p"] = value
    if "top_k" in overrides:
        value = overrides["top_k"]
        if isinstance(value, bool) or not isinstance(value, int) or value < -1:
            raise ValueError("veRL sampling override 'top_k' must be an integer greater than or equal to -1")
        result["top_k"] = value
    if "min_p" in overrides:
        value = overrides["min_p"]
        if value is not None:
            value = _number(value, "min_p")
            if not 0 <= value <= 1:
                raise ValueError("veRL sampling override 'min_p' must be in [0, 1]")
        result["min_p"] = value
    if "repetition_penalty" in overrides:
        value = _number(overrides["repetition_penalty"], "repetition_penalty")
        if value <= 0:
            raise ValueError("veRL sampling override 'repetition_penalty' must be positive")
        result["repetition_penalty"] = value
    if "presence_penalty" in overrides:
        value = _number(overrides["presence_penalty"], "presence_penalty")
        if not -2 <= value <= 2:
            raise ValueError("veRL sampling override 'presence_penalty' must be in [-2, 2]")
        result["presence_penalty"] = value
    if "logprobs" in overrides and overrides["logprobs"] is not True:
        raise ValueError("veRL Verifiers rollouts require logprobs=true")
    return result


def _effective_sampling(base: PolicySampling, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge phase overrides without allowing them to expand environment output limits."""
    max_tokens = overrides.get("max_tokens", base.max_tokens)
    if max_tokens > base.max_tokens:
        raise ValueError(
            "veRL sampling override 'max_tokens' exceeds the environment output limit: "
            f"{max_tokens} > {base.max_tokens}"
        )
    return {
        "max_tokens": max_tokens,
        "temperature": overrides.get("temperature", base.temperature),
        "top_p": overrides.get("top_p", base.top_p),
        "top_k": overrides.get("top_k", base.top_k),
        "min_p": overrides.get("min_p", base.min_p),
        "repetition_penalty": overrides.get("repetition_penalty", base.repetition_penalty),
        "presence_penalty": overrides.get("presence_penalty", base.presence_penalty),
    }


class VerlPolicyGenerator:
    """Expose veRL's already-loaded rollout server through the framework policy contract."""

    def __init__(
        self,
        server_manager: Any,
        tokenizer: Any,
        *,
        enable_thinking: bool,
        sampling_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            from renderers import Qwen35RendererConfig, create_renderer  # pyright: ignore[reportMissingImports]
        except ImportError as error:  # pragma: no cover - isolated runtime dependency
            raise RuntimeError("the veRL environment requires renderers with Qwen 3.5 support") from error
        self._server_manager = server_manager
        self._tokenizer = tokenizer
        self._renderer = create_renderer(tokenizer, Qwen35RendererConfig(enable_thinking=enable_thinking))
        self._sampling_overrides = _validated_sampling_overrides(sampling_overrides or {})

    def set_sampling_overrides(self, overrides: Mapping[str, Any]) -> None:
        """Install the veRL phase/per-row sampling controls for this episode."""
        self._sampling_overrides = _validated_sampling_overrides(overrides)

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
        sampling = _effective_sampling(request.sampling, self._sampling_overrides)
        output = await self._server_manager.generate(
            request_id=request.session_id or uuid4().hex,
            prompt_ids=list(rendered.token_ids),
            sampling_params={key: value for key, value in {
                "max_tokens": sampling["max_tokens"],
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "top_k": sampling["top_k"],
                "min_p": sampling["min_p"],
                "repetition_penalty": sampling["repetition_penalty"],
                "presence_penalty": sampling["presence_penalty"],
                "logprobs": True,
            }.items() if value is not None},
        )
        token_ids = tuple(int(value) for value in output.token_ids)
        if not token_ids:
            raise RuntimeError("the veRL rollout server returned an empty completion")
        logprobs = tuple(float(value) for value in (output.log_probs or ()))
        if len(logprobs) != len(token_ids):
            raise RuntimeError("veRL rollout log probabilities are not aligned with completion token ids")
        parsed = self._renderer.parse_response(list(token_ids), tools=renderer_tools)
        message = parsed_policy_message(parsed, token_ids, self._tokenizer)
        finish_reason = _finish_reason(
            token_ids,
            frozenset(self._renderer.get_stop_token_ids()),
            bool(message.get("tool_calls")),
            int(sampling["max_tokens"]),
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
        emit_sampo_metadata: bool = False,
        structured_algorithm: str | None = None,
        reward_component_names: list[str] | None = None,
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
        self._emit_sampo_metadata = emit_sampo_metadata
        self._structured_algorithm = structured_algorithm
        self._reward_component_names = tuple(reward_component_names or ())

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        self._generator.set_sampling_overrides(sampling_params)
        example_id = str(kwargs["example_id"])
        step = int(kwargs.get("global_steps", 0))
        model_id = str(kwargs["model_id"])
        started = perf_counter()
        groups: tuple[str, ...] = ()
        identities: tuple[str, ...] = ()
        if self._structured_algorithm is not None:
            if "uid" not in kwargs or "session_id" not in kwargs:
                raise ValueError("structured veRL rollouts require native prompt-occurrence and session identities")
            groups = (f"{self._bridge.run_id}/{step}/{kwargs['uid']}",)
            identities = (f"{groups[0]}/{kwargs['session_id']}",)
        rollouts = await self._bridge.run(
            RolloutBatch(example_ids=(example_id,), step=step, model_id=model_id,
                         prompt_group_ids=groups, rollout_ids=identities),
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
        response_mask = training_response_mask(
            rollout.env_mask,
            is_truncated=rollout.is_truncated,
            mask_truncated_completions=self._mask_truncated_completions,
            requires_complete_group=self._emit_sampo_metadata,
        )
        extra_fields: dict[str, Any] = {
            "rollout_trace_id": rollout.trace.external_id,
            "example_id": rollout.example_id,
            "is_truncated": rollout.is_truncated,
            "task_reward": rollout.reward,
            "algorithm_reward": reward,
            # veRL V1's dynamic group filter classifies completed trajectories
            # before materializing the training batch. Its streaming path reads
            # the selected metric from this native nested field rather than
            # from the token-level rm_scores tensor.
            "reward_extra_info": streaming_reward_extra_info(
                task_reward=rollout.reward,
                algorithm_reward=reward,
            ),
            "min_global_steps": step,
            "max_global_steps": step,
        }
        if self._emit_sampo_metadata:
            extra_fields.update(_sampo_metadata(rollout))
        if self._structured_algorithm is not None:
            extra_fields["structured_rewards"] = structured_reward_metadata(
                rollout, component_names=self._reward_component_names,
                require_process=self._structured_algorithm == "capo",
            )
        _append_rollout_reward_record(
            trace_id=rollout.trace.external_id,
            step=step,
            task_reward=rollout.reward,
            algorithm_reward=reward,
        )
        return AgentLoopOutput(
            prompt_ids=list(rollout.prompt_ids),
            response_ids=list(rollout.completion_ids),
            response_mask=response_mask,
            response_logprobs=list(rollout.sampling_logprobs),
            reward_score=reward,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(generate_sequences=perf_counter() - started),
            extra_fields=extra_fields,
        )


def _append_rollout_reward_record(
    *,
    trace_id: str,
    step: int,
    task_reward: float,
    algorithm_reward: float,
) -> None:
    """Journal the post-shaping scalar in the isolated worker for parent replay."""

    destination = os.environ.get("POSTTRAIN_VERL_ROLLOUT_REWARDS_PATH")
    if not destination:
        return
    payload = (
        json.dumps(
            {
                "trace_id": trace_id,
                "step": step,
                "task_reward": task_reward,
                "algorithm_reward": algorithm_reward,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    # Each small append is one complete record. The parent is the only process
    # that turns this journal into tracking evidence after the worker exits.
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload.encode("utf-8"))
    finally:
        os.close(descriptor)


def _sampo_metadata(rollout: EnvironmentRollout) -> dict[str, Any]:
    if not rollout.turns:
        raise RuntimeError("SAMPO requires sampled assistant-turn metadata")
    return {
        "sampo_prompt_group_id": rollout.example_id,
        "sampo_turn_lengths": [turn.completion_end - turn.completion_start for turn in rollout.turns],
        "sampo_turn_spans": [[turn.completion_start, turn.completion_end] for turn in rollout.turns],
        "sampo_anchor_state_keys": [turn.anchor_state_key for turn in rollout.turns],
        "sampo_step_rewards": [turn.step_reward for turn in rollout.turns],
    }


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
