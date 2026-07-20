from __future__ import annotations

import json
from pathlib import Path

import httpx
from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve import (
    LFM25_VLLM,
    QWEN35_VLLM_TEXT,
    Endpoint,
    GenerationRequest,
    LaunchRequest,
)
from posttrain.serve.backends.vllm import build_vllm_command
from posttrain.serve.online import generate, probe


def test_probe_checks_health_and_requested_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": QWEN_35_2B.artifact.repo_id}]})

    endpoint = Endpoint("http://model.test/v1", QWEN_35_2B.artifact.repo_id)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe(endpoint, client=client)

    assert result.healthy
    assert result.model_available
    assert result.models == (QWEN_35_2B.artifact.repo_id,)


def test_streaming_generation_preserves_reasoning_tools_usage_and_raw_events() -> None:
    observed_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_payload
        observed_payload = json.loads(request.read())
        body = "\n".join(
            [
                'data: {"choices":[{"delta":{"reasoning":"checking "},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"weather","arguments":"{\\"city\\":\\"Paris\\"}"}}]},"finish_reason":"tool_calls"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(200, text=body)

    endpoint = Endpoint("http://model.test/v1", QWEN_35_2B.artifact.repo_id)
    request = GenerationRequest(
        endpoint=endpoint,
        messages=({"role": "user", "content": "Weather?"},),
        max_tokens=32,
        reasoning_mode="thinking",
        tools=({"type": "function", "function": {"name": "weather", "parameters": {}}},),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = generate(request, QWEN_35_2B, client=client)

    assert observed_payload["enable_thinking"] is True
    assert observed_payload["tool_choice"] == "auto"
    assert result.reasoning == "checking "
    assert result.finish_reason == "tool_calls"
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert result.ttft_seconds is not None
    assert result.tool_call_deltas[0]["function"]["name"] == "weather"
    assert len(result.events) == 3


def test_vllm_command_contains_model_engine_frontend_and_template_contract(tmp_path: Path) -> None:
    template = tmp_path / "lfm-chat.jinja"
    command = build_vllm_command(LaunchRequest(LFM_25_12B_THINKING, LFM25_VLLM), template)

    assert command[1:3] == ("serve", LFM_25_12B_THINKING.artifact.repo_id)
    assert command[command.index("--revision") + 1] == LFM_25_12B_THINKING.artifact.revision
    assert command[command.index("--tool-call-parser") + 1] == "lfm2"
    assert command[command.index("--reasoning-parser") + 1] == "deepseek_r1"
    assert command[command.index("--chat-template") + 1] == str(template)


def test_qwen_launch_command_keeps_8gb_text_only_constraints() -> None:
    command = build_vllm_command(LaunchRequest(QWEN_35_2B, QWEN35_VLLM_TEXT))

    assert "--enforce-eager" in command
    assert "--skip-mm-profiling" in command
    mm_limits = json.loads(command[command.index("--limit-mm-per-prompt") + 1])
    assert mm_limits == {"image": 0, "video": 0}
