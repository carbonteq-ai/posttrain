from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve import (
    LFM25_VLLM,
    QWEN35_VLLM_TEXT,
    Endpoint,
    GenerationRequest,
    GenerationResult,
    LaunchRequest,
    generate_concurrently,
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
    assert result.summary()["reasoning_characters"] == len("checking ")
    assert len(result.as_json()["events"]) == 3


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
    assert "--language-model-only" in command
    assert "--skip-mm-profiling" in command
    hf_overrides = json.loads(command[command.index("--hf-overrides") + 1])
    assert hf_overrides == {"architectures": ["Qwen3_5ForCausalLM"]}
    model_class_overrides = json.loads(command[command.index("--model-class-overrides") + 1])
    assert model_class_overrides == {
        "Qwen3_5ForCausalLM": "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM"
    }
    mm_limits = json.loads(command[command.index("--limit-mm-per-prompt") + 1])
    assert mm_limits == {"image": 0, "video": 0}


def test_concurrent_generation_is_bounded_and_preserves_request_order() -> None:
    endpoint = Endpoint("http://model.test/v1", QWEN_35_2B.artifact.repo_id)
    requests = tuple(
        GenerationRequest(endpoint, ({"role": "user", "content": str(index)},), max_tokens=8) for index in range(4)
    )
    lock = threading.Lock()
    active = 0
    peak = 0

    def runner(request: GenerationRequest, model) -> GenerationResult:
        nonlocal active, peak
        assert model is QWEN_35_2B
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        content = str(request.messages[0]["content"])
        return GenerationResult(content, "", (), 1, 1, 0.01, 0.005, "stop", ())

    results = generate_concurrently(requests, QWEN_35_2B, max_concurrency=2, runner=runner)

    assert peak == 2
    assert tuple(result.content for result in results) == ("0", "1", "2", "3")
