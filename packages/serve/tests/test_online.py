"""Tests for online serving operations."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import httpx
from posttrain.common import InferenceBinding, LocalArtifactRef
from posttrain.common.variants import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve import (
    Endpoint,
    GenerationRequest,
    GenerationResult,
    ServeLaunchRequest,
    generate_concurrently,
)
from posttrain.serve.backends.vllm import build_vllm_command
from posttrain.serve.online import generate, probe


def test_probe_checks_health_and_requested_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": QWEN_35_2B.base.repo_id}]})

    endpoint = Endpoint("http://model.test/v1", QWEN_35_2B.base.repo_id)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe(endpoint, client=client)

    assert result.healthy
    assert result.model_available
    assert result.models == (QWEN_35_2B.base.repo_id,)


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

    endpoint = Endpoint("http://model.test/v1", QWEN_35_2B.base.repo_id)
    request = GenerationRequest(
        endpoint=endpoint,
        messages=({"role": "user", "content": "Weather?"},),
        max_tokens=32,
        reasoning_mode="thinking",
        tools=({"type": "function", "function": {"name": "weather", "parameters": {}}},),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = generate(request, QWEN_35_2B, client=client)

    assert observed_payload["chat_template_kwargs"] == {"enable_thinking": True}
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


def test_vllm_command_contains_model_engine_frontend_and_template_contract(
    tmp_path: Path,
    lfm_screen_binding: InferenceBinding,
) -> None:
    template = tmp_path / "lfm-chat.jinja"
    command = build_vllm_command(
        ServeLaunchRequest(lfm_screen_binding),
        template,
    )

    assert command[1:3] == ("serve", LFM_25_12B_THINKING.base.repo_id)
    assert command[command.index("--revision") + 1] == LFM_25_12B_THINKING.base.revision
    assert command[command.index("--tool-call-parser") + 1] == "lfm2"
    assert command[command.index("--reasoning-parser") + 1] == "lfm2"
    assert command[command.index("--chat-template") + 1] == str(template)


def test_qwen_launch_command_keeps_8gb_text_only_constraints(qwen_screen_binding: InferenceBinding) -> None:
    command = build_vllm_command(ServeLaunchRequest(qwen_screen_binding))

    assert "--enforce-eager" in command
    assert "--skip-mm-profiling" in command
    mm_limits = json.loads(command[command.index("--limit-mm-per-prompt") + 1])
    assert mm_limits == {"image": 0, "video": 0, "audio": 0}


def test_managed_server_uses_the_inference_binding_startup_budget(
    qwen_screen_binding: InferenceBinding,
) -> None:
    request = ServeLaunchRequest(replace(qwen_screen_binding, startup_timeout_seconds=600))
    explicit = ServeLaunchRequest(qwen_screen_binding, startup_timeout_seconds=30)

    assert request.effective_startup_timeout_seconds == 600
    assert explicit.effective_startup_timeout_seconds == 30


def test_vllm_launches_materialized_peft_adapter_without_treating_it_as_base_weights(
    tmp_path: Path,
    qwen_screen_binding: InferenceBinding,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter = replace(
        QWEN_35_2B,
        id="models/qwen3.5-2b/sft-qlora-test",
        artifact=LocalArtifactRef(adapter_dir, "a" * 64),
        form="peft-adapter",
        revision=None,
        digest="a" * 64,
        parent=QWEN_35_2B.id,
        provenance={"parameter_update_kind": "qlora"},
    )
    request = ServeLaunchRequest(replace(qwen_screen_binding, model=adapter))

    command = build_vllm_command(request)

    assert command[1:3] == ("serve", QWEN_35_2B.base.repo_id)
    assert command[command.index("--served-model-name") + 1] == QWEN_35_2B.base.repo_id
    assert command[command.index("--lora-modules") + 1] == f"{adapter.id}={adapter_dir}"
    assert "--enable-lora" in command
    assert request.endpoint.model == adapter.id


def test_vllm_launches_materialized_quantized_weights_as_the_selected_variant(
    tmp_path: Path,
    qwen_screen_binding: InferenceBinding,
) -> None:
    weights = tmp_path / "awq"
    weights.mkdir()
    quantized = replace(
        QWEN_35_2B,
        id="models/qwen3.5-2b@awq-test",
        artifact=LocalArtifactRef(weights, "b" * 64),
        form="weight-quantized",
        weight_precision="int4",
        revision=None,
        digest="b" * 64,
        quantization={"method": "awq", "bits": 4},
        parent=QWEN_35_2B.id,
    )
    request = ServeLaunchRequest(replace(qwen_screen_binding, model=quantized))

    command = build_vllm_command(request)

    assert command[1:3] == ("serve", str(weights))
    assert "--revision" not in command
    assert "--enable-lora" not in command
    assert command[command.index("--served-model-name") + 1] == quantized.id


def test_concurrent_generation_is_bounded_and_preserves_request_order() -> None:
    endpoint = Endpoint("http://model.test/v1", QWEN_35_2B.base.repo_id)
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
