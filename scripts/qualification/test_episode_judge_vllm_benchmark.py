"""Regression coverage for the two-stage episode-judge replay harness."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_judge_vllm_benchmark as benchmark


def test_assessment_frame_transport_requires_every_canonical_field():
    canonical = benchmark.EpisodeAssessmentFrame.model_json_schema()

    assert set(benchmark._ASSESSMENT_FRAME_SCHEMA["properties"]) == set(canonical["properties"])
    assert set(benchmark._ASSESSMENT_FRAME_SCHEMA["required"]) == set(canonical["required"])


def test_chat_completion_request_retries_only_alternate_route_after_404(monkeypatch):
    calls = []

    def fake_json_request(url, *_args, **_kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(url, 404, "Not Found", None, None)
        return {"choices": []}

    monkeypatch.setattr(benchmark, "_json_request", fake_json_request)

    assert benchmark._chat_completion_request(
        "http://127.0.0.1:8123", {"model": "judge"}, 1.0, api_key="credential"
    ) == {"choices": []}
    assert calls == [
        "http://127.0.0.1:8123/v1/chat/completions",
        "http://127.0.0.1:8123/chat/completions",
    ]


def test_chat_completion_request_retains_bounded_non_404_provider_diagnostic(monkeypatch):
    def fake_json_request(url, *_args, **_kwargs):
        raise HTTPError(
            url,
            400,
            "Bad Request",
            {"server": "PosttrainCostGuard/1", "x-request-id": "safe-request-id"},
            io.BytesIO(b'{"error":{"message":"unsupported response format","code":400}}'),
        )

    monkeypatch.setattr(benchmark, "_json_request", fake_json_request)
    try:
        benchmark._chat_completion_request("http://example.test", {"model": "judge"}, 1.0, api_key="credential")
    except HTTPError as error:
        assert "unsupported response format" in str(error)
        assert "server=PosttrainCostGuard/1" in str(error)
        assert "request_id=safe-request-id" in str(error)
    else:
        raise AssertionError("non-404 provider error was unexpectedly admitted")


def test_json_request_enforces_a_wall_clock_deadline(monkeypatch):
    async def never_returns(*_args, **_kwargs):
        await benchmark.asyncio.sleep(0.1)
        return {"choices": []}

    monkeypatch.setattr(benchmark, "_async_json_request", never_returns)

    try:
        benchmark._json_request("http://example.test", {"model": "judge"}, 0.001)
    except TimeoutError as error:
        assert "wall-clock deadline" in str(error)
    else:
        raise AssertionError("idle transport was not released at the deadline")


def test_sampling_payload_honors_omitted_provider_specific_fields():
    assert (
        benchmark._sampling_payload(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            omit_temperature=True,
            omit_top_p=True,
            omit_top_k=True,
            chat_template_kwargs={"enable_thinking": False},
            omit_chat_template_kwargs=True,
        )
        == {}
    )


def test_strict_response_schema_inlines_pydantic_local_definitions():
    response = benchmark._response_format(
        "WireEpisodeVerdict",
        benchmark.WireEpisodeVerdict.model_json_schema(),
        "strict_schema",
    )
    schema = response["json_schema"]["schema"]
    serialized = json.dumps(schema, sort_keys=True)
    assert '"$defs"' not in serialized
    assert '"$ref"' not in serialized
    assert "assessments" in schema["properties"]


def test_current_production_protocol_rebuilds_explicit_evidence_indexes():
    cases = [
        {
            "case_id": "sample",
            "input_digest": "old",
            "source_input_digest": "source",
            "messages": [
                {"role": "system", "content": "old prompt"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "contract": "general-agent-episode@9",
                            "valid_message_ids": ["message-0", "message-1"],
                            "trajectory": [
                                {"message_id": "message-0", "role": "system"},
                                {"message_id": "message-1", "role": "user"},
                            ],
                        }
                    ),
                },
            ],
        }
    ]

    [refreshed] = benchmark._use_current_production_protocol(cases)
    request = json.loads(refreshed["messages"][1]["content"])
    assert refreshed["messages"][0]["content"] == benchmark.GENERAL_EPISODE_JUDGE_SYSTEM_PROMPT
    assert request["contract"] == benchmark.EPISODE_PROMPT_VERSION
    assert [entry["evidence_index"] for entry in request["trajectory"]] == [0, 1]
    assert request["harness_observation_diagnostics"] == []
    assert refreshed["source_input_digest"] == "source"
    assert refreshed["input_digest"] != "old"


def test_observation_diagnostics_surface_conflicting_temporal_aliases():
    trajectory = [
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "value": {
                        "results": [
                            {
                                "dueDate": "2026-03-15",
                                "due_on": "2024-02-15",
                                "due_at": "2024-02-15T17:00:00.000Z",
                            }
                        ]
                    }
                }
            ),
        }
    ]

    assert benchmark._observation_diagnostics(trajectory) == [
        {
            "diagnostic_id": "observation-conflict-0-due_date",
            "kind": "conflicting_observation_aliases",
            "evidence_index": 0,
            "field_family": "due_date",
            "observations": [
                {
                    "path": "$.value.results[0].dueDate",
                    "value": "2026-03-15",
                    "normalized_value": "2026-03-15",
                },
                {
                    "path": "$.value.results[0].due_on",
                    "value": "2024-02-15",
                    "normalized_value": "2024-02-15",
                },
                {
                    "path": "$.value.results[0].due_at",
                    "value": "2024-02-15T17:00:00.000Z",
                    "normalized_value": "2024-02-15",
                },
            ],
        }
    ]


def test_asana_contract_declares_alias_precedence_and_resolves_template_dates():
    [definition] = benchmark.enrich_tool_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "asana_create_task",
                    "description": "Create an Asana task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dueDate": {"type": ["string", "null"]},
                            "due_on": {"type": ["string", "null"]},
                        },
                    },
                },
            }
        ]
    )
    function = definition["function"]
    assert "Canonical due_date" in function["parameters"]["properties"]["dueDate"]["description"]
    assert "Compatibility alias" in function["parameters"]["properties"]["due_on"]["description"]
    assert function["x-automationbench-contract"]["result_semantics"]["precedence"]

    trajectory = [
        {
            "role": "tool",
            "name": "asana_create_task",
            "content": json.dumps({"results": [{"dueDate": "2026-03-15", "due_on": "2024-02-15"}]}),
        }
    ]
    assert benchmark._observation_diagnostics(trajectory, [definition]) == []


def test_assessment_frame_must_preserve_machine_detected_conflict_id():
    diagnostic = {
        "diagnostic_id": "observation-conflict-3-due_date",
        "evidence_index": 3,
    }
    dimensions = (
        "logical_correctness",
        "verification_self_correction",
        "answer_quality",
    )
    missed = SimpleNamespace(
        open_discrepancies=["none"],
        requirement_observations=[SimpleNamespace(discrepancy="none")],
        dimension_subchecks=SimpleNamespace(
            **{name: [SimpleNamespace(blocks_perfection=False)] for name in dimensions}
        ),
        perfection_blockers=SimpleNamespace(**{name: ["none"] for name in dimensions}),
    )
    preserved = SimpleNamespace(
        open_discrepancies=["observation-conflict-3-due_date remains unresolved"],
        requirement_observations=[
            SimpleNamespace(discrepancy="observation-conflict-3-due_date is unresolved because aliases disagree")
        ],
        dimension_subchecks=SimpleNamespace(**{name: [SimpleNamespace(blocks_perfection=True)] for name in dimensions}),
        perfection_blockers=SimpleNamespace(
            **{name: ["Conflicting due-date aliases remain unresolved."] for name in dimensions}
        ),
    )

    assert benchmark._assessment_frame_diagnostic_errors(missed, [diagnostic]) == [
        "open_discrepancies omitted observation-conflict-3-due_date",
        "requirement_observations omitted observation-conflict-3-due_date",
    ]
    assert benchmark._assessment_frame_diagnostic_errors(preserved, [diagnostic]) == []

    falsely_resolved = SimpleNamespace(
        open_discrepancies=["observation-conflict-3-due_date"],
        requirement_observations=[
            SimpleNamespace(discrepancy=("observation-conflict-3-due_date resolved by the same message-3 aliases"))
        ],
        dimension_subchecks=missed.dimension_subchecks,
        perfection_blockers=missed.perfection_blockers,
    )
    assert benchmark._assessment_frame_diagnostic_errors(falsely_resolved, [diagnostic]) == [
        "resolved observation-conflict-3-due_date did not cite a later environment observation"
    ]


def test_truncated_assessment_frame_does_not_fall_back_to_direct_verdict(monkeypatch):
    """A verdict without a valid frame would invalidate model-native evidence."""

    calls = []

    def fake_json_request(*_args, **_kwargs):
        calls.append(1)
        return {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"episode_understanding": "partial"}'},
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(benchmark, "_json_request", fake_json_request)
    case = {
        "case_id": "frame-length",
        "input_digest": "input",
        "source_input_digest": "source",
        "messages": [
            {"role": "system", "content": "judge"},
            {"role": "user", "content": '{"valid_message_ids": ["message-0"]}'},
        ],
    }

    result = benchmark._complete(
        base_url="http://judge.invalid",
        model="judge",
        case=case,
        max_tokens=128,
        timeout=1.0,
        chat_template_kwargs={},
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        omit_temperature=False,
        omit_top_p=False,
        wire_evidence_indexes=False,
        retain_content=False,
        restatement_first=True,
        restatement_max_tokens=128,
        review_after_verdict=False,
        api_key=None,
        extra_body={},
        omit_top_k=False,
        omit_chat_template_kwargs=False,
        response_format_mode="strict_schema",
    )

    assert len(calls) == 1
    assert result["finish_reason"] == "invalid_assessment_frame"
    assert result["structured_output_valid"] is False
