"""Exercise stage plumbing with a fake generator; no learning/quality claims."""

import hashlib
import json
import sys
from types import SimpleNamespace

from generate_luna_ssr_pilot import run


def test_quality_flow_requires_evidence_before_rationale_and_retains_critique(tmp_path, monkeypatch):
    inputs = []
    calls = []

    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            inputs.append(json.loads(messages[1]["content"]))
            return [len(inputs)]

    class Engine:
        llm_engine = SimpleNamespace(vllm_config="fake")

        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling, **kwargs):
            results = []
            for prompt in prompts:
                data = inputs[prompt["prompt_token_ids"][0] - 1]
                stage = data.get("stage", "plan")
                calls.append(stage)
                if stage == "plan":
                    value = {
                        "workflow_plan": [
                            {"phase": phase, "operation": "Check evidence"}
                            for phase in ("READ", "DECIDE", "ACT", "VERIFY")
                        ],
                        "reference_issues": [],
                    }
                elif stage == "prefix_evidence_ledger":
                    assert "teacher_only_guidance" not in data
                    value = {"facts": [{"evidence_ref": "task", "quote": "Do the task"}], "missing_information": []}
                elif stage == "action_rationale":
                    assert data["verified_quote_ledger"]["facts"]
                    value = {
                        "action_index": 0,
                        "supported": True,
                        "rationale": "Follow the request.",
                        "evidence_refs": ["task"],
                    }
                else:
                    assert stage == "annotation_critique"
                    value = {"acceptable": False, "issues": [{"kind": "needs_execution"}], "missing_verifications": []}
                response = SimpleNamespace(text=json.dumps(value), token_ids=[3], finish_reason="stop")
                results.append(SimpleNamespace(outputs=[response]))
            return results

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "posttrain.common.cuda", SimpleNamespace(activate_cuda_toolkit=lambda torch: None))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())),
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    cases = tmp_path / "cases"
    cases.mkdir()
    request = {
        "system": "annotate",
        "stage": "action_rationale",
        "action_index": 0,
        "prefix_evidence": [{"id": "task", "value": "Do the task"}],
        "target_action": {"name": "search", "arguments": {}},
        "teacher_only_guidance": {},
    }
    case = {
        "task": "simple.test",
        "skeleton_request": {"system": "plan", "available_tool_definitions": []},
        "rationale_request_templates": [request],
    }
    raw = json.dumps(case).encode()
    (cases / "case.json").write_bytes(raw)
    (cases / "manifest.json").write_text(
        json.dumps({"tasks": [{"case_file": "case.json", "case_sha256": hashlib.sha256(raw).hexdigest()}]})
    )
    output = tmp_path / "out"
    run(SimpleNamespace(model=tmp_path / "model", cases=cases, output=output, quality_stages=True))
    assert calls == ["plan", "prefix_evidence_ledger", "action_rationale", "annotation_critique"]
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["completed"] and not receipt["sft_qualified"]
    assert not receipt["raw_generator_thinking_export_allowed"]
    critique = json.loads((output / "critiques.jsonl").read_text())
    assert critique["parsed"]["acceptable"] is False  # Never turned into SFT admission.
