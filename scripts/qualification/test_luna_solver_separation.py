import json
from pathlib import Path

from luna_solver_separation import argument_provenance, observation_bound_sources, separated_sources


def test_actual_sources_hide_answers_but_preserve_dependencies_and_public_values():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/evidence/luna-ssr-sft-20261018/solver-pilot2-20261019-r1/4b/materials.json"
    )
    for material in json.loads(path.read_text()).values():
        contract, actions, receipt = separated_sources(material)
        rendered = json.dumps([contract, actions])
        for value in ("+1-555-0707", "003007", "marcus.rivera@bluemtn.example.com", "msg_3007", "CPART01"):
            assert value not in rendered
        assert "returned_before_dispatch" in rendered
        assert [c["check_id"] for c in contract["checks"]] == [
            c["check_id"] for c in material["goal_manifest"]["checks"]
        ]
        assert len(actions) == len(material["reference_actions"])
        assert receipt["private_literal_count"] > 0
        if "hubspot" in material["task"]:
            assert "Vertex Labs Partnership" in rendered and "75000" in rendered


def test_same_hidden_value_keeps_equality_without_disclosing_value():
    material = {
        "user_request": "Update the contact.",
        "tool_definitions": [],
        "goal_manifest": {"value": "hidden-123", "record_id": "hidden-123"},
        "reference_actions": [],
    }
    contract, _, _ = separated_sources(material)
    assert contract["value"]["symbol"] == contract["record_id"]["symbol"]
    assert "hidden-123" not in json.dumps(contract)


def test_changing_private_answer_values_does_not_change_preparation_input():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/evidence/luna-ssr-sft-20261018/solver-pilot2-20261019-r1/4b/materials.json"
    )
    material = next(m for m in json.loads(path.read_text()).values() if "sf_contact" in m["task"])
    altered = json.loads(json.dumps(material).replace("+1-555-0707", "+9-444-9898").replace("003007", "003999"))
    original_input = separated_sources(material)
    changed_input = separated_sources(altered)
    assert original_input == changed_input


def test_argument_provenance_distinguishes_api_options_from_retrieved_values():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/evidence/luna-ssr-sft-20261018/solver-pilot2-20261019-r1/4b/materials.json"
    )
    for material in json.loads(path.read_text()).values():
        rows = argument_provenance(material)
        if "hubspot" in material["task"]:
            stage = rows[1]["argument_provenance"]["dealstage"]
            assert stage["origin"] == "demonstrated_api_option"
            assert stage["value"] == "appointmentscheduled"
        else:
            update = rows[-1]["argument_provenance"]
            assert update["id"]["origin"] == "prior_tool_result"
            assert any(e["action_id"] == "a3" for e in update["id"]["evidence"])
            assert any(
                e["action_id"] == "a2" and e["result_path"].endswith("body_plain")
                for e in update["mobile_phone"]["evidence"]
            )
            assert "+1-555-0707" not in json.dumps(rows)


def test_observation_bound_inputs_have_no_opaque_symbols_or_hidden_answers():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/evidence/luna-ssr-sft-20261018/solver-pilot2-20261019-r1/4b/materials.json"
    )
    for material in json.loads(path.read_text()).values():
        contract, actions, receipt = observation_bound_sources(material)
        text = json.dumps([contract, actions])
        assert "retrieved_" not in text
        assert "+1-555-0707" not in text and "003007" not in text and "CPART01" not in text
        assert "obtain_from_tool_result" in text
        assert "returned_before_dispatch" in text
        assert receipt["private_literal_count"] > 0
        assert "prior_read" not in text and "prior_email_read" not in text
        for check in contract["checks"]:
            for dep in check.get("operational_dependencies", []):
                assert dep["matching_conditions"]
                if dep["source"] == "reads" and "hubspot" in material["task"]:
                    assert "body_plain" in json.dumps(dep)
                    assert '"value": "body_plain"' in json.dumps(dep)
