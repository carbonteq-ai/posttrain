import json

import pytest
from calibrate_automationbench_judge import load_fixture


def test_external_fixture_has_stable_digest(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "id": "reviewed-v1",
                "scope": "held-out",
                "repetitions": 3,
                "acceptance": "all labels",
                "cases": [
                    {
                        "id": "case-1",
                        "messages": [{"role": "assistant", "content": "Done"}],
                        "erroneous": False,
                        "quality_range": [0.7, 1.0],
                    }
                ],
            }
        )
    )

    cases, repetitions, manifest = load_fixture(fixture)

    assert repetitions == 3
    assert cases[0]["id"] == "case-1"
    assert len(manifest["fixture_sha256"]) == 64


def test_fixture_rejects_duplicate_case_ids(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    case = {
        "id": "duplicate",
        "messages": [{"role": "assistant", "content": "Done"}],
        "erroneous": False,
        "quality_range": [0.7, 1.0],
    }
    fixture.write_text(json.dumps({"repetitions": 1, "cases": [case, case]}))

    with pytest.raises(ValueError, match="unique"):
        load_fixture(fixture)


def test_fixture_requires_expectation_for_every_assistant_turn(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "repetitions": 1,
                "cases": [
                    {
                        "id": "multi-turn",
                        "messages": [
                            {"role": "assistant", "content": "First"},
                            {"role": "tool", "content": "Observation"},
                            {"role": "assistant", "content": "Second"},
                        ],
                        "turn_expectations": [
                            {
                                "turn_id": "assistant-0",
                                "erroneous": False,
                                "quality_range": [0.7, 1.0],
                            }
                        ],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="every assistant turn"):
        load_fixture(fixture)
