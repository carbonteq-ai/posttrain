from __future__ import annotations

from posttrain_observatory.evaluation_contracts import read_evaluation_contract


def test_versioned_contract_is_read_from_run_inputs() -> None:
    result = read_evaluation_contract(
        {
            "evaluation": {
                "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 1},
                "plan": {"id": "ifeval-v1"},
                "environment": {"id": "ifeval"},
                "signal_manifest": {"schema_version": "evaluation-signals/v1"},
                "native_evidence": {"schema_id": "verifiers.trace", "schema_version": "v1"},
                "population": {"split": "train"},
            }
        }
    )

    assert result.state == "versioned"
    assert result.contract_id == "posttrain.eval.verifiers-observation"
    assert result.contract_version == 1
    assert result.plan["id"] == "ifeval-v1"
    assert result.population["split"] == "train"


def test_v2_contract_retains_typed_success_definition() -> None:
    result = read_evaluation_contract(
        {
            "evaluation": {
                "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 2},
                "plan": {
                    "id": "reasoning-gym-v2",
                    "success": {
                        "id": "full-credit-solution",
                        "label": "Full-credit solution",
                        "source": {"namespace": "metric", "name": "native_score"},
                        "predicate": {"operator": "gte", "value": 0.99, "tolerance": 0.0},
                        "missing": "error",
                    },
                },
                "environment": {"id": "reasoning-gym"},
                "signal_manifest": {"schema_version": "evaluation-signals/v1"},
                "native_evidence": {"schema_id": "verifiers.trace", "schema_version": "v1"},
                "population": {},
            }
        }
    )

    assert result.state == "versioned"
    assert result.contract_version == 2
    assert result.plan["success"]["source"] == {"namespace": "metric", "name": "native_score"}  # type: ignore[index]


def test_v3_contract_retains_compound_breakdowns() -> None:
    result = read_evaluation_contract(
        {
            "evaluation": {
                "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 3},
                "plan": {
                    "id": "math-python-v2",
                    "success": None,
                    "breakdowns": [
                        {
                            "id": "problem-type-by-difficulty",
                            "label": "Problem type × difficulty",
                            "dimensions": ["problem_type", "difficulty"],
                            "presentation": "matrix",
                            "multi_value": "reject",
                            "missing": "exclude",
                        }
                    ],
                },
                "environment": {"id": "math-python"},
                "signal_manifest": {"schema_version": "evaluation-signals/v1"},
                "native_evidence": {"schema_id": "verifiers.trace", "schema_version": "v1"},
                "population": {},
            }
        }
    )

    assert result.state == "versioned"
    assert result.contract_version == 3
    assert result.plan["breakdowns"][0]["dimensions"] == ["problem_type", "difficulty"]  # type: ignore[index]


def test_unknown_contract_is_inspectable_but_not_interpreted() -> None:
    result = read_evaluation_contract(
        {
            "evaluation": {
                "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 99},
                "plan": {"id": "future"},
            }
        }
    )

    assert result.state == "unsupported"
    assert result.contract_version == 99
    assert result.plan == {}
    assert result.raw["plan"] == {"id": "future"}


def test_missing_contract_is_legacy_without_catalog_lookup() -> None:
    result = read_evaluation_contract({"environment": {"selection_id": "env/legacy"}})

    assert result.state == "legacy"
    assert result.contract_id is None
    assert result.raw == {}
