from __future__ import annotations

from posttrain.common import Catalog, CatalogRef
from posttrain.environment import environment_catalog_decoders
from posttrain.eval import EvaluationPlan, evaluation_catalog_decoders


def test_catalog_decodes_typed_selection_policy() -> None:
    catalog = Catalog.open(
        {
            "layer_id": "selection-test",
            "environment": {
                "math": {
                    "id": "math",
                    "category": "reasoning",
                    "source": {
                        "package": "math-env",
                        "repository": "https://example.com/math-env.git",
                        "revision": "a" * 40,
                    },
                    "activation": {"kind": "python-factory", "reference": "math_env:create"},
                    "sampling": {"max_tokens": 256},
                    "num_tasks": 100,
                    "observation": {
                        "facets": [{"field": "subject", "dimension": "subject", "label": "Subject"}]
                    },
                }
            },
            "evaluation": {
                "math-balanced": {
                    "id": "math-balanced",
                    "kind": "domain",
                    "environments": ["math"],
                    "success": {
                        "math": {
                            "id": "correct",
                            "label": "Correct",
                            "source": {"namespace": "reward", "name": "correct"},
                            "predicate": {"operator": "gte", "value": 1.0},
                        }
                    },
                    "selection": {
                        "math": {
                            "kind": "minimum_then_proportional",
                            "num_tasks": 20,
                            "dimensions": ["subject"],
                            "minimum_per_stratum": 2,
                            "seed": 7,
                            "task_filter": {
                                "all_of": [{"dimension": "split", "operator": "eq", "values": ["test"]}]
                            },
                        }
                    },
                }
            },
        },
        decoders={**environment_catalog_decoders(), **evaluation_catalog_decoders()},
    )

    plan = catalog.resolve(CatalogRef("evaluation", "math-balanced")).value
    assert isinstance(plan, EvaluationPlan)
    policy = plan.selection_for("math")
    assert policy is not None
    assert policy.kind == "minimum_then_proportional"
    assert policy.num_tasks == 20
    assert policy.minimum_per_stratum == 2
    assert policy.task_filter.all_of[0].values == ("test",)


def test_legacy_plan_has_no_manifest_backed_selection() -> None:
    from posttrain.eval.programs import GENERAL_SMOKE

    assert GENERAL_SMOKE.selection_for(GENERAL_SMOKE.environments[0].id) is None
