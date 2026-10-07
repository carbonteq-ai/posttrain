"""Declared task labels survive the training bridge without entering prompts."""

from types import SimpleNamespace

import pytest
from posttrain.train.integrations.verifiers import _record_task_facets, _rollout_dataset, _task_facet_values


def test_multilabel_facets_survive_extraction_and_observation_recovery():
    task = SimpleNamespace(data={"domain": "finance", "capabilities": ["policy_retrieval", "arithmetic", "arithmetic"]})
    facets = _task_facet_values(task, ("domain", "capabilities"))
    assert facets == {"domain": "finance", "capabilities": ["arithmetic", "policy_retrieval"]}
    assert _record_task_facets({"task_facets": facets}) == facets


def test_tuple_and_empty_labels_are_supported_without_changing_scalar_values():
    task = SimpleNamespace(data=SimpleNamespace(labels=("routing", "approval"), guards=[], domain="hr", count=2))
    assert _task_facet_values(task, ("labels", "guards", "domain", "count")) == {
        "labels": ["approval", "routing"],
        "guards": [],
        "domain": "hr",
        "count": 2,
    }


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), [""], [" "], ["routing", 2], {"label": "routing"}])
def test_invalid_declared_facets_fail_at_extraction(value):
    with pytest.raises(ValueError, match="valid observation facet 'labels'"):
        _task_facet_values(SimpleNamespace(data={"labels": value}), ("labels",))


def test_observation_recovery_keeps_existing_invalid_field_filtering():
    assert _record_task_facets({"task_facets": {"domain": "sales", "bad": {"nested": True}}}) == {"domain": "sales"}


def test_labels_enter_dataset_metadata_without_changing_prompt():
    task = SimpleNamespace(
        data=SimpleNamespace(prompt="Find the approved recipient.", capabilities=["policy_retrieval"])
    )
    dataset = _rollout_dataset("tasks", "revision", "environment", {0: task}, ("capabilities",))
    assert dataset.examples[0].prompt == "Find the approved recipient."
    assert dataset.examples[0].metadata["capabilities"] == ["policy_retrieval"]
