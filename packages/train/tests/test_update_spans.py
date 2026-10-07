"""Derived reasoning/answer spans address original actions and select by role."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from posttrain.train.backends.policy_update_admission import AdmittedNativePopulation
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.update_records import ActionSelection, InvalidPolicyUpdate, PolicyVersions

from .test_update_admission import source
from .test_update_resolution import capabilities, settings


def _with_usage(decode, reasoning):
    def decoded(raw):
        traces = decode(raw)
        for trace in traces.values():
            trace.calls[0].usage = (
                None if reasoning is None else SimpleNamespace(completion_tokens=2, reasoning_tokens=reasoning)
            )
        return traces

    return decoded


def _admit(rollouts, evidence, decode, **selection):
    selected = replace(
        settings(),
        policy_updates=PolicyUpdateSettings(
            PolicyUpdateSchedule("episode", 1),
            PolicyExecutionBudget(2, 100, 1000),
            objective_variant="semantic-spans",
            **selection,
        ),
    )
    return AdmittedNativePopulation.from_rollouts(
        rollouts,
        selected,
        capabilities(),
        population_id="population@3",
        native_evidence_ref="artifact:episodes",
        read_evidence=lambda reference: evidence,
        decode=decode,
        template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        selector_digest="complete-groups@1",
        applied_update_offset=3,
        attempt_offset=3,
    )


def test_admission_derives_reasoning_and_answer_spans_on_original_actions():
    rollouts, evidence, decode = source()
    admitted = _admit(
        rollouts,
        evidence,
        _with_usage(decode, 1),
        policy_selection=ActionSelection("roles", roles=("reasoning",)),
        kl_selection=ActionSelection("roles", roles=("reasoning", "answer")),
    )
    snapshot = admitted.resolved.snapshot
    spans = {span.id: span for span in snapshot.spans}
    assert sorted(spans) == [
        "trace-0/node-1/answer",
        "trace-0/node-1/reasoning",
        "trace-1/node-1/answer",
        "trace-1/node-1/reasoning",
    ]
    reasoning = snapshot.span_positions(spans["trace-0/node-1/reasoning"])
    assert [(snapshot.action(int(p)).turn_id, snapshot.action(int(p)).token_index) for p in reasoning] == [
        ("trace-0/node-1", 1)
    ]
    assert {span.projection_revision for span in spans.values()} == {"verifiers.renderer-reasoning-prefix@1"}
    expected = np.zeros(snapshot.size, dtype=bool)
    for span in spans.values():
        if span.role == "reasoning":
            expected[snapshot.span_positions(span)] = True
    assert snapshot.select_roles(("reasoning",)).tolist() == expected.tolist()
    # Reasoning policy support and full KL support are selected independently.
    assert snapshot.select_roles(("reasoning", "answer")).all()
    objective = admitted.resolved.updates[0].objective
    assert objective.policy.tolist() == expected.tolist()


def test_admission_rejects_spans_without_renderer_reasoning_accounting():
    rollouts, evidence, decode = source()
    with pytest.raises(InvalidPolicyUpdate, match="reasoning-token accounting"):
        _admit(
            rollouts,
            evidence,
            _with_usage(decode, None),
            policy_selection=ActionSelection("roles", roles=("reasoning",)),
        )


def test_role_selection_is_declared_and_unique():
    with pytest.raises(InvalidPolicyUpdate):
        ActionSelection("roles")
    with pytest.raises(InvalidPolicyUpdate):
        ActionSelection("all", roles=("reasoning",))
    with pytest.raises(InvalidPolicyUpdate):
        ActionSelection("roles", roles=("answer", "answer"))
    assert json.dumps(ActionSelection("roles", roles=("answer",)).roles) == '["answer"]'
