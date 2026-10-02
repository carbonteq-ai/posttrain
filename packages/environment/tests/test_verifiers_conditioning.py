from dataclasses import replace
from types import SimpleNamespace

import pytest
from posttrain.environment.verifiers_conditioning import (
    InvalidNativeConditioning,
    materialize_native_conditioning,
    native_conditioning_records,
    native_reasoning_partition,
)


def graph():
    # Later inference sees a new summary root, not the earlier sampled path.
    nodes = [
        SimpleNamespace(parent=None, sampled=False, message={"role": "user"}, token_ids=[1, 2], mask=[False, False]),
        SimpleNamespace(parent=0, sampled=True, message={"role": "assistant"}, token_ids=[3, 4], mask=[False, True]),
        SimpleNamespace(parent=None, sampled=False, message={"role": "system"}, token_ids=[10], mask=[False]),
        SimpleNamespace(parent=2, sampled=False, message={"role": "tool"}, token_ids=[11, 12], mask=[False, False]),
        SimpleNamespace(
            parent=3, sampled=True, message={"role": "assistant"}, token_ids=[13, 14, 15], mask=[False, True, True]
        ),
    ]
    return SimpleNamespace(id="trace-original", nodes=nodes, calls=[SimpleNamespace(node=1), SimpleNamespace(node=4)])


def test_rolling_context_uses_original_physical_path() -> None:
    trace = graph()
    records = native_conditioning_records(trace, sampled_node_indices=(1, 4), context_contract="causal-text@1")
    assert records[0].prefix_node_indices == (0,)
    assert records[1].prefix_node_indices == (2, 3)
    assert records[1].sampled_token_indices == (1, 2)
    assert [record.context_tokens for record in records] == [4, 6]
    assert records[0].input_digest != records[1].input_digest
    # Editing the excluded earlier transcript does not change later conditioning.
    trace.nodes[0].token_ids[0] = 99
    later = native_conditioning_records(trace, sampled_node_indices=(4,), context_contract="causal-text@1")[0]
    assert later == records[1]
    assert replace(later, trace_id="another-trace").input_digest == later.input_digest


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda trace: setattr(trace.nodes[3], "parent", 4), "cycle"),
        (lambda trace: setattr(trace.nodes[4], "mask", [False, 1, True]), "boolean"),
        (lambda trace: setattr(trace.nodes[4], "sampled", False), "sampled assistant"),
        (lambda trace: trace.calls.append(SimpleNamespace(node=4)), "sampled assistant"),
        (lambda trace: setattr(trace.nodes[3], "multi_modal_data", {}), "multimodal"),
    ],
)
def test_unproven_native_conditioning_is_rejected(mutation, message) -> None:
    trace = graph()
    mutation(trace)
    with pytest.raises(InvalidNativeConditioning, match=message):
        native_conditioning_records(trace, sampled_node_indices=(4,), context_contract="causal-text@1")


def test_custom_attention_requires_its_own_qualified_contract() -> None:
    with pytest.raises(InvalidNativeConditioning, match="unqualified"):
        native_conditioning_records(graph(), sampled_node_indices=(1,), context_contract="custom-attention@1")


def test_materialized_coordinates_preserve_actual_causal_positions() -> None:
    trace = graph()
    record = native_conditioning_records(trace, sampled_node_indices=(4,), context_contract="causal-text@1")[0]
    inputs = materialize_native_conditioning(trace, record)
    assert inputs.token_ids == (10, 11, 12, 13, 14, 15)
    assert inputs.action_positions == ((1, 4), (2, 5))
    trace.nodes[0].token_ids[0] = 99  # Outside the actual view.
    assert materialize_native_conditioning(trace, record) == inputs
    trace.nodes[3].token_ids[0] = 99
    with pytest.raises(InvalidNativeConditioning, match="differs from the frozen"):
        materialize_native_conditioning(trace, record)


def test_materialization_rejects_forged_action_coordinate_even_with_same_tokens() -> None:
    trace = graph()
    record = native_conditioning_records(trace, sampled_node_indices=(4,), context_contract="causal-text@1")[0]
    with pytest.raises(InvalidNativeConditioning, match="differs from the frozen"):
        materialize_native_conditioning(trace, replace(record, sampled_token_indices=(2,)))


@pytest.mark.parametrize(("reasoning", "expected"), [(0, ((), (1, 2))), (1, ((1,), (2,))), (2, ((1, 2), ()))])
def test_reasoning_partition_splits_sampled_actions_at_renderer_count(reasoning, expected) -> None:
    trace = graph()
    trace.calls[1].usage = SimpleNamespace(completion_tokens=2, reasoning_tokens=reasoning)
    record = native_conditioning_records(trace, sampled_node_indices=(4,), context_contract="causal-text@1")[0]
    assert native_reasoning_partition(trace, record) == expected


@pytest.mark.parametrize(
    "usage",
    [
        None,
        SimpleNamespace(completion_tokens=2, reasoning_tokens=None),
        SimpleNamespace(completion_tokens=3, reasoning_tokens=1),
        SimpleNamespace(completion_tokens=2, reasoning_tokens=3),
    ],
)
def test_reasoning_partition_rejects_missing_or_inconsistent_accounting(usage) -> None:
    trace = graph()
    trace.calls[1].usage = usage
    record = native_conditioning_records(trace, sampled_node_indices=(4,), context_contract="causal-text@1")[0]
    with pytest.raises(InvalidNativeConditioning):
        native_reasoning_partition(trace, record)
    trace.calls[1].usage = SimpleNamespace(completion_tokens=2, reasoning_tokens=1)
    with pytest.raises(InvalidNativeConditioning, match="retained conditioning record"):
        native_reasoning_partition(trace, replace(record, sampled_token_indices=(2,)))
