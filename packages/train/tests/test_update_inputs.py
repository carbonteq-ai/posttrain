"""Native input admission authenticates bytes and every frozen context."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from posttrain.environment.verifiers_conditioning import native_conditioning_records
from posttrain.train.backends.policy_update_inputs import NativePopulationInputs
from posttrain.train.update_records import (
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationSnapshot,
)


def fixture():
    evidence = json.dumps(
        {
            "id": "original",
            "calls": [1],
            "nodes": [
                {
                    "parent": None,
                    "sampled": False,
                    "message": {"role": "system"},
                    "token_ids": [1, 2],
                    "mask": [False, False],
                },
                {
                    "parent": 0,
                    "sampled": True,
                    "message": {"role": "assistant"},
                    "token_ids": [3, 4, 5],
                    "mask": [False, True, True],
                    "logprobs": [-0.2, -0.4],
                },
            ],
        }
    ).encode()

    def decode(raw):
        data = json.loads(raw)
        trace = SimpleNamespace(
            id=data["id"],
            calls=[SimpleNamespace(node=index) for index in data["calls"]],
            nodes=[SimpleNamespace(**node) for node in data["nodes"]],
        )
        return {trace.id: trace}

    trace = decode(evidence)["original"]
    record = native_conditioning_records(trace, sampled_node_indices=(1,), context_contract="causal-text@1")[0]
    coordinates = json.dumps({"trace_id": trace.id, "prefix_nodes": [0], "node_index": 1})
    view = ConditioningView(
        "turn-1",
        "native:original",
        coordinates,
        "causal-text@1/attention",
        "causal-text@1/positions",
        "template@1",
        record.input_digest,
        5,
        "episode",
        "branch",
        (1, 2),
    )
    snapshot = PopulationSnapshot(
        "population",
        view.native_ref,
        hashlib.sha256(evidence).hexdigest(),
        (view,),
        (),
        (),
        PolicyVersions("sampler", "old", "current", None),
        "all",
    )
    return snapshot, evidence, decode


def test_reader_exposes_original_tokens_and_sampled_coordinates():
    snapshot, evidence, decode = fixture()
    reader = NativePopulationInputs.from_evidence(snapshot, evidence, decode)
    inputs = reader(snapshot.conditioning[0])
    assert inputs.token_ids == (1, 2, 3, 4, 5)
    assert inputs.action_positions == ((1, 3), (2, 4))


def test_sampler_scores_match_original_coordinates_and_reject_mutation():
    snapshot, evidence, decode = fixture()
    reader = NativePopulationInputs.from_evidence(snapshot, evidence, decode)
    scores = reader.sampling_log_scores(snapshot)
    # One score per position, in canonical (turn, native token index) order.
    assert scores.tolist() == [-0.2, -0.4]
    assert [snapshot.action(position).token_index for position in range(snapshot.size)] == [1, 2]
    assert not scores.flags.writeable
    # Rewritten by design: scores are read once from authenticated evidence at
    # admission, so mutating the decoded graph afterwards cannot change them.
    reader._traces["original"].nodes[1].logprobs[0] = -0.8
    assert reader.sampling_log_scores(snapshot).tolist() == [-0.2, -0.4]
    # A population bound to other evidence or another view is still rejected.
    with pytest.raises(InvalidPolicyUpdate, match="different native evidence"):
        reader.sampling_log_scores(replace(snapshot, native_evidence_digest="other"))
    changed = replace(snapshot, conditioning=(replace(snapshot.conditioning[0], digest="changed"),))
    with pytest.raises(InvalidPolicyUpdate, match="different native evidence"):
        reader.sampling_log_scores(changed)


@pytest.mark.parametrize("values", [None, [], [-0.2], [-0.2, float("nan")], [True, -0.4]])
def test_missing_or_misaligned_sampler_scores_reject(values):
    snapshot, evidence, decode = fixture()

    def decoder(raw):
        result = decode(raw)
        result["original"].nodes[1].logprobs = values
        return result

    reader = NativePopulationInputs.from_evidence(snapshot, evidence, decoder)
    with pytest.raises(InvalidPolicyUpdate, match="sampled log scores"):
        reader.sampling_log_scores(snapshot)


def test_hash_mismatch_rejected_before_native_decode():
    snapshot, evidence, _ = fixture()

    def forbidden(raw):
        pytest.fail("unauthenticated bytes reached native decoder")

    with pytest.raises(InvalidPolicyUpdate, match="evidence bytes"):
        NativePopulationInputs.from_evidence(snapshot, evidence + b" ", forbidden)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"native_ref": "native:other"}, "different population"),
        ({"token_ids_ref": '{"trace_id":"original","node_index":true,"prefix_nodes":[0]}'}, "exact graph"),
        ({"token_ids_ref": '{"trace_id":"original","node_index":1,"prefix_nodes":[]}'}, "frozen context"),
        ({"digest": "different"}, "frozen context"),
        ({"context_tokens": 6}, "frozen context"),
        ({"positions_ref": "custom-positions@1"}, "qualified causal"),
    ],
)
def test_reader_preflights_context_identity_and_coordinates(changes, message):
    snapshot, evidence, decode = fixture()
    # A view naming other evidence is now rejected when the snapshot is built.
    with pytest.raises(InvalidPolicyUpdate, match=message):
        snapshot = replace(snapshot, conditioning=(replace(snapshot.conditioning[0], **changes),))
        NativePopulationInputs.from_evidence(snapshot, evidence, decode)


def test_reader_rejects_context_positions_credited_as_sampled_actions():
    snapshot, evidence, decode = fixture()
    snapshot = replace(snapshot, conditioning=(replace(snapshot.conditioning[0], sampled=(0, 2)),))
    with pytest.raises(InvalidPolicyUpdate, match="sampled action coordinates"):
        NativePopulationInputs.from_evidence(snapshot, evidence, decode)


def test_reader_rejects_action_evidence_reference_substitution():
    # Actions no longer carry their own evidence reference; every view must
    # name the snapshot's evidence, so substitution fails at construction.
    snapshot, _, _ = fixture()
    with pytest.raises(InvalidPolicyUpdate, match="references different"):
        replace(snapshot, native_evidence_ref="native:other")


def test_reader_revalidates_mutable_graph_and_rejects_foreign_views():
    snapshot, evidence, decode = fixture()
    traces = decode(evidence)
    reader = NativePopulationInputs.from_evidence(snapshot, evidence, lambda _: traces)
    view = snapshot.conditioning[0]
    with pytest.raises(InvalidPolicyUpdate, match="outside its frozen population"):
        reader(replace(view, digest="changed"))
    # Rewritten by design: inputs are built once from the verified graph at
    # admission, so a later mutation of the decoded graph cannot reach scoring.
    traces["original"].nodes[0].token_ids[0] = 9
    inputs = reader(view)
    assert inputs.token_ids == (1, 2, 3, 4, 5)
    assert inputs is reader(view)
