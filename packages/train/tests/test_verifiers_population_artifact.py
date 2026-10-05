"""Mid-run replay artifacts preserve physical native envelopes and durability."""

import hashlib
import json
import sys
import threading
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import LocalArtifactRef
from posttrain.train.integrations.verifiers_population_artifact import (
    decode_native_population,
    retain_native_population,
)
from posttrain.train.online_rl import EnvironmentRollout
from posttrain.train.update_records import InvalidPolicyUpdate


@pytest.mark.parametrize("contract", [None, "causal-text@1"])
def test_portable_bridge_preserves_conditioning_contract(tmp_path, monkeypatch, contract):
    from posttrain.train.integrations.verifiers import (
        VerifiersEnvironmentRolloutBridge,
        load_verifiers_bridge_snapshot,
    )
    from posttrain.train.online_rl import PolicySampling

    # Environment activation is outside this serialization regression.
    monkeypatch.setattr(VerifiersEnvironmentRolloutBridge, "__post_init__", lambda self: None)
    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="tasks",
        revision="test@1",
        tasks={},
        environment_factory=dict,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="environment",
        run_id="run",
        sampling=PolicySampling(8),
        policy_update_context_contract=contract,
    )
    path = tmp_path / "bridge.pkl"
    bridge.write_portable_snapshot(path)
    restored = load_verifiers_bridge_snapshot(path)
    assert restored.policy_update_context_contract == contract


def test_bridge_retains_native_authority_instead_of_derived_observation(tmp_path):
    from posttrain.train.integrations.verifiers import VerifiersEnvironmentRolloutBridge

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(b'{"id":"a","derived":true}\n')
    native = b'{"id":"episode","traces":[{"id":"a","nodes":[1]}]}\n'
    trace_path.with_name("episodes.jsonl").write_bytes(native)
    bridge = SimpleNamespace(trace_path=trace_path, _write_lock=threading.Lock())
    rollout = SimpleNamespace(trace=SimpleNamespace(external_id="a"))
    artifact = VerifiersEnvironmentRolloutBridge.retain_population(
        cast(VerifiersEnvironmentRolloutBridge, bridge), (cast(EnvironmentRollout, rollout),)
    )
    assert isinstance(artifact.reference, LocalArtifactRef)
    assert artifact.reference.path.read_bytes() == native
    assert trace_path.read_bytes() == b'{"id":"a","derived":true}\n'


def test_native_episode_snapshot_preserves_bytes_and_survives_source_growth(tmp_path):
    source = tmp_path / "episodes.jsonl"
    unrelated = b'{"id":"failed","traces":[{"id":"failed-trace"}]}\n'
    selected = b'{"id": "episode", "traces": [{"id":"policy","nodes":[1,2]}, {"id":"judge"}]}\n'
    source.write_bytes(unrelated + selected)
    artifact = retain_native_population(source, tmp_path / "populations", ("policy",), episodes=True)
    assert isinstance(artifact.reference, LocalArtifactRef)
    assert artifact.reference.path.read_bytes() == selected
    assert artifact.reference.digest == hashlib.sha256(selected).hexdigest()
    assert artifact.metadata["format"] == "verifiers-native-episodes"
    source.write_bytes(source.read_bytes() + unrelated)
    repeated = retain_native_population(source, tmp_path / "populations", ("policy",), episodes=True)
    assert repeated == artifact
    assert artifact.reference.path.read_bytes() == selected
    assert not tuple((tmp_path / "populations").glob(".population-*"))


@pytest.mark.parametrize(
    "records,ids,message",
    [
        ([{"id": "other"}], ("missing",), "lacks"),
        ([{"id": "a"}, {"id": "a"}], ("a",), "duplicate retained"),
        ([{"id": "a"}], ("a", "a"), "unique"),
    ],
)
def test_invalid_membership_creates_no_artifact(tmp_path, records, ids, message):
    source = tmp_path / "traces.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(InvalidPolicyUpdate, match=message):
        retain_native_population(source, tmp_path / "populations", ids, episodes=False)
    assert not (tmp_path / "populations").exists()


def test_corrupt_existing_artifact_is_not_overwritten(tmp_path):
    source = tmp_path / "traces.jsonl"
    source.write_bytes(b'{"id":"a"}\n')
    artifact = retain_native_population(source, tmp_path / "populations", ("a",), episodes=False)
    assert isinstance(artifact.reference, LocalArtifactRef)
    artifact.reference.path.write_bytes(b"corrupt")
    with pytest.raises(InvalidPolicyUpdate, match="modified"):
        retain_native_population(source, tmp_path / "populations", ("a",), episodes=False)
    assert artifact.reference.path.read_bytes() == b"corrupt"
    assert not tuple((tmp_path / "populations").glob(".population-*"))


def test_incomplete_native_record_cannot_be_sealed(tmp_path):
    source = tmp_path / "traces.jsonl"
    source.write_bytes(b'{"id":"a"}')
    with pytest.raises(InvalidPolicyUpdate, match="incomplete"):
        retain_native_population(source, tmp_path / "populations", ("a",), episodes=False)


@pytest.mark.parametrize(
    "payload,message",
    [
        (b'{"traces":[{"nodes":[]}]}\n', "original identity"),
        (b'{"traces":[{"id":"a"},{"id":"a"}]}\n', "duplicate"),
        (b'{"traces":[]}\n', "lacks"),
        (b'{"traces":[{"id":"a"}]}', "complete JSONL"),
    ],
)
def test_decoder_rejects_invalid_identity_before_native_defaults(monkeypatch, payload, message):
    class NativeSchema:
        @staticmethod
        def model_validate(record):
            pytest.fail("invalid artifact reached native schema defaults")

    monkeypatch.setitem(sys.modules, "verifiers.v1.episode", SimpleNamespace(Episode=NativeSchema))
    monkeypatch.setitem(sys.modules, "verifiers.v1.trace", SimpleNamespace(Trace=NativeSchema))
    with pytest.raises(InvalidPolicyUpdate, match=message):
        decode_native_population(payload, format="verifiers-native-episodes")


def _episode_line(index):
    return (json.dumps({"traces": [{"id": f"t{index}", "payload": "x" * (index + 3)}]}, sort_keys=True) + "\n").encode()


def test_indexed_population_reads_only_its_records_and_matches_a_full_scan(tmp_path):
    source = tmp_path / "episodes.jsonl"
    spans, offset = {}, 0
    with source.open("wb") as stream:
        for index in range(8):
            line = _episode_line(index)
            stream.write(line)
            spans[f"t{index}"] = (offset, len(line))
            offset += len(line)
    wanted = ("t6", "t2", "t5")
    scanned = retain_native_population(source, tmp_path / "scan", wanted, episodes=True)
    indexed = retain_native_population(source, tmp_path / "index", wanted, episodes=True, spans=spans)
    assert indexed.reference.digest == scanned.reference.digest
    assert indexed.reference.path.read_bytes() == scanned.reference.path.read_bytes()
    # A span index that does not cover every trace falls back to the full scan.
    partial = retain_native_population(source, tmp_path / "partial", wanted, episodes=True, spans={"t2": spans["t2"]})
    assert partial.reference.digest == scanned.reference.digest


def test_stale_or_truncated_spans_fall_back_to_the_authoritative_scan(tmp_path):
    source = tmp_path / "episodes.jsonl"
    first, second = _episode_line(0), _episode_line(1)
    source.write_bytes(first + second)
    scanned = retain_native_population(source, tmp_path / "scan", ("t1",), episodes=True)
    for name, span in (("stale", (0, len(first))), ("truncated", (len(first), len(second) + 50))):
        retained = retain_native_population(source, tmp_path / name, ("t1",), episodes=True, spans={"t1": span})
        assert retained.reference.path.read_bytes() == scanned.reference.path.read_bytes() == second
    # A trace that is genuinely absent still fails after the fallback.
    with pytest.raises(InvalidPolicyUpdate):
        retain_native_population(source, tmp_path / "missing", ("t9",), episodes=True, spans={"t9": (0, len(first))})


def test_bridge_records_episode_spans_as_it_appends(tmp_path, monkeypatch):
    from posttrain.train.integrations import verifiers as bridge_module
    from posttrain.train.integrations.verifiers import VerifiersEnvironmentRolloutBridge
    from posttrain.train.online_rl import PolicySampling

    monkeypatch.setattr(VerifiersEnvironmentRolloutBridge, "__post_init__", lambda self: None)
    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="tasks",
        revision="test@1",
        tasks={},
        environment_factory=dict,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="environment",
        run_id="run",
        sampling=PolicySampling(8),
    )
    monkeypatch.setattr(bridge_module, "_native_record", lambda episode: episode)
    monkeypatch.setattr(
        "posttrain.train.integrations.verifiers_assessment_artifacts.retain_episode_artifacts",
        lambda *args, **kwargs: None,
    )
    for index in range(3):
        bridge._preserve_episode({"traces": [{"id": f"t{index}"}, {"id": f"s{index}"}]})
    raw = (tmp_path / "episodes.jsonl").read_bytes()
    for index in range(3):
        offset, length = bridge._episode_spans[f"t{index}"]
        assert bridge._episode_spans[f"s{index}"] == (offset, length)
        assert json.loads(raw[offset : offset + length])["traces"][0]["id"] == f"t{index}"
