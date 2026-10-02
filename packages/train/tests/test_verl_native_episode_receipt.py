"""Collection receipts preserve contexts and authenticate retained evidence."""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import LocalArtifactRef, ProducedArtifact
from posttrain.train.backends.verl.policy_rollouts import NativeEpisodeReceipt, retain_episode_receipt
from posttrain.train.online_rl import BehaviorPolicySpan
from posttrain.train.update_records import InvalidPolicyUpdate

from .test_update_admission import AdmissionIdentity, source


class ReceiptIdentity(AdmissionIdentity):
    destination: Path


def receipt_source(tmp_path):
    rollouts, evidence, _ = source()
    rollout = rollouts[0]
    path = tmp_path / "native.jsonl"
    path.write_bytes(evidence)
    artifact = ProducedArtifact(
        "native/episode",
        "evaluation-traces",
        LocalArtifactRef(path, hashlib.sha256(evidence).hexdigest()),
        metadata={
            "format": "verifiers-native-traces",
            "replay_authority": True,
            "trace_ids": [rollout.trace.external_id],
        },
    )
    bridge = SimpleNamespace(
        policy_update_context_contract="causal-text@1", retain_population=lambda selected: artifact
    )
    return bridge, rollout, artifact


def test_episode_receipt_json_preserves_native_coordinate_and_credit_evidence(tmp_path):
    bridge, rollout, artifact = receipt_source(tmp_path)
    encoded = retain_episode_receipt(bridge, rollout, sampler_step=3)
    receipt = NativeEpisodeReceipt.model_validate_json(encoded)
    assert receipt.rollout == rollout
    assert receipt.artifact == artifact
    assert isinstance(artifact.reference, LocalArtifactRef)
    assert receipt.verify_artifact() == artifact.reference.path.read_bytes()
    artifact.reference.path.write_bytes(b"modified\n")
    with pytest.raises(InvalidPolicyUpdate, match="differs from its digest"):
        receipt.verify_artifact()


def test_episode_receipt_rejects_flattened_or_mixed_version_collection(tmp_path):
    bridge, rollout, _ = receipt_source(tmp_path)
    mixed = replace(rollout, behavior_policy=BehaviorPolicySpan(2, 3))
    with pytest.raises(InvalidPolicyUpdate, match="synchronous policy"):
        retain_episode_receipt(bridge, mixed, sampler_step=3)
    flattened = replace(rollout, conditioning_records=(), selected_branch_id=None, conditioning_completion_indices=())
    with pytest.raises(InvalidPolicyUpdate, match="original contexts"):
        retain_episode_receipt(bridge, flattened, sampler_step=3)


def test_receipts_resolve_complete_groups_with_shared_native_admission(tmp_path, monkeypatch):
    import json

    from posttrain.train.backends.verl.policy_rollouts import admit_episode_receipts
    from posttrain.train.update_records import PolicyVersions

    from .test_update_resolution import capabilities, settings

    rollouts, evidence, decode = source()
    encoded = []
    for rollout, record in zip(rollouts, json.loads(evidence), strict=True):
        path = tmp_path / f"{rollout.trace.external_id}.jsonl"
        path.write_text(json.dumps(record) + "\n")
        artifact = ProducedArtifact(
            "native/episode",
            "evaluation-traces",
            LocalArtifactRef(path, hashlib.sha256(path.read_bytes()).hexdigest()),
            metadata={
                "format": "verifiers-native-traces",
                "replay_authority": True,
                "trace_ids": [rollout.trace.external_id],
            },
        )
        bridge = SimpleNamespace(
            policy_update_context_contract="causal-text@1",
            retain_population=lambda selected, artifact=artifact: artifact,
        )
        encoded.append(retain_episode_receipt(bridge, rollout, sampler_step=3))
    monkeypatch.setattr(
        "posttrain.train.integrations.verifiers_population_artifact.decode_native_population",
        lambda raw, **kwargs: decode(json.dumps([json.loads(line) for line in raw.splitlines()]).encode()),
    )
    kwargs = ReceiptIdentity(
        destination=tmp_path / "population",
        population_id="complete@3",
        template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        selector_digest="selection@1",
        applied_update_offset=3,
        attempt_offset=4,
    )
    admitted, artifact = admit_episode_receipts(tuple(encoded), settings(), capabilities(), **kwargs)
    assert admitted.resolved.snapshot.native_evidence_ref == artifact.name
    assert admitted.applied_update_offset == 3 and admitted.attempt_offset == 4
    assert len(admitted.resolved.credit.values) == 4
    assert len(admitted.resolved.snapshot.conditioning) == 2
    with pytest.raises(InvalidPolicyUpdate, match="complete prompt groups"):
        admit_episode_receipts(tuple(encoded[:1]), settings(), capabilities(), **kwargs)
    with pytest.raises(InvalidPolicyUpdate, match="duplicates selected"):
        admit_episode_receipts((encoded[0], encoded[0]), settings(), capabilities(), **kwargs)


def test_agent_loop_retains_episode_receipt_before_native_row_return(tmp_path, monkeypatch):
    import asyncio
    import importlib
    import sys
    from types import ModuleType

    module_name = "posttrain.train.backends.verl.agent_loop"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    for package in ("verl", "verl.experimental", "verl.experimental.agent_loop"):
        module = ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    native = ModuleType("verl.experimental.agent_loop.agent_loop")
    monkeypatch.setattr(native, "AgentLoopBase", object, raising=False)
    monkeypatch.setattr(native, "AgentLoopMetrics", SimpleNamespace, raising=False)
    monkeypatch.setattr(native, "AgentLoopOutput", SimpleNamespace, raising=False)
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop.agent_loop", native)
    module = importlib.import_module(module_name)
    bridge, rollout, artifact = receipt_source(tmp_path)
    observed = []

    async def collect(batch, generator):
        observed.append(batch)
        return (rollout,)

    bridge.run = collect
    bridge.run_id = "run"
    loop = object.__new__(module.PosttrainVerifiersAgentLoop)
    loop._bridge = bridge
    loop._generator = SimpleNamespace(
        begin_episode=lambda: None, set_sampling_overrides=lambda value: None, behavior_policy=BehaviorPolicySpan(3, 3)
    )
    loop._structured_algorithm = None
    loop._retain_policy_update_evidence = True
    loop._trainer_mode = "sync"
    loop.rollout_config = SimpleNamespace(prompt_length=100, response_length=100)
    loop._max_completion_tokens = 100
    loop._overlong_buffer_tokens = loop._overlong_penalty_factor = loop._truncation_penalty = None
    loop._mask_truncated_completions = loop._emit_sampo_metadata = False
    output = asyncio.run(
        loop.run({}, example_id="task", global_steps=3, model_id="actor", uid="prompt", session_id="episode")
    )
    assert observed[0].prompt_group_ids == ("run/3/prompt",)
    assert observed[0].rollout_ids == ("run/3/prompt/episode",)
    receipt = NativeEpisodeReceipt.model_validate_json(output.extra_fields["posttrain_native_episode_receipt"])
    assert receipt.rollout.conditioning_records == rollout.conditioning_records
    assert receipt.artifact == artifact
    assert output.response_ids == list(rollout.completion_ids)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
