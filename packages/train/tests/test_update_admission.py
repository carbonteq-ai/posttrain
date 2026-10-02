"""Host admission reads durable evidence before handing work to either backend."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TypedDict

import pytest
from posttrain.common import TraceObservation
from posttrain.environment.verifiers_conditioning import native_conditioning_records
from posttrain.train.backends.policy_update_admission import AdmittedNativePopulation
from posttrain.train.backends.trl.policy_updates import ResolvedTRLPopulation
from posttrain.train.online_rl import AgenticTurn, BehaviorPolicySpan, EnvironmentRollout
from posttrain.train.update_records import InvalidPolicyUpdate, PolicyVersions

from .test_update_resolution import capabilities, settings


class AdmissionIdentity(TypedDict):
    population_id: str
    template_revision: str
    versions: PolicyVersions
    sampler_step: int
    selector_digest: str
    applied_update_offset: int
    attempt_offset: int


def test_retained_artifact_handoff_binds_logical_name_and_checks_digest(tmp_path, monkeypatch):
    from posttrain.common import LocalArtifactRef, ProducedArtifact

    rollouts, evidence, decode = source()
    path = tmp_path / "population.jsonl"
    path.write_bytes(evidence)
    artifact = ProducedArtifact(
        "training/rollouts/populations/test",
        "evaluation-traces",
        LocalArtifactRef(path.resolve(), hashlib.sha256(evidence).hexdigest()),
        metadata={"format": "verifiers-native-traces", "replay_authority": True},
    )
    monkeypatch.setattr(
        "posttrain.train.integrations.verifiers_population_artifact.decode_native_population",
        lambda raw, **kwargs: decode(raw),
    )
    args = AdmissionIdentity(
        population_id="population@3",
        template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        selector_digest="complete-groups@1",
        applied_update_offset=3,
        attempt_offset=3,
    )
    admitted = AdmittedNativePopulation.from_retained_artifact(artifact, rollouts, settings(), capabilities(), **args)
    assert admitted.resolved.snapshot.native_evidence_ref == artifact.name
    assert all(action.native_ref == artifact.name for action in admitted.resolved.snapshot.actions)
    path.write_bytes(evidence + b" ")
    with pytest.raises(InvalidPolicyUpdate, match="declared digest"):
        AdmittedNativePopulation.from_retained_artifact(artifact, rollouts, settings(), capabilities(), **args)


def source():
    records = [
        {
            "id": f"trace-{index}",
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
                    "logprobs": [-1.0, -1.0],
                },
            ],
        }
        for index in range(2)
    ]
    evidence = json.dumps(records).encode()

    def decode(raw):
        return {
            record["id"]: SimpleNamespace(
                id=record["id"],
                calls=[SimpleNamespace(node=index) for index in record["calls"]],
                nodes=[SimpleNamespace(**node) for node in record["nodes"]],
            )
            for record in json.loads(raw)
        }

    traces = decode(evidence)
    rollouts = tuple(
        EnvironmentRollout(
            "task",
            (1, 2, 3),
            (4, 5),
            (-1.0, -1.0),
            (True, True),
            float(index),
            False,
            TraceObservation(
                "verifiers",
                f"trace-{index}",
                {
                    "info": {
                        "posttrain_episode_id": f"episode-{index}",
                        "posttrain_prompt_group_id": "group",
                        "posttrain_rollout_id": f"rollout-{index}",
                    }
                },
            ),
            turns=(AgenticTurn(0, 2, "shared"),),
            behavior_policy=BehaviorPolicySpan(3, 3),
            conditioning_records=native_conditioning_records(
                traces[f"trace-{index}"], sampled_node_indices=(1,), context_contract="causal-text@1"
            ),
            selected_branch_id="1",
            conditioning_completion_indices=((0, 1),),
        )
        for index in range(2)
    )
    return rollouts, evidence, decode


def admit(rollouts, evidence, decode, **kwargs):
    kwargs.setdefault("applied_update_offset", 3)
    kwargs.setdefault("attempt_offset", 3)
    return AdmittedNativePopulation.from_rollouts(
        rollouts,
        settings(),
        capabilities(),
        population_id="population@3",
        native_evidence_ref="artifact:episodes",
        read_evidence=lambda reference: evidence,
        decode=decode,
        template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        selector_digest="complete-groups@1",
        **kwargs,
    )


def test_collection_binds_retained_bytes_and_preserves_original_context_and_credit():
    rollouts, evidence, decode = source()
    admitted = admit(rollouts, evidence, decode, applied_update_offset=3, attempt_offset=5, max_overflow_retries=2)
    assert admitted.resolved.snapshot.native_evidence_digest == hashlib.sha256(evidence).hexdigest()
    assert len(admitted.resolved.updates) == 2
    assert [value.advantage for value in admitted.resolved.credit.values] == [-1, -1, 1, 1]
    for view in admitted.resolved.snapshot.conditioning:
        inputs = admitted.read_input(view)
        assert inputs.token_ids == (1, 2, 3, 4, 5)
        assert inputs.action_positions == ((1, 3), (2, 4))
    native = ResolvedTRLPopulation.from_admitted(
        admitted, score_temperature=0.8, score_contract="test@1", sampler_correction=None
    )
    assert native.credit is admitted.resolved.credit
    assert native.read_input is admitted.read_input
    assert (native.applied_update_offset, native.attempt_offset, native.max_overflow_retries) == (3, 5, 2)
    assert native.old is None and native.next_update == 0


def test_incomplete_groups_and_changed_context_rejected_before_handoff():
    rollouts, evidence, decode = source()
    with pytest.raises(InvalidPolicyUpdate, match="complete prompt groups"):
        admit(rollouts[:1], evidence, decode)
    changed = json.loads(evidence)
    changed[0]["nodes"][0]["token_ids"][0] = 9
    with pytest.raises(InvalidPolicyUpdate, match="frozen context"):
        admit(rollouts, json.dumps(changed).encode(), decode)
    with pytest.raises(InvalidPolicyUpdate, match="sampled tokens differ"):
        admit((replace(rollouts[0], completion_ids=(5, 4)), rollouts[1]), evidence, decode)


@pytest.mark.parametrize(
    "changes",
    [
        {"applied_update_offset": True},
        {"attempt_offset": -1},
        {"applied_update_offset": 2, "attempt_offset": 1},
        {"max_overflow_retries": True},
    ],
)
def test_invalid_offsets_do_not_read_artifacts(changes):
    rollouts, evidence, decode = source()
    with pytest.raises(InvalidPolicyUpdate, match="offsets"):
        admit(rollouts, evidence, lambda _: pytest.fail("invalid counters reached decoder"), **changes)


def test_stale_sampling_cannot_be_admitted_as_a_fresh_population():
    rollouts, evidence, decode = source()
    with pytest.raises(InvalidPolicyUpdate, match="sampling at the current applied boundary"):
        admit(rollouts, evidence, decode, applied_update_offset=4, attempt_offset=4)


def test_reader_failures_and_foreign_inputs_cannot_be_admitted():
    rollouts, evidence, decode = source()
    with pytest.raises(InvalidPolicyUpdate, match="original retained bytes"):
        admit(rollouts, json.loads(evidence), decode)
    admitted = admit(rollouts, evidence, decode)
    with pytest.raises(InvalidPolicyUpdate, match="outside its frozen population"):
        # Even direct wrapper construction must reject a reader from another view.
        replace(admitted, read_input=replace(admitted.read_input, _views={}))


def test_both_backend_factories_consume_the_same_admitted_contract():
    pytest.importorskip("torch")
    from posttrain.train.backends.verl.policy_updates import ResolvedVeRLPopulation

    rollouts, evidence, decode = source()
    selected = settings()
    from posttrain.train.update_plan import PolicyExecutionBudget
    from posttrain.train.update_resolution import resolve_policy_population

    assert selected.policy_updates is not None
    selected = replace(
        selected, policy_updates=replace(selected.policy_updates, execution=PolicyExecutionBudget(1, 100, 1000))
    )
    admitted = admit(rollouts, evidence, decode)
    resolved = resolve_policy_population(admitted.resolved.snapshot, admitted.resolved.credit, selected, capabilities())
    admitted = replace(admitted, resolved=resolved)
    trl = ResolvedTRLPopulation.from_admitted(
        admitted, score_temperature=0.8, score_contract="test@1", sampler_correction=None
    )
    verl = ResolvedVeRLPopulation.from_admitted(
        admitted, score_temperature=0.8, score_contract="test@1", sampler_correction=None
    )
    assert trl.updates == verl.updates and trl.credit is verl.credit
    assert trl.read_input is verl.read_input


def test_recovery_verifies_native_seal_and_original_inputs_without_estimating_credit(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from posttrain.train.backends.policy_update_recovery import population_recovery_identity, save_population_recovery

    from .test_update_scoring import CausalModel

    rollouts, evidence, decode = source()
    admitted = admit(rollouts, evidence, decode, applied_update_offset=3, attempt_offset=5)
    original = ResolvedTRLPopulation.from_admitted(
        admitted, score_temperature=0.8, score_contract="test@1", sampler_correction=None
    )
    original.loss(CausalModel(), 0, torch.device("cpu"))
    optimizer = SimpleNamespace(step_was_skipped=False)
    original.before_step(optimizer)
    original.complete_step(optimizer)
    (tmp_path / "native.bin").write_bytes(b"native checkpoint fixture")
    save_population_recovery(
        original,
        tmp_path,
        runtime_identity="admission-test@1",
        world_size=1,
        native_applied_updates=4,
        native_components=("native.bin",),
    )
    identity = population_recovery_identity(original, runtime_identity="admission-test@1", world_size=1)

    from posttrain.train.backends.policy_update_recovery import NATIVE_EVIDENCE_FILENAME

    assert (tmp_path / NATIVE_EVIDENCE_FILENAME).read_bytes() == evidence
    import copy
    import shutil

    relocated = tmp_path.parent / f"{tmp_path.name}-relocated"
    shutil.copytree(tmp_path, relocated)
    standalone = AdmittedNativePopulation.from_checkpoint(relocated, identity, sampler_correction=None, decode=decode)
    assert standalone.resolved == admitted.resolved
    assert standalone.read_input.retained_evidence == evidence
    (relocated / NATIVE_EVIDENCE_FILENAME).write_bytes(evidence + b" ")
    with pytest.raises(InvalidPolicyUpdate, match="changed|digest|differs|component"):
        AdmittedNativePopulation.from_checkpoint(
            relocated,
            identity,
            sampler_correction=None,
            decode=lambda _: pytest.fail("modified sealed evidence reached native decoding"),
        )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "native.bin").write_bytes(b"native checkpoint fixture")
    legacy_population = copy.copy(original)
    legacy_population.read_input = lambda view: admitted.read_input(view)
    save_population_recovery(
        legacy_population,
        legacy,
        runtime_identity="admission-test@1",
        world_size=1,
        native_applied_updates=4,
        native_components=("native.bin",),
    )
    with pytest.raises(InvalidPolicyUpdate, match="supply a retained artifact resolver"):
        AdmittedNativePopulation.from_checkpoint(
            legacy,
            identity,
            sampler_correction=None,
            decode=lambda _: pytest.fail("legacy checkpoint decoded without its artifact resolver"),
        )
    assert (
        AdmittedNativePopulation.from_checkpoint(
            legacy,
            identity,
            sampler_correction=None,
            read_evidence=lambda _: evidence,
            decode=decode,
        ).resolved
        == admitted.resolved
    )

    def forbidden(*args, **kwargs):
        pytest.fail("recovery must not resolve a fresh population or estimate new credit")

    monkeypatch.setattr("posttrain.train.backends.policy_update_admission.resolve_rollout_population", forbidden)
    reads = []

    def read(reference):
        reads.append(reference)
        return evidence

    restored = AdmittedNativePopulation.from_checkpoint(
        tmp_path, identity, sampler_correction=None, read_evidence=read, decode=decode
    )
    assert reads == ["artifact:episodes"]
    assert restored.resolved == admitted.resolved
    assert (restored.applied_update_offset, restored.attempt_offset) == (3, 5)
    for view in restored.resolved.snapshot.conditioning:
        assert restored.read_input(view) == admitted.read_input(view)
    with pytest.raises(InvalidPolicyUpdate, match="evidence bytes"):
        AdmittedNativePopulation.from_checkpoint(
            tmp_path, identity, sampler_correction=None, read_evidence=lambda _: evidence + b" ", decode=decode
        )
    reads.clear()
    (tmp_path / "native.bin").write_bytes(b"changed native weights")
    with pytest.raises(InvalidPolicyUpdate, match="changed|digest|differs|component"):
        AdmittedNativePopulation.from_checkpoint(
            tmp_path, identity, sampler_correction=None, read_evidence=read, decode=decode
        )
    assert reads == []
