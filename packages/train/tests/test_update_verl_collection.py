"""Native active round decisions retain discarded and selected evidence."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import LocalArtifactRef, Observer, ProducedArtifact, RunContext, TraceObservation
from posttrain.train.backends.verl.policy_collection import NativeActiveCollectionBuffer
from posttrain.train.backends.verl.policy_rollouts import NativeEpisodeReceipt
from posttrain.train.online_rl import BehaviorPolicySpan
from posttrain.train.update_records import InvalidPolicyUpdate

from .test_update_admission import source
from .test_update_verl_driver import _native_sampo_fixture

pytest.importorskip("torch")
tq = pytest.importorskip("transfer_queue")


def collection(tmp_path, monkeypatch, mode, *, corrupt=False, reject_artifact=False, receipt_change=None):
    import numpy as np
    from verl.utils.tensordict_utils import get_tensordict

    trainer_cls, config = _native_sampo_fixture()
    trainer = trainer_cls(config)
    trainer.global_steps = 1
    trainer.prompt_selector = None
    trainer.on_sample_begin = trainer.on_sample_end = lambda: None
    trainer._consume_rollout_metrics = lambda: {}
    pool = get_tensordict(
        {"example_id": np.asarray(["a", "b", "c"], dtype=object), "uid": np.asarray(["a", "b", "c"], dtype=object)}
    )
    trainer._next_train_batch = lambda count: pool
    published, dispatches, cleared, observed = [], [], [], []

    def emit(artifact):
        if reject_artifact and artifact.kind == "evaluation-traces":
            raise RuntimeError("artifact observer failed")
        published.append(artifact)

    context = RunContext(
        "project",
        "work",
        "run",
        "train.sampo",
        "job@1",
        tmp_path,
        observer=cast(Observer, SimpleNamespace(artifact=emit)),
    )
    native = trainer.replay_buffer
    native.observe_fn = lambda groups: observed.append(groups)
    if mode == "surplus":
        native.active_oversample = 1
    trainer.replay_buffer = NativeActiveCollectionBuffer(native, context, tmp_path / "collection", 2)
    tags, fields = {}, {}
    rollouts, evidence, _ = source()

    def submit(batch):
        uids = [str(uid) for uid in batch["uid"]]
        dispatches.append(uids)
        for uid in uids:
            failed = mode == "failed" and uid == "a"
            tags[uid] = {"is_prompt": True, "status": "failure" if failed else "finished", "global_steps": 0}
            if failed:
                continue
            for index, original in enumerate(rollouts):
                if receipt_change == "incomplete" and uid == "a" and index == 1:
                    continue
                trace_id = f"{uid}-{index}"
                rollout = replace(
                    original,
                    example_id=uid,
                    behavior_policy=BehaviorPolicySpan(0, 0),
                    conditioning_records=tuple(
                        replace(record, trace_id=trace_id) for record in original.conditioning_records
                    ),
                    trace=TraceObservation(
                        "verifiers",
                        trace_id,
                        {
                            "info": {
                                "posttrain_episode_id": trace_id,
                                "posttrain_prompt_group_id": f"run/0/{uid}",
                                "posttrain_rollout_id": trace_id,
                            }
                        },
                    ),
                )
                path = tmp_path / f"{trace_id}.jsonl"
                path.write_bytes(evidence)
                artifact = ProducedArtifact(
                    f"native/{trace_id}",
                    "evaluation-traces",
                    LocalArtifactRef(path, hashlib.sha256(evidence).hexdigest()),
                    metadata={"format": "verifiers-native-traces", "replay_authority": True, "trace_ids": [trace_id]},
                )
                receipt = NativeEpisodeReceipt(artifact=artifact, rollout=rollout).model_dump_json(fallback=dict)
                if uid == "a" and receipt_change in {"task", "group", "policy"}:
                    altered = json.loads(receipt)
                    if receipt_change == "task":
                        altered["rollout"]["example_id"] = "other-task"
                    elif receipt_change == "group":
                        altered["rollout"]["trace"]["payload"]["info"]["posttrain_prompt_group_id"] = "other-group"
                    else:
                        altered["rollout"]["behavior_policy"] = {"start": 1, "end": 1}
                    receipt = json.dumps(altered)
                if corrupt and uid == "a":
                    path.write_bytes(b"changed")
                reward = index if uid == "b" or mode == "surplus" else 0
                key = f"{uid}_{index}_0"
                tags[key] = {"is_prompt": False, "seq_len": 3, "global_steps": 0}
                fields[key] = {
                    "reward_extra_info": {"group_reward": reward},
                    "posttrain_native_episode_receipt": receipt,
                }

    def clear(*, partition_id, keys):
        assert partition_id == "train"
        # Every completed episode was published before its native eviction.
        promoted = {artifact.name for artifact in published if artifact.kind == "evaluation-traces"}
        for key in keys:
            if key in fields:
                uid, index, _ = key.split("_")
                assert f"native/{uid}-{index}" in promoted
            cleared.append(key)
            tags.pop(key, None)
            fields.pop(key, None)

    trainer._submit_batch_to_rollout = submit
    monkeypatch.setattr(tq, "kv_list", lambda: {"train": dict(tags)})
    monkeypatch.setattr(tq, "kv_clear", clear)
    monkeypatch.setattr(
        tq, "kv_batch_get", lambda *, keys, partition_id, select_fields: {"extra_fields": [fields[key] for key in keys]}
    )
    commits = []
    trainer.actor_rollout_wg = SimpleNamespace(
        resolved_state=lambda: [{"applied": 0, "needs_population": True}],
        resolved_update=lambda receipts, **kwargs: (
            commits.append(receipts) or [{"train/rl/applied_optimizer_updates": 1}]
        ),
    )
    return trainer, published, dispatches, cleared, observed, commits


@pytest.mark.parametrize("mode", ["uniform", "failed", "surplus"])
def test_native_refill_surplus_and_unused_candidates_are_durable(tmp_path, monkeypatch, mode):
    trainer, artifacts, dispatches, cleared, observed, commits = collection(tmp_path, monkeypatch, mode)
    batch = trainer.step({}, {})
    selected = "a" if mode == "surplus" else "b"
    assert batch.keys == [f"{selected}_0_0", f"{selected}_1_0"]
    assert len(commits) == 1 and trainer.global_steps == 1
    assert dispatches == ([["a", "b"]] if mode == "surplus" else [["a"], ["b"]])
    assert len(observed) == len(dispatches)  # Original native observer still runs.
    retained = [artifact for artifact in artifacts if artifact.kind == "training-collection"]
    final = json.loads(retained[-1].reference.path.read_bytes())
    assert final["status"] == "selected" and final["selected"] == [selected]
    assert [item["uid"] for item in final["reserved"]] == ["a", "b", "c"]
    assert {uid for item in final["rounds"] for uid in item["uids"]} == {"a", "b"}
    assert len(final["groups"]) == 2  # Includes the evicted candidate.
    assert final["groups"][0]["terminal"] == ("failed" if mode == "failed" else "finished")
    assert len(final["groups"][0]["receipts"]) == (0 if mode == "failed" else 2)
    assert final["groups"][0]["native_spread_eligible"] == (mode == "surplus")
    assert final["groups"][1]["native_spread_eligible"] is True
    assert any(key.startswith("b_" if mode == "surplus" else "a_") for key in cleared) or mode == "failed"
    for artifact in retained:
        assert hashlib.sha256(artifact.reference.path.read_bytes()).hexdigest() == artifact.reference.digest


@pytest.mark.parametrize("failure", ["corrupt", "observer"])
def test_evidence_failure_prevents_native_eviction_and_optimizer(tmp_path, monkeypatch, failure):
    trainer, _, _, cleared, _, commits = collection(
        tmp_path, monkeypatch, "uniform", corrupt=failure == "corrupt", reject_artifact=failure == "observer"
    )
    with pytest.raises((InvalidPolicyUpdate, RuntimeError), match="digest|observer failed"):
        trainer.step({}, {})
    assert not cleared and not commits and trainer.global_steps == 1


def test_reservation_rejects_duplicate_uid_before_dispatch(tmp_path, monkeypatch):
    trainer, _, dispatches, _, _, _ = collection(tmp_path, monkeypatch, "uniform")
    with pytest.raises(InvalidPolicyUpdate, match="reserved task/UID"):
        trainer.replay_buffer.begin(("a", "b"), ("same", "same"), 0)
    assert not dispatches


@pytest.mark.parametrize("change", ["task", "group", "policy", "incomplete"])
def test_native_receipt_membership_and_coverage_fail_before_eviction(tmp_path, monkeypatch, change):
    trainer, _, _, cleared, _, commits = collection(tmp_path, monkeypatch, "uniform", receipt_change=change)
    with pytest.raises(InvalidPolicyUpdate, match="identity|complete finished"):
        trainer.step({}, {})
    assert not cleared and not commits and trainer.global_steps == 1


def test_real_driver_factory_installs_recorder_with_host_context(tmp_path):
    import numpy as np
    from posttrain.train.backends.verl.policy_driver import resolved_trainer_type
    from verl.utils.tensordict_utils import get_tensordict

    unobserved, config = _native_sampo_fixture()
    settings = unobserved(config).resolved_settings
    published = []
    context = RunContext(
        "project",
        "work",
        "run",
        "train.sampo",
        "job@1",
        tmp_path,
        observer=cast(Observer, SimpleNamespace(artifact=published.append)),
    )
    config.trainer.default_local_dir = str(tmp_path / "model" / "checkpoints")
    trainer = resolved_trainer_type(type("Actor", (), {}), settings, context)(config)
    assert isinstance(trainer.replay_buffer, NativeActiveCollectionBuffer)
    trainer.global_steps = 0
    trainer._reserve_candidates = lambda count: None
    trainer._candidate_pool = get_tensordict(
        {"example_id": np.asarray(["a", "b", "c"], dtype=object), "uid": np.asarray(["a", "b", "c"], dtype=object)}
    )
    trainer._prepare_resolved_collection()
    assert len(published) == 1 and published[0].metadata["status"] == "reserved"
    assert trainer.replay_buffer.native.dispatch_fn.__self__ is trainer.replay_buffer
