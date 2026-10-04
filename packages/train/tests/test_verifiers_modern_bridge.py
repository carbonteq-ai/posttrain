"""Real local native episode lifecycle with an injected exact-token policy."""

import gzip
import hashlib
import json
import threading
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import JsonValue, LocalArtifactRef
from posttrain.train.integrations.verifiers import (
    VerifiersEnvironmentRolloutBridge,
    _compress_jsonl,
    _episode_assessment_observation,
    _project_training_branch,
)
from posttrain.train.online_rl import BehaviorPolicySpan, PolicySampling, PolicyTurnResult, RolloutBatch
from posttrain.train.rollout_execution import CollectionKey, EpisodeKey


def test_assigned_credit_allocation_preserves_mass_and_rejects_boundary():
    from dataclasses import replace

    from posttrain.train.reward_evidence import AssignedCredit, InvalidRewardEvidence

    credit = AssignedCredit(
        assignment_id="assignment",
        contribution_id="harm",
        rule_digest="rule",
        attempt_id="attempt",
        channel="guard",
        semantics="cost",
        branch_digest="path",
        status="valid",
        alignment="exact",
        allocation="fixed_mass",
        value=-2,
        weight=3,
        signal_digest="signal",
        units="reward",
        transformation="identity",
        parent_assessment_ids=("assessment",),
        attribution="exact",
        recipient_kind="turn",
        recipient_id="subject",
        source_snapshot_id="source",
        recipient_trace_id="trace",
        intervals=((0, 2), (1, 3)),
    )
    mask = (True, True, True, False)
    assert credit.token_values(mask) == (-2, -2, -2, 0)
    assert sum(credit.token_values(mask)) == -6
    assert replace(credit, allocation="broadcast").token_values(mask) == (-6, -6, -6, 0)
    with pytest.raises(InvalidRewardEvidence, match="boundary events"):
        replace(credit, allocation="turn_boundary").token_values(mask)
    with pytest.raises(InvalidRewardEvidence, match="original sampled"):
        credit.support((True, False, True, False))
    with pytest.raises(InvalidRewardEvidence, match="explicit reward conversion"):
        replace(credit, semantics="probability").token_values(mask)


def test_native_assignment_ingestion_keeps_harm_separate_from_positive_outcome():
    vf = pytest.importorskip("verifiers.v1")
    if not hasattr(vf, "CreditAssignment"):
        pytest.skip("requires native assessment candidate checkout")
    from posttrain.train.integrations.verifiers_credit import align_native_credit
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage, UserMessage

    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="exclude protected accounts")),
    )
    trace.rewards["solved"] = vf.Reward(score=1)
    trace.nodes = [
        MessageNode(message=UserMessage(content="request"), token_ids=[1], mask=[False]),
        MessageNode(
            parent=0,
            message=AssistantMessage(content="send protected conversion"),
            sampled=True,
            token_ids=[2, 3, 4, 5],
            mask=[False, True, True, True],
            logprobs=[-0.1, -0.2, -0.3],
        ),
    ]
    source = capture_trace_source(trace)
    recipient = vf.SubjectRef(
        kind="turn",
        snapshot_id=source.snapshot_id,
        episode_id=trace.episode_id,
        trace_id=trace.id,
        node_index=1,
        node_content_digest=source.nodes[1].node_content_digest,
    )
    signal = vf.SignalDefinition(
        signal_id="harm", revision="1", semantics="cost", description="protected send", units="reward"
    )
    assessment = vf.Assessment(
        assessment_id="harm",
        run_id="judge",
        subject=recipient,
        view_id="context",
        signal=signal,
        status="valid",
        value=-1,
    )
    request = vf.CreditRequest(
        source=source,
        invocation_id="assign",
        attempt_id="attempt",
        rule=vf.CreditRule(rule_id="protected-action", revision="1"),
        accepted=(assessment,),
        targets=(vf.CreditTarget(recipient=recipient, channel="guard"),),
        allocation="fixed_mass",
        overlap_policy="reject",
    )
    contribution = vf.CreditContribution(
        contribution_id="harm",
        parent_assessment_ids=("harm",),
        recipient=recipient,
        channel="guard",
        signal=signal,
        transformation="identity",
        status="valid",
        value=-1,
        allocation="fixed_mass",
        attribution="coarse",
    )
    assignment = vf.CreditAssignment(request=request, contributions=(contribution,))
    trace.credit_assignments = [assignment, assignment]
    branch = _project_training_branch(trace)
    aligned = align_native_credit(trace, branch, 2)
    assert len(aligned.contributions) == 1  # duplicate journal entry is not a second reward term
    aligned.attempts[0].require_complete()
    assert (
        aligned.select(rule_digest=request.rule.digest, invocation_id="assign", attempt_id="attempt", channel="guard")
        == aligned.contributions
    )
    assert aligned.contributions[0].channel == "guard" and trace.reward == 1
    assert aligned.contributions[0].intervals == ((0, 3),)
    assert sum(aligned.contributions[0].token_values((True, True, True))) == -1
    # Episode-owned assignment reaches the same consumer without child duplication.
    trace.credit_assignments = []
    assert align_native_credit(trace, branch, 2, assignments=(assignment,)) == aligned
    from posttrain.common import TraceObservation

    bridge = object.__new__(VerifiersEnvironmentRolloutBridge)
    bridge.technique = "grpo"
    bridge.reward_projection = None
    bridge.policy_update_context_contract = None
    observation = TraceObservation(
        trace_type="verifiers.rollout",
        external_id=trace.id,
        payload={},
        attributes={"example_id": "task", "is_truncated": False},
    )
    rollout = bridge._project(trace, observation, assignments=(assignment,))
    assert rollout.reward == 1 and rollout.assigned_credit == aligned
    assert rollout.completion_ids == (3, 4, 5) and rollout.env_mask == (True, True, True)
    from posttrain.train.assigned_rewards import CreditSelection, episode_reward

    episode_subject = vf.SubjectRef(
        kind="episode",
        snapshot_id=source.snapshot_id,
        episode_id=trace.episode_id,
    )
    outcome_signal = vf.SignalDefinition(
        signal_id="outcome",
        revision="1",
        semantics="outcome",
        description="verified completion",
        units="reward",
    )
    outcome = vf.Assessment(
        assessment_id="outcome",
        run_id="check",
        subject=episode_subject,
        view_id="context",
        signal=outcome_signal,
        status="valid",
        value=0.75,
    )
    outcome_request = vf.CreditRequest(
        source=source,
        invocation_id="outcome",
        attempt_id="attempt",
        rule=request.rule,
        accepted=(outcome,),
        targets=(vf.CreditTarget(recipient=episode_subject, channel="outcome"),),
        allocation="turn_boundary",
        overlap_policy="reject",
    )
    outcome_assignment = vf.CreditAssignment(
        request=outcome_request,
        contributions=(
            vf.CreditContribution(
                contribution_id="outcome",
                parent_assessment_ids=("outcome",),
                recipient=episode_subject,
                channel="outcome",
                signal=outcome_signal,
                transformation="identity",
                status="valid",
                value=0.75,
                allocation="turn_boundary",
                attribution="coarse",
            ),
        ),
    )
    selected_outcome = bridge._project(trace, observation, assignments=(outcome_assignment,))
    selection = CreditSelection(request.rule.digest, "outcome", "attempt", "outcome")
    assert episode_reward(selected_outcome, selection) == 0.75
    assert selected_outcome.reward == 1
    rewritten = trace.model_copy(deep=True)
    rewritten.nodes[1].token_ids[1] = 99
    invalid = align_native_credit(rewritten, _project_training_branch(rewritten), 2, assignments=(assignment,))
    assert invalid.contributions[0].alignment == "failed" and not invalid.contributions[0].intervals
    invalid_outcome = bridge._project(rewritten, observation, assignments=(outcome_assignment,))
    assert invalid_outcome.assigned_credit.contributions[0].alignment == "failed"
    with pytest.raises(ValueError, match="incompatible allocation or source alignment"):
        episode_reward(invalid_outcome, selection)
    incomplete = vf.CreditAssignment(request=request, status="failed", reason="rule_failed")
    failed = align_native_credit(trace, branch, 2, assignments=(incomplete,))
    assert failed.contributions == () and failed.attempts[0].missing_count == 1
    from posttrain.train.reward_evidence import InvalidRewardEvidence

    with pytest.raises(InvalidRewardEvidence, match="incomplete coverage"):
        failed.attempts[0].require_complete()
    changed_request = request.model_copy(update={"rule": vf.CreditRule(rule_id="changed", revision="1")})
    changed_assignment = assignment.model_copy(update={"request": changed_request})
    with pytest.raises(InvalidRewardEvidence, match="changed request"):
        align_native_credit(trace, branch, 2, assignments=(assignment, changed_assignment))


@pytest.mark.asyncio
async def test_episode_assessment_replay_without_solver_file_and_live_delivery(tmp_path):
    # No policy runtime is needed to exercise the native evidence export path.
    bridge = object.__new__(VerifiersEnvironmentRolloutBridge)
    bridge.trace_path = tmp_path / "traces.jsonl"
    bridge._write_lock = threading.Lock()
    bridge._live_observed_trace_ids = set()
    bridge._requested_by_step = {}
    record = {
        "id": "failed-episode",
        "ok": False,
        "traces": [],
        "assessment_batches": [{"run": {"status": "failed"}}],
        "assessment_finalization_state": "not_run",
    }
    episode = SimpleNamespace(id=record["id"], traces=[], to_record=lambda: record)
    bridge._preserve_episode(episode)
    replay = bridge.evidence()
    assert len(replay.traces) == 1
    assert replay.metrics == ()
    result = replay.traces[0]
    assert result.trace_type == "verifiers.assessment-results"
    assert result.payload["child_trace_ids"] == []
    assert result.payload["execution_ok"] is False

    async def unavailable(_observation):
        raise RuntimeError("observer unavailable")

    await bridge._observe_episode_assessments(record, unavailable)
    assert len(bridge.evidence().traces) == 1
    delivered = []

    async def observe(observation):
        delivered.append(observation)

    await bridge._observe_episode_assessments(record, observe)
    await bridge._observe_episode_assessments(record, observe)
    assert [item.external_id for item in delivered] == [result.external_id]
    assert bridge.evidence().traces == ()
    assert json.loads((tmp_path / "episodes.jsonl").read_text()) == record


def test_episode_assessment_envelope_links_children_without_copying_them():
    record = {
        "id": "episode-a",
        "ok": True,
        "traces": [{"id": "solver-a"}, {"id": "solver-b"}],
        "assessment_batches": [{"result": 0.0}],
        "assessment_sources": [{"snapshot_id": "sealed", "source_json": "native source"}],
        "assessment_views": [{"view_id": "view", "input_json": "native assessor context"}],
    }
    observation = _episode_assessment_observation(record)
    assert observation is not None
    assert observation.payload["child_trace_ids"] == ["solver-a", "solver-b"]
    assert observation.payload["assessment_sources"] == record["assessment_sources"]
    assert observation.payload["assessment_views"] == record["assessment_views"]
    assert "traces" not in observation.payload
    assert "rewards" not in observation.payload
    assert _episode_assessment_observation({"id": "empty", "traces": []}) is None
    credit_only = _episode_assessment_observation(
        {"id": "credit", "traces": [], "credit_assignments": [{"status": "complete"}], "credit_errors": ["failed"]}
    )
    assert credit_only is not None
    assert credit_only.payload["credit_assignments"] == [{"status": "complete"}]
    assert credit_only.payload["credit_error_count"] == 1


def test_native_compression_failure_preserves_previous_archive(tmp_path, monkeypatch):
    source = tmp_path / "episodes.jsonl"
    source.write_text('{"id":"original"}\n')
    sealed = _compress_jsonl(source)
    original = sealed.path.read_bytes()
    source.write_text('{"id":"next"}\n')

    def unavailable(**kwargs):
        raise OSError("compression interrupted")

    monkeypatch.setattr("posttrain.train.integrations.verifiers.gzip.GzipFile", unavailable)
    with pytest.raises(OSError, match="compression interrupted"):
        _compress_jsonl(source)
    assert sealed.path.read_bytes() == original
    assert list(tmp_path.glob(".episodes.jsonl.gz.*")) == []


def test_training_branch_replaces_canonical_tool_turn_with_exact_sampled_sibling() -> None:
    vf = pytest.importorskip("verifiers.v1")

    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="task")),
        nodes=[
            vf.MessageNode(message=vf.SystemMessage(content="system"), token_ids=[1], mask=[False]),
            vf.MessageNode(parent=0, message=vf.UserMessage(content="task"), token_ids=[2], mask=[False]),
            vf.MessageNode(
                parent=1,
                message=vf.AssistantMessage(content="sampled tool call"),
                sampled=True,
                token_ids=[3, 4],
                mask=[True, True],
                logprobs=[-0.1, -0.2],
            ),
            vf.MessageNode(
                parent=1,
                message=vf.AssistantMessage(content="canonical tool call"),
                token_ids=[30, 40],
                mask=[False, False],
            ),
            vf.MessageNode(
                parent=3,
                message=vf.ToolMessage(tool_call_id="call", content="result"),
                token_ids=[5],
                mask=[False],
            ),
            vf.MessageNode(
                parent=4,
                message=vf.AssistantMessage(content="done"),
                sampled=True,
                token_ids=[6],
                mask=[True],
                logprobs=[-0.3],
            ),
        ],
        is_completed=True,
        ok=True,
    )

    branch = _project_training_branch(trace)

    assert branch.token_ids == [1, 2, 3, 4, 5, 6]
    assert branch.sampled_mask == [False, False, True, True, False, True]
    assert branch.logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, -0.3]
    if not hasattr(vf, "CreditAssignment"):
        return  # The original branch regression also runs against the old pin.
    from posttrain.common import TraceObservation
    from posttrain.train.assigned_rewards import CreditSelection, local_token_rewards
    from verifiers.v1.assessment_source import capture_trace_source

    source = capture_trace_source(trace)
    parents, targets, contributions = [], [], []
    for node_index, channel, semantics, value in ((2, "progress", "progress", 1), (5, "guard", "cost", -1)):
        recipient = vf.SubjectRef(
            kind="turn",
            snapshot_id=source.snapshot_id,
            episode_id="episode",
            trace_id=trace.id,
            node_index=node_index,
            node_content_digest=source.nodes[node_index].node_content_digest,
        )
        signal = vf.SignalDefinition(
            signal_id=channel, revision="1", semantics=semantics, description=channel, units="reward"
        )
        parents.append(
            vf.Assessment(
                assessment_id=channel,
                run_id="run",
                subject=recipient,
                view_id="full-context",
                signal=signal,
                status="valid",
                value=value,
            )
        )
        targets.append(vf.CreditTarget(recipient=recipient, channel=channel))
        contributions.append(
            vf.CreditContribution(
                contribution_id=channel,
                parent_assessment_ids=(channel,),
                recipient=recipient,
                channel=channel,
                signal=signal,
                transformation="identity",
                status="valid",
                value=value,
                allocation="fixed_mass",
                attribution="coarse",
            )
        )
    rule = vf.CreditRule(rule_id="conditional-workflow", revision="1")
    request = vf.CreditRequest(
        source=source,
        invocation_id="assign",
        attempt_id="attempt",
        rule=rule,
        accepted=tuple(parents),
        targets=tuple(targets),
        allocation="fixed_mass",
        overlap_policy="reject",
    )
    trace.credit_assignments.append(vf.CreditAssignment(request=request, contributions=tuple(contributions)))
    trace.record_reward("official", 0.75)
    restored = vf.WireTrace.model_validate(trace.to_record())
    bridge = object.__new__(VerifiersEnvironmentRolloutBridge)
    bridge.technique = "sampo"
    bridge.reward_projection = None
    bridge.policy_update_context_contract = None
    observation = TraceObservation(
        "verifiers.rollout", restored.id, {}, attributes={"example_id": "task", "is_truncated": False}
    )
    rollout = bridge._project(restored, observation)
    assert rollout.completion_ids == (3, 4, 5, 6)
    assert rollout.env_mask == (True, True, False, True) and rollout.reward == 0.75
    assert [item.intervals for item in rollout.assigned_credit.contributions] == [((0, 2),), ((3, 4),)]
    assert local_token_rewards(rollout, CreditSelection(rule.digest, "assign", "attempt", "progress")) == (
        0.5,
        0.5,
        0,
        0,
    )
    assert local_token_rewards(rollout, CreditSelection(rule.digest, "assign", "attempt", "guard")) == (0, 0, 0, -1)
    # First native branch has only the sampled sibling, not the later action.
    from posttrain.train.integrations.verifiers_credit import align_native_credit

    partial = align_native_credit(restored, restored.branches[0], 2)
    assert partial.contributions[1].alignment == "absent"
    assert partial.contributions[1].intervals == ()


def test_native_judge_resolution_does_not_mutate_recoverable_selection():
    pytest.importorskip("verifiers.v1.utils.loaders")
    pytest.importorskip("automationbench_v1")
    from posttrain.environment import VerifiersV1ConfigActivation

    activation = VerifiersV1ConfigActivation(
        {
            "taskset": {
                "id": "automationbench-v1",
                "task": {
                    "judges": [
                        {
                            "id": "automationbench-v1",
                            "code_revision": "a" * 40,
                            "model_revision": "b" * 40,
                            "model": "judge",
                            "input_budget_tokens": 12_288,
                        }
                    ]
                },
            },
            "agent": {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}},
        }
    )
    before = json.dumps(activation.to_payload(), sort_keys=True)
    digest = activation.digest
    activation.activate()
    activation.activate()
    assert json.dumps(activation.to_payload(), sort_keys=True) == before
    assert activation.digest == digest


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_modern_native_episode_retains_exact_policy_tokens(tmp_path, failure):
    module = pytest.importorskip("verifiers.v1.episode")
    if not hasattr(module, "WireEpisode"):
        pytest.skip("requires modern native episode runtime")
    pytest.importorskip("reverse_text")
    from verifiers.v1.utils.loaders import load_environment, resolve_env_config

    env = load_environment(
        resolve_env_config(
            {
                "taskset": {"id": "reverse-text"},
                "agent": {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}, "max_turns": 2},
            }
        )
    )
    task = next(iter(env.taskset.load()))

    class Policy:
        async def generate(self, request):
            if failure:
                raise RuntimeError("deliberate policy failure")
            message = {"role": "assistant", "content": "ready"}
            prompt = tuple(range(100, 100 + len(request.messages)))
            return PolicyTurnResult(
                message=message,
                prompt_ids=prompt,
                completion_ids=(900, 901),
                completion_logprobs=(-0.123456789123, -0.223456789123),
                finish_reason="stop",
                reasoning_tokens=1,
                prompt_message_spans=tuple((i, i + 1) for i in range(len(prompt))),
                raw_response=cast(
                    dict[str, JsonValue],
                    {
                        "id": "policy",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "policy",
                        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    },
                ),
            )

    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="native",
        revision="1",
        tasks={0: task},
        environment_factory=lambda: env,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="reverse-text",
        run_id="run",
        sampling=PolicySampling(max_tokens=8),
    )
    batch = RolloutBatch(("train/000000",), 1, "policy", prompt_group_ids=("group",), rollout_ids=("rollout",))
    if failure:
        from posttrain.train.integrations.verifiers import VerifiersRolloutFailure

        with pytest.raises(VerifiersRolloutFailure):
            await bridge.run(batch, Policy())
        episode = module.WireEpisode.model_validate_json((tmp_path / "episodes.jsonl").read_text())
        assert not episode.ok
        assert episode.group.id == "group"
        assert episode.env.name == "reverse-text"
        assert episode.run.work.step == 1
        [artifact] = bridge.finalize()
        assert artifact.metadata["replay_authority"] is True
        return
    [rollout] = await bridge.run(batch, Policy())
    assert rollout.completion_ids == (900, 901)
    assert rollout.sampling_logprobs == (-0.123456789123, -0.223456789123)
    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    episode = module.WireEpisode.model_validate(record)
    assert episode.run.work.step == 1
    assert episode.group.id == "group"
    assert episode.traces[0].branches[0].token_ids[-2:] == [900, 901]
    assert episode.traces[0].branches[0].logprobs[-2:] == [-0.123456789123, -0.223456789123]
    # Renderer reasoning accounting reaches native call usage (thinking facts and spans).
    assert [call.usage.reasoning_tokens for call in episode.traces[0].calls if call.usage is not None] == [1]
    info = rollout.trace.payload["info"]
    assert isinstance(info, dict)
    assert info["posttrain_episode_id"] == episode.id
    # Only the replay authority is published, compressed; the derived trace
    # view was streamed to tracking and is rebuilt from the episodes.
    [artifact] = bridge.finalize()
    assert artifact.metadata["replay_authority"] is True
    assert artifact.metadata["episode_count"] == 1
    assert artifact.metadata["compression"] == "gzip"
    reference = artifact.reference
    assert isinstance(reference, LocalArtifactRef)
    assert reference.path.name == "episodes.jsonl.gz"
    assert gzip.decompress(reference.path.read_bytes()) == (tmp_path / "episodes.jsonl").read_bytes()
    assert artifact.metadata["uncompressed_bytes"] == (tmp_path / "episodes.jsonl").stat().st_size
    assert reference.digest == hashlib.sha256(reference.path.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_worker_episode_uses_the_same_native_projection_and_lineage(tmp_path):
    module = pytest.importorskip("verifiers.v1.episode")
    if not hasattr(module, "WireEpisode"):
        pytest.skip("requires modern native episode runtime")
    import verifiers.v1 as vf

    task_data = vf.TaskData(idx=0, prompt="question")
    task = SimpleNamespace(data=task_data)
    trace_task = vf.TraceTask(type="Task", data=task_data)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=trace_task,
        nodes=[
            vf.MessageNode(
                parent=None,
                message=vf.UserMessage(content="question"),
                token_ids=[10, 11],
                mask=[False, False],
            ),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content="ready"),
                sampled=True,
                token_ids=[12, 900, 901],
                mask=[False, True, True],
                logprobs=[-0.125, -0.25],
            ),
        ],
        rewards={"task": vf.Reward(score=0.75)},
        is_completed=True,
        ok=True,
    )
    episode = module.WireEpisode.model_validate(
        vf.Episode(task=trace_task, ok=True, traces=[trace]).model_dump(mode="python")
    )

    worker = VerifiersEnvironmentRolloutBridge(
        dataset_id="native",
        revision="1",
        tasks={0: task},
        environment_factory=object,
        trace_path=tmp_path / "worker" / "traces.jsonl",
        environment_id="reverse-text",
        run_id="run",
        sampling=PolicySampling(max_tokens=8),
    )
    key = EpisodeKey(
        collection=CollectionKey("run", "collection-3", "policy-3", logical_step=3),
        example_id="train/000000",
        group_id="group",
        occurrence_id="rollout",
        seed=7,
        rollout_ordinal=0,
    )
    worker_rollout = await worker.project_native_episode(
        key,
        episode,
        behavior_policy=BehaviorPolicySpan(3, 5),
    )

    assert worker_rollout.prompt_ids == (10, 11, 12)
    assert worker_rollout.completion_ids == (900, 901)
    assert worker_rollout.sampling_logprobs == (-0.125, -0.25)
    assert worker_rollout.env_mask == (True, True)
    assert worker_rollout.reward == 0.75
    assert worker_rollout.behavior_policy == BehaviorPolicySpan(3, 5)
    assert worker_rollout.trace.attributes["example_id"] == "train/000000"
    retained = module.WireEpisode.model_validate_json((tmp_path / "worker" / "episodes.jsonl").read_text())
    assert retained.run.id == "run"
    assert retained.run.work.step == 3
    assert retained.run.work.policy.start == 3
    assert retained.run.work.policy.end == 5
    assert retained.group.id == "group"
    assert retained.traces[0].info["posttrain_rollout_id"] == "rollout"
    assert retained.traces[0].info["posttrain_run"]["policy"] == {"start": 3, "end": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize("toolset", ["limited_zapier", "zapier"])
async def test_modern_automationbench_executes_tool_and_retains_turn_credit(tmp_path, toolset):
    module = pytest.importorskip("verifiers.v1.episode")
    if not hasattr(module, "WireEpisode"):
        pytest.skip("requires modern native episode runtime")
    pytest.importorskip("automationbench_v1")
    from dataclasses import asdict

    from posttrain.train.reward_evidence import RewardValue
    from posttrain.train.reward_projection import RewardComponentProjection, RewardProjection
    from posttrain.train.turn_rewards import TURN_PROJECTION, TurnAssessment, native_turn_map
    from verifiers.v1.utils.loaders import load_environment, resolve_env_config

    env = load_environment(
        resolve_env_config(
            {
                "taskset": {"id": "automationbench-v1", "domains": ["simple"], "task": {"toolset": toolset}},
                "agent": {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}, "max_turns": 3},
            }
        )
    )
    task = next(iter(env.taskset.load()))
    tool_name = "salesforce_contact_update" if toolset == "limited_zapier" else "execute_tool"
    arguments = '{"id":"003001","phone":"+1-555-0101"}'
    if toolset == "zapier":
        arguments = json.dumps({"tool_name": "salesforce_contact_update", "arguments": arguments})

    class Policy:
        async def generate(self, request):
            assert tool_name in {tool["name"] for tool in request.tools}
            after_tool = any(message.get("role") == "tool" for message in request.messages)
            message = (
                {"role": "assistant", "content": "Updated the contact."}
                if after_tool
                else {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "update",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )
            prompt = tuple(range(100, 100 + len(request.messages)))
            finish = "stop" if after_tool else "tool_calls"
            if after_tool:
                prompt = request.previous_prompt_ids + request.previous_completion_ids + (104,)
                spans = ((0, 1), (1, 2), (2, 4), (4, 5))
            else:
                spans = tuple((i, i + 1) for i in range(len(prompt)))
            native_message = (
                message
                if after_tool
                else {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "update",
                            "name": tool_name,
                            "arguments": arguments,
                        }
                    ],
                }
            )
            return PolicyTurnResult(
                message=native_message,
                prompt_ids=prompt,
                completion_ids=(902, 903) if after_tool else (900, 901),
                completion_logprobs=(-0.1, -0.2),
                finish_reason=finish,
                prompt_message_spans=spans,
                raw_response=cast(
                    dict[str, JsonValue],
                    {
                        "id": "policy",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "policy",
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    },
                ),
            )

    def score_turns(trace):
        turns = native_turn_map(trace.branches[0])
        assert len(turns) == 2
        trace.info.update(
            posttrain_scorer_digest="a" * 64,
            ratings={
                "trace_id": trace.id,
                "branch_id": "0",
                "projection_id": TURN_PROJECTION,
                "scorer_digest": "a" * 64,
                "assessments": [
                    asdict(TurnAssessment(turn.id, (RewardValue("custom", "valid", score),), "ratings"))
                    for turn, score in zip(turns, (0.25, 0.75), strict=True)
                ],
            },
        )

    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="automation",
        revision="1",
        tasks={0: task},
        environment_factory=lambda: env,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="automationbench-v1",
        run_id="run",
        sampling=PolicySampling(max_tokens=8),
        technique="sampo",
        enrichers=(score_turns,),
        reward_projection=RewardProjection(
            "turns",
            "1",
            (RewardComponentProjection("outcome", "scalar"),),
            scorer_digest="a" * 64,
            turns_info_key="ratings",
            turn_reward_key="custom",
            turn_reward_includes_terminal_outcome=False,
        ),
    )
    [rollout] = await bridge.run(RolloutBatch(("train/000000",), 1, "policy"), Policy())
    assert rollout.reward == 1.0
    assert [turn.step_reward for turn in rollout.turns] == [0.25, 0.75]
    assert tuple(
        token for token, selected in zip(rollout.completion_ids, rollout.env_mask, strict=True) if selected
    ) == (
        900,
        901,
        902,
        903,
    )
    assert False in rollout.env_mask
    episode = module.WireEpisode.model_validate_json((tmp_path / "episodes.jsonl").read_text())
    assert episode.traces[0].rewards["partial_credit"].score == 1.0
