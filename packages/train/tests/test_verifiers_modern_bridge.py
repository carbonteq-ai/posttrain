"""Real local native episode lifecycle with an injected exact-token policy."""

import json
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import JsonValue
from posttrain.train.integrations.verifiers import VerifiersEnvironmentRolloutBridge
from posttrain.train.online_rl import BehaviorPolicySpan, PolicySampling, PolicyTurnResult, RolloutBatch
from posttrain.train.rollout_execution import CollectionKey, EpisodeKey


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
    info = rollout.trace.payload["info"]
    assert isinstance(info, dict)
    assert info["posttrain_episode_id"] == episode.id
    artifacts = bridge.finalize()
    assert len(artifacts) == 2
    assert artifacts[0].metadata["replay_authority"] is True
    assert artifacts[0].metadata["episode_count"] == 1
    assert artifacts[1].metadata["replay_authority"] is False


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
