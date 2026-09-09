"""Real native Verifiers process-pool qualification with a loopback policy."""

import json
from types import SimpleNamespace

import pytest
from posttrain.common.variants import LFM_25_12B_THINKING
from posttrain.train.backends.trl.collection_runner import TrlCollectionRunner, TrlNativeRolloutCollector
from posttrain.train.backends.trl.policy_endpoint import TrlPolicyEndpoint
from posttrain.train.integrations.verifiers import VerifiersEnvironmentRolloutBridge
from posttrain.train.integrations.verifiers_workers import (
    VerifiersWorkerPool,
    create_verifiers_train_client_config,
)
from posttrain.train.online_rl import PolicySampling, RolloutBatch
from posttrain.train.profiles import LFM25_RENDERER
from posttrain.train.rollout_execution import RolloutExecutionConfig


@pytest.mark.asyncio
async def test_real_spawned_worker_returns_a_projected_native_episode(monkeypatch, tmp_path):
    pytest.importorskip("automationbench_v1")
    from transformers import AutoTokenizer
    from verifiers.v1 import Sampling
    from verifiers.v1.utils.loaders import load_environment, resolve_env_config

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = LFM_25_12B_THINKING
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model.base.repo_id,
            revision=model.base.revision,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("real worker qualification requires the selected LFM tokenizer in the local cache")
    tool_completion_ids = tokenizer.encode(
        "<|tool_call_start|>[salesforce_note_create(parent_id='001001', title='Q1 Meeting Summary', body='Discussed Q1 targets and expansion plans. Client interested in upgrading to enterprise tier.')]<|tool_call_end|>",
        add_special_tokens=False,
    )
    final_completion_ids = tokenizer.encode("Done.", add_special_tokens=False)
    activation = {
        "taskset": {
            "id": "automationbench-v1",
            "task_names": ["simple.sf_create_account_note"],
            "task": {"toolset": "limited_zapier"},
        },
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "max_turns": 2,
        },
    }
    environment = load_environment(resolve_env_config(activation))
    task = next(iter(environment.taskset.load()))

    class Session:
        def __init__(self):
            self.policy_version = None
            self.requests = 0

        async def synchronize_policy(self, version):
            self.policy_version = version

        async def open_policy(self, version):
            assert version == self.policy_version

        async def generate(self, request):
            ids = tool_completion_ids if self.requests == 0 else final_completion_ids
            self.requests += 1
            completion = SimpleNamespace(
                index=0,
                token_ids=ids,
                logprobs=[{token_id: SimpleNamespace(logprob=-0.125)} for token_id in ids],
                finish_reason="stop",
            )
            return SimpleNamespace(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                outputs=[completion],
                finished=True,
            )

        async def abort(self, _request_id):
            return False

        async def stop_admission(self):
            return None

        async def suspend_for_update(self):
            return None

        async def aclose(self):
            return None

    session = Session()
    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="automation/native-worker",
        revision="1",
        tasks={0: task},
        environment_factory=lambda: environment,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="automationbench-v1",
        run_id="run-native-worker",
        sampling=PolicySampling(
            max_tokens=max(len(tool_completion_ids), len(final_completion_ids)),
            temperature=0.1,
        ),
    )
    endpoint = TrlPolicyEndpoint(model_name=model.base.repo_id, max_model_len=4096)
    workers = VerifiersWorkerPool(
        model=model.base.repo_id,
        sampling=Sampling(max_tokens=max(len(tool_completion_ids), len(final_completion_ids)), temperature=0.1),
        project_episode=bridge.project_native_episode,
        global_limit=1,
        startup_timeout=30,
    )
    runner = TrlCollectionRunner(
        session=session,
        endpoint=endpoint,
        workers=workers,
        client_config_factory=lambda base_url: create_verifiers_train_client_config(
            base_url=base_url,
            renderer_model_name=model.base.repo_id,
            model=model,
            renderer=LFM25_RENDERER,
            multiplex=1,
        ),
        execution_config=RolloutExecutionConfig(env_workers=1, episodes_per_worker=1),
    )
    await runner.start(activation)
    try:
        collector = TrlNativeRolloutCollector(runner=runner, bridge=bridge, episode_timeout=60)
        [rollout] = await collector.collect(
            RolloutBatch(("train/000000",), 1, model.id),
            collection_id="step-1/batch-1",
            policy_version="optimizer-step-0",
        )
        assert rollout.example_id == "train/000000"
        assert rollout.reward == 1.0
        assert session.requests == 2
        assert tuple(tool_completion_ids) == rollout.completion_ids[: len(tool_completion_ids)]
        assert rollout.trace.attributes["environment_id"] == "automationbench-v1"
        assert (tmp_path / "episodes.jsonl").is_file()
        assert (tmp_path / "traces.jsonl").is_file()
        episode = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
        assistant_calls = [
            node["message"]["tool_calls"]
            for trace in episode["traces"]
            for node in trace["nodes"]
            if node["message"]["role"] == "assistant" and node["message"].get("tool_calls")
        ]
        assert assistant_calls[0][0]["name"] == "salesforce_note_create"
    finally:
        await runner.aclose()
