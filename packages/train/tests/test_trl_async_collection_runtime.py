import asyncio
import threading
from types import SimpleNamespace

from posttrain.common.variants import LFM_25_12B_THINKING
from posttrain.train.backends.trl import async_collection_runtime as runtime_module
from posttrain.train.backends.trl.async_collection_runtime import TrlAsyncCollectionRuntime
from posttrain.train.online_rl import PolicySampling, RolloutBatch
from posttrain.train.profiles import LFM25_RENDERER
from posttrain.train.rollout_execution import RolloutExecutionConfig


def test_sync_collection_runtime_reuses_one_async_loop_and_replays_observations_on_caller(monkeypatch):
    loop_ids = []
    closed = []

    class Session:
        async def aclose(self):
            closed.append("session")

    class Generation:
        model = SimpleNamespace(name_or_path="model-snapshot")

        async def create_async_session(self):
            loop_ids.append(id(asyncio.get_running_loop()))
            return Session()

    class Endpoint:
        def __init__(self, **_kwargs):
            pass

    class Workers:
        def __init__(self, *, project_episode, **_kwargs):
            self.project_episode = project_episode

    class Runner:
        def __init__(self, *, session, endpoint, workers, **_kwargs):
            self.session = session
            self.workers = workers

        async def start(self, activation):
            assert activation == {"agent": {"timeout": {"rollout": 5}}}

        async def aclose(self):
            await self.session.aclose()

    class Collector:
        def __init__(self, *, runner, **_kwargs):
            self.runner = runner

        async def collect(self, batch, *, collection_id, policy_version):
            loop_ids.append(id(asyncio.get_running_loop()))
            rollout = await self.runner.workers.project_episode(
                SimpleNamespace(collection=SimpleNamespace(run_id="run")),
                SimpleNamespace(),
            )
            return [rollout]

    class Bridge:
        run_id = "run"
        max_concurrent = 1
        sampling = PolicySampling(max_tokens=8, temperature=0.1)
        environment_factory = SimpleNamespace(config={"agent": {"timeout": {"rollout": 5}}})
        native_activation = {"agent": {"timeout": {"rollout": 5}}}

        async def project_native_episode(self, _key, _episode, *, on_completed, **_kwargs):
            await on_completed(SimpleNamespace(external_id="trace-1"))
            return SimpleNamespace(example_id="train/000000")

    monkeypatch.setattr(runtime_module, "TrlPolicyEndpoint", Endpoint)
    monkeypatch.setattr(runtime_module, "VerifiersWorkerPool", Workers)
    monkeypatch.setattr(runtime_module, "TrlCollectionRunner", Runner)
    monkeypatch.setattr(runtime_module, "TrlNativeRolloutCollector", Collector)

    runtime = TrlAsyncCollectionRuntime(
        trainer=SimpleNamespace(vllm_generation=Generation()),
        bridge=Bridge(),
        model=LFM_25_12B_THINKING,
        renderer=LFM25_RENDERER,
        renderer_model_name="model-snapshot",
        max_model_len=128,
        execution_config=RolloutExecutionConfig(env_workers=1, episodes_per_worker=1),
        episode_timeout=5,
    )
    caller_thread = threading.get_ident()
    observed_threads = []
    try:
        for ordinal in range(2):
            result = runtime.collect(
                RolloutBatch(("train/000000",), ordinal + 1, LFM_25_12B_THINKING.id),
                collection_id=f"collection-{ordinal}",
                policy_version=f"policy-{ordinal}",
                observe_trace=lambda _trace: observed_threads.append(threading.get_ident()),
            )
            assert result[0].example_id == "train/000000"
    finally:
        runtime.close()

    assert len(set(loop_ids)) == 1
    assert observed_threads == [caller_thread, caller_thread]
    assert closed == ["session"]
