"""Native TaskRunner lifecycle specialization, without launching model workers."""

from types import SimpleNamespace

import pytest
from posttrain.train.backends.verl.policy_runner import resolved_task_runner_type

tq = pytest.importorskip("transfer_queue")
pytest.importorskip("verl")


@pytest.mark.parametrize("failure", [None, "queue-init", "trainer-init", "fit"])
def test_native_task_runner_keeps_manager_lifecycle_and_closes_on_failure(monkeypatch, failure):
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import TaskRunnerV1Base

    events = []

    def event(name):
        events.append(name)
        if name == failure:
            raise RuntimeError(name)

    monkeypatch.setattr(tq, "init", lambda config: event("queue-init"))
    monkeypatch.setattr(tq, "close", lambda: event("queue-close"))
    monkeypatch.setattr("verl.utils.logging_utils.configure_verl_logging", lambda: None)

    class Trainer:
        def __init__(self, config):
            event("trainer-construct")
            self.logger = SimpleNamespace(finish=lambda exit_code: events.append(("finish", exit_code)))

        def init(self):
            event("trainer-init")

        def fit(self, manager):
            assert manager == "native-manager"
            event("fit")

    runner_cls = resolved_task_runner_type(Trainer)

    def manager(self):
        event("manager-init")
        self.agent_loop_manager = "native-manager"

    monkeypatch.setattr(TaskRunnerV1Base, "init_agent_loop_manager", manager)
    runner = runner_cls()
    assert isinstance(runner, TaskRunnerV1Base)
    config = OmegaConf.create({"transfer_queue": {"enable": False}})
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            runner.run(config)
    else:
        runner.run(config)
    assert config.transfer_queue.enable
    assert events[-1] == "queue-close"
    if failure != "queue-init":
        assert events[-2] == ("finish", 1 if failure else 0)
    if not failure:
        assert events == [
            "queue-init",
            "trainer-construct",
            "trainer-init",
            "manager-init",
            "fit",
            ("finish", 0),
            "queue-close",
        ]
