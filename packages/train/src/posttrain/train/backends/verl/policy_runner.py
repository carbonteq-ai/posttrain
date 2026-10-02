"""Local recipe specialization of veRL's intended native TaskRunner extension."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolved_task_runner_type(trainer_cls: Callable[..., Any]) -> type:
    """Retain native manager construction, TransferQueue and terminal cleanup.

    The caller supplies the resolved synchronous trainer type before Ray
    dispatch. This does not replace the global native sync trainer registry.
    """
    from verl.trainer.main_ppo import TaskRunnerV1Base  # pyright: ignore[reportMissingImports]

    class ResolvedTaskRunner(TaskRunnerV1Base):
        def run(self, config: Any):
            import transfer_queue as tq  # pyright: ignore[reportMissingImports]
            from omegaconf import OmegaConf
            from verl.utils.logging_utils import configure_verl_logging  # pyright: ignore[reportMissingImports]

            configure_verl_logging()
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            self.config = config
            succeeded = False
            try:
                tq.init(config.transfer_queue)
                self.trainer = trainer_cls(config=config)
                self.trainer.init()
                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
                succeeded = True
            finally:
                try:
                    tracking = getattr(self.trainer, "logger", None)
                    if tracking is not None:
                        tracking.finish(exit_code=0 if succeeded else 1)
                finally:
                    tq.close()

    return ResolvedTaskRunner
