#!/usr/bin/env python
"""Qualify one real TRL optimizer update from Posttrain's async group producer.

This local gate deliberately controls environment outputs and weight transfer so
one GPU can exercise the learner. Native Verifiers process transport and real
vLLM weight publication are separate qualification gates.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import torch
from datasets import Dataset
from posttrain.common import TraceObservation
from posttrain.train.backends.trl.async_worker import TrlAsyncRolloutWorker
from posttrain.train.integrations.verifiers_async_groups import VerifiersAsyncGroupProducer
from posttrain.train.online_rl import BehaviorPolicySpan, EnvironmentRollout
from posttrain.train.profiles import GRPOSettings, TrainingLoop
from posttrain.train.rollout_execution import EpisodeOutcome, EpisodeStatus, RolloutExecutionConfig
from transformers import AutoTokenizer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


class _ControlledPolicyAdmission:
    base_url = "http://127.0.0.1:1/v1"
    fatal_error = None

    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []

    async def start(self, initial_model_version: int) -> None:
        self.events.append(("start", initial_model_version))

    async def prepare_model_update(self, model_version: int) -> None:
        self.events.append(("prepare", model_version))

    async def activate_model_version(self, model_version: int) -> None:
        self.events.append(("activate", model_version))

    async def aclose(self) -> None:
        self.events.append(("close", None))


class _ControlledEnvironmentWorkers:
    """Return token-faithful episodes while retaining producer concurrency."""

    def __init__(self, prompt_ids: list[int], completion_ids: tuple[list[int], list[int]]) -> None:
        self._prompt_ids = prompt_ids
        self._completion_ids = completion_ids
        self.versions: list[int] = []

    async def start(self, activation: Any, client_config: Any, execution_config: Any) -> None:
        del activation, client_config, execution_config

    async def open_run_admission(self, run_id: str) -> None:
        del run_id

    async def stop_run_admission(self, run_id: str) -> None:
        del run_id

    async def run_episode(self, key: Any, task: Any, deadline: float) -> EpisodeOutcome:
        del task, deadline
        version = int(key.collection.policy_version)
        self.versions.append(version)
        completion_ids = self._completion_ids[key.rollout_ordinal]
        rollout = EnvironmentRollout(
            example_id=key.example_id,
            prompt_ids=tuple(self._prompt_ids),
            completion_ids=tuple(completion_ids),
            sampling_logprobs=(-0.5,) * len(completion_ids),
            env_mask=(True,) * len(completion_ids),
            reward=float(key.rollout_ordinal),
            is_truncated=False,
            trace=TraceObservation("verifiers", key.occurrence_id, {}),
            behavior_policy=BehaviorPolicySpan(version, version),
        )
        return EpisodeOutcome(key, EpisodeStatus.COMPLETED, rollout=rollout)

    async def cancel(self, key: Any) -> bool:
        del key
        return False

    async def aclose(self) -> None:
        return None


class _Bridge:
    dataset = SimpleNamespace(examples=(SimpleNamespace(id="controlled-task"),))

    def task_for_example_id(self, example_id: str) -> Any:
        return SimpleNamespace(id=example_id)


class _RecordingProducer(VerifiersAsyncGroupProducer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.consumed_group_ids: list[int] = []

    async def acknowledge_consumed_samples(self, group_ids: Any) -> None:
        await super().acknowledge_consumed_samples(group_ids)
        self.consumed_group_ids.extend(group_ids)


class _NoopWeightTransfer:
    """Exercise trainer publication ordering without requiring a second GPU."""

    def init_weight_transfer(self) -> None:
        return None

    def pause(self) -> None:
        return None

    def send_weights(self, iterator: Any) -> None:
        for _ in iterator:
            pass

    def resume(self) -> None:
        return None

    def destroy(self) -> None:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trl-internal-testing/small-Qwen2ForCausalLM-2.5")
    parser.add_argument("--output-dir", default="outputs/local-async-verifiers-update")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = [{"role": "user", "content": "Choose A or B."}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    completion_ids = (
        tokenizer.encode(" A", add_special_tokens=False),
        tokenizer.encode(" B", add_special_tokens=False),
    )
    if not isinstance(prompt_ids, list) or not all(completion_ids):
        raise RuntimeError("qualification tokenizer did not produce the required token sequences")

    settings = GRPOSettings(
        id="local-async-verifiers-update",
        loop=TrainingLoop(max_steps=1, per_device_batch_size=2),
        num_prompts_per_step=1,
        num_generations=2,
    )
    policy = _ControlledPolicyAdmission()
    environment_workers = _ControlledEnvironmentWorkers(prompt_ids, completion_ids)
    # TRL performs a cold publication before starting the worker, so version 1
    # is the first version that can legally serve a rollout.
    producer = _RecordingProducer(
        workers=environment_workers,
        policy_admission=policy,
        bridge=_Bridge(),
        activation={"id": "controlled-local-qualification"},
        client_config_factory=lambda base_url: SimpleNamespace(base_url=base_url),
        execution_config=RolloutExecutionConfig(env_workers=1, episodes_per_worker=2),
        settings=settings,
        run_id="local-async-verifiers-update",
        initial_model_version=1,
        episode_timeout_s=30,
        max_group_tokens=256,
        seed=7,
    )
    rollout_worker = TrlAsyncRolloutWorker(
        producer,
        initial_model_version=0,
        max_inflight_groups=1,
        queue_maxsize=4,
        shutdown_timeout_s=30,
    )
    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        learning_rate=0.1,
        per_device_train_batch_size=2,
        num_generations=2,
        max_steps=1,
        max_completion_length=8,
        # Fixed-count batching makes this recovery gate consume exactly one
        # complete two-sibling group in its single optimizer update.
        token_budget=-1,
        weight_sync_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=Dataset.from_list([{"prompt": prompt, "example_id": "controlled-task"}]),
        processing_class=tokenizer,
        rollout_worker=rollout_worker,
        weight_transfer=_NoopWeightTransfer(),
    )
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainer.model.named_parameters()}
    result = trainer.train()
    changed = sum(
        not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in trainer.model.named_parameters()
    )
    if result.global_step != 1 or changed == 0:
        raise RuntimeError("async qualification did not perform one parameter-changing optimizer update")
    if producer.consumed_group_ids != [0, 0]:
        raise RuntimeError(f"learner consumption acknowledgement was incorrect: {producer.consumed_group_ids}")
    print(
        json.dumps(
            {
                "changed_parameter_tensors": changed,
                "consumed_group_ids": producer.consumed_group_ids,
                "global_step": result.global_step,
                "policy_events": policy.events,
                "served_policy_versions": environment_workers.versions,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
