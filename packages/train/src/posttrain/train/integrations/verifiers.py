"""Native Verifiers v1 episode bridge for environment-driven online RL."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from posttrain.common import JsonValue, LocalArtifactRef, ProducedArtifact, TraceObservation

from ..data import RolloutDataset, RolloutExample
from ..online_rl import PolicyGenerator, PolicySampling, PolicyTurnRequest, RolloutBatch, TrainingRollout


class TraceEnricher(Protocol):
    def __call__(self, trace: Any) -> None: ...


type EnvironmentFactory = Callable[[], Any]


def _imports() -> tuple[Any, ...]:
    try:
        from verifiers.v1 import (  # pyright: ignore[reportMissingImports]
            AssistantMessage,
            Client,
            ModelContext,
            Response,
            Sampling,
            TrainRunInfo,
            TurnTokens,
            Usage,
        )
        from verifiers.v1.dialects import ChatDialect, parse_tools  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    return (
        AssistantMessage,
        Client,
        ModelContext,
        Response,
        Sampling,
        TrainRunInfo,
        TurnTokens,
        Usage,
        ChatDialect,
        parse_tools,
    )


def _rollout_dataset(
    identifier: str,
    revision: str,
    environment_id: str,
    tasks: Mapping[int, Any],
) -> RolloutDataset:
    """Project native Verifiers tasks into task-neutral prompts and stable keys."""

    examples = tuple(
        RolloutExample(
            id=f"train/{index:06d}",
            prompt=str(task.data.prompt),
            metadata={"task_index": index, "environment_id": environment_id},
        )
        for index, task in sorted(tasks.items())
    )
    return RolloutDataset(identifier, revision, examples)


class _PolicyClient:
    """Adapt Verifiers model turns to the platform policy-generator contract."""

    def __init__(self, generator: PolicyGenerator) -> None:
        self._generator = generator

    async def get_response(
        self,
        dialect: Any,
        body: dict[str, Any],
        model: str,
        sampling_args: Any,
        session_id: str | None = None,
        turn: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        del headers
        AssistantMessage, _, _, Response, _, _, TurnTokens, Usage, ChatDialect, parse_tools = _imports()
        if not isinstance(dialect, ChatDialect):
            raise NotImplementedError("the online-RL policy bridge currently supports chat-completions dialects")
        if turn is None:
            messages, tools = dialect.parse_request(body)
            anchor = None
            tail_start = 0
        else:
            messages = turn.prompt
            tools = parse_tools(body.get("tools"))
            anchor = turn.previous_token_ids()
            tail_start = turn.tail_start
        if sampling_args.max_tokens is None:
            raise ValueError("environment policy turns require an explicit max_tokens value")
        request = PolicyTurnRequest(
            messages=tuple(_record(message) for message in messages),
            sampling=PolicySampling(
                max_tokens=int(sampling_args.max_tokens),
                temperature=1.0 if sampling_args.temperature is None else float(sampling_args.temperature),
                top_p=1.0 if sampling_args.top_p is None else float(sampling_args.top_p),
            ),
            tools=tuple(_record(tool) for tool in tools or []),
            session_id=session_id,
            previous_prompt_ids=tuple(anchor[0]) if anchor else (),
            previous_completion_ids=tuple(anchor[1]) if anchor else (),
            tail_start=tail_start,
        )
        result = await self._generator.generate(request)
        response = Response(
            id=str((result.raw_response or {}).get("id", "posttrain-policy-turn")),
            created=0,
            model=model,
            message=AssistantMessage.model_validate(result.message),
            finish_reason=result.finish_reason,
            usage=Usage(prompt_tokens=len(result.prompt_ids), completion_tokens=len(result.completion_ids)),
            tokens=TurnTokens(
                prompt_ids=list(result.prompt_ids),
                completion_ids=list(result.completion_ids),
                completion_logprobs=list(result.completion_logprobs),
                message_spans=list(result.prompt_message_spans) or None,
                is_content=list(result.prompt_is_content) or None,
            ),
        )
        response.raw = dict(result.raw_response or {})
        return response

    async def close(self) -> None:
        return None


def _record(value: Any) -> Mapping[str, JsonValue]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot project {type(value).__name__} into a policy message record")


@dataclass(slots=True)
class VerifiersOnlineRLBridge:
    """Run native Verifiers episodes using an injected, already-loaded policy."""

    dataset_id: str
    revision: str
    tasks: Mapping[int, Any]
    environment_factory: EnvironmentFactory
    trace_path: Path
    environment_id: str
    run_id: str
    sampling: PolicySampling
    enrichers: tuple[TraceEnricher, ...] = ()
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _trace_count: int = field(default=0, init=False)
    _dataset: RolloutDataset = field(init=False, repr=False)
    _tasks_by_example_id: dict[str, tuple[int, Any]] = field(init=False, repr=False)
    _environment: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dataset = _rollout_dataset(self.dataset_id, self.revision, self.environment_id, self.tasks)
        self._tasks_by_example_id = {
            example.id: (index, self.tasks[index])
            for index, example in zip(sorted(self.tasks), self._dataset.examples, strict=True)
        }
        self._environment = self.environment_factory()

    @property
    def dataset(self) -> RolloutDataset:
        return self._dataset

    async def run(self, batch: RolloutBatch, generator: PolicyGenerator) -> Sequence[TrainingRollout]:
        _, _, ModelContext, _, Sampling, TrainRunInfo, _, _, _, _ = _imports()
        counts = Counter(batch.example_ids)
        order = tuple(dict.fromkeys(batch.example_ids))
        if sum(counts.values()) != len(batch.example_ids):
            raise AssertionError("rollout identity accounting failed")
        client = _PolicyClient(generator)
        context = ModelContext(
            model=batch.model_id,
            client=client,
            sampling=Sampling(
                max_tokens=self.sampling.max_tokens,
                temperature=self.sampling.temperature,
                top_p=self.sampling.top_p,
            ),
        )
        by_id: dict[str, list[TrainingRollout]] = {}
        async with self._environment.serving():
            for example_id in order:
                try:
                    task_index, task = self._tasks_by_example_id[example_id]
                except KeyError as error:
                    raise ValueError(f"unknown rollout example {example_id!r}") from error
                traces = await self._environment.episode(task, context, n=counts[example_id]).run()
                projected: list[TrainingRollout] = []
                for trace in traces:
                    trace.stamp(
                        run=TrainRunInfo(id=self.run_id, step=batch.step),
                        environment_id=self.environment_id,
                        task_index=task_index,
                        example_id=example_id,
                    )
                    for enrich in self.enrichers:
                        enrich(trace)
                    projected.append(self._project(trace, example_id, task_index))
                by_id[example_id] = projected
        positions = {identifier: 0 for identifier in by_id}
        aligned: list[TrainingRollout] = []
        for example_id in batch.example_ids:
            position = positions[example_id]
            aligned.append(by_id[example_id][position])
            positions[example_id] = position + 1
        return aligned

    def _project(self, trace: Any, example_id: str, task_index: int) -> TrainingRollout:
        branches = trace.branches
        if len(branches) != 1:
            raise ValueError(f"online-RL MVP requires one trainable trace branch, got {len(branches)}")
        branch = branches[0]
        token_ids = tuple(int(value) for value in branch.token_ids)
        sampled_mask = tuple(bool(value) for value in branch.sampled_mask)
        if len(token_ids) != len(sampled_mask):
            raise ValueError("Verifiers branch token ids and sampled mask are misaligned")
        try:
            first_sampled = sampled_mask.index(True)
        except ValueError as error:
            raise ValueError("Verifiers trace has no model-sampled tokens") from error
        prompt_ids = token_ids[:first_sampled]
        completion_ids = token_ids[first_sampled:]
        env_mask = sampled_mask[first_sampled:]
        logprobs = tuple(float(value) for value in branch.logprobs[first_sampled:])
        if len(logprobs) != len(completion_ids):
            raise ValueError("Verifiers branch logprobs are not aligned to the training sequence")
        record = trace.to_record()
        self._preserve(record)
        observation = TraceObservation(
            trace_type="verifiers",
            external_id=str(trace.id),
            payload=record,
            attributes={
                "environment_id": self.environment_id,
                "task_index": task_index,
                "example_id": example_id,
                "is_truncated": trace.is_truncated,
                "model": trace.agent.model if trace.agent is not None else "",
            },
        )
        return TrainingRollout(
            example_id=example_id,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            sampling_logprobs=logprobs,
            env_mask=env_mask,
            reward=float(trace.reward),
            is_truncated=bool(trace.is_truncated or trace.has_error),
            trace=observation,
        )

    def _preserve(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._write_lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
            self._trace_count += 1

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        if not self.trace_path.is_file():
            return ()
        digest = hashlib.sha256(self.trace_path.read_bytes()).hexdigest()
        artifact = ProducedArtifact(
            name=f"training/rollouts/{self.dataset.id}/verifiers-traces",
            kind="evaluation-traces",
            reference=LocalArtifactRef(self.trace_path.resolve(), digest),
            metadata={
                "technique": "grpo",
                "environment_id": self.environment_id,
                "dataset_id": self.dataset.id,
                "dataset_revision": self.dataset.revision,
                "trace_count": self._trace_count,
                "schema_version": 2,
            },
        )
        return (artifact,)


__all__ = ["TraceEnricher", "VerifiersOnlineRLBridge"]
