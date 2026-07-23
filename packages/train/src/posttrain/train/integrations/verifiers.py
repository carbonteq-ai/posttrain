"""Native Verifiers v1 episode bridge for environment-driven online RL."""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Protocol

from posttrain.common import JsonValue, LocalArtifactRef, ProducedArtifact, TraceObservation
from posttrain.data import RolloutDataset, RolloutExample

from ..online_rl import EnvironmentRollout, PolicyGenerator, PolicySampling, PolicyTurnRequest, RolloutBatch


class TraceEnricher(Protocol):
    def __call__(self, trace: Any) -> None: ...


type EnvironmentFactory = Callable[[], Any]


class EnvironmentSourceSelection(Protocol):
    @property
    def package(self) -> str: ...

    @property
    def revision(self) -> str: ...


class VerifiersEnvironmentSelection(Protocol):
    """Structural environment binding consumed without importing posttrain.eval."""

    @property
    def id(self) -> str: ...

    @property
    def source(self) -> EnvironmentSourceSelection: ...

    @property
    def factory(self) -> EnvironmentFactory: ...

    @property
    def num_tasks(self) -> int: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    @property
    def revision(self) -> str: ...


@dataclass(frozen=True, slots=True)
class NativeVerifiersEnvironmentFactory:
    """Pickle-safe reconstruction of a validated native Verifiers environment."""

    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", dict(self.config))

    def __call__(self) -> Any:
        EnvConfig, Environment = _environment_imports()
        return Environment(EnvConfig.model_validate(dict(self.config)))


@dataclass(frozen=True, slots=True)
class VerifiersBridgeSnapshot:
    """Pickle-safe reconstruction state for isolated trainer workers."""

    dataset_id: str
    revision: str
    tasks: Mapping[int, Any]
    environment_factory: EnvironmentFactory
    trace_path: Path
    environment_id: str
    run_id: str
    sampling: PolicySampling
    enrichers: tuple[TraceEnricher, ...]

    def create(self) -> VerifiersEnvironmentRolloutBridge:
        return VerifiersEnvironmentRolloutBridge(
            dataset_id=self.dataset_id,
            revision=self.revision,
            tasks=self.tasks,
            environment_factory=self.environment_factory,
            trace_path=self.trace_path,
            environment_id=self.environment_id,
            run_id=self.run_id,
            sampling=self.sampling,
            enrichers=self.enrichers,
        )


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


def _environment_imports() -> tuple[type[Any], type[Any]]:
    try:
        from verifiers.v1.env import EnvConfig, Environment  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    return EnvConfig, Environment


def preflight_verifiers_environment(environment: VerifiersEnvironmentSelection) -> Mapping[str, Any]:
    """Check an installed environment binding and return portable native config."""

    EnvConfig, Environment = _environment_imports()
    base = environment.factory()
    if isinstance(base, Environment):
        base = base.config
    if not isinstance(base, EnvConfig):
        raise TypeError("Verifiers environment factories must return verifiers.v1.EnvConfig")
    payload = base.model_dump(mode="python")
    _apply_training_parameters(environment, payload)
    config = EnvConfig.model_validate(payload)
    Environment(config)
    return config.model_dump(mode="python")


def create_verifiers_training_bridge(
    environment: VerifiersEnvironmentSelection,
    trace_path: Path,
    run_id: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    purpose: str = "grpo",
    tasks: Mapping[int, Any] | None = None,
) -> VerifiersEnvironmentRolloutBridge:
    """Build the existing native bridge from a public environment selection."""

    if not purpose or "/" in purpose:
        raise ValueError("Verifiers bridge purpose must be one stable path segment")
    config = preflight_verifiers_environment(environment)
    native_factory = NativeVerifiersEnvironmentFactory(config)
    selected = dict(tasks) if tasks is not None else _load_selected_tasks(environment, native_factory)
    if not selected:
        raise ValueError("Verifiers training bridge requires at least one selected task")
    if len(selected) != environment.num_tasks:
        raise ValueError(
            f"environment {environment.id!r} requests {environment.num_tasks} tasks, "
            f"but the bridge received {len(selected)}"
        )
    seed = _sampling_seed(environment)
    return VerifiersEnvironmentRolloutBridge(
        dataset_id=f"{environment.id}/{purpose}/seed-{seed}-limit-{len(selected)}",
        revision=environment.source.revision,
        tasks=selected,
        environment_factory=native_factory,
        trace_path=trace_path,
        environment_id=environment.id,
        run_id=run_id,
        sampling=PolicySampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p),
    )


def _apply_training_parameters(
    environment: VerifiersEnvironmentSelection,
    payload: dict[str, Any],
) -> None:
    parameters = environment.parameters
    for key in ("max_turns", "max_total_tokens"):
        value = parameters.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            payload[key] = value
    rollout_timeout = parameters.get("rollout_timeout_seconds")
    if isinstance(rollout_timeout, int | float) and not isinstance(rollout_timeout, bool):
        timeout = payload.setdefault("timeout", {})
        if isinstance(timeout, dict):
            timeout["rollout"] = float(rollout_timeout)
    if environment.source.package != "automationbench-v1":
        return
    taskset = payload.setdefault("taskset", {})
    if not isinstance(taskset, dict):
        raise TypeError("Verifiers taskset config must be an object")
    domains = parameters.get("domains")
    if isinstance(domains, (tuple, list)) and all(isinstance(value, str) for value in domains):
        taskset["domains"] = list(domains)
    task = taskset.setdefault("task", {})
    if not isinstance(task, dict):
        raise TypeError("AutomationBench task config must be an object")
    for key in ("toolset", "search_top_k"):
        value = parameters.get(key)
        if isinstance(value, str | int) and not isinstance(value, bool):
            task[key] = value


def _sampling_seed(environment: VerifiersEnvironmentSelection) -> int:
    seed = environment.parameters.get("sampling_seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("environment sampling_seed must be a non-negative integer")
    return seed


def _load_selected_tasks(
    environment: VerifiersEnvironmentSelection,
    factory: NativeVerifiersEnvironmentFactory,
) -> dict[int, Any]:
    available = factory().taskset.load()
    size = len(available)
    if environment.num_tasks > size:
        raise ValueError(
            f"environment {environment.id!r} requests {environment.num_tasks} tasks, "
            f"but the installed taskset exposes {size}"
        )
    indices = sorted(Random(_sampling_seed(environment)).sample(range(size), environment.num_tasks))
    return {index: available[index] for index in indices}


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
class VerifiersEnvironmentRolloutBridge:
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

    async def run(self, batch: RolloutBatch, generator: PolicyGenerator) -> Sequence[EnvironmentRollout]:
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
        by_id: dict[str, list[EnvironmentRollout]] = {}
        async with self._environment.serving():
            for example_id in order:
                try:
                    task_index, task = self._tasks_by_example_id[example_id]
                except KeyError as error:
                    raise ValueError(f"unknown rollout example {example_id!r}") from error
                traces = await self._environment.episode(task, context, n=counts[example_id]).run()
                projected: list[EnvironmentRollout] = []
                for trace in traces:
                    trace.stamp(
                        run=TrainRunInfo(id=self.run_id, step=batch.step),
                        environment_id=self.environment_id,
                        task_index=task_index,
                        example_id=example_id,
                    )
                    for enrich in self.enrichers:
                        enrich(trace)
                    self._preserve(trace.to_record())
                    projected.append(self._project(trace, example_id, task_index))
                by_id[example_id] = projected
        positions = {identifier: 0 for identifier in by_id}
        aligned: list[EnvironmentRollout] = []
        for example_id in batch.example_ids:
            position = positions[example_id]
            aligned.append(by_id[example_id][position])
            positions[example_id] = position + 1
        return aligned

    def _project(self, trace: Any, example_id: str, task_index: int) -> EnvironmentRollout:
        branches = trace.branches
        if len(branches) != 1:
            error = trace.error
            detail = f"; trace error={error.type}: {error.message}" if error is not None else ""
            raise ValueError(f"online-RL MVP requires one trainable trace branch, got {len(branches)}{detail}")
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
        return EnvironmentRollout(
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
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - Windows is not a qualified veRL target
                    fcntl = None
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    stream.write(encoded)
                    stream.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            self._trace_count += 1

    def write_portable_snapshot(self, path: Path) -> None:
        """Serialize trusted reconstruction state for an isolated veRL/Ray runtime."""

        snapshot = VerifiersBridgeSnapshot(
            dataset_id=self.dataset_id,
            revision=self.revision,
            tasks=self.tasks,
            environment_factory=self.environment_factory,
            trace_path=self.trace_path,
            environment_id=self.environment_id,
            run_id=self.run_id,
            sampling=self.sampling,
            enrichers=self.enrichers,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        if not self.trace_path.is_file():
            return ()
        with self.trace_path.open("r", encoding="utf-8") as stream:
            preserved_trace_count = sum(1 for line in stream if line.strip())
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
                "trace_count": preserved_trace_count,
                "schema_version": 2,
            },
        )
        return (artifact,)


def load_verifiers_bridge_snapshot(path: Path) -> VerifiersEnvironmentRolloutBridge:
    """Load a trusted bridge snapshot created by this package."""

    with path.open("rb") as stream:
        snapshot = pickle.load(stream)  # noqa: S301 - internal trusted artifact, never user supplied
    if not isinstance(snapshot, VerifiersBridgeSnapshot):
        raise TypeError("portable Verifiers bridge snapshot has an incompatible schema")
    return snapshot.create()


__all__ = [
    "EnvironmentSourceSelection",
    "NativeVerifiersEnvironmentFactory",
    "TraceEnricher",
    "VerifiersBridgeSnapshot",
    "VerifiersEnvironmentSelection",
    "VerifiersEnvironmentRolloutBridge",
    "create_verifiers_training_bridge",
    "load_verifiers_bridge_snapshot",
    "preflight_verifiers_environment",
]
