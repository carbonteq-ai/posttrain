"""Native Verifiers v1 episode bridge for environment-driven online RL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import pickle
import statistics
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Literal, Protocol

from posttrain.common import (
    JsonValue,
    LocalArtifactRef,
    MetricBatchObservation,
    ProducedArtifact,
    TraceObservation,
)
from posttrain.data import RolloutDataset, RolloutExample

from ..online_rl import (
    AgenticTurn,
    AsyncTerminalTraceObserver,
    EnvironmentRollout,
    EnvironmentRolloutEvidence,
    EnvironmentSampling,
    PolicyGenerator,
    PolicySampling,
    PolicyTurnRequest,
    RolloutBatch,
)

type OnlineRLTechnique = Literal["grpo", "dapo", "olmo3", "sampo", "distill"]


class VerifiersRolloutFailure(RuntimeError):
    """A terminal environment trace that cannot safely become a training sample."""

_NULL_HARNESS_UNBOUNDED_MCP = '"mcp"'
_NULL_HARNESS_MCP_V1 = '"mcp>=1.24.0,<2"'


def _apply_verifiers_runtime_compatibility() -> None:
    """Keep the pinned Verifiers null harness on its compatible MCP major."""

    try:
        from verifiers.v1.harnesses.null import harness as null_harness  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    source = null_harness.PROGRAM_SOURCE
    if _NULL_HARNESS_MCP_V1 in source:
        return
    if _NULL_HARNESS_UNBOUNDED_MCP not in source:
        raise RuntimeError(
            "the pinned Verifiers null harness dependency declaration changed; update the Posttrain compatibility guard"
        )
    null_harness.PROGRAM_SOURCE = source.replace(
        _NULL_HARNESS_UNBOUNDED_MCP,
        _NULL_HARNESS_MCP_V1,
        1,
    )


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

    def activate(self) -> Any: ...

    @property
    def num_tasks(self) -> int: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    @property
    def revision(self) -> str: ...

    @property
    def max_concurrent(self) -> int: ...

    @property
    def sampling(self) -> EnvironmentSampling: ...

    @property
    def observation(self) -> Any: ...


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
    max_concurrent: int | None
    technique: OnlineRLTechnique
    enrichers: tuple[TraceEnricher, ...]
    task_facet_fields: tuple[str, ...] = ()

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
            max_concurrent=self.max_concurrent,
            technique=self.technique,
            enrichers=self.enrichers,
            task_facet_fields=self.task_facet_fields,
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
    _apply_verifiers_runtime_compatibility()
    try:
        from verifiers.v1.env import EnvConfig, Environment  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    return EnvConfig, Environment


def preflight_verifiers_environment(environment: VerifiersEnvironmentSelection) -> Mapping[str, Any]:
    """Check an installed environment binding and return portable native config."""

    EnvConfig, Environment = _environment_imports()
    base = environment.activate()
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
    sampling: PolicySampling,
    purpose: OnlineRLTechnique = "grpo",
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
        sampling=sampling,
        max_concurrent=getattr(environment, "max_concurrent", None),
        technique=purpose,
        task_facet_fields=_task_facet_fields(environment),
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
    taskset = factory().taskset
    if bool(getattr(type(taskset), "INFINITE", False)):
        selected = tuple(
            taskset.select(
                num_tasks=environment.num_tasks,
                shuffle=False,
            )
        )
        if len(selected) != environment.num_tasks:
            raise ValueError(
                f"environment {environment.id!r} requests {environment.num_tasks} tasks, "
                f"but its infinite taskset selected {len(selected)}"
            )
        indexed: dict[int, Any] = {}
        for task in selected:
            index = getattr(getattr(task, "data", None), "idx", None)
            if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in indexed:
                raise ValueError(
                    "infinite Verifiers tasksets must select tasks with unique non-negative integer data.idx values"
                )
            indexed[index] = task
        return dict(sorted(indexed.items()))

    # Native tasksets are allowed to return any iterable.  In particular,
    # Reasoning Gym exposes a generator here rather than a sized sequence.
    # Materialize it once so selection remains deterministic and the
    # cardinality/indexing checks below apply uniformly to finite tasksets.
    available = tuple(taskset.load())
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
    task_facet_fields: tuple[str, ...] = (),
) -> RolloutDataset:
    """Project native Verifiers tasks into task-neutral prompts and stable keys."""

    examples = tuple(
        RolloutExample(
            id=f"train/{index:06d}",
            prompt=str(task.data.prompt),
            metadata={
                "task_index": index,
                "environment_id": environment_id,
                **_task_facet_values(task, task_facet_fields),
            },
        )
        for index, task in sorted(tasks.items())
    )
    return RolloutDataset(identifier, revision, examples)


def _task_facet_fields(environment: VerifiersEnvironmentSelection) -> tuple[str, ...]:
    observation = getattr(environment, "observation", None)
    facets = getattr(observation, "facets", ())
    fields = tuple(str(facet.field) for facet in facets)
    if any(not field for field in fields):
        raise ValueError("environment observation facets must declare non-empty task fields")
    return tuple(dict.fromkeys(fields))


def _task_facet_values(task: Any, fields: tuple[str, ...]) -> dict[str, JsonValue]:
    data = getattr(task, "data", None)
    values: dict[str, JsonValue] = {}
    for name in fields:
        value = data.get(name) if isinstance(data, Mapping) else getattr(data, name, None)
        if isinstance(value, str | int | bool) or (isinstance(value, float) and math.isfinite(value)):
            values[name] = value
            continue
        raise ValueError(f"task data does not expose scalar observation facet {name!r}")
    return values


def _record_task_facets(info: Mapping[str, object]) -> dict[str, JsonValue]:
    raw = info.get("task_facets")
    if not isinstance(raw, Mapping):
        return {}
    values: dict[str, JsonValue] = {}
    for name, value in raw.items():
        if isinstance(name, str) and (
            isinstance(value, str | int | bool) or (isinstance(value, float) and math.isfinite(value))
        ):
            values[name] = value
    return values


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
        min_p = getattr(sampling_args, "min_p", None)
        request = PolicyTurnRequest(
            messages=tuple(_record(message) for message in messages),
            sampling=PolicySampling(
                max_tokens=int(sampling_args.max_tokens),
                temperature=1.0 if sampling_args.temperature is None else float(sampling_args.temperature),
                top_p=1.0 if sampling_args.top_p is None else float(sampling_args.top_p),
                top_k=int(getattr(sampling_args, "top_k", 0)),
                min_p=None if min_p is None else float(min_p),
                repetition_penalty=float(getattr(sampling_args, "repetition_penalty", 1.0)),
                presence_penalty=float(getattr(sampling_args, "presence_penalty", 0.0)),
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
    max_concurrent: int | None = None
    technique: OnlineRLTechnique = "grpo"
    enrichers: tuple[TraceEnricher, ...] = ()
    task_facet_fields: tuple[str, ...] = ()
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _trace_count: int = field(default=0, init=False)
    _live_observed_trace_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _requested_by_step: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _dataset: RolloutDataset = field(init=False, repr=False)
    _tasks_by_example_id: dict[str, tuple[int, Any]] = field(init=False, repr=False)
    _environment: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dataset = _rollout_dataset(
            self.dataset_id,
            self.revision,
            self.environment_id,
            self.tasks,
            self.task_facet_fields,
        )
        self._tasks_by_example_id = {
            example.id: (index, self.tasks[index])
            for index, example in zip(sorted(self.tasks), self._dataset.examples, strict=True)
        }
        self._environment = self.environment_factory()

    @property
    def dataset(self) -> RolloutDataset:
        return self._dataset

    async def run(self, batch: RolloutBatch, generator: PolicyGenerator) -> Sequence[EnvironmentRollout]:
        return await self._run(batch, generator, on_completed=None)

    async def run_observed(
        self,
        batch: RolloutBatch,
        generator: PolicyGenerator,
        *,
        on_completed: AsyncTerminalTraceObserver,
    ) -> Sequence[EnvironmentRollout]:
        return await self._run(batch, generator, on_completed=on_completed)

    async def _run(
        self,
        batch: RolloutBatch,
        generator: PolicyGenerator,
        *,
        on_completed: AsyncTerminalTraceObserver | None,
    ) -> Sequence[EnvironmentRollout]:
        with self._write_lock:
            self._requested_by_step[batch.step] = self._requested_by_step.get(batch.step, 0) + len(batch.example_ids)
        _, _, ModelContext, _, Sampling, TrainRunInfo, _, _, _, _ = _imports()
        client = _PolicyClient(generator)
        context = ModelContext(
            model=batch.model_id,
            client=client,
            sampling=Sampling(
                max_tokens=self.sampling.max_tokens,
                temperature=self.sampling.temperature,
                top_p=self.sampling.top_p,
                top_k=self.sampling.top_k,
                min_p=self.sampling.min_p,
                repetition_penalty=self.sampling.repetition_penalty,
                presence_penalty=self.sampling.presence_penalty,
            ),
        )
        counts = Counter(batch.example_ids)
        if sum(counts.values()) != len(batch.example_ids):
            raise AssertionError("rollout identity accounting failed")

        # ``Environment.episode(..., n=G).run()`` starts all G branches for a
        # task concurrently.  Calling it once for every prompt group would
        # therefore turn a 32-group/G=8 update into 256 live subprocesses even
        # when the catalog declares max_concurrent=32.  Schedule one branch per
        # task occurrence so the declared bound covers the full rollout
        # population, not only the number of prompt groups.
        limit = self.max_concurrent
        if limit is not None and limit < 1:
            raise ValueError("Verifiers bridge max_concurrent must be positive")
        worker_count = min(limit or len(batch.example_ids), len(batch.example_ids))
        if worker_count == 0:
            return ()
        next_ordinal = 0
        next_lock = asyncio.Lock()
        fatal_error: BaseException | None = None
        fatal_lock = asyncio.Lock()
        results: dict[int, EnvironmentRollout] = {}

        async def run_example(rollout_ordinal: int, example_id: str) -> tuple[int, EnvironmentRollout]:
            try:
                task_index, task = self._tasks_by_example_id[example_id]
            except KeyError as error:
                raise ValueError(f"unknown rollout example {example_id!r}") from error
            traces = await self._environment.episode(task, context, n=1).run()
            if len(traces) != 1:
                raise ValueError(
                    f"Verifiers episode returned {len(traces)} traces for one scheduled branch; expected exactly one"
                )
            trace = traces[0]
            trace.stamp(
                run=TrainRunInfo(id=self.run_id, step=batch.step),
                environment_id=self.environment_id,
                task_index=task_index,
                example_id=example_id,
            )
            for enrich in self.enrichers:
                enrich(trace)
            record, observation = self._terminal_observation(trace, example_id, task_index, rollout_ordinal)
            # Native JSONL is the replay authority.  This happens before any
            # trainable projection, including branch/token/reward validation.
            self._preserve(record)
            if on_completed is not None:
                try:
                    await on_completed(observation)
                except Exception:
                    # A tracking outage must not erase the native evidence or
                    # turn a healthy rollout into a failed training sample.
                    pass
                else:
                    self.mark_live_observed(observation.external_id)
            return rollout_ordinal, self._project(trace, observation)

        occurrences = [example_id for example_id in batch.example_ids]

        async def worker() -> None:
            nonlocal next_ordinal, fatal_error
            while True:
                async with next_lock:
                    if fatal_error is not None or next_ordinal >= len(occurrences):
                        return
                    ordinal = next_ordinal
                    next_ordinal += 1
                try:
                    result_ordinal, rollout = await run_example(ordinal, occurrences[ordinal])
                except BaseException as error:
                    async with fatal_lock:
                        if fatal_error is None:
                            fatal_error = error
                    return
                results[result_ordinal] = rollout

        async with self._environment.serving():
            await asyncio.gather(*(worker() for _ in range(worker_count)))
        if fatal_error is not None:
            if isinstance(fatal_error, VerifiersRolloutFailure):
                raise fatal_error
            raise VerifiersRolloutFailure(f"Verifiers rollout failed: {fatal_error}") from fatal_error
        return [results[ordinal] for ordinal in range(len(occurrences))]

    def _terminal_observation(
        self,
        trace: Any,
        example_id: str,
        task_index: int,
        rollout_ordinal: int,
    ) -> tuple[dict[str, Any], TraceObservation]:
        record = trace.to_record()
        task_facets = _task_facet_values(self.tasks[task_index], self.task_facet_fields)
        info = record.setdefault("info", {})
        if not isinstance(info, dict):
            raise ValueError("Verifiers trace info must be an object")
        info["task_facets"] = task_facets
        return record, self._observation_from_record(
            record,
            example_id=example_id,
            task_index=task_index,
            rollout_ordinal=rollout_ordinal,
        )

    def _observation_from_record(
        self,
        record: Mapping[str, Any],
        *,
        example_id: str | None = None,
        task_index: int | None = None,
        rollout_ordinal: int | None = None,
    ) -> TraceObservation:
        info = record.get("info")
        info = info if isinstance(info, Mapping) else {}
        external_id = str(record.get("id") or "")
        if not external_id:
            raise ValueError("preserved Verifiers traces require stable trace ids")
        attributes: dict[str, JsonValue] = {
            "environment_id": str(info.get("environment_id") or self.environment_id),
            "task_index": int(info.get("task_index", -1) if task_index is None else task_index),
            "example_id": str(info.get("example_id") or "") if example_id is None else example_id,
            "is_truncated": _trace_is_truncated(record),
            "has_error": _trace_has_error(record),
            "model": _trace_model(record),
            **_record_task_facets(info),
        }
        if rollout_ordinal is not None:
            attributes["rollout_ordinal"] = rollout_ordinal
        return TraceObservation(
            trace_type="verifiers",
            external_id=external_id,
            payload=record,
            attributes=attributes,
        )

    def _project(self, trace: Any, observation: TraceObservation) -> EnvironmentRollout:
        if _trace_has_error(observation.payload):
            raise VerifiersRolloutFailure("Verifiers trace terminated with a harness or environment error")
        branches = trace.branches
        if len(branches) != 1:
            error = trace.error
            detail = f"; trace error={error.type}: {error.message}" if error is not None else ""
            raise VerifiersRolloutFailure(
                f"online-RL requires one trainable trace branch, got {len(branches)}{detail}"
            )
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
        if not math.isfinite(float(trace.reward)):
            raise VerifiersRolloutFailure("Verifiers trace has a non-finite scalar reward")
        is_truncated = bool(observation.attributes["is_truncated"])
        turns = _agentic_turns(branch, first_sampled) if self.technique == "sampo" else ()
        attributes = dict(observation.attributes)
        attributes.update(
            completion_token_count=len(completion_ids),
            selected_token_count=sum(env_mask),
        )
        observation = TraceObservation(
            trace_type=observation.trace_type,
            external_id=observation.external_id,
            payload=observation.payload,
            attributes=attributes,
        )
        return EnvironmentRollout(
            example_id=str(observation.attributes["example_id"]),
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            sampling_logprobs=logprobs,
            env_mask=env_mask,
            reward=float(trace.reward),
            is_truncated=is_truncated,
            trace=observation,
            turns=turns,
        )

    def mark_live_observed(self, external_id: str) -> None:
        """Mark a native trace accepted by a live host/provider submission."""

        with self._write_lock:
            self._live_observed_trace_ids.add(external_id)

    def trace_observation(self, record: Mapping[str, Any]) -> TraceObservation:
        """Reconstruct one terminal native record in a host-side observer."""

        return self._observation_from_record(record)

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
            max_concurrent=self.max_concurrent,
            technique=self.technique,
            enrichers=self.enrichers,
            task_facet_fields=self.task_facet_fields,
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
                "technique": self.technique,
                "environment_id": self.environment_id,
                "dataset_id": self.dataset.id,
                "dataset_revision": self.dataset.revision,
                "trace_count": preserved_trace_count,
                "schema_version": 2,
            },
        )
        return (artifact,)

    def evidence(self) -> EnvironmentRolloutEvidence:
        """Replay native trace records and trace-derived metrics in the host process."""

        with self._write_lock:
            live_observed_trace_ids = frozenset(self._live_observed_trace_ids)
            requested_by_step = dict(self._requested_by_step)
        if not self.trace_path.is_file():
            return EnvironmentRolloutEvidence(
                metrics=tuple(
                    MetricBatchObservation(
                        _trace_metrics((), requested=requested),
                        step=step,
                        attributes={"observation_source": "verifiers"},
                    )
                    for step, requested in sorted(requested_by_step.items())
                )
            )
        records = [
            json.loads(line) for line in self.trace_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        traces: list[TraceObservation] = []
        records_by_step: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("preserved Verifiers traces must be JSON objects")
            run = record.get("run")
            if not isinstance(run, dict) or not isinstance(run.get("step"), int):
                raise ValueError("preserved Verifiers traces require an integer run step")
            external_id = str(record.get("id") or "")
            if not external_id:
                raise ValueError("preserved Verifiers traces require stable trace ids")
            if external_id not in live_observed_trace_ids:
                traces.append(self._observation_from_record(record))
            records_by_step.setdefault(run["step"], []).append(record)
        metrics = tuple(
            MetricBatchObservation(
                _trace_metrics(step_records, requested=requested_by_step.get(step)),
                step=step,
                attributes={"observation_source": "verifiers"},
            )
            for step in sorted(set(records_by_step) | set(requested_by_step))
            for step_records in (records_by_step.get(step, []),)
        )
        return EnvironmentRolloutEvidence(metrics=metrics, traces=tuple(traces))


def _trace_model(record: Mapping[str, Any]) -> str:
    agent = record.get("agent")
    return str(agent.get("model") or "") if isinstance(agent, Mapping) else ""


def _trace_reward(record: Mapping[str, Any]) -> float | None:
    if _trace_has_error(record):
        return None
    rewards = record.get("rewards")
    if not isinstance(rewards, Mapping):
        return None
    values = [
        float(value) for value in rewards.values() if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    total = sum(values) if values else None
    return total if total is not None and math.isfinite(total) else None


def _trace_has_error(record: Mapping[str, Any]) -> bool:
    errors = record.get("errors")
    return isinstance(errors, list) and bool(errors)


def _trace_is_truncated(record: Mapping[str, Any]) -> bool:
    # A harness/environment error is terminal evidence, but it is not a model
    # completion that reached a generation boundary.  Keeping it separate
    # prevents failure counts from being silently folded into truncation rate.
    if _trace_has_error(record):
        return False
    if record.get("stop_condition") in {
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
        "harness_timeout",
    }:
        return True
    calls = record.get("calls")
    if isinstance(calls, list):
        last_successful_call = next(
            (call for call in reversed(calls) if isinstance(call, Mapping) and not call.get("error")),
            None,
        )
        if last_successful_call is not None and last_successful_call.get("finish_reason") == "length":
            return True
    return not bool(record.get("is_completed"))


def _trace_has_tool_call(record: Mapping[str, Any]) -> bool:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        return False
    return any(
        isinstance(node, Mapping)
        and bool(node.get("sampled"))
        and isinstance((message := node.get("message")), Mapping)
        and bool(message.get("tool_calls"))
        for node in nodes
    )


def _trace_has_tool_failure(record: Mapping[str, Any]) -> bool:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        return False
    for node in nodes:
        if not isinstance(node, Mapping) or bool(node.get("sampled")):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and value.get("success") is False:
            return True
    return False


def _trace_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    requested: int | None = None,
) -> dict[str, float]:
    attempted = len(records)
    if attempted == 0 and requested is None:
        raise ValueError("cannot derive rollout evidence from an empty step")
    rewards = [reward for record in records if (reward := _trace_reward(record)) is not None]
    grouped_rewards: dict[str, list[float]] = {}
    for record in records:
        reward = _trace_reward(record)
        info = record.get("info")
        example_id = str(info.get("example_id") or "") if isinstance(info, Mapping) else ""
        if reward is not None:
            grouped_rewards.setdefault(example_id, []).append(reward)
    grouped = [values for values in grouped_rewards.values() if len(values) > 1]
    values = {
        "train/rl/rollouts_requested": float(requested if requested is not None else attempted),
        "train/rl/rollouts_attempted": float(attempted),
        "train/rl/rollouts_completed": float(sum(bool(record.get("is_completed")) for record in records)),
        "train/rl/rollouts_failed": float(sum(_trace_has_error(record) for record in records)),
        "train/rl/rollouts_truncated": float(sum(_trace_is_truncated(record) for record in records)),
        "train/rl/rollouts_unscorable": float(attempted - len(rewards)),
    }
    if attempted:
        values["train/rl/tool_call_frequency"] = sum(_trace_has_tool_call(record) for record in records) / attempted
        values["train/rl/tool_failure_frequency"] = sum(_trace_has_tool_failure(record) for record in records) / attempted
    if requested is not None:
        values["train/rl/rollouts_missing"] = float(max(requested - attempted, 0))
    # Missing reward variation is evidence that no valid training population
    # existed, not a measured zero.  Do not fabricate a zero-variance signal.
    if rewards:
        values["train/rl/reward_std"] = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if grouped:
        values["train/rl/group_zero_variance_fraction"] = sum(
            statistics.pstdev(group) == 0 for group in grouped
        ) / len(grouped)
    return values


def load_verifiers_bridge_snapshot(path: Path) -> VerifiersEnvironmentRolloutBridge:
    """Load a trusted bridge snapshot created by this package."""

    with path.open("rb") as stream:
        snapshot = pickle.load(stream)  # noqa: S301 - internal trusted artifact, never user supplied
    if not isinstance(snapshot, VerifiersBridgeSnapshot):
        raise TypeError("portable Verifiers bridge snapshot has an incompatible schema")
    return snapshot.create()


def _agentic_turns(branch: Any, first_sampled: int) -> tuple[AgenticTurn, ...]:
    """Project sampled assistant nodes into flattened completion spans."""

    turns: list[AgenticTurn] = []
    offset = 0
    latest_observation: Mapping[str, JsonValue] | None = None
    for node in branch.nodes:
        record = _record(node.message)
        role = record.get("role")
        node_ids = tuple(int(value) for value in node.token_ids)
        node_mask = tuple(bool(value) for value in node.mask)
        if len(node_ids) != len(node_mask):
            raise ValueError("Verifiers message-node token ids and mask are misaligned")
        sampled_positions = [index for index, selected in enumerate(node_mask) if selected]
        if bool(getattr(node, "sampled", False)):
            if role != "assistant" or not sampled_positions:
                raise ValueError("SAMPO sampled nodes must be assistant turns with sampled tokens")
            expected = list(range(sampled_positions[0], sampled_positions[-1] + 1))
            if sampled_positions != expected:
                raise ValueError("SAMPO requires each sampled assistant turn to be one contiguous token span")
            if latest_observation is None:
                raise ValueError("SAMPO sampled assistant turns require a preceding user or tool observation")
            start = offset + sampled_positions[0] - first_sampled
            end = offset + sampled_positions[-1] + 1 - first_sampled
            turns.append(
                AgenticTurn(
                    completion_start=start,
                    completion_end=end,
                    anchor_state_key=_anchor_state_key(latest_observation),
                )
            )
        elif role in {"user", "tool"}:
            latest_observation = record
        offset += len(node_ids)
    if not turns:
        raise ValueError("SAMPO Verifiers trace has no sampled assistant turns")
    return tuple(turns)


def _anchor_state_key(observation: Mapping[str, JsonValue]) -> str:
    encoded = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


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
