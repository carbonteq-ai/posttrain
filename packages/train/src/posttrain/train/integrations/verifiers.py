"""Native Verifiers v1 episode bridge for environment-driven online RL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import statistics
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Literal, Protocol, cast

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
    EnvironmentRollout,
    EnvironmentRolloutEvidence,
    PolicyGenerator,
    PolicySampling,
    PolicyTurnRequest,
    RolloutBatch,
)

type OnlineRLTechnique = Literal["grpo", "dapo", "olmo3", "sampo", "distill"]

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
    max_tokens: int,
    temperature: float,
    top_p: float,
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
        sampling=PolicySampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p),
        max_concurrent=getattr(environment, "max_concurrent", None),
        technique=purpose,
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

    available = taskset.load()
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
        response_format = body.get("response_format")
        if response_format is not None and not isinstance(response_format, Mapping):
            raise TypeError("environment response_format must be a JSON object")
        generation_contract = _posttrain_generation_contract(body)
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
            response_format=(
                cast(Mapping[str, JsonValue], dict(response_format))
                if isinstance(response_format, Mapping)
                else None
            ),
            max_prompt_tokens=generation_contract.get("prompt_limit"),
            max_sequence_tokens=generation_contract.get("sequence_limit"),
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


def _posttrain_generation_contract(body: Mapping[str, Any]) -> dict[str, int]:
    value = body.get("posttrain_generation")
    if value is None:
        extra_body = body.get("extra_body")
        value = extra_body.get("posttrain_generation") if isinstance(extra_body, Mapping) else None
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("posttrain_generation must be a JSON object")
    result: dict[str, int] = {}
    for key in ("prompt_limit", "sequence_limit"):
        raw = value.get(key)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
            raise ValueError(f"posttrain_generation.{key} must be a positive integer")
        result[key] = raw
    return result


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
        semaphore = asyncio.Semaphore(limit) if limit is not None else None

        async def run_example(example_id: str) -> tuple[str, int, Sequence[Any]]:
            try:
                task_index, task = self._tasks_by_example_id[example_id]
            except KeyError as error:
                raise ValueError(f"unknown rollout example {example_id!r}") from error
            if semaphore is None:
                traces = await self._environment.episode(task, context, n=1).run()
            else:
                async with semaphore:
                    traces = await self._environment.episode(task, context, n=1).run()
            if len(traces) != 1:
                raise ValueError(
                    f"Verifiers episode returned {len(traces)} traces for one scheduled branch; expected exactly one"
                )
            return example_id, task_index, traces

        occurrences = [example_id for example_id in batch.example_ids]
        async with self._environment.serving():
            results = await asyncio.gather(*(run_example(example_id) for example_id in occurrences))
        for example_id, task_index, traces in results:
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
            by_id.setdefault(example_id, []).extend(projected)
        positions = {identifier: 0 for identifier in by_id}
        aligned: list[EnvironmentRollout] = []
        for example_id in batch.example_ids:
            position = positions[example_id]
            aligned.append(by_id[example_id][position])
            positions[example_id] = position + 1
        return aligned

    def _project(self, trace: Any, example_id: str, task_index: int) -> EnvironmentRollout:
        branches = trace.branches
        selected = _selected_opd_branch(trace, branches)
        if selected is None and len(branches) != 1:
            error = trace.error
            detail = f"; trace error={error.type}: {error.message}" if error is not None else ""
            raise ValueError(f"online-RL MVP requires one trainable trace branch, got {len(branches)}{detail}")
        branch_index, branch, selected_node = selected or (0, branches[0], None)
        token_ids, sampled_mask, logprobs = _selected_training_sequence(branch, selected_node)
        if len(token_ids) != len(sampled_mask):
            raise ValueError("Verifiers branch token ids and sampled mask are misaligned")
        try:
            first_sampled = sampled_mask.index(True)
        except ValueError as error:
            raise ValueError("Verifiers trace has no model-sampled tokens") from error
        prompt_ids = token_ids[:first_sampled]
        completion_ids = token_ids[first_sampled:]
        env_mask = sampled_mask[first_sampled:]
        completion_logprobs = tuple(float(value) for value in logprobs[first_sampled:])
        if len(completion_logprobs) != len(completion_ids):
            raise ValueError("Verifiers branch logprobs are not aligned to the training sequence")
        record = trace.to_record()
        turns = _agentic_turns(branch, first_sampled) if self.technique == "sampo" else ()
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
                "selected_branch_index": branch_index,
            },
        )
        rollout = EnvironmentRollout(
            example_id=example_id,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            sampling_logprobs=completion_logprobs,
            env_mask=env_mask,
            reward=float(trace.reward),
            is_truncated=(
                False if selected is not None else bool(trace.is_truncated or trace.has_error)
            ),
            trace=observation,
            turns=turns,
        )
        if selected is not None and not all(rollout.env_mask):
            raise ValueError("OPD selected completion contains non-target loss tokens")
        return rollout

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

        if not self.trace_path.is_file():
            return EnvironmentRolloutEvidence()
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
            info = record.get("info")
            info = info if isinstance(info, dict) else {}
            traces.append(
                TraceObservation(
                    trace_type="verifiers",
                    external_id=str(record.get("id") or ""),
                    payload=record,
                    attributes={
                        "environment_id": str(info.get("environment_id") or self.environment_id),
                        "task_index": int(info.get("task_index", -1)),
                        "example_id": str(info.get("example_id") or ""),
                        "is_truncated": _trace_is_truncated(record),
                        "model": _trace_model(record),
                    },
                )
            )
            records_by_step.setdefault(run["step"], []).append(record)
        if any(not trace.external_id for trace in traces):
            raise ValueError("preserved Verifiers traces require stable trace ids")
        metrics = tuple(
            MetricBatchObservation(
                _trace_metrics(step_records),
                step=step,
                attributes={"observation_source": "verifiers"},
            )
            for step, step_records in sorted(records_by_step.items())
        )
        return EnvironmentRolloutEvidence(metrics=metrics, traces=tuple(traces))


def _trace_model(record: Mapping[str, Any]) -> str:
    agent = record.get("agent")
    return str(agent.get("model") or "") if isinstance(agent, Mapping) else ""


def _trace_reward(record: Mapping[str, Any]) -> float | None:
    rewards = record.get("rewards")
    if not isinstance(rewards, Mapping):
        return None
    values = [
        float(value) for value in rewards.values() if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return sum(values) if values else None


def _trace_is_truncated(record: Mapping[str, Any]) -> bool:
    return not bool(record.get("is_completed")) and not bool(record.get("errors"))


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


def _trace_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    attempted = len(records)
    if attempted == 0:
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
    zero_variance_groups = sum(statistics.pstdev(values) == 0 for values in grouped)
    return {
        "train/rl/reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "train/rl/group_zero_variance_fraction": (zero_variance_groups / len(grouped) if grouped else 0.0),
        "train/rl/rollouts_attempted": float(attempted),
        "train/rl/rollouts_completed": float(sum(bool(record.get("is_completed")) for record in records)),
        "train/rl/rollouts_failed": float(sum(bool(record.get("errors")) for record in records)),
        "train/rl/rollouts_truncated": float(sum(_trace_is_truncated(record) for record in records)),
        "train/rl/rollouts_unscorable": float(attempted - len(rewards)),
        "train/rl/tool_call_frequency": sum(_trace_has_tool_call(record) for record in records) / attempted,
        "train/rl/tool_failure_frequency": sum(_trace_has_tool_failure(record) for record in records) / attempted,
    }


def load_verifiers_bridge_snapshot(path: Path) -> VerifiersEnvironmentRolloutBridge:
    """Load a trusted bridge snapshot created by this package."""

    with path.open("rb") as stream:
        snapshot = pickle.load(stream)  # noqa: S301 - internal trusted artifact, never user supplied
    if not isinstance(snapshot, VerifiersBridgeSnapshot):
        raise TypeError("portable Verifiers bridge snapshot has an incompatible schema")
    return snapshot.create()


def _selected_opd_branch(
    trace: Any,
    branches: Sequence[Any],
) -> tuple[int, Any, Any] | None:
    info = trace.info
    policy = info.get("policy_prism") if isinstance(info, Mapping) else None
    marker = policy.get("opd") if isinstance(policy, Mapping) else None
    if not isinstance(marker, Mapping):
        return None
    if marker.get("admission_status") != "accepted":
        raise ValueError("OPD trace target was not admitted")
    branch_index = marker.get("selected_branch_index")
    node_index = marker.get("selected_call_node")
    if not isinstance(branch_index, int) or not 0 <= branch_index < len(branches):
        raise ValueError("OPD selected branch index is invalid")
    if not isinstance(node_index, int) or not 0 <= node_index < len(trace.nodes):
        raise ValueError("OPD selected call node is invalid")
    branch = branches[branch_index]
    selected_node = trace.nodes[node_index]
    if not any(node is selected_node for node in branch.nodes):
        raise ValueError("OPD selected node is absent from the selected branch")
    if not bool(getattr(selected_node, "sampled", False)):
        raise ValueError("OPD selected node is not model sampled")
    prompt_ids, completion_ids = _selected_opd_turn_token_ids(branch, selected_node)
    expected_completion = marker.get("selected_completion_sha256")
    actual_completion = _token_ids_sha256(completion_ids)
    if not isinstance(expected_completion, str) or expected_completion != actual_completion:
        raise ValueError("OPD selected completion digest is missing or inconsistent")
    expected_prompt = marker.get("selected_prompt_sha256")
    if not isinstance(expected_prompt, str) or expected_prompt != _token_ids_sha256(prompt_ids):
        raise ValueError("OPD selected prompt digest is missing or inconsistent")
    return branch_index, branch, selected_node


def _selected_opd_turn_token_ids(branch: Any, selected_node: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a Verifiers assistant node into its prompt scaffold and sampled completion."""
    selected_position = next(
        index for index, node in enumerate(branch.nodes) if node is selected_node
    )
    node_ids = tuple(int(value) for value in selected_node.token_ids)
    node_mask = tuple(bool(value) for value in selected_node.mask)
    if len(node_ids) != len(node_mask):
        raise ValueError("OPD selected node tokens and mask are misaligned")
    sampled_positions = tuple(index for index, value in enumerate(node_mask) if value)
    if not sampled_positions or sampled_positions != tuple(range(sampled_positions[0], len(node_mask))):
        raise ValueError("OPD selected node must end in one contiguous sampled span")
    split = sampled_positions[0]
    prompt_ids = tuple(
        int(token_id)
        for node in branch.nodes[:selected_position]
        for token_id in node.token_ids
    ) + node_ids[:split]
    return prompt_ids, node_ids[split:]


def _selected_training_sequence(
    branch: Any,
    selected_node: Any | None,
) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[float, ...]]:
    token_ids = tuple(int(value) for value in branch.token_ids)
    sampled_mask = tuple(bool(value) for value in branch.sampled_mask)
    logprobs = tuple(float(value) for value in branch.logprobs)
    if selected_node is None:
        return token_ids, sampled_mask, logprobs
    endpoint = 0
    selected_start: int | None = None
    selected_mask: tuple[bool, ...] = ()
    for node in branch.nodes:
        node_ids = tuple(int(value) for value in node.token_ids)
        if node is selected_node:
            selected_start = endpoint
            selected_mask = tuple(bool(value) for value in node.mask)
            endpoint += len(node_ids)
            break
        endpoint += len(node_ids)
    if selected_start is None or not selected_mask or not any(selected_mask):
        raise ValueError("OPD selected node has no sampled token span")
    if len(selected_mask) != endpoint - selected_start:
        raise ValueError("OPD selected node tokens and mask are misaligned")
    target_mask = (False,) * selected_start + selected_mask
    return token_ids[:endpoint], target_mask, logprobs[:endpoint]


def _token_ids_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "big", signed=False))
    return digest.hexdigest()


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
