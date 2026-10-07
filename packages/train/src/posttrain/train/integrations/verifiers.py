"""Native Verifiers v1 episode bridge for environment-driven online RL."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import shlex
import statistics
import tempfile
import threading
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from typing import Any, Literal, Protocol

from posttrain.common import (
    JsonValue,
    LocalArtifactRef,
    MetricBatchObservation,
    ProducedArtifact,
    SignalSource,
    TraceObservation,
)
from posttrain.data import RolloutDataset, RolloutExample
from posttrain.environment import (
    project_verifiers_trace_facts,
    verifiers_episode_ending,
    verifiers_trace_attributes,
    verifiers_trace_has_error,
    verifiers_trace_is_truncated,
)
from posttrain.environment.verifiers_conditioning import native_conditioning_records

from ..grpo_observations import episode_ending_metrics
from ..online_rl import (
    AgenticTurn,
    AsyncTerminalTraceObserver,
    BehaviorPolicySpan,
    EnvironmentRollout,
    EnvironmentRolloutEvidence,
    EnvironmentSampling,
    PartialRolloutBatchError,
    PolicyGenerator,
    PolicySampling,
    PolicyTurnRequest,
    RolloutBatch,
)
from ..reward_projection import RewardProjection
from ..rollout_execution import EpisodeKey, InvalidNativeEpisode
from ..turn_rewards import native_turn_map
from .native_records import (
    EncodedEpisode,
    encode_episode,
    encode_episode_on_workers,
)
from .native_records import (
    episode_assessment_observation as _episode_assessment_observation,
)
from .native_records import (
    native_record as _native_record,
)

type OnlineRLTechnique = Literal["grpo", "dapo", "olmo3", "sampo", "gdpo", "capo", "distill"]


class VerifiersRolloutFailure(PartialRolloutBatchError):
    """A terminal environment trace that cannot safely become a training sample."""


_VERIFIERS_UV_ORIGINAL: str | None = None


_COMPRESS_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _CompressedJsonl:
    path: Path
    digest: str
    records: int
    uncompressed_bytes: int


def _compress_jsonl(source: Path) -> _CompressedJsonl:
    """Write ``<source>.gz`` in bounded chunks and hash what was written.

    Level 1 compresses native traces about 5x at about 200 MB/s; a multi-GB
    run file is never held in memory.
    """

    target = source.with_name(f"{source.name}.gz")
    records = 0
    uncompressed = 0
    last = b"\n"
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as raw:
        temporary = Path(raw.name)
        try:
            with source.open("rb") as reader:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as writer:
                    while chunk := reader.read(_COMPRESS_CHUNK_BYTES):
                        records += chunk.count(b"\n")
                        uncompressed += len(chunk)
                        last = chunk[-1:]
                        writer.write(chunk)
            raw.flush()
            os.fsync(raw.fileno())
            raw.close()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if last != b"\n":
        records += 1
    digest = hashlib.sha256()
    with target.open("rb") as written:
        while chunk := written.read(_COMPRESS_CHUNK_BYTES):
            digest.update(chunk)
    return _CompressedJsonl(target, digest.hexdigest(), records, uncompressed)


def _native_failure_detail(episode: Any) -> str | None:
    """Return the most specific already-sanitized native failure summary."""

    errors = list(getattr(episode, "errors", ()))
    for trace in getattr(episode, "traces", ()):
        error = getattr(trace, "last_error", None)
        if error is not None:
            errors.append(error)
    if not errors:
        return None
    error = errors[-1]
    error_type = str(getattr(error, "type", type(error).__name__))
    message = " ".join(str(getattr(error, "message", error)).split())
    detail = f"{error_type}: {message}" if message else error_type
    return detail[:700]


def _apply_verifiers_runtime_compatibility() -> None:
    """Apply bounded compatibility fixes for the pinned Verifiers runtime."""

    from posttrain.environment.verifiers_preinstalled import configure_preinstalled_runtime
    from posttrain.environment.verifiers_runtime import enable_verifiers_fork_server

    enable_verifiers_fork_server()
    # In a packed job (TRL control or veRL backend environment) harness scripts
    # run from this process's locked interpreter: no pip, uv or PyPI at a rollout.
    if configure_preinstalled_runtime():
        return
    uv_executable = os.environ.get("POSTTRAIN_UV_EXECUTABLE")
    if uv_executable is None:
        return
    uv_path = Path(uv_executable)
    if not uv_path.is_absolute() or not uv_path.is_file() or not os.access(uv_path, os.X_OK):
        raise RuntimeError("POSTTRAIN_UV_EXECUTABLE must name an absolute executable file")
    try:
        from verifiers.v1.runtimes import base as runtime_base  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    global _VERIFIERS_UV_ORIGINAL
    if _VERIFIERS_UV_ORIGINAL is None:
        _VERIFIERS_UV_ORIGINAL = runtime_base._ENSURE_UV
    uv_directory = shlex.quote(str(uv_path.parent))
    runtime_base._ENSURE_UV = (
        f'export PATH={uv_directory}:"$HOME/.local/bin:$PATH" '
        'UV_INSTALL_DIR="$HOME/.local/bin"; '
        f"command -v uv >/dev/null 2>&1 || {{ {_VERIFIERS_UV_ORIGINAL}; }}"
    )


class TraceEnricher(Protocol):
    def __call__(self, trace: Any) -> None | Awaitable[None]: ...


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

    @property
    def reward_component_sources(self) -> Mapping[str, SignalSource]: ...


@dataclass(frozen=True, slots=True)
class NativeVerifiersEnvironmentFactory:
    """Pickle-safe reconstruction of a validated native Verifiers environment."""

    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", dict(self.config))

    def __call__(self) -> Any:
        EnvConfig, Environment = _environment_imports()
        if not hasattr(Environment, "episode"):
            from verifiers.v1.utils.loaders import load_environment, resolve_env_config

            return load_environment(resolve_env_config(dict(self.config)))
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
    model_identity: Mapping[str, JsonValue] = field(default_factory=dict)
    reward_component_sources: Mapping[str, SignalSource] = field(default_factory=dict)
    reward_projection: RewardProjection | None = None
    policy_update_context_contract: str | None = None
    record_encoding: Literal["thread", "process"] = "thread"

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
            model_identity=self.model_identity,
            reward_component_sources=self.reward_component_sources,
            reward_projection=self.reward_projection,
            policy_update_context_contract=self.policy_update_context_contract,
            record_encoding=self.record_encoding,
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
    from posttrain.environment.verifiers_runtime import verifiers_environment_types

    return verifiers_environment_types()


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
    config = type(base).model_validate(payload)
    NativeVerifiersEnvironmentFactory(config.model_dump(mode="python"))()
    return config.model_dump(mode="python")


def create_verifiers_training_bridge(
    environment: VerifiersEnvironmentSelection,
    trace_path: Path,
    run_id: str,
    *,
    sampling: PolicySampling,
    purpose: OnlineRLTechnique = "grpo",
    tasks: Mapping[int, Any] | None = None,
    model_identity: Mapping[str, JsonValue] | None = None,
    reward_projection: RewardProjection | None = None,
    policy_update_context_contract: str | None = None,
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
        model_identity=dict(model_identity or {}),
        reward_component_sources=dict(environment.reward_component_sources),
        reward_projection=reward_projection,
        policy_update_context_contract=policy_update_context_contract,
        record_encoding="process",
    )


def _apply_training_parameters(
    environment: VerifiersEnvironmentSelection,
    payload: dict[str, Any],
) -> None:
    parameters = environment.parameters
    limits = payload.get("agent", payload)
    if not isinstance(limits, dict):
        raise TypeError("Verifiers agent configuration must be an object")
    selected_token_limits = {
        key
        for key in ("max_input_tokens", "max_output_tokens", "max_total_tokens")
        if isinstance(parameters.get(key), int) and not isinstance(parameters.get(key), bool)
    }
    if "max_output_tokens" in selected_token_limits and "max_total_tokens" in selected_token_limits:
        raise ValueError("Verifiers training parameters must select output-token or total-token limits, not both")
    if "max_output_tokens" in selected_token_limits:
        limits.pop("max_total_tokens", None)
    elif "max_total_tokens" in selected_token_limits:
        limits.pop("max_output_tokens", None)
    for key in ("max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens"):
        value = parameters.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            limits[key] = value
    rollout_timeout = parameters.get("rollout_timeout_seconds")
    if isinstance(rollout_timeout, int | float) and not isinstance(rollout_timeout, bool):
        timeout = limits.setdefault("timeout", {})
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


def _normalized_task_facet(value: object) -> JsonValue:
    if isinstance(value, str | int | bool) or (isinstance(value, float) and math.isfinite(value)):
        return value
    if isinstance(value, list | tuple) and all(isinstance(item, str) and item.strip() for item in value):
        return sorted(set(value))
    raise ValueError("task observation facets must be finite scalars or lists of non-empty strings")


def _task_facet_values(task: Any, fields: tuple[str, ...]) -> dict[str, JsonValue]:
    data = getattr(task, "data", None)
    values: dict[str, JsonValue] = {}
    for name in fields:
        value = data.get(name) if isinstance(data, Mapping) else getattr(data, name, None)
        try:
            values[name] = _normalized_task_facet(value)
        except ValueError as error:
            raise ValueError(f"task data does not expose a valid observation facet {name!r}") from error
    return values


def _record_task_facets(info: Mapping[str, object]) -> dict[str, JsonValue]:
    raw = info.get("task_facets")
    if not isinstance(raw, Mapping):
        return {}
    values: dict[str, JsonValue] = {}
    for name, value in raw.items():
        if isinstance(name, str):
            try:
                values[name] = _normalized_task_facet(value)
            except ValueError:
                continue
    return values


class _PolicyClient:
    """Adapt Verifiers model turns to the platform policy-generator contract."""

    def __init__(self, generator: PolicyGenerator) -> None:
        self._generator = generator

    async def get_response(
        self,
        dialect: Any,
        body: dict[str, Any],
        model: Any,
        sampling_args: Any = None,
        session_id: str | None = None,
        turn: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        del headers
        if sampling_args is None:
            sampling_args = model
            model = body.get("model")
            if not isinstance(model, str) or not model:
                raise ValueError("native policy request requires the resolved model identity")
        AssistantMessage, _, _, Response, _, _, TurnTokens, Usage, ChatDialect, parse_tools = _imports()
        if not isinstance(dialect, ChatDialect):
            raise NotImplementedError("the online-RL policy bridge currently supports chat-completions dialects")
        if turn is None:
            parsed = dialect.parse_request(body)
            if hasattr(parsed, "messages"):
                messages = parsed.messages
                tools = parsed.tools
            else:
                messages, tools = parsed
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
            # The OpenAI wire shapes Verifiers' own train client renders (message_to_wire,
            # tool_to_wire), so an in-process policy (veRL, synchronous TRL) sees the same
            # prompt as the TRL policy endpoint; flat Tool records render shorter tool lists.
            messages=tuple(_wire_message(message) for message in messages),
            sampling=PolicySampling(
                max_tokens=int(sampling_args.max_tokens),
                temperature=1.0 if sampling_args.temperature is None else float(sampling_args.temperature),
                top_p=1.0 if sampling_args.top_p is None else float(sampling_args.top_p),
                top_k=int(getattr(sampling_args, "top_k", None) or 0),
                min_p=None if min_p is None else float(min_p),
                repetition_penalty=float(getattr(sampling_args, "repetition_penalty", None) or 1.0),
                presence_penalty=float(getattr(sampling_args, "presence_penalty", None) or 0.0),
            ),
            tools=tuple(_wire_tool(tool) for tool in tools or []),
            session_id=session_id,
            previous_prompt_ids=tuple(anchor[0]) if anchor else (),
            previous_completion_ids=tuple(anchor[1]) if anchor else (),
            tail_start=tail_start,
        )
        result = await self._generator.generate(request)
        token_evidence = {}
        if result.parser_evidence is not None:
            from .verifiers_generation import decode_parser_evidence

            if not {"generated_calls", "generated_call_producer"}.issubset(TurnTokens.model_fields):
                raise ValueError("selected native runtime cannot retain generated-call evidence")
            decoded = decode_parser_evidence(result.parser_evidence)
            token_evidence["generated_calls"] = decoded.attempts
            token_evidence["generated_call_producer"] = decoded.producer
        response = Response(
            id=str((result.raw_response or {}).get("id", "posttrain-policy-turn")),
            created=0,
            model=model,
            message=AssistantMessage.model_validate(result.message),
            finish_reason=result.finish_reason,
            usage=Usage(
                prompt_tokens=len(result.prompt_ids),
                completion_tokens=len(result.completion_ids),
                reasoning_tokens=result.reasoning_tokens,
            ),
            tokens=TurnTokens(
                prompt_ids=list(result.prompt_ids),
                completion_ids=list(result.completion_ids),
                completion_logprobs=list(result.completion_logprobs),
                message_spans=list(result.prompt_message_spans) or None,
                is_content=list(result.prompt_is_content) or None,
                **token_evidence,
            ),
        )
        response.raw = dict(result.raw_response or {})
        return response

    async def close(self) -> None:
        return None


def _wire_message(message: Any) -> Mapping[str, JsonValue]:
    """A Verifiers message in the OpenAI chat wire shape its train client renders."""

    if isinstance(message, Mapping):
        return dict(message)
    from verifiers.v1.dialects.chat import message_to_wire  # pyright: ignore[reportMissingImports]

    return message_to_wire(message)


def _wire_tool(tool: Any) -> Mapping[str, JsonValue]:
    """A Verifiers tool in the OpenAI function-tool wire shape its train client renders."""

    if isinstance(tool, Mapping):
        return dict(tool)
    from verifiers.v1.clients.train import tool_to_wire  # pyright: ignore[reportMissingImports]

    return tool_to_wire(tool)


def _record(value: Any) -> Mapping[str, JsonValue]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot project {type(value).__name__} into a policy message record")


def _project_training_branch(trace: Any) -> Any:
    """Resolve Verifiers' sampled/raw plus canonical tool-turn graph into one trajectory.

    Modern tool environments retain the exact sampled assistant response as one
    physical leaf and attach subsequent tool messages to a canonical, unsampled
    assistant sibling. For training, replace that canonical sibling with its
    unique sampled sibling on the deepest terminal path. Single-branch
    environments remain unchanged; ambiguous graph shapes fail closed.
    """
    from verifiers.v1.trace import Branch  # pyright: ignore[reportAttributeAccessIssue]

    if getattr(trace.agent, "execution_purpose", "solver") != "solver":
        raise VerifiersRolloutFailure("assessment execution cannot supply policy training tokens")
    branches = [branch for branch in trace.branches if branch.trainable and any(branch.sampled_mask)]
    if not branches:
        raise VerifiersRolloutFailure("online-RL trace has no trainable sampled branch")
    if len(branches) == 1:
        return branches[0]
    max_depth = max(len(branch.nodes) for branch in branches)
    terminal = [branch for branch in branches if len(branch.nodes) == max_depth]
    if len(terminal) != 1:
        raise VerifiersRolloutFailure(f"online-RL trace has {len(terminal)} equally deep trainable terminal branches")

    node_positions = {id(node): index for index, node in enumerate(trace.nodes)}
    expected_sampled_ids = {
        id(node) for branch in branches for node in branch.nodes if bool(getattr(node, "sampled", False))
    }
    resolved_nodes: list[Any] = []
    for node in terminal[0].nodes:
        role = getattr(getattr(node, "message", None), "role", None)
        if role != "assistant" or bool(getattr(node, "sampled", False)):
            resolved_nodes.append(node)
            continue
        position = node_positions[id(node)]
        sampled_siblings = [
            candidate
            for candidate in trace.nodes[:position]
            if candidate.parent == node.parent
            and bool(getattr(candidate, "sampled", False))
            and id(candidate) in expected_sampled_ids
            and getattr(getattr(candidate, "message", None), "role", None) == "assistant"
        ]
        if not sampled_siblings:
            resolved_nodes.append(node)
            continue
        if len(sampled_siblings) != 1:
            raise VerifiersRolloutFailure(
                "online-RL cannot resolve an unsampled assistant turn with multiple sampled siblings"
            )
        resolved_nodes.append(sampled_siblings[0])

    sampled_nodes = [node for node in resolved_nodes if bool(getattr(node, "sampled", False))]
    if {id(node) for node in sampled_nodes} != expected_sampled_ids:
        raise VerifiersRolloutFailure("online-RL terminal trajectory does not cover every sampled assistant turn")
    call_by_node = {call.node: call for call in trace.calls if call.node is not None}
    resolved_positions = [node_positions[id(node)] for node in resolved_nodes]
    return Branch(
        index=terminal[0].index,
        nodes=resolved_nodes,
        calls=[call_by_node[index] for index in resolved_positions if index in call_by_node],
        trainable=True,
        mm_token_type_id_map=trace.mm_token_type_id_map,
    )


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
    model_identity: Mapping[str, JsonValue] = field(default_factory=dict)
    reward_component_sources: Mapping[str, SignalSource] = field(default_factory=dict)
    reward_projection: RewardProjection | None = None
    policy_update_context_contract: str | None = None
    # "process" serializes each episode's replay record on the shared native
    # record worker pool; "thread" serializes it in a thread of this process.
    record_encoding: Literal["thread", "process"] = "thread"
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _trace_count: int = field(default=0, init=False)
    _live_observed_trace_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _requested_by_step: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    # Byte span and SHA-256 of each native episode record in episodes.jsonl,
    # keyed by every trace id in the record, so retaining a population reads
    # and proves only its own records instead of re-parsing the run's episodes.
    _episode_spans: dict[str, tuple[int, int, str]] = field(default_factory=dict, init=False, repr=False)
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

    @property
    def native_activation(self) -> Mapping[str, Any]:
        """Return the validated portable config used by isolated workers."""

        config = getattr(self.environment_factory, "config", None)
        if not isinstance(config, Mapping):
            raise RuntimeError("this Verifiers bridge does not have a portable native activation")
        return config

    @property
    def rollout_timeout_seconds(self) -> float:
        """Return the environment-owned deadline for one complete episode."""

        agent = self.native_activation.get("agent")
        timeout = agent.get("timeout") if isinstance(agent, Mapping) else None
        rollout = timeout.get("rollout") if isinstance(timeout, Mapping) else None
        if (
            isinstance(rollout, bool)
            or not isinstance(rollout, int | float)
            or not math.isfinite(rollout)
            or rollout <= 0
        ):
            raise ValueError("native Verifiers execution requires agent.timeout.rollout to be a positive number")
        return float(rollout)

    def task_for_example_id(self, example_id: str) -> Any:
        """Return the selected native task for one stable rollout example id."""

        try:
            _task_index, task = self._tasks_by_example_id[example_id]
        except KeyError as error:
            raise ValueError(f"unknown rollout example {example_id!r}") from error
        return task

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
        if self.reward_projection is not None and self.technique != "sampo" and not batch.prompt_group_ids:
            raise ValueError("structured RL requires explicit prompt-group and rollout identities")
        _, _, ModelContext, _, Sampling, TrainRunInfo, _, _, _, _ = _imports()
        client = _PolicyClient(generator)
        modern = hasattr(self._environment, "run_episode")
        context_client = client
        if modern:
            from verifiers.v1.configs.client import EvalClientConfig

            context_client = EvalClientConfig(
                base_url="http://posttrain-policy.invalid/v1",
                api_key_var="POSTTRAIN_INPROCESS_POLICY",
            )
        context = ModelContext(
            model=batch.model_id,
            client=context_client,
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
        failures: dict[int, str] = {}

        async def run_example(rollout_ordinal: int, example_id: str) -> tuple[int, EnvironmentRollout]:
            try:
                task_index, task = self._tasks_by_example_id[example_id]
            except KeyError as error:
                raise ValueError(f"unknown rollout example {example_id!r}") from error
            if modern:
                episode = await self._environment.run_episode(task, context)
                try:
                    rollout = await self._project_modern_episode(
                        episode,
                        task=task,
                        task_index=task_index,
                        example_id=example_id,
                        logical_step=batch.step,
                        rollout_ordinal=rollout_ordinal,
                        group_id=(batch.prompt_group_ids[rollout_ordinal] if batch.prompt_group_ids else None),
                        rollout_id=(batch.rollout_ids[rollout_ordinal] if batch.rollout_ids else None),
                        behavior_policy=batch.behavior_policy,
                        on_completed=on_completed,
                    )
                except InvalidNativeEpisode as error:
                    raise VerifiersRolloutFailure(str(error)) from error
                return rollout_ordinal, rollout

            traces = await self._environment.episode(task, context, n=1).run()
            if len(traces) != 1:
                raise ValueError(
                    f"Verifiers episode returned {len(traces)} traces for one scheduled branch; expected exactly one"
                )
            trace = traces[0]
            trace.info.update(
                posttrain_run={"type": "train", "id": self.run_id, "step": batch.step},
                environment_id=self.environment_id,
                task_index=task_index,
                example_id=example_id,
            )
            if batch.behavior_policy is not None:
                trace.info["posttrain_run"]["policy"] = {
                    "start": batch.behavior_policy.start,
                    "end": batch.behavior_policy.end,
                }
            if batch.prompt_group_ids:
                trace.info.update(
                    posttrain_prompt_group_id=batch.prompt_group_ids[rollout_ordinal],
                    posttrain_rollout_id=batch.rollout_ids[rollout_ordinal],
                )
            enrichment_error: Exception | asyncio.CancelledError | None = None
            try:
                for enrich in self.enrichers:
                    pending = enrich(trace)
                    if pending is not None:
                        await pending
            except (Exception, asyncio.CancelledError) as error:
                # Preserve the terminal native evidence even when derived
                # annotation fails; a missing critique is never successful credit.
                enrichment_error = error
                trace.info.update(posttrain_enrichment_error=type(error).__name__)
            record, observation = self._terminal_observation(trace, example_id, task_index, rollout_ordinal)
            # Native JSONL is the replay authority.  This happens before any
            # trainable projection, including branch/token/reward validation.
            await self._preserve_off_loop(record)
            if isinstance(enrichment_error, asyncio.CancelledError):
                raise enrichment_error
            if on_completed is not None:
                try:
                    await on_completed(observation)
                except Exception:
                    # A tracking outage must not erase the native evidence or
                    # turn a healthy rollout into a failed training sample.
                    pass
                else:
                    self.mark_live_observed(observation.external_id)
            if enrichment_error is not None:
                raise VerifiersRolloutFailure("native trace retained after enrichment failure") from enrichment_error
            return rollout_ordinal, replace(self._project(trace, observation), behavior_policy=batch.behavior_policy)

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
                except VerifiersRolloutFailure as error:
                    failures[ordinal] = str(error)
                    continue
                except BaseException as error:
                    async with fatal_lock:
                        if fatal_error is None:
                            fatal_error = error
                    return
                results[result_ordinal] = rollout

        def policy_client_factory(config: Any) -> Any:
            if config == context_client:
                return _PolicyClient(generator)
            from verifiers.v1.clients import resolve_client

            return resolve_client(config)

        serving = (
            self._environment.serving(client_factory=policy_client_factory) if modern else self._environment.serving()
        )
        async with serving:
            await asyncio.gather(*(worker() for _ in range(worker_count)))
        if fatal_error is not None:
            if isinstance(fatal_error, asyncio.CancelledError):
                raise fatal_error
            raise VerifiersRolloutFailure(f"Verifiers rollout failed: {fatal_error}") from fatal_error
        if failures:
            raise VerifiersRolloutFailure(
                f"{len(failures)} of {len(occurrences)} Verifiers rollouts failed: {sorted(set(failures.values()))}",
                completed=results,
                failures=failures,
            )
        return [results[ordinal] for ordinal in range(len(occurrences))]

    async def project_native_episode(
        self,
        key: EpisodeKey,
        episode: Any,
        *,
        behavior_policy: BehaviorPolicySpan | None = None,
        on_completed: AsyncTerminalTraceObserver | None = None,
    ) -> EnvironmentRollout:
        """Project a worker-returned native episode through the direct bridge contract."""

        if key.collection.run_id != self.run_id:
            raise ValueError("native episode collection belongs to a different training run")
        try:
            task_index, task = self._tasks_by_example_id[key.example_id]
        except KeyError as error:
            raise ValueError(f"unknown native rollout example {key.example_id!r}") from error
        return await self._project_modern_episode(
            episode,
            task=task,
            task_index=task_index,
            example_id=key.example_id,
            logical_step=key.collection.logical_step,
            rollout_ordinal=key.rollout_ordinal,
            group_id=key.group_id,
            rollout_id=key.occurrence_id,
            behavior_policy=behavior_policy,
            on_completed=on_completed,
        )

    async def _project_modern_episode(
        self,
        episode: Any,
        *,
        task: Any,
        task_index: int,
        example_id: str,
        logical_step: int,
        rollout_ordinal: int,
        group_id: str | None,
        rollout_id: str | None,
        behavior_policy: BehaviorPolicySpan | None,
        on_completed: AsyncTerminalTraceObserver | None,
    ) -> EnvironmentRollout:
        """Retain and project one modern episode, regardless of process placement."""

        from verifiers.v1.episode import GroupInfo  # pyright: ignore[reportAttributeAccessIssue]

        *_, TrainRunInfo, _, _, _, _ = _imports()
        policy_record = (
            None if behavior_policy is None else {"start": behavior_policy.start, "end": behavior_policy.end}
        )
        episode.record_run(
            TrainRunInfo(
                id=self.run_id,
                work={"type": "train", "step": logical_step, "policy": policy_record},
            )
        )
        episode.env.name = self.environment_id
        if group_id is not None:
            episode.group = GroupInfo(id=group_id)
        for native_trace in episode.traces:
            native_trace.info.update(
                posttrain_episode_id=episode.id,
                posttrain_run={
                    "type": "train",
                    "id": self.run_id,
                    "step": logical_step,
                    **({"policy": policy_record} if policy_record is not None else {}),
                },
                environment_id=self.environment_id,
                task_index=task_index,
                example_id=example_id,
                task_facets=_task_facet_values(task, self.task_facet_fields),
            )
            if group_id is not None and rollout_id is not None:
                native_trace.info.update(
                    posttrain_prompt_group_id=group_id,
                    posttrain_rollout_id=rollout_id,
                )
        traces = [
            trace
            for trace in episode.traces
            if trace.agent.trainable and getattr(trace.agent, "execution_purpose", "solver") == "solver"
        ]
        invalid_assessment_standing = any(
            trace.agent.trainable and getattr(trace.agent, "execution_purpose", "solver") == "assessment"
            for trace in episode.traces
        )
        if not episode.ok or len(traces) != 1 or invalid_assessment_standing:
            failure_detail = _native_failure_detail(episode)
            reason = (
                "native episode is not trainable "
                f"(ok={episode.ok}, total_traces={len(episode.traces)}, trainable_traces={len(traces)})"
            )
            if failure_detail is not None:
                reason += f"; native_error={failure_detail}"
            for native_trace in episode.traces:
                native_trace.info.update(posttrain_admission_error=reason)
            encoded = await self._preserve_episode_off_loop(episode)
            await self._observe_episode_assessments(encoded.assessment, on_completed)
            raise InvalidNativeEpisode(reason)
        trace = traces[0]
        enrichment_error: Exception | asyncio.CancelledError | None = None
        try:
            for enrich in self.enrichers:
                pending = enrich(trace)
                if pending is not None:
                    await pending
        except (Exception, asyncio.CancelledError) as error:
            enrichment_error = error
            trace.info.update(posttrain_enrichment_error=type(error).__name__)
        encoded = await self._preserve_episode_off_loop(
            episode,
            trace_id=str(trace.id),
            task_facets=_task_facet_values(self.tasks[task_index], self.task_facet_fields),
        )
        record = encoded.trace_record
        assert record is not None and encoded.trace_line is not None
        observation = self._observation_from_record(
            record,
            example_id=example_id,
            task_index=task_index,
            rollout_ordinal=rollout_ordinal,
        )
        await asyncio.to_thread(self._preserve_line, encoded.trace_line)
        if isinstance(enrichment_error, asyncio.CancelledError):
            raise enrichment_error
        await self._observe_episode_assessments(encoded.assessment, on_completed)
        if on_completed is not None:
            try:
                await on_completed(observation)
            except Exception:
                pass
            else:
                self.mark_live_observed(observation.external_id)
        if enrichment_error is not None:
            raise InvalidNativeEpisode("native trace retained after enrichment failure") from enrichment_error
        try:
            rollout = self._project(
                trace,
                observation,
                assignments=getattr(episode, "credit_assignments", ()),
                source_traces={str(child.id): child for child in episode.traces},
            )
            return replace(rollout, behavior_policy=behavior_policy)
        except VerifiersRolloutFailure as error:
            raise InvalidNativeEpisode(str(error)) from error

    def _terminal_observation(
        self,
        trace: Any,
        example_id: str,
        task_index: int,
        rollout_ordinal: int,
    ) -> tuple[dict[str, Any], TraceObservation]:
        record = _native_record(trace)
        trace_info = getattr(trace, "info", {})
        if isinstance(trace_info.get("posttrain_run"), Mapping):
            # Versioned derived trace view for existing observation consumers;
            # the untouched native episode is retained alongside this view.
            record["run"] = dict(trace_info["posttrain_run"])
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
            **verifiers_trace_attributes(record),
            **self.model_identity,
            "environment_id": str(info.get("environment_id") or self.environment_id),
            "task_index": int(info.get("task_index", -1) if task_index is None else task_index),
            "example_id": str(info.get("example_id") or "") if example_id is None else example_id,
            **_record_task_facets(info),
        }
        if rollout_ordinal is not None:
            attributes["rollout_ordinal"] = rollout_ordinal
        facts = project_verifiers_trace_facts(
            record,
            attributes=attributes,
            reward_component_sources=self.reward_component_sources,
        )
        return TraceObservation(
            trace_type="verifiers",
            external_id=external_id,
            payload=record,
            attributes=attributes,
            facts=(facts,),
        )

    def _project(
        self,
        trace: Any,
        observation: TraceObservation,
        *,
        assignments: Iterable[Any] = (),
        source_traces: Mapping[str, Any] | None = None,
    ) -> EnvironmentRollout:
        if _trace_has_error(observation.payload):
            raise VerifiersRolloutFailure("Verifiers trace terminated with a harness or environment error")
        branch = _project_training_branch(trace)
        conditioning = ()
        if self.policy_update_context_contract is not None:
            original_positions = {id(node): index for index, node in enumerate(trace.nodes)}
            conditioning = native_conditioning_records(
                trace,
                sampled_node_indices=tuple(original_positions[id(node)] for node in branch.nodes if node.sampled),
                context_contract=self.policy_update_context_contract,
            )
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
        from .verifiers_credit import align_native_credit

        assigned_credits = align_native_credit(
            trace,
            branch,
            first_sampled,
            assignments=assignments,
            traces=source_traces,
        )
        conditioning_completion_indices = []
        if conditioning:
            offset = 0
            for node in branch.nodes:
                if node.sampled:
                    conditioning_completion_indices.append(
                        tuple(offset + index - first_sampled for index, eligible in enumerate(node.mask) if eligible)
                    )
                offset += len(node.token_ids)
        logprobs = tuple(float(value) for value in branch.logprobs[first_sampled:])
        if len(logprobs) != len(completion_ids):
            raise ValueError("Verifiers branch logprobs are not aligned to the training sequence")
        if not math.isfinite(float(trace.reward)):
            raise VerifiersRolloutFailure("Verifiers trace has a non-finite scalar reward")
        is_truncated = bool(observation.attributes["is_truncated"])
        turns = (
            _agentic_turns(branch, first_sampled)
            if (self.technique == "sampo" or self.policy_update_context_contract is not None)
            else ()
        )
        native_turns = (
            native_turn_map(branch)
            if self.reward_projection is not None and self.reward_projection.turns_info_key is not None
            else ()
        )
        turn_ids = tuple(turn.id for turn in native_turns)
        if turns and self.reward_projection is not None:
            local_rewards = self.reward_projection.project_turn_rewards(observation, turn_ids)
            if local_rewards is not None:
                turns = tuple(
                    replace(turn, step_reward=reward) for turn, reward in zip(turns, local_rewards, strict=True)
                )
            state_keys = self.reward_projection.project_turn_state_keys(observation, turn_ids)
            if state_keys is not None:
                turns = tuple(replace(turn, state_key=key) for turn, key in zip(turns, state_keys, strict=True))
            outcomes = self.reward_projection.project_turn_outcomes(observation, turn_ids)
            if outcomes is not None:
                turns = tuple(
                    replace(turn, goal_credits=goals, harm_debit=harm)
                    for turn, (goals, harm) in zip(turns, outcomes, strict=True)
                )
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
            facts=observation.facts,
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
            conditioning_records=conditioning,
            selected_branch_id=str(branch.index) if conditioning else None,
            conditioning_completion_indices=tuple(conditioning_completion_indices),
            assigned_credit=assigned_credits,
            turns=turns,
            reward_evidence=(
                self.reward_projection.project(
                    observation,
                    scalar_reward=float(trace.reward),
                    turn_ids=turn_ids,
                    native_turns=native_turns,
                )
                if self.reward_projection is not None and self.technique != "sampo"
                else None
            ),
        )

    def mark_live_observed(self, external_id: str) -> None:
        """Mark a native trace accepted by a live host/provider submission."""

        with self._write_lock:
            self._live_observed_trace_ids.add(external_id)

    async def _preserve_episode_off_loop(
        self,
        episode: Any,
        *,
        trace_id: str | None = None,
        task_facets: Mapping[str, JsonValue] | None = None,
    ) -> EncodedEpisode:
        """Encode and append a native episode without blocking the event loop.

        Episode records carry every turn's token ids and the assessment
        archive (megabytes of JSON). The rollout event loop also serves the
        policy engine, so the record is serialized on the shared worker pool
        (or a thread) and only the append happens here. Awaiting it keeps the
        write ordered before any projection of the episode.
        """
        if self.record_encoding == "process":
            encoded = await encode_episode_on_workers(episode, trace_id=trace_id, task_facets=task_facets)
        else:
            encoded = await asyncio.to_thread(encode_episode, episode, trace_id=trace_id, task_facets=task_facets)
        await asyncio.to_thread(self._write_episode, episode, encoded)
        return encoded

    async def _observe_episode_assessments(
        self, observation: TraceObservation | None, on_completed: AsyncTerminalTraceObserver | None
    ) -> None:
        if on_completed is None or observation is None:
            return
        with self._write_lock:
            if observation.external_id in self._live_observed_trace_ids:
                return
        try:
            await on_completed(observation)
        except Exception:
            # Retained native episodes allow evidence() to retry submission.
            return
        self.mark_live_observed(observation.external_id)

    async def _preserve_off_loop(self, record: dict[str, Any]) -> None:
        await asyncio.to_thread(self._preserve, record)

    def _preserve_episode(self, episode: Any) -> EncodedEpisode:
        """Encode and append one native episode in this thread."""
        encoded = encode_episode(episode)
        self._write_episode(episode, encoded)
        return encoded

    def _write_episode(self, episode: Any, encoded: EncodedEpisode) -> None:
        from .verifiers_assessment_artifacts import retain_episode_artifacts

        with self._write_lock:
            retain_episode_artifacts(episode, self.trace_path.parent / "assessment-evidence")
        offset, length = self._append_line(self.trace_path.with_name("episodes.jsonl"), encoded.line)
        with self._write_lock:
            try:
                spans = self._episode_spans
            except AttributeError:  # constructed without dataclass initialization
                spans = self._episode_spans = {}
            for identity in encoded.trace_ids:
                spans[identity] = (offset, length, encoded.digest)

    def trace_observation(self, record: Mapping[str, Any]) -> TraceObservation:
        """Reconstruct one terminal native record in a host-side observer."""

        return self._observation_from_record(record)

    def _preserve(self, record: dict[str, Any]) -> None:
        self._preserve_line((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))

    def _preserve_line(self, line: bytes) -> None:
        self._append_line(self.trace_path, line)
        with self._write_lock:
            self._trace_count += 1

    def _append_line(self, path: Path, encoded: bytes) -> tuple[int, int]:
        """Keep native and derived JSONL intact across concurrent rollout workers.

        Returns the line's (byte offset, byte length) in the file."""
        with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as stream:
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - Windows is not a qualified veRL target
                    fcntl = None
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    offset = stream.seek(0, os.SEEK_END)
                    stream.write(encoded)
                    stream.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return offset, len(encoded)

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
            model_identity=self.model_identity,
            reward_component_sources=self.reward_component_sources,
            reward_projection=self.reward_projection,
            policy_update_context_contract=self.policy_update_context_contract,
            record_encoding=self.record_encoding,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def retain_population(self, rollouts: Sequence[EnvironmentRollout]) -> ProducedArtifact:
        """Seal admitted native episodes before any resolved update consumes them.

        This does not finalize the collector or publish through an observer.
        The host publishes the returned artifact and admits its retained bytes.
        """
        from .verifiers_population_artifact import retain_native_population

        with self._write_lock:
            episodes_path = self.trace_path.with_name("episodes.jsonl")
            episodes = episodes_path.is_file()
            return retain_native_population(
                episodes_path if episodes else self.trace_path,
                self.trace_path.parent / "populations",
                tuple(rollout.trace.external_id for rollout in rollouts),
                episodes=episodes,
                spans=dict(getattr(self, "_episode_spans", {})) if episodes else None,
            )

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        """Publish the replay authority, compressed, once per run.

        The native episode envelope is the replay authority; the derived trace
        view is rebuilt from it and was streamed to tracking as each rollout
        finished, so it is not published a second time. Runtimes without
        episode envelopes publish their trace records as the authority.
        """

        from .verifiers_assessment_artifacts import seal_assessment_artifacts

        with self._write_lock:
            evidence_bundle = seal_assessment_artifacts(
                self.trace_path.parent / "assessment-evidence",
                name=f"training/rollouts/{self.dataset.id}/verifiers-assessment-evidence",
            )
        retained_evidence = () if evidence_bundle is None else (evidence_bundle,)
        episodes_path = self.trace_path.with_name("episodes.jsonl")
        common = {
            "technique": self.technique,
            "environment_id": self.environment_id,
            "dataset_id": self.dataset.id,
            "dataset_revision": self.dataset.revision,
            "replay_authority": True,
            "compression": "gzip",
        }
        if episodes_path.is_file():
            compressed = _compress_jsonl(episodes_path)
            return retained_evidence + (
                ProducedArtifact(
                    name=f"training/rollouts/{self.dataset.id}/verifiers-episodes",
                    kind="evaluation-traces",
                    reference=LocalArtifactRef(compressed.path.resolve(), compressed.digest),
                    metadata={
                        **common,
                        "episode_count": compressed.records,
                        "uncompressed_bytes": compressed.uncompressed_bytes,
                        "format": "verifiers-native-episodes",
                    },
                ),
            )
        if not self.trace_path.is_file():
            return retained_evidence
        compressed = _compress_jsonl(self.trace_path)
        return retained_evidence + (
            ProducedArtifact(
                name=f"training/rollouts/{self.dataset.id}/verifiers-traces",
                kind="evaluation-traces",
                reference=LocalArtifactRef(compressed.path.resolve(), compressed.digest),
                metadata={
                    **common,
                    "trace_count": compressed.records,
                    "uncompressed_bytes": compressed.uncompressed_bytes,
                    "schema_version": 2,
                },
            ),
        )

    def evidence(self) -> EnvironmentRolloutEvidence:
        """Replay native trace records and trace-derived metrics in the host process."""

        with self._write_lock:
            live_observed_trace_ids = frozenset(self._live_observed_trace_ids)
            requested_by_step = dict(self._requested_by_step)
        episode_observations: dict[str, TraceObservation] = {}
        episodes_path = self.trace_path.with_name("episodes.jsonl")
        if episodes_path.is_file():
            with episodes_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    episode = json.loads(line)
                    if not isinstance(episode, dict):
                        raise TypeError("preserved native episodes must be JSON objects")
                    observation = _episode_assessment_observation(episode)
                    if observation is not None and observation.external_id not in live_observed_trace_ids:
                        episode_observations[observation.external_id] = observation
        if not self.trace_path.is_file():
            return EnvironmentRolloutEvidence(
                traces=tuple(episode_observations.values()),
                metrics=tuple(
                    MetricBatchObservation(
                        _trace_metrics((), requested=requested),
                        step=step,
                        attributes={"observation_source": "verifiers"},
                    )
                    for step, requested in sorted(requested_by_step.items())
                ),
            )
        # Stream the trace file: a long run's file is ~100 GB, and holding every record
        # (as this replay once did) OOM-killed a 400-update run after training completed.
        # Per step, only each record's metric summary is kept.
        traces: list[TraceObservation] = list(episode_observations.values())
        records_by_step: dict[int, list[_TraceSummary]] = {}
        with self.trace_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
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
                records_by_step.setdefault(run["step"], []).append(_summarize_trace(record))
        metrics = tuple(
            MetricBatchObservation(
                _summary_metrics(step_records, requested=requested_by_step.get(step)),
                step=step,
                attributes={"observation_source": "verifiers"},
            )
            for step in sorted(set(records_by_step) | set(requested_by_step))
            for step_records in (records_by_step.get(step, []),)
        )
        return EnvironmentRolloutEvidence(metrics=metrics, traces=tuple(traces))


def _trace_model(record: Mapping[str, Any]) -> str:
    return str(verifiers_trace_attributes(record)["model"] or "")


def _trace_reward(record: Mapping[str, Any]) -> float | None:
    reward = project_verifiers_trace_facts(record).measures["task_reward"]
    return float(reward) if reward is not None else None


def _trace_has_error(record: Mapping[str, Any]) -> bool:
    return verifiers_trace_has_error(record)


def _trace_is_truncated(record: Mapping[str, Any]) -> bool:
    return verifiers_trace_is_truncated(record)


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
        if content.lstrip().startswith("Error executing tool"):
            return True
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and (value.get("success") is False or bool(value.get("error"))):
            return True
    return False


@dataclass(frozen=True, slots=True)
class _TraceSummary:
    """The few values of one trace record that per-step rollout metrics use."""

    reward: float | None
    example_id: str
    completed: bool
    failed: bool
    truncated: bool
    ending: Any
    tool_call: bool
    tool_failure: bool


def _summarize_trace(record: Mapping[str, Any]) -> _TraceSummary:
    info = record.get("info")
    return _TraceSummary(
        reward=_trace_reward(record),
        example_id=str(info.get("example_id") or "") if isinstance(info, Mapping) else "",
        completed=bool(record.get("is_completed")),
        failed=_trace_has_error(record),
        truncated=_trace_is_truncated(record),
        ending=verifiers_episode_ending(record),
        tool_call=_trace_has_tool_call(record),
        tool_failure=_trace_has_tool_failure(record),
    )


def _trace_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    requested: int | None = None,
) -> dict[str, float]:
    return _summary_metrics([_summarize_trace(record) for record in records], requested=requested)


def _summary_metrics(
    summaries: Sequence[_TraceSummary],
    *,
    requested: int | None = None,
) -> dict[str, float]:
    attempted = len(summaries)
    if attempted == 0 and requested is None:
        raise ValueError("cannot derive rollout evidence from an empty step")
    rewards = [summary.reward for summary in summaries if summary.reward is not None]
    grouped_rewards: dict[str, list[float]] = {}
    for summary in summaries:
        if summary.reward is not None:
            grouped_rewards.setdefault(summary.example_id, []).append(summary.reward)
    grouped = [values for values in grouped_rewards.values() if len(values) > 1]
    values = {
        "train/rl/rollouts_requested": float(requested if requested is not None else attempted),
        "train/rl/rollouts_attempted": float(attempted),
        "train/rl/rollouts_completed": float(sum(summary.completed for summary in summaries)),
        "train/rl/rollouts_failed": float(sum(summary.failed for summary in summaries)),
        "train/rl/rollouts_truncated": float(sum(summary.truncated for summary in summaries)),
        "train/rl/rollouts_unscorable": float(attempted - len(rewards)),
        **episode_ending_metrics(summary.ending for summary in summaries),
    }
    if attempted:
        values["train/rl/tool_call_frequency"] = sum(summary.tool_call for summary in summaries) / attempted
        values["train/rl/tool_failure_frequency"] = sum(summary.tool_failure for summary in summaries) / attempted
    if requested is not None:
        values["train/rl/rollouts_missing"] = float(max(requested - attempted, 0))
    # Missing reward variation is evidence that no valid training population
    # existed, not a measured zero.  Do not fabricate a zero-variance signal.
    if rewards:
        values["train/rl/reward_std"] = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if grouped:
        values["train/rl/group_zero_variance_fraction"] = sum(statistics.pstdev(group) == 0 for group in grouped) / len(
            grouped
        )
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
    observations: list[Mapping[str, JsonValue]] = []
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
            if not observations:
                raise ValueError("SAMPO sampled assistant turns require a preceding user or tool observation")
            start = offset + sampled_positions[0] - first_sampled
            end = offset + sampled_positions[-1] + 1 - first_sampled
            turns.append(
                AgenticTurn(
                    completion_start=start,
                    completion_end=end,
                    anchor_state_key=_observation_bundle_key(observations),
                )
            )
            observations = []
        elif role in {"user", "tool"}:
            observations.append(record)
        offset += len(node_ids)
    if not turns:
        raise ValueError("SAMPO Verifiers trace has no sampled assistant turns")
    return tuple(turns)


def _anchor_state_key(observation: Mapping[str, JsonValue]) -> str:
    """Compatibility helper for a single observation proxy, not environment state."""
    return _observation_bundle_key([observation])


_SAMPLE_IDENTIFIER = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _observation_bundle_key(observations: Sequence[Mapping[str, JsonValue]]) -> str:
    """Hash the complete ordered observation bundle preceding an assistant action.

    Transport tool-call IDs and minted UUIDs retain the existing AutomationBench
    normalization. Repeated observations remain a proxy for state equality;
    the prefix versions this changed grouping contract for retained native rows.
    """

    content = [
        {key: value for key, value in observation.items() if key != "tool_call_id"} for observation in observations
    ]
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    encoded = _SAMPLE_IDENTIFIER.sub("<id>", encoded)
    return "observation-bundle@2:" + hashlib.sha256(encoded.encode()).hexdigest()


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
