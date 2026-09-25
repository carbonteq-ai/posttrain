"""Settings calculator: suggest engine settings for hardware, model and task."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from posttrain.advisor import HubModelReader, Task, review, suggest
from posttrain.advisor.calculator import Hardware
from posttrain.advisor.review import recommendation_payload
from posttrain.common import CatalogRef, ContractError, ExecutionTarget, ModelVariant

from ..context import CliState
from ..output import emit
from ..work_runtime import load_work_package_bundle
from .work_package import finding_lines

_PURPOSES = ("rollout", "eval", "screen", "judge")


def _render(payload: Mapping[str, Any]) -> str:
    if "unavailable" in payload:
        return f"No recommendation: {payload['unavailable']}"
    rows = payload["settings"]
    lines = ["engine:"]
    for row in rows:
        value = row["suggested"] if isinstance(row["suggested"], str) else json.dumps(row["suggested"])
        was = ""
        if "changed" in row:
            was = "  # unchanged" if not row["changed"] else f"  # was {json.dumps(row.get('current'))}"
        lines.append(f"  {row['key']}: {value}{was}")
    if payload["environment"]:
        lines.append("environment:")
        lines.extend(f"  {key}: {value}" for key, value in payload["environment"].items())
    lines.append("")
    lines.append("Why:")
    lines.extend(f"  {row['key']}: {row['reason']}" for row in rows)
    lines.append("")
    lines.append("Memory per GPU (GB): " + ", ".join(f"{key} {value:g}" for key, value in payload["memory_gb"].items()))
    if payload["max_concurrency"] is not None:
        lines.append(f"Typical-length sequences that fit in the KV budget: {payload['max_concurrency']}")
    if payload["decode_tokens_per_s_upper_bound"] is not None:
        lines.append(
            "Decode throughput upper bound (memory bandwidth, estimate): "
            f"{payload['decode_tokens_per_s_upper_bound']:,.0f} tok/s"
        )
    lines.extend(f"Note: {note}" for note in payload["notes"])
    step = payload.get("step")
    if step:
        lines.append("")
        lines.append(f"Step sizing: {step['reason']}.")
        lines.append("  option        prompts  oversampled groups  rows  waves  step time  rows/s")
        lines.extend(
            f"  {option['label']:<12}  {option['prompts_per_step']:>7}  {option['oversample_groups']:>18}  "
            f"{option['rows']:>4}  {option['waves']:>5}  {option['relative_step_time']:>8.2f}x  "
            f"{option['relative_rows_per_second']:>5.2f}x"
            for option in step["options"]
        )
        lines.append("  Step time and rows/s are decode-bound estimates relative to the current step.")
    return "\n".join(lines)


def _override_target(snapshot: dict[str, Any], memory_gb: float | None, gpu: str | None, gpus: int | None) -> None:
    targets = snapshot.get("execution_targets", {}).get("targets", [])
    for target in targets:
        if memory_gb is not None:
            target["memory_gb"] = memory_gb
        hardware = target.setdefault("hardware", {})
        if gpu is not None:
            hardware["accelerator_model"] = gpu
        if gpus is not None:
            hardware["accelerator_count"] = gpus


def register(app: typer.Typer) -> None:
    settings_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="calculate engine settings for hardware, model and task"
    )
    app.add_typer(settings_app, name="settings")

    @settings_app.command(
        "suggest",
        help="suggest vLLM engine settings, with reasons, a memory budget and step options, for a model and task",
    )
    def settings_suggest_cmd(
        ctx: typer.Context,
        model: Annotated[
            str | None, typer.Option("--model", help="catalog model variant, e.g. models/lfm2.5-2.6b@bf16")
        ] = None,
        target: Annotated[str | None, typer.Option("--target", help="catalog execution target")] = None,
        memory_gb: Annotated[
            float | None, typer.Option("--memory-gb", help="GPU memory instead of the target's")
        ] = None,
        gpu: Annotated[str | None, typer.Option("--gpu", help="accelerator model, e.g. RTXPRO6000, H100")] = None,
        gpus: Annotated[int | None, typer.Option("--gpus", help="GPUs the engine shards across")] = None,
        purpose: Annotated[str, typer.Option("--purpose", help="rollout, eval, screen or judge")] = "eval",
        concurrency: Annotated[int, typer.Option("--concurrency", help="concurrent sequences")] = 8,
        prompt_tokens: Annotated[int, typer.Option("--prompt-tokens")] = 4096,
        completion_tokens: Annotated[int, typer.Option("--completion-tokens")] = 1024,
        temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
        colocated_trainer_gb: Annotated[
            float | None,
            typer.Option("--colocated-trainer-gb", help="memory a colocated trainer keeps during rollouts"),
        ] = None,
        reproducible: Annotated[bool, typer.Option("--reproducible", help="require batch-invariant logprobs")] = False,
        lora_rank: Annotated[int | None, typer.Option("--lora-rank")] = None,
        kv_cache_dtype: Annotated[str, typer.Option("--kv-cache-dtype", help="auto, or a TurboQuant format")] = "auto",
        work_package: Annotated[
            Path | None,
            typer.Option(
                "--work-package",
                help="review a package's recorded selections: rule and calculator findings, and each binding's diff",
            ),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        reader = HubModelReader()
        if work_package is not None:
            from posttrain.work import resolve_work_package

            _, catalog, _, package = load_work_package_bundle(state, work_package)
            snapshot = copy.deepcopy(dict(resolve_work_package(catalog, package).snapshot))
            _override_target(snapshot, memory_gb, gpu, gpus)
            result = review(snapshot, reader)
            if not result.recommendations:
                raise ContractError("the work package binds no locally served inference seat")
            findings = [issue.as_dict() for issue in result.findings]
            blocks = [f"# {role}\n{_render(payload)}" for role, payload in result.recommendations.items()]
            if result.findings:
                blocks.append("Findings:\n" + "\n".join(finding_lines(result.findings, prefix="  ")))
            emit(state, {"recommendations": dict(result.recommendations), "findings": findings}, "\n\n".join(blocks))
            return
        if model is None:
            raise ContractError("give --model (and --target or --memory-gb), or --work-package")
        _, catalog = state.open_catalog()
        variant = catalog.resolve(CatalogRef("model", model)).value
        if not isinstance(variant, ModelVariant):
            raise ContractError(f"{model} is not a model variant")
        target_value = catalog.resolve(CatalogRef("target", target)).value if target else None
        if target_value is not None and not isinstance(target_value, ExecutionTarget):
            raise ContractError(f"{target} is not an execution target")
        declared = target_value.hardware if target_value is not None else None
        memory = memory_gb or (target_value.memory_gb if target_value is not None else None)
        if memory is None:
            raise ContractError("give --target with declared memory, or --memory-gb")
        hardware = Hardware(
            memory_gb=float(memory),
            accelerator_model=gpu or (declared.accelerator_model if declared is not None else None),
            accelerator_count=gpus
            or (declared.accelerator_count if declared is not None and declared.accelerator_count else 1),
            supports_bf16=declared.supports_bf16 if declared is not None else None,
        )
        if purpose not in _PURPOSES:
            raise ContractError("--purpose must be rollout, eval, screen or judge")
        task = Task(
            purpose,  # type: ignore[arg-type]
            concurrency,
            prompt_tokens,
            completion_tokens,
            temperature,
            colocated_trainer_gb=colocated_trainer_gb,
            reproducible_logprobs=reproducible,
            lora_rank=lora_rank,
            kv_cache_dtype=kv_cache_dtype,
        )
        architecture = reader(variant.base.repo_id, variant.base.revision)
        payload = recommendation_payload(variant.id, hardware, task, suggest(hardware, architecture, task))
        emit(state, payload, _render(payload))


__all__ = ["register"]
