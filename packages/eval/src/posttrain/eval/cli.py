"""Run Verifiers v1 environments against an OpenAI-compatible model endpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import trackio
from common import PROFILES_DIR, ProfileResolver, TrackedRun

from .results import summarize_traces
from .suites import EnvironmentSpec, load_suite
from .trace_sync import TraceSyncStats, VerifiersTraceSynchronizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reusable Verifiers evaluation selection.")
    parser.add_argument("model_profile", help="Reference under profiles/models")
    parser.add_argument(
        "--eval-profile",
        default=None,
        help="Reference under profiles/eval; defaults to the model profile's general eval",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", help="Endpoint model name; defaults to the Hub repository")
    parser.add_argument(
        "--serve-profile",
        help="Serving config used by the endpoint; defaults to model defaults.eval.serve",
    )
    parser.add_argument(
        "--served-context-window",
        type=int,
        help="Override the endpoint context limit when it differs from the selected serve profile",
    )
    parser.add_argument("--api-key-var", default="LOCAL_INFERENCE_API_KEY")
    parser.add_argument("--reasoning-mode")
    parser.add_argument("--environment", action="append", dest="environment_ids")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan only")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Ask Verifiers to resolve packages and validate native configs without model calls",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable Verifiers debug logging")
    return parser


def _hf_repository(artifact: str) -> str:
    if not artifact.startswith("hf://"):
        raise ValueError(f"evaluation currently requires an hf:// model artifact: {artifact!r}")
    repository, separator, revision = artifact.removeprefix("hf://").rpartition("@")
    if not separator or not repository or not revision:
        raise ValueError("Hugging Face model artifacts must include an immutable revision")
    return repository


def _eval_reference(model: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    reference = model.get("defaults", {}).get("eval", {}).get("general")
    if not isinstance(reference, str) or not reference:
        raise ValueError("model profile does not declare defaults.eval.general")
    return reference


def _eval_path(reference: str) -> Path:
    value = Path(reference)
    if value.parts and value.parts[0] == "eval":
        value = Path(*value.parts[1:])
    if not str(value).endswith((".yaml", ".yml")):
        value = Path(f"{value}.yaml")
    path = (PROFILES_DIR / "eval" / value).resolve()
    if not path.is_relative_to((PROFILES_DIR / "eval").resolve()):
        raise ValueError("eval profile reference escapes profiles/eval")
    return path


def _reasoning_settings(model: dict[str, Any], requested: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    reasoning = model.get("prompting", {}).get("reasoning", {})
    mode = requested or reasoning.get("default", "native")
    modes = reasoning.get("modes", {})
    if not isinstance(mode, str) or mode not in modes:
        available = ", ".join(sorted(str(value) for value in modes))
        raise ValueError(f"unsupported reasoning mode {mode!r}; expected one of: {available}")
    settings = modes[mode]
    if not isinstance(settings, dict):
        raise ValueError(f"reasoning mode {mode!r} must resolve to a mapping")
    chat_template_kwargs = settings.get("chat_template_kwargs", {})
    sampling = settings.get("sampling", {})
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError(f"reasoning mode {mode!r}.chat_template_kwargs must be a mapping")
    if not isinstance(sampling, dict):
        raise ValueError(f"reasoning mode {mode!r}.sampling must be a mapping")
    return mode, dict(chat_template_kwargs), dict(sampling)


def _serve_reference(model: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    defaults = model.get("defaults", {})
    reference = defaults.get("eval", {}).get("serve") or defaults.get("serve", {}).get("vllm")
    if not isinstance(reference, str) or not reference:
        raise ValueError("model profile does not declare a default serving profile")
    return reference


def _served_context_window(serve: dict[str, Any], explicit: int | None) -> int:
    value = explicit if explicit is not None else serve.get("engine", {}).get("max_model_len")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("served context window must be a positive integer")
    return value


def build_verifiers_config(
    environment: EnvironmentSpec,
    *,
    model: str,
    base_url: str,
    api_key_var: str,
    output_dir: Path,
    chat_template_kwargs: dict[str, Any],
    sampling_overrides: dict[str, Any] | None = None,
    context_window: int | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    sampling = dict(environment.sampling)
    if sampling_overrides:
        sampling.update(sampling_overrides)
    if chat_template_kwargs:
        sampling["chat_template_kwargs"] = dict(chat_template_kwargs)
    config = {
        "model": model,
        "client": {"type": "eval", "base_url": base_url, "api_key_var": api_key_var},
        "sampling": sampling,
        "taskset": dict(environment.taskset),
        "harness": dict(environment.harness),
        "timeout": dict(environment.timeout),
        "num_tasks": environment.num_tasks,
        "num_rollouts": environment.num_rollouts,
        "max_concurrent": environment.max_concurrent,
        "rich": False,
        "verbose": verbose,
        "push": False,
        "server": False,
        "dry_run": dry_run,
        "output_dir": str(output_dir.resolve()),
    }
    if context_window is not None:
        config["max_total_tokens"] = context_window
    return config


def _validate_eval_budgets(environments: tuple[EnvironmentSpec, ...], context_window: int) -> None:
    for environment in environments:
        max_tokens = environment.sampling.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError(f"{environment.id}.sampling.max_tokens must be a positive integer")
        if max_tokens >= context_window:
            raise ValueError(
                f"{environment.id}.sampling.max_tokens ({max_tokens}) must be smaller than "
                f"the served context window ({context_window})"
            )


def _write_toml(config: dict[str, Any], path: Path) -> None:
    try:
        import tomli_w
    except ImportError as error:
        raise RuntimeError(
            "Verifiers dependencies are not installed; sync packages/eval with --extra verifiers"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(config), encoding="utf-8")


def _run_native(
    config: dict[str, Any],
    config_path: Path,
    log_path: Path,
    trace_sync: VerifiersTraceSynchronizer | None = None,
) -> TraceSyncStats | None:
    _write_toml(config, config_path)
    executable = Path(sys.executable).with_name("eval")
    if not executable.is_file():
        raise RuntimeError("Verifiers eval executable is unavailable; sync packages/eval with --extra verifiers")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(executable), "@", str(config_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while process.poll() is None:
                if trace_sync is not None:
                    trace_sync.drain()
                time.sleep(0.25)
        finally:
            if trace_sync is not None:
                trace_sync.finalize()
        returncode = process.returncode
    if returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Verifiers evaluation failed with exit code {returncode}:\n{tail}")
    return trace_sync.stats if trace_sync is not None else None


def _select(environments: tuple[EnvironmentSpec, ...], ids: list[str] | None) -> tuple[EnvironmentSpec, ...]:
    selected = tuple(item for item in environments if not ids or item.id in ids)
    if not selected:
        raise ValueError("environment filters selected no evaluations")
    unknown = set(ids or ()) - {item.id for item in environments}
    if unknown:
        raise ValueError(f"unknown environment IDs: {', '.join(sorted(unknown))}")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolver = ProfileResolver(PROFILES_DIR)
    model_profile = resolver.resolve("models", args.model_profile)
    suite = load_suite(_eval_path(_eval_reference(model_profile.data, args.eval_profile)))
    environments = _select(suite.environments, args.environment_ids)
    repository = _hf_repository(model_profile.data["model"]["artifact"])
    served_model = args.served_model or repository
    reasoning_mode, chat_template_kwargs, sampling_overrides = _reasoning_settings(
        model_profile.data, args.reasoning_mode
    )
    serve_reference = _serve_reference(model_profile.data, args.serve_profile)
    serve_profile = resolver.resolve("serve", serve_reference)
    if serve_profile.data.get("backend") != "vllm":
        raise ValueError("the current eval client requires an OpenAI-compatible vLLM endpoint")
    context_window = _served_context_window(serve_profile.data, args.served_context_window)
    native_context = model_profile.data["model"]["capabilities"]["context_window"]
    if context_window > native_context:
        raise ValueError(f"served context window ({context_window}) exceeds model capability ({native_context})")
    _validate_eval_budgets(environments, context_window)
    execution_id = uuid.uuid4().hex[:10]
    plan = {
        "suite_id": suite.id,
        "suite_execution_id": execution_id,
        "evaluation_kind": suite.evaluation_kind,
        "model_profile": model_profile.data["id"],
        "served_model": served_model,
        "serve_profile": serve_profile.reference,
        "served_context_window": context_window,
        "base_url": args.base_url,
        "reasoning_mode": reasoning_mode,
        "environments": [item.resolved_config() for item in environments],
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    if args.validate_only:
        root = Path("runs") / "eval-validation" / execution_id
        for environment in environments:
            output = root / environment.id / "native"
            config = build_verifiers_config(
                environment,
                model=served_model,
                base_url=args.base_url,
                api_key_var=args.api_key_var,
                output_dir=output,
                chat_template_kwargs=chat_template_kwargs,
                sampling_overrides=sampling_overrides,
                context_window=context_window,
                verbose=args.verbose,
                dry_run=True,
            )
            _run_native(config, root / environment.id / "input.toml", root / environment.id / "validate.log")
        return 0

    failures: list[str] = []
    for environment in environments:
        run_config = {
            **plan,
            "environment": environment.resolved_config(),
            "api_key_var": args.api_key_var,
        }
        run_kind = "general-eval" if suite.evaluation_kind == "general" else "domain-eval"
        run_name = f"{model_profile.data['id']}-{environment.id}-{execution_id}"
        try:
            with TrackedRun.start(
                run_kind,
                run_config,
                resolved_profile=model_profile,
                name=run_name,
            ) as run:
                native_dir = run.context.output_dir / "native"
                native_config = build_verifiers_config(
                    environment,
                    model=served_model,
                    base_url=args.base_url,
                    api_key_var=args.api_key_var,
                    output_dir=native_dir,
                    chat_template_kwargs=chat_template_kwargs,
                    sampling_overrides=sampling_overrides,
                    context_window=context_window,
                    verbose=args.verbose,
                )
                trace_sync = VerifiersTraceSynchronizer(
                    native_dir / "traces.jsonl",
                    lambda records: run.log(
                        {"eval/verifiers_trace": [trackio.VerifiersTrace(record) for record in records]}
                    ),
                )
                sync_stats = _run_native(
                    native_config,
                    run.context.recovery_dir / "verifiers-input.toml",
                    run.context.output_dir / "verifiers-driver.log",
                    trace_sync,
                )
                metrics = summarize_traces(native_dir / "traces.jsonl")
                sync_metrics = sync_stats.metrics() if sync_stats is not None else {}
                summary: dict[str, Any] = {
                    **metrics,
                    **sync_metrics,
                    "eval/category": environment.category,
                    "eval/environment_id": environment.id,
                }
                (run.context.output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                run.log({**metrics, **sync_metrics})
                run.log_artifact(
                    run.context.output_dir,
                    name=f"{run_name}-evaluation",
                    artifact_type="evaluation",
                    aliases=("latest",),
                    metadata={
                        "suite_id": suite.id,
                        "environment_id": environment.id,
                        "category": environment.category,
                    },
                )
                print(json.dumps({"run": run_name, **summary}, sort_keys=True), flush=True)
        except Exception:
            failures.append(environment.id)
            if not args.continue_on_error:
                raise
    if failures:
        print(json.dumps({"failed_environments": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
