"""Teacher in the evaluation loop (plan v4 P4): the teacher plays the agent inside the 4B's own evaluation loop.

The 4B's evaluation (wt-eval2 `eval-v2` sdpo_ab_runner.py, Runtime.collect) calls verifiers run_eval with
sdpo_ab_replay.environment(...) and an `eval` client pointed at a local /v1/chat/completions endpoint. This module
builds the same EvalConfig with the client pointed at teacher_proxy.py instead, so the teacher gets the 4B's exact
system prompt, tools, tool execution, turn budget, limits and scoring; episodes are written in the runner's
episodes.jsonl.gz format, and episode_rows.py turns them into training rows.

The evaluation code is imported read-only from the wt-eval2 worktree; its HEAD and the bytes of the files used are
checked before anything runs. The environment comes from PYTHONPATH (the pinned env-fix worktree, via
tools/pyfix/python), and its commit is recorded.

usage (environment interpreter, CPU):
  POSTTRAIN_VERIFIERS_PREINSTALLED=1 .../tools/pyfix/python teacher_runner.py --tasks T1 T2 --attempts 1 \
      --base-url http://127.0.0.1:PORT/v1 --output DIR [--label teacher] [--concurrency 4] \
      [--max-turns 50 --context-limit 131072 --reply-limit 32768 --rollout-timeout 10800]
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from settings import ENV_WT, REPO as WT_EVAL  # noqa: E402

EVAL_HEAD = os.environ.get("SFTGEN_EVAL_HEAD", "a89d7ad6cc9ed86bb4a3416c14e2a44cb8e2a507")  # a89d7ad6 = c9f8bc37 + thinking-aware runaway flag (runner only; replay files unchanged)
EVAL_FILES = ("scripts/qualification/sdpo_ab_replay.py", "scripts/qualification/sdpo_frozen_environment.py",
              "scripts/qualification/sdpo_ab_execution.py")
WORLD_DATE = "2026-10-28"  # D31: the world date of every evaluation run (launch_when_free_wt.sh SDPO_AB_WORLD_DATE)
PREINSTALLED_DIR = WT_EVAL / "packages/environment/src/posttrain/environment"


class FrozenCodeMismatch(RuntimeError):
    pass


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def frozen_code_check(root: Path = WT_EVAL, expected_head: str = EVAL_HEAD) -> dict:
    head = git(root, "rev-parse", "HEAD")
    if head != expected_head:
        raise FrozenCodeMismatch(f"{root} HEAD is {head}, expected {expected_head}")
    digests = {}
    for rel in EVAL_FILES:
        committed = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{rel}"], capture_output=True,
                                   check=True).stdout
        on_disk = (root / rel).read_bytes()
        if committed != on_disk:
            raise FrozenCodeMismatch(f"{root / rel} differs from its committed bytes")
        digests[rel] = hashlib.sha256(on_disk).hexdigest()
    return {"worktree": str(root), "head": head, "files_sha256": digests}


def environment_check(root: Path = ENV_WT) -> dict:
    if git(root, "status", "--porcelain"):
        raise FrozenCodeMismatch(f"{root} has uncommitted changes")
    src = str(root / "environments/automationbench_v1/src")
    if src not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        raise FrozenCodeMismatch(f"PYTHONPATH does not load the pinned environment {src}")
    return {"worktree": str(root), "commit": git(root, "rev-parse", "HEAD")}


def import_frozen(root: Path = WT_EVAL):
    qualification = str(root / "scripts/qualification")
    if qualification not in sys.path:
        sys.path.insert(0, qualification)
    import sdpo_ab_execution
    import sdpo_ab_replay

    for module in (sdpo_ab_replay, sdpo_ab_execution):
        if Path(module.__file__).resolve().parent != (root / "scripts/qualification").resolve():
            raise FrozenCodeMismatch(f"{module.__name__} imported from {module.__file__}, not {root}")
    return sdpo_ab_replay.environment, sdpo_ab_execution.pin_world_clock


def eval_config_dict(environment, names, attempts: int, *, base_url: str, concurrency: int, output: Path,
                     label: str, max_turns: int, context_limit: int, reply_limit: int, rollout_timeout: int) -> dict:
    """Runtime.collect's EvalConfig (eval-v2) with the teacher proxy as the endpoint. Sampling and max_tokens are
    set again by the proxy; the values here only satisfy the client."""
    env = environment(names, max_turns, max_output_tokens=context_limit, rollout_timeout=rollout_timeout)
    return {
        "env": env, "model": "teacher",
        "client": {"type": "eval", "base_url": base_url, "api_key_var": "SFTGEN_PROXY_KEY"},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "max_tokens": reply_limit},
        "num_tasks": len(names), "task_keys": list(names), "num_rollouts": attempts,
        "max_concurrent": concurrency, "shuffle": False, "output_dir": output / "native",
        "run": {"name": label, "dir": "."}, "push": False, "rich": None, "serve": None}


def configure_runtime() -> bool:
    if os.environ.get("POSTTRAIN_VERIFIERS_PREINSTALLED") != "1":
        raise RuntimeError("set POSTTRAIN_VERIFIERS_PREINSTALLED=1 (no network installs during a rollout)")
    # Load the one helper module by path: putting its folder on sys.path would shadow the `requests` library with
    # posttrain/environment/requests.py and break the environment's imports.
    import importlib.util

    spec = importlib.util.spec_from_file_location("verifiers_preinstalled", PREINSTALLED_DIR / "verifiers_preinstalled.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configure_preinstalled_runtime(interpreter=sys.executable)


async def collect(names, attempts, *, base_url, output: Path, label="teacher", concurrency=1, max_turns=50,
                  context_limit=131072, reply_limit=32768, rollout_timeout=10800) -> list[dict]:
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.cli.eval import EvalConfig

    frozen = frozen_code_check()
    env_commit = environment_check()
    environment, pin_world_clock = import_frozen()
    if os.environ.get("SDPO_AB_WORLD_DATE", WORLD_DATE) != WORLD_DATE:
        raise FrozenCodeMismatch(f"SDPO_AB_WORLD_DATE is {os.environ['SDPO_AB_WORLD_DATE']}, evaluations use {WORLD_DATE}")
    os.environ["SDPO_AB_WORLD_DATE"] = WORLD_DATE
    pinned = pin_world_clock()  # as the runner's main(): world clock 2026-10-28 UTC, TZ=UTC
    if str(pinned) != WORLD_DATE or os.environ.get("TZ") != "UTC":
        raise FrozenCodeMismatch(f"world clock not pinned ({pinned}, TZ={os.environ.get('TZ')})")
    configure_runtime()
    os.environ.setdefault("SFTGEN_PROXY_KEY", "local-proxy")  # the proxy holds the provider key; this is not it
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config_dict = eval_config_dict(environment, names, attempts, base_url=base_url, concurrency=concurrency,
                                   output=output, label=label, max_turns=max_turns, context_limit=context_limit,
                                   reply_limit=reply_limit, rollout_timeout=rollout_timeout)
    started = time.time()
    episodes = await run_eval(EvalConfig.model_validate(config_dict))
    records = [e.model_dump(mode="json") for e in episodes]
    with gzip.open(output / "episodes.jsonl.gz", "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary = {"label": label, "tasks": list(names), "attempts": attempts, "episodes": len(records),
               "base_url": base_url, "eval_code": frozen, "environment": env_commit,
               "world_date": WORLD_DATE, "tz": os.environ.get("TZ"),
               "limits": {"max_turns": max_turns, "context": context_limit, "reply": reply_limit,
                          "rollout_timeout_s": rollout_timeout},
               "elapsed_s": time.time() - started, "config": json.loads(json.dumps(config_dict, default=str))}
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="teacher")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--context-limit", type=int, default=131072)
    parser.add_argument("--reply-limit", type=int, default=32768)
    parser.add_argument("--rollout-timeout", type=int, default=10800)
    args = parser.parse_args(argv)
    records = asyncio.run(collect(args.tasks, args.attempts, base_url=args.base_url, output=args.output,
                                  label=args.label, concurrency=args.concurrency, max_turns=args.max_turns,
                                  context_limit=args.context_limit, reply_limit=args.reply_limit,
                                  rollout_timeout=args.rollout_timeout))
    print(json.dumps({"episodes": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
