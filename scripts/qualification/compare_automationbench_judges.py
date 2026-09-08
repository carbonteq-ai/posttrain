"""Compare frozen AutomationBench turn-judge profiles without training.

The AutomationBench plugin owns the rubric, request schema, retries, and result
validation. This runner owns only sequential inference lifecycle and immutable
model/runtime selection. The five-case fixture is an engineering calibration,
not a held-out benchmark or proof that either model improves a policy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from automationbench_v1.judge import AutomationBenchTurnJudge, TurnQualityConfig
from calibrate_automationbench_judge import run as run_calibration

RUNTIME_REVISION = "62f6de733d7ae63b759329993bc209e67afdf431"
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
JUDGE_CODE_BLOB = "59dd00e20b42ee83d2e76f845d339185e51b3bc7"


@dataclass(frozen=True, slots=True)
class JudgeProfile:
    name: str
    repo_id: str
    revision: str
    reasoning_parser: str


PROFILES = (
    JudgeProfile(
        "nanbeige-3b",
        "Nanbeige/Nanbeige4.2-3B",
        "3384e426066d1a49c3aea90a7190b81260a6533f",
        "nanbeige",
    ),
    JudgeProfile(
        "gemma4-12b",
        "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "gemma4",
    ),
)


def _gpu_name() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()


def _wait_ready(process: subprocess.Popen[str], port: int, log_path: Path) -> float:
    started = time.monotonic()
    deadline = started + 1_200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"judge server exited with code {process.returncode}:\n"
                + log_path.read_text(errors="replace")[-20_000:]
            )
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
                f"http://127.0.0.1:{port}/health",
                timeout=1,
            ):
                return time.monotonic() - started
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError("judge server readiness timed out")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def _command(profile: JudgeProfile, port: int) -> list[str]:
    return [
        "vllm",
        "serve",
        profile.repo_id,
        "--revision",
        profile.revision,
        "--served-model-name",
        "automationbench-judge",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.80",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--enable-chunked-prefill",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "8192",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0, "audio": 0}, separators=(",", ":")),
        "--skip-mm-profiling",
        "--reasoning-parser",
        profile.reasoning_parser,
    ]


def _judge_blob() -> str:
    source = inspect.getsourcefile(AutomationBenchTurnJudge)
    if source is None:
        raise RuntimeError("cannot resolve installed AutomationBench judge source")
    payload = Path(source).read_bytes()
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def _selection(
    profile: JudgeProfile,
    port: int,
    startup_seconds: float,
    *,
    enable_thinking: bool,
) -> dict[str, object]:
    sampling = {
        "temperature": 0.0,
        "max_tokens": 4096,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    }
    config = TurnQualityConfig(
        id="automationbench-v1",
        name="quality",
        model="automationbench-judge",
        model_revision=profile.revision,
        code_revision=JUDGE_CODE_BLOB,
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key_var="POSTTRAIN_CALIBRATION_API_KEY",
        sampling=sampling,
        input_budget_tokens=12_288,
        timeout_seconds=300,
    )
    judge = AutomationBenchTurnJudge(config)
    return {
        "scope": "provisional engineering calibration; not training or a held-out benchmark",
        "profile": profile.name,
        "judge_thinking": enable_thinking,
        "judge": config.model_dump(mode="json"),
        "projection": {"scorer_digest": judge.scorer_digest},
        "runtime": {
            "repository": "https://github.com/Nanbeige/vllm",
            "revision": RUNTIME_REVISION,
            "startup_seconds": startup_seconds,
        },
    }


def _run_profile(
    profile: JudgeProfile,
    output: Path,
    port: int,
    *,
    enable_thinking: bool,
    fixture: Path | None,
) -> dict[str, object]:
    root = output / profile.name
    root.mkdir(parents=True, exist_ok=False)
    command = _command(profile, port)
    log_path = root / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603 - frozen internal command
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            startup_seconds = _wait_ready(process, port, log_path)
            selection = _selection(
                profile,
                port,
                startup_seconds,
                enable_thinking=enable_thinking,
            )
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True))
            asyncio.run(
                run_calibration(
                    selection_path,
                    root / "calibration",
                    fixture=fixture,
                )
            )
        finally:
            _stop(process)
    summary = json.loads((root / "calibration" / "summary.json").read_text())
    fixture_manifest = json.loads((root / "calibration" / "fixture.json").read_text())
    return {
        "profile": profile.name,
        "judge_thinking": enable_thinking,
        "passed": summary["passed"],
        "scorer_digest": summary["scorer_digest"],
        "fixture_id": fixture_manifest.get("id"),
        "fixture_sha256": fixture_manifest["fixture_sha256"],
        "startup_seconds": startup_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if "@sha256:" not in args.runtime_image:
        parser.error("--runtime-image must be an immutable OCI digest reference")
    if _gpu_name() != EXPECTED_GPU_NAME:
        raise RuntimeError(f"comparison requires {EXPECTED_GPU_NAME}")
    if _judge_blob() != JUDGE_CODE_BLOB:
        raise RuntimeError("installed AutomationBench judge differs from frozen source blob")
    args.output.mkdir(parents=True, exist_ok=False)
    results = [
        _run_profile(
            profile,
            args.output,
            args.port,
            enable_thinking=args.enable_thinking,
            fixture=args.fixture,
        )
        for profile in PROFILES
    ]
    scope = (
        "external frozen fixture comparison; acceptance is owned by that fixture"
        if args.fixture is not None
        else "provisional engineering calibration; no winner declaration"
    )
    summary = {
        "scope": scope,
        "runtime_image": args.runtime_image,
        "judge_code_blob": JUDGE_CODE_BLOB,
        "judge_thinking": args.enable_thinking,
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
