from __future__ import annotations

from pathlib import Path

from scripts.qualification.algorithm_scenarios import scenario_by_id
from scripts.qualification.run_algorithm_scenario import render_launch


def test_automationbench_launch_hides_raw_task_selection_from_operator() -> None:
    scenario = scenario_by_id("automationbench-qwen35-08b-grpo-10")

    launch = render_launch(
        scenario,
        run_id="run-1",
        provider="local",
        target="pop-os.lan",
        workspace=Path("/tmp/qualification-run"),
        python_executable=Path("/runtime/python"),
        trackio_server_url="https://trackio.lan",
    )

    assert launch.command[-2:] == ("--max-steps", "10")
    assert launch.command[launch.command.index("--num-generations") + 1] == "4"
    assert launch.command[launch.command.index("--task-indices") + 1 : -4] == (
        "194",
        "198",
    )


def test_dstack_launch_uses_container_runtime_and_run_workspace() -> None:
    scenario = scenario_by_id("automationbench-qwen35-08b-grpo-10")

    launch = render_launch(
        scenario,
        run_id="run-1",
        provider="dstack",
        target="carbonteq-ai-workstation.lan",
        workspace=Path("/tmp/qualification-run"),
        python_executable=Path("/runtime/python"),
        trackio_server_url="https://trackio.lan",
    )

    assert launch.command[0] == "/opt/venv/bin/python"
    assert launch.command[1] == "tools/run_verl_grpo_qualification.py"
    assert launch.job_workspace == Path("/opt/posttrain/run")
    assert launch.command[
        launch.command.index("--verl-worktree") + 1
    ] == "/workspace/verl-upstream"


def test_gsm8k_launch_uses_the_same_trainer_with_a_math_environment_adapter() -> None:
    scenario = scenario_by_id("gsm8k-qwen35-08b-grpo-15")

    launch = render_launch(
        scenario,
        run_id="run-1",
        provider="local",
        target="pop-os.lan",
        workspace=Path("/tmp/qualification-run"),
        python_executable=Path("/runtime/python"),
        trackio_server_url="https://trackio.lan",
    )

    assert launch.command[1].endswith("run_verl_grpo_qualification.py")
    assert launch.command[launch.command.index("--environment") + 1] == "gsm8k"
    assert launch.command[launch.command.index("--max-steps") + 1] == "15"
    assert launch.command[launch.command.index("--task-indices") + 1 : -4] == (
        "0",
        "1",
    )


def test_cli_default_preserves_virtual_environment_entrypoint() -> None:
    from scripts.qualification.run_algorithm_scenario import DEFAULT_GPU_PYTHON

    assert str(DEFAULT_GPU_PYTHON) == "/home/hammad/projects/verl/.venv313/bin/python"
