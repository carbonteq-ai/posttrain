from __future__ import annotations

from pathlib import Path

from posttrain_lab.qualification.scenarios import scenario_by_id

from scripts.qualification.run_algorithm_scenario import render_launch


def test_automationbench_launch_uses_posttrain_job_cli() -> None:
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

    assert launch.command[:5] == ("uv", "run", "posttrain", "job", "run")
    assert launch.command[5] == (".posttrain/work_packages/automationbench_zapier_grpo.yaml")
    assert launch.command[launch.command.index("--job") + 1] == "grpo"
    assert launch.command[launch.command.index("--provider") + 1] == "local"
    assert launch.command[launch.command.index("--run-id") + 1] == "run-1"


def test_dstack_launch_uses_container_run_workspace() -> None:
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

    assert launch.command[launch.command.index("--provider") + 1] == "dstack"
    assert launch.command[launch.command.index("--target") + 1] == ("carbonteq-ai-workstation.lan")
    assert launch.job_workspace == Path("/opt/posttrain/run")


def test_gsm8k_launch_uses_the_same_cli_with_its_work_package() -> None:
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

    assert launch.command[5] == (".posttrain/work_packages/gsm8k_qwen08b_grpo_qualification.yaml")
    assert launch.command[launch.command.index("--job") + 1] == "grpo"


def test_cli_default_preserves_virtual_environment_entrypoint() -> None:
    from scripts.qualification.run_algorithm_scenario import DEFAULT_GPU_PYTHON

    assert str(DEFAULT_GPU_PYTHON) == "/home/hammad/projects/verl/.venv313/bin/python"
