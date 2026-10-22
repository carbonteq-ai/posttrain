"""A valid zero-score text-only policy remains evaluable without training transitions."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from self_campaign_config import comparisons, summarize


def test_native_budget_and_text_only_failure_population_are_preserved(tmp_path):
    source = ast.parse(Path(__file__).with_name("self_reserved_transfer_eval.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "ReservedEvaluation")
    names = [f"task-{i}" for i in range(24)]
    episodes = [{"id": f"{name}-{i}", "task": {"data": {"name": name}}, "traces": [{
        "stop_condition": "agent_completed", "metrics": {"task_completed_correctly": 0.},
        "rewards": {"partial_credit": {"score": 0.}}, "nodes": []}]} for name in names for i in range(2)]
    observed = {}

    class Config:
        @staticmethod
        def model_validate(value):
            observed.update(value)
            return value

    async def run_eval(config):
        return episodes

    def write(path, value):
        path.write_text(json.dumps(value))

    def environment(turns, tools):
        assert turns == 8 and tools == "limited_zapier"
        return {"agent": {}, "taskset": {"task": {}}}

    namespace = {"Campaign": object, "environment_config": environment, "EvalConfig": Config,
                 "run_eval": run_eval, "write": write, "summarize": summarize, "comparisons": comparisons}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "reserved-evaluation", "exec"), namespace)
    owner = namespace["ReservedEvaluation"](SimpleNamespace(output=tmp_path, run_name="test"))
    owner.rollout = SimpleNamespace(version=0)
    asyncio.run(owner.collect(0, names))
    assert observed["sampling"] == {"temperature": 0., "max_tokens": 2048}
    assert observed["env"]["agent"]["max_output_tokens"] == 12288
    assert observed["num_rollouts"] == 2 and observed["max_concurrent"] == 32
    assert observed["task_keys"] == names
    assert owner.curve[0]["summary"]["metrics"]["task_completed_correctly"]["full_population_task_mean"] == 0
    admission = json.loads((tmp_path / "step-0000/reserve_transfer/evaluation-admission.json").read_text())
    assert admission["optimizer_updates"] == 0 and admission["training_allowed"] is False
