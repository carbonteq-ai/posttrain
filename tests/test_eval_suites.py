from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common import PROFILES_DIR
from eval.cli import build_verifiers_config
from eval.results import summarize_traces
from eval.suites import EvalSuiteError, load_suite


class EvalSuiteTest(unittest.TestCase):
    def test_general_smoke_has_four_pinned_categories(self) -> None:
        suite = load_suite(PROFILES_DIR / "eval" / "general-smoke-v1.yaml")

        self.assertEqual(suite.evaluation_kind, "general")
        self.assertEqual(len(suite.environments), 4)
        self.assertEqual(
            {environment.category for environment in suite.environments},
            {"math_reasoning", "instruction_following", "code_generation", "multi_turn_state"},
        )
        self.assertTrue(all(len(environment.source["revision"]) == 40 for environment in suite.environments))

    def test_code_execution_is_docker_isolated_and_group_scored(self) -> None:
        suite = load_suite(PROFILES_DIR / "eval" / "general-smoke-v1.yaml")
        code = next(item for item in suite.environments if item.id == "code-execution")

        self.assertEqual(code.harness["runtime"]["type"], "docker")
        self.assertEqual(code.num_rollouts, 2)
        self.assertEqual(code.max_concurrent, 1)

    def test_native_config_applies_model_chat_template_kwargs(self) -> None:
        suite = load_suite(PROFILES_DIR / "eval" / "general-smoke-v1.yaml")
        config = build_verifiers_config(
            suite.environments[0],
            model="Qwen/Qwen3.5-2B",
            base_url="http://127.0.0.1:8000/v1",
            api_key_var="LOCAL_INFERENCE_API_KEY",
            output_dir=Path("/tmp/eval"),
            chat_template_kwargs={"enable_thinking": False},
            sampling_overrides={"top_p": 0.95},
            context_window=32768,
            verbose=True,
        )

        self.assertEqual(
            config["sampling"]["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(config["sampling"]["top_p"], 0.95)
        self.assertEqual(config["max_total_tokens"], 32768)
        self.assertTrue(config["verbose"])
        self.assertFalse(config["push"])
        self.assertFalse(config["rich"])

    def test_rejects_unpinned_environment_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.yaml"
            path.write_text(
                """schema_version: 1
id: broken
evaluation_kind: general
defaults: {num_tasks: 1, num_rollouts: 1, max_concurrent: 1}
environments:
  - id: x
    category: x
    taskset: {id: x}
    harness: {id: 'null'}
    source: {repository: https://example.test/x}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvalSuiteError, "source.revision"):
                load_suite(path)


class EvalResultsTest(unittest.TestCase):
    def test_summarizes_native_trace_records(self) -> None:
        traces = [
            {
                "is_completed": True,
                "errors": [],
                "rewards": {"correct": 1.0},
                "calls": [
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": 10,
                            "cached_input_tokens": 2,
                            "completion_tokens": 4,
                            "reasoning_tokens": 3,
                            "cost": 0.01,
                        },
                        "time": {"start": 1.0, "end": 3.0},
                    }
                ],
                "metrics": {"format": 0.75},
                "timing": {
                    "boot": {"start": 1.0, "end": 2.0},
                    "setup": {"start": 2.0, "end": 3.0},
                    "generation": {
                        "start": 3.0,
                        "end": 7.0,
                        "model": {"duration": 2.0},
                        "harness": {"duration": 2.0},
                    },
                    "finalize": {"start": 7.0, "end": 8.0},
                    "scoring": {"start": 8.0, "end": 9.0},
                },
                "stop_condition": "agent_completed",
            },
            {
                "is_completed": False,
                "errors": [{"type": "ProviderError"}],
                "rewards": {"correct": 0.0},
                "calls": [
                    {
                        "finish_reason": "length",
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                        "time": {"start": 2.0, "end": 6.0},
                    }
                ],
                "metrics": {"format": 0.25},
                "timing": {},
                "stop_condition": "error",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in traces) + "\n")
            summary = summarize_traces(path)

        self.assertEqual(summary["eval/rollouts"], 2)
        self.assertEqual(summary["eval/completed"], 1)
        self.assertEqual(summary["eval/errors"], 1)
        self.assertEqual(summary["eval/mean_reward"], 0.5)
        self.assertEqual(summary["eval/input_tokens"], 17)
        self.assertEqual(summary["eval/output_tokens"], 5)
        self.assertEqual(summary["eval/model_calls"], 2)
        self.assertEqual(summary["eval/length_finished_calls"], 1)
        self.assertEqual(summary["eval/truncated_rollouts"], 1)
        self.assertEqual(summary["eval/truncated_rollout_rate"], 0.5)
        self.assertEqual(summary["eval/reasoning_tokens"], 3)
        self.assertEqual(summary["eval/provider_cost"], 0.01)
        self.assertEqual(summary["eval/model_call_latency_mean_seconds"], 3.0)
        self.assertEqual(summary["eval/environment_metric/format"], 0.5)
        self.assertEqual(summary["eval/reward/correct"], 0.5)
        self.assertEqual(summary["eval/stop/agent_completed"], 1)


if __name__ == "__main__":
    unittest.main()
