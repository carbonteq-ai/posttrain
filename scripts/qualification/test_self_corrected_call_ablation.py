"""Qualification requires native improvement and retention, not just lower loss."""

import ast
import copy
import unittest
from pathlib import Path


def qualify(signal, baseline, after):
    source = ast.parse(Path(__file__).with_name("self_corrected_call_ablation.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "qualify")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "qualification", "exec"), namespace)
    return namespace["qualify"](signal, baseline, after)


class AblationChecks(unittest.TestCase):
    def setUp(self):
        self.baseline = {}
        for cohort in ["failures", "retention"]:
            value = 0. if cohort == "failures" else 1.
            metrics = {name: {"complete_task_means": dict.fromkeys("abcd", value),
                              "full_population_task_mean": value}
                       for name in ["task_completed_correctly", "partial_credit"]}
            self.baseline[cohort] = {"summary": {"metrics": metrics}}
        self.after = copy.deepcopy(self.baseline)
        for metric in self.after["failures"]["summary"]["metrics"].values():
            metric["complete_task_means"]["a"] = 1.
            metric["full_population_task_mean"] = .25

    def test_positive_gradients_and_native_improvement_can_qualify(self):
        self.assertTrue(qualify(dict.fromkeys("abcd", 1.), self.baseline, self.after)["qualified"])

    def test_no_reward_gain_or_bad_gradient_rejects(self):
        self.assertFalse(qualify(dict.fromkeys("abcd", 1.), self.baseline, self.baseline)["qualified"])
        self.assertFalse(qualify({"a": -1., "b": 1.}, self.baseline, self.after)["qualified"])

    def test_retention_regression_rejects(self):
        metric = self.after["retention"]["summary"]["metrics"]["partial_credit"]
        metric["complete_task_means"]["a"] = 0.
        self.assertFalse(qualify(dict.fromkeys("abcd", 1.), self.baseline, self.after)["qualified"])

    def test_missing_task_scores_reject(self):
        del self.after["failures"]["summary"]["metrics"]["partial_credit"]["complete_task_means"]["a"]
        with self.assertRaises(ValueError):
            qualify(dict.fromkeys("abcd", 1.), self.baseline, self.after)


if __name__ == "__main__":
    unittest.main()
