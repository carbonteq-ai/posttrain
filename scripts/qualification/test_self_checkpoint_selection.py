"""Selection must not peek at heldout or hide missing development scores."""

import copy
import unittest

from self_checkpoint_selection import choose


class SelectionChecks(unittest.TestCase):
    def setUp(self):
        self.tasks = [f"task-{i}" for i in range(24)]
        self.curve = []
        for step in range(0, 257, 32):
            metrics = {}
            for name, value in [("task_completed_correctly", .5), ("partial_credit", .6), ("tool_mistakes", 1.)]:
                metrics[name] = {"episode_coverage": 48, "total_episodes": 48,
                                 "complete_task_means": dict.fromkeys(self.tasks, value),
                                 "full_population_task_mean": value}
            self.curve.append({"step": step, "cohort": "development",
                               "summary": {"tasks": 24, "episodes": 48, "errors": [], "metrics": metrics}})

    def test_ties_select_earlier_trained_checkpoint_and_add_it_to_sweep(self):
        result = choose(self.curve, self.tasks)
        self.assertEqual(result["primary_checkpoint"], 32)
        self.assertEqual(result["evaluation_steps"], [0, 32, 64, 128, 192, 256])

    def test_non_development_outcomes_cannot_change_selection(self):
        self.curve.append({"step": 256, "cohort": "heldout", "summary": "ignored"})
        self.assertEqual(choose(self.curve, self.tasks)["primary_checkpoint"], 32)

    def test_primary_order_prefers_success_then_partial_then_tool_mistakes(self):
        better = self.curve[4]["summary"]["metrics"]
        for key, value in [("task_completed_correctly", .75), ("partial_credit", .8), ("tool_mistakes", 0.)]:
            better[key]["full_population_task_mean"] = value
            better[key]["complete_task_means"] = dict.fromkeys(self.tasks, value)
        self.assertEqual(choose(self.curve, self.tasks)["primary_checkpoint"], 128)

    def test_incomplete_or_duplicate_rounds_rejected(self):
        for curve in [self.curve[:-1], self.curve + [self.curve[-1]]]:
            with self.assertRaises(ValueError):
                choose(curve, self.tasks)

    def test_missing_scores_are_not_dropped_or_zero_filled(self):
        curve = copy.deepcopy(self.curve)
        del curve[1]["summary"]["metrics"]["partial_credit"]["complete_task_means"][self.tasks[0]]
        with self.assertRaises(ValueError):
            choose(curve, self.tasks)


if __name__ == "__main__":
    unittest.main()
