"""Statistical contract checks for the native pilot reducer."""

import unittest

from analyze_self_pilot import paired_bootstrap


class PairedUncertaintyChecks(unittest.TestCase):
    def test_constant_improvement_has_separate_absolute_and_delta_intervals(self):
        result = paired_bootstrap({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}, repeats=100)
        self.assertEqual(result["before_absolute_ci95"], [0.0, 0.0])
        self.assertEqual(result["after_absolute_ci95"], [1.0, 1.0])
        self.assertEqual(result["paired_delta_ci95"], [1.0, 1.0])

    def test_zero_average_can_still_have_uncertain_task_changes(self):
        result = paired_bootstrap({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0}, repeats=1000)
        self.assertEqual(result["delta"], 0.0)
        self.assertLess(result["paired_delta_ci95"][0], 0.0)
        self.assertGreater(result["paired_delta_ci95"][1], 0.0)

    def test_missing_tasks_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_bootstrap({"a": 1.0, "b": 0.0}, {"a": 1.0})


if __name__ == "__main__":
    unittest.main()
