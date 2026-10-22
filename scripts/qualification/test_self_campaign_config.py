"""Regression checks for campaign exposure accounting and missing-score reporting."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from self_campaign_config import comparisons, schedule, summarize


def episode(identity, name, success):
    return {"id": identity, "task": {"data": {"name": name}}, "traces": [
        {"stop_condition": "agent_completed", "metrics": {"task_completed_correctly": success},
         "rewards": {"partial_credit": {"score": success}}}]}


class CampaignChecks(unittest.TestCase):
    def test_loader_cannot_prefetch_next_round(self):
        # Exercise the actual iterator without importing GPU/runtime dependencies.
        source = ast.parse(Path(__file__).with_name("self_campaign.py").read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RoundDataset")
        namespace = {"IterableDataset": object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "campaign-iterator", "exec"), namespace)
        owner = SimpleNamespace(schedule=[(0, 1, 2), (1, 2, 2)],
                                samples={0: ["a", "b"]},
                                trainer=SimpleNamespace(state=SimpleNamespace(global_step=0)))
        iterator = iter(namespace["RoundDataset"](owner))
        self.assertEqual([next(iterator), next(iterator)], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "expected number of updates"):
            next(iterator)
        owner.trainer.state.global_step = 0
        iterator = iter(namespace["RoundDataset"](owner))
        next(iterator)
        next(iterator)
        owner.trainer.state.global_step = 1
        owner.samples[1] = ["fresh-c", "fresh-d"]
        self.assertEqual([next(iterator), next(iterator)], ["fresh-c", "fresh-d"])
        owner.trainer.state.global_step = 2
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_exposure_and_round_boundaries(self):
        rounds = schedule(256, 32, 16)
        self.assertEqual(len(rounds), 8)
        self.assertEqual(sum(r[2] for r in rounds), 4096)
        self.assertEqual(rounds[0], (0, 32, 512))
        self.assertEqual(rounds[-1], (224, 256, 512))
        with self.assertRaises(ValueError):
            schedule(255, 32, 16)

    def test_missing_scores_do_not_become_zero(self):
        result = summarize([episode("a", "one", 1), episode("b", "one", None)], ["one"], 2)
        metric = result["metrics"]["task_completed_correctly"]
        self.assertEqual(metric["episode_coverage"], 1)
        self.assertIsNone(metric["full_population_task_mean"])
        self.assertEqual(metric["complete_task_means"], {})
        self.assertIsNone(comparisons(result, result)["task_completed_correctly"]["paired"])

    def test_cohort_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize([episode("a", "one", 1)], ["one"], 2)
        with self.assertRaises(ValueError):
            summarize([episode("a", "one", 1), episode("a", "one", 1)], ["one"], 2)


if __name__ == "__main__":
    unittest.main()
