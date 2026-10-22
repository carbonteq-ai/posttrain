"""Admission and exposure checks, with no model/runtime dependency."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from self_pilot_config import admit_sequences, resolve_pilot_tasks, sample_transitions, validate_sequences


class PilotChecks(unittest.TestCase):
    def test_actual_protocol_and_leak_rejection(self):
        root = Path(__file__).resolve().parents[2]
        protocol_path = root / "docs/research/evidence/self-vllm-20261017/task-protocol.json"
        heldout_path = root / "scripts/qualification/fixtures/automationbench_heldout62_v1.json"
        names = resolve_pilot_tasks(protocol_path, heldout_path)
        self.assertEqual(len(names), 32)
        protocol = json.loads(protocol_path.read_text())
        protocol["cohorts"]["development"]["task_names"].append(names[0])
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "protocol.json"
            candidate.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "leaks"):
                resolve_pilot_tasks(candidate, heldout_path)

    def test_long_trajectory_does_not_dominate(self):
        rows = [{"trajectory_id": "short"}] + [{"trajectory_id": "long", "turn": i} for i in range(100)]
        sampled = sample_transitions(rows, 8, seed=17)
        self.assertEqual(sum(row["trajectory_id"] == "long" for row in sampled), 4)
        self.assertEqual(sampled, sample_transitions(rows, 8, seed=17))

    def test_teacher_context_is_checked_before_updates(self):
        row = dict.fromkeys(("student_prefix_ids", "teacher_prefix_ids", "feedback_prefix_ids", "action_ids", "feedback_ids"), [1])
        validate_sequences([row], 4)
        oversized = copy.deepcopy(row)
        oversized["teacher_prefix_ids"] = [1] * 4
        with self.assertRaisesRegex(ValueError, "teacher_prefix_ids"):
            validate_sequences([oversized], 4)

    def test_overlong_transition_is_excluded_without_mutation(self):
        row = dict.fromkeys(("student_prefix_ids", "teacher_prefix_ids", "feedback_prefix_ids", "action_ids", "feedback_ids"), [1])
        row["trajectory_id"] = "episode-1"
        original = copy.deepcopy(row)
        long_row = copy.deepcopy(row)
        long_row["feedback_ids"] = [2] * 16384
        admitted, excluded = admit_sequences([row, long_row], 16384)
        self.assertEqual(admitted, [original])
        self.assertIs(admitted[0], row)
        self.assertEqual(len(long_row["feedback_ids"]), 16384)
        self.assertEqual(len(excluded), 1)


if __name__ == "__main__":
    unittest.main()
