"""CPU checks run inside the staged native qualification runtime."""

import copy
import unittest
from types import SimpleNamespace

import torch
from self_feedback_native import collect_transitions


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}


class TransitionChecks(unittest.TestCase):
    def setUp(self):
        self.call = {
            "messages": [{"role": "user", "content": "Update the sheet."}],
            "tools": [],
            "prefix_ids": [1, 2],
            "action_ids": [3, 4],
            "complete": True,
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "random-transport-uuid",
                        "type": "function",
                        "function": {"name": "update_sheet", "arguments": "{}"},
                    }
                ],
            },
        }
        self.value = {
            "id": "episode-1",
            "traces": [
                {
                    "errors": [],
                    "stop_condition": "completed",
                    "nodes": [
                        {
                            "message": {
                                "role": "tool",
                                "tool_call_id": "random-transport-uuid",
                                "name": "update_sheet",
                                "content": '{"updated": true}',
                            }
                        }
                    ],
                }
            ],
        }

    def collect(self, value=None, call=None):
        episode = SimpleNamespace(model_dump=lambda **kwargs: self.value if value is None else value)
        return collect_transitions([self.call if call is None else call], [episode], Tokenizer(), "version-1")

    def test_original_action_and_observed_feedback(self):
        rows, omitted = self.collect()
        self.assertEqual(rows[0]["action_ids"], [3, 4])
        self.assertEqual(rows[0]["policy_revision"], "version-1")
        target = bytes(rows[0]["feedback_ids"]).decode()
        self.assertIn("updated", target)
        self.assertNotIn("random-transport-uuid", target)
        self.assertEqual(omitted, [])

    def test_missing_response_is_rejected(self):
        value = copy.deepcopy(self.value)
        value["traces"][0]["nodes"] = []
        with self.assertRaisesRegex(ValueError, "lacks its observed response"):
            self.collect(value=value)

    def test_truncated_action_is_rejected(self):
        call = dict(self.call, complete=False)
        with self.assertRaisesRegex(ValueError, "incomplete collection"):
            self.collect(call=call)

    def test_native_error_is_rejected(self):
        value = copy.deepcopy(self.value)
        value["traces"][0]["errors"] = [{"type": "TaskError"}]
        with self.assertRaisesRegex(ValueError, "native rollout has errors"):
            self.collect(value=value)

    def test_incomplete_action_can_be_excluded_and_retained(self):
        call = dict(self.call, complete=False, limit_reason="context_limit")
        call["action_ids"] = [7, 8, 9]
        episode = SimpleNamespace(model_dump=lambda **kwargs: self.value)
        exclusions = []
        rows, _ = collect_transitions([call], [episode], Tokenizer(), "version-1", exclusions)
        self.assertEqual(rows, [])
        self.assertEqual(exclusions[0]["reason"], "context_limit")
        self.assertEqual(call["action_ids"], [7, 8, 9])


if __name__ == "__main__":
    unittest.main()
