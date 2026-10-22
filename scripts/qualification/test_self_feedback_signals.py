"""Check budget contracts and conservative goal-linked failure feedback."""

import hashlib
import json
import unittest

from self_feedback_signals import authority_errors, goal_links, repair_candidates, requested_budget, teacher_message


class Signals(unittest.TestCase):
    def test_authority_mismatch_is_not_a_goal_failure_label(self):
        initial = {'gmail': {'messages': []}}
        digest = hashlib.sha256(json.dumps(initial, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        manifest = {'bindings': [{'path': ['task_evidence', 'initial'], 'canonical_sha256': digest}]}
        self.assertEqual(authority_errors(manifest, initial, []), [])
        self.assertEqual(authority_errors(manifest, {}, [])[0]['reason'], 'manifest_authority_digest_mismatch')

    def test_repair_proposals_use_observed_resources_and_do_not_override_explicit_addresses(self):
        call = {"message": {"tool_calls": [{"function": {"name": "google_sheets_get_spreadsheet_by_id",
                "arguments": '{"spreadsheet":"Employee Departures (ss_offboard)"}'}}]}}
        feedback = json.dumps([{"tool": "google_sheets_get_spreadsheet_by_id", "response": json.dumps({
            "success": False, "error": "Not found. Known: ['Employee Departures (ss_offboard)']"})}])
        self.assertEqual(repair_candidates(call, feedback)[0]['arguments']['spreadsheet'], 'ss_offboard')
        call = {"messages": [{"role": "user", "content": "Find Anika Patel's email"}],
                "message": {"tool_calls": [{"function": {"name": "gmail_find_email",
                              "arguments": '{"query":"from:anika.patel@company.com"}'}}]}}
        feedback = json.dumps([{"tool": "gmail_find_email", "response": '{"success":true,"messages":[],"result_count":0}'}])
        self.assertEqual(repair_candidates(call, feedback)[0]['arguments']['query'], 'from:anika.patel')
        call['messages'][0]['content'] = 'Find anika.patel@company.com'
        self.assertEqual(repair_candidates(call, feedback), [])

    def setUp(self):
        self.goals = {"declared_checks": [
            {"id": "read", "role": "goal", "source": "reads", "join_sources": []},
            {"id": "post", "role": "goal", "source": "posts", "join_sources": ["reads"]}],
            "sources": {"reads": {"adapter": "gmail.message_reads@1"},
                        "posts": {"adapter": "slack.messages@1", "kind": "channel_message"}}}

    def test_related_tool_is_not_automatic_completion(self):
        links = goal_links("gmail_find_email", self.goals)
        self.assertEqual([v["relation"] for v in links], ["candidate_evidence_for_check", "prerequisite_evidence"])
        self.assertTrue(all(v["completion"] == "not evaluated" for v in links))
        self.assertEqual(goal_links("unknown_tool", self.goals), [])

    def test_empty_success_is_no_evidence_not_a_failed_global_goal(self):
        feedback = json.dumps([{"tool": "gmail_find_email", "response": json.dumps({
            "success": True, "messages": [], "result_count": 0})}])
        message = teacher_message({}, feedback, self.goals)
        self.assertIn("empty_search_result", message)
        self.assertIn("no matching message-read evidence produced", message)
        self.assertIn('"check_id": "read"', message)
        self.assertNotIn('"completion": "failed"', message)
        self.assertIn("unfinished goal alone does not", message)
        call = {"message": {"tool_calls": [{"function": {"name": "gmail_find_email", "arguments": '{"query":"wrong"}'}}]},
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "prior", "function": {
                        "name": "gmail_find_email", "arguments": '{"query":"wrong"}'}}]},
                    {"role": "tool", "tool_call_id": "prior", "content": json.loads(feedback)[0]["response"]}]}
        repeated = teacher_message(call, feedback, self.goals)
        self.assertIn('"status": "repeated_empty_search"', repeated)
        self.assertIn('"prior_identical_empty_results": 1', repeated)

    def test_request_budget_and_context_are_both_enforced(self):
        self.assertEqual(requested_budget({"max_tokens": 127}, 2048, 16384, 16000), (127, True))
        self.assertEqual(requested_budget({"max_tokens": 9000}, 2048, 16384, 15000), (2048, False))
        self.assertEqual(requested_budget({}, 2048, 16384, 14336), (2048, True))
        self.assertEqual(requested_budget({"max_completion_tokens": None, "max_tokens": 100}, 2048, 16384, 0), (100, True))
        with self.assertRaises(ValueError):
            requested_budget({"max_tokens": 0}, 2048, 16384, 0)

    def test_minimal_control_is_unchanged(self):
        self.assertEqual(teacher_message({}, "ERROR"),
                         "Observed tool response to an attempted action: ERROR\nChoose the action for the original request.")


if __name__ == "__main__":
    unittest.main()
