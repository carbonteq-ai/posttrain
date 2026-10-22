"""Independent finite-difference and consumed-sample checks for failure diagnostics."""

import ast
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch


def helpers():
    tree = ast.parse(Path(__file__).with_name("self_failure_diagnosis.py").read_text())
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef)
             and n.name in {"key", "cases_from", "projection"})
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in n.targets))]
    namespace = {"json": json, "Counter": Counter,
                 "repair_candidates": lambda *a: [{"function": "lookup", "arguments": {"key": "good"}}]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "diagnosis-functions", "exec"), namespace)
    return namespace


class DiagnosisChecks(unittest.TestCase):
    def test_projection_matches_independent_parameter_perturbation(self):
        values = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
        def margin(x):
            return (x[0] ** 2 + 2 * x[1]).sin()
        action = (values - torch.tensor([1., 2.])).square().mean()
        feedback = (values + torch.tensor([3., 4.])).square().mean()
        reference, = torch.autograd.grad(margin(values), values, retain_graph=True)
        g_action, = torch.autograd.grad(action, values, retain_graph=True)
        g_feedback, = torch.autograd.grad(feedback, values)
        direction = helpers()["projection"]([g_action + .2 * g_feedback], [reference])
        epsilon = 1e-7
        observed = (margin(values.detach() - epsilon * (g_action + .2 * g_feedback))
                    - margin(values.detach())).item() / epsilon
        self.assertAlmostEqual(direction, observed, places=5)

    def test_unconsumed_planned_examples_cannot_be_selected(self):
        ns = helpers()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "step-0064/train"
            (folder / "native").mkdir(parents=True)
            episodes, calls, rows = [], [], []
            consumed = []
            unused = []
            for index, task in enumerate(sorted(ns["TASKS"])):
                eid = f"episode-{index}"
                nodes = []
                for turn in [0, 1]:
                    tid = f"tool-{index}-{turn}"
                    nodes.append({"message": {"role": "tool", "tool_call_id": tid,
                                             "name": "lookup", "content": '{"success": false}'}})
                    calls.append({"complete": True, "prefix_ids": [index, turn], "action_ids": [3],
                                  "message": {"tool_calls": [{"id": tid, "function": {"name": "lookup"}}]}})
                    row = {"trajectory_id": eid, "student_prefix_ids": [index, turn], "action_ids": [3]}
                    rows.append(row)
                    (unused if turn == 0 else consumed).append(row)
                episodes.append({"id": eid, "task": {"data": {"name": task}}, "traces": [{"nodes": nodes}]})
            (folder / "native/traces.jsonl").write_text("\n".join(map(json.dumps, episodes)))
            (folder / "generation-calls.jsonl").write_text("\n".join(map(json.dumps, calls)))
            (folder / "transitions.json").write_text(json.dumps(rows))
            (root / "step-0064/sampled-training-transitions.json").write_text(json.dumps(consumed * 112 + unused * 16))
            (root / "updates.jsonl").write_text("{}\n" * 92)
            result = ns["cases_from"](root, 64)
            self.assertEqual(len(result), 4)
            self.assertTrue(all(r["row"]["student_prefix_ids"][-1] == 1 for r in result))
            self.assertTrue(all(r["sample_exposures"] == 112 and r["round_exposures_consumed"] == 448 for r in result))


if __name__ == "__main__":
    unittest.main()
