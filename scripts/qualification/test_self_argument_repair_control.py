"""Token-span selection must exclude unchanged syntax and preserve full inputs."""

import ast
import unittest
from pathlib import Path


def span(original, corrected):
    tree = ast.parse(Path(__file__).with_name("self_argument_repair_control.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "changed_span")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "span-selection", "exec"), namespace)
    return namespace["changed_span"](original, corrected)


class SpanChecks(unittest.TestCase):
    def test_different_length_repair_only_supervises_changed_tokens(self):
        self.assertEqual(span([1, 2, 9, 3, 4], [1, 2, 7, 8, 3, 4]), (2, 4))

    def test_unchanged_suffix_not_accidentally_supervised(self):
        self.assertEqual(span([1, 9, 5, 6], [1, 3, 5, 6]), (1, 2))

    def test_missing_change_rejected(self):
        with self.assertRaises(StopIteration):
            span([1, 2], [1, 2])


if __name__ == "__main__":
    unittest.main()
