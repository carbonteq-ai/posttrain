"""Transfer selection is fixed by task identity, not adaptation outcomes."""

import ast
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch


def select(protocol, heldout):
    tree = ast.parse(Path(__file__).with_name("self_repair_transfer_eval.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "transfer_tasks")
    namespace = {"TASKS": set(range(4)), "RETENTION": list(range(4, 8))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "transfer-selection", "exec"), namespace)
    return namespace["transfer_tasks"](protocol, heldout)


class TransferChecks(unittest.TestCase):
    def test_excludes_adaptation_and_control_tasks(self):
        protocol = {"cohorts": {"pilot": {"task_names": list(range(32))},
                                 "development": {"task_names": list(range(32, 56))}}}
        self.assertEqual(select(protocol, list(range(56, 118))), list(range(8, 32)))

    def test_no_hidden_development_or_heldout_overlap(self):
        protocol = {"cohorts": {"pilot": {"task_names": list(range(32))},
                                 "development": {"task_names": [9]}}}
        with self.assertRaises(ValueError):
            select(protocol, [])
        protocol["cohorts"]["development"]["task_names"] = []
        with self.assertRaises(ValueError):
            select(protocol, [10])

    def test_export_check_detects_stale_or_wrong_student_adapter(self):
        tree = ast.parse(Path(__file__).with_name("self_repair_transfer_eval.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "matching_export")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, exported = root / "candidate", root / "export"
            for folder in [candidate, exported]:
                folder.mkdir()
                (folder / "adapter_model.safetensors").write_bytes(folder.name.encode())
            states = {str(candidate / "adapter_model.safetensors"): {"lora": torch.tensor([1.])},
                      str(exported / "adapter_model.safetensors"): {"lora": torch.tensor([2.])}}
            namespace = {"torch": torch, "hashlib": hashlib, "load_file": states.__getitem__}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "export-check", "exec"), namespace)
            with self.assertRaises(ValueError):
                namespace["matching_export"](candidate, exported)
            states[str(exported / "adapter_model.safetensors")] = {"lora": torch.tensor([1.])}
            self.assertTrue(namespace["matching_export"](candidate, exported)["tensor_equality_verified"])


if __name__ == "__main__":
    unittest.main()
