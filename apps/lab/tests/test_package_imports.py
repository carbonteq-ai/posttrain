"""Import boundaries for worker-safe lab subpackages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_environment_submodule_does_not_eagerly_import_tracking() -> None:
    repository = Path(__file__).resolve().parents[3]
    source_paths = (
        repository / "packages" / "common" / "src",
        repository / "packages" / "data" / "src",
        repository / "packages" / "train" / "src",
        repository / "apps" / "lab" / "src",
    )
    script = """
import sys
import posttrain_lab.environments.gsm8k_grpo
assert 'posttrain.tracking' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ":".join(str(path) for path in source_paths)},
    )
    assert completed.returncode == 0, completed.stderr
