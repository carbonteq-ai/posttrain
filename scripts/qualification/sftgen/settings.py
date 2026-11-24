"""Locations the SFT data pipeline reads and writes, each overridable through an environment variable.

| Variable | Default | What it is |
| --- | --- | --- |
| `SFTGEN_WORK` | `./sftgen-work` | folder for generation rounds, logs and built rows |
| `SFTGEN_REPO` | this repository | checkout holding the evaluation code (`scripts/qualification`) |
| `SFTGEN_ENV_WT` | `../verifiers-environments` | verifiers-environments checkout with the AutomationBench task fixes |
| `SFTGEN_ENV_PYTHON` | `python3` | interpreter that imports `automationbench_v1` from `SFTGEN_ENV_WT` |
| `SFTGEN_FRAMEWORK_PYTHON` | `.venv/bin/python` of this repository | interpreter with the posttrain framework |
| `SFTGEN_TOKENIZER` | `Qwen/Qwen3.5-4B` | tokenizer folder or Hugging Face id (revision `851bf6e8`) |
| `SFTGEN_CAPPED` | unset | optional command prefix that limits memory, e.g. `systemd-run --user --scope -p MemoryMax=8G` |
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

REPO = Path(os.environ.get("SFTGEN_REPO", Path(__file__).resolve().parents[3]))
WORK = Path(os.environ.get("SFTGEN_WORK", "sftgen-work"))
ENV_WT = Path(os.environ.get("SFTGEN_ENV_WT", REPO.parent / "verifiers-environments"))
ENV_PYTHON = os.environ.get("SFTGEN_ENV_PYTHON", "python3")
FRAMEWORK_PYTHON = os.environ.get("SFTGEN_FRAMEWORK_PYTHON", str(REPO / ".venv/bin/python"))
TOKENIZER = os.environ.get("SFTGEN_TOKENIZER", "Qwen/Qwen3.5-4B")
CAPPED = shlex.split(os.environ.get("SFTGEN_CAPPED", ""))
FROZEN_SETS = REPO / "scripts/qualification/fixtures/automationbench/sets-phase2-frozen.json"
