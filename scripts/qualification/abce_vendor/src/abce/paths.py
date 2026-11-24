"""Locations of the read-only sources this engine consumes.

Everything is overridable through environment variables so the engine can run against another
checkout. Nothing here writes to those locations.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO.parent

FORK = Path(os.environ.get("ABCE_FORK", RESEARCH / "automationbench"))
ENVIRONMENT = Path(
    os.environ.get("ABCE_ENVIRONMENT", RESEARCH / "verifiers-environments/environments/automationbench_v1")
)
CONTRACTS = ENVIRONMENT / "src/automationbench_v1/contracts"
CATALOG = CONTRACTS / "catalog.json"
EVIDENCE = Path(
    os.environ.get("ABCE_EVIDENCE", RESEARCH / "posttrain-rft-warm-start/docs/research/evidence")
)

DATA = REPO / "data"
EXCLUDED_NAMES = DATA / "automationbench_heldout62_v1.json"  # names only; used as an exclusion filter
EXCLUDED_TEST = DATA / "automationbench_test60b_v1.json"  # the new test (D66), sealed the same way
POOL_NAMES = DATA / "automationbench_train_pool_luna07_v1.json"
CACHE = Path(os.environ.get("ABCE_CACHE", REPO / ".cache"))
