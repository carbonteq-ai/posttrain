import json
from collections import Counter
from pathlib import Path

import pytest
from prepare_self_transfer_reserve import QUOTAS, select


def inputs():
    root = Path(__file__).resolve().parents[2]
    return [json.loads((root / p).read_text()) for p in [
        "docs/research/evidence/self-vllm-20261017/task-protocol.json",
        "scripts/qualification/fixtures/automationbench_train_pool_luna07_v1.json",
        "scripts/qualification/fixtures/automationbench_heldout62_v1.json"]]


def test_reserved_population_disjoint_balanced_and_order_invariant():
    protocol, pool, heldout = inputs()
    names = select(protocol, pool, heldout)
    assert len(names) == len(set(names)) == 24
    excluded = set(protocol["cohorts"]["pilot"]["task_names"]) | set(protocol["cohorts"]["development"]["task_names"]) | set(heldout["task_names"])
    assert not set(names) & excluded
    assert dict(Counter(n.split(".")[0] for n in names)) == QUOTAS
    pool["task_names"].reverse()
    assert select(protocol, pool, heldout) == names


def test_final_test_contamination_rejected():
    protocol, pool, heldout = inputs()
    heldout["task_names"][0] = pool["task_names"][0]
    with pytest.raises(ValueError, match="separation"):
        select(protocol, pool, heldout)
