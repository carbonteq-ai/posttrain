"""Prepare a proposed SELF pilot without changing the canonical train/test split."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "scripts/qualification/fixtures"
OUTPUT = ROOT / "docs/research/evidence/self-vllm-20261017/task-protocol.json"
DEV = {"finance": 2, "hr": 2, "marketing": 2, "operations": 3, "sales": 1, "support": 2, "simple": 12}
PILOT = {"finance": 3, "hr": 2, "marketing": 2, "operations": 4, "sales": 2, "support": 3, "simple": 16}


def digest(names):
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def select(names, counts, scores, seed):
    """Divide each score-ordered domain into equal bins; hash-pick one per bin."""
    selected = []
    for domain, count in counts.items():
        ordered = sorted(
            (n for n in names if n.split(".")[0] == domain),
            key=lambda n: (scores[n], n),
        )
        assert len(ordered) >= count
        for index in range(count):
            group = ordered[index * len(ordered) // count : (index + 1) * len(ordered) // count]
            selected.append(min(group, key=lambda n: hashlib.sha256(f"{seed}:{n}".encode()).digest()))
    return sorted(selected)


def main():
    files = {
        "new_training": "automationbench_train_pool_luna07_v1.json",
        "final_test": "automationbench_heldout62_v1.json",
        "legacy": "automationbench_manifest_luna_v2.json",
    }
    source = {key: json.loads((FIXTURES / name).read_text()) for key, name in files.items()}
    train, test = source["new_training"]["task_names"], source["final_test"]["task_names"]
    scores = source["new_training"]["luna_partial_credit"]
    assert len(train) == len(set(train)) == 124
    assert len(test) == len(set(test)) == 62
    assert not set(train) & set(test)
    dev = select(train, DEV, scores, "self-dev-20261017-v1")
    eligible = sorted(set(train) - set(dev))
    pilot = select(eligible, PILOT, scores, "self-pilot-20261017-v1")
    assert len(dev) == 24 and len(eligible) == 100 and len(pilot) == 32
    assert not set(dev) & set(pilot)
    legacy = set(source["legacy"]["task_names"])
    assert not legacy & set(test)
    cohorts = {"development": dev, "new_update_eligible": eligible, "pilot": pilot}
    result = {
        "status": "proposed; no collection or training launched",
        "environment_revision": source["new_training"]["environment_revision"],
        "source_files": {
            key: {"path": str((FIXTURES / name).relative_to(ROOT)),
                  "sha256": hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()}
            for key, name in files.items()
        },
        "selection": "Within-domain Luna-score bins, stable hash selection; no heldout performance used.",
        "novelty_audit_cutoff": "2026-10-11; not reverified against later runs",
        "cohorts": {
            key: {"count": len(names), "domains": dict(sorted(Counter(n.split('.')[0] for n in names).items())),
                  "mean_historical_luna": sum(scores[n] for n in names) / len(names),
                  "sorted_names_sha256": digest(names), "task_names": names}
            for key, names in cohorts.items()
        },
        "final_test": {"id": source["final_test"]["id"], "count": 62, "overlap_with_new_training": []},
        "legacy_inventory": {
            "fixture_training_count": len(legacy),
            "overlap_with_new124": sorted(legacy & set(train)),
            "overlap_with_final62": sorted(legacy & set(test)),
            "older_reserved20_overlap_new124": sorted(set(source["legacy"]["reserved_evaluation_tasks"]) & set(train)),
            "admission": "Candidates only: review current manifests, score health, Luna scores and training provenance before union. Historical ~300/167 claim not verified by this fixture.",
        },
        "pilot_collection": {"tasks": 32, "fresh_trajectories_per_task_per_round": 2,
                             "episodes_per_round": 64, "development_tasks_never_updated": True},
        "measurement": "Track distinct tasks, episodes, executed transitions, action/feedback tokens and update exposures separately. No saturation claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'task_names'} for key, value in result['cohorts'].items()}, indent=2))


if __name__ == "__main__":
    main()
