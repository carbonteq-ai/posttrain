"""Build task mix automationbench-manifest-luna-v2: v1 plus Simple training tasks up to 20.

    python docs/research/verifiers-assessment-qualification/reward-candidate/simple-manifest-select.py \
        docs/research/verifiers-assessment-qualification/reward-candidate/simple-manifest-replay-20261006.json

Acceptance is v1's rule on the replay rows (``simple-manifest-replay.py``): the manifest
binds the 16-turn task, scores Luna's episode without assessment or credit errors, decides at least
one required goal and every required goal, agrees with a Luna full pass (every required goal passes,
no harm), and Luna fully passed the task or scored at least 0.5. Accepted tasks outside v1 are ranked
by Luna's score, ties broken by sha256("20261006:" + task name); the first 18 join v1's training tasks.
Evaluation is unchanged. Deterministic; writes the v2 fixture next to v1's.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FIXTURES = ROOT / "scripts/qualification/fixtures"
SIMPLE_TRAINING_TASKS = 20
SEED = "20261006"


def rejection(row: dict) -> str | None:
    full = row.get("luna_class") == "official_full"
    reasons = []
    if "error" in row or "required" not in row:
        return "error" if "error" in row else "no_luna_episode"
    if row["binding_16"] is not None:
        reasons.append("binding")
    if row["assessment_errors"] or row["credit_errors"]:
        reasons.append("errors")
    if row["required"] == 0:
        reasons.append("no_required_goals")
    elif row["undecided"]:
        reasons.append("undecided_goals")
    if full and row["passed"] != row["required"]:
        reasons.append("disagrees_with_full_pass")
    if full and row["harms"]:
        reasons.append("harm_on_full_pass")
    if not full and (row.get("luna_class") != "official_partial" or (row.get("luna_score") or 0) < 0.5):
        reasons.append("luna_not_solved")
    return ",".join(reasons) or None


def tier(row: dict, prior: dict[str, list[float]]) -> dict:
    if means := prior.get(row["task"]):
        mean = statistics.mean(means)
        return {
            "tier": "easy" if mean >= 0.6 else "medium" if mean >= 0.2 else "hard",
            "basis": f"2.6b_mean={mean:.2f}",
        }
    full = row["luna_class"] == "official_full"
    label = "hard" if not full else "medium" if row["required"] >= 8 else "easy"
    return {"tier": label, "basis": f"luna={row['luna_score'] or 0:.2f},goals={row['required']}"}


def main() -> None:
    replay_path = Path(sys.argv[1])
    rows = json.loads(replay_path.read_text())
    v1 = json.loads((FIXTURES / "automationbench_manifest_luna_v1.json").read_text())
    taken = set(v1["task_names"]) | set(v1["reserved_evaluation_tasks"])
    prior: dict[str, list[float]] = collections.defaultdict(list)
    for path in glob.glob(str(ROOT / "docs/research/sampo-multirun-analysis/evidence/groups-*.json")):
        for group in json.loads(Path(path).read_text()):
            prior[group["task_name"]].append(group["mean"])

    accepted, rejected = [], []
    for row in rows:
        if (reason := rejection(row)) is None:
            accepted.append(row)
        else:
            rejected.append({"task": row["task"], "reasons": reason})
    ranked = sorted(
        (row for row in accepted if row["task"] not in taken),
        key=lambda row: (-(row["luna_score"] or 0.0), hashlib.sha256(f"{SEED}:{row['task']}".encode()).hexdigest()),
    )
    present = sum(name.startswith("simple.") for name in v1["task_names"])
    added = ranked[: SIMPLE_TRAINING_TASKS - present]

    names = sorted(v1["task_names"] + [row["task"] for row in added])
    tiers = dict(v1["tiers"]) | {row["task"]: tier(row, prior) for row in added}
    train_outcomes = collections.Counter(v1["luna_outcomes"]["train"])
    train_outcomes.update(row["luna_class"] for row in added)
    luna = dict(v1["luna_outcomes"]) | {"train": dict(sorted(train_outcomes.items()))}

    v2 = dict(v1)
    v2 |= {
        "id": "automationbench-manifest-luna-v2",
        "selection_method": v1["selection_method"]
        + (
            " v2 adds Simple training tasks up to 20: every Simple draft or installed manifest replayed on its"
            " Luna episode with the training step-credit reader (record goals count; drafts bound to the user"
            " message when the whole-prompt binding was the only mismatch), accepted by the same rule, ranked"
            f' by Luna score with ties broken by sha256("{SEED}:" + task name). Evaluation is v1\'s.'
        ),
        "num_train": len(names),
        "task_names": names,
        "train_domains": dict(sorted(collections.Counter(name.split(".")[0] for name in names).items())),
        "train_tiers": dict(sorted(collections.Counter(tiers[name]["tier"] for name in names).items())),
        "tiers": dict(sorted(tiers.items())),
        "luna_outcomes": luna,
        "selected_names_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
        "simple_selection": {
            "replay": str(replay_path.relative_to(ROOT)) if replay_path.is_absolute() else str(replay_path),
            "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            "seed": SEED,
            "accepted": len(accepted),
            "added": [row["task"] for row in added],
            "ranked_unused": [row["task"] for row in ranked[len(added) :]],
            "rejected": rejected,
        },
    }
    out = FIXTURES / "automationbench_manifest_luna_v2.json"
    out.write_text(json.dumps(v2, indent=1) + "\n")
    print(json.dumps({key: v2[key] for key in ("num_train", "train_domains", "train_tiers")}, indent=1))
    print("added:", [row["task"] for row in added])
    print("rejected:", collections.Counter(item["reasons"] for item in rejected).most_common())
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
