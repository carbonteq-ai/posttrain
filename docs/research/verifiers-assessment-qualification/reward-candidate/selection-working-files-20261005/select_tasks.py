"""Accept replayed manifests, tier difficulty, and draw a 20-task eval split (>=2 per domain,
balanced easy/medium/hard) with the rest for training. Deterministic (seed 20261005)."""

import collections
import glob
import json
import random
import statistics
import sys

rows = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
COV = {
    e["task_name"]: e
    for e in json.load(
        open(
            "/home/hammad/projects/rl/docs/research/verifiers-assessment-qualification/luna-development-review-coverage.json"
        )
    )["entries"]
}
prior = collections.defaultdict(list)
for f in glob.glob("/home/hammad/projects/rl/docs/research/sampo-multirun-analysis/evidence/groups-*.json"):
    for g in json.load(open(f)):
        prior[g["task_name"]].append(g["mean"])


def luna_score(n):
    rw = COV[n].get("official_rewards") or [{}]
    return float(rw[0].get("score") or 0.0)


accepted, rejected = [], []
for r in rows:
    if "error" in r:
        rejected.append((r["task"], "error"))
        continue
    full = r.get("luna_class") == "official_full"
    reasons = []
    if r["binding"] is not None:
        reasons.append("binding")
    if r["assessment_errors"] or r["credit_errors"]:
        reasons.append("errors")
    if r["required"] == 0:
        reasons.append("no_required_goals")
    elif r["required_valid"] != r["required"]:
        reasons.append("undecided_goals")
    if full and r["required_pass"] != r["required"]:
        reasons.append("disagrees_with_full_pass")
    if full and r["harms"]:
        reasons.append("harm_on_full_pass")
    if not full and (r.get("luna_class") != "official_partial" or luna_score(r["task"]) < 0.5):
        reasons.append("luna_not_solved")
    if reasons:
        rejected.append((r["task"], ",".join(reasons)))
        continue
    n = r["task"]
    p = prior.get(n)
    if p:
        m = statistics.mean(p)
        tier = "easy" if m >= 0.6 else "medium" if m >= 0.2 else "hard"
        basis = f"2.6b_mean={m:.2f}"
    else:
        ls = luna_score(n)
        if not full:
            tier = "hard"
        elif r["required"] >= 8:
            tier = "medium"
        else:
            tier = "easy"
        basis = f"luna={ls:.2f},goals={r['required']}"
    accepted.append(
        {
            "task": n,
            "domain": n.split(".")[0],
            "tier": tier,
            "basis": basis,
            "luna_class": r["luna_class"],
            "required": r["required"],
        }
    )
rng = random.Random(20261005)
by_domain = collections.defaultdict(list)
for a in accepted:
    by_domain[a["domain"]].append(a)
for v in by_domain.values():
    rng.shuffle(v)
evals = []
# Two per domain, preferring different tiers within a domain.
for _domain, items in sorted(by_domain.items()):
    picked = []
    for tier in ("hard", "medium", "easy"):
        for a in items:
            if a["tier"] == tier and a not in picked and len(picked) < 2 and not any(p["tier"] == tier for p in picked):
                picked.append(a)
    for a in items:
        if len(picked) >= 2:
            break
        if a not in picked:
            picked.append(a)
    evals += picked
# Fill to 20 balancing tiers.
target = {"easy": 7, "medium": 7, "hard": 6}
# Simple tasks stay at the two-per-domain minimum (the user wants few simple tasks).
pool = [a for a in accepted if a not in evals and a["domain"] != "simple"]
rng.shuffle(pool)
while len(evals) < 20 and pool:
    counts = collections.Counter(a["tier"] for a in evals)
    need = max(target, key=lambda t: target[t] - counts[t])
    cand = next((a for a in pool if a["tier"] == need), pool[0])
    evals.append(cand)
    pool.remove(cand)
eval_names = sorted(a["task"] for a in evals)
train = sorted(a["task"] for a in accepted if a["task"] not in eval_names)
summary = {
    "selection": "replayed manifests accepted (binding, no errors, decided goals, agreement with Luna full passes, no harm on full passes, Luna full pass or partial score >= 0.5); eval = 2 per domain then tier-balanced fill to 20 without more simple tasks; seed 20261005",
    "accepted": len(accepted),
    "rejected": len(rejected),
    "train": train,
    "eval": eval_names,
    "eval_tiers": dict(collections.Counter(a["tier"] for a in evals)),
    "eval_domains": dict(collections.Counter(a["domain"] for a in evals)),
    "train_domains": dict(collections.Counter(n.split(".")[0] for n in train)),
    "train_tiers": dict(collections.Counter(a["tier"] for a in accepted if a["task"] in train)),
    "tasks": accepted,
    "rejected_reasons": rejected,
}
json.dump(summary, open(OUT, "w"), indent=1)
print(
    json.dumps({k: v for k, v in summary.items() if k not in ("tasks", "train", "eval", "rejected_reasons")}, indent=1)
)
print("rejected:", collections.Counter(r for _, r in rejected))
