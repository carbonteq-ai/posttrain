"""Task-resampled (paired) bootstrap over the ten-attempt held-out evals."""
import gzip, json, random, sys
from collections import defaultdict
from statistics import mean
from pathlib import Path
src = (Path(sys.argv[1]) / "eval_components.py").read_text()
ns = {"__file__": str(Path(sys.argv[1]) / "eval_components.py")}
exec(src[: src.index("def run_summary")], ns)
DOWNLOADS, episode_scores, model_call_failed = ns["DOWNLOADS"], ns["episode_scores"], ns["model_call_failed"]

LABELS = ["base", "sft-v1", "sft-v2", "dpo-v1", "dpo-v3-step48", "dpo-v3", "cont100"]
METRICS = ["success", "partial_credit", "goal_rate", "goals_met", "harms", "harm_free"]
runs = {}
for label in LABELS:
    path = DOWNLOADS / f"luna2-heldout-64k20t-t05-x10-{label}-20261012" / "traces.jsonl.gz"
    if not path.exists():
        path = path.with_suffix("")
    by_task = defaultdict(list)
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as stream:
        for line in stream:
            if line.strip() and not model_call_failed(line):
                e = json.loads(line)
                by_task[e["task"]["data"]["task_name"]].append(episode_scores((e.get("traces") or [{}])[0]))
    runs[label] = by_task
tasks = sorted(set.intersection(*(set(r) for r in runs.values())))

def stat(by_task, task_sample, ep_samples, key):
    vals = []
    for t, idx in zip(task_sample, ep_samples):
        eps = by_task[t]
        xs = [eps[i % len(eps)][key] for i in idx[t]]
        xs = [x for x in xs if x is not None]
        if xs: vals.append(mean(xs))
    return mean(vals)

rng = random.Random(0)
B = 5000
draws = {l: {k: [] for k in METRICS} for l in LABELS}
for _ in range(B):
    ts = [rng.choice(tasks) for _ in tasks]
    idx = [{t: [rng.randrange(10) for _ in range(10)]} for t in ts]
    for l in LABELS:
        for k in METRICS:
            draws[l][k].append(stat(runs[l], ts, idx, k))
point = {l: {k: stat(runs[l], tasks, [{t: list(range(10))} for t in tasks], k) for k in METRICS} for l in LABELS}
q = lambda xs: (sorted(xs)[int(.025*len(xs))], sorted(xs)[int(.975*len(xs))])
print(f"{len(tasks)} tasks, 10 attempts each, {B} paired bootstrap draws\n")
for k in METRICS:
    print(f"== {k}")
    for l in LABELS:
        lo, hi = q(draws[l][k])
        line = f"  {l:14s} {point[l][k]:.3f} [{lo:.3f}, {hi:.3f}]"
        for ref in ("base", "sft-v1"):
            if l == ref: continue
            d = [a - b for a, b in zip(draws[l][k], draws[ref][k])]
            dlo, dhi = q(d)
            mark = "*" if dlo > 0 or dhi < 0 else " "
            line += f" | vs {ref}: {point[l][k]-point[ref][k]:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{mark}"
        print(line)
