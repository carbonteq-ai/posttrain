"""DPO preference pairs from the chosen and rejected pools.

One pair = one task's shared start (system message, user message, tool list) + a whole chosen attempt + a whole rejected
attempt (ETO: whole trajectories; step-level pairs collapsed there). Written in the framework's preference format
(posttrain.data.PreferenceExample: id, prompt, chosen, rejected, tools, metadata).
  type A  chosen = an admitted P9 success, rejected = an admitted P9 failure of the same task
  type B  chosen = the task's SFT data-v4 row (397B, gate-checked), only where the task has no admitted P9 success
Rules (plan 5.1 item 5 and 7.2, approved):
  - the shared start must be byte-identical on both sides (JSON of system, user and tools, key order kept: the chat
    template renders them in order), else the pair is refused and counted;
  - pairs whose tool-call lists (name + arguments) are identical are dropped (they would differ only in wording);
  - per task at most 4 pairs, each attempt in at most 2 pairs; selection (seed 20261103) first covers every failure
    class among the task's rejected attempts (wrong content, missed action, forbidden action, call never made), then
    fills the cap; the GPU similarity ranking (CHES, plan 5.1 item 3) later reorders within the same caps;
  - about 10% of the tasks with pairs (seeded, in proportion per type) are held out: their pairs are never trained and
    measure held-out pair accuracy during training.
Masks: chosen assistant turns not in the row's trainable_message_indices (failed calls, wrong-then-corrected turns,
SFT's own masks) are listed in metadata.chosen_untrained (indices into `chosen`); every rejected assistant turn is
trained. Phase 3's renderer must honour chosen_untrained (the framework renderer today trains every assistant turn).
usage: build_pairs.py CHOSEN.jsonl REJECTED.jsonl EXPERT.jsonl OUT_PAIRS.jsonl OUT_MANIFEST.json [OUT_CANDIDATES.jsonl]
"""
import hashlib
import json
import random
import sys
from collections import defaultdict

SEED = 20261103
CAP_TASK, CAP_ATTEMPT, HELDOUT_SHARE = 4, 2, 0.10
CLASS_ORDER = ("forbidden action", "missed action", "call never made", "wrong content", "unclassified")


def start(row: dict) -> str:
    return json.dumps([row["messages"][0], row["messages"][1], row.get("tools") or []], ensure_ascii=False)


def calls(row: dict) -> list:
    return [(c["function"]["name"], json.dumps(c["function"]["arguments"], sort_keys=True, ensure_ascii=False))
            for m in row["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []]


def untrained(row: dict) -> list[int]:
    """Indices (into messages[2:]) of assistant turns that carry no loss."""
    trained = set(row["trainable_message_indices"])
    return [i - 2 for i, m in enumerate(row["messages"]) if i >= 2 and m["role"] == "assistant" and i not in trained]


def classes(row: dict) -> set[str]:
    return set((row["metadata"].get("failure_classes") or {}).keys())


def first_difference(a: list, b: list) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def select(chosen: list[dict], rejected: list[dict], rng: random.Random) -> list[tuple[dict, dict]]:
    """Up to CAP_TASK pairs, each attempt in <= CAP_ATTEMPT pairs, every failure class covered first."""
    uses = defaultdict(int)
    picked, covered = [], set()
    pool = list(rejected)
    rng.shuffle(pool)

    def next_chosen(r):
        free = [c for c in chosen if uses[c["id"]] < CAP_ATTEMPT and (c["id"], r["id"]) not in pairs_made]
        if not free:
            return None
        low = min(uses[c["id"]] for c in free)
        return rng.choice([c for c in free if uses[c["id"]] == low])

    pairs_made = set()

    def take(r):
        if uses[r["id"]] >= CAP_ATTEMPT or len(picked) >= CAP_TASK:
            return False
        c = next_chosen(r)
        if c is None:
            return False
        picked.append((c, r))
        pairs_made.add((c["id"], r["id"]))
        uses[c["id"]] += 1
        uses[r["id"]] += 1
        covered.update(classes(r))
        return True

    for cls in CLASS_ORDER:  # 1. one rejected attempt per not-yet-covered failure class
        if cls in covered:
            continue
        options = [r for r in pool if cls in classes(r) and uses[r["id"]] == 0]
        if options:
            take(max(options, key=lambda r: len(classes(r) - covered)))
    for r in pool:  # 2. fill the cap with unused rejected attempts, then second uses
        if len(picked) >= CAP_TASK:
            break
        if uses[r["id"]] == 0:
            take(r)
    for r in pool:
        if len(picked) >= CAP_TASK:
            break
        take(r)
    return picked


def pair_record(task: str, kind: str, c: dict, r: dict) -> dict:
    cc, rc = calls(c), calls(r)
    pid = f"dpo-pair-{task}-{c['metadata']['episode_id'][:8]}-{r['metadata']['episode_id'][:8]}"
    return {"id": pid, "prompt": c["messages"][:2], "chosen": c["messages"][2:], "rejected": r["messages"][2:],
            "tools": c.get("tools") or [],
            "chosen_trace_id": c["id"], "rejected_trace_id": r["id"],
            "metadata": {"task": task, "type": kind, "chosen_id": c["id"], "rejected_id": r["id"],
                         "chosen_untrained": untrained(c), "rejected_failure_classes": r["metadata"]["failure_classes"],
                         "chosen_calls": len(cc), "rejected_calls": len(rc),
                         "first_differing_call": first_difference(cc, rc),
                         "calls_only_in_chosen": len(set(cc) - set(rc)), "calls_only_in_rejected": len(set(rc) - set(cc)),
                         "chosen_form_metadata": {k: c["metadata"].get(k) for k in ("masked_turns", "masked_reasons",
                                                                                    "edits", "listed_for_ali")
                                                  if c["metadata"].get(k)}}}


def build(chosen_rows, rejected_rows, expert_rows):
    by = lambda rows: {t: [r for r in rows if r["metadata"]["task"] == t] for t in {r["metadata"]["task"] for r in rows}}
    ch, rj, ex = by(chosen_rows), by(rejected_rows), by(expert_rows)
    rng = random.Random(SEED)
    pairs, manifest, candidates = [], {}, []
    for task in sorted(set(ch) | set(rj) | set(ex)):
        rejected = rj.get(task, [])
        kind, chosen = ("A", ch[task]) if task in ch else (("B", ex[task]) if task in ex else (None, []))
        entry = {"type": kind, "chosen": len(chosen), "rejected": len(rejected)}
        if not kind or not rejected:
            entry["pairs"], entry["reason"] = 0, "no chosen side" if not kind else "no rejected attempt"
            manifest[task] = entry
            continue
        refused = [(c["id"], r["id"]) for c in chosen for r in rejected if start(c) != start(r)]
        same_calls = [(c["id"], r["id"]) for c in chosen for r in rejected if calls(c) == calls(r)]
        bad = set(refused) | set(same_calls)
        ok_rejected = [r for r in rejected if any((c["id"], r["id"]) not in bad for c in chosen)]
        ok_chosen = [c for c in chosen if any((c["id"], r["id"]) not in bad for r in rejected)]
        picked = [(c, r) for c, r in select(ok_chosen, ok_rejected, rng) if (c["id"], r["id"]) not in bad]
        entry.update(candidates=len(chosen) * len(rejected), refused_start=len(refused), same_calls=len(same_calls),
                     pairs=len(picked),
                     classes_available=sorted(set().union(*[classes(r) for r in rejected])),
                     classes_covered=sorted(set().union(*[classes(r) for _, r in picked])) if picked else [])
        manifest[task] = entry
        pairs += [pair_record(task, kind, c, r) for c, r in picked]
        candidates += [pair_record(task, kind, c, r) for c in ok_chosen for r in ok_rejected
                       if (c["id"], r["id"]) not in bad]
    # held-out tasks: ~10% of tasks with pairs, per type, seeded
    with_pairs = defaultdict(list)
    for task, e in manifest.items():
        if e.get("pairs"):
            with_pairs[e["type"]].append(task)
    held = set()
    for kind in sorted(with_pairs):
        tasks = sorted(with_pairs[kind])
        held |= set(random.Random(f"{SEED}-{kind}").sample(tasks, round(HELDOUT_SHARE * len(tasks))))
    for p in pairs + candidates:
        p["metadata"]["split"] = "heldout" if p["metadata"]["task"] in held else "train"
    for task in held:
        manifest[task]["split"] = "heldout"
    return pairs, manifest, candidates


def main(chosen_path, rejected_path, expert_path, out, manifest_path, candidates_path=None):
    load = lambda p: [json.loads(line) for line in open(p)]
    pairs, manifest, candidates = build(load(chosen_path), load(rejected_path), load(expert_path))
    if candidates_path:  # every admissible pair within the caps' pools, for checks before the CHES ranking picks
        with open(candidates_path, "w") as w:
            for p in candidates:
                w.write(json.dumps(p, ensure_ascii=False) + "\n")
    ids = [p["id"] for p in pairs]
    assert len(ids) == len(set(ids)), "duplicate pair ids"
    with open(out, "w") as w:
        for p in pairs:
            w.write(json.dumps(p, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    split = {s: sum(p["metadata"]["split"] == s for p in pairs) for s in ("train", "heldout")}
    types = {k: sum(p["metadata"]["type"] == k for p in pairs) for k in ("A", "B")}
    summary = {"pairs": len(pairs), "by_type": types, "by_split": split,
               "tasks_with_pairs": sum(bool(e.get("pairs")) for e in manifest.values()),
               "heldout_tasks": sorted(t for t, e in manifest.items() if e.get("split") == "heldout"),
               "refused_start": sum(e.get("refused_start", 0) for e in manifest.values()),
               "same_calls": sum(e.get("same_calls", 0) for e in manifest.values()), "sha256": sha}
    json.dump({"summary": summary, "tasks": manifest}, open(manifest_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:7]))
