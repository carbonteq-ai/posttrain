"""The final training file from the CHES scores (run on CPU).

  1. candidates (exact-token file) longer than CAP tokens (prompt + longer side) are left out: 32,768 normally; a
     smaller cap only if the memory proof did not fit (plan 5.1 item 4's fallback, pairs lost reported);
  2. ches_rank.select: per task the caps (<= 4 pairs, each attempt in <= 2), every failure class covered first, then
     lowest CHES; every pair is trained (Ali 19 Nov);
  3. writes the training file and the memory-proof file (its 10 longest pairs) into the lab project, and sets the
     main settings' max_steps = ceil(3 x pairs / 16) (every pair seen at least 3 times) and checkpoint_steps =
     ceil(pairs / 16) (about one pass) in the DPO catalog file.
usage: prepare_train.py CANDIDATES_TOK.jsonl CHES_PREFIX CAP LAB_DIR WORK_DIR
"""
import json
import math
import re
import sys
from pathlib import Path

import ches_rank

CATALOG = ".posttrain/catalog/qwen35-4b-automationbench-dpo-p9.yaml"
TRAIN = ".posttrain/datasets/qwen35-4b-ab-dpo-p9-train.jsonl"
PROOF = ".posttrain/datasets/qwen35-4b-ab-dpo-p9-longest10.jsonl"
SETTINGS = "qwen3.5-4b/agent-dpo-p9-e3"


def length(row: dict) -> int:
    t = row["metadata"]["token_ids"]
    return len(t["prompt"]) + max(len(t["chosen"]), len(t["rejected"]))


def set_steps(text: str, steps: int, checkpoint: int) -> str:
    """Set max_steps and checkpoint_steps inside the SETTINGS block only."""
    start = text.index(f"  {SETTINGS}:\n")
    end = text.find("\n\n", start)
    block = text[start:end]
    new = re.sub(r"(\n      max_steps: )\d+", rf"\g<1>{steps}", block, count=1)
    new = re.sub(r"(\n      checkpoint_steps: )\d+", rf"\g<1>{checkpoint}", new, count=1)
    if new.count(f"max_steps: {steps}") != 1 or new.count(f"checkpoint_steps: {checkpoint}") != 1:
        raise ValueError("could not set the steps in the settings block")
    return text[:start] + new + text[end:]


def main(cand_path, ches_prefix, cap, lab, work):
    cap, lab, work = int(cap), Path(lab), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    kept, dropped = work / f"candidates-cap{cap}.jsonl", 0
    with open(cand_path) as src, open(kept, "w") as out:
        for line in src:
            if length(json.loads(line)) <= cap:
                out.write(line)
            else:
                dropped += 1
    selected, report = work / f"selected-cap{cap}.jsonl", work / f"selected-cap{cap}.json"
    code = ches_rank.select(str(kept), ches_prefix, str(selected), str(report))
    if code:
        raise SystemExit(f"selection left a failure class uncovered (see {report})")
    rows = [json.loads(line) for line in open(selected)]
    (lab / TRAIN).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    longest = sorted(rows, key=lambda r: (-length(r), r["id"]))[:10]
    (lab / PROOF).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in longest))
    steps, checkpoint = math.ceil(3 * len(rows) / 16), math.ceil(len(rows) / 16)
    catalog = lab / CATALOG
    catalog.write_text(set_steps(catalog.read_text(), steps, checkpoint))
    summary = {"cap": cap, "candidates_over_cap": dropped, "pairs": len(rows),
               "tasks": len({r["metadata"]["task"] for r in rows}),
               "by_type": {k: sum(r["metadata"]["type"] == k for r in rows) for k in ("A", "B")},
               "max_steps": steps, "checkpoint_steps": checkpoint,
               "proof_lengths": [length(r) for r in longest], **json.load(open(report))}
    json.dump(summary, open(work / f"prepare-cap{cap}.json", "w"), indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
