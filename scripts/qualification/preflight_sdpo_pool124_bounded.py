"""CPU replay broad-task admission on retained native actions, never new model calls."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from sdpo_agent_ablation import associate
from sdpo_luna_context import native_task_bindings
from sdpo_pool124_config import diagnostic_groups, step_tasks, task_batch, validate_pool
from transformers import AutoTokenizer
from verifiers.v1.episode import Episode


def main(args):
    started = time.monotonic()
    root = args.capsule
    cohort = json.loads((root / "cohorts.json").read_text())
    tasks = validate_pool(cohort, json.loads((root / "train-pool.json").read_text()),
                          json.loads((root / "heldout62.json").read_text()))
    groups = diagnostic_groups(tasks, json.loads((root / "heldout62.json").read_text())["task_names"],
                               "pool124-bounded", cohort["reference_diagnostic"])
    if [(name, len(names)) for name, names in groups] != [("reference_train", 5), ("heldout62", 62)]:
        raise ValueError("targeted training or full heldout diagnostic population changed")
    references = json.loads((root / "luna-reviewed.json").read_text())["references"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    populations, covered, reference_covered = [], set(), set()
    for step in range(1, 5):
        folder = args.prior / f"step-{step:04d}/online_train"
        calls = [json.loads(s) for s in (folder / "generation-calls.jsonl").read_text().splitlines()]
        retained = [json.loads(s) for s in (folder / "native/traces.jsonl").read_text().splitlines()]
        episodes = [Episode.model_validate(e).model_dump(mode="json") for e in retained]
        bindings = native_task_bindings(episodes, retained)
        expected = step_tasks(tasks, step)
        for mode in ("solution-or-feedback", "luna-or-solution"):
            rows, exclusions = associate(calls, episodes, tokenizer, mode, references, bindings, 32768, False)
            batch = task_batch(rows, expected, offset=step-1)
            if Counter(row["task"] for row in batch) != Counter(expected):
                raise ValueError("broad task admission changed")
            for row in rows:
                call = calls[row["generation_index"]]
                if (row["student_prefix_ids"] != call["prefix_ids"] or row["action_ids"] != call["action_ids"]
                        or max(len(row[k])+len(row["action_ids"]) for k in
                               ("student_prefix_ids", "teacher_prefix_ids")) > 32768):
                    raise ValueError("student action changed or context exceeded; no truncation")
            used = {row["task"] for row in batch if row["offline_luna_reference_used"]}
            reference_covered |= used
            covered |= {row["task"] for row in batch}
            populations.append({"step": step, "mode": mode, "native_episodes": len(episodes),
                "selected": len(batch), "distinct_selected_tasks": len(expected), "exclusions": exclusions,
                "luna_selected_tasks": sorted(used),
                "max_teacher_tokens_plus_action": max(len(row["teacher_prefix_ids"])+len(row["action_ids"])
                                                      for row in rows)})
    if covered != set(tasks) or reference_covered != set(references):
        raise ValueError("full task or admitted-reference coverage absent from CPU replay")
    result = {"passed": True, "gpu_used": False, "model_generations": 0, "optimizer_updates": 0,
              "distinct_training_tasks": len(covered), "reference_tasks_exercised": sorted(reference_covered),
              "declared_diagnostics": {name: len(names) for name, names in groups},
              "populations": populations, "student_prefix_action_parity": True, "no_truncation": True,
              "elapsed_s": time.monotonic()-started,
              "scope": "Re-rendered32K teacher contexts on retained16K/nonthinking native actions; "
                       "not4K generation, full32K backward, thinking rollout or learning qualification"}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "populations"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
