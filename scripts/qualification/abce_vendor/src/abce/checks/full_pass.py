"""M7 full pass, streamed: every (or a sample of) recorded episodes through Record -> both renderers,
with leak and probe accounting, aggregated over all tasks and over pool124.

Memory discipline (mandatory): episodes are read file by file and one at a time; nothing but
counters, one successful reference per task per file (its messages only) and the per-file result
rows are kept. Results are appended to disk per file; the aggregate is computed from those rows.

Reported per set (all / pool124): failed episodes and the share with a non-empty per-action
feedback in each style; raw byte-equality with the handover's `feedback_text`; untemplated share
(failed items whose statement fell back to the generic form); leaks in the hidden-values
rendering; probe statuses; located harms; disagreements; skip reasons.
"""

from __future__ import annotations

import collections
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from .. import paths
from ..context import build_context, feedback_for_action, solution_from
from ..record import Episode, build_record, iter_episodes
from ..render import common
from ..tasks import pool_names
from .leak import leaks


class _Reference:
    """A successful sibling reduced to what the renderers need (id, success, messages)."""

    def __init__(self, episode: Episode):
        self.id = episode.id
        self.success = True
        self.messages = episode.messages
        self.record = {"id": episode.id, "traces": [{"nodes": [{"message": m} for m in episode.messages],
                                                    "metrics": {"task_completed_correctly": 1.0}}]}


def _untemplated(item) -> bool:
    if item.type == "tool_error":
        return False
    if item.origin == "native":
        return item.probe.status != "full"
    return not item.address and not item.content


def _handover():
    try:
        sys.path.insert(0, str(paths.EVIDENCE.parents[2] / "scripts/qualification"))
        import sdpo_ab_handover  # read-only

        return sdpo_ab_handover
    except ImportError:
        return None


def _handover_text(handover, ep: Episode, reference: _Reference | None) -> str:
    solution = handover.solution_json(reference.record) if reference is not None else None
    assistants = [m for m in ep.messages if m["role"] == "assistant"]
    observations = []
    if assistants:
        current = {c.get("id") for c in assistants[0].get("tool_calls") or []}
        responses = [m for m in ep.messages if m["role"] == "tool" and m.get("tool_call_id") in current]
        observations = [{"tool": m.get("name"), "response": m.get("content")} for m in responses]
    return handover.feedback_text(ep.native_results, observations, solution)


def _select(path: str, sample: int | None, seed: int) -> tuple[list[int], dict[str, _Reference]]:
    """First streaming pass over one file: which line indexes to process (≤ `sample` failed episodes
    per task, seeded) and one successful reference per task."""
    rng = random.Random(f"{seed}:{Path(path).name}")
    failed_by_task: dict[str, list[int]] = collections.defaultdict(list)
    references: dict[str, _Reference] = {}
    successes: list[int] = []
    for index, ep in enumerate(iter_episodes([path])):
        if ep.success:
            if ep.task not in references:
                references[ep.task] = _Reference(ep)
            successes.append(index)
        else:
            failed_by_task[ep.task].append(index)
    chosen: list[int] = []
    for task, indexes in failed_by_task.items():
        if sample is not None and len(indexes) > sample:
            indexes = sorted(rng.sample(indexes, sample))
        chosen.extend(indexes)
    if sample is None:
        chosen.extend(successes)
    return sorted(chosen), references


def process_file(path: str, out_rows: list[dict], *, sample: int | None, seed: int, placement: str,
                 expected_values: bool, handover) -> None:
    chosen, references = _select(path, sample, seed)
    wanted = set(chosen)
    for index, ep in enumerate(iter_episodes([path])):
        if index not in wanted:
            continue
        record = build_record(ep, probe=True, manifest=True)
        reference = references.get(ep.task)
        siblings = [reference] if reference is not None else []
        row: dict[str, Any] = {"file": Path(path).parent.name, "task": ep.task, "episode": ep.id, "success": record.success,
                               "turns": record.turns}
        row["probe"] = dict(collections.Counter(i.probe.status for i in record.items if i.origin == "native" and i.type != "tool_error"))
        failed_items = [i for i in record.items if i.outcome == "failed"]
        row["failed_items"] = len(failed_items)
        row["untemplated_items"] = sum(_untemplated(i) for i in failed_items)
        row["untemplated_types"] = sorted({f"{i.origin}:{i.type}" for i in failed_items if _untemplated(i)})
        harms = [i for i in failed_items if i.polarity in ("forbid", "keep") and i.type != "tool_error"]
        row["harm_items"] = len(harms)
        row["harm_items_located"] = sum(i.location.kind == "at_call" for i in harms)
        row["disagreements"] = sum(1 for n in record.notes if n.startswith("disagreement"))
        raw_action = feedback_for_action(record, siblings, 0, style="raw", expected_values=True, episode=ep)
        plain_action = feedback_for_action(record, siblings, 0, style="plain", expected_values=True, episode=ep)
        row["action_raw_text"] = raw_action["text"] is not None
        row["action_plain_text"] = plain_action["text"] is not None
        row["action_plain_skip"] = plain_action["skip_reason"]
        row["action_solution"] = raw_action["solution_episode"] is not None
        row["action_raw_equal"] = (raw_action["text"] == _handover_text(handover, ep, reference)) if handover and raw_action["text"] else None
        # Hidden-values rendering, both styles: leak detectors.
        for style in common.STYLES:
            ctx = build_context(record, siblings, style=style, placement=placement, expected_values=expected_values, episode=ep)
            found = leaks(record, ctx, episode_messages=ep.messages, task_assertions=ep.task_data.get("assertions"),
                          tool_names=tuple(ep.task_data.get("zapier_tools") or ()))
            row[f"{style}_context"] = ctx["skip_reason"] is None
            row[f"{style}_skip"] = ctx["skip_reason"]
            row[f"{style}_leaks"] = len(found)
            if found:
                row[f"{style}_leak_examples"] = found[:2]
        out_rows.append(row)
        del record, ep
    gc.collect()


def aggregate(rows: list[dict]) -> dict[str, Any]:
    pool = pool_names()
    out: dict[str, Any] = {}
    for name, keep in (("all", lambda r: True), ("pool124", lambda r: r["task"] in pool)):
        subset = [r for r in rows if keep(r)]
        c: collections.Counter = collections.Counter()
        untemplated_types: collections.Counter = collections.Counter()
        skips: collections.Counter = collections.Counter()
        for r in subset:
            c["episodes"] += 1
            c["failed_episodes"] += not r["success"]
            for k, v in r["probe"].items():
                c[f"probe_{k}"] += v
            c["failed_items"] += r["failed_items"]
            c["untemplated_items"] += r["untemplated_items"]
            untemplated_types.update(r["untemplated_types"])
            c["harm_items"] += r["harm_items"]
            c["harm_items_located"] += r["harm_items_located"]
            c["disagreements"] += r["disagreements"]
            c["action_raw_checked"] += r["action_raw_equal"] is not None
            c["action_raw_equal"] += r["action_raw_equal"] is True
            c["action_plain_text"] += r["action_plain_text"]
            c["action_plain_failed_with_text"] += r["action_plain_text"] and not r["success"]
            c["action_with_solution"] += r["action_solution"]
            if r["action_plain_skip"]:
                skips[f"plain_action:{r['action_plain_skip']}"] += 1
            for style in common.STYLES:
                c[f"{style}_contexts"] += r[f"{style}_context"]
                c[f"{style}_failed_with_context"] += r[f"{style}_context"] and not r["success"]
                c[f"{style}_leaking_contexts"] += r[f"{style}_leaks"] > 0
                c[f"{style}_leaks"] += r[f"{style}_leaks"]
                if r[f"{style}_skip"]:
                    skips[f"{style}:{r[f'{style}_skip']}"] += 1
        failed = c["failed_episodes"] or 1
        out[name] = {**dict(c), "tasks": len({r["task"] for r in subset}),
                     "action_raw_equal_share": c["action_raw_equal"] / (c["action_raw_checked"] or 1),
                     "action_plain_failed_share": c["action_plain_failed_with_text"] / failed,
                     "raw_failed_with_context_share": c["raw_failed_with_context"] / failed,
                     "plain_failed_with_context_share": c["plain_failed_with_context"] / failed,
                     "untemplated_share": c["untemplated_items"] / (c["failed_items"] or 1),
                     "harm_located_share": c["harm_items_located"] / (c["harm_items"] or 1),
                     "untemplated_types": dict(untemplated_types.most_common(30)), "skip_reasons": dict(skips)}
    return out


def full_pass(files: list[str] | None = None, *, out_dir: str | Path, sample: int | None = 3, seed: int = 20261030,
              placement: str = "decision_point", expected_values: bool = False, resume: bool = True) -> dict[str, Any]:
    """Stream every file; append one JSON-lines file of rows per evidence file under `out_dir`;
    return the aggregate. With `resume`, files that already have a rows file are not re-run."""
    from ..record import episode_files

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handover = _handover()
    start = time.time()
    files = files if files is not None else episode_files()
    for path in files:
        try:
            relative = Path(path).resolve().relative_to(paths.EVIDENCE.resolve())
        except ValueError:
            relative = Path(path)
        target = out_dir / ("__".join(relative.parts[:-1]) + ".jsonl")
        if resume and target.exists():
            continue
        rows: list[dict] = []
        process_file(path, rows, sample=sample, seed=seed, placement=placement, expected_values=expected_values,
                     handover=handover)
        with open(target, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        del rows
        gc.collect()
    rows = []
    for target in sorted(out_dir.glob("*.jsonl")):
        with open(target) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    result = aggregate(rows)
    result["files"] = len(list(out_dir.glob("*.jsonl")))
    result["sample_per_task_per_file"] = sample
    result["seconds"] = round(time.time() - start, 1)
    with open(out_dir / "aggregate.json", "w") as handle:
        json.dump(result, handle, indent=1)
    return result
