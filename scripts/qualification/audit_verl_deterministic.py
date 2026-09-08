"""Independent Decimal-vs-live aggregate advantage audit for one-group veRL toys.

Aggregate agreement is not a per-token update or checkpoint-resume proof.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from audit_deterministic_inputs import audit as audit_inputs
from audit_structured_toy import normalize
from verifiers.v1.trace import WireTrace


def audit(root):
    inputs = audit_inputs(root)
    selection = json.loads((root / "selection.json").read_text())
    settings = selection["settings"]
    algorithm = "gdpo" if "component_names" in settings else "capo"
    trainer = root / "training" / algorithm / "trainer"
    metrics = [json.loads(line) for line in (trainer / "verl-metrics.jsonl").read_text().splitlines()]
    metrics = {row["step"]: row["data"] for row in metrics if "actor/grad_norm" in row["data"]}
    assert set(metrics) == {1, 2, 3, 4, 5}
    groups = defaultdict(list)
    for line in (root / "traces.jsonl").read_text().splitlines():
        record = json.loads(line)
        groups[record["info"]["posttrain_prompt_group_id"]].append(record)
    rejected = {group: [{"trace_id": record["id"], "stop_condition": record["stop_condition"]} for record in records]
                for group, records in groups.items()
                if any(record["stop_condition"] != "agent_completed" for record in records)}
    groups = {group: records for group, records in groups.items() if group not in rejected}
    assert len(groups) == 5, "ambiguous admission: audit requires exactly five complete groups"
    assert {int(group.split("/")[1]) for group in groups} == {1, 2, 3, 4, 5}
    assert settings["num_prompts_per_step"] == 1
    checked = []
    for group, records in groups.items():
        step = int(group.split("/")[1])
        assert len(records) == settings["num_generations"]
        rows, scores = [], []
        for record in records:
            branch = WireTrace.model_validate(record).branches[0]
            panel = record["info"]["qualification_turn_fixture"]
            count = sum(branch.sampled_mask)
            assert count > 0
            scores.append([math.fsum(item["components"][column]["value"]
                                     for item in panel["assessments"]) / len(panel["assessments"])
                           for column in (0, 1)])
            values, turn = [], 0
            for node in branch.nodes:
                if not any(node.mask):
                    continue
                error = f"assistant-{turn}" in panel["erroneous_turn_ids"]
                outcome = record["metrics"]["task_completed_correctly"]
                assert outcome in (0, 1)
                reward = settings.get("outcome_weight", 2.0) * outcome - settings.get("process_weight", 1.0) * error
                values.extend([reward] * sum(node.mask))
                turn += 1
            rows.append(values)
        if algorithm == "gdpo":
            columns = [normalize([row[column] for row in scores], settings["epsilon"]) for column in (0, 1)]
            combined = [sum(weight * column[index] for weight, column in
                            zip(settings["component_weights"], columns, strict=True)) for index in range(len(rows))]
            expected = normalize(combined, settings["epsilon"])
            tokens = [value for value, row in zip(expected, rows, strict=True) for _ in row]
        else:
            tokens = normalize([value for row in rows for value in row], settings["epsilon"])
        actual = metrics[step]
        for name, expected in (("mean", math.fsum(tokens) / len(tokens)), ("min", min(tokens)), ("max", max(tokens))):
            assert math.isclose(actual[f"critic/advantages/{name}"], expected, rel_tol=2e-5, abs_tol=2e-6), (
                step, name, actual[f"critic/advantages/{name}"], expected,
            )
        assert math.isfinite(actual["actor/grad_norm"])
        checked.append(step)
    assert any(row["actor/grad_norm"] > 0 for row in metrics.values())
    assert (root / "result.json").is_file(), "operation did not finalize successfully"
    return {"algorithm": algorithm, "updates": sorted(checked), "input_audit": inputs, "rejected_groups": rejected,
            "gradient_norms": [metrics[step]["actor/grad_norm"] for step in sorted(metrics)],
            "advantage_aggregate_oracle": "passed", "checkpoint_resume_qualified": False,
            "per_token_update_qualified": False, "release_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    (args.root / "deterministic-verl-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
