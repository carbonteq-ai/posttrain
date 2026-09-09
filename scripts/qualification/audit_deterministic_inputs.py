"""Audit retained synthetic inputs and native token provenance, not optimizer math.

This deliberately does not import the fixture implementation. It may inspect a
failed attempt; passing this audit never certifies five updates or a backend.
"""

import argparse
import hashlib
import json
from pathlib import Path

from verifiers.v1.trace import WireTrace


def audit(root):
    selection = json.loads((root / "selection.json").read_text())
    source_sha = hashlib.sha256((root / "fixture-source.py").read_bytes()).hexdigest()
    expected_digest = hashlib.sha256(
        json.dumps(
            {"fixture": selection["fixture"], "source_sha256": source_sha},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert selection["projection"]["scorer_digest"] == expected_digest
    episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    native = {trace["id"]: trace for episode in episodes for trace in episode["traces"]}
    checked, cases, turns_count, observation_tokens = 0, set(), 0, 0
    for line in (root / "traces.jsonl").read_text().splitlines():
        record = json.loads(line)
        panel = record["info"].get("qualification_turn_fixture")
        if panel is None:
            continue
        trace = WireTrace.model_validate(record)
        authority = WireTrace.model_validate(native[trace.id])
        assert len(trace.branches) == len(authority.branches) == 1
        branch = trace.branches[0]
        assert branch.token_ids == authority.branches[0].token_ids
        assert branch.sampled_mask == authority.branches[0].sampled_mask
        assert [node.logprobs for node in branch.nodes] == [node.logprobs for node in authority.branches[0].nodes]
        assert panel["synthetic"] is True and panel["source_sha256"] == source_sha
        assert panel["scorer_digest"] == expected_digest
        assert panel["trace_id"] == trace.id and panel["branch_id"] == "0"
        assert panel["projection_id"] == "assistant-turns@1"
        assert panel["rollout_id"] == record["info"]["posttrain_rollout_id"]
        case = hashlib.sha256(panel["rollout_id"].encode()).digest()[0] % 4
        assert panel["case"] == case
        pairs = ((0.25, -0.5), (-0.75, 1.0), (0.0, 0.0), (1.0, -0.25))
        policy_nodes = []
        for node in branch.nodes:
            assert len(node.token_ids) == len(node.mask)
            if any(node.mask):
                assert node.sampled and node.message.role == "assistant"
                policy_nodes.append(node)
            else:
                observation_tokens += len(node.token_ids)
        assert len(panel["assessments"]) == len(policy_nodes)
        expected_errors = []
        for index, assessment in enumerate(panel["assessments"]):
            identity = f"assistant-{index}"
            assert assessment["turn_id"] == identity
            assert assessment["components"] == [
                {"name": "fixture_a", "status": "valid", "value": pairs[case][index % 2]},
                {"name": "fixture_b", "status": "valid", "value": pairs[case][(index + 1) % 2]},
            ]
            if case == 2 or (case == 1 and index % 2 == 0) or (case == 3 and index == len(policy_nodes) - 1):
                expected_errors.append(identity)
        assert panel["erroneous_turn_ids"] == expected_errors
        checked += 1
        cases.add(case)
        turns_count += len(policy_nodes)
    assert checked > 0, "no synthetic inputs were exercised"
    return {
        "scope": "input-and-native-provenance-only",
        "backend_qualified": False,
        "traces_checked": checked,
        "turns_checked": turns_count,
        "unsampled_context_tokens": observation_tokens,
        "cases_observed": sorted(cases),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    (args.root / "deterministic-input-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
