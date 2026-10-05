import argparse
import asyncio
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import verifiers.v1 as vf
import verifiers.v1._validation_scope as scopes
from verifiers.v1.assessments import SourceSnapshot

parser = argparse.ArgumentParser()
parser.add_argument("--default-loader", action="store_true")
parser.add_argument("--output", default="/tmp/assessment-reload-performance-20261005/rejection-controls.json")
args = parser.parse_args()
raw = json.loads(
    Path("/tmp/automationbench-luna-sales-20261004/scored-roundtrips/sales.apply_project_label.json").read_bytes()
)
raw["traces"][0]["assessment_batches"] = raw["traces"][0]["assessment_batches"][:2]
rows = []


def check(name, mutate, expected=True):
    payload = copy.deepcopy(raw)
    mutate(payload["traces"][0])
    with nullcontext() if args.default_loader else scopes.validation_scope():
        try:
            vf.WireEpisode.model_validate(payload)
            rejected = False
        except (ValueError, TypeError):
            rejected = True
    assert rejected == expected, (name, rejected)
    assert scopes._owner.get() is None
    rows.append({"case": name, "rejected": rejected})


check("valid_control", lambda t: None, expected=False)
check(
    "noncanonical_source",
    lambda t: t["assessment_sources"][0].update(source_json=" " + t["assessment_sources"][0]["source_json"]),
)
check("wrong_source_digest", lambda t: t["assessment_sources"][0].update(source_digest="0" * 64))
check("wrong_snapshot_id", lambda t: t["assessment_sources"][0].update(snapshot_id="0" * 64))
check("boolean_execution_count", lambda t: t["assessment_sources"][0]["executions"][0].update(event_count=True))
check(
    "noncanonical_view",
    lambda t: t["assessment_views"][0].update(input_json=" " + t["assessment_views"][0]["input_json"]),
)
check("wrong_view_digest", lambda t: t["assessment_views"][0].update(input_digest="0" * 64))
check("view_wrong_source", lambda t: t["assessment_views"][0].update(snapshot_id="0" * 64))
check("run_wrong_source", lambda t: t["assessment_batches"][1]["run"].update(snapshot_id="0" * 64))


def conflicting(t):
    inline = copy.deepcopy(t["assessment_sources"][0])
    inline["source_digest"] = "0" * 64
    t["assessment_batches"][1]["source"] = inline


check("conflicting_inline_source_after_pool_admission", conflicting)

with scopes.validation_scope():
    source = SourceSnapshot.model_validate(raw["traces"][0]["assessment_sources"][0])
    assert scopes.intrinsic_proof(source).hit
    forged = source.model_copy(update={"source_digest": "0" * 64})
    try:
        SourceSnapshot.model_validate(forged)
        raise AssertionError("forged model accepted")
    except ValueError:
        rows.append({"case": "forged_copy_after_proof_hit", "rejected": True})
assert scopes._owner.get() is None


async def isolation():
    async def child():
        return scopes.intrinsic_proof(source).hit

    with scopes.validation_scope():
        SourceSnapshot.model_validate(source)
        assert scopes.intrinsic_proof(source).hit
        assert not await asyncio.create_task(child())
    assert scopes._owner.get() is None


asyncio.run(isolation())
rows.append({"case": "unplanned_async_child_cannot_reuse_proof", "rejected": True})
Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
