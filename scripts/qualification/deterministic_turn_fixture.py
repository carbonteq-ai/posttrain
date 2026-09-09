"""Synthetic reward inputs for algorithm qualification; never semantic ratings.

No mutable counter, RNG, model client or task-specific success override. The
rollout identity selects a case; native task rewards remain untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from posttrain.train.turn_rewards import TURN_PROJECTION, native_turn_map

FIXTURE = {
    "id": "deterministic-turn-inputs@1",
    "purpose": "synthetic-algorithm-qualification-not-reasoning-quality",
    "case_selection": "sha256(rollout_id) first byte modulo 4",
    "scores": [[0.25, -0.5], [-0.75, 1.0], [0.0, 0.0], [1.0, -0.25]],
    "errors": ["none", "alternating-even", "all", "last"],
}
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCORER_DIGEST = hashlib.sha256(
    json.dumps({"fixture": FIXTURE, "source_sha256": SOURCE_SHA256}, sort_keys=True).encode()
).hexdigest()
ANNOTATION_KEY = "qualification_turn_fixture"


class DeterministicTurnFixture:
    def __call__(self, trace):
        if len(trace.branches) != 1:
            raise ValueError("fixture requires one native branch")
        if ANNOTATION_KEY in trace.info:
            raise ValueError("fixture evidence already exists")
        rollout_id = trace.info["posttrain_rollout_id"]
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("fixture requires a stable rollout identity")
        turns = native_turn_map(trace.branches[0])
        case = hashlib.sha256(rollout_id.encode()).digest()[0] % 4
        scores = FIXTURE["scores"][case]
        errors = [
            turn.id
            for index, turn in enumerate(turns)
            if (case == 1 and index % 2 == 0) or case == 2 or (case == 3 and index == len(turns) - 1)
        ]
        evidence_ref = f"{trace.id}/info/{ANNOTATION_KEY}"
        trace.info.setdefault("posttrain_scorer_digests", {})[ANNOTATION_KEY] = SCORER_DIGEST
        trace.info[ANNOTATION_KEY] = {
            "trace_id": trace.id,
            "branch_id": "0",
            "projection_id": TURN_PROJECTION,
            "scorer_digest": SCORER_DIGEST,
            "synthetic": True,
            "fixture": FIXTURE,
            "source_sha256": SOURCE_SHA256,
            "rollout_id": rollout_id,
            "case": case,
            "erroneous_turn_ids": errors,
            "assessments": [
                {
                    "turn_id": turn.id,
                    "evidence_ref": evidence_ref,
                    "components": [
                        {"name": "fixture_a", "status": "valid", "value": scores[index % 2]},
                        {"name": "fixture_b", "status": "valid", "value": scores[(index + 1) % 2]},
                    ],
                }
                for index, turn in enumerate(turns)
            ],
        }
