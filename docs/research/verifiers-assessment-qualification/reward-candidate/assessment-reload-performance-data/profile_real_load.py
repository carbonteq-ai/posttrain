import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import verifiers.v1 as vf
import verifiers.v1.assessments as assessment_types
from verifiers.v1._validation_scope import validation_scope

parser = argparse.ArgumentParser()
parser.add_argument("artifact")
parser.add_argument("--mode", choices=("baseline", "scoped"), required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
raw = Path(args.artifact).read_bytes()
original_loads, original_dumps = json.loads, json.dumps
original_proof = assessment_types.intrinsic_proof
metrics = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "input_bytes": 0})
proofs = Counter()

def loads(value, *a, **kw):
    start = time.perf_counter()
    try:
        return original_loads(value, *a, **kw)
    finally:
        bucket = "loads_large" if len(value) > 1_000_000 else "loads_small"
        row = metrics[bucket]
        row["calls"] += 1
        row["seconds"] += time.perf_counter() - start
        row["input_bytes"] += len(value)

def dumps(value, *a, **kw):
    start = time.perf_counter()
    result = original_dumps(value, *a, **kw)
    row = metrics["dumps_large" if len(result) > 1_000_000 else "dumps_small"]
    row["calls"] += 1
    row["seconds"] += time.perf_counter() - start
    row["input_bytes"] += len(result)
    return result

def proof(value):
    result = original_proof(value)
    proofs[f"{type(value).__name__}:{'hit' if result.hit else 'miss'}"] += 1
    return result

json.loads, json.dumps = loads, dumps
assessment_types.intrinsic_proof = proof
start = time.perf_counter()
print(f"starting {args.mode} {len(raw)} bytes", flush=True)
try:
    if args.mode == "scoped":
        with validation_scope():
            episode = vf.WireEpisode.model_validate_json(raw)
    else:
        episode = vf.WireEpisode.model_validate_json(raw)
    elapsed = time.perf_counter() - start
finally:
    json.loads, json.dumps = original_loads, original_dumps
    assessment_types.intrinsic_proof = original_proof

serialized = episode.model_dump_json().encode()
result = {
    "mode": args.mode,
    "input_bytes": len(raw),
    "input_sha256": hashlib.sha256(raw).hexdigest(),
    "load_seconds": elapsed,
    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "json_work": dict(metrics),
    "intrinsic_proofs": dict(proofs),
    "roundtrip_sha256": hashlib.sha256(serialized).hexdigest(),
    "roundtrip_bytes": len(serialized),
    "traces": len(episode.traces),
    "batches": sum(len(t.assessment_batches) for t in episode.traces),
    "credit_assignments": sum(len(t.credit_assignments) for t in episode.traces),
}
Path(args.output).write_text(original_dumps(result, indent=2) + "\n")
print(original_dumps(result, indent=2), flush=True)
