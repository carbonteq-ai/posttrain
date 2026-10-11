# ruff: noqa: E701, E702, E741, B905
import collections
import json
import sys


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


P = load(sys.argv[1])
C = load(sys.argv[2]) if len(sys.argv) > 2 else {}
rows = list(P.values())
errs = [r for r in rows if r.get("error")]
print("errors", len(errs), [(r["task"], r["error"][:150]) for r in errs][:6])
val = [r for r in rows if r.get("recorded_harms") is not None and not r.get("error")]
agree = sum(r["recorded_harms"] == r.get("harms") for r in val)
bin_agree = sum((r["recorded_harms"] > 0) == (r.get("harms", 0) > 0) for r in val)
print(
    f"validation (recorded manifest_harms present): n={len(val)} exact={agree} binary={bin_agree} "
    f"recorded_harmful={sum(r['recorded_harms'] > 0 for r in val)} total_recorded={sum(r['recorded_harms'] for r in val)} total_replayed={sum(r.get('harms', 0) for r in val)}"
)
for r in val:
    if r["recorded_harms"] != r.get("harms"):
        print(
            "  DISAGREE",
            r["task"],
            r["run"][-30:],
            "rec",
            r["recorded_harms"],
            "rep",
            r.get("harms"),
            "fid",
            r.get("fidelity"),
            r.get("harm_checks"),
        )
old = [r for r in rows if r.get("recorded_harms") is None]
print(
    f"old episodes: n={len(old)} errors={sum(bool(r.get('error')) for r in old)} harmful={sum((r.get('harms') or 0) > 0 for r in old)} harms={[r.get('harms') for r in old]}"
)
fid = collections.Counter(
    "clean" if r.get("fidelity") == [0, 0] else "minor" if r.get("fidelity") and sum(r["fidelity"]) <= 2 else "diverged"
    for r in rows
)
print(
    "fidelity pinned all:",
    dict(fid),
    " old only:",
    dict(
        collections.Counter(
            "clean"
            if r.get("fidelity") == [0, 0]
            else "minor"
            if r.get("fidelity") and sum(r["fidelity"]) <= 2
            else "diverged"
            for r in old
        )
    ),
)
if C:
    print(
        "fidelity current-tools old only:",
        dict(
            collections.Counter(
                "clean"
                if C[r["id"]].get("fidelity") == [0, 0]
                else "minor"
                if C[r["id"]].get("fidelity") and sum(C[r["id"]]["fidelity"]) <= 2
                else "diverged"
                for r in old
            )
        ),
    )
    diff = [
        (r["task"], r.get("harms"), C[r["id"]].get("harms")) for r in rows if r.get("harms") != C[r["id"]].get("harms")
    ]
    print("pinned vs current harm differences:", diff)
secs = [(r.get("replay_seconds") or 0) + (r.get("score_seconds") or 0) for r in rows]
print(
    f"seconds/episode: mean={sum(secs) / len(secs):.3f} max={max(secs):.3f} (replay mean {sum(r.get('replay_seconds') or 0 for r in rows) / len(rows):.3f}, score mean {sum(r.get('score_seconds') or 0 for r in rows) / len(rows):.3f})"
)
for r in rows:
    if r.get("fidelity") and sum(r["fidelity"]) > 2:
        print("  DIVERGED", r["task"], r["environment_id"], r["fidelity"], r.get("stats"))
