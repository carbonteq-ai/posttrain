"""build_pairs: caps (4 per task, 2 per attempt), class coverage first, no duplicate pairs, byte-identical start,
identical call lists dropped, chosen masks carried, held-out split by task."""
import random

from build_pairs import CAP_ATTEMPT, CAP_TASK, build, select, untrained


def row(rid, task, kind, calls=(("a", "1"),), classes=None, trained=None, system="S"):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "U"}]
    for name, arg in calls:
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {"name": name,
                  "arguments": {"v": arg}}}]}, {"role": "tool", "content": "ok", "tool_call_id": "x"}]
    msgs.append({"role": "assistant", "content": "done"})
    meta = {"task": task, "episode_id": rid * 4}
    if classes:
        meta["failure_classes"] = {c: 1 for c in classes}
    tr = trained if trained is not None else [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    return {"id": rid, "messages": msgs, "tools": [{"name": "t"}], "trainable_message_indices": tr, "metadata": meta}


def test_caps_and_no_duplicates():
    ch = [row(f"c{i}", "t", "c", calls=[("ok", "1")]) for i in range(3)]
    rj = [row(f"r{i}", "t", "r", calls=[("bad", str(i))], classes=["wrong content"]) for i in range(5)]
    picked = select(ch, rj, random.Random(1))
    assert len(picked) == CAP_TASK
    ids = [(c["id"], r["id"]) for c, r in picked]
    assert len(set(ids)) == len(ids)
    for a in ch + rj:
        assert sum(a["id"] in p for p in ids) <= CAP_ATTEMPT


def test_every_failure_class_covered_first():
    ch = [row("c0", "t", "c", calls=[("ok", "1")]), row("c1", "t", "c", calls=[("ok", "2")])]
    rj = [row(f"r{i}", "t", "r", calls=[("bad", str(i))], classes=["wrong content"]) for i in range(6)]
    rj.append(row("rF", "t", "r", calls=[("bad", "F")], classes=["forbidden action"]))
    rj.append(row("rM", "t", "r", calls=[("bad", "M")], classes=["missed action"]))
    picked = select(ch, rj, random.Random(3))
    got = {r["id"] for _, r in picked}
    assert {"rF", "rM"} <= got


def test_start_mismatch_and_same_calls_are_not_paired():
    ch = [row("c0", "t", "c", calls=[("same", "1")])]
    rj = [row("r0", "t", "r", calls=[("same", "1")], classes=["wrong content"]),
          row("r1", "t", "r", calls=[("x", "2")], classes=["wrong content"], system="OTHER")]
    pairs, manifest, candidates = build(ch, rj, [])
    assert pairs == [] and manifest["t"]["same_calls"] == 1 and manifest["t"]["refused_start"] == 1


def test_expert_only_where_no_p9_success_and_masks_carried():
    ex = [row("e0", "t1", "c", calls=[("ok", "1"), ("ok", "2")], trained=[4, 6])]  # turn 2 masked
    ex2 = [row("e1", "t2", "c", calls=[("ok", "1")])]
    ch = [row("c2", "t2", "c", calls=[("ok", "9")])]
    rj = [row("r1", "t1", "r", calls=[("bad", "1")], classes=["wrong content"]),
          row("r2", "t2", "r", calls=[("bad", "1")], classes=["wrong content"])]
    pairs, _, _ = build(ch, rj, ex + ex2)
    by = {p["metadata"]["task"]: p for p in pairs}
    assert by["t1"]["metadata"]["type"] == "B" and by["t1"]["metadata"]["chosen_untrained"] == [0]
    assert by["t2"]["metadata"]["type"] == "A" and by["t2"]["metadata"]["chosen_id"] == "c2"
    assert untrained(ex2[0]) == []


def test_candidates_are_every_admissible_pair_and_include_the_picked_ones():
    ch = [row(f"c{i}", "t", "c", calls=[("ok", str(i))]) for i in range(3)]
    rj = [row(f"r{i}", "t", "r", calls=[("bad", str(i))], classes=["wrong content"]) for i in range(4)]
    rj.append(row("rS", "t", "r", calls=[("ok", "0")], classes=["wrong content"]))  # same calls as c0: never paired
    pairs, _, candidates = build(ch, rj, [])
    got = {(p["metadata"]["chosen_id"], p["metadata"]["rejected_id"]) for p in candidates}
    assert len(got) == len(candidates) == 3 * 5 - 1 and ("c0", "rS") not in got
    assert {(p["metadata"]["chosen_id"], p["metadata"]["rejected_id"]) for p in pairs} <= got
    assert all("split" in p["metadata"] for p in candidates)
