"""The chosen pool: the SFT model's successes that pass every SFT data check.

An attempt is admitted only if
  1. every non-advisory SFT gate passes (sftgen/validate_row.py --thinking, replay on the pinned environment) on the
     form of the attempt the pool uses: as recorded; with failed-call turns masked (SFT's rule: the gate accepts a
     masked turn only with a recorded reason); and, since gate 1b (Ali 18 Nov: chosen attempts get SFT's edit rules),
     with the final reply edited delete-only (reply_edits.py) or a wrong-then-corrected turn masked (mask_turns.py).
     Each form is a LAYER (rows + their own gate reports); a later layer replaces an earlier form of the same attempt.
     The one exception is `no_harness_words` when its hits equal, exactly, the hits I read and cleared
     (review/harness-clearances-*.json); any other or additional hit still fails;
  2. it is not dropped by my reading (review/chosen-review-*.json: per attempt, or the whole task; a later review's
     `undrop` lifts an earlier per-attempt drop, with its reason).
Attempts listed for Ali in a review file are admitted and carry `listed_for_ali` with the reason.
Writes OUT.jsonl (the admitted rows, in their layer's form) and MANIFEST.json (every success: admitted or the reason).
usage: chosen_pool.py OUT.jsonl MANIFEST.json --layer FORM ROWS.jsonl REPORT.json[,REPORT.json...] [--layer ...]
                      --review REVIEW.json [--review ...] --clearances CLEARANCES.json [--clearances ...]
"""
import argparse
import json
import sys

ADVISORY = {"call_numbers_shown"}


def failures(report: dict) -> list[str]:
    return sorted(g for g, v in report["gates"].items() if not v["pass"] and g not in ADVISORY)


def admit(report: dict, cleared: dict) -> tuple[bool, str]:
    failed = failures(report)
    if not failed:
        return True, "every gate passes"
    if failed == ["no_harness_words"]:
        hits = report["gates"]["no_harness_words"].get("hits")
        if report["id"] in cleared and cleared[report["id"]] == hits:
            return True, "every gate passes; harness-word hits read and cleared"
        return False, "no_harness_words (hits not cleared exactly)"
    return False, "gates failed: " + ", ".join(failed)


def merge_reviews(reviews: list[dict]) -> dict:
    """In order; a later review's `undrop` (id -> reason) lifts an earlier per-attempt drop (gate 1b: an attempt
    dropped on 17 Nov because masking was not allowed yet is admitted in its masked form)."""
    out = {"drop": {}, "drop_tasks": {}, "keep_listed_for_ali": {}}
    for rev in reviews:
        for rid in rev.get("undrop") or {}:
            if rid not in out["drop"]:
                raise ValueError(f"undrop of an attempt no earlier review dropped: {rid}")
            del out["drop"][rid]
        for key in out:
            out[key].update(rev.get(key) or {})
    return out


def build(layers, reviews, clearances):
    """layers: [(form, rows: {id: row}, reports: {id: report})] in order; reviews/clearances: loaded dicts."""
    rows, reports = {}, {}
    for form, layer_rows, layer_reports in layers:
        for rid, row in layer_rows.items():
            if rid not in layer_reports:
                raise ValueError(f"{form}: {rid} has no gate report")
            rows[rid] = row
            reports[rid] = (form, layer_reports[rid])
    cleared = {}
    for c in clearances:
        cleared.update(c["cleared"])
    rev = merge_reviews(reviews)
    chosen, manifest = [], {}
    for rid, row in rows.items():
        form, report = reports[rid]
        ok, why = admit(report, cleared)
        task = row["metadata"]["task"]
        if ok and rid in rev["drop"]:
            ok, why = False, "dropped on reading: " + rev["drop"][rid]
        if ok and task in rev["drop_tasks"]:
            ok, why = False, "task dropped on reading: " + rev["drop_tasks"][task]
        entry = {"task": task, "admitted": ok, "reason": why, "form": form}
        if ok and rid in rev["keep_listed_for_ali"]:
            entry["listed_for_ali"] = rev["keep_listed_for_ali"][rid]
            row["metadata"]["listed_for_ali"] = entry["listed_for_ali"]
        manifest[rid] = entry
        if ok:
            chosen.append(row)
    return chosen, manifest


def load_layer(form, rows_path, report_paths):
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(rows_path)}
    reports = {}
    for p in report_paths.split(","):
        for r in json.load(open(p))["rows"]:
            reports[r["id"]] = r
    return form, rows, {k: v for k, v in reports.items() if k in rows}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("manifest")
    ap.add_argument("--layer", nargs=3, action="append", required=True, metavar=("FORM", "ROWS", "REPORTS"))
    ap.add_argument("--review", action="append", required=True)
    ap.add_argument("--clearances", action="append", required=True)
    a = ap.parse_args(argv)
    layers = [load_layer(*x) for x in a.layer]
    chosen, manifest = build(layers, [json.load(open(p)) for p in a.review],
                             [json.load(open(p)) for p in a.clearances])
    with open(a.out, "w") as w:
        for row in chosen:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    json.dump(manifest, open(a.manifest, "w"), indent=1, ensure_ascii=False)
    forms = {}
    for e in manifest.values():
        if e["admitted"]:
            forms[e["form"]] = forms.get(e["form"], 0) + 1
    print(json.dumps({"successes": len(manifest), "admitted": len(chosen),
                      "tasks": len({r["metadata"]["task"] for r in chosen}), "admitted_by_form": forms,
                      "listed_for_ali": sum("listed_for_ali" in e for e in manifest.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
