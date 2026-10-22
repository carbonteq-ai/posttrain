"""Render the completed local four-context diagnosis without modifying raw evidence."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "docs/research/evidence/combined-teacher-qualification-20261019-r1"


def main():
    audit = json.loads((STUDY / "native-audit.json").read_text())
    gradients = json.loads((STUDY / "tool-gradient-audit/receipt.json").read_text())
    review = json.loads((STUDY / "same-decision-source-review.json").read_text())
    assert gradients["completed"] and gradients["optimizer_updates"] == 0
    rows = gradients["results"]
    for task in review["cases"]:
        matched = [r for r in rows if r["task"] == task]
        assert len(matched) == 2
        assert matched[0]["tool_mean_logprob_before"] == matched[1]["tool_mean_logprob_before"]
    verdict = {
        "runtime_completed": True, "qualified_for_restart": False,
        "training_restarted": False, "distinct_optimizer_updates": 2,
        "read_only_gradient_audit_updates": 0, "native_episodes": 48,
        "reasons": ["No failure-task success gain over base",
                    "Blog partial-credit regression: 75% to 62.5%",
                    "Campaign teacher still names forbidden items; actual Adam update raises their conditional log probability",
                    "Sales teacher repeats discovery rather than completing the already-supported update",
                    "Competitor retrieval uses a private tracking code present only in hindsight feedback"],
        "native_checks": audit["checks"],
        "scope": "One isolated update per arm, four failure and four retention training tasks, two greedy replicas each. Not heldout evidence.",
        "limits": ["Legacy partial-credit denominators can depend on excluded checks; native scores preserved",
                   "Applicable-manifest harm coverage not established; no safety qualification",
                   "No authoritative transport retry counter", "Context authoring is manually reviewed, not an automated qualified pipeline"],
    }
    (STUDY / "qualification-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    body = ["<h1>Combined teacher context: qualification results</h1>",
            "<p><strong>Not qualified to restart training.</strong> Partial credit improved, but success did not exceed base and one task regressed. Both diagnostic arms completed one isolated AdamW update; the final gradient audit applied zero updates.</p>",
            "<p>Same original 32-row batch; only four teacher prefixes changed. Student prompts and actions stayed unchanged. All 48 native episodes scored, all initial task states matched, no native scoring errors. These are small training-task diagnostics, not heldout results.</p>",
            "<h2>Full native student rollouts</h2><table><tr><th>Student</th><th>Failure-task success</th><th>Failure-task partial credit</th><th>Retention success / partial</th></tr>"]
    for arm, label in [("baseline", "Base"), ("original_teacher", "Original feedback: one update"),
                       ("combined_teacher", "Combined context: one update")]:
        c = audit["cohorts"][arm]
        metrics = c["failures"]["metrics"]
        body.append(f"<tr><td>{label}</td><td>{100 * metrics['task_completed_correctly']['task_mean']:.2f}%</td>"
                    f"<td>{100 * metrics['partial_credit']['task_mean']:.2f}%</td><td>100% / 100%</td></tr>")
    body.append("</table><h2>Four failures: native partial credit</h2><table><tr><th>Task</th><th>Base</th><th>Original</th><th>Combined</th></tr>")
    for r in audit["task_table"]:
        if r["cohort"] == "failures":
            body.append("<tr><td>" + html.escape(r["task"]) + "</td>" + "".join(
                f"<td>{100*r[a]['partial_credit']:.2f}%</td>" for a in
                ["baseline", "original_teacher", "combined_teacher"]) + "</tr>")
    body.append("</table><h2>Actual teacher decisions</h2>")
    for task, case in review["cases"].items():
        body.append(f"<h3>{html.escape(task)}</h3><p>{html.escape(case['combined_teacher'])}</p><p>{html.escape(case['verdict'])}</p>")
    body.append("<p>In full Sales rollouts, the combined student updates the correct contact phone, but omits the required note. In blog rollouts, it still names forbidden content and additionally omits the required Benchmarking item. API validity alone misses both problems.</p>")
    body.append("<h2>Tool-token audit</h2><p>Both updates are measured against the SAME combined-teacher proposal. Reasoning tokens are excluded. Positive means increased conditional mean log probability; it is beneficial only for a correct proposal. The projection is a moment-derived clipped-gradient approximation, not an actual Adam prediction.</p><table><tr><th>Task</th><th>Arm</th><th>Projected gradient direction</th><th>Actual Adam Δ mean log probability</th></tr>")
    for r in rows:
        body.append(f"<tr><td>{html.escape(r['task'])}</td><td>{r['arm']}</td><td>{r['full_batch_gradient_direction_toward_proposal_tool_logprob']:+.6f}</td><td>{r['actual_adam_tool_mean_logprob_delta']:+.6f}</td></tr>")
    body.append("</table><p><strong>Wrong-direction signal:</strong> the campaign proposal includes Holiday Offer, Upcoming Feature and VIP Early despite native exclusion checks. Their conditional log probability rose +0.004770 with the original update and +0.006025 with combined context. This does not estimate the probability of a whole-task violation, but it fails the local corrective-signal test.</p>")
    body.append("<h2>Next changes suggested by evidence</h2><ul><li>Include the common system policy in guidance preparation, especially prohibitions on naming excluded items.</li><li>Make feedback specific to the current decision and distinguish completed evidence gathering from outstanding actions.</li><li>Select the semantic failure action: this campaign training row was a successful sheet read, while the mistake occurred later in its summary.</li><li>Separate privileged expected values from observable evidence; do not credit hindsight-only policy-code retrieval as grounded planning.</li></ul><p>Training remains on hold. No automated context revision or additional optimizer run was launched.</p>")
    body.append("<h2>Limits and evidence</h2><p>Native partial-credit exclusions can change denominators. Scores were preserved; missing scores were not zero-filled. Retention is four tasks only. Applicable guard coverage and retries remain unverified. Raw traces, update receipts, source review and tool-gradient results are retained alongside this report.</p><p><a href='four-full-teacher-contexts.html'>Review the four full teacher contexts</a></p>")
    document = "<!doctype html><meta charset='utf-8'><title>Teacher-context qualification</title><style>body{font:16px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#202734}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border:1px solid #ccd2db;text-align:left}th{background:#edf1f7}h2{margin-top:36px}p{max-width:1000px}</style>" + "".join(body)
    (STUDY / "qualification-results.html").write_text(document)
    target = ROOT / "docs/research/evidence/combined-teacher-four-20261019-r1/qualification-results.html"
    target.write_text(document)


if __name__ == "__main__":
    main()
