"""Build the adaptive experiment index and exact per-experiment trace viewers."""

import argparse
import html
import json
from pathlib import Path

from build_luna_solver_viewer import build


def run(root, output):
    ledger = json.loads((root / "ledger.json").read_text())
    summary_path = root / "summary.json"
    summaries = (
        {e["number"]: e for e in json.loads(summary_path.read_text())["experiments"]} if summary_path.exists() else {}
    )
    rows, options = [], []
    for entry in ledger["experiments"]:
        number = entry["number"]
        counts = summaries.get(number, {}).get("models", {})
        folder = root / f"experiment{number:02d}"
        if not (folder / "launch.json").exists():
            continue
        build(folder, folder / "viewer.html")
        task_count = len(json.loads((folder / "inputs.json").read_text())["tasks"])

        def population(size, folder=folder, counts=counts, task_count=task_count):
            if not (folder / size / "materials.json").exists():
                return "Not run"
            return f"{counts.get(size, {}).get('guided_executed', '?')}/{task_count}"
        path = folder.relative_to(output.parent) / "viewer.html"
        options.append(
            f'<option value="{html.escape(str(path))}">Experiment {number}: {html.escape(entry["variant"])}</option>'
        )
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    number,
                    entry["variant"],
                    entry["status"],
                    population("4b"),
                    population("9b"),
                    entry.get("result", "Inspection pending"),
                    entry.get("next_reason", "Pending previous result"),
                )
            )
            + "</tr>"
        )
    page = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive solver generation experiments</title><style>
body{font:15px/1.55 system-ui,sans-serif;margin:24px;color:#182234;background:#f7f8fa}h1{font-size:26px}p{max-width:100ch}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #cbd2dc;padding:10px;text-align:left;vertical-align:top}select{font:inherit;padding:10px;max-width:100%;margin:12px 0}iframe{width:100%;height:1100px;border:1px solid #cbd2dc;background:white}td{overflow-wrap:anywhere}.table{overflow:auto}a{color:#2456a6}@media(max-width:600px){body{margin:12px}table{min-width:700px}}
</style><h1>Five adaptive solver-generation experiments</h1>
<p>Two fixed tasks, fresh Qwen3.5-4B and9B, no training or corpus scaling. Private demonstration values are separated from solver instructions. Each next experiment follows inspection of the previous result. Native scores, structural admission and semantic trace quality are distinct.</p>
<p>Guided counts are executed episodes out of two possible tasks, not success rates.04/05 permit reviewer-negative simulated diagnostics; none receive automatic SFT approval.</p>
<div class="table"><table><thead><tr><th>Experiment</th><th>Variant</th><th>Status</th><th>4B guided</th><th>9B guided</th><th>Finding</th><th>Why next</th></tr></thead><tbody>ROWS</tbody></table></div>
<p>Use the experiment selector below, then the model/task selectors inside its viewer. All snapshots are retained evidence, not live status.</p>
<label for="experiment">Experiment</label><br><select id="experiment">OPTIONS</select><br>
<a id="open" target="_blank" rel="noopener">Open selected trace viewer separately</a><iframe id="viewer" title="Selected experiment trace viewer"></iframe>
<script>const s=document.getElementById('experiment'),f=document.getElementById('viewer'),a=document.getElementById('open');function show(){f.src=s.value;a.href=s.value}s.addEventListener('change',show);s.selectedIndex=s.options.length-1;show()</script>"""
    page = page.replace("<style>", "<style>*{box-sizing:border-box}td:nth-child(3){white-space:nowrap}")
    presentation = ledger.get("presentation", {})
    if presentation:
        page = page.replace("Five adaptive solver-generation experiments", html.escape(presentation["title"]))
        page = page.replace("Two fixed tasks, fresh Qwen3.5-4B and9B, no training or corpus scaling. Private demonstration values are separated from solver instructions. Each next experiment follows inspection of the previous result. Native scores, structural admission and semantic trace quality are distinct.", html.escape(presentation["introduction"]))
        page = page.replace("Guided counts are executed episodes out of two possible tasks, not success rates.04/05 permit reviewer-negative simulated diagnostics; none receive automatic SFT approval.", html.escape(presentation["count_note"]))
    output.write_text(page.replace("ROWS", "".join(rows)).replace("OPTIONS", "".join(options)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
