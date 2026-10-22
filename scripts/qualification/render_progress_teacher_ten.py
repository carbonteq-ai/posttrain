"""Inspect full teacher prefixes and actual same-state native calls."""

import argparse
import html
import json
from pathlib import Path


def render(root, output):
    cases = json.loads((root / "source-cases.json").read_text())["cases"]
    proposals = json.loads((root / "same-decision-proposals.json").read_text())
    esc = html.escape
    body = ["<h1>Ten-task teacher-context qualification</h1><p>This is a zero-update native teacher/source diagnostic. "
            "No student learning or ten-step training result is established by this page. "
            "All tasks come from the124training pool; no recorded evaluation/development/reserve task is used.</p>",
            "<p>Each task compares the same original decision state across student, original-feedback teacher and revised teacher. "
            "The revised context carries verbatim system restrictions, compact source-bound requirements, prior-observation "
            "citations and retrospective feedback with hidden expected parameters withheld. Private compiler sidecars are not passed to the teacher.</p>"]
    for case in cases:
        task = case["row"]["task"]
        context = root / "contexts" / case["context_key"]
        body.append(f"<section><h2>{esc(task)}</h2><p>Native episode {case['row']['episode_id']}; "
                    f"generation {case['row']['generation_index']}; sampled episode success {case['row']['native_success']}.</p>")
        for filename, title in [("teacher-supplement.txt", "Revised guidance + decision feedback"),
                                ("full-teacher-context.txt", "Full revised teacher prefix, no omissions")]:
            body.append(f"<details><summary>{title}</summary><pre>{esc((context / filename).read_text())}</pre></details>")
        for mode in ["student", "original_teacher", "revised_teacher"]:
            p = next(p for p in proposals if p["task"] == task and p["mode"] == mode)
            body.append(f"<h3>{mode}</h3><details><summary>Actual generated reasoning and calls</summary><pre>{esc(p['raw'])}</pre></details>"
                        f"<details><summary>Native tool responses and one-decision scoring</summary><pre>{esc(json.dumps(p.get('native_replay', p.get('native_error')), indent=2))}</pre></details>")
        body.append("</section>")
    body.append("<p>Successful API calls alone do not establish a correct action. One-decision scores retain earlier "
                "state and native exclusions; they are not complete new student trajectories or a fixed-obligation benchmark. "
                "Incomplete outputs/errors remain visible. Independent semantic review and actual gradient/update evidence "
                "are required before training.</p>")
    output.write_text("<!doctype html><meta charset=utf-8><title>Ten teacher contexts and native calls</title>"
                      "<style>body{font:16px/1.55 system-ui;max-width:1150px;margin:36px auto;padding:0 22px;color:#243144}"
                      "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:16px}"
                      "section{border-top:2px solid #d8e0ea;margin:35px 0}summary{cursor:pointer;font-weight:650;margin:12px 0}</style>"
                      + "".join(body))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.output)
