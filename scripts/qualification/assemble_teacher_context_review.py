"""Assemble inspectable candidate teacher prefixes from retained real SDPO rows.

No inference, model update or training-runtime edits. Historical feedback and
rendered history are preserved exactly; only the teacher supplement is replaced.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

from tokenizers import Tokenizer

GENERATION_PREFIX = "<|im_start|>assistant\n<think>\n"
BOUNDARY = "Factual hindsight from this student episode:\n"


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def parse_feedback(text):
    if text.count(BOUNDARY) != 1:
        raise ValueError("ambiguous historical feedback boundary")
    facts, _ = json.JSONDecoder().raw_decode(text.split(BOUNDARY, 1)[1])
    if set(facts) != {"terminal_native_assertions", "native_manifest_findings", "current_action_observations"}:
        raise ValueError("unexpected factual feedback fields")
    return facts


def compose(historical_teacher, old_feedback, guidance, facts):
    if historical_teacher.count(old_feedback) != 1:
        raise ValueError("historical feedback not uniquely present in decoded teacher prefix")
    supplement = (
        "Teacher-only task guidance follows. This is a reviewed specification, not a tool observation, "
        "completed action, successful demonstration or proof that the student's action is wrong. "
        "Concrete IDs, policy rules and values must come from the public request or actual tool results.\n"
        + json.dumps({"task_guidance": guidance}, ensure_ascii=False, indent=2)
        + "\nFactual hindsight from this student episode:\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\nEvidence boundaries:\n"
        "The original conversation above is the student's decision-time history. "
        "current_action_observations contains the recorded result associated with the selected action "
        "(or the preceding tool response for a terminal answer); this is retrospective feedback, "
        "not an observation already present at the original decision point. "
        "terminal_native_assertions describes the completed attempt. Parameters in those assertions "
        "can contain privileged expected values: they are evaluation hindsight, not values retrieved "
        "by the student and not permission to bypass discovery. Preserve missing manifest signals as "
        "unknown. A failed episode is not proof that this selected action caused the failure. "
        "Tool success does not establish correct scope or content. Empty results do not establish that "
        "no eligible records exist. An error does not establish that the task is impossible. "
        "Do not invent unavailable read-back tools or silently repeat an irreversible write.\n"
        "Correctly continue the original task at the original decision point. Produce the next "
        "assistant action in the native reasoning/tool format, not a critique. Do not claim later "
        "events have already occurred, cite this teacher-only guidance, or expose evaluation labels."
    )
    candidate = historical_teacher.replace(old_feedback, supplement, 1)
    before, after = historical_teacher.split(old_feedback, 1)
    if candidate != before + supplement + after or not candidate.endswith(GENERATION_PREFIX):
        raise ValueError("historical conversation/template changed during composition")
    return candidate, supplement, before, after


def build(root, selected_path, guidance_path, tokenizer_path, output):
    rows = json.loads(selected_path.read_text())
    guidance_bundle = json.loads(guidance_path.read_text())
    authored = root / "docs/research/evidence/deepseek-context32-20261019-r1"
    inputs = {i["task"]: i for i in json.loads((authored / "inputs.json").read_text())}
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output.mkdir(parents=True, exist_ok=False)
    results, sections = [], []
    for task, reviewed in guidance_bundle["tasks"].items():
        matching = [r for r in rows if r["task"] == task and r["native_success"] == 0]
        if not matching:
            raise ValueError(f"no recorded failed episode for {task}")
        row = matching[0]
        inp = inputs[task]
        public = inp["public_request"]
        student = tokenizer.decode(row["student_prefix_ids"], skip_special_tokens=False)
        historical = tokenizer.decode(row["teacher_prefix_ids"], skip_special_tokens=False)
        action = tokenizer.decode(row["action_ids"], skip_special_tokens=False)
        if public not in student or public not in historical:
            raise ValueError(f"public instruction binding mismatch: {task}")
        known_tools = {t["function"]["name"] for t in inp["available_tools"]}
        guide = reviewed["guidance"]
        for step in guide["workflow"]:
            if not set(step["tools"]) <= known_tools:
                raise ValueError(f"unavailable workflow tool for {task}")
        manifest_checks = inp["goal_harm_contract"]["checks"]
        if {x["check_id"]: x["role"] for x in guide["contract_scope"]} != {
                x["check_id"]: x["role"] for x in manifest_checks}:
            raise ValueError(f"manifest check coverage mismatch: {task}")
        facts = parse_feedback(row["teacher_feedback"])
        if not any(a.get("passed") is False and not a.get("excluded")
                   for a in facts["terminal_native_assertions"]):
            raise ValueError("no decided non-excluded failed assertion")
        candidate, supplement, history, suffix = compose(historical, row["teacher_feedback"], guide, facts)
        folder = output / task
        folder.mkdir()
        for name, content in [("full-teacher-context.txt", candidate), ("teacher-supplement.txt", supplement),
                              ("historical-teacher-context.txt", historical), ("original-student-prefix.txt", student),
                              ("sampled-action-review-only.txt", action)]:
            (folder / name).write_text(content)
        (folder / "actual-feedback.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")
        metadata = {"task": task, "episode_id": row["episode_id"], "update": 1,
                    "generation_index": row["generation_index"], "policy_version": row["policy_version"],
                    "native_success": row["native_success"], "terminal_text": row["terminal_text"],
                    "observed_tool_error": row["observed_tool_error"],
                    "historical_source": str(selected_path), "historical_source_sha256": digest_bytes(selected_path.read_bytes()),
                    "public_request_sha256": digest_bytes(public.encode()),
                    "original_deepseek_context_sha256": digest_bytes((authored / task / "context.json").read_bytes()),
                    "guidance_provenance": guidance_bundle["provenance"], "review_edits": reviewed["edits"],
                    "guidance_file_sha256": digest_bytes(guidance_path.read_bytes()),
                    "candidate_sha256": digest_bytes(candidate.encode()),
                    "history_outside_teacher_supplement_unchanged": True,
                    "rendered_history_sha256": digest_bytes(history.encode()),
                    "historical_template_suffix_sha256": digest_bytes(suffix.encode()),
                    "feedback_preserved_exactly_as_json": parse_feedback(supplement) == facts,
                    "candidate_input_token_count_offline": len(tokenizer.encode(candidate, add_special_tokens=False).ids),
                    "scope": "OFFLINE CANDIDATE; never submitted to a model or used in training",
                    "sampled_action_in_separate_review_panel_only": True,
                    "source_state_binding": "Exact public request and recorded episode/policy are bound; no fresh environment-state requalification performed.",
                    "review_note": reviewed["episode_note"]}
        if metadata["candidate_input_token_count_offline"] + len(row["action_ids"]) > 32768:
            raise ValueError("candidate plus original selected action exceeds prior training context limit")
        (folder / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results.append(metadata)
        esc = html.escape
        sections.append(f'<section id="{esc(task)}"><h2>{esc(task)}</h2>'
                        f'<p>Real episode <code>{row["episode_id"]}</code> · update1 · native success0</p>'
                        f'<p><a href="{esc(task)}/full-teacher-context.txt">Open full teacher context as text</a> · '
                        f'<a href="{esc(task)}/teacher-supplement.txt">Open guidance + feedback as text</a></p>'
                        f'<p><b>Review:</b> {esc(reviewed["episode_note"])}</p>'
                        f'<details><summary>Guidance edits and provenance</summary><pre>{esc(json.dumps(metadata, indent=2))}</pre></details>'
                        f'<details open><summary>Teacher-only supplement: guidance + complete actual feedback</summary><pre>{esc(supplement)}</pre></details>'
                        f'<details><summary>FULL candidate teacher context: tools, original history and supplement (no omissions)</summary><pre>{esc(candidate)}</pre></details>'
                        f'<details><summary>Sampled student action — inspection only, not inserted into teacher prefix</summary><pre>{esc(action)}</pre></details>'
                        f'<details><summary>Historical teacher context for comparison</summary><pre>{esc(historical)}</pre></details></section>')
    receipt = {"status": "four_offline_candidates_assembled", "model_calls": 0, "cost_usd": 0,
               "training_restarted": False, "native_or_gradient_qualified": False, "episodes": results}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    navigation = " · ".join(f'<a href="#{html.escape(r["task"])}">{html.escape(r["task"])}</a>' for r in results)
    page = ('<!doctype html><meta charset="utf-8"><title>Four real episodes: combined teacher contexts</title>'
            '<style>body{max-width:1180px;margin:36px auto;padding:0 22px;font:16px/1.55 system-ui;color:#17273b;background:#f7f9fc}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;border:1px solid #d5deea;border-radius:7px}'
            'section{border-top:2px solid #cbd7e6;margin:42px 0}summary{cursor:pointer;font-weight:650;margin:15px 0}'
            '.note{background:#fff1d6;padding:16px}a{color:#135abd}</style>'
            '<h1>Four real failed episodes: candidate teacher contexts</h1>'
            '<p>Original experiment03, update1. Full recorded tools/history, actual tool results and terminal feedback; '
            'DeepSeek task guidance corrected by source review. No placeholders or generated tool responses.</p>'
            '<p class="note"><b>Offline review only; training remains on hold.</b> These are proposed new prompts, '
            'not prompts previously used in training. No teacher inference, gradients or solver improvements were measured. '
            'Terminal feedback retains historical expected-value parameters; these are privileged hindsight, not student observations. '
            'A failed episode does not prove its selected action caused the failure.</p>'
            f'<nav>{navigation}</nav>' + "".join(sections))
    (output / "four-full-teacher-contexts.html").write_text(page)
    print(json.dumps({"status": receipt["status"], "model_calls": 0,
                      "episodes": [{k: r[k] for k in ["task", "episode_id", "candidate_input_token_count_offline"]} for r in results]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("/tmp/sdpo-qwen-tokenizer/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Path.cwd(), args.selected, args.guidance, args.tokenizer, args.output)
