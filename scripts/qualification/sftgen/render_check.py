"""Render check on the MATERIALIZED training file (extends J-sft/rootcause/render_check.py).

For built rows (session_to_row output, or any SupervisedDataset JSONL rows) this:
  1. materializes them through the framework's own catalog path (decode_dataset_selection, jsonl source,
     materialize_dataset -> state/datasets/<fingerprint>/data.jsonl), unless --materialized gives an existing file;
  2. round trip: each materialized line equals the built row serialized the same way, with only the adapter's two
     metadata fields (source_row, source_format) appended; messages, tools and trainable indices are byte-identical;
  3. loads the materialized file as the training job does (load_materialized_dataset) and renders it with the
     framework (render_supervised, QWEN35_RENDERER, thinking off, the real Qwen3.5-4B tokenizer 851bf6e);
  4. per row: token ids identical to the runner's chat-template render of the BUILT row (apply_chat_template with
     the tools, enable_thinking=False; the framework drops the template's trailing newline after the last target);
     for every assistant message k the runner's prompt at that turn (apply_chat_template(messages[:k],
     add_generation_prompt=True, enable_thinking=False)) is exactly the token prefix before the k-th trainable span;
     the loss mask is exactly the assistant turns: one trainable span per assistant message, each decoding to that
     message's body (text, tool calls, final reply) ending in <|im_end|>, with no system/user/tool text inside.

Usage (wt-sft .venv; CPU):
  CAPPED_MEM=4G .../bin/capped .venv/bin/python scripts/qualification/sftgen/render_check.py ROWS.jsonl OUTDIR \
      [--tokenizer DIR] [--materialized DATA.jsonl] [--dump ID ...]
  render_check.py --teacher-shaped SRC.jsonl OUT.jsonl [--count 3]   # build synthetic teacher-shaped test rows
Writes OUTDIR/render-check.json; exit 0 only if every row passes every check.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
from settings import TOKENIZER  # noqa: E402
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GEN_PROMPT_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
GEN_PROMPT_TAIL_THINKING = "<|im_start|>assistant\n<think>\n"
THINKING = False  # --thinking: QWEN35_THINKING_RENDERER and the template with enable_thinking=True
EOS = "<|im_end|>"


def framework_model():
    from posttrain.common.artifacts import HubModelRef
    from posttrain.common.variants.qwen35 import QWEN_35_2B, QWEN35_THINKING_RENDERER_CONTRACT

    ref = HubModelRef(repo_id="Qwen/Qwen3.5-4B", revision=MODEL_REVISION)
    # models/qwen3.5-4b@bf16 as the base catalog declares it (renderer contract qwen3.5-tools-thinking@1)
    return replace(QWEN_35_2B, id="qwen3.5-4b", artifact=ref, base=ref, parameters=4_000_000_000,
                   renderer=QWEN35_THINKING_RENDERER_CONTRACT)


def materialize(rows_path: Path, out: Path) -> tuple[Path, object, object]:
    """Copy the rows into a scratch project and materialize them with the framework's catalog code."""
    from posttrain.common import CatalogRef
    from posttrain.data import decode_dataset_selection, materialize_dataset

    project = out / "project"
    (project / "data").mkdir(parents=True, exist_ok=True)
    (project / "data" / "rows.jsonl").write_bytes(rows_path.read_bytes())
    plan = decode_dataset_selection(
        CatalogRef("dataset", "datasets/sftgen-render-check@1"),
        {"id": "datasets/sftgen-render-check@1", "revision": "1", "kind": "supervised",
         "source": {"kind": "jsonl", "path": "data/rows.jsonl"}, "format": {"kind": "messages"}}, {})
    materialization = materialize_dataset(plan, state_dir=out / "state", project_root=project)
    return materialization.path, plan, materialization


def load_like_the_job(plan, materialization):
    from posttrain.data import load_materialized_dataset

    return load_materialized_dataset(plan, materialization)


def round_trip(built: list[dict], materialized_text: str) -> list[dict]:
    lines = materialized_text.splitlines(keepends=True)
    out = []
    for index, row in enumerate(built):
        expected = {**row, "metadata": {**row.get("metadata", {}), "source_row": index, "source_format": "messages"}}
        line = lines[index] if index < len(lines) else ""
        fields = {f: json.dumps(row[f]) in line for f in ("messages", "tools", "trainable_message_indices")}
        out.append({"id": row["id"], "line_equal": line == json.dumps(expected) + "\n", **{f"{k}_bytes_equal": v
                                                                                           for k, v in fields.items()}})
    if len(lines) != len(built):
        out.append({"id": "(count)", "line_equal": False, "materialized_lines": len(lines), "built_rows": len(built)})
    return out


def spans(ids, labels):
    runs, current, flag, start = [], [], None, 0
    for position, (token, label) in enumerate(zip(ids, labels)):
        trainable = label != -100
        if trainable != flag and current:
            runs.append((flag, start, current))
            current = []
        if not current:
            start = position
        flag = trainable
        current.append(token)
    if current:
        runs.append((flag, start, current))
    return runs


def check_row(row: dict, sample, tok) -> dict:
    messages, tools = row["messages"], row["tools"]
    reference = tok.apply_chat_template(messages, tools=tools, tokenize=True, enable_thinking=THINKING,
                                        return_dict=True)["input_ids"]
    reference = list(reference)
    newline = tok("\n", add_special_tokens=False)["input_ids"]
    if reference[-len(newline):] == newline:
        reference = reference[:-len(newline)]  # the framework drops tokens after the last target
    ids, labels = list(sample.input_ids), list(sample.labels)
    first_diff = next((i for i, (a, b) in enumerate(zip(ids, reference)) if a != b), None)
    if first_diff is None and len(ids) != len(reference):
        first_diff = min(len(ids), len(reference))
    result = {"id": row["id"], "tokens": len(ids), "ids_equal_runner_render": first_diff is None}
    if first_diff is not None:
        result["near_framework"] = tok.decode(ids[max(0, first_diff - 10):first_diff + 10])
        result["near_runner"] = tok.decode(reference[max(0, first_diff - 10):first_diff + 10])
    runs = [r for r in spans(ids, labels) if r[0]]
    assistant = list(row["trainable_message_indices"])  # all assistant messages unless turns are masked (13 Nov)
    if any(messages[k]["role"] != "assistant" for k in assistant):
        return {**result, "mask_exact": False, "mask_problems": ["a trainable index is not an assistant message"],
                "pass": False}
    problems = []
    if len(runs) != len(assistant):
        problems.append(f"{len(runs)} trainable spans for {len(assistant)} trainable assistant messages")
    for (_, start, tokens), k in zip(runs, assistant, strict=False):
        prompt = tok.apply_chat_template(messages[:k], tools=tools, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=THINKING, return_dict=True)["input_ids"]
        if list(prompt) != ids[:start]:
            problems.append(f"message {k}: runner prompt at this turn is not the context before its target")
        prefix_text = tok.apply_chat_template(messages[:k], tools=tools, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=THINKING)
        full_text = tok.apply_chat_template(messages[:k + 1], tools=tools, tokenize=False, enable_thinking=THINKING)
        target_text = full_text[len(prefix_text):].rstrip("\n")
        decoded = tok.decode(tokens)
        if not prefix_text.endswith(GEN_PROMPT_TAIL_THINKING if THINKING else GEN_PROMPT_TAIL) \
                or not full_text.startswith(prefix_text):
            problems.append(f"message {k}: template history is not the thinking-off prompt extended")
        if decoded != target_text:
            problems.append(f"message {k}: trainable span differs from the message body")
        if not decoded.endswith(EOS) or any(s in decoded for s in ("<tool_response>", "<|im_start|>", "<think>\n")):
            problems.append(f"message {k}: span has context tokens or no {EOS}")
    result["mask_exact"] = not problems
    result["mask_problems"] = problems
    result["trainable_tokens"] = sum(label != -100 for label in labels)
    result["pass"] = result["ids_equal_runner_render"] and result["mask_exact"]
    return result


def dump(sample, tok, path: Path) -> None:
    text = "".join(("[[T]]" + tok.decode(t) + "[[/T]]") if f else tok.decode(t)
                   for f, _, t in spans(list(sample.input_ids), list(sample.labels)))
    path.write_text(text)


def run(rows_path: Path, out: Path, tokenizer_dir: Path, materialized: Path | None, dump_ids: set[str]) -> dict:
    from posttrain.data import supervised_from_huggingface
    from posttrain.train import QWEN35_RENDERER, QWEN35_THINKING_RENDERER, render_supervised
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    built = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if materialized is None:
        data_path, plan, materialization = materialize(rows_path, out)
        dataset = load_like_the_job(plan, materialization)
    else:
        data_path = materialized
        rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
        dataset = supervised_from_huggingface(rows, dataset_id="sftgen-render-check", revision="1",
                                              format="messages")
    trips = round_trip(built, data_path.read_text())
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)  # framework's (it may set
    # the variant's template on it); the reference render uses its own, untouched tokenizer
    ref_tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    template_file = tokenizer_dir / "chat_template.jinja"
    template_text = template_file.read_text(encoding="utf-8") if template_file.exists() else None
    # the reference render must use exactly the model's chat_template.jinja (the runner's vLLM uses the same file)
    template_is_file = ref_tok.chat_template == template_text
    samples = render_supervised(tok, framework_model(), dataset,
                                QWEN35_THINKING_RENDERER if THINKING else QWEN35_RENDERER, max_length=10**9)
    framework_template_is_file = tok.chat_template == template_text
    by_id = {s.id: s for s in samples}
    rows = []
    for row in built:
        sample = by_id[row["id"]]
        rows.append(check_row(row, sample, ref_tok))
        if row["id"] in dump_ids or (not dump_ids and len(rows) <= 2):
            dump(sample, tok, out / (row["id"].replace("/", "_") + ".rendered.txt"))
    summary = {"rows": len(built), "materialized_file": str(data_path),
               "round_trip_line_equal": sum(t["line_equal"] for t in trips if t["id"] != "(count)"),
               "ids_equal_runner_render": sum(r["ids_equal_runner_render"] for r in rows),
               "mask_exact": sum(r["mask_exact"] for r in rows), "passed": sum(r["pass"] for r in rows),
               "tokenizer": str(tokenizer_dir), "chat_template_file": str(template_file),
               "chat_template_is_file": template_is_file,
               "framework_tokenizer_template_is_file": framework_template_is_file,
               "rows_with_multi_call_turns": sum(any(len(m.get("tool_calls") or []) > 1 for m in r["messages"])
                                                 for r in built)}
    summary["all_pass"] = (summary["passed"] == len(built) and summary["round_trip_line_equal"] == len(built)
                           and all(t["line_equal"] for t in trips) and template_is_file)
    report = {"summary": summary, "round_trip": trips, "rows": rows}
    (out / "render-check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


# ----------------------------------------------------------------------------- synthetic teacher-shaped rows
def teacher_shaped(rows: list[dict], count: int = 3) -> list[dict]:
    """Test rows in the Route B shape built from existing rows: consecutive calls merged into multi-call turns,
    a short note before every calling turn, the row's final reply kept. Synthetic (for render tests only)."""
    picked = []
    for row in rows:
        calls = [m for m in row["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
        if len(calls) >= 3 and row["messages"][-1]["role"] == "assistant" and not row["messages"][-1].get("tool_calls"):
            picked.append(row)
        if len(picked) == count:
            break
    out = []
    for n, row in enumerate(picked):
        messages = [copy.deepcopy(m) for m in row["messages"][:2]]
        pending = []  # (assistant calls, tool messages)
        body = row["messages"][2:-1]
        i = 0
        while i < len(body):
            m = body[i]
            k = len(m.get("tool_calls") or [])
            pending.append((m["tool_calls"], body[i + 1:i + 1 + k]))
            i += 1 + k
        group = 2 + n % 2  # 2 or 3 calls per turn
        for g in range(0, len(pending), group):
            chunk = pending[g:g + group]
            calls = [copy.deepcopy(c) for cs, _ in chunk for c in cs]
            note = (f"Next I will run {len(calls)} calls together: " +
                    ", ".join(c["function"]["name"] for c in calls) + ". Café “ünïcode” – noted.")
            messages.append({"role": "assistant", "content": note, "tool_calls": calls})
            messages += [copy.deepcopy(t) for _, ts in chunk for t in ts]
        messages.append(copy.deepcopy(row["messages"][-1]))
        out.append({"id": f"sftgen-teacher-shaped/{n}", "messages": messages, "tools": copy.deepcopy(row["tools"]),
                    "trainable_message_indices": [i for i, m in enumerate(messages) if m["role"] == "assistant"],
                    "metadata": {"synthetic": "teacher-shaped render test row", "from": row["id"]}})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="?")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--tokenizer", default=str(TOKENIZER))
    parser.add_argument("--materialized", help="check this already-materialized data.jsonl instead of materializing")
    parser.add_argument("--dump", nargs="*", default=[])
    parser.add_argument("--teacher-shaped", nargs=2, metavar=("SRC", "OUT"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--thinking", action="store_true", help="thinking-on rows (reasoning_content on every turn)")
    args = parser.parse_args(argv)
    global THINKING
    THINKING = args.thinking
    if args.teacher_shaped:
        src, dst = map(Path, args.teacher_shaped)
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        shaped = teacher_shaped([r for r in rows if str(r["metadata"].get("source", "")).startswith("claude")],
                                args.count)
        dst.write_text("".join(json.dumps(r) + "\n" for r in shaped))
        print(json.dumps({"rows": len(shaped), "out": str(dst)}))
        return 0
    report = run(Path(args.rows), Path(args.out), Path(args.tokenizer),
                 Path(args.materialized) if args.materialized else None, set(args.dump))
    print(json.dumps(report["summary"], indent=1))
    for r in report["rows"]:
        if not r["pass"]:
            print("FAIL", r["id"], {k: v for k, v in r.items() if k not in ("id",)})
    for t in report["round_trip"]:
        if not t["line_equal"]:
            print("ROUND-TRIP FAIL", t)
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
