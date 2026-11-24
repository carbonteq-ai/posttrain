"""Shape report: a set of SFT rows against the 4B's own clean successes (the reference interface shape).

The ladder of 12 Nov collapsed the 4B's planning text and parallel calls because the training rows had neither
(OVERNIGHT-REPORT root cause 1) and because materialization key-sorted tools and arguments (root cause 3). This report
measures those shape properties on a candidate row set and on the reference (rows whose metadata source starts with
"own" in J-sft data/sft-phase2a-armT-final.jsonl: the untrained 4B's own clean successes) and applies bars:

  text_share            share of calling turns with visible text          candidate >= reference - 0.05
  multi_call_share      share of calling turns with 2+ calls              candidate >= 0.5 x reference
  mean_calls_per_turn   calls per calling turn                            candidate <= 2.0 x reference
  note_chars_median     median text length of calling turns (with text)   0.5x .. 2.0x reference
  reply_chars_median    median final-reply length                         0.5x .. 2.0x reference
  sorted_args_share     multi-key argument objects whose keys are sorted  candidate <= reference + 0.20
  tools_key_order       (with --runner-prompts) tools byte-equal to the runner's tools for the task: every row;
                        without it: tool entries keep {type, function{name, description, parameters}} order and
                        parameter properties are not all sorted where the reference's are not

The bars are PROVISIONAL (set 12 Nov, before any effect pilot): they flag rows that teach a shape the 4B does not
have; they say nothing about task success. Override with --bar NAME=VALUE (the multiplier or offset).

  python shape_report.py ROWS.jsonl [--own SFT.jsonl] [--runner-prompts runner_prompts.jsonl] [--json OUT.json]
Pure Python. Exit 0 only if every bar passes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import os
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OWN = Path(os.environ.get("SFTGEN_OWN_ROWS", "sftgen-work/own-rows.jsonl"))
BARS = {"text_share_offset": 0.05, "multi_call_share_ratio": 0.5, "mean_calls_ratio": 2.0, "note_chars_low": 0.5,
        "note_chars_high": 2.0, "reply_chars_low": 0.5, "reply_chars_high": 2.0, "sorted_args_offset": 0.20}
HISTOGRAM = ("1", "2", "3", "4", "5", "6+")


def load(path: Path) -> list[dict]:
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def own_rows(path: Path) -> list[dict]:
    return [r for r in load(path) if str(r.get("metadata", {}).get("source", "")).startswith("own")]


def _arguments(call: dict):
    args = call["function"].get("arguments")
    return json.loads(args) if isinstance(args, str) else args


def _objects(value):
    """Every mapping inside a value (the value itself first), depth first."""
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _objects(v)
    elif isinstance(value, list):
        for v in value:
            yield from _objects(v)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def measure(rows: list[dict], runner: dict | None = None) -> dict:
    turns, texts, replies, sizes = 0, [], [], []
    with_text, multi = 0, 0
    sorted_args = multikey_args = 0
    tool_entries_ordered = tool_entries = props_sorted = props_multi = 0
    tools_equal = tools_compared = 0
    for row in rows:
        messages = row["messages"]
        for m in messages:
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            turns += 1
            n = len(m["tool_calls"])
            sizes.append(n)
            multi += n >= 2
            text = (m.get("content") or "").strip()
            if text:
                with_text += 1
                texts.append(len(text))
            for call in m["tool_calls"]:
                for obj in _objects(_arguments(call)):
                    if len(obj) >= 2:
                        multikey_args += 1
                        sorted_args += list(obj) == sorted(obj)
        last = messages[-1]
        if last.get("role") == "assistant" and not last.get("tool_calls"):
            replies.append(len((last.get("content") or "").strip()))
        for tool in row.get("tools", []):
            tool_entries += 1
            fn = tool.get("function", {})
            tool_entries_ordered += list(tool) == ["type", "function"] and list(fn)[:3] == ["name", "description",
                                                                                            "parameters"]
            props = (fn.get("parameters") or {}).get("properties") or {}
            if len(props) >= 2:
                props_multi += 1
                props_sorted += list(props) == sorted(props)
        if runner is not None:
            reference = runner.get(row.get("metadata", {}).get("task"))
            if reference is not None:
                tools_compared += 1
                tools_equal += (json.dumps(row["tools"], ensure_ascii=False) ==
                                json.dumps(reference["tools"], ensure_ascii=False))
    histogram = Counter(str(s) if s < 6 else "6+" for s in sizes)
    return {
        "rows": len(rows), "calling_turns": turns,
        "text_share": with_text / turns if turns else 0.0,
        "multi_call_share": multi / turns if turns else 0.0,
        "mean_calls_per_turn": sum(sizes) / turns if turns else 0.0,
        "calls_per_turn_histogram": {k: histogram.get(k, 0) for k in HISTOGRAM},
        "max_calls_in_turn": max(sizes, default=0),
        "note_chars_median": statistics.median(texts) if texts else 0.0,
        "note_chars_p90": _percentile(texts, 0.9),
        "reply_chars_median": statistics.median(replies) if replies else 0.0,
        "reply_chars_p90": _percentile(replies, 0.9),
        "rows_with_final_reply": len(replies),
        "sorted_args_share": sorted_args / multikey_args if multikey_args else 0.0,
        "multikey_argument_objects": multikey_args,
        "tool_entries_key_order_kept": f"{tool_entries_ordered}/{tool_entries}",
        "sorted_properties_share": props_sorted / props_multi if props_multi else 0.0,
        "tools_equal_runner": f"{tools_equal}/{tools_compared}" if runner is not None else "not compared",
        "_tools": (tool_entries_ordered, tool_entries, tools_equal, tools_compared),
    }


def apply_bars(cand: dict, ref: dict, bars: dict, runner_given: bool) -> list[dict]:
    out = []

    def bar(name: str, ok: bool, rule: str) -> None:
        out.append({"metric": name, "candidate": cand.get(name), "reference": ref.get(name), "bar": rule,
                    "pass": bool(ok)})

    low = ref["text_share"] - bars["text_share_offset"]
    bar("text_share", cand["text_share"] >= low, f">= {low:.3f}")
    low = ref["multi_call_share"] * bars["multi_call_share_ratio"]
    bar("multi_call_share", cand["multi_call_share"] >= low, f">= {low:.3f}")
    high = ref["mean_calls_per_turn"] * bars["mean_calls_ratio"]
    bar("mean_calls_per_turn", cand["mean_calls_per_turn"] <= high, f"<= {high:.3f}")
    lo, hi = ref["note_chars_median"] * bars["note_chars_low"], ref["note_chars_median"] * bars["note_chars_high"]
    bar("note_chars_median", lo <= cand["note_chars_median"] <= hi, f"{lo:.0f} .. {hi:.0f}")
    lo, hi = ref["reply_chars_median"] * bars["reply_chars_low"], ref["reply_chars_median"] * bars["reply_chars_high"]
    bar("reply_chars_median", lo <= cand["reply_chars_median"] <= hi, f"{lo:.0f} .. {hi:.0f}")
    high = ref["sorted_args_share"] + bars["sorted_args_offset"]
    bar("sorted_args_share", cand["sorted_args_share"] <= high, f"<= {high:.3f}")
    ordered, entries, equal, compared = cand["_tools"]
    if runner_given:
        ok = compared == cand["rows"] and equal == compared
        out.append({"metric": "tools_key_order", "candidate": f"{equal}/{compared} rows byte-equal to the runner "
                    f"(of {cand['rows']})", "reference": ref["tools_equal_runner"], "bar": "every row", "pass": ok})
    else:
        ok = ordered == entries and not (cand["sorted_properties_share"] == 1.0 and ref["sorted_properties_share"] < 1)
        out.append({"metric": "tools_key_order", "candidate": f"entries {ordered}/{entries}; sorted properties "
                    f"{cand['sorted_properties_share']:.3f}", "reference": f"sorted properties "
                    f"{ref['sorted_properties_share']:.3f}", "bar": "entries in order; properties not all sorted",
                    "pass": ok})
    return out


def histogram_lines(name: str, hist: dict, width: int = 40) -> list[str]:
    total = sum(hist.values()) or 1
    lines = [f"  calls per turn, {name}:"]
    for key in HISTOGRAM:
        share = hist[key] / total
        lines.append(f"    {key:>2} {'#' * round(share * width):<{width}} {hist[key]:5d} ({share:.1%})")
    return lines


def report(cand_rows: list[dict], ref_rows: list[dict], runner: dict | None, bars: dict) -> dict:
    cand, ref = measure(cand_rows, runner), measure(ref_rows, runner)
    results = apply_bars(cand, ref, bars, runner is not None)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    return {"candidate": strip(cand), "reference": strip(ref), "bars": results, "bar_settings": bars,
            "all_pass": all(r["pass"] for r in results)}


def render(result: dict) -> str:
    c, r = result["candidate"], result["reference"]
    lines = [f"candidate: {c['rows']} rows, {c['calling_turns']} calling turns; reference (own clean successes): "
             f"{r['rows']} rows, {r['calling_turns']} calling turns", ""]
    lines.append(f"  {'metric':<22} {'candidate':>32} {'reference':>14}  {'bar':<26} result")
    for b in result["bars"]:
        cv, rv = b["candidate"], b["reference"]
        cv = f"{cv:.3f}" if isinstance(cv, float) else str(cv)
        rv = f"{rv:.3f}" if isinstance(rv, float) else str(rv)
        lines.append(f"  {b['metric']:<22} {cv:>32} {rv:>14}  {b['bar']:<26} {'PASS' if b['pass'] else 'FAIL'}")
    lines.append("")
    lines += histogram_lines("candidate", c["calls_per_turn_histogram"])
    lines += histogram_lines("reference", r["calls_per_turn_histogram"])
    lines.append("")
    lines.append(f"  note chars p90: candidate {c['note_chars_p90']}, reference {r['note_chars_p90']}; reply chars p90: "
                 f"candidate {c['reply_chars_p90']}, reference {r['reply_chars_p90']}; max calls in a turn: candidate "
                 f"{c['max_calls_in_turn']}, reference {r['max_calls_in_turn']}")
    lines.append(f"  rows with a final reply: candidate {c['rows_with_final_reply']}/{c['rows']}")
    lines.append("")
    lines.append("SHAPE " + ("PASS" if result["all_pass"] else "FAIL") + " (bars are provisional; see the docstring)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows")
    parser.add_argument("--own", default=str(DEFAULT_OWN), help="file whose rows with source own* are the reference")
    parser.add_argument("--runner-prompts", help="runner_prompts.jsonl: compare each row's tools with the runner's")
    parser.add_argument("--bar", action="append", default=[], metavar="NAME=VALUE", help=f"override: {sorted(BARS)}")
    parser.add_argument("--json", help="write the full result here")
    args = parser.parse_args(argv)
    bars = dict(BARS)
    for item in args.bar:
        name, _, value = item.partition("=")
        if name not in bars:
            parser.error(f"unknown bar {name!r}")
        bars[name] = float(value)
    runner = None
    if args.runner_prompts:
        runner = {r["task"]: r for r in load(Path(args.runner_prompts))}
    ref_rows = own_rows(Path(args.own))
    if not ref_rows:
        parser.error(f"no rows with source own* in {args.own}")
    result = report(load(Path(args.rows)), ref_rows, runner, bars)
    print(render(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
