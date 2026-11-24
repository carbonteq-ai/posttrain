"""Make an attempt's tool-call arguments render exactly as the SFT model sampled them.

The rows hold the runner's parsed call arguments; the chat template renders a non-string value in its own spelling
(JSON `null`/`true`/`false` as Python `None`/`True`/`False`, `4200.00` as `4200.0`, `0.0` as `0`), so the training text
would differ from the tokens P9 wrote. For every non-string argument whose sampled text (the `<parameter=NAME>` block of
P9's own action segment in its recorded chain) parses as JSON to exactly the same value, the argument is replaced by that
sampled text (a string renders verbatim). Meaning is unchanged (the parsed values are equal); only the spelling becomes
P9's own. Anything else is left as is and reported (pair_parity.py then flags it). The rows are only rendered for DPO,
never replayed (replay ran on the original rows in phase 1).
usage (pyfix11): sampled_spelling.py ROWS.jsonl CHAINS.jsonl.gz TOKENIZER.json OUT.jsonl
"""
import gzip
import json
import re
import sys

CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)


def sampled_calls(action_text: str) -> list[tuple[str, dict[str, str]]]:
    return [(m.group(1), {p.group(1): p.group(2) for p in PARAM.finditer(m.group(2))})
            for m in CALL.finditer(action_text)]


def restore(row: dict, action_texts: list[str]) -> tuple[dict, int, list[str]]:
    """(row with sampled spellings, values replaced, problems). action_texts: decoded action segments in order, one per
    assistant message."""
    row = json.loads(json.dumps(row))
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    if len(assistants) != len(action_texts):
        return row, 0, [f"{len(assistants)} assistant messages vs {len(action_texts)} action segments"]
    n, problems = 0, []
    for k, (msg, text) in enumerate(zip(assistants, action_texts)):
        calls = msg.get("tool_calls") or []
        sampled = sampled_calls(text)
        if len(sampled) != len(calls):
            if calls:
                problems.append(f"turn {k + 1}: {len(calls)} calls vs {len(sampled)} sampled blocks")
            continue
        for call, (name, params) in zip(calls, sampled):
            if call["function"]["name"] != name:
                problems.append(f"turn {k + 1}: call {call['function']['name']} vs sampled {name}")
                continue
            args = call["function"]["arguments"]
            for key, value in list(args.items()):
                if isinstance(value, str) or key not in params:
                    continue
                try:
                    same = json.loads(params[key]) == value and type(json.loads(params[key])) is not str
                except ValueError:
                    same = False
                if same:  # even when JSON spells it the same: the template renders the parsed value (None, True)
                    args[key] = params[key]
                    n += 1
    return row, n, problems


def main(rows_path, chains_path, tokenizer_path, out):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    rows = [json.loads(line) for line in open(rows_path)]
    want = {r["metadata"]["episode_id"] for r in rows}
    actions = {}
    with gzip.open(chains_path, "rt") as s:
        for line in s:
            c = json.loads(line)
            if c["episode_id"] in want:
                actions[c["episode_id"]] = [tok.decode(seg["ids"], skip_special_tokens=False)
                                            for seg in c["segments"] if seg["kind"] == "action"]
    total, changed, problems = 0, 0, {}
    with open(out, "w") as w:
        for r in rows:
            new, n, prob = restore(r, actions[r["metadata"]["episode_id"]])
            if n:
                new["metadata"]["sampled_spelling_restored"] = n
                changed += 1
            total += n
            if prob:
                problems[r["id"]] = prob
            w.write(json.dumps(new, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "rows_changed": changed, "values_restored": total,
                      "rows_with_problems": len(problems), "problems_sample": dict(list(problems.items())[:5])}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
