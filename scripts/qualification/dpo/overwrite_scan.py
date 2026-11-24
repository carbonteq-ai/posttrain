"""Overwrite scan: a chosen attempt that writes the same record twice with different values did a wrong action and corrected it later.
twice with different values did a wrong action and corrected it later (po_three_way_match: generic statuses, then the
policy's labels). A TRAINED turn's write is compared with the earlier writes of any turn; in a row whose wrong turn
is already masked (SFT's "wrong action, corrected later"), the trained correction still flags, which is expected and
read as such; an untrained (masked) rewrite never flags.
A write is a call whose tool name contains update/set/change/patch/edit/modify; the record is the tool plus its
identifying arguments (spreadsheet, worksheet, row, card, ticket, ... and every key that is `id` or ends in `_id`/`_key`).
The remaining arguments are compared field by field (JSON-string arguments such as a sheet's `cells` are parsed; values
compared as text, so "2" and 2 are equal): only a field written before and now written with another value flags.
usage: overwrite_scan.py ROWS.jsonl [ROWS.jsonl ...]   (prints one line per flagged attempt; exit 0)"""
import json
import re
import sys

WRITE = re.compile(r"update|_set|set_|change|patch|edit|modify", re.I)
IDENT = re.compile(r"^(id|row|spreadsheet|worksheet|board|item|card|ticket|deal|issue|page|conversation|contact|task|"
                   r"record|message|file|event)$|_id$|_key$", re.I)


def fields(args: dict) -> dict:
    """Non-identifying arguments flattened to field -> text (JSON-string values parsed one level)."""
    out = {}
    for k, v in args.items():
        if IDENT.search(k):
            continue
        if isinstance(v, str) and v[:1] in "{[":
            try:
                v = json.loads(v)
            except ValueError:
                pass
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}" if k not in ("cells", "additional_properties_json", "properties") else k2] = str(v2)
        else:
            out[k] = str(v)
    return out


def overwrites(row: dict) -> list[dict]:
    trained = set(row["trainable_message_indices"])
    seen, found = {}, []
    for i, m in enumerate(row["messages"]):
        for c in m.get("tool_calls") or [] if m["role"] == "assistant" else []:
            name, args = c["function"]["name"], c["function"]["arguments"]
            if not WRITE.search(name) or not isinstance(args, dict):
                continue
            ident = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in args.items() if IDENT.search(k)))
            if not ident:
                continue
            rec = (name, ident)
            for f, value in fields(args).items():
                if (rec, f) in seen and seen[(rec, f)][1] != value and i in trained:
                    found.append({"record": rec, "field": f, "first_turn": seen[(rec, f)][0], "turn": i,
                                  "before": seen[(rec, f)][1][:100], "after": value[:100]})
                seen[(rec, f)] = (i, value)
    return found


def main(*paths):
    for path in paths:
        n = flagged = 0
        for line in open(path):
            row = json.loads(line)
            n += 1
            hits = overwrites(row)
            if hits:
                flagged += 1
                for h in hits:
                    print(f"{row['id']} | {h['record'][0]} {dict(h['record'][1])} {h['field']} | turn {h['first_turn']} -> "
                          f"{h['turn']} | {h['before']} -> {h['after']}")
        print(json.dumps({"file": path, "rows": n, "flagged": flagged}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
