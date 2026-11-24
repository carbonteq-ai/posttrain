"""The SFT masked-turn rule for successes that did a wrong action and corrected it later.

A masked turn stays in the conversation (the student sees it) but carries no loss: it is removed from
trainable_message_indices and listed in metadata.masked_turns with the reason "wrong action, corrected later", the
reason validate_row.py's zero_tool_errors gate accepts (13 Nov v3 review). Masks already on a row (failed calls) are
kept. Only assistant turns before the final reply can be masked. Nothing else in the row changes.
usage: mask_turns.py MASKS.json OUT.jsonl ROWS.jsonl [ROWS.jsonl ...]   (MASKS: attempt id -> [message indices])
"""
import json
import sys

REASON = "wrong action, corrected later"


def mask(row: dict, turns: list[int]) -> dict:
    row = json.loads(json.dumps(row))
    msgs = row["messages"]
    final = len(msgs) - 1
    for t in turns:
        if not (0 <= t < final and msgs[t]["role"] == "assistant"):
            raise ValueError(f"message {t} is not an assistant turn before the final reply")
    meta = row["metadata"]
    masked = set(meta.get("masked_turns") or []) | set(turns)
    reasons = {str(k): v for k, v in (meta.get("masked_reasons") or {}).items()}
    for t in turns:
        reasons[str(t)] = REASON
    meta["masked_turns"] = sorted(masked)
    meta["masked_reasons"] = reasons
    row["trainable_message_indices"] = [i for i in row["trainable_message_indices"] if i not in masked]
    return row


def main(masks_path, out, *row_files):
    masks = {k: v for k, v in json.load(open(masks_path)).items() if not k.startswith("_")}
    rows = {}
    for f in row_files:  # later files win (a masked-for-failed-calls form replaces the as-recorded one)
        for line in open(f):
            r = json.loads(line)
            rows[r["id"]] = r
    with open(out, "w") as w:
        for rid, turns in sorted(masks.items()):
            w.write(json.dumps(mask(rows[rid], turns), ensure_ascii=False) + "\n")
    print(json.dumps({"masked_rows": len(masks)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
