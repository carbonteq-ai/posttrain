"""Delete-only edits of a chosen success's final reply (the SFT data edit rules).

Each edit (hand-written after reading the reply; edits files map attempt id -> {"lines": [...], "fragments": [[line,
text], ...]} or {"drop": reason}) removes whole lines and/or exact fragments of a line of the final assistant message's
visible text. The reasoning, every call and every tool result stay byte-identical. Refused, as in SFT's
apply_reply_edits.py (13 Nov): a fragment that does not occur exactly once in its line, and any result that is not a
pure deletion of the original (difflib opcodes other than equal/delete). Every deleted text is logged in
metadata.edits; runs of blank lines left behind collapse to one and the reply is stripped (whitespace only).
usage: reply_edits.py MANIFEST.json SUCC_AS_RECORDED.jsonl SUCC_MASKED.jsonl OUT.jsonl REVIEW.md EDITS.json [EDITS.json...]
"""
import difflib
import json
import re
import sys


def delete_only(a: str, b: str) -> bool:
    return {op for op, *_ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()} <= {"equal", "delete"}


def edit_reply(reply: str, edit: dict) -> tuple[str, list[str]]:
    lines = reply.split("\n")
    deleted = []
    for ln, frag in edit.get("fragments", []):
        if lines[ln].count(frag) != 1:
            raise ValueError(f"fragment occurs {lines[ln].count(frag)} times in line {ln}: {frag!r}")
        lines[ln] = lines[ln].replace(frag, "")
        deleted.append(frag)
    for ln in sorted(edit.get("lines", []), reverse=True):
        deleted.append(lines[ln])
        lines[ln] = None
    new = re.sub(r"\n{3,}", "\n\n", "\n".join(x for x in lines if x is not None)).strip()
    if not delete_only(reply, new):
        raise ValueError("edit is not a pure deletion")
    if not new:
        raise ValueError("edit empties the reply")
    return new, deleted


def main(manifest_path, recorded, masked, out, review, *edit_files):
    manifest = json.load(open(manifest_path))
    rows = {}
    for path, form in ((recorded, "as recorded"), (masked, "failed-call turns masked")):
        for line in open(path):
            r = json.loads(line)
            rows[(r["id"], form)] = r
    edits = {}
    for f in edit_files:
        for k, v in json.load(open(f)).items():
            assert k not in edits, f"edit given twice: {k}"
            edits[k] = v
    n = 0
    with open(out, "w") as w, open(review, "w") as rv:
        rv.write("# Gate 1b: delete-only edits of P9 final replies (every deleted text)\n\n")
        for rid, edit in sorted(edits.items()):
            if "drop" in edit:
                rv.write(f"## {rid}: DROPPED ({edit['drop']})\n\n")
                continue
            row = json.loads(json.dumps(rows[(rid, manifest[rid]["form"])]))
            final = row["messages"][-1]
            assert final["role"] == "assistant" and not final.get("tool_calls"), rid
            new, deleted = edit_reply(final["content"], edit)
            final["content"] = new
            row["metadata"]["edits"] = row["metadata"].get("edits", []) + [{
                "kind": "final reply: delete-only (gate 1b, 18 Nov)", "message": len(row["messages"]) - 1,
                "deleted": deleted}]
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            rv.write(f"## {rid}\n\n" + "\n".join(f"- deleted: `{d}`" for d in deleted) + "\n\n")
    print(json.dumps({"edited": n, "dropped": sum("drop" in e for e in edits.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
