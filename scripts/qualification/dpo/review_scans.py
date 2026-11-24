"""The SFT review scans on candidate chosen attempts.

The SFT data (v3 report, 13 Nov) was reviewed beyond the gates with these scans; each flag is printed for reading and
written to OUT.json. Only TRAINED assistant turns are scanned (masked turns carry no loss).
  years        a year (20xx) in a trained turn that the system prompt, request and tool results do not contain
               (accept_v3.py rule; it caught the invented 2025-01-15 date)
  provenance   every email, phone number, identifier-like token (letters with _ or - and digits) and URL inside a
               trained call's arguments must appear earlier in the conversation (system incl. tools, request, tool
               results, the agent's own earlier call arguments); provenance_scan.py rule (it caught an invented address)
  assumptions  phrases of assuming or inventing values ("assume", "placeholder", "made up", "make up", "invent",
               "hypothetical", "dummy", "example.com" when the context lacks it)
  corrections  admitted mistakes and correction messages: reasoning that says an earlier action was wrong, and
               outgoing messages that correct or retract an earlier one ("correction", "corrected", "disregard",
               "apolog", "sent in error", "mistake")
usage: python3 -I review_scans.py ROWS.jsonl OUT.json"""
import json
import re
import sys

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b")
URL = re.compile(r"https?://\S+")
ASSUME = re.compile(r"\bassum\w*|\bplaceholder\b|\bmade[- ]up\b|\bmake up\b|\binvent(?:s|ed|ing|ion|ions)?\b|\bhypothetical\b|\bdummy\b", re.I)
MISTAKE = re.compile(r"\b(I|we) (made a mistake|made an error|made a wrong|was wrong|were wrong|mistakenly|accidentally|"
                     r"incorrectly|wrongly|erroneously)\b|\b(I|we) (?:initially |already |previously )?sent (?:her|him|them|"
                     r"a|an|the)[^.]{0,60}\b(?:wrong|incorrect)\b|\b(?:was|were) (?:wrong|incorrect|an error|a mistake)\b|"
                     r"\b(sent|posted|created|updated|deleted) [^.]{0,40}\b(by mistake|in error|incorrectly|wrongly)\b|"
                     r"\b(that|this) was (a mistake|wrong|an error)\b|\boops\b", re.I)
CORRECTION = re.compile(r"\bcorrect(ion|ed)\b|\bdisregard\b|\bapolog\w*|\bsent in error\b|\bmistake\b|\bretract\w*",
                        re.I)


def txt(c):
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)


def scan(row):
    m = row["messages"]
    trained = set(row["trainable_message_indices"])
    ctx = txt(m[0]["content"]) + "\n" + json.dumps(row.get("tools") or [], ensure_ascii=False) + "\n" + txt(m[1]["content"])
    world = "".join(txt(x.get("content")) for x in m if x["role"] in ("system", "user", "tool"))
    flags = []
    for i, x in enumerate(m[2:], 2):
        if x["role"] == "tool":
            ctx += "\n" + txt(x["content"])
            continue
        if x["role"] != "assistant":
            continue
        args = [c["function"]["arguments"] for c in x.get("tool_calls") or []]
        if i in trained:
            reasoning = x.get("reasoning_content") or ""
            own = reasoning + txt(x.get("content")) + json.dumps(args, ensure_ascii=False)
            for y in sorted(set(re.findall(r"\b20\d\d\b", own))):
                if y not in world:
                    flags.append({"scan": "years", "message": i, "value": y})
            for a in args:
                for s in strings(a):
                    for pat, kind in ((EMAIL, "email"), (PHONE, "phone"), (URL, "url"), (TOKEN, "token")):
                        for v in pat.findall(s):
                            v = v if isinstance(v, str) else v[0]
                            if v.strip() and v not in ctx:
                                flags.append({"scan": "provenance", "message": i, "kind": kind, "value": v[:120]})
            for mo in ASSUME.finditer(reasoning + "\n" + txt(x.get("content"))):
                src = reasoning + "\n" + txt(x.get("content"))
                flags.append({"scan": "assumptions", "message": i, "text": src[max(0, mo.start() - 120):mo.end() + 80]})
            for mo in MISTAKE.finditer(reasoning):
                flags.append({"scan": "corrections", "message": i, "where": "reasoning",
                              "text": reasoning[max(0, mo.start() - 120):mo.end() + 80]})
            for s in (s for a in args for s in strings(a)):
                for mo in CORRECTION.finditer(s):
                    if mo.group(0).lower() not in ctx.lower():
                        flags.append({"scan": "corrections", "message": i, "where": "message",
                                      "text": s[max(0, mo.start() - 120):mo.end() + 80]})
        ctx += "\n" + json.dumps(args, ensure_ascii=False)
    return flags


def main(rows_path, out_path):
    out = {}
    for line in open(rows_path):
        row = json.loads(line)
        out[row["id"]] = {"task": row["metadata"]["task"], "flags": scan(row)}
    json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
    counts = {}
    for r in out.values():
        for f in r["flags"]:
            counts[f["scan"]] = counts.get(f["scan"], 0) + 1
    print(json.dumps({"rows": len(out), "rows_flagged": sum(bool(r["flags"]) for r in out.values()), "flags": counts}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
