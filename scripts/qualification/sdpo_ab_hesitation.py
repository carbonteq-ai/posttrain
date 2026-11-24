"""D36 behaviour check on chains files: reconsidering phrases per reply, reply-limit hits and reply length.

Arm A r2 learned to second-guess in free text ("Wait", "Actually, let me re-check"): 0.13 -> 1.38 phrases per
transfer reply and 0 -> 31% of episodes cut by the reply limit. This prints the same numbers for any chains files.

Usage: sdpo_ab_hesitation.py NAME=<chains.jsonl.gz> [NAME=<chains.jsonl.gz> ...]
"""
import gzip
import json
import re
import sys

HESITATION = re.compile(r"\b(Wait|Actually|Let me re-?(?:check|read|consider|verify)"
                        r"|I need to (?:re-?check|reconsider|verify))\b")


def stats(path):
    replies, episodes, cut = [], 0, 0
    for line in gzip.open(path, "rt"):
        chain = json.loads(line)
        actions = [s for s in chain["segments"] if s["kind"] == "action"]
        episodes += 1
        cut += any(a.get("limit_reason") == "reply_limit" for a in actions)
        replies += [a.get("raw") or "" for a in actions]
    lengths = sorted(len(r) for r in replies)
    return {"episodes": episodes, "replies": len(replies),
            "hesitation_per_reply": round(sum(len(HESITATION.findall(r)) for r in replies) / len(replies), 3),
            "reply_limit_episode_share": round(cut / episodes, 4),
            "median_reply_chars": lengths[len(lengths) // 2], "p90_reply_chars": lengths[int(0.9 * len(lengths))]}


if __name__ == "__main__":
    print(json.dumps({name: stats(path) for name, path in (a.split("=", 1) for a in sys.argv[1:])}, indent=1))
