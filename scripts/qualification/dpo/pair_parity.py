"""Render every pair through the framework's own DPO path and compare it with what the 4B saw.

Path: pairs.jsonl -> posttrain.data.preferences_from_huggingface (format trl) -> posttrain.train.render_preferences
with QWEN35_THINKING_RENDERER and the real Qwen3.5-4B tokenizer (the exact rendering the DPO job uses). Per side:
  P9 side      prompt_ids + side_ids against P9's recorded token chain (W6 chains.jsonl.gz; context segments = prompt,
               tool results and generation prompts the runner rendered; action segments = P9's own sampled tokens):
               `tokens_equal` (ids identical) or, as SFT's accepted bar (decisions-20261112: renderer parity), the full
               decoded text identical with only non-canonical token splits differing; trained spans = the action
               segments in order (decoded text equal).
  edited reply a chosen side whose final reply was edited (gate 1b): everything before the final action identical as
               above; the final action's decoded text differs from P9's only by deletions (difflib equal/delete), and
               the deleted characters are exactly the logged deleted texts (whitespace aside).
  expert side  (type B) the same chosen messages rendered by the framework SFT path (render_supervised, same renderer)
               as data v4 was verified with: ids identical.
Masks (the intended DPO mask; phase 3 makes the renderer produce it): the rendered side's trained spans, minus the
spans of metadata.chosen_untrained turns, minus the "<think>\n" opener that the runner's generation prompt forces at the
start of every turn after the first (the chain records it as context; the framework renderer trains it today).
`mask_ok` (P9 sides) = every intended trained span decodes exactly to one of P9's kept action segments, in order, so no
prompt, tool-result, opener or untrained-turn token can be inside it (intended_mask() and the span check are tested
both ways in test_pair_parity.py). Rendered text = chain text plus the template's untrained final "\n".
Lengths: prompt + side tokens, <= 32,768 (the SFT limit, plan 7.3), and the chosen/rejected length ratio.
usage (wt-sft .venv python, CPU, under capped): pair_parity.py PAIRS.jsonl CHAINS.jsonl.gz EXPERT_ROWS.jsonl TOKENIZER_DIR OUT.json
"""
import difflib
import gzip
import json
import re
import sys
from dataclasses import replace

MAX_TOKENS = 32768


def spans(mask: list[int]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(list(mask) + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def intended_mask(side_mask: list[int], untrained_ordinals: set[int], ids: list[int] | None = None,
                  opener: list[int] | None = None) -> list[int]:
    """The rendered side mask with (1) the spans of untrained assistant turns (by assistant ordinal) zeroed and
    (2) the thinking opener ("<think>\\n") at the start of every turn after the first zeroed: the runner's generation
    prompt inserts it (the chain records it as context), so the 4B never sampled it; the first turn's opener is
    already inside the prompt."""
    out = list(side_mask)
    for k, (a, b) in enumerate(spans(side_mask)):
        if k in untrained_ordinals:
            out[a:b] = [0] * (b - a)
        elif k > 0 and ids is not None and opener and list(ids[a:a + len(opener)]) == opener:
            out[a:a + len(opener)] = [0] * len(opener)
    return out


def mask_ok(mask: list[int], allowed: list[tuple[int, int]]) -> bool:
    """True iff the set positions are exactly the union of the allowed spans."""
    want = [0] * len(mask)
    for a, b in allowed:
        want[a:b] = [1] * (b - a)
    return list(mask) == want


def deletion_matches(before: str, after: str, deleted: list[str]) -> bool:
    """`after` is `before` with deletions only (difflib equal/delete), and removing each logged deleted text once from
    `before` gives `after` (whitespace aside: whole-line deletions collapse blank lines)."""
    ops = difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
    if {op for op, *_ in ops} - {"equal", "delete"}:
        return False
    head, sep, reply = before.rpartition("</think>")  # the edits are in the visible reply, never the reasoning
    for d in sorted(deleted, key=len, reverse=True):
        if d not in reply:
            return False
        reply = reply.replace(d, "", 1)
    squash = lambda s: re.sub(r"\s+", "", s)
    return squash(head + sep + reply) == squash(after)


LITERAL = re.compile(r"(<parameter=[^>\n]+>\n)(null|true|false)(\n</parameter>)")
PY_LITERAL = {"null": "None", "true": "True", "false": "False"}


def python_literals(text: str) -> tuple[str, int]:
    """The chat template renders a parsed JSON literal parameter value in Python spelling (decisions-20261112 accepted
    true -> True; null -> None and false -> False are the same mechanism). Returns the text so normalized + count."""
    n = len(LITERAL.findall(text))
    return LITERAL.sub(lambda m: m.group(1) + PY_LITERAL[m.group(2)] + m.group(3), text), n


def load_chains(path, episodes):
    out = {}
    with gzip.open(path, "rt") as s:
        for line in s:
            c = json.loads(line)
            if c["episode_id"] in episodes:
                out[c["episode_id"]] = c
    return out


def check_p9_side(tok, prompt_ids, side_ids, side_mask, chain, untrained_ord, edit_deleted, opener):
    full = list(prompt_ids) + list(side_ids)
    want = [t for seg in chain["segments"] for t in seg["ids"]]
    actions = [seg for seg in chain["segments"] if seg["kind"] == "action"]
    dec = tok.decode
    mask = intended_mask(list(side_mask), set(untrained_ord), list(side_ids), opener)
    trained = [(a + len(prompt_ids), b + len(prompt_ids)) for a, b in spans(mask)]
    kept_actions = [a for k, a in enumerate(actions) if k not in untrained_ord]
    res = {"tokens_equal": full == want, "trained_spans": len(trained), "trained_actions": len(kept_actions)}
    a_text, b_text = dec(full), dec(want)
    if a_text not in (b_text, b_text + "\n"):  # fallback only: sampled_spelling.py normally restores P9's spelling
        normalized, literals = python_literals(b_text)
        if a_text in (normalized, normalized + "\n"):
            b_text = normalized
            res["json_literal_spellings"] = literals
    # the template closes the final message with "<|im_end|>\n"; generation stopped at <|im_end|>: one untrained "\n"
    res["final_newline_untrained"] = a_text == b_text + "\n" and not side_mask[-1]
    if edit_deleted is None:
        res["text_equal"] = a_text == b_text or res["final_newline_untrained"]
        res["spans_equal"] = len(trained) == len(kept_actions) and all(
            dec(full[a:b]) in (dec(act["ids"]), python_literals(dec(act["ids"]))[0])
            for (a, b), act in zip(trained, kept_actions))
    else:  # edited final reply: everything before the final action identical, the final action delete-only
        last_seg = max(i for i, seg in enumerate(chain["segments"]) if seg["kind"] == "action")
        chain_last_start = sum(len(seg["ids"]) for seg in chain["segments"][:last_seg])
        a0 = trained[-1][0]
        res["prefix_text_equal"] = dec(full[:a0]) in (dec(want[:chain_last_start]),
                                                      python_literals(dec(want[:chain_last_start]))[0])
        res["final_delete_only"] = deletion_matches(dec(actions[-1]["ids"]), dec(full[a0:trained[-1][1]]),
                                                    edit_deleted)
        res["text_equal"] = res["prefix_text_equal"] and res["final_delete_only"]
        res["spans_equal"] = len(trained) == len(kept_actions) and all(
            dec(full[a:b]) in (dec(act["ids"]), python_literals(dec(act["ids"]))[0])
            for (a, b), act in zip(trained[:-1], kept_actions[:-1]))
    res["mask_ok"] = res["spans_equal"] and sum(mask) > 0
    # token identity of the trained turns (reported, not a bar: SFT's accepted bar is text identity; a sampled
    # non-canonical split is re-tokenized canonically by the renderer)
    res["trained_turns_token_equal"] = sum(list(full[a:b]) == list(act["ids"]) for (a, b), act in zip(trained, kept_actions))
    res["pass"] = res["text_equal"] and res["spans_equal"] and res["mask_ok"]
    return res, mask


def main(pairs_path, chains_path, expert_path, tok_dir, out_path):
    from transformers import AutoTokenizer
    from posttrain.common.artifacts import HubModelRef
    from posttrain.common.variants.qwen35 import QWEN_35_2B, QWEN35_THINKING_RENDERER_CONTRACT
    from posttrain.data import preferences_from_huggingface, supervised_from_huggingface
    from posttrain.train import QWEN35_THINKING_RENDERER, render_preferences, render_supervised

    ref = HubModelRef(repo_id="Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    model = replace(QWEN_35_2B, id="qwen3.5-4b", artifact=ref, base=ref, parameters=4_000_000_000,
                    renderer=QWEN35_THINKING_RENDERER_CONTRACT)
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    opener = tok.encode("<think>\n", add_special_tokens=False)
    rows = [json.loads(line) for line in open(pairs_path)]
    meta = {r["id"]: r["metadata"] for r in rows}
    dataset = preferences_from_huggingface(rows, dataset_id="dpo-pairs-phase2", revision="1", format="trl")
    samples = render_preferences(tok, model, dataset, QWEN35_THINKING_RENDERER, max_length=1 << 20)
    rowsrc = {}
    for p in ("chosen_id", "rejected_id"):
        for r in rows:
            rowsrc[r["metadata"][p]] = None
    episodes = {i.rsplit("/", 1)[-1] for i in rowsrc if i.startswith("dpo-w6/")}
    chains = load_chains(chains_path, episodes)
    expert = {json.loads(l)["id"]: json.loads(l) for l in open(expert_path)}
    expert_render = {}
    needed = {m["chosen_id"] for m in meta.values() if m["type"] == "B"}
    if needed:
        sup = supervised_from_huggingface([expert[i] for i in needed], dataset_id="dpo-expert-check", revision="1",
                                          format="messages")
        for s in render_supervised(tok, model, sup, QWEN35_THINKING_RENDERER, max_length=1 << 20):
            expert_render[s.id] = list(s.input_ids)
    report = []
    for s in samples:
        m = meta[s.id]
        assistant_ord = lambda msgs, idx: sum(1 for x in msgs[:idx] if x["role"] == "assistant")
        row = next(r for r in rows if r["id"] == s.id)
        unt = {assistant_ord(row["chosen"], i) for i in m["chosen_untrained"]}
        edits = [e for e in (m.get("chosen_form_metadata") or {}).get("edits") or []
                 if str(e.get("kind", "")).startswith("final reply: delete-only (gate 1b")]  # P9 sides edited in gate 1b
        deleted = [d for e in edits for d in e["deleted"]] if edits else None
        entry = {"id": s.id, "task": m["task"], "type": m["type"], "split": m["split"],
                 "prompt_tokens": len(s.prompt_ids), "chosen_tokens": len(s.prompt_ids) + len(s.chosen_ids),
                 "rejected_tokens": len(s.prompt_ids) + len(s.rejected_ids)}
        entry["length_ok"] = max(entry["chosen_tokens"], entry["rejected_tokens"]) <= MAX_TOKENS
        entry["length_ratio"] = round(len(s.chosen_ids) / len(s.rejected_ids), 3)
        if m["type"] == "A":
            entry["chosen"], _ = check_p9_side(tok, s.prompt_ids, s.chosen_ids, s.chosen_mask,
                                               chains[m["chosen_id"].rsplit("/", 1)[-1]], unt, deleted, opener)
        else:
            want = expert_render[m["chosen_id"]]
            got = list(s.prompt_ids) + list(s.chosen_ids)
            ok_ids = got == want or (got[:-1] == want and tok.decode(got[-1:]) == "\n" and not s.chosen_mask[-1])
            mask = intended_mask(list(s.chosen_mask), unt, list(s.chosen_ids), opener)
            allowed = spans(mask)
            entry["chosen"] = {"tokens_equal_sft_render": ok_ids, "mask_ok": mask_ok(mask, allowed) and sum(mask) > 0,
                               "pass": ok_ids and sum(mask) > 0}
        entry["rejected"], _ = check_p9_side(tok, s.prompt_ids, s.rejected_ids, s.rejected_mask,
                                             chains[m["rejected_id"].rsplit("/", 1)[-1]], set(), None, opener)
        entry["pass"] = entry["chosen"]["pass"] and entry["rejected"]["pass"] and entry["length_ok"]
        report.append(entry)
    summary = {"pairs": len(report), "pass": sum(e["pass"] for e in report),
               "length_over": sum(not e["length_ok"] for e in report),
               "tokens_equal_sides": sum(e[s].get("tokens_equal", e[s].get("tokens_equal_sft_render", False))
                                         for e in report for s in ("chosen", "rejected")),
               "edited_final_delete_only": sum(e["chosen"].get("final_delete_only", False) for e in report),
               "json_literal_spellings": sum(e[s].get("json_literal_spellings", 0) for e in report
                                             for s in ("chosen", "rejected")),
               "sides_with_literal_spelling": sum(bool(e[s].get("json_literal_spellings")) for e in report
                                                  for s in ("chosen", "rejected")),
               "max_tokens": max(max(e["chosen_tokens"], e["rejected_tokens"]) for e in report),
               "p9_trained_turns": sum(e[s].get("trained_actions", 0) for e in report for s in ("chosen", "rejected")),
               "p9_trained_turns_token_equal": sum(e[s].get("trained_turns_token_equal", 0) for e in report
                                                   for s in ("chosen", "rejected"))}
    json.dump({"summary": summary, "pairs": report}, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0 if summary["pass"] == summary["pairs"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
