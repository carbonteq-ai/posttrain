"""Every pair's token ids and loss masks; the SFT model's sides use its own sampled tokens, not re-tokenised text.

Per pair (written to metadata.token_ids = {prompt, chosen, chosen_mask, rejected, rejected_mask}):
  prompt       the rejected attempt's chain up to its first action (system, tools, task and the generation prompt
               "<|im_start|>assistant\\n<think>\\n"), identical to the chosen chain's for type A
  P9 side      the rest of its chain: context segments (tool results, the next generation prompt with its forced
               "<think>\\n") mask 0; action segments (P9's sampled reasoning, text, calls, <|im_end|>) mask 1, except
               the chosen side's untrained turns (metadata.chosen_untrained)
  edited reply a chosen P9 side whose final reply was edited in gate 1b: the chain up to the final action, then the
               final assistant span as the framework renders the edited message (it was never sampled)
  type B chosen the framework's own rendering (the SFT row), with pair_parity.intended_mask (opener of later turns and
               untrained turns at 0)
Checks, every side (refuse the pair on any failure): the ids decode to exactly the text the framework renders for
the same messages (a trailing untrained "\\n" after the last <|im_end|> aside); every trained span decodes to text
inside the rendered side's assistant spans, in order (so no prompt, tool-result or opener token is trained); the
prompt plus the longer side fits in 32,768 tokens. The training job repeats the text and span checks before using
the ids (posttrain.train.rendering.apply_token_overrides).
usage (wt-sft .venv python with wt-dpo packages first on PYTHONPATH, CPU, under capped):
  exact_tokens.py PAIRS.jsonl CHAINS.jsonl.gz TOKENIZER_DIR OUT_PAIRS.jsonl OUT_REPORT.json
"""
import json
import sys
from dataclasses import replace

from pair_parity import MAX_TOKENS, intended_mask, load_chains, spans


def chain_parts(chain: dict) -> tuple[list[int], list[dict]]:
    """(prompt ids, the segments after the prompt); context and action segments alternate."""
    segs = chain["segments"]
    first = next(i for i, s in enumerate(segs) if s["kind"] == "action")
    prompt = [t for s in segs[:first] for t in s["ids"]]
    return prompt, segs[first:]


def side_from_chain(segments: list[dict], untrained: set[int]) -> tuple[list[int], list[int]]:
    ids, mask, k = [], [], 0
    for seg in segments:
        trained = seg["kind"] == "action" and k not in untrained
        ids += seg["ids"]
        mask += [int(trained)] * len(seg["ids"])
        k += seg["kind"] == "action"
    return ids, mask


def trained_text_inside(decode, ids, mask, rendered_ids, rendered_mask) -> bool:
    """Every trained span of (ids, mask) decodes to text found, in order, inside the rendered assistant spans."""
    allowed = [decode(rendered_ids[a:b]) for a, b in spans(rendered_mask)]
    k, pos = 0, 0
    for a, b in spans(mask):
        text = decode(ids[a:b])
        while k < len(allowed):
            at = allowed[k].find(text, pos)
            if at >= 0:
                pos = at + len(text)
                break
            k, pos = k + 1, 0
        else:
            return False
    return True


def text_equal(decode, ids, rendered_ids) -> bool:
    a, b = decode(ids), decode(rendered_ids)
    return a == b or b == a + "\n"


def convert(tok, sample, row, chains, opener):
    """(token_ids dict, report dict) for one rendered pair; raises ValueError on any failed check."""
    decode = lambda ids: tok.decode(list(ids), skip_special_tokens=False)
    meta = row["metadata"]
    ordinal = lambda msgs, idx: sum(1 for x in msgs[:idx] if x["role"] == "assistant")
    unt = {ordinal(row["chosen"], i) for i in meta["chosen_untrained"]}
    edited = any(str(e.get("kind", "")).startswith("final reply: delete-only (gate 1b")
                 for e in (meta.get("chosen_form_metadata") or {}).get("edits") or [])
    rid = meta["rejected_id"].rsplit("/", 1)[-1]
    prompt, rej_segments = chain_parts(chains[rid])
    if decode(prompt) != decode(sample.prompt_ids):
        raise ValueError("chain prompt text differs from the rendered prompt")
    rejected, rejected_mask = side_from_chain(rej_segments, set())
    report = {"prompt_tokens_equal": prompt == list(sample.prompt_ids)}
    if meta["type"] == "A":
        cid = meta["chosen_id"].rsplit("/", 1)[-1]
        c_prompt, c_segments = chain_parts(chains[cid])
        if c_prompt != prompt:
            raise ValueError("type A chosen and rejected chains start with different prompt tokens")
        chosen, chosen_mask = side_from_chain(c_segments, unt)
        if edited:  # replace the final action with the rendered (edited) final assistant span
            last = max(i for i, s in enumerate(c_segments) if s["kind"] == "action")
            keep = sum(len(s["ids"]) for s in c_segments[:last])
            r_mask = intended_mask(list(sample.chosen_mask), unt, list(sample.chosen_ids), opener)
            a, b = spans(r_mask)[-1]
            chosen = chosen[:keep] + list(sample.chosen_ids[a:b])
            chosen_mask = chosen_mask[:keep] + [1] * (b - a)
        report["chosen_source"] = "chain+rendered final" if edited else "chain"
    else:
        chosen = list(sample.chosen_ids)
        chosen_mask = intended_mask(list(sample.chosen_mask), unt, chosen, opener)
        report["chosen_source"] = "rendered (type B)"
    for side, ids, mask, r_ids, r_mask in (("chosen", chosen, chosen_mask, sample.chosen_ids, sample.chosen_mask),
                                           ("rejected", rejected, rejected_mask, sample.rejected_ids,
                                            sample.rejected_mask)):
        if not text_equal(decode, prompt + ids, list(sample.prompt_ids) + list(r_ids)):
            raise ValueError(f"{side}: token ids do not decode to the rendered text")
        if not trained_text_inside(decode, ids, mask, list(r_ids), list(r_mask)):
            raise ValueError(f"{side}: a trained span is not inside the rendered assistant text")
        if not any(mask):
            raise ValueError(f"{side}: no trained token")
        report[f"{side}_tokens"] = len(prompt) + len(ids)
        report[f"{side}_trained"] = sum(mask)
        report[f"{side}_ids_equal_rendered"] = ids == list(r_ids)
    if len(prompt) + max(len(chosen), len(rejected)) > MAX_TOKENS:
        raise ValueError("longer than the 32,768-token limit")
    return {"prompt": prompt, "chosen": chosen, "chosen_mask": chosen_mask,
            "rejected": rejected, "rejected_mask": rejected_mask}, report


def main(pairs_path, chains_path, tok_dir, out_path, report_path):
    from transformers import AutoTokenizer
    from posttrain.common.artifacts import HubModelRef
    from posttrain.common.variants.qwen35 import QWEN_35_2B, QWEN35_THINKING_RENDERER_CONTRACT
    from posttrain.data import preferences_from_huggingface
    from posttrain.train import QWEN35_THINKING_RENDERER, render_preferences

    ref = HubModelRef(repo_id="Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    model = replace(QWEN_35_2B, id="qwen3.5-4b", artifact=ref, base=ref, parameters=4_000_000_000,
                    renderer=QWEN35_THINKING_RENDERER_CONTRACT)
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    opener = tok.encode("<think>\n", add_special_tokens=False)
    rows = [json.loads(line) for line in open(pairs_path)]
    by_id = {r["id"]: r for r in rows}
    episodes = {r["metadata"][k].rsplit("/", 1)[-1] for r in rows for k in ("chosen_id", "rejected_id")
                if r["metadata"][k].startswith("dpo-w6/")}
    chains = load_chains(chains_path, episodes)
    dataset = preferences_from_huggingface(rows, dataset_id="dpo-pairs-phase3", revision="1", format="trl")
    samples = render_preferences(tok, model, dataset, QWEN35_THINKING_RENDERER, max_length=1 << 20)
    reports, failures = {}, {}
    with open(out_path, "w") as w:
        for s in samples:
            row = json.loads(json.dumps(by_id[s.id]))
            try:
                token_ids, rep = convert(tok, s, row, chains, opener)
            except ValueError as error:
                failures[s.id] = str(error)
                continue
            row["metadata"]["token_ids"] = token_ids
            reports[s.id] = rep
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    count = lambda key, value: sum(r.get(key) == value for r in reports.values())
    summary = {"pairs": len(samples), "converted": len(reports), "failed": len(failures),
               "prompt_tokens_equal_rendered": count("prompt_tokens_equal", True),
               "chosen_from_chain": count("chosen_source", "chain"),
               "chosen_chain_plus_rendered_final": count("chosen_source", "chain+rendered final"),
               "chosen_rendered_type_b": count("chosen_source", "rendered (type B)"),
               "rejected_ids_differ_from_rendered": count("rejected_ids_equal_rendered", False),
               "chosen_ids_differ_from_rendered": count("chosen_ids_equal_rendered", False),
               "max_tokens": max(max(r["chosen_tokens"], r["rejected_tokens"]) for r in reports.values())}
    json.dump({"summary": summary, "failures": failures, "pairs": reports}, open(report_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
