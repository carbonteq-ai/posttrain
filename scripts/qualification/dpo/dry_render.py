"""Dry render: the pair file through the training job's own data path, on CPU.

pairs-tok.jsonl -> preferences_from_huggingface (trl) -> render_preferences -> apply_token_overrides (the job's
steps in posttrain.train.backends.trl.dpo.run_dpo), then: every pair's ids and masks equal the converter's
metadata.token_ids; TRL's collator plus the framework's masked_completion put loss exactly on those masks (checked
on every pair, one pair per batch, chosen row then rejected row); lengths within the limit; lengths per type.
usage (wt-sft .venv python, wt-dpo packages first on PYTHONPATH, CPU, under capped):
  dry_render.py PAIRS_TOK.jsonl TOKENIZER_DIR OUT.json
"""
import json
import statistics
import sys
from dataclasses import replace

MAX_LENGTH = 32768


def main(pairs_path, tok_dir, out_path):
    from transformers import AutoTokenizer
    from trl.trainer.dpo_trainer import DataCollatorForPreference
    from posttrain.common.artifacts import HubModelRef
    from posttrain.common.variants.qwen35 import QWEN_35_2B, QWEN35_THINKING_RENDERER_CONTRACT
    from posttrain.data import preferences_from_huggingface
    from posttrain.train import QWEN35_THINKING_RENDERER, apply_token_overrides, render_preferences
    from posttrain.train.backends.trl.dpo import masked_completion

    ref = HubModelRef(repo_id="Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    model = replace(QWEN_35_2B, id="qwen3.5-4b", artifact=ref, base=ref, parameters=4_000_000_000,
                    renderer=QWEN35_THINKING_RENDERER_CONTRACT)
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    rows = [json.loads(line) for line in open(pairs_path)]
    want = {r["id"]: r["metadata"]["token_ids"] for r in rows}
    kind = {r["id"]: r["metadata"]["type"] for r in rows}
    dataset = preferences_from_huggingface(rows, dataset_id="dpo-pairs-dry", revision="1", format="trl")
    samples = render_preferences(tok, model, dataset, QWEN35_THINKING_RENDERER, max_length=MAX_LENGTH)
    samples = apply_token_overrides(tok, samples, dataset, max_length=MAX_LENGTH)
    collator = DataCollatorForPreference(pad_token_id=tok.pad_token_id or tok.eos_token_id, max_length=MAX_LENGTH,
                                         truncation_mode="keep_start")
    bad, lengths = {}, {"A": [], "B": []}
    for s in samples:
        w = want[s.id]
        same = (list(s.prompt_ids) == w["prompt"] and list(s.chosen_ids) == w["chosen"]
                and list(s.rejected_ids) == w["rejected"] and list(s.chosen_mask) == w["chosen_mask"]
                and list(s.rejected_mask) == w["rejected_mask"])
        example = {"prompt_ids": list(s.prompt_ids), "chosen_ids": list(s.chosen_ids),
                   "rejected_ids": list(s.rejected_ids), "chosen_mask": list(s.chosen_mask),
                   "rejected_mask": list(s.rejected_mask)}
        batch = collator.torch_call([example])
        mask = masked_completion(batch["completion_mask"], [example], MAX_LENGTH, "keep_start")
        p = len(s.prompt_ids)
        rows_ok = all(
            mask[row, : p + len(side)].tolist() == [0] * p + list(side_mask)
            and int(mask[row, p + len(side):].sum()) == 0
            and batch["input_ids"][row, : p + len(side)].tolist() == list(s.prompt_ids) + list(side)
            for row, side, side_mask in ((0, s.chosen_ids, s.chosen_mask), (1, s.rejected_ids, s.rejected_mask)))
        if not (same and rows_ok):
            bad[s.id] = {"ids_and_masks_equal_converter": same, "collated_rows_ok": rows_ok}
        lengths[kind[s.id]].append((p + len(s.chosen_ids), p + len(s.rejected_ids), sum(s.chosen_mask),
                                    sum(s.rejected_mask)))
    table = {k: {"pairs": len(v),
                 "chosen_tokens_median": statistics.median(x[0] for x in v),
                 "rejected_tokens_median": statistics.median(x[1] for x in v),
                 "chosen_trained_median": statistics.median(x[2] for x in v),
                 "rejected_trained_median": statistics.median(x[3] for x in v),
                 "max_tokens": max(max(x[0], x[1]) for x in v)} for k, v in lengths.items() if v}
    out = {"pairs": len(samples), "failed": len(bad), "failures": bad, "lengths": table}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
