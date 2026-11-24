"""Rank candidate pairs by length-normalised CHES under the SFT model and pick pairs within the caps.

CHES (Razin et al., ICLR 2025, arXiv 2410.08847; code princeton-nlp/unintentional-unalignment ches/ches.py): with
h the last-layer hidden state at each position that predicts a response token, m+ and m- the mean of those states
over the chosen and the rejected response, CHES = m+ . m- - ||m+||^2. Higher = more similar pair = more likelihood
displacement. Our adaptation (phase 0, recorded): the mean runs over the positions predicting trained tokens only
(P9's own reasoning, text and calls; tool results carry no loss), not over the padded length.

  score  (GPU, P9 = Qwen3.5-4B + the e3 adapter merged, bf16): one forward per distinct side of every candidate
         pair in a token-id pair file (exact_tokens.py output); writes OUT.tensors (mean vectors, float32) + OUT.json
  select (CPU): CHES per candidate; per task the same caps as build_pairs.py (<= 4 pairs, each attempt in <= 2, no
         pair twice), every failure class of the task's candidates covered first, then lowest CHES; held-out split
         kept; writes the training pair file and a report
usage: ches_rank.py score CANDIDATES_TOK.jsonl BASE_MODEL_DIR ADAPTER_DIR OUT_PREFIX
       ches_rank.py select CANDIDATES_TOK.jsonl OUT_PREFIX PAIRS_OUT.jsonl REPORT_OUT.json
"""
import hashlib
import json
import sys
from collections import defaultdict

from build_pairs import CAP_ATTEMPT, CAP_TASK, CLASS_ORDER


def side_key(prompt: list[int], ids: list[int], mask: list[int]) -> str:
    return hashlib.sha256(json.dumps([prompt, ids, mask]).encode()).hexdigest()


def sides(row: dict):
    t = row["metadata"]["token_ids"]
    for side in ("chosen", "rejected"):
        yield side, side_key(t["prompt"], t[side], t[f"{side}_mask"]), t["prompt"], t[side], t[f"{side}_mask"]


def predicting_positions(prompt_len: int, mask: list[int]) -> list[int]:
    """Positions (in prompt + side) whose hidden state predicts a trained side token: token j is predicted at j-1."""
    return [prompt_len + j - 1 for j, m in enumerate(mask) if m]


def mean_hidden(model, prompt, ids, mask, device):
    import torch

    seq = torch.tensor([list(prompt) + list(ids)], device=device)
    with torch.no_grad():
        hidden = model.model(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False).last_hidden_state[0]
    pos = torch.tensor(predicting_positions(len(prompt), mask), device=device)
    return hidden.index_select(0, pos).float().mean(dim=0).cpu()


def ches(m_plus, m_minus) -> float:
    return float(m_plus @ m_minus - m_plus @ m_plus)


def require_attached(peft_model, adapter_dir) -> None:
    """Every tensor of adapter_model.safetensors is in the model with equal values (as the framework's
    require_loaded_adapter): an adapter that silently failed to attach would score the base model."""
    import torch
    from safetensors.torch import load_file

    adapter, params = peft_model.active_adapter, dict(peft_model.named_parameters())
    for key, saved in load_file(f"{adapter_dir}/adapter_model.safetensors").items():
        stem, _, leaf = key.rpartition(".")
        param = params.get(f"{stem}.{adapter}.{leaf}", params.get(key))
        if param is None or not torch.equal(param.detach().cpu().to(saved.dtype), saved):
            raise ValueError(f"adapter {adapter_dir} did not attach: {key}")


def score(cand_path, base_dir, adapter_dir, out_prefix):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText  # the class the evaluation runner loads P9 with

    device = "cuda" if torch.cuda.is_available() else "cpu"  # cpu only in the tests
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=dtype, device_map={"": device},
                                                       attn_implementation="sdpa")
    adapted = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    require_attached(adapted, adapter_dir)
    model = adapted.merge_and_unload().eval()
    todo = {}
    for line in open(cand_path):
        for _, key, prompt, ids, mask in sides(json.loads(line)):
            todo.setdefault(key, (prompt, ids, mask))
    keys, vectors = sorted(todo), []
    for n, key in enumerate(keys):
        vectors.append(mean_hidden(model, *todo[key], device=device))
        if n % 50 == 0:
            print(json.dumps({"sides_done": n, "of": len(keys)}), flush=True)
    torch.save(torch.stack(vectors), out_prefix + ".tensors")  # not .pt: fetch.sh skips *.pt
    json.dump({"keys": keys, "base": base_dir, "adapter": adapter_dir}, open(out_prefix + ".json", "w"))
    print(json.dumps({"sides": len(keys)}))
    return 0


def pick(cands: list[dict]) -> list[dict]:
    """Within one task: cover each failure class first (the rejected attempt adding the most uncovered classes, as
    build_pairs; lowest CHES among equals), then fill by lowest CHES;
    each attempt in <= CAP_ATTEMPT pairs, <= CAP_TASK pairs, no pair twice."""
    uses, picked, covered = defaultdict(int), [], set()
    ordered = sorted(cands, key=lambda p: (p["metadata"]["ches"], p["id"]))

    def ok(p):
        m = p["metadata"]
        return (p not in picked and len(picked) < CAP_TASK
                and uses[m["chosen_id"]] < CAP_ATTEMPT and uses[m["rejected_id"]] < CAP_ATTEMPT)

    def take(p):
        m = p["metadata"]
        picked.append(p)
        uses[m["chosen_id"]] += 1
        uses[m["rejected_id"]] += 1
        covered.update(m["rejected_failure_classes"])

    for cls in CLASS_ORDER:  # as build_pairs: the attempt covering the most uncovered classes; CHES breaks ties
        if cls in covered:
            continue
        options = [p for p in ordered if cls in p["metadata"]["rejected_failure_classes"] and ok(p)]
        if options:
            take(max(options, key=lambda p: (len(set(p["metadata"]["rejected_failure_classes"]) - covered),
                                             -p["metadata"]["ches"])))
    for p in ordered:
        if ok(p):
            take(p)
    return picked


def select(cand_path, prefix, out_path, report_path):
    import statistics

    import torch

    vectors = torch.load(prefix + ".tensors")
    index = {k: i for i, k in enumerate(json.load(open(prefix + ".json"))["keys"])}
    rows = [json.loads(line) for line in open(cand_path)]
    by_task = defaultdict(list)
    for r in rows:
        (_, kc, *_), (_, kr, *_) = sides(r)
        r["metadata"]["ches"] = ches(vectors[index[kc]], vectors[index[kr]])
        by_task[r["metadata"]["task"]].append(r)
    chosen = [p for task in sorted(by_task) for p in pick(by_task[task])]
    with open(out_path, "w") as w:
        for p in chosen:
            w.write(json.dumps(p, ensure_ascii=False) + "\n")
    classes = lambda ps: set().union(*[set(p["metadata"]["rejected_failure_classes"]) for p in ps])
    uncovered = [t for t, ps in by_task.items() if classes(ps) - classes([p for p in chosen if p["metadata"]["task"] == t])]
    report = {"candidates": len(rows), "pairs": len(chosen), "tasks": len(by_task),
              "by_type": {k: sum(p["metadata"]["type"] == k for p in chosen) for k in ("A", "B")},
              "by_split": {k: sum(p["metadata"]["split"] == k for p in chosen) for k in ("train", "heldout")},
              "tasks_with_a_class_uncovered": uncovered,
              "ches_selected_median": statistics.median(p["metadata"]["ches"] for p in chosen),
              "ches_all_median": statistics.median(r["metadata"]["ches"] for r in rows),
              "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest()}
    json.dump(report, open(report_path, "w"), indent=1)
    print(json.dumps(report, indent=1))
    return 0 if not uncovered else 1


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    sys.exit(score(*args) if mode == "score" else select(*args))
