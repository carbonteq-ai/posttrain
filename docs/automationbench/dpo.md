# AutomationBench: DPO

DPO of the Qwen3.5-4B SFT adapter on preference pairs built from its own attempts on the training tasks. This branch
builds on `feat/automationbench-sft`; evaluation, the environment and the SFT data are described in
[README.md](README.md).

| Adapter | test120 | guard10 |
| --- | --- | --- |
| [carbonteq/qwen3.5-4b-automationbench-sft](https://huggingface.co/carbonteq/qwen3.5-4b-automationbench-sft) (start and reference) | 42.4% | 78.8% |
| [carbonteq/qwen3.5-4b-automationbench-dpo](https://huggingface.co/carbonteq/qwen3.5-4b-automationbench-dpo) | 43.3% | 77.5% |

DPO added 0.9 points on test120. The pairs taught habits that every preferred side shared, such as shorter final
replies and longer reasoning, rather than the decisions that fail tasks. This line of work was closed after this run.

## Layout

| Path | What it is |
| --- | --- |
| `scripts/qualification/dpo/` | pair pipeline: pools, pairs, exact tokens, checks, ranking (one test file per module) |
| `packages/train/src/posttrain/train/backends/trl/chunked_dpo.py` | DPO loss computed in chunks of 256 tokens over the exact sampled tokens, with no full-vocabulary logits |
| `packages/train`, `packages/jobs` | DPO loads the same image-text model class as SFT; an adapter whose weights do not all load is refused; a downloaded adapter must match its recorded digest |
| `apps/lab/.posttrain/catalog/qwen35-4b-automationbench-dpo-p9.yaml` | the start adapter, the pairs, the DPO settings and the LoRA binding |
| `apps/lab/.posttrain/work_packages/qwen35_4b_automationbench_dpo_p9_e3.yaml` | the DPO job |

## Build the pairs

The inputs are the SFT model's recorded attempts on the 204 training tasks (8 attempts per task).

1. `provenance_check.py` and `replay_check.py` exclude any attempt that does not trace back to its run or does not
   replay on the pinned environment.
2. `chosen_pool.py` collects the successes that pass every SFT data check; `review_scans.py`, `overwrite_scan.py`,
   `mask_turns.py` and `reply_edits.py` apply the SFT data rules to them.
3. `rejected_pool.py` collects the failures that are real, reproduced failures.
4. `build_pairs.py` pairs a whole chosen attempt with a whole rejected attempt of the same task, through
   `attempt_rows.py`.
5. `exact_tokens.py` and `sampled_spelling.py` keep the SFT model's own sampled tokens on both sides.
   `pair_parity.py` and `dry_render.py` check every pair against the framework's DPO rendering.
6. `phase4_ches.py` scores the candidates on the GPU: it first checks the adapter with `p9_adapter_check.py`, then
   computes CHES. `ches_rank.py` and `prepare_train.py` then select 322 pairs on 111 tasks: 277 from the model's own
   attempts, plus 45 with an SFT row as the preferred side.

The selected pairs are `dpo/pairs.jsonl` in
[carbonteq/automationbench-posttrain-data](https://huggingface.co/datasets/carbonteq/automationbench-posttrain-data).

## Train

Put the pairs at `apps/lab/.posttrain/datasets/qwen35-4b-ab-dpo-p9-train.jsonl`, then submit the job:

```bash
cd apps/lab
posttrain job run .posttrain/work_packages/qwen35_4b_automationbench_dpo_p9_e3.yaml --job train \
  --provider dstack --timeout-seconds 43200 --run-id <run id>
```

| Setting | Value |
| --- | --- |
| Loss | sigmoid DPO with beta 0.1, plus an SFT term on the preferred side (weight 1); the frozen SFT adapter is the reference, scored once before training |
| Optimiser | learning rate 1e-5, cosine schedule, 10% warm-up, effective batch 16, 61 steps (each pair seen about 3 times) |
| LoRA | rank 16, alpha 32, the SFT adapter's modules; seed 20261103; 32,768-token sides |

After training, the framework's tracking upload can hang while the job still holds the GPU. Once the adapter is saved,
cancel the run with `posttrain run cancel <run id>`.
