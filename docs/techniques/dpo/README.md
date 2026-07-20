# DPO

Direct Preference Optimization over renderer-pretokenized preference pairs.

- Target engine: `packages/train` with TRL `DPOTrainer`
- Public operation: `posttrain.train.dpo(context, request)`
- Lab smoke entrypoints: `gsm8k-qwen-dpo-smoke` and `gsm8k-lfm-dpo-smoke`

The job derives a strict chosen/rejected pair from an authoritative
demonstration and a retained Verifiers rollout trace. The reusable trainer sees
only the typed preference dataset and a materialized parent adapter.

## Verified Qwen smoke

Trackio run `9a89fda28de34c6d9254995402becba9` consumed SFT adapter
`training-qwen3.5-2b-sft-adapter:v0` and rejected trace
`fe56257e31fb4f5797238da5f3906d76` from rollout run
`70a557c4ad4644ecb2ab8e9e3d3df1c8`. It completed two steps with final loss
`0.34744`, final logged gradient norm `0.13867`, and produced a DPO adapter,
step-2 recovery checkpoint, and summary artifact.

```bash
uv run --package posttrain-lab --extra gpu-posttrain \
  posttrain-lab gsm8k-qwen-dpo-smoke \
  --tracked --project posttrain-platform --adapter-version v0 \
  --rollout-run-id 70a557c4ad4644ecb2ab8e9e3d3df1c8 \
  --rejected-trace-id fe56257e31fb4f5797238da5f3906d76
```

Qwen uses the Torch DPO loss because the measured Liger backward path creates a
large FP32 language-head gradient even when that head is frozen. LFM retains a
separate model-family profile and still requires runtime qualification.
