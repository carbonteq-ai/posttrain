# posttrain-data

`posttrain-data` owns trainer-neutral, immutable data contracts for supervised
fine-tuning, preference optimization, and task-neutral rollout selection. It is
imported as `posttrain.data`.

The package deliberately separates semantic records from storage containers:
Hugging Face Dataset, JSONL, Parquet, Trackio artifacts, and object storage can
all carry the same canonical snapshot.

## Canonical SFT example

```python
SupervisedExample(
    id="customer-support/000001",
    messages=(
        {"role": "user", "content": "Cancel my order"},
        {"role": "assistant", "content": "I can help with that."},
    ),
    trainable_message_indices=(1,),
    tools=(),
    metadata={"source": "reviewed-conversation"},
)
```

## Canonical preference example

```python
PreferenceExample(
    id="customer-support/000001",
    prompt=({"role": "user", "content": "Cancel my order"},),
    chosen=({"role": "assistant", "content": "I can help with that."},),
    rejected=({"role": "assistant", "content": "No."},),
    chosen_trace_id="trace-good",
    rejected_trace_id="trace-bad",
)
```

## Adapter boundary

- `supervised_from_huggingface` accepts messages, prompt-completion, Alpaca,
  and ShareGPT rows.
- `preferences_from_huggingface` accepts TRL, Tulu, and NeMo-ranked rows.
- `supervised_from_nemo` and `to_nemo_*` make NeMo conversion explicit.
- Catalog `source.kind: nemo` loads project-relative NeMo JSONL through those
  adapters (`messages` / `nemo-ranked`, or `auto`) and materializes the same
  canonical cache as other sources.
- `supervised_from_verifiers` projects selected native trace branches into SFT
  examples; `supervised_from_verifiers_jsonl` validates a native trace artifact
  before projection (not yet a catalog source kind).

Model chat templates, tokenization, packing, and token-level loss masks remain
the responsibility of the consuming training engine.
