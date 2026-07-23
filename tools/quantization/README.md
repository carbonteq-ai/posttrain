# Isolated quantization runtime

The AWQ and RTN transforms run outside the training environment because LLM Compressor
and the TRL fork do not share one compatible `datasets` dependency set.

Create the runtime and expose it to the lab host:

```bash
uv sync --project tools/quantization --locked
export POSTTRAIN_QUANTIZATION_PYTHON="$PWD/tools/quantization/.venv/bin/python"
uv run posttrain-lab qwen-awq-transform --tracked --project <project>
uv run posttrain-lab qwen-rtn-transform --tracked --project <project>
```

The runtime is pinned to an exact LLM Compressor Git revision because the
released packages available to this workspace still target Transformers 4.x.
The exact modifier recipes are checked in under `recipes/`; their file digests
are validated against the catalog so the recipe name cannot drift from the
executed configuration unnoticed.
