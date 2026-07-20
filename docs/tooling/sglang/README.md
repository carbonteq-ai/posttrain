# SGLang

SGLang is an optional backend of `packages/serve`:

```bash
uv sync --package serve --extra sglang --python 3.12
```

Backend-native settings and model compatibility belong in typed definitions
shipped with `packages/serve`; observations remain attached to the exact model
artifact and resolved execution context. SGLang and vLLM are mutually exclusive
dependency variants in the uv workspace.

The new SGLang launcher will be added only after the initial model profiles exist. The removed prototype launcher and normalized-result adapter are not compatibility requirements.
