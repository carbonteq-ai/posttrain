# Model profiles

Each YAML file identifies one reusable loadable target:

- a foundation profile references an immutable external model revision;
- a derived profile references an intentionally promoted adapter, merged model, quantized model, or checkpoint;
- optional `defaults` select recommended train, eval, and serve configs.

Ordinary trainer checkpoints remain Trackio artifacts and do not receive profile files.
