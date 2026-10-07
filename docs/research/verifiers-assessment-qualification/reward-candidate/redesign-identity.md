# Reward redesign identity

The isolated candidate adds `calibration/reward_revision.py` to bind actual
adapter/native Python sources, the environment lock, canonical configuration,
declared native provenance and runtime versions. Historical Luna collection
identities stay unchanged. This closes the old scorer fingerprint's omission
of new reward modules. Identity integrity is separate from design approval.

Nine tests pass; scoped Ruff and Pyright pass. Tests cover source/configuration/
lock changes, stable configuration ordering, malformed re-sealed records and
nested mutation on serialized revalidation. Revalidate at consumption: frozen
models still contain mutable dictionaries.

From `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

```sh
.venv/bin/python -m pytest tests/test_calibration_reward_revision.py -q
.venv/bin/ruff check src/automationbench_v1/calibration/reward_revision.py tests/test_calibration_reward_revision.py
.venv/bin/pyright --pythonpath .venv/bin/python src/automationbench_v1/calibration/reward_revision.py
```

Source-bound eligibility remains in progress. This does not approve a full-bank
reward revision or qualify Qwen execution, publication or dependency adoption.
