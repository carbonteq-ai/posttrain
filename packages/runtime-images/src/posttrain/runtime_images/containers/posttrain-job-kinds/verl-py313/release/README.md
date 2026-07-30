# veRL release inputs

This directory holds the dependency-only Python 3.13.12 project and lock for
the dormant `online-rl-verl-py313` kind image.

Current inputs:

1. `pyproject.toml` selects CarbonTeq veRL
   `c3f49b9117b882fa888e25e4a771461e13167848`, CarbonTeq vLLM
   `7817d845727af570352622dc8d58f2d43c76d89d`, and Verifiers core
   `284a868d6a9022109b749710672a0460e8a996d4` with no concrete environment
   packages and no editable or path sources.
2. `uv.lock` is generated for exact Python `3.13.12`.
3. `backend-constraints.txt` is the exact, hash-bound export of that lock used
   while packaging veRL environment wheels. It prevents workspace dependency
   constraints from replacing backend-owned versions such as
   `antlr4-python3-runtime==4.9.3`.
4. `profile.toml` records the fork revision and both content digests.
5. The image verifies the upstream vLLM 0.25.1 x86_64 ABI3 wheel at SHA-256
   `16fc7a28df1576eb6f7ca0455026551b8f9adb674c19c66059359ef3e964bd1e`
   and uses its compiled extensions as the binary base for the Python-only
   CarbonTeq fork delta.

The shared kind image may be added to the BuildKit publication graph only after
all of the following are true:

1. the CarbonTeq veRL changes and fork ledger are committed together;
2. that full commit is reachable from the configured CarbonTeq remote;
3. `pyproject.toml` in this directory selects that full Git commit and contains no
   concrete Verifiers environment;
4. `uv.lock` is generated for Python 3.13.12 without editable or path sources;
5. `profile.toml` records the fork commit, exact lock digest, and exact backend
   constraints digest; and
6. the ready profile is present in the job-kind Bake publication graph; and
7. `release_gate.py --release` passes against a clean checkout and the remote,
   including the real Docker/Bake smoke.

The actual-job image must keep the Python 3.12 framework control environment
and install the exact selected environment-wheel closure plus the
content-addressed `common`, `data`, and `train` source projection into
`/opt/posttrain-verl`. `profile.toml` and `release_gate.py` bind the projection
path, Python-path variable, package set, worker module, and dormant/ready Bake
state to the actual Docker definition. The shared kind image must not contain
GSM8K, AutomationBench, or another concrete environment package.

Regenerate the lock after changing `pyproject.toml`:

```bash
cd containers/posttrain-job-kinds/verl-py313/release
uv lock --python 3.13.12
uv export \
  --frozen \
  --no-dev \
  --no-emit-project \
  --no-hashes \
  --no-annotate \
  --format requirements-txt \
  --output-file backend-constraints.txt
sha256sum uv.lock
sha256sum backend-constraints.txt
```

Then update `dependency_lock_sha256` and `backend_constraints_sha256` in
`../profile.toml` to match.
