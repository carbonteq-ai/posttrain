# veRL release inputs

This directory holds the dependency-only Python 3.13.12 project and lock for
the dormant `online-rl-verl-py313` kind image.

Current inputs:

1. `pyproject.toml` selects CarbonTeq veRL
   `1dcdf67e9473db5297c98c9c88cf4dae6c4a8932` and Verifiers core
   `284a868d6a9022109b749710672a0460e8a996d4` with no concrete environment
   packages and no editable or path sources.
2. `uv.lock` is generated for exact Python `3.13.12`.
3. `profile.toml` records the fork revision and lock digest
   `0e293b1d7beadc0e9548481236117c8fe6a888b7806029f054ee68935362f8f8`.

The shared kind image may be added to the BuildKit publication graph only after
all of the following are true:

1. the CarbonTeq veRL changes and fork ledger are committed together;
2. that full commit is reachable from the configured CarbonTeq remote;
3. `pyproject.toml` in this directory selects that full Git commit and contains no
   concrete Verifiers environment;
4. `uv.lock` is generated for Python 3.13.12 without editable or path sources;
5. `profile.toml` records the fork commit and exact lock digest; and
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
sha256sum uv.lock
```

Then update `dependency_lock_sha256` in `../profile.toml` to match.
