# veRL release inputs

This directory deliberately contains no `uv.lock` yet.

The shared `online-rl-verl-py313` image may be added to the BuildKit
publication graph only after all of the following are true:

1. the CarbonTeq veRL changes and fork ledger are committed together;
2. that full commit is reachable from the configured CarbonTeq remote;
3. `pyproject.toml` in this directory selects that full Git commit and contains no
   concrete Verifiers environment;
4. `uv.lock` is generated for Python 3.13.12 without editable or path sources;
5. `profile.toml` records the fork commit and exact lock digest; and
6. the ready profile is present in the job-kind Bake publication graph; and
7. `release_gate.py --release` passes against a clean checkout and the remote.

The actual-job image must keep the Python 3.12 framework control environment
and install the exact selected environment-wheel closure plus the
content-addressed `common`, `data`, and `train` source projection into
`/opt/posttrain-verl`. `profile.toml` and `release_gate.py` bind the projection
path, Python-path variable, package set, worker module, and dormant/ready Bake
state to the actual Docker definition. The shared kind image must not contain
GSM8K, AutomationBench, or another concrete environment package.
