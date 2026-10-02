# Correctness investigation reports

This directory retains written findings, mathematical explanations, source
identities and qualification limits. Experimental runners, raw sweeps, module
traces and copied runner snapshots are local research material and are not
committed to the product repository.

The archive on this workstation is
`/home/hammad/experiments/posttrain-correctness/2026-10-01`.
It preserves all removed files byte-for-byte, their hash manifest, the original
unpublished Git history as a verified bundle, and the pre-cleanup dirty diff.
It is a local archive, not a published or shared artifact service.

Before using reproduction commands in these reports, set:

```bash
export POSTTRAIN_CORRECTNESS_ROOT=/home/hammad/experiments/posttrain-correctness/2026-10-01
cd /home/hammad/projects/rl
```

Runner references use `${POSTTRAIN_CORRECTNESS_ROOT}/tools/correctness/`.
Raw receipt filenames are relative to the archive's
`docs/research/evidence/correctness-matrix/` directory. These filenames are
references to local evidence, not links to files shipped with this repository.
The archive's `packages` symlink lets the existing runners resolve framework
sources without copying or committing them. Existing ignored runtime outputs
under `.posttrain/state/correctness/` remain local as well.

Future experiments should write runners and raw outputs outside the repository.
Commit actual source repairs, meaningful regression tests and concise findings.
Do not recreate trace dumps or experimental runner snapshots in this directory.
