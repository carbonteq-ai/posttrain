# Post-training jobs

A job is a versioned Python definition for one objective or workstream. It can
compose shared model definitions, train/eval/serve profiles, published
Verifiers environments, and exact artifacts across many actions and runs.

```text
jobs/<job-id>/
  job.py       Job identity and explicit actions
  README.md    Objective, constraints, conclusions, and human rationale
```

Actions are normal Python functions discovered with `@job.action`. They invoke
typed operations from the reusable train, eval, or serve packages; they do not
define a hidden workflow DAG.
One action invocation may create several Trackio runs, and repeated invocations
remain under the same job.

Model branches can remain in one job when they answer the same objective. Use a
new job for a distinct objective, owner, or lifecycle. Trackio observes runs,
metrics, traces, artifacts, and lineage; it does not define or execute the job.
