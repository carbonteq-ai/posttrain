template: generic@1
---
```data span
measures: [duration_seconds]
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Started | {{run.started_at}} |
| Finished | {{run.finished_at | default "not finished"}} |
| Error | {{run.error | default "none"}} |
