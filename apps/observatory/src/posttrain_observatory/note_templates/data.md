template: data@1
---
```data result
measures: [data_examples, data_bytes, duration_seconds]
```

**{{run.job_kind}}** · {{run.status}} · {{result.duration_seconds | duration}} · work package {{run.work_package}}

| | |
| --- | --- |
| Examples | {{result.data_examples | default "not recorded"}} |
| Bytes | {{result.data_bytes | default "not recorded"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |
