template: data@2
---
```sql result
select data_examples, data_bytes, duration_seconds from runs
```

**{{run.job_kind}}** · {{run.status}} · {{result.duration_seconds | duration}} · work package {{run.work_package}}

| | |
| --- | --- |
| Examples | {{result.data_examples | default "not recorded"}} |
| Bytes | {{result.data_bytes | default "not recorded"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |
