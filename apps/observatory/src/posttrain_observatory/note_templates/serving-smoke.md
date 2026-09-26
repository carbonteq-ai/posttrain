template: serving-smoke@1
---
```data result
measures: [probe_healthy, probe_model_available, probe_latency_seconds, duration_seconds]
```

**{{run.job_kind}}** · {{run.status}} · {{result.duration_seconds | duration}} · work package {{run.work_package}}

| | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Inference | {{run.inference_binding | default "not recorded"}} |
| Healthy | {{result.probe_healthy | default "not recorded"}} |
| Model available | {{result.probe_model_available | default "not recorded"}} |
| Probe latency | {{result.probe_latency_seconds | round 3 | default "—"}} s |
| Error | {{run.error | default "none"}} |
