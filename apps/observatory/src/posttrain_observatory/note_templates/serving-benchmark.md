template: serving-benchmark@2
---
```sql result
select serve_requests, serve_output_tokens, serve_measurement_seconds, serve_peak_vram_bytes, serve_concurrency,
       serve_context_tokens, duration_seconds
from runs
```

**{{run.job_kind}}** · {{run.status}} · {{result.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Inference | {{run.inference_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Requests | {{result.serve_requests | default "—"}} at concurrency {{result.serve_concurrency | default "—"}} |
| Output tokens | {{result.serve_output_tokens | default "—"}} |
| Measured for | {{result.serve_measurement_seconds | duration | default "—"}} |
| Context | {{result.serve_context_tokens | default "—"}} tokens |
| Peak VRAM | {{result.serve_peak_vram_bytes | default "—"}} bytes |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

Load levels and the throughput frontier are in the serving view above.
