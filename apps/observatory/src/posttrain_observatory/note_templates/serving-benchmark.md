template: serving-benchmark@1
---
```data result
measures: [serve_requests, serve_output_tokens, serve_measurement_seconds, serve_peak_vram_bytes, duration_seconds]
```

```data levels
measures: [throughput, failure_rate, completed_requests, ttft_p50_ms, tpot_p50_ms]
by: [load_level.concurrency, load_level.context_tokens]
order_by: [load_level.concurrency]
```

**{{run.job_kind}}** · {{run.status}} · {{result.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Inference | {{run.inference_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Requests | {{result.serve_requests | default "—"}} |
| Output tokens | {{result.serve_output_tokens | default "—"}} |
| Measured for | {{result.serve_measurement_seconds | duration | default "—"}} |
| Peak VRAM | {{result.serve_peak_vram_bytes | default "—"}} bytes |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```table
data: levels
title: Load levels
```

```chart
data: levels
x: load_level.concurrency
y: throughput
type: bar
title: Output tokens per second by concurrency
```
