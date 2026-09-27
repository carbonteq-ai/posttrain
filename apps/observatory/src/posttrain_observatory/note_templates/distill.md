template: distill@1
---
```data result
measures: [update_seconds:count, distill_loss:first, distill_loss:last, reverse_kl:last, teacher_failures:sum]
```

```data span
measures: [duration_seconds]
```

```data curve
measures: [distill_loss, reverse_kl]
by: update.step
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Student | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Environment | {{run.environment | default "not recorded"}} |
| Learning rate | {{run.learning_rate | sci 1 | default "not recorded"}} |
| Training / inference | {{run.training_binding | default "not recorded"}} / {{run.inference_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Updates | {{result.update_seconds_count | default "0"}} of {{run.max_updates | default "?"}} |
| Distillation loss | {{result.distill_loss_first | round 4 | default "—"}} → {{result.distill_loss_last | round 4 | default "—"}} |
| Reverse KL (last) | {{result.reverse_kl_last | round 4 | default "—"}} |
| Teacher failures | {{result.teacher_failures_sum | default "0"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: curve
x: update.step
y: [distill_loss, reverse_kl]
title: Distillation loss by update
```
